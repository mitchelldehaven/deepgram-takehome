"""Slice WER by metadata, and first check the metadata is worth slicing on.

The order matters. "Compute SNR, bin it, report WER per bin" silently assumes
the SNR estimate is accurate enough that the bins mean something. Here that
assumption is tested first -- two independent estimators per field, compared
on the continuous value and on the bin assignment -- and any field that fails
is reported as unreliable rather than quietly used anyway.

Also computes the added metric (keyword error rate), since the keyword set is
derived from the same reference-side metadata.
"""
from collections import Counter

import dirtygit
import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import OmegaConf
from scipy.stats import spearmanr

from src.config import load_config, setup_logging
from src.data.metadata import WORD
from src.data.utils import load_jsonl
from src.eval.metrics import (
    bootstrap_ci,
    counts,
    keyword_counts,
    keyword_error_rate,
)
from src.eval.score import aligned, load_transcripts
from src.normalization.factory import build_normalizer
from src.paths import CONFIG_DIR, METADATA_FILE, SCORES_DIR, SUBSET_FILE

DEFAULT_CONFIG = CONFIG_DIR / "eval" / "openasr.yaml"

# Fields computed two independent ways in src/data/metadata.py.
ESTIMATOR_PAIRS = {
    "snr_db": ("snr_percentile_db", "snr_wada_db"),
    "speech_rate": ("speech_rate_wall", "speech_rate_voiced"),
}

# Agreement below this and we do not trust the field enough to slice on it.
MIN_SPEARMAN = 0.7
MIN_BIN_AGREEMENT = 0.6

N_BINS = 3
BIN_LABELS = ["low", "mid", "high"]


def tertiles(values):
    """Bin into terciles, per dataset-free global quantiles."""
    finite = np.isfinite(values)
    out = np.full(len(values), "n/a", dtype=object)
    if finite.sum() < N_BINS:
        return out
    edges = np.quantile(values[finite], np.linspace(0, 1, N_BINS + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    # Quantile edges collapse when a field is heavily tied; drop duplicates.
    edges = np.unique(edges)
    if len(edges) - 1 < 2:
        return out
    labels = BIN_LABELS[: len(edges) - 1]
    out[finite] = np.array(labels)[np.digitize(values[finite], edges[1:-1])]
    return out


def cohen_kappa(a, b):
    """Agreement on bin assignment, corrected for chance."""
    categories = sorted(set(a) | set(b))
    index = {c: i for i, c in enumerate(categories)}
    matrix = np.zeros((len(categories), len(categories)))
    for x, y in zip(a, b):
        matrix[index[x], index[y]] += 1
    total = matrix.sum()
    if not total:
        return float("nan")
    observed = np.trace(matrix) / total
    expected = (matrix.sum(0) @ matrix.sum(1)) / total**2
    return float((observed - expected) / (1 - expected)) if expected < 1 else float("nan")


def check_reliability(metadata):
    """Do the two estimators of each field agree enough to bin on?"""
    rows = []
    for field, (first, second) in ESTIMATOR_PAIRS.items():
        a = np.array([row[first] for row in metadata], dtype=float)
        b = np.array([row[second] for row in metadata], dtype=float)
        both = np.isfinite(a) & np.isfinite(b)
        rho = spearmanr(a[both], b[both]).statistic if both.sum() > 2 else float("nan")
        bins_a, bins_b = tertiles(a), tertiles(b)
        agreement = float((bins_a[both] == bins_b[both]).mean()) if both.sum() else float("nan")
        kappa = cohen_kappa(bins_a[both], bins_b[both]) if both.sum() else float("nan")
        reliable = bool(rho >= MIN_SPEARMAN and agreement >= MIN_BIN_AGREEMENT)
        rows.append(
            {
                "field": field,
                "estimator_a": first,
                "estimator_b": second,
                "n": int(both.sum()),
                "spearman": rho,
                "bin_agreement": agreement,
                "cohen_kappa": kappa,
                "reliable": reliable,
            }
        )
    return pd.DataFrame(rows)


def build_keyword_vocab(references, max_df, min_length):
    """Rare content words, with rarity measured inside the evaluation set.

    An earlier version used mid-sentence capitalised tokens as a proxy for
    named entities. That failed on contact with the data: GigaSpeech
    references are entirely lowercase, so 400 of 1200 utterances were silently
    excluded, and on the rest it picked up "yeah" and "i'll" rather than
    entities.

    Document frequency does not care about casing, so it works across all
    three reference conventions. Words appearing in at most `max_df` of the
    1200 references are the names, jargon and product terms a buyer is
    actually paying for; common words are exactly the ones WER already
    over-weights.
    """
    frequencies = Counter()
    for reference in references:
        frequencies.update(set(WORD.findall(reference.lower())))
    return {
        word
        for word, count in frequencies.items()
        if count <= max_df and len(word) >= min_length
    }


def keyword_set(reference, vocab):
    return {word for word in WORD.findall(reference.lower()) if word in vocab}


def breakdown_table(cfg, subset, metadata, transcripts, normalizer, reliability):
    by_id = {row["id"]: row for row in metadata}
    rows = []
    for system_name, hypotheses in transcripts.items():
        references, hyps, datasets = aligned(subset, hypotheses, normalizer)
        kept = [row for row in subset if row["id"] in hypotheses]
        pairs = counts(references, hyps)

        for field, (first, _) in ESTIMATOR_PAIRS.items():
            is_reliable = bool(
                reliability.loc[reliability.field == field, "reliable"].iloc[0]
            )
            values = np.array([by_id[row["id"]][first] for row in kept], dtype=float)
            bins = tertiles(values)
            for label in BIN_LABELS:
                mask = bins == label
                if mask.sum() < 10:
                    continue
                point, lower, upper = bootstrap_ci(
                    pairs[mask],
                    cfg.scoring.bootstrap_samples,
                    cfg.scoring.confidence,
                    cfg.scoring.seed,
                )
                rows.append(
                    {
                        "system": system_name,
                        "field": field,
                        "bin": label,
                        "n": int(mask.sum()),
                        "median_value": float(np.nanmedian(values[mask])),
                        "wer": 100 * point,
                        "ci_low": 100 * lower,
                        "ci_high": 100 * upper,
                        "field_reliable": is_reliable,
                    }
                )

        # Dataset is a label we are given rather than one we estimate, so it
        # needs no reliability check and is always safe to slice on.
        for dataset in dict.fromkeys(datasets):
            mask = datasets == dataset
            point, lower, upper = bootstrap_ci(
                pairs[mask],
                cfg.scoring.bootstrap_samples,
                cfg.scoring.confidence,
                cfg.scoring.seed,
            )
            rows.append(
                {
                    "system": system_name,
                    "field": "dataset",
                    "bin": dataset,
                    "n": int(mask.sum()),
                    "median_value": float("nan"),
                    "wer": 100 * point,
                    "ci_low": 100 * lower,
                    "ci_high": 100 * upper,
                    "field_reliable": True,
                }
            )
    return pd.DataFrame(rows)


def keyword_table(cfg, subset, transcripts, normalizer):
    """Keyword error rate next to WER, for the same systems and utterances."""
    vocab = build_keyword_vocab(
        [normalizer(row["reference"]) for row in subset],
        cfg.scoring.keyword_max_df,
        cfg.scoring.keyword_min_length,
    )
    logger.info(f"keyword vocabulary: {len(vocab)} rare content words")

    rows = []
    for system_name, hypotheses in transcripts.items():
        references, hyps, datasets = aligned(subset, hypotheses, normalizer)
        keywords = [keyword_set(reference, vocab) for reference in references]
        covered = [i for i, k in enumerate(keywords) if k]
        pairs = counts(references, hyps)
        keyword_pairs = keyword_counts(references, hyps, keywords)
        ker, ker_low, ker_high = bootstrap_ci(
            keyword_pairs,
            cfg.scoring.bootstrap_samples,
            cfg.scoring.confidence,
            cfg.scoring.seed,
        )
        rows.append(
            {
                "system": system_name,
                "dataset": "ALL",
                "wer": 100 * (pairs[:, 0].sum() / max(pairs[:, 1].sum(), 1)),
                "keyword_error_rate": 100 * ker,
                "ker_ci_low": 100 * ker_low,
                "ker_ci_high": 100 * ker_high,
                "n_utterances_with_keywords": len(covered),
            }
        )
        # Per dataset too: the three differ enough that a pooled number hides
        # which corpus is driving the gap.
        for dataset in dict.fromkeys(datasets):
            mask = datasets == dataset
            indices = [i for i in range(len(references)) if mask[i]]
            rows.append(
                {
                    "system": system_name,
                    "dataset": dataset,
                    "wer": 100
                    * (pairs[mask][:, 0].sum() / max(pairs[mask][:, 1].sum(), 1)),
                    "keyword_error_rate": 100
                    * keyword_error_rate(
                        [references[i] for i in indices],
                        [hyps[i] for i in indices],
                        [keywords[i] for i in indices],
                    ),
                    "n_utterances_with_keywords": sum(
                        1 for i in indices if keywords[i]
                    ),
                }
            )
    return pd.DataFrame(rows)


def main(cfg):
    logger.info(f"githash={cfg.githash}")
    subset = load_jsonl(SUBSET_FILE)
    metadata = load_jsonl(METADATA_FILE)
    transcripts = load_transcripts(cfg)
    normalizer = build_normalizer("leaderboard")
    SCORES_DIR.mkdir(parents=True, exist_ok=True)

    reliability = check_reliability(metadata)
    reliability.to_csv(SCORES_DIR / "metadata_reliability.csv", index=False)
    logger.info("metadata reliability (two estimators per field):")
    for _, row in reliability.iterrows():
        verdict = "OK" if row.reliable else "UNRELIABLE -- do not slice on this"
        logger.info(
            f"  {row.field:<14} spearman={row.spearman:+.3f} "
            f"bin_agreement={row.bin_agreement:.3f} kappa={row.cohen_kappa:+.3f}  {verdict}"
        )

    table = breakdown_table(cfg, subset, metadata, transcripts, normalizer, reliability)
    table.to_csv(SCORES_DIR / "breakdown.csv", index=False)
    for field in table.field.unique():
        subtable = table[table.field == field]
        reliable = bool(subtable.field_reliable.iloc[0])
        note = "" if reliable else "  (field failed its reliability check)"
        logger.info(f"WER by {field}{note}:")
        for _, row in subtable.iterrows():
            logger.info(
                f"  {row.system:<24} {str(row['bin']):<18} n={row.n:<4} "
                f"{row.wer:5.2f} [{row.ci_low:.2f}, {row.ci_high:.2f}]"
            )

    keywords = keyword_table(cfg, subset, transcripts, normalizer)
    keywords.to_csv(SCORES_DIR / "keyword_error_rate.csv", index=False)
    logger.info("WER vs keyword error rate:")
    for _, row in keywords[keywords.dataset == "ALL"].iterrows():
        logger.info(
            f"  {row.system:<24} WER {row.wer:5.2f}  "
            f"KER {row.keyword_error_rate:5.2f} "
            f"[{row.ker_ci_low:.2f}, {row.ker_ci_high:.2f}]  "
            f"(n={row.n_utterances_with_keywords})"
        )
    for dataset in [d for d in keywords.dataset.unique() if d != "ALL"]:
        logger.info(f"  -- {dataset}")
        for _, row in keywords[keywords.dataset == dataset].iterrows():
            logger.info(
                f"     {row.system:<22} WER {row.wer:5.2f}  "
                f"KER {row.keyword_error_rate:5.2f}  (n={row.n_utterances_with_keywords})"
            )

    logger.info(f"tables -> {SCORES_DIR}")


if __name__ == "__main__":
    githash = dirtygit.check()
    cfg = load_config(DEFAULT_CONFIG)
    OmegaConf.update(cfg, "githash", githash)
    setup_logging()
    main(cfg)
