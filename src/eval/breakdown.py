"""Slice WER by metadata, and first check the metadata is worth slicing on.

The order matters. "Compute SNR, bin it, report WER per bin" silently assumes
the SNR estimate is accurate enough that the bins mean something. Here that
assumption is tested first -- two independent estimators per field, compared
on the continuous value and on the bin assignment -- and any field that fails
is reported as unreliable rather than quietly used anyway.

Slices every metric in src.eval.metrics.METRICS, not just WER, because the
reason for having the other three is that they can disagree with it -- and a
disagreement that only shows up in one acoustic condition is the kind a pooled
number hides.
"""
import dirtygit
import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import OmegaConf
from scipy.stats import spearmanr

from src.config import load_config, setup_logging
from src.data.utils import load_jsonl
from src.eval.metrics import METRICS, bootstrap_ci
from src.eval.score import aligned, metric_kwargs, load_transcripts
from src.normalization.factory import build_normalizer
from src.paths import CONFIG_DIR, METADATA_FILE, SCORES_DIR, SUBSET_FILE

DEFAULT_CONFIG = CONFIG_DIR / "eval" / "openasr.yaml"

# Acoustic properties we *estimate*. Each is computed two independent ways in
# src/data/metadata.py so the estimate can be checked before it is used.
ESTIMATOR_PAIRS = {
    "snr_db": ("snr_percentile_db", "snr_wada_db"),
    "speech_rate": ("speech_rate_wall", "speech_rate_voiced"),
}

# Properties we *measure*. A duration is a sample count over a sample rate and
# a word count is a word count: there is no estimator to disagree with, so the
# two-estimator check neither applies nor is needed, and these are exact.
# Keeping the distinction explicit matters, because the reliability gate was
# originally the only route into the breakdown table -- which silently meant
# that a field without a second estimator could not be sliced on at all, and
# the most obvious conditioning variable on a short-form benchmark (how long
# the utterance is) went unexamined for that reason alone.
MEASURED_FIELDS = {
    "duration_s": "utterance duration in seconds",
    "n_words": "reference length in words",
    "disfluency_rate": "fraction of reference words that are fillers",
    "numeric_rate": "fraction of reference tokens containing a digit",
}

# Estimated, but with only one estimator, so it cannot be checked the way the
# two above are. Sliced on and reported, flagged unreliable -- the project's
# rule is to report a field as unchecked rather than to use it quietly.
UNCHECKED_FIELDS = {"voiced_fraction": "fraction of frames above the energy floor"}

# Agreement below this and we do not trust the field enough to slice on it.
MIN_SPEARMAN = 0.7
MIN_BIN_AGREEMENT = 0.6

N_BINS = 3
BIN_LABELS = ["low", "mid", "high"]

# Below this many utterances a bin is not reported at all.
MIN_BIN_SIZE = 10


def bin_values(values):
    """Three bins, chosen to suit the field's distribution.

    Terciles are the default, but they are the wrong tool for a zero-inflated
    field: `disfluency_rate` is exactly zero on 86% of utterances and
    `numeric_rate` on 96%, so both their 33rd and 67th percentiles are zero
    and all the structure collapses into one bin. There the informative split
    is none-vs-some: utterances at exactly zero form a bin of their own, and
    the rest are halved at their median.

    Returns (bins, labels). The `bin` names differ between the two schemes on
    purpose, so a reader cannot mistake "none" for "low".
    """
    finite = np.isfinite(values)
    zero_inflated = finite.sum() and (values[finite] == 0).mean() > 1 / N_BINS
    if not zero_inflated:
        return tertiles(values), BIN_LABELS

    out = np.full(len(values), "n/a", dtype=object)
    out[finite & (values == 0)] = "none"
    nonzero = finite & (values > 0)
    if nonzero.sum() < 2 * MIN_BIN_SIZE:
        out[nonzero] = "some"
        return out, ["none", "some"]
    split = np.median(values[nonzero])
    out[nonzero & (values <= split)] = "low"
    out[nonzero & (values > split)] = "high"
    return out, ["none", "low", "high"]


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


def slice_fields(reliability):
    """Every field we slice on, as (name, metadata key, is_reliable).

    Three kinds, and the difference is the point: an estimated field carries
    the verdict of its two-estimator check, a measured field is exact and
    needs none, and a single-estimator field is reported as unchecked.
    """
    for field, (first, _) in ESTIMATOR_PAIRS.items():
        verdict = reliability.loc[reliability.field == field, "reliable"]
        yield field, first, bool(verdict.iloc[0]) if len(verdict) else False
    for field in MEASURED_FIELDS:
        yield field, field, True
    for field in UNCHECKED_FIELDS:
        yield field, field, False


def breakdown_table(cfg, subset, metadata, transcripts, normalizer, reliability):
    by_id = {row["id"]: row for row in metadata}
    rows = []
    for system_name, hypotheses in transcripts.items():
        references, hyps, datasets = aligned(subset, hypotheses, normalizer)
        kept = [row for row in subset if row["id"] in hypotheses]

        for metric, function in METRICS.items():
            pairs = function(references, hyps, **metric_kwargs(cfg, metric))

            for field, source, is_reliable in slice_fields(reliability):
                values = np.array(
                    [by_id[row["id"]][source] for row in kept], dtype=float
                )
                bins, labels = bin_values(values)
                for label in labels:
                    mask = bins == label
                    if mask.sum() < MIN_BIN_SIZE:
                        continue
                    point, lower, upper = bootstrap_ci(
                        pairs[mask],
                        cfg.scoring.bootstrap_samples,
                        cfg.scoring.confidence,
                        cfg.scoring.seed,
                    )
                    rows.append(
                        {
                            "metric": metric,
                            "system": system_name,
                            "field": field,
                            "bin": label,
                            "n": int(mask.sum()),
                            "median_value": float(np.nanmedian(values[mask])),
                            "value": 100 * point,
                            "ci_low": 100 * lower,
                            "ci_high": 100 * upper,
                            "field_reliable": is_reliable,
                        }
                    )

            # Dataset is a label we are given rather than one we estimate, so
            # it needs no reliability check and is always safe to slice on.
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
                        "metric": metric,
                        "system": system_name,
                        "field": "dataset",
                        "bin": dataset,
                        "n": int(mask.sum()),
                        "median_value": float("nan"),
                        "value": 100 * point,
                        "ci_low": 100 * lower,
                        "ci_high": 100 * upper,
                        "field_reliable": True,
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
    for metric in METRICS:
        for field in table.field.unique():
            subtable = table[(table.field == field) & (table.metric == metric)]
            if subtable.empty:
                continue
            reliable = bool(subtable.field_reliable.iloc[0])
            note = "" if reliable else "  (field failed its reliability check)"
            logger.info(f"{metric} by {field}{note}:")
            for _, row in subtable.iterrows():
                logger.info(
                    f"  {row.system:<26} {str(row['bin']):<18} n={row.n:<4} "
                    f"{row.value:5.2f} [{row.ci_low:.2f}, {row.ci_high:.2f}]"
                )

    logger.info(f"tables -> {SCORES_DIR}")


if __name__ == "__main__":
    githash = dirtygit.check()
    cfg = load_config(DEFAULT_CONFIG)
    OmegaConf.update(cfg, "githash", githash)
    setup_logging()
    main(cfg)
