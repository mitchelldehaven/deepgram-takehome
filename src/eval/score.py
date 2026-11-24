"""Score the cached transcripts, under each normalizer, against the published numbers.

Produces five tables in outputs/scores/:
  metrics.csv            metric x system x dataset x normalizer, with bootstrap CIs
  error_composition.csv  substitutions / deletions / insertions behind each WER
  published.csv          ours vs the leaderboard's, and the gap
  ablation.csv           WER attributable to each individual normalization rule
  pairwise.csv           paired bootstrap on every system pair, per metric

All four metrics in src.eval.metrics.METRICS are scored, not just WER: the
leaderboard's number is one of them, and the point of the other three is that
they disagree with it about which system to ship.

Nothing here re-runs a model. Changing a normalization rule and rescoring is
seconds, which is the point of caching raw hypotheses in transcribe.py.
"""
import itertools

import dirtygit
import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import OmegaConf

from src.config import load_config, setup_logging
from src.data.utils import load_jsonl
from src.eval.metrics import (
    METRICS,
    bootstrap_ci,
    counts,
    error_breakdown,
    paired_bootstrap,
    wer_from_counts,
)
from src.normalization.factory import build_normalizer, load_rules
from src.paths import CONFIG_DIR, PUBLISHED_FILE, SCORES_DIR, SUBSET_FILE, TRANSCRIPTS_DIR

DEFAULT_CONFIG = CONFIG_DIR / "eval" / "openasr.yaml"


def load_transcripts(cfg):
    """Systems that actually produced output, keyed by name."""
    available = {}
    for system_cfg in cfg.systems:
        path = TRANSCRIPTS_DIR / f"{system_cfg.name}.jsonl"
        if not path.exists():
            logger.warning(f"{system_cfg.name}: no transcripts at {path}, skipping")
            continue
        available[system_cfg.name] = {
            row["id"]: row["hypothesis"] for row in load_jsonl(path)
        }
    if not available:
        raise RuntimeError("No transcripts found; run src/eval/transcribe.py first")
    return available


def retained(subset, hypotheses, normalizer):
    """The subset rows that survive scoring, in order.

    Two conditions: the system produced a hypothesis for the utterance, and
    the reference does not normalize to the empty string. The second is what
    the leaderboard's own harness does and what this pipeline originally did
    not. Empty references are not harmless -- they contribute nothing to the
    WER denominator, but whatever the system emitted against them still
    counts as insertions in the numerator, so keeping them inflates corpus
    WER. There are 90 such rows in the AMI test split (1.2% of it), enough to
    move its WER by 0.15 and to account for ~8% of the reproduction gap this
    report attributes to configuration.

    This is the single definition of "a scorable utterance". `aligned` is
    written in terms of it, and anything that needs the rows behind a score
    (breakdown slices, padding conditions) must call this rather than
    re-deriving the predicate -- the bug above survived as long as it did
    because two code paths each had their own copy.
    """
    return [
        row
        for row in subset
        if row["id"] in hypotheses and normalizer(row["reference"]).strip()
    ]


def aligned(subset, hypotheses, normalizer):
    """Normalized (reference, hypothesis) pairs plus the dataset label, in order."""
    kept = retained(subset, hypotheses, normalizer)
    references = [normalizer(row["reference"]) for row in kept]
    hyps = [normalizer(hypotheses[row["id"]]) for row in kept]
    return references, hyps, np.array([row["dataset"] for row in kept])


def empty_reference_counts(subset, normalizers):
    """How many references each normalizer empties out, per dataset.

    Reported as pipeline health, not swept up into a WER. A reference that
    normalizes away is usually one whose entire content was a filler or a
    non-speech marker, so the count is a property of the normalizer crossed
    with the corpus's transcription conventions, and it differs between the
    two normalizers compared here.
    """
    rows = []
    for norm_name, normalizer in normalizers.items():
        for row in subset:
            rows.append(
                {
                    "normalizer": norm_name,
                    "dataset": row["dataset"],
                    "is_empty": not normalizer(row["reference"]).strip(),
                }
            )
    frame = pd.DataFrame(rows)
    return (
        frame.groupby(["normalizer", "dataset"])
        .agg(n=("is_empty", "size"), n_empty=("is_empty", "sum"))
        .assign(pct=lambda d: 100 * d.n_empty / d.n)
        .reset_index()
    )


def metric_kwargs(cfg, metric):
    """Thresholds live in config, so a reviewer can move them and rescore."""
    if metric == "catastrophic":
        return {"threshold": cfg.scoring.catastrophic_threshold}
    if metric == "hallucination":
        return {"ratio": cfg.scoring.hallucination_ratio}
    if metric == "edge_error":
        return {"edge_words": cfg.scoring.edge_words}
    return {}


def score_table(cfg, subset, transcripts, normalizers):
    """Every metric, every system, every dataset, under every normalizer.

    `denominator` is reported alongside the value because the four metrics do
    not share one: WER is per reference word, two are per utterance, and the
    edge metric is per edge word slot. A reader comparing two rows needs to
    know which.
    """
    rows = []
    for norm_name, normalizer in normalizers.items():
        for system_name, hypotheses in transcripts.items():
            references, hyps, datasets = aligned(subset, hypotheses, normalizer)
            for metric, function in METRICS.items():
                pairs = function(references, hyps, **metric_kwargs(cfg, metric))
                for dataset in list(dict.fromkeys(datasets)) + ["ALL"]:
                    mask = slice(None) if dataset == "ALL" else (datasets == dataset)
                    subset_pairs = pairs[mask]
                    point, lower, upper = bootstrap_ci(
                        subset_pairs,
                        cfg.scoring.bootstrap_samples,
                        cfg.scoring.confidence,
                        cfg.scoring.seed,
                    )
                    rows.append(
                        {
                            "metric": metric,
                            "normalizer": norm_name,
                            "system": system_name,
                            "dataset": dataset,
                            "n_utterances": len(subset_pairs),
                            "denominator": int(subset_pairs[:, 1].sum()),
                            "value": 100 * point,
                            "ci_low": 100 * lower,
                            "ci_high": 100 * upper,
                        }
                    )
    return pd.DataFrame(rows)


def composition_table(cfg, subset, transcripts, normalizers):
    """What the WER is made of. Two systems at one WER can be very different
    products: one that deletes is quietly lossy, one that inserts is noisily
    wrong."""
    rows = []
    for norm_name, normalizer in normalizers.items():
        for system_name, hypotheses in transcripts.items():
            references, hyps, datasets = aligned(subset, hypotheses, normalizer)
            for dataset in list(dict.fromkeys(datasets)) + ["ALL"]:
                mask = slice(None) if dataset == "ALL" else (datasets == dataset)
                errors = error_breakdown(
                    list(np.array(references)[mask]), list(np.array(hyps)[mask])
                )
                rows.append(
                    {
                        "normalizer": norm_name,
                        "system": system_name,
                        "dataset": dataset,
                        "sub": 100 * errors["sub"],
                        "del": 100 * errors["del"],
                        "ins": 100 * errors["ins"],
                    }
                )
    return pd.DataFrame(rows)


def published_comparison(cfg, wer_df):
    """Ours vs theirs, per dataset, for the systems the board actually ranks."""
    if not PUBLISHED_FILE.exists():
        logger.warning(f"{PUBLISHED_FILE} missing; run src/data/fetch_published.py")
        return pd.DataFrame()

    published = pd.read_csv(PUBLISHED_FILE).set_index("model")
    rows = []
    for system_cfg in cfg.systems:
        if not system_cfg.get("leaderboard_id"):
            continue
        if system_cfg.leaderboard_id not in published.index:
            continue
        for _, row in wer_df[
            (wer_df.metric == "wer")
            & (wer_df.system == system_cfg.name)
            & (wer_df.normalizer == "leaderboard")
            & (wer_df.dataset != "ALL")
        ].iterrows():
            if row.dataset not in published.columns:
                continue
            theirs = pd.to_numeric(
                published.loc[system_cfg.leaderboard_id, row.dataset], errors="coerce"
            )
            rows.append(
                {
                    "system": system_cfg.name,
                    "dataset": row.dataset,
                    "ours": row.value,
                    "ci_low": row.ci_low,
                    "ci_high": row.ci_high,
                    "published": theirs,
                    "gap": row.value - theirs,
                    "published_in_ci": bool(row.ci_low <= theirs <= row.ci_high),
                }
            )
    return pd.DataFrame(rows)


def ablation(cfg, subset, transcripts):
    """Flip one normalization rule at a time; report the WER it is worth.

    Measured against our custom normalizer, not the leaderboard's, so each
    number is the marginal effect of that one rule rather than of the whole
    bundle of differences at once.

    Reported per dataset as well as pooled, because a rule's effect is a
    property of the *reference text*, not of the rule. Pooling hides that:
    `remove_fillers` is worth 2.8-3.1 WER on AMI and 0.00 on SPGISpeech,
    whose references contain two filler words in 948k, and the pooled mean
    understates the AMI effect by roughly 6x. The pooled rows are kept for
    comparability with the ranking tables, which are also pooled.
    """
    base_rules = load_rules()
    baseline = build_normalizer("custom")

    def wer_by_scope(references, hyps, datasets):
        """WER pooled and within each dataset, as {scope: wer}."""
        out = {"ALL": wer_from_counts(counts(references, hyps))}
        for dataset in np.unique(datasets):
            mask = datasets == dataset
            out[dataset] = wer_from_counts(
                counts(
                    [r for r, m in zip(references, mask) if m],
                    [h for h, m in zip(hyps, mask) if m],
                )
            )
        return out

    base_wers = {}
    for system_name, hypotheses in transcripts.items():
        base_wers[system_name] = wer_by_scope(*aligned(subset, hypotheses, baseline))

    rows = []
    for rule in base_rules.keys():
        overrides = {rule: not bool(base_rules[rule])}
        flipped = build_normalizer("custom", overrides)
        for system_name, hypotheses in transcripts.items():
            flip_wers = wer_by_scope(*aligned(subset, hypotheses, flipped))
            for scope, flip_wer in flip_wers.items():
                base_wer = base_wers[system_name][scope]
                rows.append(
                    {
                        "rule": rule,
                        "default": bool(base_rules[rule]),
                        "system": system_name,
                        "dataset": scope,
                        "wer_default": 100 * base_wer,
                        "wer_flipped": 100 * flip_wer,
                        "delta": 100 * (flip_wer - base_wer),
                    }
                )
    return pd.DataFrame(rows)


def pairwise(cfg, subset, transcripts, normalizers):
    """Is the gap between two systems bigger than the sample can resolve?

    Run for every metric, because the answer differs: a sample that cannot
    separate two systems on WER may separate them cleanly on how often they
    fail badly, which is the more decision-relevant of the two.

    Run pooled *and* per dataset, and record how many comparisons were made
    so that `separated` is read in context. That exists because of a measured
    failure: on 394 AMI utterances this comparison separated Parakeet from
    Qwen3 at p=0.023, and the full 7,805-utterance split reversed the sign.
    Two reasons it was wrong, both visible in this table -- the gap was
    smaller than the sample could resolve, and it was one of 40 comparisons,
    a couple of which are expected to clear p<0.05 by luck.
    """
    rows = []
    for norm_name, normalizer in normalizers.items():
        for metric, function in METRICS.items():
            pairs_by_system, datasets = {}, None
            for system_name, hypotheses in transcripts.items():
                references, hyps, datasets = aligned(subset, hypotheses, normalizer)
                pairs_by_system[system_name] = function(
                    references, hyps, **metric_kwargs(cfg, metric)
                )
            scopes = [("ALL", np.ones(len(datasets), dtype=bool))]
            scopes += [(d, datasets == d) for d in np.unique(datasets)]
            for scope, mask in scopes:
                for a, b in itertools.combinations(sorted(pairs_by_system), 2):
                    delta, lower, upper, p = paired_bootstrap(
                        pairs_by_system[a][mask],
                        pairs_by_system[b][mask],
                        cfg.scoring.bootstrap_samples,
                        cfg.scoring.confidence,
                        cfg.scoring.seed,
                    )
                    rows.append(
                        {
                            "metric": metric,
                            "normalizer": norm_name,
                            "dataset": scope,
                            "system_a": a,
                            "system_b": b,
                            "delta": 100 * delta,
                            "ci_low": 100 * lower,
                            "ci_high": 100 * upper,
                            "p_value": p,
                            "separated": bool(lower > 0 or upper < 0),
                        }
                    )
    frame = pd.DataFrame(rows)

    # Every pair x every scope within one metric and normalizer, because the
    # report quotes pooled and per-dataset claims interchangeably. Counting
    # only the pooled pairs would understate how many chances were taken.
    frame["n_comparisons"] = frame.groupby(
        ["metric", "normalizer"]
    ).p_value.transform("size")
    return frame


def log_rankings(table):
    """Rank under each metric. Where the orderings disagree, the leaderboard's
    single number is making a choice it does not disclose."""
    for norm_name in table.normalizer.unique():
        for metric in METRICS:
            overall = (
                table[
                    (table.normalizer == norm_name)
                    & (table.dataset == "ALL")
                    & (table.metric == metric)
                ]
                .sort_values("value")
                .reset_index(drop=True)
            )
            logger.info(f"ranking under '{norm_name}' by {metric}:")
            for rank, row in overall.iterrows():
                logger.info(
                    f"  {rank + 1}. {row.system:<26} "
                    f"{row.value:5.2f}  [{row.ci_low:.2f}, {row.ci_high:.2f}]"
                )


def main(cfg):
    logger.info(f"githash={cfg.githash}")
    subset = load_jsonl(SUBSET_FILE)
    transcripts = load_transcripts(cfg)
    logger.info(f"scoring {sorted(transcripts)} on {len(subset)} utterances")

    normalizers = {name: build_normalizer(name) for name in cfg.normalization}
    SCORES_DIR.mkdir(parents=True, exist_ok=True)

    table = score_table(cfg, subset, transcripts, normalizers)
    table.to_csv(SCORES_DIR / "metrics.csv", index=False)
    log_rankings(table)

    composition = composition_table(cfg, subset, transcripts, normalizers)
    composition.to_csv(SCORES_DIR / "error_composition.csv", index=False)

    comparison = published_comparison(cfg, table)
    if not comparison.empty:
        comparison.to_csv(SCORES_DIR / "published.csv", index=False)
        logger.info("ours vs published (leaderboard normalizer):")
        for _, row in comparison.iterrows():
            flag = "" if row.published_in_ci else "   <- outside our CI"
            logger.info(
                f"  {row.system:<24} {row.dataset:<32} "
                f"ours {row.ours:5.2f}  published {row.published:5.2f}  "
                f"gap {row.gap:+5.2f}{flag}"
            )

    ablation_df = ablation(cfg, subset, transcripts)
    ablation_df.to_csv(SCORES_DIR / "ablation.csv", index=False)
    logger.info("normalization rules, mean |delta WER| across systems, by dataset:")
    scoped = (
        ablation_df.assign(abs_delta=ablation_df.delta.abs())
        .pivot_table(index="rule", columns="dataset", values="abs_delta", aggfunc="mean")
        .sort_values("ALL", ascending=False)
    )
    datasets = ["ALL"] + [c for c in scoped.columns if c != "ALL"]
    logger.info("  " + f"{'rule':<22}" + "".join(f"{d[:18]:>20}" for d in datasets))
    for rule, row in scoped.iterrows():
        logger.info(
            f"  {rule:<22}" + "".join(f"{row[d]:>20.2f}" for d in datasets)
        )

    pairwise_df = pairwise(cfg, subset, transcripts, normalizers)
    pairwise_df.to_csv(SCORES_DIR / "pairwise.csv", index=False)
    for metric in METRICS:
        for norm_name in pairwise_df.normalizer.unique():
            rows = pairwise_df[
                (pairwise_df.metric == metric)
                & (pairwise_df.normalizer == norm_name)
            ]
            pooled = rows[rows.dataset == "ALL"]
            logger.info(
                f"{metric} [{norm_name}]: {int(pooled.separated.sum())} of "
                f"{len(pooled)} pooled pairs separated; "
                f"{int(rows.separated.sum())} of {len(rows)} including per-dataset "
                f"({len(rows)} comparisons taken)"
            )

    empty_df = empty_reference_counts(subset, normalizers)
    empty_df.to_csv(SCORES_DIR / "empty_references.csv", index=False)
    for _, row in empty_df[empty_df.n_empty > 0].iterrows():
        logger.warning(
            f"{row.normalizer} normalizer empties {row.n_empty} of {row.n} "
            f"{row.dataset} references ({row.pct:.2f}%) -- dropped from scoring"
        )

    logger.info(f"tables -> {SCORES_DIR}")


if __name__ == "__main__":
    githash = dirtygit.check()
    cfg = load_config(DEFAULT_CONFIG)
    OmegaConf.update(cfg, "githash", githash)
    setup_logging()
    main(cfg)
