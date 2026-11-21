"""Score the cached transcripts, under each normalizer, against the published numbers.

Produces four tables in outputs/scores/:
  wer.csv          system x dataset x normalizer, with bootstrap CIs
  published.csv    ours vs the leaderboard's, and the gap
  ablation.csv     WER attributable to each individual normalization rule
  pairwise.csv     paired bootstrap on every system pair, per normalizer

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


def aligned(subset, hypotheses, normalizer):
    """Normalized (reference, hypothesis) pairs plus the dataset label, in order."""
    references, hyps, datasets = [], [], []
    for row in subset:
        if row["id"] not in hypotheses:
            continue
        references.append(normalizer(row["reference"]))
        hyps.append(normalizer(hypotheses[row["id"]]))
        datasets.append(row["dataset"])
    return references, hyps, np.array(datasets)


def score_table(cfg, subset, transcripts, normalizers):
    rows = []
    for norm_name, normalizer in normalizers.items():
        for system_name, hypotheses in transcripts.items():
            references, hyps, datasets = aligned(subset, hypotheses, normalizer)
            pairs = counts(references, hyps)
            for dataset in list(dict.fromkeys(datasets)) + ["ALL"]:
                mask = slice(None) if dataset == "ALL" else (datasets == dataset)
                subset_pairs = pairs[mask]
                point, lower, upper = bootstrap_ci(
                    subset_pairs,
                    cfg.scoring.bootstrap_samples,
                    cfg.scoring.confidence,
                    cfg.scoring.seed,
                )
                errors = error_breakdown(
                    list(np.array(references)[mask]), list(np.array(hyps)[mask])
                )
                rows.append(
                    {
                        "normalizer": norm_name,
                        "system": system_name,
                        "dataset": dataset,
                        "n": len(subset_pairs),
                        "wer": 100 * point,
                        "ci_low": 100 * lower,
                        "ci_high": 100 * upper,
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
            (wer_df.system == system_cfg.name)
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
                    "ours": row.wer,
                    "ci_low": row.ci_low,
                    "ci_high": row.ci_high,
                    "published": theirs,
                    "gap": row.wer - theirs,
                    "published_in_ci": bool(row.ci_low <= theirs <= row.ci_high),
                }
            )
    return pd.DataFrame(rows)


def ablation(cfg, subset, transcripts):
    """Flip one normalization rule at a time; report the WER it is worth.

    Measured against our custom normalizer, not the leaderboard's, so each
    number is the marginal effect of that one rule rather than of the whole
    bundle of differences at once.
    """
    base_rules = load_rules()
    baseline = build_normalizer("custom")
    base_wers = {}
    for system_name, hypotheses in transcripts.items():
        base_refs, base_hyps, _ = aligned(subset, hypotheses, baseline)
        base_wers[system_name] = wer_from_counts(counts(base_refs, base_hyps))

    rows = []
    for rule in base_rules.keys():
        overrides = {rule: not bool(base_rules[rule])}
        flipped = build_normalizer("custom", overrides)
        for system_name, hypotheses in transcripts.items():
            flip_refs, flip_hyps, _ = aligned(subset, hypotheses, flipped)
            base_wer = base_wers[system_name]
            flip_wer = wer_from_counts(counts(flip_refs, flip_hyps))
            rows.append(
                {
                    "rule": rule,
                    "default": bool(base_rules[rule]),
                    "system": system_name,
                    "wer_default": 100 * base_wer,
                    "wer_flipped": 100 * flip_wer,
                    "delta": 100 * (flip_wer - base_wer),
                }
            )
    return pd.DataFrame(rows)


def pairwise(cfg, subset, transcripts, normalizers):
    """Is the gap between two systems bigger than the sample can resolve?"""
    rows = []
    for norm_name, normalizer in normalizers.items():
        pairs_by_system = {}
        for system_name, hypotheses in transcripts.items():
            references, hyps, _ = aligned(subset, hypotheses, normalizer)
            pairs_by_system[system_name] = counts(references, hyps)
        for a, b in itertools.combinations(sorted(pairs_by_system), 2):
            delta, lower, upper, p = paired_bootstrap(
                pairs_by_system[a],
                pairs_by_system[b],
                cfg.scoring.bootstrap_samples,
                cfg.scoring.confidence,
                cfg.scoring.seed,
            )
            rows.append(
                {
                    "normalizer": norm_name,
                    "system_a": a,
                    "system_b": b,
                    "delta_wer": 100 * delta,
                    "ci_low": 100 * lower,
                    "ci_high": 100 * upper,
                    "p_value": p,
                    "separated": bool(lower > 0 or upper < 0),
                }
            )
    return pd.DataFrame(rows)


def log_rankings(wer_df):
    for norm_name in wer_df.normalizer.unique():
        overall = (
            wer_df[(wer_df.normalizer == norm_name) & (wer_df.dataset == "ALL")]
            .sort_values("wer")
            .reset_index(drop=True)
        )
        logger.info(f"ranking under '{norm_name}':")
        for rank, row in overall.iterrows():
            logger.info(
                f"  {rank + 1}. {row.system:<24} "
                f"{row.wer:5.2f}  [{row.ci_low:.2f}, {row.ci_high:.2f}]"
            )


def main(cfg):
    logger.info(f"githash={cfg.githash}")
    subset = load_jsonl(SUBSET_FILE)
    transcripts = load_transcripts(cfg)
    logger.info(f"scoring {sorted(transcripts)} on {len(subset)} utterances")

    normalizers = {name: build_normalizer(name) for name in cfg.normalization}
    SCORES_DIR.mkdir(parents=True, exist_ok=True)

    wer_df = score_table(cfg, subset, transcripts, normalizers)
    wer_df.to_csv(SCORES_DIR / "wer.csv", index=False)
    log_rankings(wer_df)

    comparison = published_comparison(cfg, wer_df)
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
    logger.info("normalization rules, mean |delta WER| across systems:")
    summary = (
        ablation_df.assign(abs_delta=ablation_df.delta.abs())
        .groupby("rule")
        .abs_delta.mean()
        .sort_values(ascending=False)
    )
    for rule, value in summary.items():
        logger.info(f"  {rule:<22} {value:5.2f}")

    pairwise_df = pairwise(cfg, subset, transcripts, normalizers)
    pairwise_df.to_csv(SCORES_DIR / "pairwise.csv", index=False)
    unresolved = pairwise_df[~pairwise_df.separated]
    if len(unresolved):
        logger.info("pairs this sample cannot separate:")
        for _, row in unresolved.iterrows():
            logger.info(
                f"  [{row.normalizer}] {row.system_a} vs {row.system_b}: "
                f"{row.delta_wer:+.2f} [{row.ci_low:+.2f}, {row.ci_high:+.2f}] p={row.p_value:.3f}"
            )

    logger.info(f"tables -> {SCORES_DIR}")


if __name__ == "__main__":
    githash = dirtygit.check()
    cfg = load_config(DEFAULT_CONFIG)
    OmegaConf.update(cfg, "githash", githash)
    setup_logging()
    main(cfg)
