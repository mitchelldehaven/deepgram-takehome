"""Diff two scoring arms on the same utterances, and rank where they disagree.

The score tables say *that* two arms differ. This says *where*. It decomposes
the gap between them into an exact per-utterance contribution, sorts by it,
and prints the text -- so the twenty lines responsible for a 0.6-point WER
difference can be read directly instead of inferred from an aggregate.

An arm is a (system, normalizer) pair, written `system:normalizer`, which
makes one tool answer three questions:

  one system, two normalizers   what a normalization rule actually changes
  two systems, one normalizer   where one recognizer beats another
  either, any of the metrics    where a hallucination or edge gap comes from

The decomposition is exact rather than a heuristic. Every metric here is
sum(numerators) / sum(denominators), so utterance i contributes

    n_a[i] / D_a  -  n_b[i] / D_b

to the gap between the arms, and those contributions sum to the gap exactly.
Doing it this way (rather than differencing per-utterance rates) matters
whenever the arms have different denominators -- which is always, when the
normalizers differ, because normalization changes the reference word count.

Usage:

    python src/eval/compare.py \
        compare.a=deepgram_nova_3_raw:leaderboard \
        compare.b=deepgram_nova_3_raw:custom

    python src/eval/compare.py \
        compare.a=parakeet_tdt_0_6b_v2 compare.b=qwen3_asr_1_7b \
        compare.metric=hallucination compare.dataset=AMI-Cleaned

A normalizer arm can also have individual rules flipped, which turns a row of
the ablation table into the utterances behind it:

    python src/eval/compare.py \
        compare.a=deepgram_nova_3_raw:custom \
        compare.b=deepgram_nova_3_raw:custom \
        compare.b_rules.remove_fillers=false
"""
import re

import dirtygit
import numpy as np
import pandas as pd
from jiwer import process_words
from loguru import logger
from omegaconf import OmegaConf

from src.config import load_config, setup_logging
from src.data.utils import load_jsonl
from src.eval.metrics import METRICS, paired_bootstrap
from src.eval.score import load_transcripts, metric_kwargs
from src.normalization.factory import build_normalizer
from src.paths import CONFIG_DIR, SCORES_DIR, SUBSET_FILE

DEFAULT_CONFIG = CONFIG_DIR / "eval" / "openasr.yaml"


def parse_arm(spec, default_normalizer):
    """`system:normalizer`, or bare `system` for the config's default."""
    system, _, normalizer = str(spec).partition(":")
    return system, normalizer or default_normalizer


def arm_rows(subset, hypotheses, normalizer):
    """Normalized rows for one arm, keyed by utterance id.

    The reference is normalized per arm, not once: when the arms differ in
    normalizer, the references differ too, and hiding that would misattribute
    the gap to the systems.
    """
    rows = {}
    for row in subset:
        if row["id"] not in hypotheses:
            continue
        rows[row["id"]] = {
            "dataset": row["dataset"],
            "reference": normalizer(row["reference"]),
            "hypothesis": normalizer(hypotheses[row["id"]]),
        }
    return rows


def mark(reference, hypothesis):
    """A hypothesis rendered against its reference, errors marked inline.

        {ref>hyp}   substitution
        [-word]     deletion
        [+word]     insertion

    Reading this next to the raw text is usually enough to classify a failure
    in one glance -- a run of [-...] at the end is a truncated endpoint, a run
    of [+...] is a decoder that kept going.
    """
    ref_words = str(reference).split()
    hyp_words = str(hypothesis).split()
    if not ref_words:
        return " ".join(f"[+{w}]" for w in hyp_words)

    result = process_words([" ".join(ref_words)], [" ".join(hyp_words) or " "])
    out = []
    for chunk in result.alignments[0]:
        ref_span = ref_words[chunk.ref_start_idx : chunk.ref_end_idx]
        hyp_span = hyp_words[chunk.hyp_start_idx : chunk.hyp_end_idx]
        if chunk.type == "equal":
            out.extend(ref_span)
        elif chunk.type == "substitute":
            out.extend(f"{{{r}>{h}}}" for r, h in zip(ref_span, hyp_span))
        elif chunk.type == "delete":
            out.extend(f"[-{w}]" for w in ref_span)
        else:
            out.extend(f"[+{w}]" for w in hyp_span)
    return " ".join(out)


def rate(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else float("nan")


def compare_table(cfg, rows_a, rows_b, metric):
    """Per-utterance contributions to the gap, largest absolute first.

    `contribution` is in the same units as the headline (percentage points)
    and sums to the headline gap, so a reader can check the top rows account
    for as much of it as the prose claims they do.
    """
    shared = [i for i in rows_a if i in rows_b]
    if not shared:
        raise RuntimeError("the two arms share no utterances")

    function = METRICS[metric]
    kwargs = metric_kwargs(cfg, metric)
    pairs_a = function([rows_a[i]["reference"] for i in shared],
                       [rows_a[i]["hypothesis"] for i in shared], **kwargs)
    pairs_b = function([rows_b[i]["reference"] for i in shared],
                       [rows_b[i]["hypothesis"] for i in shared], **kwargs)

    total_a = pairs_a[:, 1].sum()
    total_b = pairs_b[:, 1].sum()

    table = pd.DataFrame(
        {
            "id": shared,
            "dataset": [rows_a[i]["dataset"] for i in shared],
            "contribution": 100 * (
                pairs_a[:, 0] / (total_a or np.nan)
                - pairs_b[:, 0] / (total_b or np.nan)
            ),
            "num_a": pairs_a[:, 0],
            "den_a": pairs_a[:, 1],
            "num_b": pairs_b[:, 0],
            "den_b": pairs_b[:, 1],
            "rate_a": [100 * rate(n, d) for n, d in pairs_a],
            "rate_b": [100 * rate(n, d) for n, d in pairs_b],
            "reference_a": [rows_a[i]["reference"] for i in shared],
            "reference_b": [rows_b[i]["reference"] for i in shared],
            "hypothesis_a": [rows_a[i]["hypothesis"] for i in shared],
            "hypothesis_b": [rows_b[i]["hypothesis"] for i in shared],
        }
    )
    table = table.reindex(
        table.contribution.abs().sort_values(ascending=False).index
    ).reset_index(drop=True)
    return table, pairs_a, pairs_b


def log_headline(cfg, label_a, label_b, metric, pairs_a, pairs_b):
    value_a = 100 * rate(pairs_a[:, 0].sum(), pairs_a[:, 1].sum())
    value_b = 100 * rate(pairs_b[:, 0].sum(), pairs_b[:, 1].sum())
    delta, lower, upper, p = paired_bootstrap(
        pairs_a,
        pairs_b,
        cfg.scoring.bootstrap_samples,
        cfg.scoring.confidence,
        cfg.scoring.seed,
    )
    logger.info(f"{metric} over {len(pairs_a)} shared utterances")
    logger.info(f"  A  {label_a:<42} {value_a:6.2f}  (denominator {pairs_a[:, 1].sum()})")
    logger.info(f"  B  {label_b:<42} {value_b:6.2f}  (denominator {pairs_b[:, 1].sum()})")
    logger.info(
        f"  A - B  {100 * delta:+6.2f}  "
        f"[{100 * lower:+.2f}, {100 * upper:+.2f}]  p={p:.4f}  "
        f"{'separated' if (lower > 0 or upper < 0) else 'not separated'}"
    )


def log_disagreements(table, top):
    """The utterances behind the gap, worst first, with the text.

    Both references are printed when they differ, which is the whole point of
    a normalizer diff: the reference moving is often the entire effect, and an
    aggregate table cannot show that.
    """
    head = table.head(top)
    share = head.contribution.sum()
    total = table.contribution.sum()
    logger.info(
        f"top {len(head)} of {len(table)} utterances account for "
        f"{share:+.2f} of the {total:+.2f} gap"
    )
    for _, row in head.iterrows():
        logger.info("")
        logger.info(
            f"  {row.id}  [{row.dataset}]  contribution {row.contribution:+.3f}  "
            f"A {row.rate_a:.0f} vs B {row.rate_b:.0f}  "
            f"({row.num_a}/{row.den_a} vs {row.num_b}/{row.den_b})"
        )
        logger.info(f"    ref   {row.reference_a}")
        if row.reference_b != row.reference_a:
            logger.info(f"    ref*  {row.reference_b}   <- B's reference differs")
        logger.info(f"    A     {mark(row.reference_a, row.hypothesis_a)}")
        logger.info(f"    B     {mark(row.reference_b, row.hypothesis_b)}")


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def build_arm(cfg, spec, rules_key, subset, transcripts, default_normalizer):
    system, normalizer_name = parse_arm(spec, default_normalizer)
    if system not in transcripts:
        raise RuntimeError(
            f"no transcripts for '{system}'; have {sorted(transcripts)}"
        )
    overrides = cfg.compare.get(rules_key)
    overrides = OmegaConf.to_container(overrides) if overrides else None
    if overrides and normalizer_name != "custom":
        raise RuntimeError(f"{rules_key} only applies to the custom normalizer")
    normalizer = build_normalizer(normalizer_name, overrides)
    label = f"{system}:{normalizer_name}"
    if overrides:
        label += "+" + ",".join(f"{k}={v}" for k, v in sorted(overrides.items()))
    return label, arm_rows(subset, transcripts[system], normalizer)


def main(cfg):
    logger.info(f"githash={cfg.githash}")
    metric = cfg.compare.metric
    if metric not in METRICS:
        raise RuntimeError(f"unknown metric '{metric}'; have {sorted(METRICS)}")

    subset = load_jsonl(SUBSET_FILE)
    if cfg.compare.get("dataset"):
        subset = [row for row in subset if row["dataset"] == cfg.compare.dataset]
        if not subset:
            raise RuntimeError(f"no utterances in dataset '{cfg.compare.dataset}'")
    transcripts = load_transcripts(cfg)

    default_normalizer = cfg.normalization[0]
    label_a, rows_a = build_arm(
        cfg, cfg.compare.a, "a_rules", subset, transcripts, default_normalizer
    )
    label_b, rows_b = build_arm(
        cfg, cfg.compare.b, "b_rules", subset, transcripts, default_normalizer
    )
    if label_a == label_b:
        logger.warning("both arms resolve to the same thing; the gap will be zero")

    table, pairs_a, pairs_b = compare_table(cfg, rows_a, rows_b, metric)
    log_headline(cfg, label_a, label_b, metric, pairs_a, pairs_b)
    log_disagreements(table, cfg.compare.top)

    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    out = SCORES_DIR / f"compare_{slug(label_a)}_vs_{slug(label_b)}_{metric}.csv"
    table.to_csv(out, index=False)
    logger.info("")
    logger.info(f"all {len(table)} utterances -> {out}")


if __name__ == "__main__":
    githash = dirtygit.check()
    cfg = load_config(DEFAULT_CONFIG)
    OmegaConf.update(cfg, "githash", githash)
    setup_logging()
    main(cfg)
