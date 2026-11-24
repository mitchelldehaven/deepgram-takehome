"""Did padding Nova-3's input with silence recover the words it was dropping?

Nova-3's AMI deficit against Parakeet is entirely omission: 6.52 deletions per
100 edge slots against 2.59, with insertion rates identical. Two explanations
fit. Either the API's endpointing truncates against a hard cut -- these
utterances are slices out of continuous recordings, so the first and last
words can begin mid-word at sample zero -- or the words are not in the audio
and no amount of context recovers them.

Padding with digital silence separates the two, because it adds runway without
adding information. If edge deletions fall, the loss was an artifact of the
cut. If they do not, the audio is the limit.

The probe is two-sided on purpose. Silence is also where a model is freest to
hallucinate, and Nova-3 demonstrably does: on the five AMI utterances with
over 1.5 s of edge silence it emits 3.0 words beyond the reference on average.
So this reports insertions alongside deletions, and WER alongside both. A fix
that trades deletions for hallucinations is not a fix.

SPGISpeech is the control. Its segments are cleanly cut from single-speaker
earnings calls, so padding should do nothing there. If it helps everywhere
equally, the effect is not about segment boundaries and this whole reading is
wrong.
"""
import dirtygit
import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import OmegaConf

from src.config import load_config, setup_logging
from src.data.utils import load_jsonl
from src.eval.metrics import (
    counts,
    edge_breakdown,
    paired_bootstrap,
    wer_from_counts,
)
from src.eval.score import aligned
from src.normalization.factory import build_normalizer
from src.paths import CONFIG_DIR, SCORES_DIR, SUBSET_FILE, TRANSCRIPTS_DIR

DEFAULT_CONFIG = CONFIG_DIR / "eval" / "padding_probe.yaml"


def load_hypotheses(name):
    path = TRANSCRIPTS_DIR / f"{name}.jsonl"
    if not path.exists():
        return None
    return {row["id"]: row["hypothesis"] for row in load_jsonl(path)}


def condition_table(subset, transcripts, normalizer):
    rows = []
    for name, hypotheses in transcripts.items():
        references, hyps, datasets = aligned(subset, hypotheses, normalizer)
        for dataset in list(dict.fromkeys(datasets)) + ["ALL"]:
            mask = slice(None) if dataset == "ALL" else (datasets == dataset)
            refs = list(np.array(references)[mask])
            hs = list(np.array(hyps)[mask])
            edge, middle, edge_slots, middle_slots = edge_breakdown(refs, hs)
            rows.append(
                {
                    "condition": name,
                    "dataset": dataset,
                    "n": len(refs),
                    "wer": 100 * wer_from_counts(counts(refs, hs)),
                    "edge_del": 100 * edge["del"] / max(edge_slots, 1),
                    "edge_ins": 100 * edge["ins"] / max(edge_slots, 1),
                    "edge_sub": 100 * edge["sub"] / max(edge_slots, 1),
                    "mid_del": 100 * middle["del"] / max(middle_slots, 1),
                    "mid_ins": 100 * middle["ins"] / max(middle_slots, 1),
                }
            )
    return pd.DataFrame(rows)


def main(cfg):
    logger.info(f"githash={cfg.githash}")
    subset = load_jsonl(SUBSET_FILE)
    normalizer = build_normalizer("leaderboard")

    names = [cfg.baseline] + [s.name for s in cfg.systems]
    transcripts = {}
    for name in names:
        hypotheses = load_hypotheses(name)
        if hypotheses is None:
            logger.warning(f"{name}: no transcripts; run transcribe.py for it first")
            continue
        transcripts[name] = hypotheses
    if cfg.baseline not in transcripts:
        raise RuntimeError(f"baseline {cfg.baseline} has no transcripts to compare against")

    table = condition_table(subset, transcripts, normalizer)
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(SCORES_DIR / "padding_probe.csv", index=False)

    for dataset in ["AMI-Cleaned", "Gigaspeech-Cleaned", "SPGISpeech", "ALL"]:
        rows = table[table.dataset == dataset]
        if rows.empty:
            continue
        logger.info(f"-- {dataset}")
        logger.info("   condition                   WER   edge_del  edge_ins   mid_del")
        for _, row in rows.iterrows():
            logger.info(
                f"   {row.condition:<24} {row.wer:6.2f}   {row.edge_del:6.2f}   "
                f"{row.edge_ins:6.2f}   {row.mid_del:6.2f}"
            )

    # Paired against the unpadded run on the same utterances, so the shared
    # difficulty cancels and the interval is on the change itself.
    base_refs, base_hyps, base_ds = aligned(subset, transcripts[cfg.baseline], normalizer)
    base_pairs = counts(base_refs, base_hyps)
    logger.info("change in WER vs the unpadded run (paired bootstrap):")
    deltas = []
    for name in [s.name for s in cfg.systems if s.name in transcripts]:
        refs, hyps, _ = aligned(subset, transcripts[name], normalizer)
        pairs = counts(refs, hyps)
        # The masks below index `pairs` with the baseline's dataset labels, so
        # the two runs must have retained the same utterances in the same
        # order. They do unless a condition returned no hypothesis for some
        # id, which would silently shift every comparison by one row -- so
        # fail loudly instead of scoring misaligned pairs.
        if len(pairs) != len(base_pairs):
            raise RuntimeError(
                f"{name} retained {len(pairs)} utterances against the "
                f"baseline's {len(base_pairs)}; cannot pair them"
            )
        for dataset in list(dict.fromkeys(base_ds)) + ["ALL"]:
            mask = slice(None) if dataset == "ALL" else (base_ds == dataset)
            delta, lower, upper, p = paired_bootstrap(
                pairs[mask],
                base_pairs[mask],
                cfg.scoring.bootstrap_samples,
                cfg.scoring.confidence,
                cfg.scoring.seed,
            )
            deltas.append(
                {
                    "condition": name,
                    "dataset": dataset,
                    "delta_wer": 100 * delta,
                    "ci_low": 100 * lower,
                    "ci_high": 100 * upper,
                    "p_value": p,
                    "separated": bool(lower > 0 or upper < 0),
                }
            )
            verdict = "" if (lower > 0 or upper < 0) else "   (not resolved)"
            logger.info(
                f"   {name:<24} {dataset:<20} {100 * delta:+6.2f} "
                f"[{100 * lower:+.2f}, {100 * upper:+.2f}] p={p:.3f}{verdict}"
            )
    pd.DataFrame(deltas).to_csv(SCORES_DIR / "padding_probe_deltas.csv", index=False)
    logger.info(f"tables -> {SCORES_DIR}")


if __name__ == "__main__":
    githash = dirtygit.check()
    cfg = load_config(DEFAULT_CONFIG)
    OmegaConf.update(cfg, "githash", githash)
    setup_logging()
    main(cfg)
