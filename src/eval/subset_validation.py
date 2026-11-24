"""Measure what the 1200-utterance subset actually cost, against the full splits.

Section 1 of REPORT.md argues that 400 utterances per dataset is enough, and
that the +0.58 gap to the published numbers is not sampling. Both are
arguments. This measures them, by running the identical pipeline over every
row of all three test splits (~65,900 utterances, ~144 h) and comparing.

The comparison is a chain rather than a single number, because three different
things separate our subset result from the published one and they can be
isolated one at a time:

    A  subset ids, subset run      the number in REPORT.md
    B  subset ids, full run        identical audio, identical config, but each
                                   utterance sat in a different batch
    C  all ids,    full run        the whole split
    D  published

  A -> B  is batch composition. Nothing about the audio or the configuration
          differs; only which other utterances shared the forward pass.
          REPORT.md asserts that this matters for Whisper. Here it is
          measured instead.
  B -> C  is sampling, cleanly isolated: same transcription run, the only
          difference is which rows are scored.
  C -> D  is whatever is left -- decoding configuration, model version,
          anything else. Sampling cannot appear in this term, which is the
          point of the exercise.

The headline check is separate from the chain: does the full-split WER fall
inside the confidence interval that the 400-row sample predicted? That is a
direct calibration test of every interval in the report.

Run after build_subset/transcribe/score with RUN_TAG=full; it reads both runs
and needs no tag of its own. See config/eval/full.yaml.
"""
import os

import dirtygit
import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import OmegaConf

from src.config import load_config, setup_logging
from src.data.utils import load_jsonl
from src.eval.metrics import bootstrap_ci, counts, paired_bootstrap
from src.normalization.factory import build_normalizer
from src.paths import CONFIG_DIR, PUBLISHED_FILE, ROOT_DIR

DEFAULT_CONFIG = CONFIG_DIR / "eval" / "full.yaml"

# Both runs' artefacts, by the RUN_TAG convention in src/paths.py.
FULL_TAG = os.environ.get("FULL_TAG", "full")
SUBSET_PROCESSED = ROOT_DIR / "data" / "processed"
SUBSET_OUTPUTS = ROOT_DIR / "outputs"
FULL_PROCESSED = ROOT_DIR / "data" / f"processed-{FULL_TAG}"
FULL_OUTPUTS = ROOT_DIR / f"outputs-{FULL_TAG}"


def load_run(processed_dir, outputs_dir, systems):
    """(rows by id, {system: {id: hypothesis}}) for one run."""
    subset_file = processed_dir / "subset.jsonl"
    if not subset_file.exists():
        raise RuntimeError(f"{subset_file} missing; run build_subset.py for that run")
    rows = {row["id"]: row for row in load_jsonl(subset_file)}

    hypotheses = {}
    for name in systems:
        path = outputs_dir / "transcripts" / f"{name}.jsonl"
        if not path.exists():
            logger.warning(f"{name}: no transcripts at {path}, skipping")
            continue
        hypotheses[name] = {r["id"]: r["hypothesis"] for r in load_jsonl(path)}
    if not hypotheses:
        raise RuntimeError(f"no transcripts under {outputs_dir}")
    return rows, hypotheses


def pairs_for(ids, rows, hypotheses, normalizer):
    """Per-utterance (errors, reference words) for a given set of ids.

    Drops utterances whose reference normalizes to empty, exactly as
    `score.aligned` does. This function duplicated that logic and originally
    duplicated the omission too, which is worth noting: the empty-reference
    bug was found precisely *because* two implementations of the same
    quantity disagreed, and then it had to be fixed twice. The lesson is that
    a second implementation is a good cross-check and a bad place to keep
    shared logic.
    """
    references, hyps = [], []
    for i in ids:
        reference = normalizer(rows[i]["reference"])
        if not reference.strip():
            continue
        references.append(reference)
        hyps.append(normalizer(hypotheses[i]))
    return counts(references, hyps)


def wer_of(pairs):
    denominator = pairs[:, 1].sum()
    return 100 * pairs[:, 0].sum() / denominator if denominator else float("nan")


def main(cfg):
    logger.info(f"githash={cfg.githash}")
    systems = [s.name for s in cfg.systems]
    normalizer = build_normalizer("leaderboard")

    subset_rows, subset_hyps = load_run(SUBSET_PROCESSED, SUBSET_OUTPUTS, systems)
    full_rows, full_hyps = load_run(FULL_PROCESSED, FULL_OUTPUTS, systems)
    shared = [i for i in subset_rows if i in full_rows]
    logger.info(
        f"subset run: {len(subset_rows)} utterances; full run: {len(full_rows)}; "
        f"subset ids present in the full run: {len(shared)}"
    )
    if len(shared) != len(subset_rows):
        logger.warning(
            f"{len(subset_rows) - len(shared)} subset ids are absent from the full "
            "run, which should not happen -- the subset is drawn from these splits"
        )

    published = (
        pd.read_csv(PUBLISHED_FILE).set_index("model") if PUBLISHED_FILE.exists() else None
    )
    by_leaderboard_id = {s.name: s.get("leaderboard_id") for s in cfg.systems}

    datasets = list(dict.fromkeys(row["dataset"] for row in full_rows.values()))
    rows = []
    for name in sorted(set(subset_hyps) & set(full_hyps)):
        for dataset in datasets + ["ALL"]:
            def keep(i, pool):
                return dataset == "ALL" or pool[i]["dataset"] == dataset

            subset_ids = [i for i in shared if keep(i, subset_rows)]
            full_ids = [i for i in full_rows if keep(i, full_rows)]
            if not subset_ids or not full_ids:
                continue

            a = pairs_for(subset_ids, subset_rows, subset_hyps[name], normalizer)
            b = pairs_for(subset_ids, full_rows, full_hyps[name], normalizer)
            c = pairs_for(full_ids, full_rows, full_hyps[name], normalizer)

            _, a_low, a_high = bootstrap_ci(
                a,
                cfg.scoring.bootstrap_samples,
                cfg.scoring.confidence,
                cfg.scoring.seed,
            )
            _, c_low, c_high = bootstrap_ci(
                c,
                cfg.scoring.bootstrap_samples,
                cfg.scoring.confidence,
                cfg.scoring.seed,
            )
            # A vs B is paired: identical utterances, two transcription runs.
            delta_batch, _, _, p_batch = paired_bootstrap(
                a,
                b,
                cfg.scoring.bootstrap_samples,
                cfg.scoring.confidence,
                cfg.scoring.seed,
            )

            leaderboard_id = by_leaderboard_id.get(name)
            theirs = float("nan")
            if (
                published is not None
                and leaderboard_id in getattr(published, "index", [])
                and dataset in published.columns
            ):
                theirs = pd.to_numeric(
                    published.loc[leaderboard_id, dataset], errors="coerce"
                )

            rows.append(
                {
                    "system": name,
                    "dataset": dataset,
                    "n_subset": len(subset_ids),
                    "n_full": len(full_ids),
                    "a_subset_run": wer_of(a),
                    "a_ci_low": 100 * a_low,
                    "a_ci_high": 100 * a_high,
                    "b_subset_ids_full_run": wer_of(b),
                    "c_full_split": wer_of(c),
                    "c_ci_low": 100 * c_low,
                    "c_ci_high": 100 * c_high,
                    "d_published": theirs,
                    "batch_effect_a_to_b": 100 * delta_batch,
                    "p_batch_effect": p_batch,
                    "sampling_effect_b_to_c": wer_of(b) - wer_of(c),
                    "residual_c_to_d": wer_of(c) - theirs,
                    "full_inside_subset_ci": bool(
                        100 * a_low <= wer_of(c) <= 100 * a_high
                    ),
                }
            )

    table = pd.DataFrame(rows)
    out_dir = FULL_OUTPUTS / "scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "subset_validation.csv", index=False)

    per_dataset = table[table.dataset != "ALL"]
    inside = per_dataset.full_inside_subset_ci.sum()
    logger.info("")
    logger.info(
        f"CALIBRATION: the full-split WER falls inside the 400-row 95% CI in "
        f"{inside} of {len(per_dataset)} system x dataset cells"
    )
    logger.info("")
    logger.info("the chain, per system x dataset (leaderboard normalizer):")
    logger.info(
        f"  {'system':<22} {'dataset':<20} {'A':>6} {'B':>6} {'C':>6} {'D':>6}   "
        f"{'A->B':>6} {'B->C':>6} {'C->D':>6}  in CI"
    )
    for _, r in table.iterrows():
        logger.info(
            f"  {r.system:<22} {r.dataset:<20} "
            f"{r.a_subset_run:6.2f} {r.b_subset_ids_full_run:6.2f} "
            f"{r.c_full_split:6.2f} {r.d_published:6.2f}   "
            f"{r.batch_effect_a_to_b:+6.2f} {r.sampling_effect_b_to_c:+6.2f} "
            f"{r.residual_c_to_d:+6.2f}  {r.full_inside_subset_ci}"
        )

    logger.info("")
    for label, column in [
        ("batch composition (A->B)", "batch_effect_a_to_b"),
        ("sampling (B->C)", "sampling_effect_b_to_c"),
        ("residual, config/version (C->D)", "residual_c_to_d"),
    ]:
        values = per_dataset[column].dropna()
        logger.info(
            f"  {label:<34} mean {values.mean():+.2f}  "
            f"mean |.| {values.abs().mean():.2f}  "
            f"max |.| {values.abs().max():.2f}"
        )

    # Per cell, not a ratio of mean widths. The three splits grow by very
    # different factors (AMI ~20x, GigaSpeech ~47x, SPGISpeech ~98x), and the
    # widest intervals are the ones that grow least, so dividing the mean
    # widths understates the tightening badly -- 5.2x against a true ~6.8x,
    # which is easy to misread as evidence that errors cluster. They do not:
    # per cell the scaling is 0.96 of sqrt(n).
    width_subset = (per_dataset.a_ci_high - per_dataset.a_ci_low).mean()
    width_full = (per_dataset.c_ci_high - per_dataset.c_ci_low).mean()
    per_cell = (
        (per_dataset.a_ci_high - per_dataset.a_ci_low)
        / (per_dataset.c_ci_high - per_dataset.c_ci_low)
    )
    predicted = (per_dataset.n_full / per_dataset.n_subset) ** 0.5
    logger.info("")
    logger.info(
        f"  mean 95% CI width: {width_subset:.2f} WER on 400 rows, "
        f"{width_full:.2f} on the full split"
    )
    logger.info(
        f"  per cell, intervals tighten {per_cell.mean():.2f}x against "
        f"sqrt(n)'s {predicted.mean():.2f}x = {(per_cell / predicted).mean():.2f} "
        f"of sqrt(n); naively dividing the mean widths would say "
        f"{width_subset / width_full:.1f}x"
    )
    logger.info(f"table -> {out_dir / 'subset_validation.csv'}")


if __name__ == "__main__":
    githash = dirtygit.check()
    cfg = load_config(DEFAULT_CONFIG)
    OmegaConf.update(cfg, "githash", githash)
    setup_logging()
    main(cfg)
