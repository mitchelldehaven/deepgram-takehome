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
from src.eval.metrics import METRICS, bootstrap_ci, counts, edge_breakdown
from src.eval.score import aligned, metric_kwargs, load_transcripts, retained
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
        kept = retained(subset, hypotheses, normalizer)

        for metric, function in METRICS.items():
            pairs = function(references, hyps, **metric_kwargs(cfg, metric))

            for field, source, is_reliable in slice_fields(reliability):
                all_values = np.array(
                    [by_id[row["id"]][source] for row in kept], dtype=float
                )
                # Pooled *and* within each dataset. Pooling alone is
                # misleading here, because these fields are not independent of
                # the corpus: the shortest duration tercile is 70% AMI and 0%
                # SPGISpeech, and both nonzero numeric_rate bins are 100%
                # SPGISpeech. A pooled slice on those is a dataset slice
                # wearing a different label, and the only way to tell is to
                # bin inside each corpus and see whether the effect survives.
                scopes = [("ALL", np.ones(len(all_values), dtype=bool))]
                scopes += [
                    (name, datasets == name) for name in dict.fromkeys(datasets)
                ]
                for scope, scope_mask in scopes:
                    values = np.where(scope_mask, all_values, np.nan)
                    bins, labels = bin_values(all_values[scope_mask])
                    full = np.full(len(all_values), "n/a", dtype=object)
                    full[scope_mask] = bins
                    bins = full
                    for label in labels:
                        mask = (bins == label) & scope_mask
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
                                "dataset": scope,
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
                        "dataset": dataset,
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


# Fixed absolute-duration buckets, in seconds. Terciles cannot answer "is AMI
# harder than GigaSpeech at the same clip length", because the corpora barely
# overlap in duration -- AMI's longest tercile has a shorter median than
# SPGISpeech's shortest -- so each corpus's terciles cover a different range.
# Fixed edges put every corpus on one scale, at the cost of empty cells.
DURATION_BUCKETS = [(0, 2), (2, 4), (4, 8), (8, 16)]


def duration_buckets(subset, metadata, transcripts, normalizer):
    """WER per system x dataset x fixed duration bucket.

    This is what supports the claim that most of "AMI is the hard corpus" is
    utterance length rather than acoustics. Hand-building this table is how
    the effect-size table went stale; computed here so the claim is
    reproducible.
    """
    by_id = {row["id"]: row for row in metadata}
    rows = []
    for system_name, hypotheses in transcripts.items():
        kept = retained(subset, hypotheses, normalizer)
        references = [normalizer(row["reference"]) for row in kept]
        hyps = [normalizer(hypotheses[row["id"]]) for row in kept]
        pairs = METRICS["wer"](references, hyps)
        durations = np.array([by_id[row["id"]]["duration_s"] for row in kept], float)
        datasets = np.array([row["dataset"] for row in kept])
        for dataset in dict.fromkeys(datasets):
            for low, high in DURATION_BUCKETS:
                mask = (datasets == dataset) & (durations >= low) & (durations < high)
                if mask.sum() < MIN_BIN_SIZE:
                    continue
                denominator = pairs[mask][:, 1].sum()
                rows.append(
                    {
                        "system": system_name,
                        "dataset": dataset,
                        "bucket": f"{low}-{high}s",
                        "n": int(mask.sum()),
                        "wer": 100 * pairs[mask][:, 0].sum() / denominator,
                    }
                )
    return pd.DataFrame(rows)


def duration_matched(subset, metadata, transcripts, normalizer, reference="Gigaspeech-Cleaned"):
    """Each dataset's WER reweighted to another's duration distribution.

    The bucket table says whether one corpus is harder than another *at a
    given length*; this says how much of the pooled difference that accounts
    for. AMI's per-bucket WER is reweighted by the reference corpus's share
    of reference words per bucket, which asks: what would AMI score if its
    clips were as long as GigaSpeech's?

    Worth computing rather than eyeballing the bucket table, because the
    buckets disagree across systems, and reading one system's row invites an
    over-generalisation: length is the whole AMI-GigaSpeech gap for four of
    the five systems, and a third of it for Whisper.
    """
    by_id = {row["id"]: row for row in metadata}
    rows = []
    for system_name, hypotheses in transcripts.items():
        kept = retained(subset, hypotheses, normalizer)
        pairs = METRICS["wer"](
            [normalizer(row["reference"]) for row in kept],
            [normalizer(hypotheses[row["id"]]) for row in kept],
        )
        durations = np.array([by_id[row["id"]]["duration_s"] for row in kept], float)
        datasets = np.array([row["dataset"] for row in kept])

        def wer_of(mask):
            denominator = pairs[mask][:, 1].sum()
            return 100 * pairs[mask][:, 0].sum() / denominator if denominator else np.nan

        in_bucket = lambda name, low, high: (
            (datasets == name) & (durations >= low) & (durations < high)
        )
        reference_wer = wer_of(datasets == reference)
        for dataset in dict.fromkeys(datasets):
            if dataset == reference:
                continue
            numerator = denominator = 0.0
            for low, high in DURATION_BUCKETS:
                here, there = in_bucket(dataset, low, high), in_bucket(reference, low, high)
                if here.sum() < MIN_BIN_SIZE or there.sum() < MIN_BIN_SIZE:
                    continue
                weight = pairs[there][:, 1].sum()
                numerator += wer_of(here) * weight
                denominator += weight
            if not denominator:
                continue
            matched = numerator / denominator
            observed = wer_of(datasets == dataset)
            rows.append(
                {
                    "system": system_name,
                    "dataset": dataset,
                    "reference_dataset": reference,
                    "wer": observed,
                    "reference_wer": reference_wer,
                    "gap": observed - reference_wer,
                    "wer_at_reference_durations": matched,
                    "gap_at_reference_durations": matched - reference_wer,
                    "points_from_duration": observed - matched,
                }
            )
    return pd.DataFrame(rows)

# Reference-length buckets for the decomposition below. Lower bound is
# MIN_REF_WORDS, since shorter utterances have no edge/middle split.
LENGTH_BUCKETS = [(6, 10), (10, 16), (16, 26), (26, 10_000)]


def length_decomposition(subset, transcripts, normalizer):
    """Why short utterances score so much worse, split into its two causes.

    Short utterances score far worse than long ones, and that is the largest
    single effect in the metadata. Two different things produce it and they
    need separating:

      1. Composition. The edge metric in src/eval/metrics.py counts errors in
         the first and last two reference words. For a 6-word utterance that
         is 4 of 6 word slots; for a 40-word one it is 4 of 40. A short
         utterance is *mostly* edge, and edges run about twice the error rate
         of middles at every length.
      2. Difficulty. Short utterances are also harder word-for-word in the
         middle, where composition cannot be the explanation.

    So this reports the edge rate, the middle rate and the edge share per
    bucket, plus a counterfactual: what the shortest bucket's WER would be if
    it had the *edge share* of the longest one but kept its own edge and
    middle rates. The difference between that and its real WER is the part of
    the short-utterance penalty that is geometry rather than difficulty.
    """
    rows = []
    for system_name, hypotheses in transcripts.items():
        references, hyps, _ = aligned(subset, hypotheses, normalizer)
        lengths = np.array([len(r.split()) for r in references])
        pairs = counts(references, hyps)
        for low, high in LENGTH_BUCKETS:
            mask = (lengths >= low) & (lengths < high)
            if mask.sum() < MIN_BIN_SIZE:
                continue
            kept_refs = [r for r, keep in zip(references, mask) if keep]
            kept_hyps = [h for h, keep in zip(hyps, mask) if keep]
            edge, middle, edge_slots, middle_slots = edge_breakdown(
                kept_refs, kept_hyps
            )
            slots = edge_slots + middle_slots
            rows.append(
                {
                    "system": system_name,
                    "ref_words": f"{low}-{high - 1}" if high < 10_000 else f"{low}+",
                    "n": int(mask.sum()),
                    "edge_share": 100 * edge_slots / slots if slots else float("nan"),
                    "wer": 100 * pairs[mask][:, 0].sum() / pairs[mask][:, 1].sum(),
                    "edge_error": 100 * sum(edge.values()) / edge_slots
                    if edge_slots
                    else float("nan"),
                    "middle_error": 100 * sum(middle.values()) / middle_slots
                    if middle_slots
                    else float("nan"),
                }
            )

    table = pd.DataFrame(rows)
    # The counterfactual, per system: shortest bucket's error rates, longest
    # bucket's shape.
    counterfactual = []
    for system_name, group in table.groupby("system"):
        short, long = group.iloc[0], group.iloc[-1]
        share = long.edge_share / 100
        implied = share * short.edge_error + (1 - share) * short.middle_error
        counterfactual.append(
            {
                "system": system_name,
                "short_bucket": short.ref_words,
                "long_bucket": long.ref_words,
                "short_wer": short.wer,
                "long_wer": long.wer,
                "short_wer_at_long_edge_share": implied,
                "gap": short.wer - long.wer,
                "gap_from_composition": short.wer - implied,
                "share_of_gap_from_composition": 100 * (short.wer - implied)
                / (short.wer - long.wer),
            }
        )
    return table, pd.DataFrame(counterfactual)


def effect_sizes(table, metrics_path):
    """How many WER points each clip property is worth, next to the model and
    normalizer effects, so "X matters" is a measured claim rather than an
    impression.

    The effect of a property is the gap between its worst and best bin,
    averaged over systems. That is deliberately the same unit as every other
    number in the report -- WER points -- rather than a ratio or a variance
    share, both of which are harder to interpret and neither of which answers
    "how much worse does this make my transcripts".

    The dataset label is excluded on purpose. It would top the table, but it
    is not a property of a clip: it is the name of the whole bundle of
    properties, so it cannot compete with its own contents.

    Hand-deriving this meant it silently went stale when scoring changed.
    Computed here so it cannot.
    """
    pooled = table[(table.dataset == "ALL") & (table.metric == "wer")]
    rows = [
        {"effect": field, "kind": "clip property", "wer_points": value}
        for field, value in (
            pooled.groupby(["field", "system"])
            .value.agg(lambda v: v.max() - v.min())
            .groupby("field")
            .mean()
            .sort_values(ascending=False)
            .items()
        )
    ]
    metrics = pd.read_csv(metrics_path)
    wer_rows = metrics[(metrics.metric == "wer") & (metrics.dataset == "ALL")]
    for norm_name, group in wer_rows.groupby("normalizer"):
        rows.append(
            {
                "effect": f"all system configurations ({norm_name} normalizer)",
                "kind": "for comparison",
                "wer_points": group.value.max() - group.value.min(),
            }
        )
    wide = wer_rows.pivot_table(index="system", columns="normalizer", values="value")
    if {"custom", "leaderboard"} <= set(wide.columns):
        rows.append(
            {
                "effect": "choice of normalizer",
                "kind": "for comparison",
                "wer_points": (wide["custom"] - wide["leaderboard"]).abs().mean(),
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

    buckets = duration_buckets(subset, metadata, transcripts, normalizer)
    buckets.to_csv(SCORES_DIR / "duration_buckets.csv", index=False)
    matched = duration_matched(subset, metadata, transcripts, normalizer)
    matched.to_csv(SCORES_DIR / "duration_matched.csv", index=False)
    logger.info("each corpus reweighted to GigaSpeech's duration distribution:")
    for _, row in matched.iterrows():
        logger.info(
            f"  {row.system:<26} {row.dataset:<20} {row.wer:5.2f} -> "
            f"{row.wer_at_reference_durations:5.2f}  "
            f"(gap {row.gap:+.2f} -> {row.gap_at_reference_durations:+.2f}; "
            f"{row.points_from_duration:+.2f} points were length)"
        )

    effects = effect_sizes(table, SCORES_DIR / "metrics.csv")
    effects.to_csv(SCORES_DIR / "effect_sizes.csv", index=False)
    logger.info("effect sizes (worst bin - best bin, mean over systems), WER points:")
    for _, row in effects.iterrows():
        logger.info(f"  {row.effect:<48} {row.wer_points:5.2f}  ({row.kind})")
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

    lengths, counterfactual = length_decomposition(subset, transcripts, normalizer)
    lengths.to_csv(SCORES_DIR / "length_decomposition.csv", index=False)
    counterfactual.to_csv(SCORES_DIR / "length_counterfactual.csv", index=False)
    logger.info("why short utterances score worse (leaderboard normalizer):")
    for _, row in lengths.iterrows():
        logger.info(
            f"  {row.system:<26} {row.ref_words:>6} words  n={row.n:<4} "
            f"edge_share={row.edge_share:5.1f}%  wer={row.wer:5.2f}  "
            f"edge={row.edge_error:5.2f}  middle={row.middle_error:5.2f}"
        )
    logger.info("how much of the short-long gap is composition, not difficulty:")
    for _, row in counterfactual.iterrows():
        logger.info(
            f"  {row.system:<26} gap {row.gap:5.2f} -> "
            f"{row.gap_from_composition:5.2f} from edge share "
            f"({row.share_of_gap_from_composition:4.0f}%)"
        )

    logger.info(f"tables -> {SCORES_DIR}")


if __name__ == "__main__":
    githash = dirtygit.check()
    cfg = load_config(DEFAULT_CONFIG)
    OmegaConf.update(cfg, "githash", githash)
    setup_logging()
    main(cfg)
