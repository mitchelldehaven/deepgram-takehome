"""Scoring primitives: four metrics that share one shape.

WER is what the leaderboard reports. The other three exist because a corpus
WER is an average, and an average answers "how wrong overall" rather than
"wrong how, and how often badly" -- which is what someone deciding whether to
ship a system is actually asking.

Every metric here is a function (references, hypotheses) -> per-utterance
array of (numerator, denominator). That one convention is what makes the suite
uniform: `bootstrap_ci` and `paired_bootstrap` work on any of them without
knowing which, so a new metric needs one function and one entry in METRICS to
arrive with confidence intervals and significance tests already attached.

The denominators differ on purpose, and that is the interesting part:

  wer            reference words      -- how much of the text is wrong
  catastrophic   utterances           -- how often an utterance is unusable
  hallucination  utterances           -- how often output is invented
  edge_error     edge word slots      -- whether errors sit at segment cuts

Rates over words and rates over utterances answer different questions, and a
system can look good on one while being unshippable on the other.
"""
import numpy as np
from jiwer import process_words

# An utterance's first and last EDGE_WORDS reference words are its "edges".
# Utterances shorter than MIN_REF_WORDS are skipped by the edge metric, since
# below that the edges and the middle overlap and the split means nothing.
EDGE_WORDS = 2
MIN_REF_WORDS = 6

# Defaults; config/eval/openasr.yaml overrides them.
CATASTROPHIC_THRESHOLD = 0.5   # per-utterance WER above this is unusable
HALLUCINATION_RATIO = 1.2      # hypothesis this many times longer than reference


def counts(references, hypotheses):
    """Per-utterance (errors, reference_length), so corpus WER can be re-derived.

    Corpus WER is the ratio of summed errors to summed reference words, not the
    mean of per-utterance WERs -- the leaderboard uses the former and the two
    differ a lot on sets with short utterances.
    """
    out = []
    for ref, hyp in zip(references, hypotheses):
        ref = str(ref).strip()
        hyp = str(hyp).strip()
        if not ref:
            # No reference words: insertions are unscoreable as a rate.
            out.append((len(hyp.split()), 0))
            continue
        result = process_words([ref], [hyp if hyp else " "])
        errors = result.substitutions + result.deletions + result.insertions
        out.append((errors, len(ref.split())))
    return np.array(out, dtype=np.int64)


def wer_from_counts(pairs):
    errors = pairs[:, 0].sum()
    length = pairs[:, 1].sum()
    return float(errors) / float(length) if length else float("nan")


def wer(references, hypotheses):
    return wer_from_counts(counts(references, hypotheses))


def error_breakdown(references, hypotheses):
    """Substitutions / deletions / insertions as rates, for diagnosing a WER.

    Two systems at the same WER can be very different products: one that
    deletes is quietly lossy, one that inserts is noisily wrong.
    """
    subs = dels = ins = length = 0
    for ref, hyp in zip(references, hypotheses):
        ref, hyp = str(ref).strip(), str(hyp).strip()
        if not ref:
            continue
        r = process_words([ref], [hyp if hyp else " "])
        subs += r.substitutions
        dels += r.deletions
        ins += r.insertions
        length += len(ref.split())
    if not length:
        return {"sub": float("nan"), "del": float("nan"), "ins": float("nan")}
    return {"sub": subs / length, "del": dels / length, "ins": ins / length}


def catastrophic_counts(references, hypotheses, threshold=CATASTROPHIC_THRESHOLD):
    """Per-utterance (is this utterance unusable, 1).

    A corpus WER of 6% is compatible with "every utterance is slightly wrong"
    and with "19 in 20 are perfect and the twentieth is garbage". Those are
    different products. This counts the second case: utterances whose own WER
    exceeds `threshold`.

    The denominator is utterances rather than words, so the result reads as a
    rate of bad calls -- which is the unit a review queue is staffed in.
    """
    out = []
    for ref, hyp in zip(references, hypotheses):
        ref, hyp = str(ref).strip(), str(hyp).strip()
        if not ref:
            out.append((0, 0))
            continue
        pair = counts([ref], [hyp])[0]
        utterance_wer = pair[0] / pair[1] if pair[1] else 0.0
        out.append((int(utterance_wer > threshold), 1))
    return np.array(out, dtype=np.int64)


def hallucination_counts(references, hypotheses, ratio=HALLUCINATION_RATIO):
    """Per-utterance (did the system invent material, 1).

    Flags utterances where the hypothesis runs at least `ratio` times the
    reference length. Length is a blunt instrument, but it catches the failure
    that matters -- a decoder continuing past the end of the speech -- without
    needing to decide which specific words were invented.

    Deliberately separate from the insertion rate in `error_breakdown`.
    Insertions averaged over a corpus hide this: a handful of utterances with
    runaway output contribute the same insertion mass as a slight tendency to
    add a word everywhere, and only one of those is a product risk.

    Note the short-utterance sensitivity: against a 2-word reference, a 3-word
    hypothesis is 50% over and gets flagged. Pass a `min_ref_words` filter at
    the call site to check how much of a result depends on that.
    """
    out = []
    for ref, hyp in zip(references, hypotheses):
        ref_words = str(ref).split()
        hyp_words = str(hyp).split()
        if not ref_words:
            out.append((0, 0))
            continue
        out.append((int(len(hyp_words) >= ratio * len(ref_words)), 1))
    return np.array(out, dtype=np.int64)


def edge_index(n_words, edge_words=EDGE_WORDS):
    """Reference positions counted as edges for an utterance of n_words."""
    return set(range(edge_words)) | set(range(n_words - edge_words, n_words))


def _edge_split(reference, hypothesis, edge_words=EDGE_WORDS):
    """One utterance's errors, split into edge and middle by reference position.

    Returns (edge, middle, edge_slots, middle_slots) with the first two keyed
    sub/del/ins, or None when the utterance is too short to split.

    The fiddly case is an insertion, which occupies no reference position. It
    is anchored to the reference index it was inserted at, clamped to the last
    real position -- so material appended past the end of the reference counts
    as a trailing-edge error rather than a middle one. That clamp matters: a
    decoder running on past the end of the speech is the single most important
    thing this metric is for, and it lands at exactly ref_start_idx == n.
    """
    ref_words = str(reference).split()
    n = len(ref_words)
    if n < MIN_REF_WORDS:
        return None

    index = edge_index(n, edge_words)
    edge = {"sub": 0, "del": 0, "ins": 0}
    middle = {"sub": 0, "del": 0, "ins": 0}
    kinds = {"substitute": "sub", "delete": "del", "insert": "ins"}

    hypothesis = str(hypothesis).strip()
    result = process_words([" ".join(ref_words)], [hypothesis if hypothesis else " "])
    for chunk in result.alignments[0]:
        if chunk.type == "equal":
            continue
        kind = kinds[chunk.type]
        if kind == "ins":
            weight = chunk.hyp_end_idx - chunk.hyp_start_idx
            span = [min(chunk.ref_start_idx, n - 1)]
        else:
            weight = chunk.ref_end_idx - chunk.ref_start_idx
            span = range(chunk.ref_start_idx, chunk.ref_end_idx)
        target = edge if any(i in index for i in span) else middle
        target[kind] += weight

    return edge, middle, 2 * edge_words, n - 2 * edge_words


def edge_breakdown(references, hypotheses, edge_words=EDGE_WORDS):
    """Substitutions, deletions and insertions at the edges vs the middle.

    Returns (edge, middle, edge_slots, middle_slots), the first two being dicts
    keyed sub/del/ins. Kept separate from `edge_error_counts` because
    diagnosing an edge effect needs the error *type*: deletions at a cut mean
    the endpointer truncated, insertions mean it kept going.
    """
    edge = {"sub": 0, "del": 0, "ins": 0}
    middle = {"sub": 0, "del": 0, "ins": 0}
    edge_slots = middle_slots = 0

    for reference, hypothesis in zip(references, hypotheses):
        split = _edge_split(reference, hypothesis, edge_words)
        if split is None:
            continue
        one_edge, one_middle, one_edge_slots, one_middle_slots = split
        for kind in edge:
            edge[kind] += one_edge[kind]
            middle[kind] += one_middle[kind]
        edge_slots += one_edge_slots
        middle_slots += one_middle_slots

    return edge, middle, edge_slots, middle_slots


def edge_error_counts(references, hypotheses, edge_words=EDGE_WORDS):
    """Per-utterance (errors at the edges, edge word slots).

    These corpora are pre-segmented: every utterance is a slice cut out of a
    longer recording, so its first and last words can begin or end mid-word at
    the cut. A system that does its own endpointing meets a hard onset at
    sample zero and can truncate against it, which is a property of the
    benchmark's segmentation rather than of the recognizer.

    Comparing this against the middle-of-utterance rate separates the two.
    src/eval/padding_probe.py uses exactly that to show a large part of
    Nova-3's edge deletions are recoverable by padding with silence, which adds
    no information and therefore cannot be a recognition improvement.

    Counts substitutions, deletions and insertions, on the same footing as
    WER, so -- like WER -- the rate can in principle exceed 100%.
    """
    out = []
    for reference, hypothesis in zip(references, hypotheses):
        split = _edge_split(reference, hypothesis, edge_words)
        if split is None:
            out.append((0, 0))
            continue
        edge, _, edge_slots, _ = split
        out.append((sum(edge.values()), edge_slots))
    return np.array(out, dtype=np.int64)


# The suite. A new metric is one function above and one entry here; everything
# downstream -- CIs, paired tests, per-dataset tables, metadata breakdowns --
# picks it up without further change.
METRICS = {
    "wer": counts,
    "catastrophic": catastrophic_counts,
    "hallucination": hallucination_counts,
    "edge_error": edge_error_counts,
}


def bootstrap_ci(pairs, samples, confidence, seed):
    """Percentile bootstrap over utterances.

    Resampling utterances (not words) is the right unit: utterances are what
    was sampled, and words inside one are correlated.
    """
    rng = np.random.default_rng(seed)
    n = len(pairs)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, n, size=(samples, n))
    errors = pairs[:, 0][idx].sum(axis=1)
    lengths = pairs[:, 1][idx].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        draws = np.where(lengths > 0, errors / lengths, np.nan)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.nanpercentile(draws, [100 * alpha, 100 * (1 - alpha)])
    return wer_from_counts(pairs), float(lower), float(upper)


def paired_bootstrap(pairs_a, pairs_b, samples, confidence, seed):
    """CI on the WER difference between two systems on the same utterances.

    Paired, because the systems saw identical audio: the shared difficulty
    cancels and the interval on the gap is much tighter than comparing two
    independent CIs, which is the comparison a leaderboard invites you to make
    by eye and which overstates how uncertain the ranking is.
    """
    rng = np.random.default_rng(seed)
    n = len(pairs_a)
    idx = rng.integers(0, n, size=(samples, n))
    ea, la = pairs_a[:, 0][idx].sum(axis=1), pairs_a[:, 1][idx].sum(axis=1)
    eb, lb = pairs_b[:, 0][idx].sum(axis=1), pairs_b[:, 1][idx].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        draws = np.where(la > 0, ea / la, np.nan) - np.where(lb > 0, eb / lb, np.nan)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.nanpercentile(draws, [100 * alpha, 100 * (1 - alpha)])
    point = wer_from_counts(pairs_a) - wer_from_counts(pairs_b)
    # p-value for "no difference": how often the resampled gap crosses zero.
    p = float(2 * min((draws < 0).mean(), (draws > 0).mean()))
    return point, float(lower), float(upper), p
