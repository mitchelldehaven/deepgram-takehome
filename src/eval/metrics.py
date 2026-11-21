"""Scoring primitives.

WER is what the leaderboard reports. Everything else here exists because a
corpus WER answers "how wrong on average" and not "wrong about what", which is
the question an actual buyer is asking.
"""
import numpy as np
from jiwer import process_words


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


def keyword_error_rate(references, hypotheses, keywords):
    """Recall-oriented error rate over a set of content words (our added metric).

    WER weights "the" and a company name identically. A contact-centre or
    meeting-notes buyer does not: missing the entity is the failure that costs
    them money. This counts, over reference occurrences of each keyword, how
    often the hypothesis does not contain it at least as many times.

    Bag-of-words on purpose: word order is already covered by WER, and an
    alignment-based version would inherit WER's sensitivity to the surrounding
    text, which is exactly what we are trying to factor out.
    """
    missed = total = 0
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = str(ref).split()
        hyp_tokens = str(hyp).split()
        for kw in keywords:
            ref_n = ref_tokens.count(kw)
            if not ref_n:
                continue
            hyp_n = hyp_tokens.count(kw)
            total += ref_n
            missed += max(0, ref_n - hyp_n)
    return missed / total if total else float("nan")


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
