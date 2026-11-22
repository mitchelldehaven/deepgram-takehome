"""Tests for the four-metric suite.

The point of these is the shared contract: every metric returns per-utterance
(numerator, denominator), which is what lets bootstrap_ci and paired_bootstrap
work on any of them. If that breaks, the CIs silently become wrong rather than
absent, so it is worth asserting directly.
"""
import numpy as np
import pytest

from src.eval.metrics import (
    METRICS,
    bootstrap_ci,
    catastrophic_counts,
    counts,
    edge_breakdown,
    edge_error_counts,
    hallucination_counts,
    wer_from_counts,
)


def test_wer_is_corpus_level_not_mean_of_utterances():
    """The leaderboard sums errors over summed reference words. The mean of
    per-utterance WERs is a different number, and much higher on short
    utterances -- which is most of a short-form benchmark."""
    refs = ["a b c d e f g h i j", "one"]
    hyps = ["a b c d e f g h i j", "wrong"]
    pairs = counts(refs, hyps)
    corpus = wer_from_counts(pairs)
    per_utterance = np.mean([p[0] / p[1] for p in pairs])
    assert corpus == pytest.approx(1 / 11)
    assert per_utterance == pytest.approx(0.5)


def test_catastrophic_counts_thresholds_per_utterance_wer():
    refs = ["a b c d", "a b c d", "a b c d"]
    hyps = ["a b c d", "a b x y", "z z z z"]   # 0%, 50%, 100%
    pairs = catastrophic_counts(refs, hyps, threshold=0.5)
    # Strictly greater than the threshold, so the 50% utterance is not counted.
    assert [int(p[0]) for p in pairs] == [0, 0, 1]
    assert [int(p[1]) for p in pairs] == [1, 1, 1]


def test_catastrophic_threshold_is_honoured():
    refs = ["a b c d"]
    hyps = ["a b x y"]   # exactly 50%
    assert catastrophic_counts(refs, hyps, threshold=0.4)[0][0] == 1
    assert catastrophic_counts(refs, hyps, threshold=0.5)[0][0] == 0


def test_hallucination_counts_flags_runaway_length():
    refs = ["a b c d e"] * 3
    hyps = ["a b c d e", "a b c d e f", "a b c d e f g h i j"]   # 1.0x, 1.2x, 2.0x
    pairs = hallucination_counts(refs, hyps, ratio=1.2)
    assert [int(p[0]) for p in pairs] == [0, 1, 1]


def test_hallucination_is_not_the_insertion_rate():
    """One runaway utterance and a corpus-wide drift can carry identical
    insertion mass; only the first is a product risk, and only this metric
    distinguishes them."""
    refs = ["a b c d e"] * 4
    concentrated = ["a b c d e", "a b c d e", "a b c d e", "a b c d e x x x x"]
    spread = ["a b c d e x", "a b c d e x", "a b c d e x", "a b c d e x"]
    assert sum(len(h.split()) for h in concentrated) == sum(
        len(h.split()) for h in spread
    )
    assert hallucination_counts(refs, concentrated)[:, 0].sum() == 1
    assert hallucination_counts(refs, spread)[:, 0].sum() == 4


def test_edge_error_counts_only_scores_the_edges():
    ref = "one two three four five six seven eight"
    # Wrong in the middle only: edges are intact.
    middle_wrong = "one two WRONG WRONG WRONG WRONG seven eight"
    pairs = edge_error_counts([ref], [middle_wrong])
    assert int(pairs[0][0]) == 0
    assert int(pairs[0][1]) == 4   # 2 words at each end

    # Wrong at both edges only.
    edge_wrong = "X X three four five six Y Y"
    assert int(edge_error_counts([ref], [edge_wrong])[0][0]) == 4


def test_edge_metric_skips_utterances_too_short_to_split():
    pairs = edge_error_counts(["one two three"], ["totally different"])
    assert int(pairs[0][1]) == 0   # zero denominator: contributes nothing


def test_edge_breakdown_separates_deletion_from_insertion():
    ref = "one two three four five six seven eight"
    truncated = "three four five six seven eight"          # lost the leading edge
    overrun = "one two three four five six seven eight nine ten"
    edge, _, slots, _ = edge_breakdown([ref], [truncated])
    assert edge["del"] == 2 and edge["ins"] == 0 and slots == 4
    edge, _, _, _ = edge_breakdown([ref], [overrun])
    assert edge["ins"] == 2 and edge["del"] == 0


@pytest.mark.parametrize("metric", sorted(METRICS))
def test_every_metric_returns_bootstrappable_pairs(metric):
    """The shared contract. A metric that violates it would still produce a
    number, but the confidence interval would be meaningless."""
    refs = ["one two three four five six", "seven eight nine ten eleven twelve"]
    hyps = ["one two three four five six", "seven eight WRONG ten eleven"]
    pairs = METRICS[metric](refs, hyps)
    assert pairs.shape == (2, 2)
    assert pairs.dtype == np.int64
    assert (pairs[:, 0] <= np.maximum(pairs[:, 1], 1)).all() or metric == "wer"
    point, low, high = bootstrap_ci(pairs, 200, 0.95, 42)
    assert low <= point <= high


def test_empty_reference_contributes_nothing():
    for metric, function in METRICS.items():
        pairs = function([""], ["some invented words"])
        assert int(pairs[0][1]) == 0, metric
