"""Tests for the arm-diff tool.

The property worth protecting is the decomposition: per-utterance
contributions must sum to the headline gap, for every metric, including when
the two arms have different denominators. If that breaks, the tool is sorting
by a number that means nothing, and it would still look plausible.
"""
import numpy as np
import pytest
from omegaconf import OmegaConf

from src.eval.compare import compare_table, mark, parse_arm, rate
from src.eval.metrics import METRICS

CFG = OmegaConf.create(
    {
        "scoring": {
            "bootstrap_samples": 200,
            "confidence": 0.95,
            "seed": 0,
            "catastrophic_threshold": 0.5,
            "hallucination_ratio": 1.2,
            "edge_words": 2,
        }
    }
)

REFERENCES = [
    "the quick brown fox jumps over the lazy dog today",
    "she sells sea shells by the sea shore every morning",
    "a man a plan a canal panama and then some more",
    "we will very much limit what you can do with it",
]
HYPS_A = [
    "the quick brown fox jumps over the lazy dog today",
    "she sells sea shells by the sea shore",
    "a man a plan a canal panama and then some more extra words here now",
    "we will very much limit what you can do with it",
]
HYPS_B = [
    "the quick brown fox jumped over the lazy dog",
    "she sells sea shells by the sea shore every morning",
    "a man a plan a canal panama and then some more",
    "totally different words appear here instead of the reference text",
]


def arm(references, hypotheses):
    return {
        str(i): {"dataset": "test", "reference": r, "hypothesis": h}
        for i, (r, h) in enumerate(zip(references, hypotheses))
    }


@pytest.mark.parametrize("metric", sorted(METRICS))
def test_contributions_sum_to_the_gap(metric):
    rows_a, rows_b = arm(REFERENCES, HYPS_A), arm(REFERENCES, HYPS_B)
    table, pairs_a, pairs_b = compare_table(CFG, rows_a, rows_b, metric)
    gap = 100 * (
        rate(pairs_a[:, 0].sum(), pairs_a[:, 1].sum())
        - rate(pairs_b[:, 0].sum(), pairs_b[:, 1].sum())
    )
    assert table.contribution.sum() == pytest.approx(gap, abs=1e-9)


def test_contributions_sum_to_the_gap_with_different_denominators():
    """The normalizer-diff case: the arms do not share a reference.

    Differencing per-utterance rates would be wrong here. Dividing each arm's
    numerator by its own corpus denominator is not.
    """
    short = [" ".join(r.split()[:6]) for r in REFERENCES]
    rows_a, rows_b = arm(REFERENCES, HYPS_A), arm(short, HYPS_B)
    table, pairs_a, pairs_b = compare_table(CFG, rows_a, rows_b, "wer")
    assert pairs_a[:, 1].sum() != pairs_b[:, 1].sum()
    gap = 100 * (
        rate(pairs_a[:, 0].sum(), pairs_a[:, 1].sum())
        - rate(pairs_b[:, 0].sum(), pairs_b[:, 1].sum())
    )
    assert table.contribution.sum() == pytest.approx(gap, abs=1e-9)


def test_table_is_sorted_by_absolute_contribution():
    table, _, _ = compare_table(CFG, arm(REFERENCES, HYPS_A), arm(REFERENCES, HYPS_B), "wer")
    magnitudes = table.contribution.abs().to_numpy()
    assert (np.diff(magnitudes) <= 1e-12).all()


def test_only_shared_utterances_are_compared():
    rows_a = arm(REFERENCES, HYPS_A)
    rows_b = arm(REFERENCES, HYPS_B)
    rows_b.pop("0")
    table, pairs_a, _ = compare_table(CFG, rows_a, rows_b, "wer")
    assert len(table) == len(pairs_a) == len(REFERENCES) - 1
    assert "0" not in set(table.id)


def test_no_shared_utterances_is_an_error():
    with pytest.raises(RuntimeError):
        compare_table(CFG, arm(REFERENCES[:2], HYPS_A[:2]), {}, "wer")


def test_mark_labels_each_error_type():
    assert mark("one two three", "one two three") == "one two three"
    assert "{two>too}" in mark("one two three", "one too three")
    assert "[-three]" in mark("one two three", "one two")
    assert "[+four]" in mark("one two three", "one two three four")


def test_mark_handles_an_empty_side():
    assert mark("", "hello there") == "[+hello] [+there]"
    assert mark("hello there", "") == "[-hello] [-there]"


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("system:custom", ("system", "custom")),
        ("system", ("system", "leaderboard")),
        ("system:leaderboard", ("system", "leaderboard")),
    ],
)
def test_parse_arm(spec, expected):
    assert parse_arm(spec, "leaderboard") == expected
