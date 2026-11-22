"""Regression tests for the custom normalizer's number handling.

All three cases below were live bugs, and all three survived a careful read of
the code. They were found by printing both normalizers' output on handwritten
sentences and looking at it. These tests exist so that stays fixed.
"""
import pytest

from src.normalization import custom
from src.normalization.factory import load_rules


def norm(text, **overrides):
    return custom.normalize(text, rules=load_rules(overrides or None))


def test_comma_grouped_thousands():
    """"1,200" was matched as two digit runs and became "one two hundred"."""
    assert norm("1,200 orders") == "one thousand two hundred orders"
    assert norm("10,000 locations") == "ten thousand locations"


def test_currency_unit_follows_magnitude():
    """The unit went after the digits: "$5 million" -> "five dollars million"."""
    assert norm("$5 million") == "five million dollars"
    assert norm("$5") == "five dollars"
    assert norm("£3 billion") == "three billion pounds"


@pytest.mark.parametrize("collapse", [True, False])
def test_digits_and_words_tokenize_identically(collapse):
    """The point of canonicalizing numbers is that the same quantity scores the
    same whichever way the source wrote it. Spelling out used to run *after*
    hyphen collapsing, so "twenty-five" gave one token and "25" gave two.
    """
    words = norm("twenty-five orders", collapse_hyphenation=collapse)
    digits = norm("25 orders", collapse_hyphenation=collapse)
    assert words == digits


def test_toggles_are_honoured():
    """Each rule is ablated independently, so each must actually do something
    observable on its own."""
    assert norm("um i don't know", remove_fillers=True) == "i don't know"
    assert norm("um i don't know", remove_fillers=False) == "um i don't know"
    assert norm("i don't know", expand_contractions=True) == "i do not know"
    assert norm("i don't know", keep_apostrophes=False) == "i dont know"
    assert norm("the colour", british_to_american=True) == "the color"
    assert norm("re-run", collapse_hyphenation=True) == "rerun"
    assert norm("re-run", collapse_hyphenation=False) == "re run"
    assert norm("[laughter] hello", strip_non_speech=True) == "hello"


def test_normalizer_is_idempotent_on_its_own_output():
    """Normalizing twice must not drift, or scoring would depend on how many
    times the text happened to be passed through."""
    for text in ["$5 million in 1,200 re-run orders", "um, we'll re-open it"]:
        once = norm(text)
        assert norm(once) == once
