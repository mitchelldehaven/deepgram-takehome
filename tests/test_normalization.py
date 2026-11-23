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


def test_digits_and_words_tokenize_identically():
    """The point of canonicalizing numbers is that the same quantity scores the
    same whichever way the source wrote it. Spelling out used to run *after*
    hyphen collapsing, so "twenty-five" gave one token and "25" gave two.
    """
    assert norm("twenty-five orders") == norm("25 orders") == "twenty five orders"


def test_hyphens_are_word_boundaries():
    """A hyphen separates words that are spoken separately, so it becomes a
    space. Collapsing it instead produced tokens no recognizer emits -- see
    REPORT.md; the diff tool found this, reading the rule did not."""
    assert norm("re-run") == "re run"
    assert norm("a forty-two-year-old") == "a forty two year old"
    assert norm("self-control") == "self control"
    # And the same for hyphens num2words introduces, not just source ones.
    assert norm("42-year-old") == "forty two year old"


def test_toggles_are_honoured():
    """Each rule is ablated independently, so each must actually do something
    observable on its own."""
    assert norm("um i don't know", remove_fillers=True) == "i don't know"
    assert norm("um i don't know", remove_fillers=False) == "um i don't know"
    assert norm("i don't know", expand_contractions=True) == "i do not know"
    assert norm("i don't know", keep_apostrophes=False) == "i dont know"
    assert norm("the colour", british_to_american=True) == "the color"
    assert norm("[laughter] hello", strip_non_speech=True) == "hello"
    # Hyphen splitting happens before the per-token rules, so a hyphenated
    # British spelling is still mapped.
    assert norm("colour-coded", british_to_american=True) == "color coded"


def test_normalizer_is_idempotent_on_its_own_output():
    """Normalizing twice must not drift, or scoring would depend on how many
    times the text happened to be passed through."""
    for text in ["$5 million in 1,200 re-run orders", "um, we'll re-open it"]:
        once = norm(text)
        assert norm(once) == once
