"""An alternative normalizer, written to disagree with the leaderboard's.

Each disagreement is a single toggle in config/normalization/custom.yaml so it
can be ablated on its own. The point is not that these choices are better --
it is that they are defensible, nobody publishes which ones were made, and the
ranking moves when they change.
"""
import re
import unicodedata

from num2words import num2words

from src.normalization.vendor.english_abbreviations import english_spelling_normalizer

NON_SPEECH = re.compile(r"\[[^\]]*\]|\([^)]*\)|<[^>]*>")
FILLERS = {"um", "uh", "mm", "hmm", "mhm", "uhhuh", "erm", "er", "ah", "eh"}
PUNCT = re.compile(r"[^\w\s']", flags=re.UNICODE)
APOSTROPHES = re.compile(r"[‘’ʼ´`]")
NUMBER = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
MAGNITUDE = r"(?:hundred|thousand|million|billion|trillion)"
WHITESPACE = re.compile(r"\s+")

CONTRACTIONS = {
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "won't": "will not",
    "can't": "can not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "wouldn't": "would not",
    "couldn't": "could not",
    "shouldn't": "should not",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "he's": "he is",
    "she's": "she is",
    "what's": "what is",
    "let's": "let us",
    "i'm": "i am",
    "i've": "i have",
    "we've": "we have",
    "they've": "they have",
    "you've": "you have",
    "i'll": "i will",
    "we'll": "we will",
    "they'll": "they will",
    "you'll": "you will",
    "i'd": "i would",
    "we're": "we are",
    "they're": "they are",
    "you're": "you are",
}

CURRENCY = {"$": "dollars", "£": "pounds", "€": "euros"}


def spell_out(match):
    value = match.group(0).replace(",", "")
    try:
        if "." in value:
            return num2words(float(value))
        return num2words(int(value))
    except (ValueError, OverflowError):
        return value


def normalize(text, rules):
    text = unicodedata.normalize("NFKC", str(text))

    if rules.strip_non_speech:
        text = NON_SPEECH.sub(" ", text)

    text = APOSTROPHES.sub("'", text)
    text = text.lower()

    # A hyphen is a word boundary, so it becomes a space rather than being
    # closed up. "forty-two-year-old" is spoken as four words and every
    # recognizer emits it as four; collapsing it to one token would charge
    # four errors for a difference in orthography that nobody uttered.
    # Done here, before the rules below, so that per-token rules such as
    # british_to_american see "colour coded" rather than one unknown token.
    #
    # This is also what the leaderboard's normalizer does (a hyphen is
    # unicode category Pd, which its remove_symbols_and_diacritics maps to a
    # space), so hyphenation is no longer one of our disagreements with it and
    # no longer has a toggle. See REPORT.md for why it used to.
    text = text.replace("-", " ")

    if rules.normalize_currency:
        for symbol, word in CURRENCY.items():
            # "$5 million" is read "five million dollars": the unit follows the
            # magnitude word when there is one, not the digits.
            text = re.sub(
                rf"\{symbol}\s?({NUMBER.pattern})(\s+{MAGNITUDE})?",
                lambda m, w=word: f"{m.group(1)}{m.group(2) or ''} {w}",
                text,
            )

    if rules.expand_contractions:
        for short, long in CONTRACTIONS.items():
            text = re.sub(rf"\b{re.escape(short)}\b", long, text)

    if rules.british_to_american:
        text = " ".join(english_spelling_normalizer.get(w, w) for w in text.split())

    if rules.spell_out_numbers:
        text = NUMBER.sub(spell_out, text)
        # num2words emits "one thousand, two hundred" and "twenty-five". Both
        # separators become spaces, exactly as a hyphen in the source text
        # does above -- otherwise "25" and "twenty-five" would tokenize
        # differently and canonicalizing numbers would accomplish nothing.
        text = text.replace(",", " ").replace("-", " ")

    text = PUNCT.sub(" ", text)

    if not rules.keep_apostrophes:
        text = text.replace("'", "")

    tokens = text.split()

    if rules.remove_fillers:
        tokens = [t for t in tokens if t.strip("'") not in FILLERS]

    return WHITESPACE.sub(" ", " ".join(tokens)).strip()
