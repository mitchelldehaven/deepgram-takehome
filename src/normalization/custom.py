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
        # num2words emits "one thousand, two hundred"; the hyphens it also
        # emits are left for the hyphenation rule below to treat exactly as it
        # treats hyphens that were in the source text.
        text = text.replace(",", " ")

    if rules.collapse_hyphenation:
        text = re.sub(r"(\w)-(\w)", r"\1\2", text)

    text = PUNCT.sub(" ", text)

    if not rules.keep_apostrophes:
        text = text.replace("'", "")

    tokens = text.split()

    if rules.remove_fillers:
        tokens = [t for t in tokens if t.strip("'") not in FILLERS]

    return WHITESPACE.sub(" ", " ".join(tokens)).strip()
