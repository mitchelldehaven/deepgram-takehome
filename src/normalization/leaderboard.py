"""The leaderboard's own normalizer, wrapped so it looks like ours."""
from src.normalization.vendor import EnglishTextNormalizer

_normalizer = EnglishTextNormalizer()


def normalize(text):
    return _normalizer(str(text))
