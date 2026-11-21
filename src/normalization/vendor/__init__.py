"""Vendored verbatim from huggingface/open_asr_leaderboard, normalizer/.

Pinned so the leaderboard baseline cannot drift underneath us.
Upstream commit: c4696193a333f7948cb113dc93bb54f661bd9a0b
Fetched: 2026-11-21

Only the English path is kept; the multilingual and Armenian normalizers and
the model-id helpers are dropped. The two files are otherwise unmodified.
"""
from .normalizer import EnglishTextNormalizer

__all__ = ["EnglishTextNormalizer"]
