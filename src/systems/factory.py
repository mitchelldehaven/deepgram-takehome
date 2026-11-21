from src.systems.deepgram import DeepgramSystem
from src.systems.parakeet import ParakeetSystem
from src.systems.qwen3_asr import Qwen3ASRSystem
from src.systems.whisper import WhisperSystem

BACKENDS = {
    "whisper": WhisperSystem,
    "qwen3_asr": Qwen3ASRSystem,
    "parakeet": ParakeetSystem,
    "deepgram": DeepgramSystem,
}


def build_system(cfg):
    if cfg.backend not in BACKENDS:
        raise ValueError(f"Unknown backend: {cfg.backend} (have {sorted(BACKENDS)})")
    return BACKENDS[cfg.backend](cfg)
