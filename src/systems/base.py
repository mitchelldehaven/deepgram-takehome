"""Common interface every ASR system implements, so adding one is a single file."""
from abc import ABC, abstractmethod

import numpy as np
import soundfile as sf
from tqdm import tqdm

SAMPLE_RATE = 16_000


def read_audio(item):
    """Attach decoded audio to an item, in place, and return the item.

    Audio is loaded per batch rather than for the whole set up front. At
    subset size that is an irrelevant difference; on a full test split the
    eager version is ~33 GB of float32 resident before the first token is
    decoded, which is the kind of limit that only shows up once the thing is
    actually run at scale.
    """
    if "audio" not in item:
        audio, _ = sf.read(item["audio_path"], dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        item["audio"] = np.asarray(audio, dtype=np.float32)
    return item


class ASRSystem(ABC):
    """A batch of items is a list of {"id", "audio_path", "reference", "dataset"}.

    `transcribe` receives items with "audio" already attached. A system that
    overrides `transcribe_all` is responsible for its own loading, and may
    prefer not to load at all: ParakeetSystem hands the paths straight to
    NeMo.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.name = cfg.name

    def load(self):
        """Bring up weights or clients. Separate from __init__ so configuration
        errors surface before anything expensive happens."""

    @abstractmethod
    def transcribe(self, batch):
        """Return one raw hypothesis string per item, in order.

        Raw means exactly what the system emits: casing, punctuation and number
        formatting are left alone. Every such decision belongs in
        src/normalization, where it can be varied, which is the whole point of
        this exercise.
        """

    def transcribe_all(self, items):
        """Whole-set entry point. Overridden by systems that batch differently:
        a subprocess that should load weights once, or an API we fan out over.
        """
        hypotheses = []
        batch_size = self.cfg.get("batch_size", 8)
        for start in tqdm(range(0, len(items), batch_size), desc=self.name):
            batch = [read_audio(item) for item in items[start : start + batch_size]]
            hypotheses.extend(self.transcribe(batch))
            for item in batch:
                item.pop("audio", None)
        return hypotheses
