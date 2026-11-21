"""Common interface every ASR system implements, so adding one is a single file."""
from abc import ABC, abstractmethod

from tqdm import tqdm

SAMPLE_RATE = 16_000


class ASRSystem(ABC):
    """A batch of items is a list of {"id", "audio" (float32 @16k), "reference"}."""

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
            hypotheses.extend(self.transcribe(items[start : start + batch_size]))
        return hypotheses
