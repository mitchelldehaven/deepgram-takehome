import io
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf
from deepgram import DeepgramClient
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

from src.systems.base import SAMPLE_RATE, ASRSystem


class DeepgramSystem(ASRSystem):
    """Nova via the prerecorded REST API.

    The API formats by default -- punctuation, casing, numerals -- while the
    leaderboard's open models emit bare lowercase text. That asymmetry is not
    cosmetic: the leaderboard normalizer treats the two differently, so the
    flags below are config, not constants, and config/eval/openasr.yaml turns
    them off to match how the open models are scored.
    """

    def load(self):
        load_dotenv()
        key = os.environ.get("DEEPGRAM_API_KEY")
        if not key:
            raise RuntimeError("DEEPGRAM_API_KEY not set (see .env, and the README)")
        self.client = DeepgramClient(api_key=key)

    def pad(self, audio):
        """Optionally bracket the utterance with digital silence.

        These utterances are slices cut out of continuous recordings, so the
        first and last words can start or stop mid-word at the cut. A
        prerecorded API does its own endpointing and expects whole-call audio,
        so it meets a hard onset at sample zero. Padding gives the endpointer
        runway *without adding any information*: if edge deletions fall, they
        were an artifact of the cut rather than of the speech.

        It is not a free lunch -- silence is also where a model is most free to
        hallucinate -- so src/eval/padding_probe.py measures both directions.
        """
        seconds = float(self.cfg.get("pad_silence_s", 0.0))
        if seconds <= 0:
            return audio
        silence = np.zeros(int(seconds * SAMPLE_RATE), dtype=audio.dtype)
        return np.concatenate([silence, audio, silence])

    def transcribe_one(self, item):
        buffer = io.BytesIO()
        sf.write(buffer, self.pad(item["audio"]), SAMPLE_RATE, format="WAV", subtype="PCM_16")
        try:
            response = self.client.listen.v1.media.transcribe_file(
                request=buffer.getvalue(),
                model=self.cfg.model,
                language=self.cfg.get("language", "en"),
                punctuate=self.cfg.get("punctuate", False),
                smart_format=self.cfg.get("smart_format", False),
                numerals=self.cfg.get("numerals", False),
                filler_words=self.cfg.get("filler_words", True),
            )
            channel = response.results.channels[0]
            return channel.alternatives[0].transcript.strip()
        except Exception as exc:
            # One failed request should not void a run of several thousand;
            # score.py counts empties so they cannot be silently ignored.
            logger.warning(f"{item['id']}: {type(exc).__name__}: {str(exc)[:160]}")
            return ""

    def transcribe(self, batch):
        return [self.transcribe_one(item) for item in batch]

    def transcribe_all(self, items):
        workers = self.cfg.get("concurrency", 8)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(
                tqdm(pool.map(self.transcribe_one, items), total=len(items), desc=self.name)
            )
