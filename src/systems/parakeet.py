import json
import subprocess
import tempfile
from pathlib import Path

import soundfile as sf

from src.paths import ROOT_DIR
from src.systems.base import SAMPLE_RATE, ASRSystem

RUNNER = ROOT_DIR / "scripts" / "run_parakeet.py"


class ParakeetSystem(ASRSystem):
    """Shells out to .venv-nemo; see scripts/run_parakeet.py for why."""

    def load(self):
        self.python = ROOT_DIR / self.cfg.venv / "bin" / "python"
        if not self.python.exists():
            raise FileNotFoundError(
                f"{self.python} missing -- create it with the NeMo step in the README"
            )

    def transcribe(self, batch):
        return self.transcribe_all(batch)

    def transcribe_all(self, items):
        """One subprocess for the whole set: NeMo takes ~30s to load weights,
        so per-batch spawning would cost more than the inference does."""
        batch = items
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            manifest = tmp / "manifest.jsonl"
            out_path = tmp / "out.jsonl"
            with open(manifest, "w") as f:
                for item in batch:
                    wav = tmp / f"{item['id']}.wav"
                    sf.write(wav, item["audio"], SAMPLE_RATE)
                    f.write(
                        json.dumps({"id": item["id"], "audio_filepath": str(wav)}) + "\n"
                    )
            subprocess.run(
                [
                    str(self.python),
                    str(RUNNER),
                    self.cfg.model,
                    str(manifest),
                    str(out_path),
                    str(self.cfg.batch_size),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            by_id = {
                json.loads(line)["id"]: json.loads(line)["hypothesis"]
                for line in open(out_path)
            }
        return [by_id[item["id"]] for item in batch]
