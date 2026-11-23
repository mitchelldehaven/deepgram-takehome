import os
from pathlib import Path

# RUN_TAG separates one run's derived artefacts from another's, so a
# full-split run does not overwrite the subset run that the report is written
# from. It tags only what a run *produces* -- the sampled audio, the
# transcripts and the score tables. Downloaded inputs (the HF cache, the
# published leaderboard table) and model weights stay shared, because they are
# identical between runs and large.
#
#   RUN_TAG=full python src/data/build_subset.py   ->  data/processed-full/,
#                                                      outputs-full/
RUN_TAG = os.environ.get("RUN_TAG", "").strip()
SUFFIX = f"-{RUN_TAG}" if RUN_TAG else ""

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / f"processed{SUFFIX}"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / f"outputs{SUFFIX}"
LOGS_DIR = ROOT_DIR / "logs"
CONFIG_DIR = ROOT_DIR / "config"

SUBSET_FILE = PROCESSED_DIR / "subset.jsonl"
AUDIO_DIR = PROCESSED_DIR / "audio"
METADATA_FILE = PROCESSED_DIR / "metadata.jsonl"
TRANSCRIPTS_DIR = OUTPUTS_DIR / "transcripts"
SCORES_DIR = OUTPUTS_DIR / "scores"
PUBLISHED_FILE = RAW_DIR / "published_leaderboard.csv"


if __name__ == "__main__":
    dirs = [
        DATA_DIR,
        RAW_DIR,
        PROCESSED_DIR,
        MODELS_DIR,
        OUTPUTS_DIR,
        LOGS_DIR,
        AUDIO_DIR,
        TRANSCRIPTS_DIR,
        SCORES_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"Created: {d}")
