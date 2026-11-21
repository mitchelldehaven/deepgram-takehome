from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
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
