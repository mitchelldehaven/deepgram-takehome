"""Sample the evaluation subset from the leaderboard's own test parquets.

Sampling design
---------------
The upstream repo sorts every split by audio length, globally and descending.
That makes naive shortcuts actively wrong: reading the first shard, or the
first N rows, gives only the longest utterances, and duration correlates with
WER. It also makes a cheap sample possible. Because the frame is sorted on a
variable we care about, a *systematic* sample over it -- take row groups at
even intervals across the whole sorted order, then sample rows inside each --
spans the duration distribution by construction, and is at least as efficient
as simple random sampling on that variable.

It is also the only affordable option: the three splits are ~17 GB, and
fetching row groups at intervals touches a few hundred MB instead.

Size is set in config and checked after the fact, not asserted here: see the
bootstrap CIs in src/eval/score.py, which are what actually establish whether
400 utterances resolve the gaps between these systems.
"""
import io

import dirtygit
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfFileSystem
from loguru import logger
from omegaconf import OmegaConf
from tqdm import tqdm

from src.config import load_config, setup_logging
from src.data.utils import save_jsonl
from src.paths import AUDIO_DIR, CONFIG_DIR, SUBSET_FILE
from src.systems.base import SAMPLE_RATE

DEFAULT_CONFIG = CONFIG_DIR / "eval" / "openasr.yaml"

COLUMNS = ["audio", "text", "id", "audio_length_s"]


def list_row_groups(fs, repo, config, split):
    """Footer-only pass: the ordered list of (path, row_group, n_rows)."""
    paths = sorted(fs.glob(f"datasets/{repo}/{config}/{split}-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No {split} parquet for {config} in {repo}")
    groups = []
    for path in tqdm(paths, desc=f"{config}: footers"):
        with fs.open(path, "rb") as handle:
            metadata = pq.ParquetFile(handle).metadata
            for index in range(metadata.num_row_groups):
                groups.append((path, index, metadata.row_group(index).num_rows))
    return groups


def select_row_groups(groups, n_groups):
    """Evenly spaced positions across the length-sorted order."""
    if n_groups >= len(groups):
        return list(range(len(groups)))
    positions = np.linspace(0, len(groups) - 1, n_groups)
    return sorted({int(round(p)) for p in positions})


def sample_dataset(fs, repo, dataset, split, n_utterances, n_groups, rng):
    groups = list_row_groups(fs, repo, dataset.config, split)
    total_rows = sum(rows for _, _, rows in groups)
    chosen = select_row_groups(groups, n_groups)
    per_group = int(np.ceil(n_utterances / len(chosen)))
    logger.info(
        f"{dataset.name}: {total_rows} utterances in {len(groups)} row groups; "
        f"reading {len(chosen)} of them, up to {per_group} rows each"
    )

    records = []
    for position in tqdm(chosen, desc=f"{dataset.name}: rows"):
        path, group_index, n_rows = groups[position]
        with fs.open(path, "rb") as handle:
            table = pq.ParquetFile(handle).read_row_group(group_index, columns=COLUMNS)
        take = min(per_group, table.num_rows)
        rows = rng.choice(table.num_rows, size=take, replace=False)
        batch = table.take(sorted(rows)).to_pylist()
        for row in batch:
            records.append((dataset.name, position, row))

    # Trim to the requested size without re-biasing toward any part of the
    # sorted order: shuffle, cut, then restore the original order.
    rng.shuffle(records)
    records = records[:n_utterances]
    records.sort(key=lambda r: r[1])
    return records


def write_records(records, audio_dir):
    audio_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for dataset_name, _, row in tqdm(records, desc="decoding audio"):
        audio, sample_rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"{row['id']}: expected {SAMPLE_RATE} Hz, got {sample_rate}")

        utterance_id = f"{dataset_name}:{row['id']}".replace("/", "_")
        wav_path = audio_dir / f"{utterance_id}.wav"
        sf.write(wav_path, audio, SAMPLE_RATE, subtype="PCM_16")
        out.append(
            {
                "id": utterance_id,
                "dataset": dataset_name,
                "reference": row["text"],
                "audio_path": str(wav_path.relative_to(audio_dir.parent.parent)),
                "duration_s": float(row["audio_length_s"]),
            }
        )
    return out


def main(cfg):
    logger.info(f"githash={cfg.githash}")
    rng = np.random.default_rng(cfg.subset.seed)
    fs = HfFileSystem()

    records = []
    for dataset in cfg.subset.datasets:
        records.extend(
            sample_dataset(
                fs,
                cfg.subset.dataset_path,
                dataset,
                cfg.subset.split,
                cfg.subset.utterances_per_dataset,
                cfg.subset.row_groups_per_dataset,
                rng,
            )
        )

    rows = write_records(records, AUDIO_DIR)
    SUBSET_FILE.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(rows, SUBSET_FILE)

    hours = sum(r["duration_s"] for r in rows) / 3600
    logger.info(f"{len(rows)} utterances ({hours:.2f} h) -> {SUBSET_FILE}")
    for dataset in cfg.subset.datasets:
        subset = [r for r in rows if r["dataset"] == dataset.name]
        durations = np.array([r["duration_s"] for r in subset])
        logger.info(
            f"  {dataset.name}: n={len(subset)} "
            f"duration min/median/max = "
            f"{durations.min():.1f}/{np.median(durations):.1f}/{durations.max():.1f}s"
        )


if __name__ == "__main__":
    githash = dirtygit.check()
    cfg = load_config(DEFAULT_CONFIG)
    OmegaConf.update(cfg, "githash", githash)
    setup_logging()
    main(cfg)
