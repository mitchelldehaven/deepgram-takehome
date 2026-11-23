"""Sample the evaluation subset from the leaderboard's own test parquets.

Sampling design
---------------
The upstream repo sorts every split by audio length, globally and descending,
and splits it into shards in that order. That makes the obvious shortcuts
wrong -- the first shard, or the first N rows, is all long utterances, and
duration correlates with WER -- but it also makes a cheap sample possible.

So: pick shards at even intervals across the sorted order, then sample rows
uniformly at random inside each. Because the frame is ordered on a variable we
care about, this systematic design spans the duration distribution by
construction rather than by luck, and on that variable it is no worse than
simple random sampling over the whole split.

The alternative, fetching individual row groups over HTTP range requests,
turned out to be ~8x slower per byte than pulling whole shards, so we pull
whole shards and sample inside them.

Whether 400 utterances per dataset is *enough* is not asserted here. It is
answered after the fact by the bootstrap CIs in src/eval/score.py, which show
directly which system gaps this sample can and cannot resolve.
"""
import io

import dirtygit
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download, list_repo_files
from loguru import logger
from omegaconf import OmegaConf
from tqdm import tqdm

from src.config import load_config, setup_logging
from src.data.utils import save_jsonl
from src.paths import AUDIO_DIR, CONFIG_DIR, SUBSET_FILE
from src.systems.base import SAMPLE_RATE

DEFAULT_CONFIG = CONFIG_DIR / "eval" / "openasr.yaml"

COLUMNS = ["audio", "text", "id", "audio_length_s"]


def list_shards(repo, config, split):
    files = list_repo_files(repo, repo_type="dataset")
    shards = sorted(f for f in files if f.startswith(f"{config}/{split}-"))
    if not shards:
        raise FileNotFoundError(f"No {split} parquet for {config} in {repo}")
    return shards


def select_evenly(items, k):
    """k positions spread across the whole ordered list, endpoints included."""
    if k >= len(items):
        return list(range(len(items)))
    positions = np.linspace(0, len(items) - 1, k)
    return sorted({int(round(p)) for p in positions})


def sample_dataset(repo, dataset, split, n_utterances, n_shards, rng):
    """Yield (dataset_name, shard_position, row) one shard at a time.

    A generator rather than a list because `n_utterances: null` means "the
    whole split", and holding ~10 GB of encoded audio in memory to then write
    it out is not necessary. The caller materialises only when it has to,
    which is exactly when a sample is being trimmed.
    """
    shards = list_shards(repo, dataset.config, split)
    take_all = n_utterances is None
    chosen = list(range(len(shards))) if not n_shards else select_evenly(shards, n_shards)
    per_shard = None if take_all else int(np.ceil(n_utterances / len(chosen)))
    logger.info(
        f"{dataset.name}: {len(shards)} shards, reading {len(chosen)} of them "
        f"({[c for c in chosen]}), "
        + ("every row" if take_all else f"up to {per_shard} rows each")
    )

    for position in tqdm(chosen, desc=f"{dataset.name}: shards"):
        path = hf_hub_download(repo, shards[position], repo_type="dataset")
        if take_all:
            # Stream in batches: peak memory is one batch, not one shard.
            for batch in pq.ParquetFile(path).iter_batches(
                batch_size=32, columns=COLUMNS
            ):
                for row in batch.to_pylist():
                    yield dataset.name, position, row
            continue
        table = pq.ParquetFile(path).read(columns=COLUMNS)
        take = min(per_shard, table.num_rows)
        rows = rng.choice(table.num_rows, size=take, replace=False)
        for row in table.take(sorted(rows)).to_pylist():
            yield dataset.name, position, row


def trim(records, n_utterances, rng):
    """Cut an oversampled set to size without re-biasing toward any part of
    the length-sorted order: shuffle, cut, then restore the original order."""
    records = list(records)
    rng.shuffle(records)
    records = records[:n_utterances]
    records.sort(key=lambda record: record[1])
    return records


def write_records(records, audio_dir, total=None):
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset_name, _, row in tqdm(records, desc="decoding audio", total=total):
        audio, sample_rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"{row['id']}: expected {SAMPLE_RATE} Hz, got {sample_rate}")

        utterance_id = f"{dataset_name}:{row['id']}".replace("/", "_")
        wav_path = audio_dir / f"{utterance_id}.wav"
        sf.write(wav_path, audio, SAMPLE_RATE, subtype="PCM_16")
        rows.append(
            {
                "id": utterance_id,
                "dataset": dataset_name,
                "reference": row["text"],
                "audio_path": str(wav_path.relative_to(audio_dir.parent.parent)),
                "duration_s": float(row["audio_length_s"]),
            }
        )
    return rows


def main(cfg):
    logger.info(f"githash={cfg.githash}")
    rng = np.random.default_rng(cfg.subset.seed)

    wanted = cfg.subset.utterances_per_dataset
    rows = []
    for dataset in cfg.subset.datasets:
        records = sample_dataset(
            cfg.subset.dataset_path,
            dataset,
            cfg.subset.split,
            wanted,
            cfg.subset.shards_per_dataset,
            rng,
        )
        # Oversampled runs are trimmed, which needs them all in hand. A
        # whole-split run is streamed straight to disk instead.
        if wanted is not None:
            records = trim(records, wanted, rng)
        rows.extend(write_records(records, AUDIO_DIR, total=wanted))

    SUBSET_FILE.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(rows, SUBSET_FILE)

    hours = sum(row["duration_s"] for row in rows) / 3600
    logger.info(f"{len(rows)} utterances ({hours:.2f} h) -> {SUBSET_FILE}")
    for dataset in cfg.subset.datasets:
        subset = [row for row in rows if row["dataset"] == dataset.name]
        durations = np.array([row["duration_s"] for row in subset])
        words = np.array([len(row["reference"].split()) for row in subset])
        logger.info(
            f"  {dataset.name}: n={len(subset)}  "
            f"duration min/median/max {durations.min():.1f}/"
            f"{np.median(durations):.1f}/{durations.max():.1f}s  "
            f"{words.sum()} reference words"
        )


if __name__ == "__main__":
    githash = dirtygit.check()
    cfg = load_config(DEFAULT_CONFIG)
    OmegaConf.update(cfg, "githash", githash)
    setup_logging()
    main(cfg)
