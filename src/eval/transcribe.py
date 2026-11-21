"""Run each configured system over the subset and cache the raw hypotheses.

Transcription is the only expensive, non-deterministic-cost step here, so it
is cached per system and never redone implicitly: everything downstream
(normalization, scoring, breakdowns) reads these files. That split is what
makes it cheap to change a normalization rule and rescore, which is most of
what this exercise asks for.
"""
import dirtygit
import numpy as np
import soundfile as sf
from loguru import logger
from omegaconf import OmegaConf

from src.config import load_config, setup_logging
from src.data.utils import load_jsonl, save_jsonl
from src.paths import CONFIG_DIR, PROCESSED_DIR, SUBSET_FILE, TRANSCRIPTS_DIR
from src.systems.factory import build_system

DEFAULT_CONFIG = CONFIG_DIR / "eval" / "openasr.yaml"


def load_items(subset):
    items = []
    for row in subset:
        audio, _ = sf.read(PROCESSED_DIR.parent / row["audio_path"], dtype="float32")
        items.append(
            {
                "id": row["id"],
                "audio": np.asarray(audio, dtype=np.float32),
                "reference": row["reference"],
                "dataset": row["dataset"],
            }
        )
    return items


def run_system(system_cfg, items, overwrite):
    out_path = TRANSCRIPTS_DIR / f"{system_cfg.name}.jsonl"
    if out_path.exists() and not overwrite:
        logger.info(f"{system_cfg.name}: cached, skipping ({out_path})")
        return

    system = build_system(system_cfg)
    system.load()
    hypotheses = system.transcribe_all(items)

    empty = sum(1 for h in hypotheses if not h.strip())
    if empty:
        logger.warning(f"{system_cfg.name}: {empty}/{len(hypotheses)} empty hypotheses")

    rows = [
        {"id": item["id"], "dataset": item["dataset"], "hypothesis": hypothesis}
        for item, hypothesis in zip(items, hypotheses)
    ]
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    save_jsonl(rows, out_path)
    logger.info(f"{system_cfg.name}: {len(rows)} hypotheses -> {out_path}")


def main(cfg):
    logger.info(f"githash={cfg.githash}")
    subset = load_jsonl(SUBSET_FILE)
    logger.info(f"{len(subset)} utterances from {SUBSET_FILE}")
    items = load_items(subset)

    only = cfg.get("only")
    overwrite = bool(cfg.get("overwrite", False))
    for system_cfg in cfg.systems:
        if only and system_cfg.name != only:
            continue
        try:
            run_system(system_cfg, items, overwrite)
        except Exception as exc:
            # A missing API key or a broken backend should not cost us the
            # systems that do work; score.py reports which are present.
            logger.error(f"{system_cfg.name} failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    githash = dirtygit.check()
    cfg = load_config(DEFAULT_CONFIG)
    OmegaConf.update(cfg, "githash", githash)
    setup_logging()
    main(cfg)
