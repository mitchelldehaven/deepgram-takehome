"""Compute per-utterance metadata used to break down the results.

Each field is paired with a reliability check in src/eval/breakdown.py, since a
breakdown is only as trustworthy as the metadata it slices on.
"""
import dirtygit
from loguru import logger
from omegaconf import OmegaConf

from src.config import load_config, setup_logging
from src.paths import CONFIG_DIR, METADATA_FILE

DEFAULT_CONFIG = CONFIG_DIR / "eval" / "openasr.yaml"


def estimate_snr(audio, sample_rate):
    # TODO: VAD-gated speech/noise power ratio
    raise NotImplementedError


def speech_rate(reference, duration):
    # TODO: words per second against the reference transcript
    raise NotImplementedError


def oov_rate(reference, vocab):
    # TODO: fraction of reference tokens outside a common-word vocabulary
    raise NotImplementedError


def main(cfg):
    logger.info(f"githash={cfg.githash}")
    # TODO: iterate SUBSET_FILE, compute fields, write METADATA_FILE
    raise NotImplementedError


if __name__ == "__main__":
    githash = dirtygit.check()
    cfg = load_config(DEFAULT_CONFIG)
    OmegaConf.update(cfg, "githash", githash)
    setup_logging()
    main(cfg)
