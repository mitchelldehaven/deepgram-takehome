"""Slice scores by metadata, and check the metadata is worth slicing on.

The reliability check matters as much as the breakdown: an SNR estimate or an
accent label that disagrees with a second estimator cannot carry a conclusion.
"""
import dirtygit
from loguru import logger
from omegaconf import OmegaConf

from src.config import load_config, setup_logging
from src.paths import CONFIG_DIR, METADATA_FILE

DEFAULT_CONFIG = CONFIG_DIR / "eval" / "openasr.yaml"


def check_metadata_reliability(metadata):
    # TODO: agreement against a second estimator / hand-labeled sample, per field
    raise NotImplementedError


def main(cfg):
    logger.info(f"githash={cfg.githash}")
    # TODO: join scores with metadata, report per-bin WER with CIs
    raise NotImplementedError


if __name__ == "__main__":
    githash = dirtygit.check()
    cfg = load_config(DEFAULT_CONFIG)
    OmegaConf.update(cfg, "githash", githash)
    setup_logging()
    main(cfg)
