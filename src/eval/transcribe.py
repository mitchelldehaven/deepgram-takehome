"""Run each configured system over the subset and cache raw hypotheses."""
import dirtygit
from loguru import logger
from omegaconf import OmegaConf

from src.config import load_config, setup_logging
from src.paths import CONFIG_DIR, TRANSCRIPTS_DIR

DEFAULT_CONFIG = CONFIG_DIR / "eval" / "openasr.yaml"


def main(cfg):
    logger.info(f"githash={cfg.githash}")
    # TODO: for each system, transcribe SUBSET_FILE, write TRANSCRIPTS_DIR/{name}.jsonl
    raise NotImplementedError


if __name__ == "__main__":
    githash = dirtygit.check()
    cfg = load_config(DEFAULT_CONFIG)
    OmegaConf.update(cfg, "githash", githash)
    setup_logging()
    main(cfg)
