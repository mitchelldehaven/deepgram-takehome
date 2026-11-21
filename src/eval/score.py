"""Score cached transcripts under each normalizer and write the comparison table."""
import dirtygit
from loguru import logger
from omegaconf import OmegaConf

from src.config import load_config, setup_logging
from src.paths import CONFIG_DIR, SCORES_DIR

DEFAULT_CONFIG = CONFIG_DIR / "eval" / "openasr.yaml"


def main(cfg):
    logger.info(f"githash={cfg.githash}")
    # TODO: system x normalizer grid, bootstrap CIs, diff against PUBLISHED_FILE
    raise NotImplementedError


if __name__ == "__main__":
    githash = dirtygit.check()
    cfg = load_config(DEFAULT_CONFIG)
    OmegaConf.update(cfg, "githash", githash)
    setup_logging()
    main(cfg)
