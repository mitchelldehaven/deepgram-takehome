import sys
from datetime import datetime
from pathlib import Path

from loguru import logger
from omegaconf import OmegaConf

from src.paths import LOGS_DIR


def load_config(default_config):
    argv = sys.argv[1:]
    config_path = Path(default_config)

    if argv and argv[0] == "--config":
        config_path = Path(argv[1])
        argv = argv[2:]

    base_cfg = OmegaConf.load(config_path)
    cli_cfg = OmegaConf.from_dotlist(argv)
    return OmegaConf.merge(base_cfg, cli_cfg)


def setup_logging(script_path=None):
    if script_path is None:
        script_path = sys.argv[0]

    script = Path(script_path)
    name = script.stem if script.suffix == ".py" else script.name

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = LOGS_DIR / name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{timestamp}.log"

    logger.add(log_file, format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}")
    logger.info(f"Logging to {log_file}")
    return log_file
