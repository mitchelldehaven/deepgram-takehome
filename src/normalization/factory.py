from functools import partial

from omegaconf import OmegaConf

from src.normalization import custom, leaderboard
from src.paths import CONFIG_DIR

CUSTOM_RULES = CONFIG_DIR / "normalization" / "custom.yaml"


def load_rules(overrides=None):
    rules = OmegaConf.load(CUSTOM_RULES)
    if overrides:
        rules = OmegaConf.merge(rules, OmegaConf.create(dict(overrides)))
    return rules


def build_normalizer(name, overrides=None):
    if name == "leaderboard":
        return leaderboard.normalize
    if name == "custom":
        return partial(custom.normalize, rules=load_rules(overrides))
    raise ValueError(f"Unknown normalizer: {name}")
