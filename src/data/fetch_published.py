"""Pull the published leaderboard table so we have something to compare against.

The results CSVs the Space reads from (hf-audio/open-asr-leaderboard-results,
hf-audio/asr_leaderboard_h200) are gated, so we go through the Space's own
Gradio endpoint instead, which serves the same table publicly.
"""
import dirtygit
import pandas as pd
from gradio_client import Client
from loguru import logger
from omegaconf import OmegaConf

from src.config import load_config, setup_logging
from src.paths import CONFIG_DIR, PUBLISHED_FILE

DEFAULT_CONFIG = CONFIG_DIR / "eval" / "openasr.yaml"

SPACE = "hf-audio/open_asr_leaderboard"
ENDPOINT = "/_main_table_update"


def find_dataframe(obj):
    """The endpoint returns nested gradio component payloads; pull out the table."""
    if isinstance(obj, dict):
        if "headers" in obj and "data" in obj:
            return obj
        for value in obj.values():
            found = find_dataframe(value)
            if found:
                return found
    if isinstance(obj, (list, tuple)):
        for value in obj:
            found = find_dataframe(value)
            if found:
                return found
    return None


def fetch(version_id, columns):
    client = Client(SPACE, verbose=False)
    payload = client.predict(
        version_id=version_id,
        search_query="",
        show_proprietary=True,
        show_llm=True,
        selected_columns=list(columns),
        max_size=33.7,
        api_name=ENDPOINT,
    )
    table = find_dataframe(payload)
    if table is None:
        raise RuntimeError("No dataframe in the Space response")
    df = pd.DataFrame(table["data"], columns=table["headers"])
    # The model cell carries HTML (links, 🤗 and NEW badges); keep the bare id.
    df["model"] = (
        df["model"].astype(str).str.replace(r"<[^>]+>", "", regex=True).str.strip()
    )
    df["model"] = df["model"].str.replace(r"(🤗|NEW)+$", "", regex=True).str.strip()
    return df


def main(cfg):
    logger.info(f"githash={cfg.githash}")
    columns = [cfg.leaderboard.rtfx_column] + list(cfg.leaderboard.published_columns)
    df = fetch(cfg.leaderboard.version_id, columns)
    PUBLISHED_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PUBLISHED_FILE, index=False)
    logger.info(f"{len(df)} systems -> {PUBLISHED_FILE}")

    wanted = [s.leaderboard_id for s in cfg.systems if s.leaderboard_id]
    missing = [w for w in wanted if w not in set(df["model"])]
    if missing:
        logger.warning(f"Not in the published table: {missing}")
    for _, row in df[df["model"].isin(wanted)].iterrows():
        logger.info(f"  rank={row['Rank']} {row['model']} avg_wer={row['Average WER ⬇️']}")


if __name__ == "__main__":
    githash = dirtygit.check()
    cfg = load_config(DEFAULT_CONFIG)
    OmegaConf.update(cfg, "githash", githash)
    setup_logging()
    main(cfg)
