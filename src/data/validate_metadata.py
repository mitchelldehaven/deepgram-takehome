"""Validate the SNR estimators against known ground truth.

Two estimators agreeing is weak evidence: they can share an assumption and be
wrong together. The stronger test is to manufacture ground truth -- take the
cleanest utterances in the set, add noise at a known SNR, and ask each
estimator to recover the number we put in.

That is what this does. It reports, per estimator, the bias and absolute error
in dB and the rank correlation with true SNR, because the breakdowns in
src/eval/breakdown.py only need the *ordering* to be right: if an estimator
ranks utterances correctly but is biased by a constant, tercile bins are still
sound. Those are different standards and the table separates them.

Caveat worth keeping in view: additive white and babble noise is a friendlier
problem than the reverberation and overlapping speech that actually make AMI
hard. Recovery here is an upper bound on how well these estimators do in the
wild.
"""
import dirtygit
import numpy as np
import pandas as pd
import soundfile as sf
from loguru import logger
from omegaconf import OmegaConf
from tqdm import tqdm

from src.config import load_config, setup_logging
from src.data.metadata import snr_percentile, snr_wada
from src.data.utils import load_jsonl
from src.paths import CONFIG_DIR, PROCESSED_DIR, SCORES_DIR, SUBSET_FILE

DEFAULT_CONFIG = CONFIG_DIR / "eval" / "openasr.yaml"

ESTIMATORS = {"snr_percentile_db": snr_percentile, "snr_wada_db": snr_wada}
TRUE_SNRS = [0.0, 5.0, 10.0, 15.0, 20.0]
N_UTTERANCES = 60
CLEAN_PERCENTILE = 75  # only mix into utterances that start out clean


def babble(rng, length, pool):
    """Overlapped speech from other utterances: the noise ASR actually meets."""
    out = np.zeros(length, dtype=np.float64)
    for _ in range(6):
        source = pool[rng.integers(len(pool))]
        if len(source) < length:
            source = np.pad(source, (0, length - len(source)), mode="wrap")
        start = rng.integers(0, max(1, len(source) - length + 1))
        out += source[start : start + length]
    return out


def mix_at_snr(clean, noise, snr_db):
    clean_power = np.mean(clean**2)
    noise_power = np.mean(noise**2)
    if noise_power <= 0 or clean_power <= 0:
        return None
    scale = np.sqrt(clean_power / (noise_power * 10 ** (snr_db / 10)))
    return clean + scale * noise


def main(cfg):
    logger.info(f"githash={cfg.githash}")
    rng = np.random.default_rng(cfg.scoring.seed)
    subset = load_jsonl(SUBSET_FILE)

    audio = {}
    for row in subset:
        signal, _ = sf.read(PROCESSED_DIR.parent / row["audio_path"], dtype="float32")
        audio[row["id"]] = np.asarray(signal, dtype=np.float64)

    # Mix into the cleanest utterances only: adding 20 dB noise to audio that
    # is already at 10 dB does not produce a 20 dB sample.
    baseline = {uid: snr_percentile(signal) for uid, signal in audio.items()}
    threshold = np.percentile(list(baseline.values()), CLEAN_PERCENTILE)
    clean_ids = [uid for uid, value in baseline.items() if value >= threshold]
    chosen = list(rng.choice(clean_ids, size=min(N_UTTERANCES, len(clean_ids)), replace=False))
    pool = [audio[uid] for uid in clean_ids]
    logger.info(
        f"{len(chosen)} source utterances above the {CLEAN_PERCENTILE}th percentile "
        f"({threshold:.1f} dB by the percentile estimator)"
    )

    rows = []
    for uid in tqdm(chosen, desc="injecting noise"):
        clean = audio[uid]
        for noise_type in ["white", "babble"]:
            noise = (
                rng.standard_normal(len(clean))
                if noise_type == "white"
                else babble(rng, len(clean), pool)
            )
            for true_snr in TRUE_SNRS:
                mixed = mix_at_snr(clean, noise, true_snr)
                if mixed is None:
                    continue
                for name, estimator in ESTIMATORS.items():
                    rows.append(
                        {
                            "id": uid,
                            "noise": noise_type,
                            "true_snr_db": true_snr,
                            "estimator": name,
                            "estimated_snr_db": estimator(mixed.astype(np.float32)),
                        }
                    )

    df = pd.DataFrame(rows)
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SCORES_DIR / "snr_validation.csv", index=False)

    summary = []
    for (name, noise_type), group in df.groupby(["estimator", "noise"]):
        error = group.estimated_snr_db - group.true_snr_db
        summary.append(
            {
                "estimator": name,
                "noise": noise_type,
                "n": len(group),
                "bias_db": error.mean(),
                "mae_db": error.abs().mean(),
                "spearman_with_truth": group[["estimated_snr_db", "true_snr_db"]]
                .corr(method="spearman")
                .iloc[0, 1],
            }
        )
    summary = pd.DataFrame(summary)
    summary.to_csv(SCORES_DIR / "snr_validation_summary.csv", index=False)

    logger.info("SNR estimator recovery of injected noise:")
    for _, row in summary.iterrows():
        logger.info(
            f"  {row.estimator:<20} {row.noise:<7} "
            f"bias {row.bias_db:+6.1f} dB  MAE {row.mae_db:5.1f} dB  "
            f"spearman {row.spearman_with_truth:+.3f}"
        )
    logger.info(f"tables -> {SCORES_DIR}")


if __name__ == "__main__":
    githash = dirtygit.check()
    cfg = load_config(DEFAULT_CONFIG)
    OmegaConf.update(cfg, "githash", githash)
    setup_logging()
    main(cfg)
