"""Per-utterance metadata for slicing the results.

Every acoustic field is computed twice, by two estimators that do not share a
failure mode. That is deliberate: a breakdown is only worth as much as the
labels it slices on, and "compute SNR, bin it, report WER per bin" quietly
assumes the SNR estimate is good enough to bin on. src/eval/breakdown.py
checks that assumption against these pairs before reporting anything.
"""
import re
from functools import lru_cache

import dirtygit
import numpy as np
import soundfile as sf
from loguru import logger
from omegaconf import OmegaConf
from tqdm import tqdm

from src.config import load_config, setup_logging
from src.data.utils import load_jsonl, save_jsonl
from src.paths import CONFIG_DIR, METADATA_FILE, PROCESSED_DIR, SUBSET_FILE
from src.systems.base import SAMPLE_RATE

DEFAULT_CONFIG = CONFIG_DIR / "eval" / "openasr.yaml"

FRAME = 400          # 25 ms at 16 kHz
HOP = 160            # 10 ms
EPS = 1e-10

DISFLUENCIES = {"um", "uh", "mm", "hmm", "mhm", "erm", "er", "ah", "eh", "uhhuh"}
WORD = re.compile(r"[a-z']+")
HAS_DIGIT = re.compile(r"\d")


def frame_energies(audio):
    if len(audio) < FRAME:
        audio = np.pad(audio, (0, FRAME - len(audio)))
    n_frames = 1 + (len(audio) - FRAME) // HOP
    strides = np.lib.stride_tricks.as_strided(
        audio,
        shape=(n_frames, FRAME),
        strides=(audio.strides[0] * HOP, audio.strides[0]),
    )
    return (strides.astype(np.float64) ** 2).mean(axis=1)


def snr_percentile(audio):
    """Energy-percentile SNR: the 95th percentile frame is taken as speech and
    the 10th as noise. Cheap and assumption-light, but it breaks down when the
    utterance is nearly all speech, which is common in read corpora.
    """
    energies = frame_energies(audio)
    speech = np.percentile(energies, 95)
    noise = np.percentile(energies, 10)
    return float(10 * np.log10((speech + EPS) / (noise + EPS)))


def wada_ratio(audio):
    """The WADA statistic: arithmetic mean of |x| over its geometric mean.

    Under Kim & Stern's model -- clean speech amplitudes Gamma-distributed with
    shape 0.4, additive Gaussian noise -- this ratio falls monotonically as SNR
    falls, and it reads the *shape* of the amplitude distribution rather than
    assuming the quiet parts are noise. That assumption is what the percentile
    estimator makes, so this is a genuinely independent second opinion and not
    a restatement of the first.
    """
    audio = audio[np.abs(audio) > EPS]
    if len(audio) < FRAME:
        return float("nan")
    absolute = np.abs(audio).astype(np.float64)
    return float(np.mean(absolute) / np.exp(np.mean(np.log(absolute))))


@lru_cache(maxsize=1)
def wada_curve(n_samples=200_000, seed=0):
    """Calibrate ratio -> SNR by simulating the model, rather than hardcoding
    the published lookup table (which I would be copying without being able to
    check it). Monte Carlo is reproducible, self-contained and verifiable:
    tests/test_snr.py checks it recovers injected SNR on real speech.
    """
    rng = np.random.default_rng(seed)
    speech = rng.gamma(shape=0.4, scale=1.0, size=n_samples)
    speech *= rng.choice([-1.0, 1.0], size=n_samples)
    noise = rng.standard_normal(n_samples)

    speech_power = np.mean(speech**2)
    noise_power = np.mean(noise**2)

    snr_grid = np.arange(-20.0, 101.0, 1.0)
    ratios = []
    for snr in snr_grid:
        scale = np.sqrt(speech_power / (noise_power * 10 ** (snr / 10)))
        mixed = np.abs(speech + scale * noise)
        mixed = mixed[mixed > EPS]
        ratios.append(np.mean(mixed) / np.exp(np.mean(np.log(mixed))))
    ratios = np.array(ratios)

    # np.interp needs an increasing x; the curve is monotone increasing in SNR.
    order = np.argsort(ratios)
    return ratios[order], snr_grid[order]


def snr_wada(audio):
    ratio = wada_ratio(audio)
    if not np.isfinite(ratio):
        return float("nan")
    ratios, snrs = wada_curve()
    return float(np.interp(ratio, ratios, snrs))


def voiced_fraction(audio):
    """Fraction of frames above an energy floor set relative to the utterance."""
    energies = frame_energies(audio)
    floor = np.percentile(energies, 10) * 4
    return float((energies > floor).mean())


def text_features(reference):
    lowered = reference.lower()
    words = WORD.findall(lowered)
    n_words = max(len(words), 1)
    return {
        "n_words": len(words),
        "disfluency_rate": sum(w in DISFLUENCIES for w in words) / n_words,
        "numeric_rate": sum(bool(HAS_DIGIT.search(t)) for t in reference.split()) / n_words,
        # Mid-sentence capitalised tokens: a crude proxy for named entities,
        # and the keyword set our added metric scores against. Crude on
        # purpose -- breakdown.py measures how crude.
        "n_caps": sum(
            1
            for token in reference.split()[1:]
            if token[:1].isupper() and not token.isupper()
        ),
    }


def compute(row):
    audio, _ = sf.read(PROCESSED_DIR.parent / row["audio_path"], dtype="float32")
    duration = len(audio) / SAMPLE_RATE
    voiced = voiced_fraction(audio)
    features = text_features(row["reference"])

    return {
        "id": row["id"],
        "dataset": row["dataset"],
        "duration_s": duration,
        # Two SNR estimators, to be cross-checked rather than averaged.
        "snr_percentile_db": snr_percentile(audio),
        "snr_wada_db": snr_wada(audio),
        # Two speech-rate estimators: over the whole utterance, and over only
        # the part of it that carries energy. They diverge when an utterance
        # is mostly silence, which is where a rate label is least trustworthy.
        "speech_rate_wall": features["n_words"] / max(duration, EPS),
        "speech_rate_voiced": features["n_words"] / max(duration * voiced, EPS),
        "voiced_fraction": voiced,
        **features,
    }


def main(cfg):
    logger.info(f"githash={cfg.githash}")
    subset = load_jsonl(SUBSET_FILE)
    rows = [compute(row) for row in tqdm(subset, desc="metadata")]
    METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(rows, METADATA_FILE)
    logger.info(f"{len(rows)} rows -> {METADATA_FILE}")

    for dataset in sorted({r["dataset"] for r in rows}):
        subset_rows = [r for r in rows if r["dataset"] == dataset]
        snr = np.array([r["snr_percentile_db"] for r in subset_rows])
        rate = np.array([r["speech_rate_wall"] for r in subset_rows])
        logger.info(
            f"  {dataset}: median SNR {np.median(snr):.1f} dB, "
            f"median rate {np.median(rate):.2f} w/s"
        )


if __name__ == "__main__":
    githash = dirtygit.check()
    cfg = load_config(DEFAULT_CONFIG)
    OmegaConf.update(cfg, "githash", githash)
    setup_logging()
    main(cfg)
