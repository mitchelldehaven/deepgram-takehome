# Reproducing the Open ASR Leaderboard

A small reproduction of the [🤗 Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)
(English short-form track), built to test how much of a published WER is the
system and how much is the measurement around it.

Findings and limitations are in [REPORT.md](REPORT.md). This file is how to rerun it.

## What it does

1. **Recreate.** Three ranked systems (Whisper large-v3, Qwen3-ASR-1.7B,
   Parakeet TDT 0.6b v2) plus one the board does not rank (Deepgram Nova-3),
   on a sample of three of its evaluation datasets, compared against the
   published numbers with bootstrap confidence intervals.
2. **Test the normalization.** A second normalizer that disagrees with the
   leaderboard's on six specific, individually ablatable rules.
3. **Look past the metric.** Three metrics scored alongside WER --
   catastrophic failure rate, hallucination rate and edge error rate -- plus
   the substitution/deletion/insertion split behind each WER, and a probe
   testing how much of one vendor's edge penalty is the benchmark's own
   segmentation.
4. **Characterize the data.** Per-utterance metadata, used to slice every
   metric — pooled and within each dataset, because the fields are confounded
   with the corpus — and checked for reliability before it is used, with two
   independent estimators per acoustic field. Six of the eight conditioning
   variables move WER more than the entire spread across the five system
   configurations; a decomposition shows roughly half of the largest one is
   geometry rather than difficulty.
5. **Read the disagreements.** A diff tool that takes any two
   (system, normalizer) arms, decomposes the gap between them into exact
   per-utterance contributions, and prints the text behind the largest ones.

## Setup

Requires Python 3.11 and a CUDA GPU (the open models run locally; ~8 GB VRAM
is enough at the default batch sizes).

```bash
uv venv --python 3.11
uv sync
```

NeMo pins `transformers` far below what Qwen3-ASR needs, so Parakeet runs in
its own environment and talks to the pipeline over files
(`src/systems/parakeet.py` → `scripts/run_parakeet.py`):

```bash
uv venv --python 3.11 .venv-nemo
VIRTUAL_ENV=.venv-nemo uv pip install "nemo_toolkit[asr]" soundfile loguru
```

Deepgram needs a key in `.env` at the repo root (gitignored):

```
DEEPGRAM_API_KEY=<your key>
```

Without it, that one system is skipped with a warning and everything else
still runs.

## Running

```bash
bash run_eval.sh
```

Or one stage at a time (every entrypoint takes `--config <path>` and
`key=value` overrides, via OmegaConf):

```bash
export PYTHONPATH=.
.venv/bin/python src/data/fetch_published.py          # published numbers to compare against
.venv/bin/python src/data/build_subset.py             # sample the evaluation subset
.venv/bin/python src/data/metadata.py                 # per-utterance metadata
.venv/bin/python src/eval/transcribe.py               # run the systems (cached per system)
.venv/bin/python src/eval/score.py                    # four metrics, CIs, published comparison, ablation
.venv/bin/python src/eval/breakdown.py                # metadata reliability, metrics sliced by it
```

Two tools sit outside the default run because they answer follow-up questions
rather than produce the headline tables:

```bash
.venv/bin/python src/eval/compare.py                  # diff two arms, utterance by utterance
.venv/bin/python src/eval/padding_probe.py            # is an edge penalty the model or the segmentation?
```

Useful overrides:

```bash
.venv/bin/python src/eval/transcribe.py only=whisper_large_v3    # one system
.venv/bin/python src/eval/transcribe.py overwrite=true           # ignore the cache
.venv/bin/python src/data/build_subset.py subset.utterances_per_dataset=100
```

## Validating the subset against the full splits

`config/eval/full.yaml` + `src/eval/subset_validation.py` run the identical
pipeline over **every row of all three test splits** (~65,900 utterances,
~144 h, ~16 GB of audio, open models only) and compare it to the
1,200-utterance subset the report is written from.

**This is provided as runnable code, not as a result. REPORT.md does not rely
on it and makes no claim from it.** It was written last and the run takes a
few hours on one GPU; if the numbers are not in REPORT.md, the run had not
finished.

What it measures, isolating three effects that are otherwise confounded:

```
A  subset ids, subset run   the number in REPORT.md
B  subset ids, full run     same audio, same config, different batch neighbours
C  all ids,    full run     the whole split
D  published

A->B  batch composition -- REPORT.md asserts this matters for Whisper
B->C  sampling, cleanly isolated
C->D  what is left: decoding config, model version. Sampling cannot be here.
```

plus the calibration check that motivated it: **does the full-split WER fall
inside the 95% CI the 400-row sample predicted?**

```bash
export PYTHONPATH=. RUN_TAG=full
.venv/bin/python src/data/build_subset.py --config config/eval/full.yaml
.venv/bin/python src/eval/transcribe.py   --config config/eval/full.yaml
.venv/bin/python src/eval/score.py        --config config/eval/full.yaml
.venv/bin/python src/eval/subset_validation.py
```

`RUN_TAG` (see `src/paths.py`) tags only what a run *produces* —
`data/processed-full/`, `outputs-full/` — so a full run cannot overwrite the
subset run. Batch sizes in `full.yaml` are deliberately identical to
`openasr.yaml`: batching changes padding and padding changes Whisper's output,
so raising them would confound "full split vs subset" with "different
decoding configuration".

## Comparing two arms

`src/eval/compare.py` answers "where, exactly, do these two differ". An *arm*
is a `system:normalizer` pair, so one tool covers normalizer diffs, system
diffs and single-rule diffs:

```bash
export PYTHONPATH=.
P=.venv/bin/python

# one system, two normalizers: what does our normalizer change?
$P src/eval/compare.py \
    compare.a=deepgram_nova_3_raw:leaderboard \
    compare.b=deepgram_nova_3_raw:custom

# two systems, one normalizer, on one metric and one dataset
$P src/eval/compare.py \
    compare.a=deepgram_nova_3_raw compare.b=parakeet_tdt_0_6b_v2 \
    compare.metric=hallucination compare.dataset=AMI-Cleaned

# one normalization rule, flipped: the utterances behind an ablation row
$P src/eval/compare.py \
    compare.a=deepgram_nova_3_raw:custom compare.b=deepgram_nova_3_raw:custom \
    compare.b_rules.remove_fillers=false
```

It prints a headline (both values, the paired-bootstrap gap and whether the
sample separates them), then the top utterances with errors marked inline —
`{ref>hyp}` substitution, `[-word]` deletion, `[+word]` insertion — and writes
every utterance to a CSV in `outputs/scores/`.

The per-utterance `contribution` column is exact, not a heuristic: each metric
is `sum(numerators) / sum(denominators)`, so an utterance contributes
`n_a[i]/D_a - n_b[i]/D_b` to the gap and those contributions sum to it. That
holds even when the arms have different denominators, which they do whenever
the normalizers differ — normalization changes the reference word count.

Entrypoints assert a clean git tree (via `dirtygit`) and stamp the commit hash
into the log, so every result is traceable to the code that produced it. While
iterating, `IGNORE_DIRTYGIT=true` skips the assert.

## Layout

```
config/
  eval/openasr.yaml            datasets, systems, sampling and scoring settings
  normalization/custom.yaml    our normalization rules, one toggle each
src/
  paths.py                     every path in the project
  config.py                    config loading + logging
  data/
    fetch_published.py         pull the published leaderboard table
    build_subset.py            sample the evaluation subset
    metadata.py                per-utterance metadata, two estimators per field
  systems/                     one file per ASR system, behind a shared interface
  normalization/
    vendor/                    the leaderboard's normalizer, pinned
    custom.py                  ours, rule-driven
  eval/
    transcribe.py              run systems, cache raw hypotheses
    metrics.py                 the four metrics, bootstrap, paired tests
    score.py                   scoring, published comparison, rule ablation
    breakdown.py               metadata reliability, slices, length decomposition
    compare.py                 diff two arms, utterance by utterance
    subset_validation.py       subset vs full splits (runnable; not run in the report)
    padding_probe.py           is an edge penalty the model or the segmentation?
scripts/run_parakeet.py        NeMo inference, runs inside .venv-nemo
tests/                         regression tests: normalizer, metrics, diff tool
outputs/                       transcripts and score tables (gitignored)
```

## Tests

```bash
PYTHONPATH=. uv run pytest tests/ -q
```

They cover three things: the custom normalizer's number handling, where three
bugs lived long enough to reach a draft of the report; the metric contract,
where a test caught an insertion being charged to the middle of an utterance
rather than its trailing edge; and the diff tool's gap decomposition, which
has to sum exactly to the headline or it is sorting by a meaningless number. The useful property to preserve
is that the same quantity normalizes identically whether the source wrote
`twenty-five` or `25` -- otherwise canonicalizing numbers accomplishes
nothing. The SNR estimators are validated separately, against injected
ground truth, by `src/data/validate_metadata.py`.

## Notes on the data

The upstream test parquets are sorted by audio length, globally and
descending. Reading the first shard, or the first N rows, therefore gives only
the longest utterances. `build_subset.py` samples shards at even intervals
across that sorted order and then samples rows within each, which spans the
duration distribution by construction. See the module docstring for why that
beats both naive slicing and row-group range reads.

Two of the board's default datasets are not in the aggregated repo and are not
used here: `Earnings22-Cleaned-AA-chunked` (scored by concatenating chunks
against a separate full-transcript dataset) and `Voice Arena Monsoon` (its own
repo). Two more of its columns are private. So the "Average WER" column is not
reproducible by anyone outside Hugging Face; this repo compares per-dataset
numbers instead, which are.

## Adding things

- **A system**: one file in `src/systems/` implementing `ASRSystem.transcribe`,
  one entry in `BACKENDS`, one block in the config. Override `transcribe_all`
  if it batches differently (an API to fan out, a subprocess to load once).
- **A normalization rule**: a flag in `config/normalization/custom.yaml` and a
  branch in `src/normalization/custom.py`. The ablation in `score.py` picks it
  up automatically.
- **A metric**: a function in `src/eval/metrics.py` returning per-utterance
  `(numerator, denominator)`, plus one entry in `METRICS`. That is the whole
  job -- bootstrap CIs, paired significance tests, per-dataset tables and the
  metadata breakdowns all pick it up with no further change. The four current
  metrics deliberately use different denominators (reference words,
  utterances, utterances, edge word slots) to keep that contract honest.
