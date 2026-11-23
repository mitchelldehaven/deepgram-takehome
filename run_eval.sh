### Run the end-to-end Open ASR Leaderboard reproduction
set -eou pipefail
export PYTHONPATH=.
PY=.venv/bin/python

# prepare directories
$PY src/paths.py

# pull the published table we are going to be compared against
$PY src/data/fetch_published.py

# sample the evaluation subset and characterise it
$PY src/data/build_subset.py
$PY src/data/metadata.py

# check the metadata is reliable enough to slice on, against injected ground truth
$PY src/data/validate_metadata.py

# transcribe with every configured system (cached per system in outputs/transcripts)
# add `only=<system_name>` to run one, `overwrite=true` to force a re-run
$PY src/eval/transcribe.py

# score under both normalizers, with bootstrap CIs, against the published numbers
$PY src/eval/score.py

# break the results down by metadata, after testing the metadata itself
$PY src/eval/breakdown.py

# follow-up tools, not part of the default run:
#
# where do two (system, normalizer) arms actually differ, utterance by utterance?
# $PY src/eval/compare.py compare.a=<system>[:<normalizer>] compare.b=<system>[:<normalizer>]
#
# was Nova-3's edge penalty the model, or the benchmark's segmentation?
# costs Deepgram API calls, so it is not part of the default run
# $PY src/eval/transcribe.py --config config/eval/padding_probe.yaml
# $PY src/eval/padding_probe.py
