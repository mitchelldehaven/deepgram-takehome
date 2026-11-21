### Run the end-to-end leaderboard reproduction
set -eou pipefail
export PYTHONPATH=.

# prepare directories
python src/paths.py

# sample the evaluation subset and compute its metadata
python src/data/build_subset.py
python src/data/metadata.py

# transcribe with every configured system (cached per system)
python src/eval/transcribe.py

# score under both normalizers, with bootstrap CIs, against the published numbers
python src/eval/score.py

# break results down by metadata, and test the metadata itself
python src/eval/breakdown.py
