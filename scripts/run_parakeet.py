"""Parakeet inference, run inside .venv-nemo.

NeMo pins a much older transformers than the rest of the project needs, so it
lives in its own environment and talks to the pipeline over files rather than
imports. src/systems/parakeet.py is the caller.

Usage: python scripts/run_parakeet.py <model_id> <manifest.jsonl> <out.jsonl> <batch_size>
"""
import json
import sys

import nemo.collections.asr as nemo_asr


def main(model_id, manifest_path, out_path, batch_size):
    with open(manifest_path) as f:
        items = [json.loads(line) for line in f]

    model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)
    model.eval()

    paths = [item["audio_filepath"] for item in items]
    outputs = model.transcribe(paths, batch_size=batch_size)

    with open(out_path, "w") as f:
        for item, out in zip(items, outputs):
            text = out.text if hasattr(out, "text") else str(out)
            f.write(json.dumps({"id": item["id"], "hypothesis": text.strip()}) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
