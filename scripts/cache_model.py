"""Explicit, pinned model download for setup only; never receives resume data."""
import json
from pathlib import Path

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "7dbbc90392e2f80f3d3c277d6e90027e55de9125"
TARGET = Path(__file__).resolve().parents[1] / "models/all-MiniLM-L6-v2"


def main():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_ID, revision=REVISION, device="cpu", trust_remote_code=False)
    model.save(str(TARGET))
    (TARGET / "matchwise-model.json").write_text(
        json.dumps({"model_id": MODEL_ID, "revision": REVISION, "device": "cpu"}, indent=2), encoding="utf-8")
    print(f"Model ready: {TARGET}")


if __name__ == "__main__":
    main()
