"""Download Qwen3-8B weights and tokenizer from HuggingFace to ./weights/"""
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

REPO_ID = "Qwen/Qwen3-8B"
LOCAL_DIR = Path(__file__).parent / "weights"

IGNORE = [
    "*.msgpack",
    "*.h5",
    "flax_model*",
    "tf_model*",
    "rust_model*",
    "onnx/",
    "original/",
]

if __name__ == "__main__":
    token = sys.argv[1] if len(sys.argv) > 1 else None
    print(f"Downloading {REPO_ID} → {LOCAL_DIR}")
    path = snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(LOCAL_DIR),
        ignore_patterns=IGNORE,
        token=token,
    )
    print(f"Done: {path}")
