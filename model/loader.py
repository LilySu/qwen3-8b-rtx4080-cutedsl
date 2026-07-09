import json
import re
from pathlib import Path

import torch
from safetensors import safe_open

from .config import Qwen3Config
from .qwen3 import Qwen3


def _map_key(hf_key: str) -> str:
    key = hf_key.removeprefix("model.")
    key = re.sub(r"layers\.(\d+)\.input_layernorm", r"layers.\1.ln1", key)
    key = re.sub(r"layers\.(\d+)\.post_attention_layernorm", r"layers.\1.ln2", key)
    key = re.sub(r"layers\.(\d+)\.self_attn", r"layers.\1.attn", key)
    return key


def load_from_hf_dir(weights_dir: str | Path, device: str = "cpu") -> Qwen3:
    weights_dir = Path(weights_dir)

    with open(weights_dir / "config.json") as f:
        raw = json.load(f)

    config = Qwen3Config(
        vocab_size=raw["vocab_size"],
        hidden_size=raw["hidden_size"],
        intermediate_size=raw["intermediate_size"],
        num_hidden_layers=raw["num_hidden_layers"],
        num_attention_heads=raw["num_attention_heads"],
        num_key_value_heads=raw["num_key_value_heads"],
        head_dim=raw.get("head_dim", raw["hidden_size"] // raw["num_attention_heads"]),
        max_position_embeddings=raw["max_position_embeddings"],
        rms_norm_eps=raw.get("rms_norm_eps", 1e-6),
        rope_theta=raw.get("rope_theta", 1_000_000.0),
        tie_word_embeddings=raw.get("tie_word_embeddings", False),
    )

    model = Qwen3(config)

    index_path = weights_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            weight_map: dict[str, str] = json.load(f)["weight_map"]
        shard_files = sorted(set(weight_map.values()))
    else:
        shard_files = ["model.safetensors"]

    state_dict: dict[str, torch.Tensor] = {}
    for shard in shard_files:
        with safe_open(weights_dir / shard, framework="pt", device=device) as f:
            for hf_key in f.keys():
                state_dict[_map_key(hf_key)] = f.get_tensor(hf_key)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[loader] missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[loader] unexpected keys ({len(unexpected)}): {unexpected[:5]}")

    return model.to(device)
