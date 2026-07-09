"""
Smoke-test the implementation without requiring downloaded weights.

1. Imports
2. Architecture forward pass with a tiny synthetic config
3. Generate a short sequence (greedy) with the tiny model
4. Tokenizer round-trip with the real tokenizer.json
5. Loader key-mapping correctness
"""
import sys
import torch

WEIGHTS_DIR = "weights"

# ── 1. Imports ────────────────────────────────────────────────────────────────
print("1. imports ...", end=" ", flush=True)
from model.config import Qwen3Config
from model.norm import RMSNorm
from model.rope import precompute_freqs_cis, apply_rotary_emb
from model.mlp import MLP
from model.attention import Attention
from model.block import Block
from model.qwen3 import Qwen3
from model.loader import _map_key
print("OK")

# ── 2. Tiny model forward pass ────────────────────────────────────────────────
print("2. architecture forward pass ...", end=" ", flush=True)
cfg = Qwen3Config(
    vocab_size=256,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    max_position_embeddings=128,
    rms_norm_eps=1e-6,
    rope_theta=10_000.0,
)
model = Qwen3(cfg).eval()

B, T = 2, 12
ids = torch.randint(0, cfg.vocab_size, (B, T))
with torch.no_grad():
    logits, kv = model(ids)

assert logits.shape == (B, T, cfg.vocab_size), f"bad logits shape: {logits.shape}"
assert len(kv) == cfg.num_hidden_layers
k0, v0 = kv[0]
assert k0.shape == (B, T, cfg.num_key_value_heads, cfg.head_dim), f"bad K shape: {k0.shape}"
print("OK")

# ── 3. Autoregressive generate ────────────────────────────────────────────────
print("3. generate (greedy, 10 tokens) ...", end=" ", flush=True)
prompt = torch.randint(0, cfg.vocab_size, (1, 4))
with torch.no_grad():
    out = model.generate(prompt, max_new_tokens=10, temperature=1.0, top_p=1.0, eos_token_id=999)
assert out.shape[1] == 4 + 10, f"unexpected output length: {out.shape}"
print("OK")

# ── 4. KV-cache decode step (single token after prefill) ─────────────────────
print("4. KV-cache single-step decode ...", end=" ", flush=True)
with torch.no_grad():
    _, kv_after_prefill = model(prompt)
    next_tok = out[:, -1:].clone()
    logits2, _ = model(next_tok, kv_caches=kv_after_prefill, start_pos=prompt.shape[1])
assert logits2.shape == (1, 1, cfg.vocab_size)
print("OK")

# ── 5. Tokenizer round-trip ───────────────────────────────────────────────────
print("5. tokenizer round-trip ...", end=" ", flush=True)
import os
tok_path = os.path.join(WEIGHTS_DIR, "tokenizer.json")
if not os.path.exists(tok_path):
    print(f"SKIP (tokenizer.json not found at {tok_path})")
else:
    from tokenizer import Qwen3Tokenizer
    tok = Qwen3Tokenizer(tok_path)
    text = "Hello, Qwen3!"
    ids_enc = tok.encode(text)
    decoded = tok.decode(ids_enc)
    assert text in decoded or decoded in text, f"round-trip mismatch: {repr(decoded)}"
    print(f"OK  ({repr(text)} → {ids_enc} → {repr(decoded)})")

# ── 6. Chat template ──────────────────────────────────────────────────────────
print("6. chat template ...", end=" ", flush=True)
if not os.path.exists(tok_path):
    print("SKIP")
else:
    msgs = [{"role": "user", "content": "What is RoPE?"}]
    chat_ids = tok.apply_chat_template(msgs)
    assert isinstance(chat_ids, list) and len(chat_ids) > 0
    print(f"OK  ({len(chat_ids)} tokens)")

# ── 7. Loader key mapping ─────────────────────────────────────────────────────
print("7. loader key mapping ...", end=" ", flush=True)
cases = {
    "model.embed_tokens.weight":                        "embed_tokens.weight",
    "model.layers.3.input_layernorm.weight":            "layers.3.ln1.weight",
    "model.layers.3.post_attention_layernorm.weight":   "layers.3.ln2.weight",
    "model.layers.3.self_attn.q_proj.weight":           "layers.3.attn.q_proj.weight",
    "model.layers.3.self_attn.q_norm.weight":           "layers.3.attn.q_norm.weight",
    "model.layers.3.mlp.gate_proj.weight":              "layers.3.mlp.gate_proj.weight",
    "model.norm.weight":                                "norm.weight",
    "lm_head.weight":                                   "lm_head.weight",
}
for hf, expected in cases.items():
    got = _map_key(hf)
    assert got == expected, f"  {hf!r} → {got!r}, expected {expected!r}"
print("OK")

print("\nAll checks passed.")
