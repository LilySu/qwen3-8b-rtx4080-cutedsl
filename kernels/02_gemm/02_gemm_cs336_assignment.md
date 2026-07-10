# Linear Projection (GEMM) — CS336 Assignment 1 Reference

*Source: Stanford CS336 Spring 2026 — Assignment 1: Building a Transformer LM, §3.2.1, §3.3.2*

---

## 3.2.1 Mathematical Notation and Memory Ordering

Most ML papers use **row vectors**, which mesh with NumPy/PyTorch's row-major memory ordering.
With row vectors, a linear transformation is:

$$\mathbf{y} = \mathbf{x} W^\top \tag{1}$$

for row-major $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$ and row-vector
$\mathbf{x} \in \mathbb{R}^{1 \times d_\text{in}}$. This batches naturally by replacing $\mathbf{x}$
with $X \in \mathbb{R}^{\text{batch} \times d_\text{in}}$.

Linear algebra convention uses **column vectors**:

$$\mathbf{y} = W \mathbf{x} \tag{2}$$

CS336 uses column vectors in math. In PyTorch (row-major), this becomes $Y = X W^\top$.

---

## 3.3.2 Linear Module

Linear layers perform:

$$\mathbf{y} = W \mathbf{x} \tag{3}$$

Modern LLMs (PaLM, LLaMA, Qwen) **omit bias terms** from linear layers.

### Recommended interface

```python
def __init__(self, in_features: int, out_features: int, device=None, dtype=None)
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Store the parameter as $W$ (not $W^\top$). Initialize with `torch.nn.init.trunc_normal_`.

---

## Einsum notation (§3.2)

CS336 strongly recommends einsum for self-documenting, batch-safe code:

```python
from einops import einsum

# Linear projection — works with any leading batch dims
Y = einsum(X, W, "... d_in, d_out d_in -> ... d_out")

# Attention scores — batched over heads
scores = einsum(Q, K, "batch heads seq_q d_k, batch heads seq_k d_k -> batch heads seq_q seq_k")
```

---

## Qwen3-8B linear projection shapes

All projections are **no-bias**. At `d_model = 4096`, `d_kv = 1024` (8 heads × 128), `d_ff = 12288`:

| Projection | Shape (out × in) | Role |
|-----------|-----------------|------|
| `q_proj` | (4096, 4096) | 32 Q heads × 128 |
| `k_proj` | (1024, 4096) | 8 KV heads × 128 |
| `v_proj` | (1024, 4096) | 8 KV heads × 128 |
| `o_proj` | (4096, 4096) | concat heads → d_model |
| `gate_proj` | (12288, 4096) | SwiGLU gate branch |
| `up_proj` | (12288, 4096) | SwiGLU up branch |
| `down_proj` | (4096, 12288) | SwiGLU output |
| `lm_head` | (151936, 4096) | vocab logits |

### Decode vs prefill regime

At **batch=1, seq_len=1** (decode), each projection degenerates from GEMM to **GEMV**
(matrix-vector product). Tensor cores cannot be efficiently utilized for M=1.
The roofline ceiling is pure memory bandwidth: $\sim$432 GB/s.

At **seq_len ≥ 64** (prefill), the operation becomes a proper GEMM and tensor cores engage.
Arithmetic intensity crosses the roofline at approximately batch ≥ 64 for BF16.

$$\text{Arithmetic Intensity} = \frac{2MNK}{(MK + NK + MN) \cdot \text{bytes}}$$
