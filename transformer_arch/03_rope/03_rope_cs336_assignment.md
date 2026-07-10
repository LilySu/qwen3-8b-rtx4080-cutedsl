# Rotary Position Embeddings (RoPE) — CS336 Assignment 1 Reference

*Source: Stanford CS336 Spring 2026 — Assignment 1: Building a Transformer LM, §3.4.3*

---

## 3.4.3 Relative Positional Embeddings

We implement **Rotary Position Embeddings** (Su et al., 2021), called RoPE. For a query token
$\mathbf{q}^{(i)} = W_q \mathbf{x}^{(i)} \in \mathbb{R}^d$ at position $i$, we apply a pairwise
rotation matrix $R_i$:

$$\mathbf{q}^{\prime(i)} = R_i \mathbf{q}^{(i)} = R_i W_q \mathbf{x}^{(i)}$$

$R_i$ rotates pairs of embedding elements $q^{(i)}_{2k-1:2k}$ as 2D vectors by angle:

$$\theta_{i,k} = \frac{i}{\Theta^{(2k-2)/d}}, \quad k \in \left\{1, \ldots, \frac{d}{2}\right\}$$

for some constant $\Theta$. $R_i$ is a block-diagonal matrix of size $d \times d$ with blocks
$R_i^k$ for $k \in \{1, \ldots, d/2\}$:

$$R_i^k = \begin{pmatrix} \cos(\theta_{i,k}) & -\sin(\theta_{i,k}) \\ \sin(\theta_{i,k}) & \cos(\theta_{i,k}) \end{pmatrix} \tag{8}$$

The full rotation matrix:

$$R_i = \begin{pmatrix} R_i^1 & 0 & 0 & \cdots & 0 \\ 0 & R_i^2 & 0 & \cdots & 0 \\ 0 & 0 & R_i^3 & \cdots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \cdots & R_i^{d/2} \end{pmatrix} \tag{9}$$

### Key implementation insight

Do **not** construct the full $d \times d$ matrix. Use the block-diagonal structure directly:
split the vector into pairs, apply 2D rotations in-place. cos/sin values can be **precomputed
and reused** across layers and batches. Use `register_buffer(persistent=False)`.

### Recommended interface

```python
def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None)
    # Precompute cos/sin buffers: shape (max_seq_len, d_k // 2)

def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor
    # x: (..., seq_len, d_k)
    # token_positions: (..., seq_len)  — enables arbitrary position offsets for KV cache
    # returns: same shape as x
```

### Application in multi-head attention

RoPE is applied to **Q and K only, not V**. The head dimension acts as a batch dimension —
the same rotation applies to each head independently:

```
x  →  q_proj  →  Q  →  RoPE(Q, positions)  →  Attention
x  →  k_proj  →  K  →  RoPE(K, positions)  →  Attention
x  →  v_proj  →  V  →  (no RoPE)           →  Attention
```

### Qwen3-8B specifics

- $\Theta = 1{,}000{,}000$ (long-context RoPE, vs 10,000 in original paper)
- `d_k = head_dim = 128`
- `max_seq_len = 40,960`
- Applied after QK-norm (Qwen3 applies per-head RMSNorm to Q and K *before* RoPE)
- Precomputed buffer: `(40960, 64)` complex64 — ~10 MB, fits permanently in SM89's 36–48 MB L2

### Why relative position matters

Since $R_i^\top R_j = R_{j-i}$, the dot product $\mathbf{q}^{(i)\top} \mathbf{k}^{(j)}$ after
RoPE depends only on the **relative position** $j - i$, not absolute positions. This is the
key property that makes RoPE effective for length generalization.
