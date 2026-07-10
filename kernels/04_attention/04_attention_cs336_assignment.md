# Attention — CS336 Assignment 1 Reference

*Source: Stanford CS336 Spring 2026 — Assignment 1: Building a Transformer LM, §3.4.4–3.4.5*

---

## Softmax (§3.4.4 preliminary)

Softmax converts unnormalized scores to a normalized distribution:

$$\text{softmax}(\mathbf{v})_i = \frac{\exp(v_i)}{\sum_{j=1}^{n} \exp(v_j)} \tag{10}$$

**Numerical stability**: $\exp(v_i)$ overflows for large $v_i$. Softmax is invariant to adding
any constant $c$ to all inputs, so subtract the row maximum before exponentiating:

$$\text{softmax}(\mathbf{v})_i = \frac{\exp(v_i - \max_j v_j)}{\sum_{j=1}^{n} \exp(v_j - \max_j v_j)}$$

This keeps all exponents $\leq 0$, preventing overflow.

---

## 3.4.4 Scaled Dot-Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V \tag{11}$$

where $Q \in \mathbb{R}^{n \times d_k}$, $K \in \mathbb{R}^{m \times d_k}$, $V \in \mathbb{R}^{m \times d_v}$.

### Masking

A boolean mask $M \in \{\text{True}, \text{False}\}^{n \times m}$: `True` at $(i,j)$ means
query $i$ attends to key $j$. Apply by adding $-\infty$ to pre-softmax scores where `False`:

```python
scores = (Q @ K.T) / math.sqrt(d_k)          # (n, m)
scores = scores.masked_fill(~mask, float('-inf'))
attn = softmax(scores, dim=-1) @ V
```

### Interface

```python
def scaled_dot_product_attention(
    Q: Tensor,    # (batch, ..., seq_q, d_k)
    K: Tensor,    # (batch, ..., seq_k, d_k)
    V: Tensor,    # (batch, ..., seq_k, d_v)
    mask: Tensor | None = None,   # (seq_q, seq_q) bool
) -> Tensor       # (batch, ..., seq_q, d_v)
```

---

## 3.4.5 Causal Multi-Head Self-Attention

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \tag{12}$$

$$\text{head}_i = \text{Attention}(Q_i, K_i, V_i) \tag{13}$$

$$\text{MultiHeadSelfAttention}(x) = W_O \, \text{MultiHead}(W_Q x,\, W_K x,\, W_V x) \tag{14}$$

Learnable parameters: $W_Q \in \mathbb{R}^{hd_k \times d_\text{model}}$,
$W_K \in \mathbb{R}^{hd_k \times d_\text{model}}$, $W_V \in \mathbb{R}^{hd_v \times d_\text{model}}$,
$W_O \in \mathbb{R}^{d_\text{model} \times hd_v}$. Following Vaswani et al., set
$d_k = d_v = d_\text{model} / h$.

### Causal masking

Prevent attending to future tokens using a lower-triangular boolean mask:

```python
mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
# True at (i,j) where j <= i — query i can attend to positions 0..i
```

### Applying RoPE

RoPE applies to Q and K only, not V. The head dimension is a batch dimension — the same
rotation applies to each head independently.

### Interface

```python
def __init__(self, d_model: int, num_heads: int, ...)
    # d_k = d_v = d_model // num_heads

def forward(self, x: Tensor, token_positions: Tensor) -> Tensor
    # x: (batch, seq_len, d_model)
```

---

## Qwen3-8B specifics: GQA + QK-Norm

Qwen3-8B uses **Grouped Query Attention (GQA)**: 32 Q heads, 8 KV heads (ratio 4).
Each KV head is shared by 4 Q heads:

$$\text{kv\_head} = \left\lfloor h / 4 \right\rfloor$$

**QK-Norm** (Qwen3-specific): per-head RMSNorm applied to Q and K *after* projection and
*before* RoPE. This is not in CS336's baseline but is Qwen3's architectural addition for
training stability.

```
x → q_proj → Q_raw → QK-Norm(Q_raw) → RoPE(Q) → Attention
x → k_proj → K_raw → QK-Norm(K_raw) → RoPE(K) → Attention  (8 KV heads, shared across 4 Q heads)
x → v_proj → V                                  → Attention
           → o_proj → output
```

### Flash Attention for SM89

SM89 (100 KB SMEM) cannot run the standard FA2 tile size of `n_block=128`.
At `n_block=64` the SMEM budget is ~65 KB — safe within 100 KB:

$$\text{SMEM} = \text{stages} \times (n_\text{block} \times d_\text{head}) \times \text{bytes} = 3 \times 64 \times 128 \times 2 = 49{,}152 \text{ bytes}$$

### Decode occupancy problem

At batch=1, 8 KV heads on 58 SMs = ~14% occupancy with one CTA per head. Requires
Flash-Decoding-style split-K: split the KV sequence across CTAs, reduce with a second pass.
