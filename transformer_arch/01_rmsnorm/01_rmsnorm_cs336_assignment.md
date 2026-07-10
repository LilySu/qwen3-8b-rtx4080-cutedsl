# RMSNorm — CS336 Assignment 1 Reference

*Source: Stanford CS336 Spring 2026 — Assignment 1: Building a Transformer LM, §3.4.1*

---

## 3.4.1 Root Mean Square Layer Normalization

The original Transformer implementation of Vaswani et al. uses layer normalization to normalize
activations. Following Touvron et al. (LLaMA), we use **root mean square layer normalization**
(RMSNorm; Zhang et al., equation 4).

Given a vector $\mathbf{a} \in \mathbb{R}^{d_\text{model}}$ of activations, RMSNorm rescales each
activation $a_i$ as:

$$\text{RMSNorm}(a_i) = \frac{a_i}{\text{RMS}(\mathbf{a})} \, g_i \tag{4}$$

where

$$\text{RMS}(\mathbf{a}) = \sqrt{\frac{1}{d_\text{model}} \sum_{i=1}^{d_\text{model}} a_i^2 + \varepsilon}$$

- $g_i$ is a learnable **gain** parameter ($d_\text{model}$ parameters total)
- $\varepsilon$ is a numerical stability hyperparameter, typically `1e-5`

### Implementation note

Upcast input to `torch.float32` before squaring to prevent overflow, then downcast the result
back to the original dtype:

```python
in_dtype = x.dtype
x = x.to(torch.float32)
# ... RMSNorm computation ...
return result.to(in_dtype)
```

### Recommended interface

```python
def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None)
def forward(self, x: torch.Tensor) -> torch.Tensor
    # x: (batch_size, sequence_length, d_model)
    # returns: same shape
```

### Qwen3-8B specifics

- Applied **twice per transformer block**: before attention (`ln1`) and before MLP (`ln2`)
- Applied **per-head** to Q and K after projection (**QK-norm**, Qwen3-specific)
- Applied once after all 36 layers (final norm before `lm_head`)
- `d_model = 4096`, `eps = 1e-6` (Qwen3 uses 1e-6, not the CS336 default of 1e-5)
- Total RMSNorm instances: 36 × 2 (block) + 36 × 2 (QK-norm) + 1 (final) = **145**

### Pre-norm vs post-norm

CS336 implements the **pre-norm** Transformer block (Figure 2 in the assignment), where
normalization is applied to the *input* of each sub-layer rather than the output. This is now
standard in LLMs (GPT-3, LLaMA, Qwen3, etc.) because it provides a "clean residual stream"
that improves gradient flow.

```
Input
  │
  ├──► RMSNorm ──► MultiHeadAttention ──► (+) ──► RMSNorm ──► FFN ──► (+) ──►
  │                                        ▲                             ▲
  └────────────────────────────────────────┘─────────────────────────────┘
                    residual connections
```
