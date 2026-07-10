# SwiGLU MLP — CS336 Assignment 1 Reference

*Source: Stanford CS336 Spring 2026 — Assignment 1: Building a Transformer LM, §3.4.2*

---

## 3.4.2 Position-Wise Feed-Forward Network

Modern LLMs use the **SwiGLU** activation (Shazeer, 2020), combining SiLU/Swish with a
Gated Linear Unit (GLU). This is used in LLaMA 3, Qwen 2.5, and Qwen3.

### SiLU (Swish) activation

$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}} \tag{5}$$

SiLU is similar to ReLU but smooth at zero (no discontinuous gradient at the origin).

### Gated Linear Unit (GLU)

Originally defined by Dauphin et al.:

$$\text{GLU}(x, W_1, W_2) = \sigma(W_1 x) \odot W_2 x \tag{6}$$

where $\odot$ denotes element-wise multiplication. GLUs "reduce the vanishing gradient problem
for deep architectures by providing a linear path for the gradients while retaining non-linear
capabilities."

### SwiGLU (replaces ReLU GLU with SiLU)

$$\text{FFN}(x) = \text{SwiGLU}(x, W_1, W_2, W_3) = W_2 \left(\text{SiLU}(W_1 x) \odot W_3 x\right) \tag{7}$$

where $x \in \mathbb{R}^{d_\text{model}}$,
$W_1, W_3 \in \mathbb{R}^{d_\text{ff} \times d_\text{model}}$,
$W_2 \in \mathbb{R}^{d_\text{model} \times d_\text{ff}}$,
and canonically $d_\text{ff} = \frac{8}{3} d_\text{model}$ (round to nearest multiple of 64).

> "We offer no explanation as to why these architectures seem to work; we attribute their
> success, as all else, to divine benevolence." — Shazeer, 2020

### Implementation note

CS336 permits `torch.sigmoid` for numerical stability in the SiLU computation.
No bias terms, following PaLM and LLaMA conventions.

---

## Qwen3-8B specifics

| Parameter | Value |
|-----------|-------|
| `d_model` | 4096 |
| `d_ff` (`intermediate_size`) | 12288 |
| `d_ff / d_model` ratio | 3.0 (Qwen3 uses exactly 3×, not 8/3×) |
| Projections | `gate_proj` ($W_1$), `up_proj` ($W_3$), `down_proj` ($W_2$) |
| Bias | None |

In code:

```python
def swiglu_mlp(x, gate_w, up_w, down_w):
    gate = F.silu(x @ gate_w.T)   # SiLU(W1 x)
    up   = x @ up_w.T             # W3 x
    return (gate * up) @ down_w.T  # W2 (SiLU(W1 x) ⊙ W3 x)
```

### Naming convention

In HuggingFace / Qwen3 checkpoint keys:
- `gate_proj` = $W_1$ (the branch that goes through SiLU)
- `up_proj` = $W_3$ (the branch that multiplies with the gate)
- `down_proj` = $W_2$ (the output projection)

### Decode vs prefill

At **decode** (M=1): three GEMVs — all bandwidth-bound. `gate_proj` and `up_proj` weights
together are $2 \times 12288 \times 4096 \times 2$ bytes = ~200 MB streamed per token.

At **prefill** (M=seq_len): three GEMMs — compute-bound above ~batch 64. SwiGLU epilogue
fuses the `silu(gate) * up` elementwise into the `gate_proj` GEMM output before `down_proj`,
avoiding a separate memory round-trip. See `dual_gemm_swiglu_epilogue.py` in cutelearning.

### FLOPs per token

$$\text{FLOPs per token} = 2 \times d_\text{model} \times d_\text{ff} \times 3 = 2 \times 4096 \times 12288 \times 3 \approx 302 \text{ MFLOPs}$$

(×36 layers = ~10.9 GFLOPs total for MLP across all layers per token)
