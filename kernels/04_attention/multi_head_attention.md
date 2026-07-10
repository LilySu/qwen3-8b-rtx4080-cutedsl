# Multi-Head Attention

Multi-head attention is the standard transformer attention block. It runs
several attention operations in parallel, each with its own learned projections,
then combines their outputs.

## Why Multiple Heads?

A single attention operation produces one attention pattern. Multiple heads let
the model learn several patterns at once.

For example, different heads may learn to focus on:

```text
nearby tokens
subject-verb relationships
copying names or numbers
syntax-like dependencies
long-range topic information
```

This is not hard-coded. The heads are learned. The point is that multiple heads
increase the number of independent relational patterns the block can represent.

## Input Shape

Start with token representations:

$$
X \in \mathbb{R}^{B \times T \times d_{\text{model}}}
$$

where:

| Symbol | Meaning |
|---|---|
| $B$ | Batch size. |
| $T$ | Sequence length. |
| $d_{\text{model}}$ | Model width. |

For a single sequence, you can ignore $B$ and think:

$$
X \in \mathbb{R}^{T \times d_{\text{model}}}
$$

## Head Dimensions

Let:

$$
h = \text{number of heads}
$$

Usually:

$$
d_{\text{head}}
=
\frac{d_{\text{model}}}{h}
$$

Each head works in a lower-dimensional subspace:

$$
d_{\text{head}} < d_{\text{model}}
$$

Example:

```text
d_model = 768
h = 12
d_head = 64
```

## Projection Matrices

A common implementation uses three large projections:

$$
Q = XW_Q
$$

$$
K = XW_K
$$

$$
V = XW_V
$$

with:

$$
W_Q, W_K, W_V
\in
\mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}
$$

Then each result is reshaped into heads:

$$
Q \rightarrow
\mathbb{R}^{B \times h \times T \times d_{\text{head}}}
$$

and similarly for $K$ and $V$.

Verbally:

```text
Project once into a big Q matrix.
Then split the last dimension into heads.
```

This is often faster than launching separate projections per head.

## Per-Head Attention

For head $a$, define:

$$
Q^{(a)}, K^{(a)}, V^{(a)}
\in
\mathbb{R}^{T \times d_{\text{head}}}
$$

The head output is:

$$
O^{(a)}
=
\operatorname{softmax}
\left(
\frac{Q^{(a)}(K^{(a)})^T}{\sqrt{d_{\text{head}}}}
\right)
V^{(a)}
$$

Each head has output shape:

$$
O^{(a)} \in \mathbb{R}^{T \times d_{\text{head}}}
$$

## Concatenating Heads

After all heads compute their outputs:

$$
O^{(0)}, O^{(1)}, \ldots, O^{(h-1)}
$$

they are concatenated along the feature dimension:

$$
O_{\text{concat}}
=
\operatorname{Concat}
\left(
O^{(0)}, O^{(1)}, \ldots, O^{(h-1)}
\right)
$$

Shape:

$$
O_{\text{concat}}
\in
\mathbb{R}^{T \times (h d_{\text{head}})}
$$

Since:

$$
h d_{\text{head}} = d_{\text{model}}
$$

we get:

$$
O_{\text{concat}}
\in
\mathbb{R}^{T \times d_{\text{model}}}
$$

## Output Projection

The concatenated output is mixed by another learned matrix:

$$
Y = O_{\text{concat}}W_O
$$

where:

$$
W_O
\in
\mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}
$$

This lets information from different heads interact.

## Full Multi-Head Formula

The standard compact definition is:

$$
\operatorname{MultiHead}(X)
=
\operatorname{Concat}
\left(
\operatorname{head}_0,
\operatorname{head}_1,
\ldots,
\operatorname{head}_{h-1}
\right)W_O
$$

where:

$$
\operatorname{head}_a
=
\operatorname{Attention}
\left(
XW_Q^{(a)},
XW_K^{(a)},
XW_V^{(a)}
\right)
$$

Verbally:

```text
Each head creates its own Q, K, V.
Each head performs scaled dot-product attention.
The head outputs are concatenated.
A final projection mixes the heads.
```

## Causal Multi-Head Attention

Decoder-only LLMs use causal masking in each head:

$$
\operatorname{head}_a
=
\operatorname{softmax}
\left(
\frac{Q^{(a)}(K^{(a)})^T}{\sqrt{d_{\text{head}}}}
+ M_{\text{causal}}
\right)V^{(a)}
$$

where:

$$
M_{\text{causal},i,j}
=
\begin{cases}
0, & j \le i \\
-\infty, & j > i
\end{cases}
$$

The mask is added before softmax.

## PyTorch-Style Shape Walkthrough

Suppose:

```text
B = 2
T = 128
d_model = 768
h = 12
d_head = 64
```

Input:

```text
X: (2, 128, 768)
```

After Q/K/V projections:

```text
Q: (2, 128, 768)
K: (2, 128, 768)
V: (2, 128, 768)
```

Reshape into heads:

```text
Q: (2, 128, 12, 64)
```

Often transpose to put heads before sequence:

```text
Q: (2, 12, 128, 64)
K: (2, 12, 128, 64)
V: (2, 12, 128, 64)
```

Scores per head:

```text
Q @ K.transpose(-2, -1): (2, 12, 128, 128)
```

Attention probabilities:

```text
P: (2, 12, 128, 128)
```

Weighted values:

```text
P @ V: (2, 12, 128, 64)
```

Transpose and merge heads:

```text
(2, 128, 12, 64) -> (2, 128, 768)
```

Output projection:

```text
Y: (2, 128, 768)
```

## Parameter Count

For standard MHA:

| Projection | Shape | Parameters |
|---|---:|---:|
| $W_Q$ | $d_{\text{model}} \times d_{\text{model}}$ | $d_{\text{model}}^2$ |
| $W_K$ | $d_{\text{model}} \times d_{\text{model}}$ | $d_{\text{model}}^2$ |
| $W_V$ | $d_{\text{model}} \times d_{\text{model}}$ | $d_{\text{model}}^2$ |
| $W_O$ | $d_{\text{model}} \times d_{\text{model}}$ | $d_{\text{model}}^2$ |

Total:

$$
4d_{\text{model}}^2
$$

ignoring bias terms.

## Computational Cost

Attention has two major costs.

Projection cost:

$$
O(BT d_{\text{model}}^2)
$$

Attention score/value mixing cost:

$$
O(BhT^2d_{\text{head}})
$$

Since:

$$
h d_{\text{head}} = d_{\text{model}}
$$

attention cost is:

$$
O(BT^2d_{\text{model}})
$$

The $T^2$ term is the long-context problem.

## MHA vs MQA vs GQA

Many modern LLMs reduce key/value head count to save memory bandwidth during
inference.

| Variant | Query Heads | Key/Value Heads | Main Benefit |
|---|---:|---:|---|
| MHA | many | same number as query heads | Most flexible standard form. |
| MQA | many | 1 | Smaller KV cache, less memory bandwidth. |
| GQA | many | fewer groups than query heads | Middle ground between MHA and MQA. |

In grouped-query attention, several query heads share one K/V head.

This matters during autoregressive decoding because the model repeatedly reads
the KV cache. Reducing K/V heads reduces memory traffic.

## Source-Backed View Of MHA, MQA, And GQA

The original Transformer paper defines multi-head attention as several attention
heads run in parallel, concatenated, then projected with $W_O$:

$$
\operatorname{MultiHead}(Q,K,V)
=
\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_h)W_O
$$

where each head has its own learned projections. Source:
[Attention Is All You Need](https://arxiv.org/abs/1706.03762).

Modern inference work often changes the K/V head structure:

| Variant | K/V Sharing Pattern | Why It Exists |
|---|---|---|
| MHA | Every query head has its own key head and value head. | Maximum flexibility, standard Transformer form. |
| MQA | All query heads share one key head and one value head. | Reduces KV-cache size and decoding memory bandwidth. |
| GQA | Groups of query heads share K/V heads. | Interpolates between MHA quality/flexibility and MQA efficiency. |

The GQA paper, [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245),
frames grouped-query attention as an interpolation between multi-head attention
and multi-query attention. Sebastian Raschka's
[GQA note](https://sebastianraschka.com/llm-architecture-gallery/gqa/) gives a
clear implementation-level explanation: keep more query heads than key/value
heads so multiple query heads share K/V projections and KV-cache entries.

This matters because decoding is often memory-bandwidth sensitive. During
generation, each new token reads stored K/V vectors from previous tokens:

```text
more K/V heads -> larger KV cache -> more memory traffic per generated token
fewer K/V heads -> smaller KV cache -> less memory traffic
```

That is why GQA is common in modern LLMs even though the basic attention
equation remains recognizable.

## PyTorch Implementation Note

PyTorch's [`nn.MultiheadAttention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
is the high-level module. PyTorch also exposes
[`torch.nn.functional.scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html),
which is the lower-level attention primitive that may use optimized fused
kernels.

For learning, implementing MHA manually helps you understand:

```text
linear Q/K/V projections
reshape into heads
scaled dot-product attention per head
merge heads
output projection
```

For performance, prefer the framework SDPA primitive unless you are explicitly
studying kernel implementation.

## Summary

Multi-head attention is:

```text
several scaled dot-product attention operations in parallel
plus a final projection that mixes their outputs
```

The core formula is:

$$
\operatorname{MultiHead}(X)
=
\operatorname{Concat}
\left(
\operatorname{head}_0,\ldots,\operatorname{head}_{h-1}
\right)W_O
$$

with:

$$
\operatorname{head}_a
=
\operatorname{softmax}
\left(
\frac{Q^{(a)}(K^{(a)})^T}{\sqrt{d_{\text{head}}}}
\right)V^{(a)}
$$

The important mental model:

```text
Each head learns a different way to route information across tokens.
```

## References

| Resource | Why It Matters |
|---|---|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Original multi-head attention definition. |
| [GQA paper](https://arxiv.org/abs/2305.13245) | Formal grouped-query attention source. |
| [Sebastian Raschka: GQA](https://sebastianraschka.com/llm-architecture-gallery/gqa/) | Clear implementation-level explanation of GQA and KV-cache motivation. |
| [PyTorch `nn.MultiheadAttention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) | Framework MHA API. |
| [PyTorch `scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) | Lower-level SDPA primitive used by modern implementations. |
