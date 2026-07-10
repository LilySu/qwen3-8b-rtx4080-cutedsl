# Attention Foundations

This note explains attention as a deep-learning operation before worrying about
GPU kernels. The goal is to understand what attention computes, why it is useful
in language models, and how the main equations should be read.

## The Core Problem

Neural networks often process a sequence of token vectors:

$$
X =
\begin{bmatrix}
x_0 \\
x_1 \\
\vdots \\
x_{T-1}
\end{bmatrix}
\in \mathbb{R}^{T \times d_{\text{model}}}
$$

Here:

| Symbol | Meaning |
|---|---|
| $T$ | Number of tokens in the sequence. |
| $d_{\text{model}}$ | Width of each token representation. |
| $x_i$ | Vector representation of token position $i$. |

The question attention answers is:

```text
When updating token i, which other tokens should token i read from?
```

A simple feed-forward layer updates each token independently. Attention lets
each token form a context-dependent mixture of other token vectors.

## Query, Key, Value

Attention starts by projecting input vectors into three different roles:

$$
Q = XW_Q
$$

$$
K = XW_K
$$

$$
V = XW_V
$$

where:

| Symbol | Shape | Meaning |
|---|---:|---|
| $X$ | $T \times d_{\text{model}}$ | Input token representations. |
| $W_Q$ | $d_{\text{model}} \times d_k$ | Learned query projection. |
| $W_K$ | $d_{\text{model}} \times d_k$ | Learned key projection. |
| $W_V$ | $d_{\text{model}} \times d_v$ | Learned value projection. |
| $Q$ | $T \times d_k$ | Queries: what each token is looking for. |
| $K$ | $T \times d_k$ | Keys: what each token can be matched by. |
| $V$ | $T \times d_v$ | Values: information each token can contribute. |

Verbally:

```text
Query: what am I looking for?
Key: what do I contain that others may match?
Value: what information should I provide if selected?
```

The query/key/value split is not three separate input sequences in ordinary
self-attention. They are usually three learned views of the same input $X$.

## Dot-Product Similarity

For a query token $i$ and key token $j$, attention computes:

$$
s_{i,j} = q_i \cdot k_j
$$

Expanded:

$$
s_{i,j}
=
\sum_{r=0}^{d_k-1} Q_{i,r}K_{j,r}
$$

This score is large when query $i$ and key $j$ point in similar directions.

For all tokens at once:

$$
S = QK^T
$$

Shape check:

$$
Q \in \mathbb{R}^{T \times d_k},
\qquad
K^T \in \mathbb{R}^{d_k \times T}
$$

so:

$$
S \in \mathbb{R}^{T \times T}
$$

The score matrix has one row per query and one column per key.

## Scaled Dot-Product Attention

The standard transformer attention operation is:

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
$$

Read this in four steps:

| Step | Formula | Verbal Meaning |
|---:|---|---|
| 1 | $S = QK^T$ | Compare every query with every key. |
| 2 | $\tilde{S} = S/\sqrt{d_k}$ | Keep score magnitudes controlled. |
| 3 | $P = \operatorname{softmax}(\tilde{S})$ | Convert scores into probabilities. |
| 4 | $O = PV$ | Mix value vectors using those probabilities. |

The output shape is:

$$
O \in \mathbb{R}^{T \times d_v}
$$

Each output row is:

$$
o_i
=
\sum_{j=0}^{T-1} p_{i,j}v_j
$$

So token $i$ receives a weighted average of value vectors.

## Why Softmax?

Softmax converts arbitrary scores into positive weights that sum to one:

$$
p_{i,j}
=
\frac{\exp(\tilde{s}_{i,j})}
{\sum_{u=0}^{T-1}\exp(\tilde{s}_{i,u})}
$$

For each query row $i$:

$$
\sum_{j=0}^{T-1}p_{i,j}=1
$$

and:

$$
p_{i,j}\ge0
$$

That makes each row a distribution over source positions.

Softmax attention is therefore a differentiable lookup:

```text
The model softly selects which tokens to read from.
```

## Why Scale By $\sqrt{d_k}$?

The dot product sums $d_k$ terms:

$$
s_{i,j}
=
\sum_{r=0}^{d_k-1} Q_{i,r}K_{j,r}
$$

If the feature products have variance around $1$, then the sum has variance
around $d_k$:

$$
\operatorname{Var}(s_{i,j}) \approx d_k
$$

So the typical score size grows like:

$$
\sqrt{d_k}
$$

Dividing by $\sqrt{d_k}$ keeps the score scale roughly stable:

$$
\operatorname{Var}
\left(
\frac{s_{i,j}}{\sqrt{d_k}}
\right)
\approx
1
$$

Without scaling, large $d_k$ can make softmax too sharp. A too-sharp softmax
puts nearly all probability on one token and gives tiny gradients to the rest.

## Self-Attention

Self-attention means $Q$, $K$, and $V$ all come from the same sequence:

$$
Q = XW_Q,
\qquad
K = XW_K,
\qquad
V = XW_V
$$

This lets every token read from other tokens in the same sequence.

Example:

```text
The token "it" can attend to a previous noun.
The token "bank" can attend to context that disambiguates river bank vs finance bank.
```

Self-attention is content-dependent. The weights are not fixed by position
alone; they are computed from learned token representations.

## Cross-Attention

Cross-attention uses queries from one sequence and keys/values from another:

$$
Q = X_{\text{target}}W_Q
$$

$$
K = X_{\text{source}}W_K
$$

$$
V = X_{\text{source}}W_V
$$

Then:

$$
O =
\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
$$

This appears in encoder-decoder transformers. Decoder tokens query encoder
tokens.

## Causal Attention

Language models generate tokens left to right. Token $i$ should not read future
tokens $j > i$ during training or inference.

This is handled with a causal mask:

$$
\tilde{s}_{i,j} =
\begin{cases}
s_{i,j}/\sqrt{d_k}, & j \le i \\
-\infty, & j > i
\end{cases}
$$

Then softmax gives future positions probability zero:

$$
\exp(-\infty)=0
$$

Verbally:

```text
Mask future tokens before softmax.
Then each position can only attend to itself and earlier positions.
```

## Attention As Dynamic Weighted Averaging

A useful mental model:

```text
Attention computes dynamic weighted averages.
The weights depend on the current input.
```

Convolution uses fixed local patterns. Attention uses learned content-based
patterns. That is why attention can connect distant tokens directly.

## Attention Cost

For sequence length $T$ and head dimension $d_k$:

$$
QK^T \text{ cost} = O(T^2d_k)
$$

The score matrix has size:

$$
T \times T
$$

So memory can scale as:

$$
O(T^2)
$$

This quadratic cost is why efficient-attention kernels and approximations
matter for long contexts.

## Source-Backed Notes

The original Transformer paper, [Attention Is All You Need](https://arxiv.org/abs/1706.03762),
introduced scaled dot-product attention and multi-head attention as the central
replacement for recurrence-heavy sequence modeling. In that paper's notation,
the attention operation is:

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
$$

Two implementation details from modern frameworks are worth knowing:

1. PyTorch exposes this operation directly as
   [`torch.nn.functional.scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).
   The function may dispatch to different backend kernels depending on input
   dtype, device, mask, dropout, and shape.
2. NVIDIA cuDNN treats scaled dot-product attention as a first-class deep
   learning primitive and documents an SDPA operation with FlashAttention-style
   implementations in the [cuDNN Attention documentation](https://docs.nvidia.com/deeplearning/cudnn/latest/operations/Attention.html).

That is a practical signal: attention is no longer just a few matrix operations
written in Python. In modern GPU software stacks, SDPA is a fused primitive with
specialized kernels.

## What This Means For Learning

When first learning attention, write it as:

```python
scores = Q @ K.T / math.sqrt(d_k)
P = softmax(scores)
O = P @ V
```

When learning GPU performance, reinterpret the same expression as a dataflow
problem:

```text
QK^T creates scores.
softmax needs row max and row sum.
PV consumes probabilities and values.
Efficient kernels try not to write the full score/probability matrices to HBM.
```

This is why attention connects directly to the GEMM and memory-hierarchy topics
in the SM89/Ampere notes.

## Summary

Attention is:

```text
content-based routing of information between tokens
```

The core equation is:

$$
O =
\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
$$

The useful interpretation is:

```text
QK^T decides where to look.
softmax decides how strongly to look.
V supplies what information is read.
```

## References

| Resource | Why It Matters |
|---|---|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Original Transformer paper introducing scaled dot-product and multi-head attention. |
| [PyTorch `scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) | Framework-level SDPA API and backend behavior. |
| [NVIDIA cuDNN Attention](https://docs.nvidia.com/deeplearning/cudnn/latest/operations/Attention.html) | Shows SDPA as a modern optimized GPU primitive. |
