# Softmax Attention

This note explains the LeetGPU softmax-attention exercise implemented in:

```text
softmax_attention_torch.py
softmax_attention_jax.py
softmax_attention_cutedsl.py
```

The mathematical operation is scaled dot-product attention:

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\!\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

Verbally:

```text
For each query vector, compare it with every key vector.
Turn those comparison scores into probabilities with softmax.
Use the probabilities to take a weighted average of the value vectors.
```

This is the core operation inside transformer self-attention and
cross-attention.

## Big Picture

Attention answers this question:

```text
For this token/query, which other tokens/keys should I look at,
and how much information should I take from each value vector?
```

The three input matrices play different conceptual roles:

| Matrix | Shape | Role | Intuition |
|---|---:|---|---|
| $Q$ | $M \times d$ | queries | What each output position is looking for. |
| $K$ | $N \times d$ | keys | What each source position offers as a match target. |
| $V$ | $N \times d$ | values | The information to mix after deciding which keys matter. |
| $O$ | $M \times d$ | output | One mixed vector per query row. |

Here:

| Symbol | Meaning |
|---|---|
| $M$ | Number of query rows. In self-attention this is often the number of target tokens. |
| $N$ | Number of key/value rows. In self-attention this is often the same token count as $M$. |
| $d$ | Feature dimension of each query/key/value vector in this simplified exercise. |
| $Q_{i,r}$ | Feature $r$ of query row $i$. |
| $K_{j,r}$ | Feature $r$ of key row $j$. |
| $V_{j,r}$ | Feature $r$ of value row $j$. |
| $O_{i,r}$ | Feature $r$ of output row $i$. |

In a full transformer, there are usually extra dimensions:

```text
batch
heads
sequence positions
head dimension
```

This exercise intentionally removes batch and heads so the math is easier to
see:

```text
Q: (M, d)
K: (N, d)
V: (N, d)
O: (M, d)
```

## One Row At A Time

Pick one query row $i$. That row is a vector:

$$
q_i =
\begin{bmatrix}
Q_{i,0} & Q_{i,1} & \cdots & Q_{i,d-1}
\end{bmatrix}
$$

For every key row $j$, we have:

$$
k_j =
\begin{bmatrix}
K_{j,0} & K_{j,1} & \cdots & K_{j,d-1}
\end{bmatrix}
$$

The first question is: how similar is $q_i$ to $k_j$?

The similarity score is a dot product:

$$
s_{i,j}
=
q_i \cdot k_j
=
\sum_{r=0}^{d-1} Q_{i,r}K_{j,r}
$$

Verbally:

```text
Multiply matching features of the query and key.
Add those products.
The result is one scalar compatibility score.
```

If $q_i$ and $k_j$ point in similar directions, their dot product is large. If
they point in unrelated or opposite directions, the score is smaller.

Doing this for all keys creates one row of scores:

$$
s_i =
\begin{bmatrix}
s_{i,0} & s_{i,1} & \cdots & s_{i,N-1}
\end{bmatrix}
$$

This row has length $N$ because query $i$ compares itself against every key.

## Matrix Form Of The Scores

The score matrix is:

$$
S = QK^T
$$

Check the shapes:

$$
Q \in \mathbb{R}^{M \times d}
$$

$$
K \in \mathbb{R}^{N \times d}
$$

Therefore:

$$
K^T \in \mathbb{R}^{d \times N}
$$

and:

$$
QK^T
\in
\mathbb{R}^{M \times d}
\mathbb{R}^{d \times N}
=
\mathbb{R}^{M \times N}
$$

So:

$$
S \in \mathbb{R}^{M \times N}
$$

Each entry is:

$$
S_{i,j}
=
\sum_{r=0}^{d-1} Q_{i,r}K^T_{r,j}
=
\sum_{r=0}^{d-1} Q_{i,r}K_{j,r}
$$

This is why the PyTorch code uses:

```python
Q @ K.T
```

The transpose is not decorative. It turns $K$ from shape $N \times d$ into
shape $d \times N$ so the matrix multiplication lines up.

## Why Divide By $\sqrt{d}$?

The scaled score is:

$$
\tilde{s}_{i,j}
=
\frac{s_{i,j}}{\sqrt{d}}
$$

The full scaled score matrix is:

$$
\tilde{S}
=
\frac{QK^T}{\sqrt{d}}
$$

The reason is numerical and statistical.

Assume, as a rough initialization-time model, that query and key features have
mean $0$ and variance $1$:

$$
\mathbb{E}[Q_{i,r}] = 0,
\qquad
\mathbb{E}[K_{j,r}] = 0
$$

$$
\operatorname{Var}(Q_{i,r}) = 1,
\qquad
\operatorname{Var}(K_{j,r}) = 1
$$

The unscaled dot product is:

$$
s_{i,j}
=
\sum_{r=0}^{d-1} Q_{i,r}K_{j,r}
$$

If the products are roughly independent, the variance of the sum is the sum of
the variances:

$$
\operatorname{Var}(s_{i,j})
\approx
\sum_{r=0}^{d-1}
\operatorname{Var}(Q_{i,r}K_{j,r})
$$

For independent unit-variance variables, each product has variance about $1$:

$$
\operatorname{Var}(Q_{i,r}K_{j,r}) \approx 1
$$

So:

$$
\operatorname{Var}(s_{i,j})
\approx
d
$$

That means the typical magnitude of the dot product grows like:

$$
\sqrt{\operatorname{Var}(s_{i,j})}
\approx
\sqrt{d}
$$

If we divide by $\sqrt{d}$:

$$
\operatorname{Var}\!\left(\frac{s_{i,j}}{\sqrt{d}}\right)
=
\frac{\operatorname{Var}(s_{i,j})}{d}
\approx
1
$$

Verbally:

```text
As the vector dimension grows, raw dot products get larger.
Large scores make softmax too sharp.
Dividing by sqrt(d) keeps score magnitudes in a healthier range.
```

Without this scaling, softmax can become almost one-hot early in training. That
causes tiny gradients for most keys because the model assigns nearly all
probability to one position.

## Softmax Turns Scores Into Probabilities

For one query row $i$, softmax is:

$$
p_{i,j}
=
\frac{\exp(\tilde{s}_{i,j})}
{\sum_{u=0}^{N-1}\exp(\tilde{s}_{i,u})}
$$

Here:

| Symbol | Meaning |
|---|---|
| $p_{i,j}$ | Attention probability from query row $i$ to key/value row $j$. |
| $\exp(\tilde{s}_{i,j})$ | Positive score for key $j$. |
| $\sum_{u=0}^{N-1}\exp(\tilde{s}_{i,u})$ | Normalizer across all keys for query $i$. |

The denominator is row-specific. For each fixed $i$:

$$
\sum_{j=0}^{N-1} p_{i,j} = 1
$$

and:

$$
p_{i,j} \ge 0
$$

So each row $p_i$ is a probability distribution over the $N$ key/value rows.

In matrix form:

$$
P
=
\operatorname{softmax}(\tilde{S})
$$

where softmax is applied row by row:

$$
P \in \mathbb{R}^{M \times N}
$$

The PyTorch code expresses this as:

```python
torch.softmax((Q @ K.T) / math.sqrt(d), dim=-1)
```

The argument `dim=-1` means:

```text
Apply softmax across the last dimension.
For scores shaped (M, N), the last dimension is N.
So each query row normalizes across all keys.
```

## Stable Softmax

The CuTeDSL implementation uses a numerically stable softmax. Instead of
computing:

$$
\frac{\exp(\tilde{s}_{i,j})}
{\sum_{u=0}^{N-1}\exp(\tilde{s}_{i,u})}
$$

directly, it first computes:

$$
m_i = \max_{0 \le u < N} \tilde{s}_{i,u}
$$

Then it computes:

$$
p_{i,j}
=
\frac{\exp(\tilde{s}_{i,j} - m_i)}
{\sum_{u=0}^{N-1}\exp(\tilde{s}_{i,u} - m_i)}
$$

This gives the same answer. To see why, multiply numerator and denominator by
$\exp(-m_i)$:

$$
\frac{\exp(\tilde{s}_{i,j})}
{\sum_{u=0}^{N-1}\exp(\tilde{s}_{i,u})}
\cdot
\frac{\exp(-m_i)}{\exp(-m_i)}
=
\frac{\exp(\tilde{s}_{i,j} - m_i)}
{\sum_{u=0}^{N-1}\exp(\tilde{s}_{i,u} - m_i)}
$$

The benefit is that the largest shifted score is:

$$
\max_j(\tilde{s}_{i,j} - m_i) = 0
$$

So the largest exponential is:

$$
\exp(0) = 1
$$

That avoids overflow from very large positive scores.

## Values Are Mixed With The Probabilities

After computing probabilities, attention forms the output row:

$$
O_{i,r}
=
\sum_{j=0}^{N-1} p_{i,j}V_{j,r}
$$

For one query row $i$, this means:

```text
Take value row 0 times probability p[i,0].
Take value row 1 times probability p[i,1].
...
Take value row N-1 times probability p[i,N-1].
Add them.
```

For the whole matrix:

$$
O = PV
$$

Check the shapes:

$$
P \in \mathbb{R}^{M \times N}
$$

$$
V \in \mathbb{R}^{N \times d}
$$

Therefore:

$$
PV
\in
\mathbb{R}^{M \times N}
\mathbb{R}^{N \times d}
=
\mathbb{R}^{M \times d}
$$

That matches the expected output:

$$
O \in \mathbb{R}^{M \times d}
$$

## Complete Elementwise Formula

Combining everything:

$$
O_{i,r}
=
\sum_{j=0}^{N-1}
\left(
\frac{
\exp\!\left(
\frac{1}{\sqrt{d}}
\sum_{t=0}^{d-1} Q_{i,t}K_{j,t}
\right)
}{
\sum_{u=0}^{N-1}
\exp\!\left(
\frac{1}{\sqrt{d}}
\sum_{t=0}^{d-1} Q_{i,t}K_{u,t}
\right)
}
\right)
V_{j,r}
$$

This looks intimidating, but it is just the four operations repeated:

```text
dot product Q row with K row
scale by sqrt(d)
softmax over all keys
weighted sum of V rows
```

## Tiny Numerical Example

Suppose one query row produces three scaled scores:

$$
\tilde{s}_i =
\begin{bmatrix}
2 & 1 & 0
\end{bmatrix}
$$

The softmax denominator is:

$$
Z_i = e^2 + e^1 + e^0
$$

Numerically:

$$
Z_i \approx 7.389 + 2.718 + 1 = 11.107
$$

The probabilities are:

$$
p_i
\approx
\begin{bmatrix}
7.389/11.107 &
2.718/11.107 &
1/11.107
\end{bmatrix}
$$

So:

$$
p_i
\approx
\begin{bmatrix}
0.665 & 0.245 & 0.090
\end{bmatrix}
$$

If the value rows are $v_0$, $v_1$, and $v_2$, then:

$$
o_i
=
0.665v_0 + 0.245v_1 + 0.090v_2
$$

Verbally:

```text
The first value vector contributes the most.
The second contributes some.
The third contributes little.
```

## PyTorch Version

The PyTorch implementation is:

```python
def softmax_attention_torch(Q, K, V):
    d = Q.shape[1]
    return torch.softmax((Q @ K.T) / math.sqrt(d), dim=-1) @ V
```

Line by line:

```python
d = Q.shape[1]
```

This reads the feature dimension.

If:

```text
Q.shape == (M, d)
```

then:

```text
Q.shape[1] == d
```

Next:

```python
Q @ K.T
```

computes all query-key dot products:

$$
QK^T \in \mathbb{R}^{M \times N}
$$

Next:

```python
(Q @ K.T) / math.sqrt(d)
```

scales the scores:

$$
\frac{QK^T}{\sqrt{d}}
$$

Next:

```python
torch.softmax(..., dim=-1)
```

normalizes each row across keys:

$$
P_{i,j}
=
\frac{\exp(\tilde{S}_{i,j})}
{\sum_u \exp(\tilde{S}_{i,u})}
$$

Finally:

```python
P @ V
```

mixes the value rows:

$$
O = PV
$$

## JAX Version

The JAX implementation is mathematically the same:

```python
out = jax.nn.softmax((Q @ K.T) / math.sqrt(d), axis=-1) @ V
```

The main vocabulary difference is:

| PyTorch | JAX | Meaning |
|---|---|---|
| `dim=-1` | `axis=-1` | Apply softmax over the last dimension. |
| `torch.randn` | `jax.random.normal` | Create random tensors. |
| `torch.testing.assert_close` | `np.testing.assert_allclose` | Check numerical agreement. |

The math is unchanged:

$$
O =
\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt{d}}
\right)V
$$

## CuTeDSL Version

The CuTeDSL version is intentionally direct and educational:

```text
one GPU thread -> one scalar output O[row, d_out]
```

The output matrix has shape:

$$
O \in \mathbb{R}^{M \times d}
$$

So there are:

$$
M \cdot d
$$

scalar output elements.

The kernel flattens those scalar outputs into one linear index:

```python
i = bidx * THREADS_PER_CTA + tidx
```

Then it maps the flat index back to a 2D output coordinate:

```python
row = i // d
d_out = i % d
```

Mathematically:

$$
\text{row} = \left\lfloor \frac{i}{d} \right\rfloor
$$

$$
d_{\text{out}} = i \bmod d
$$

So the thread computes:

$$
O_{\text{row}, d_{\text{out}}}
$$

### Pass 1: Find The Row Maximum

The first pass computes:

$$
m_{\text{row}}
=
\max_{0 \le j < N}
\left(
\frac{1}{\sqrt{d}}
\sum_{t=0}^{d-1}
Q_{\text{row},t}K_{j,t}
\right)
$$

In code:

```python
row_max = cutlass.Float32(-3.4028234663852886e38)
for k_row in cutlass.range(N, unroll=1):
    score = cutlass.Float32(0.0)
    for dim in cutlass.range(d, unroll=1):
        score = score + mQ[row, dim] * mK[k_row, dim]
    score = score * scale
    if score > row_max:
        row_max = score
```

The large negative starting value is approximately the most negative finite
FP32 value. It acts like:

$$
-\infty
$$

for the purpose of computing a maximum.

### Pass 2: Compute Denominator And Weighted Value Sum

The second pass recomputes the scores and accumulates two things:

1. The stable softmax denominator:

$$
\operatorname{denom}
=
\sum_{j=0}^{N-1}
\exp(\tilde{s}_{\text{row},j} - m_{\text{row}})
$$

2. The numerator for one output feature:

$$
\operatorname{acc}
=
\sum_{j=0}^{N-1}
\exp(\tilde{s}_{\text{row},j} - m_{\text{row}})
V_{j,d_{\text{out}}}
$$

Then the output is:

$$
O_{\text{row},d_{\text{out}}}
=
\frac{\operatorname{acc}}{\operatorname{denom}}
$$

That equals:

$$
\sum_{j=0}^{N-1}
p_{\text{row},j}
V_{j,d_{\text{out}}}
$$

because:

$$
p_{\text{row},j}
=
\frac{
\exp(\tilde{s}_{\text{row},j} - m_{\text{row}})
}{
\operatorname{denom}
}
$$

## Why The CuTeDSL Version Recomputes Scores

The teaching kernel computes each scalar output independently. That makes the
mapping easy:

```text
thread 0 computes one O[row, d_out]
thread 1 computes another O[row, d_out]
...
```

But it means many threads repeat the same score calculation.

For a fixed row, every output feature $d_{\text{out}}$ uses the same attention
probabilities:

$$
p_{\text{row},0},
p_{\text{row},1},
\ldots,
p_{\text{row},N-1}
$$

Only the value column changes:

$$
V_{j,d_{\text{out}}}
$$

So a faster attention kernel would share work across threads:

```text
1. Cooperatively compute scores for a query block.
2. Cooperatively compute row max.
3. Cooperatively compute softmax denominator.
4. Reuse probabilities or online softmax state while multiplying by V.
5. Tile K and V through shared memory.
```

The current CuTeDSL implementation does not do that. It prioritizes:

```text
simple indexing
clear math
stable softmax
framework comparison
```

over performance.

## What Production Attention Kernels Do Differently

Production attention kernels, such as FlashAttention-style kernels, avoid
materializing the full score matrix $S$ and probability matrix $P$ when possible.

The naive matrix expression:

$$
O =
\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt{d}}
\right)V
$$

suggests these intermediate matrices:

```text
S = QK^T        shape (M, N)
P = softmax(S)  shape (M, N)
O = P V         shape (M, d)
```

For long sequences, $M \times N$ can be very large. A production kernel often
streams over blocks of $K$ and $V$, maintaining enough online softmax state to
produce the correct result without storing all of $S$ or $P$ in global memory.

The key performance ideas are:

| Idea | Why It Helps |
|---|---|
| Tile $Q$, $K$, and $V$ | Reuse data from shared memory or registers instead of rereading global memory. |
| Avoid writing $S$ and $P$ to global memory | Reduces memory traffic. |
| Use online softmax | Allows block-by-block processing while preserving exact softmax semantics. |
| Use vectorized/coalesced loads | Improves global-memory efficiency. |
| Share row reductions across threads | Avoids repeated max and denominator work. |
| Use Tensor Cores for $QK^T$ and $PV$ when shapes/dtypes allow | Improves math throughput. |

This exercise is therefore a conceptual stepping stone:

```text
direct scalar attention
-> tiled attention
-> online softmax attention
-> FlashAttention-style IO-aware attention
```

## Relationship To Transformer Architectures

In a transformer block, attention is usually applied after projecting token
embeddings into query, key, and value vectors:

$$
Q = XW_Q
$$

$$
K = XW_K
$$

$$
V = XW_V
$$

Here:

| Symbol | Meaning |
|---|---|
| $X$ | Input token representations. |
| $W_Q$ | Learned query projection matrix. |
| $W_K$ | Learned key projection matrix. |
| $W_V$ | Learned value projection matrix. |

Then attention computes:

$$
O =
\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt{d}}
\right)V
$$

Then another learned projection usually follows:

$$
Y = OW_O
$$

In multi-head attention, this process happens separately for multiple heads:

```text
head 0 has its own Q, K, V slices
head 1 has its own Q, K, V slices
...
outputs are concatenated and projected
```

The LeetGPU exercise is one head, one batch, no mask, no dropout, and equal
query/value feature dimensions. That is deliberate: it isolates the core math.

## Complexity

For the simplified shapes:

```text
Q: (M, d)
K: (N, d)
V: (N, d)
O: (M, d)
```

The score computation $QK^T$ costs:

$$
O(MNd)
$$

The final multiply $PV$ also costs:

$$
O(MNd)
$$

So total arithmetic is:

$$
O(MNd)
$$

The full score matrix would require:

$$
O(MN)
$$

memory if materialized.

The teaching CuTeDSL kernel avoids explicitly storing $S$ and $P$, but it
recomputes scores separately for each output feature. Because there are $d$
output features per row, that repeated work is expensive:

```text
for each O[row, d_out]:
    recompute all N dot products over d features
```

So its work is closer to:

$$
O(MNd^2)
$$

for the score recomputation pattern, rather than the ideal:

$$
O(MNd)
$$

This is acceptable for a learning kernel but not for a performance kernel.

## Checklist For Understanding The Code

When reading `softmax_attention_cutedsl.py`, track these correspondences:

| Math | Code |
|---|---|
| $M$ | `cute.size(mOut, mode=[0])` |
| $d$ | `cute.size(mOut, mode=[1])` |
| $N$ | `cute.size(mK, mode=[0])` |
| $1/\sqrt{d}$ | `scale` |
| Output coordinate $O_{i,r}$ | `row`, `d_out` |
| Dot product $\sum_t Q_{i,t}K_{j,t}$ | inner loop over `dim` |
| Row max $m_i$ | `row_max` |
| Softmax denominator | `denom` |
| Weighted value numerator | `acc` |
| Final output | `mOut[row, d_out] = acc / denom` |

## Run Commands

PyTorch:

```bash
python learning/leetgpu/softmax_attention_torch.py --M 4 --N 8 --d 16
```

JAX:

```bash
python learning/leetgpu/softmax_attention_jax.py --M 4 --N 8 --d 16
```

CuTeDSL:

```bash
python learning/leetgpu/softmax_attention_cutedsl.py --M 4 --N 8 --d 16
```

The CuTeDSL version compares against:

```python
torch.softmax((Q @ K.T) * scale_value, dim=-1) @ V
```

That reference is the compact mathematical definition written in PyTorch.

## Summary

Softmax attention has four conceptual steps:

| Step | Formula | Meaning |
|---:|---|---|
| 1 | $S = QK^T$ | Compare every query with every key. |
| 2 | $\tilde{S} = S/\sqrt{d}$ | Keep score magnitudes controlled. |
| 3 | $P = \operatorname{softmax}(\tilde{S})$ | Convert scores into row-wise probabilities. |
| 4 | $O = PV$ | Mix value vectors according to those probabilities. |

The teaching CuTeDSL kernel maps one thread to one scalar output and implements
stable softmax directly. That makes the math visible. The cost is repeated
score computation and poor data reuse, which is exactly what more advanced
tiled and FlashAttention-style kernels are designed to fix.
