# 09 Attention: Math To Python To CuTeDSL

Matching script: `../09_attention_naive_cutedsl.py`

## What This Component Does

Scaled dot-product attention lets each token read information from earlier
tokens. The script uses shape:

```text
q: (B, H, T, D)
k: (B, H, T, D)
v: (B, H, T, D)
y: (B, H, T, D)
```

Where:

- `B` is batch size.
- `H` is number of heads.
- `T` is sequence length.
- `D` is per-head dimension.

## Math

For a query position `t` and key position `s`, the attention score is:

$$
\operatorname{score}_{b,h,t,s}
=
\frac{1}{\sqrt{D}}
\sum_{d=0}^{D-1}
Q_{b,h,t,d} K_{b,h,s,d}
$$

Causal attention masks future positions:

$$
s \le t
$$

The probability for key position `s` is:

$$
P_{b,h,t,s}
=
\frac{\exp(\operatorname{score}_{b,h,t,s})}
{\sum_{u \le t}\exp(\operatorname{score}_{b,h,t,u})}
$$

The output is a weighted sum of values:

$$
Y_{b,h,t,d}
=
\sum_{s \le t} P_{b,h,t,s} V_{b,h,s,d}
$$

## Python Translation

Plain loop version for one scalar output:

```python
for b in range(B):
    for h in range(H):
        for t in range(T):
            for d_out in range(D):
                scores = []
                for s in range(T):
                    if s <= t:
                        score = 0.0
                        for d in range(D):
                            score += q[b, h, t, d] * k[b, h, s, d]
                        scores.append(score / math.sqrt(D))

                probs = softmax(scores)

                acc = 0.0
                for local_idx, s in enumerate(range(t + 1)):
                    acc += probs[local_idx] * v[b, h, s, d_out]
                y[b, h, t, d_out] = acc
```

PyTorch reference:

```python
scores = torch.einsum("bhtd,bhsd->bhts", q, k) / math.sqrt(D)
mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device="cuda"), diagonal=1)
scores = scores.masked_fill(mask[None, None, :, :], float("-inf"))
y = torch.softmax(scores, dim=-1) @ v
```

## CuTeDSL Translation

The beginner kernel assigns one GPU thread to one scalar output:

```text
one thread -> y[b, h, q_pos, d_out]
```

It flattens `(B, H, T, D)`:

```python
i = bidx * THREADS_PER_CTA + tidx
```

Then reconstructs the logical coordinates:

```python
d_out = i % d_model
tmp = i // d_model
q_pos = tmp % seq_len
tmp = tmp // seq_len
h = tmp % heads
b = tmp // heads
```

Pass 1 computes the max score for stable softmax:

```python
row_max = cutlass.Float32(-3.4028234663852886e38)
for k_pos in cutlass.range(seq_len, unroll=1):
    if (not causal) or k_pos <= q_pos:
        score = cutlass.Float32(0.0)
        for d in cutlass.range(d_model, unroll=1):
            score = score + mQ[b, h, q_pos, d].to(cutlass.Float32) * mK[b, h, k_pos, d].to(cutlass.Float32)
        score = score * scale
        if score > row_max:
            row_max = score
```

Pass 2 computes the denominator and the weighted value sum for this `d_out`:

```python
denom = cutlass.Float32(0.0)
acc = cutlass.Float32(0.0)
for k_pos in cutlass.range(seq_len, unroll=1):
    if (not causal) or k_pos <= q_pos:
        score = ...
        p = cute.math.exp(score * scale - row_max, fastmath=True)
        denom = denom + p
        acc = acc + p * mV[b, h, k_pos, d_out].to(cutlass.Float32)
```

Final store:

```python
mY[b, h, q_pos, d_out] = acc / denom
```

## Why This Version Is Educational, Not Fast

This version recomputes the same attention scores for every `d_out`. That makes
the code easy to follow, but expensive.

Production attention kernels use tiling:

```text
load blocks of Q, K, V
-> compute score tiles
-> maintain online softmax statistics
-> accumulate value tiles
-> avoid materializing the full T x T probability matrix
```

That is the idea behind FlashAttention-style kernels. This file is the readable
baseline before that optimization.
