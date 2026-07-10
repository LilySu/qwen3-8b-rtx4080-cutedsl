# 04 RoPE: Math To Python To CuTeDSL

Matching script: `../04_rope_cutedsl.py`

## What This Component Does

RoPE means rotary positional embedding. It encodes token position by rotating
pairs of hidden dimensions. It is commonly applied to query and key vectors
before attention.

Shapes:

```text
x:   (B, T, D)
cos: (T, D / 2)
sin: (T, D / 2)
y:   (B, T, D)
```

`D` must be even because dimensions are processed in pairs.

## Math

For pair index `p`, the two hidden dimensions are:

$$
d_0 = 2p,\quad d_1 = 2p + 1
$$

For each batch `b` and position `t`:

$$
y_{b,t,d_0} =
\cos_{t,p} x_{b,t,d_0} - \sin_{t,p} x_{b,t,d_1}
$$

$$
y_{b,t,d_1} =
\sin_{t,p} x_{b,t,d_0} + \cos_{t,p} x_{b,t,d_1}
$$

This is a 2D rotation matrix:

$$
\begin{bmatrix}
y_0 \\
y_1
\end{bmatrix}
=
\begin{bmatrix}
\cos \theta & -\sin \theta \\
\sin \theta & \cos \theta
\end{bmatrix}
\begin{bmatrix}
x_0 \\
x_1
\end{bmatrix}
$$

## Python Translation

Plain loop version:

```python
for b in range(B):
    for t in range(T):
        for p in range(D // 2):
            x0 = x[b, t, 2 * p]
            x1 = x[b, t, 2 * p + 1]
            c = cos[t, p]
            s = sin[t, p]
            y[b, t, 2 * p] = c * x0 - s * x1
            y[b, t, 2 * p + 1] = s * x0 + c * x1
```

PyTorch reference:

```python
x_pair = x.reshape(B, T, D // 2, 2)
y_pair = torch.empty_like(x_pair)
y_pair[..., 0] = cos[None, :, :] * x_pair[..., 0] - sin[None, :, :] * x_pair[..., 1]
y_pair[..., 1] = sin[None, :, :] * x_pair[..., 0] + cos[None, :, :] * x_pair[..., 1]
y = y_pair.reshape_as(x)
```

## CuTeDSL Translation

The beginner kernel assigns one GPU thread to one scalar output
`y[b, t, d]`.

It flattens `(B, T, D)` into a single linear index:

```python
i = bidx * THREADS_PER_CTA + tidx
```

Then reconstructs coordinates:

```python
d = i % d_model
bt = i // d_model
t = bt % seq_len
b = bt // seq_len
```

The pair index and parity are:

```python
half = d // 2
is_odd = d % 2
```

Even dimensions compute:

```python
x0 = mX[b, t, d]
x1 = mX[b, t, d + 1]
mY[b, t, d] = c * x0 - s * x1
```

Odd dimensions compute:

```python
x0 = mX[b, t, d - 1]
x1 = mX[b, t, d]
mY[b, t, d] = s * x0 + c * x1
```

## What To Notice

This implementation duplicates some loads: the even thread and odd thread both
read the same pair. That is acceptable for a first version.

A next version would assign one thread to one pair and write both outputs, or
use vectorized loads such as `float2`-style access for coalesced pair reads.
