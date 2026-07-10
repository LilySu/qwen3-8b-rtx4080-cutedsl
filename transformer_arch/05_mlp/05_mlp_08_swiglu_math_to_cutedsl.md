# 08 SwiGLU: Math To Python To CuTeDSL

Matching script: `../08_swiglu_cutedsl.py`

## What This Component Does

SwiGLU is an activation used in many Transformer MLP blocks. A typical gated MLP
computes two projections, called here `a` and `b`, then combines them:

```text
y = a * silu(b)
```

## Math

The sigmoid function:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

The SiLU function:

$$
\operatorname{silu}(x) = x \sigma(x)
$$

SwiGLU:

$$
y_i = a_i \cdot \operatorname{silu}(b_i)
$$

Expanded:

$$
y_i = a_i \cdot \frac{b_i}{1 + e^{-b_i}}
$$

## Python Translation

Plain loop:

```python
for i in range(n):
    y[i] = a[i] * (b[i] / (1.0 + math.exp(-b[i])))
```

PyTorch:

```python
y = a * torch.nn.functional.silu(b)
```

## CuTeDSL Translation

This is an elementwise kernel, so it uses the same launch pattern as vector
affine:

```python
i = bidx * THREADS_PER_CTA + tidx
```

Each thread computes one element:

```python
b = mB[i]
mY[i] = mA[i] * (b / (1.0 + cute.math.exp(-b, fastmath=True)))
```

The only device math function needed is:

```python
cute.math.exp(...)
```

## How This Fits A Transformer MLP

A common MLP block is:

$$
\operatorname{MLP}(x) =
\left(W_{\text{up}}x \odot
\operatorname{silu}(W_{\text{gate}}x)\right) W_{\text{down}}
$$

This script implements only the middle elementwise activation:

```text
a = W_up x
b = W_gate x
y = a * silu(b)
```

The linear projections around it are handled by linear/GEMM kernels.
