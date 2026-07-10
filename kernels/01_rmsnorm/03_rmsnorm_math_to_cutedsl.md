# 03 RMSNorm: Math To Python To CuTeDSL

Matching script: `../03_rmsnorm_row_cutedsl.py`

## What This Component Does

RMSNorm rescales each row of hidden states by the row's root-mean-square value,
then applies a learned per-channel weight.

Shapes:

```text
x:      (M, D)
weight: (D,)
y:      (M, D)
```

Here `M` usually means flattened `(batch * sequence)` rows, and `D` is model
width.

## Math

For row `m`, first compute the mean square:

$$
\operatorname{ms}_m = \frac{1}{D}\sum_{d=0}^{D-1} x_{m,d}^2
$$

Then compute the inverse RMS:

$$
r_m = \frac{1}{\sqrt{\operatorname{ms}_m + \epsilon}}
$$

Finally normalize and scale each feature:

$$
y_{m,d} = x_{m,d} \cdot r_m \cdot w_d
$$

RMSNorm differs from LayerNorm because it does not subtract the row mean.

## Python Translation

Plain loop version:

```python
for m in range(M):
    ss = 0.0
    for d in range(D):
        ss += x[m, d] * x[m, d]

    inv_rms = 1.0 / math.sqrt(ss / D + eps)

    for d in range(D):
        y[m, d] = x[m, d] * inv_rms * weight[d]
```

PyTorch version:

```python
inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
y = x * inv_rms * weight
```

## CuTeDSL Translation

The beginner CuTeDSL version assigns one CUDA block to one row:

```text
block_idx.x -> row m
```

But it launches only one thread per block:

```python
rmsnorm_kernel(...).launch(grid=(rows, 1, 1), block=(1, 1, 1))
```

Inside the kernel, that one thread performs the two row loops:

```python
ss = cutlass.Float32(0.0)
for d in cutlass.range(d_model, unroll=1):
    x = mX[row, d].to(cutlass.Float32)
    ss = ss + x * x
```

Then it computes the reciprocal square root:

```python
inv_rms = cute.math.rsqrt(ss / d_model + eps, fastmath=True)
```

And writes the normalized row:

```python
for d in cutlass.range(d_model, unroll=1):
    x = mX[row, d].to(cutlass.Float32)
    w = mWeight[d].to(cutlass.Float32)
    mY[row, d] = x * inv_rms * w
```

## Why This Version Is Slow

Only one thread works on each row, so the row reduction is serial. That is good
for learning because the code looks like the math, but it leaves most of the
GPU idle.

The optimized path is:

```text
one thread per row
-> many threads per row
-> each thread sums part of the row
-> reduce partial sums in shared memory or warp primitives
-> all threads write normalized elements
```

The versioned files under `learning/kernels/rmsnorm/` start walking in that
direction.
