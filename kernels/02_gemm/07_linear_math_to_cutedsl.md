# 07 Linear Layer: Math To Python To CuTeDSL

Matching script: `../07_linear_naive_cutedsl.py`

## What This Component Does

A Transformer linear layer applies a learned matrix to every row of hidden
states. In CS336/PyTorch convention, the weight has shape `(d_out, d_in)`.

Shapes:

```text
x:      (M, K)
weight: (N, K)
y:      (M, N)
```

Where:

- `M` is number of rows, often `batch * sequence`.
- `K` is input width.
- `N` is output width.

## Math

For each output row `m` and output channel `n`:

$$
y_{m,n} = \sum_{k=0}^{K-1} x_{m,k} W_{n,k}
$$

This is matrix multiplication:

$$
Y = X W^T
$$

The transpose appears because `W` is stored as `(N, K)`.

## Python Translation

Plain loops:

```python
for m in range(M):
    for n in range(N):
        acc = 0.0
        for k in range(K):
            acc += x[m, k] * weight[n, k]
        y[m, n] = acc
```

PyTorch:

```python
y = x @ weight.T
```

## CuTeDSL Translation

The beginner CuTeDSL kernel assigns one GPU thread to one output scalar:

```text
one thread -> y[row, col]
```

The output matrix is flattened:

```python
i = bidx * THREADS_PER_CTA + tidx
```

Then converted back to `(row, col)`:

```python
n = cute.size(mY, mode=[1])
row = i // n
col = i % n
```

The thread computes the full dot product:

```python
acc = cutlass.Float32(0.0)
for k in cutlass.range(k_dim, unroll=1):
    acc = acc + mX[row, k].to(cutlass.Float32) * mWeight[col, k].to(cutlass.Float32)
mY[row, col] = acc
```

## Why This Is The Right Learning Version

This is the clearest possible GPU matmul:

```text
one output element = one dot product = one thread
```

It is not fast because:

- It rereads the same `x` and `weight` values many times.
- It does not use shared memory.
- It does not use Tensor Cores.
- It has one thread doing a whole `K` loop serially.

The optimized path is tiled GEMM:

```text
CTA owns tile of Y
-> load tile of X and W into shared memory
-> many threads cooperate
-> Tensor Cores compute MMA fragments
-> store tile of Y
```

The 500-line fp16 GEMM tutorial files are sophisticated versions of this same
math.
