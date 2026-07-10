# 05 Softmax: Math To Python To CuTeDSL

Matching script: `../05_softmax_row_cutedsl.py`

## What This Component Does

Softmax converts logits into probabilities along a row. In Transformers, it is
used in attention and in classification/language-model output distributions.

Shapes:

```text
x: (M, N)
y: (M, N)
```

Each row is independent.

## Math

Direct softmax for row `m`:

$$
y_{m,j} = \frac{\exp(x_{m,j})}{\sum_{k=0}^{N-1}\exp(x_{m,k})}
$$

The numerically stable form subtracts the row maximum:

$$
a_m = \max_{k} x_{m,k}
$$

$$
y_{m,j} =
\frac{\exp(x_{m,j} - a_m)}
{\sum_{k=0}^{N-1}\exp(x_{m,k} - a_m)}
$$

Subtracting `a_m` does not change the result, but it prevents overflow when
logits are large.

## Python Translation

Plain loop version:

```python
for m in range(M):
    row_max = max(x[m, j] for j in range(N))

    denom = 0.0
    tmp = [0.0] * N
    for j in range(N):
        tmp[j] = math.exp(x[m, j] - row_max)
        denom += tmp[j]

    for j in range(N):
        y[m, j] = tmp[j] / denom
```

PyTorch:

```python
y = torch.softmax(x, dim=-1)
```

## CuTeDSL Translation

The beginner kernel launches one block per row and one thread per block:

```python
row, _, _ = cute.arch.block_idx()
```

The single active thread performs three passes over the row.

Pass 1 finds the max:

```python
row_max = cutlass.Float32(-3.4028234663852886e38)
for col in cutlass.range(cols, unroll=1):
    x = mX[row, col].to(cutlass.Float32)
    if x > row_max:
        row_max = x
```

Pass 2 computes exponentials and the denominator:

```python
denom = cutlass.Float32(0.0)
for col in cutlass.range(cols, unroll=1):
    e = cute.math.exp(mX[row, col].to(cutlass.Float32) - row_max, fastmath=True)
    denom = denom + e
    mY[row, col] = e
```

Pass 3 normalizes:

```python
for col in cutlass.range(cols, unroll=1):
    mY[row, col] = mY[row, col].to(cutlass.Float32) / denom
```

## Optimization Direction

The math naturally contains row reductions:

```text
max over columns
sum over columns
```

The beginner version does those reductions serially in one thread. Faster
versions use many threads per row:

```text
each thread loads some columns
-> reduce local maxima to row max
-> compute local exp sums
-> reduce sums to denominator
-> normalize columns in parallel
```
