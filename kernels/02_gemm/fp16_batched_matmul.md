# FP16 Batched Matrix Multiplication

## Problem

For each batch item, compute:

$$
C_b = A_b B_b
$$

## Symbols

$$
A \in \mathbb{R}^{B \times M \times K}
$$

$A$ is the batch of left matrices.

$$
B \in \mathbb{R}^{B \times K \times N}
$$

$B$ is the batch of right matrices.

$$
C \in \mathbb{R}^{B \times M \times N}
$$

$C$ is the output batch.

$$
b,m,n,k
$$

$b$ selects the batch, $m$ selects an output row, $n$ selects an output column,
and $k$ is the reduction dimension.

## Equation

$$
C_{b,m,n} = \sum_{k=0}^{K-1} A_{b,m,k}B_{b,k,n}
$$

## Derivation

Step 1: Matrix multiplication for one batch item is:

$$
C_b = A_bB_b
$$

Step 2: One scalar of $C_b$ is row $m$ of $A_b$ dotted with column $n$ of $B_b$.

$$
C_{b,m,n}
=
[A_{b,m,0}, \dots, A_{b,m,K-1}]
\cdot
[B_{b,0,n}, \dots, B_{b,K-1,n}]
$$

Step 3: Expand the dot product.

$$
C_{b,m,n}
=
A_{b,m,0}B_{b,0,n}
+ \cdots +
A_{b,m,K-1}B_{b,K-1,n}
$$

Step 4: Write as a summation.

$$
C_{b,m,n} = \sum_{k=0}^{K-1} A_{b,m,k}B_{b,k,n}
$$

## FP16 / FP32 Detail

Inputs and output are FP16:

$$
A,B,C \text{ stored as fp16}
$$

Accumulation should be FP32:

$$
\operatorname{acc}_{b,m,n} \in \mathbb{R}_{fp32}
$$

Then convert:

$$
C_{b,m,n} = \operatorname{fp16}(\operatorname{acc}_{b,m,n})
$$

## Visual

```text
For one batch b:

A row:    [a0 a1 a2]
B column: [b0 b1 b2]

C scalar = a0*b0 + a1*b1 + a2*b2
```

## Python

```python
C = torch.bmm(A.float(), B.float()).half()
```

## CuTeDSL Mapping

One thread owns:

```text
C[b, m, n]
```

The thread loops:

```python
acc = 0.0  # Float32
for k in range(K):
    acc += A[b, m, k] * B[b, k, n]
C[b, m, n] = acc
```

This is correct but not tensor-core optimized. The next optimization is tiled
GEMM using shared memory and MMA instructions.
