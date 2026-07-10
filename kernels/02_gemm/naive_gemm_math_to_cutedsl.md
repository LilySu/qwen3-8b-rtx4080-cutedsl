# LeetGPU GEMM: Math To PyTorch/JAX To CuTeDSL

Matching implementations:

- `naive_gemm_pytorch.py`
- `naive_gemm_jax.py`

This is one combined teacher-style note for the PyTorch and JAX versions.
It also maps the screenshots you pasted to the underlying math patterns.

The two Python files implement the FP32 GEMM core with CuTe DSL / CUTLASS
`SGemm`. They use PyTorch or JAX only as the owner of GPU memory and as the
reference checker.

## GPU And Container Context

The implementation was checked in container `bcc425628202` on:

```text
GPU: NVIDIA GeForce RTX 4080 Laptop GPU
Driver: 581.83
CUDA reported by nvidia-smi: 13.0
PyTorch: 2.11.0+cu130
JAX: 0.10.2 with CUDA 13 backend
```

The successful commands were:

```bash
cd /workspace/cutelearning/learning/leetgpu
python naive_gemm_pytorch.py --mnk 512,512,512
python naive_gemm_jax.py --mnk 512,512,512
```

## What Is Actually Implemented

The implemented CuTe kernel is GEMM:

$$
C = A B
$$

Elementwise:

$$
C_{m,o} = \sum_{r=0}^{R-1} A_{m,r} B_{r,o}
$$

This directly covers the matrix multiplication pattern that appears in:

- FP32 matrix multiplication.
- The two matrix multiplies inside attention: $QK^T$ and $PV$.
- The non-batched form of batched matrix multiplication.
- Dense matrix-vector multiplication when the output has one vector column.

It does not claim to implement the separate LeetGPU kernels for matrix copy,
sigmoid, vector softmax, or sparse storage. Those are related equations, and
they are explained below, but the runnable CuTe files here are GEMM-focused.

## Screenshot Map

| Screenshot topic | Formula | Relationship to this implementation |
|---|---|---|
| Matrix Copy | $B_{i,j}=A_{i,j}$ | Elementwise copy, no reduction. Not GEMM. |
| Sigmoid Activation | $\sigma(x)=\frac{1}{1+\exp(-x)}$ | Elementwise nonlinear function. Not GEMM. |
| Softmax | $\sigma(x)_i=\frac{\exp(x_i)}{\sum_j\exp(x_j)}$ | Reduction plus normalization. Used inside attention, but not GEMM. |
| Softmax Attention | $\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^T}{\sqrt d}\right)V$ | Contains two GEMMs. This file implements the GEMM pattern used by $QK^T$ and by multiplying probabilities with $V$. |
| FP16 Batched Matmul | $C_b=A_bB_b$ | Same GEMM equation repeated over batch index $b$. This file implements the single-batch FP32 version. |
| Sparse Matrix-Vector | $y_i=\sum_j A_{i,j}x_j$ | Same dot-product pattern as GEMM, but with sparse data handling. This file implements dense GEMM, not sparse compression. |

## Core GEMM Symbols

Shapes:

```text
A: (M, R)
B: (R, O)
C: (M, O)
```

| Symbol | Meaning | Transformer-style interpretation |
|---|---|---|
| $A$ | Left matrix | Hidden states, queries, or activations |
| $B$ | Right matrix | Weights, keys/values, or another activation matrix |
| $C$ | Output matrix | Projected activations, attention scores, or attention output |
| $M$ | Number of rows in $A$ and $C$ | Usually tokens, rows, or batch-times-sequence |
| $R$ | Reduction dimension | Shared feature width; often called $K$ in GEMM |
| $O$ | Number of columns in $B$ and $C$ | Output width or number of output channels |
| $m$ | Row index | Which token/output row we are computing |
| $r$ | Reduction index | Which feature position is being multiplied |
| $o$ | Output column index | Which output channel/column we are computing |
| $A_{m,r}$ | One scalar from row $m$ of $A$ | One feature of one token |
| $B_{r,o}$ | One scalar from column $o$ of $B$ | One matching weight/key/value component |
| $C_{m,o}$ | One output scalar | One produced output feature |

Index ranges:

$$
0 \le m < M
$$

$$
0 \le r < R
$$

$$
0 \le o < O
$$

Teacher wording:

```text
To compute one output number, choose one row from A and one column from B.
Multiply matching entries, then add all those products.
```

## Deriving GEMM One Scalar At A Time

Start with one output location:

$$
C_{m,o}
$$

Say it verbally:

```text
This is the output at row m and output column o.
```

That output sees row $m$ of $A$:

$$
\left[
A_{m,0},
A_{m,1},
A_{m,2},
\ldots,
A_{m,R-1}
\right]
$$

It also sees column $o$ of $B$:

$$
\left[
\begin{array}{c}
B_{0,o} \\
B_{1,o} \\
B_{2,o} \\
\vdots \\
B_{R-1,o}
\end{array}
\right]
$$

Now multiply matching positions:

$$
A_{m,0}B_{0,o},\quad
A_{m,1}B_{1,o},\quad
A_{m,2}B_{2,o},\quad
\ldots,\quad
A_{m,R-1}B_{R-1,o}
$$

Then add them:

$$
C_{m,o}
= A_{m,0}B_{0,o}
+ A_{m,1}B_{1,o}
+ A_{m,2}B_{2,o}
+ \cdots
+ A_{m,R-1}B_{R-1,o}
$$

The summation symbol is just shorthand for that repeated pattern:

$$
C_{m,o} = \sum_{r=0}^{R-1} A_{m,r}B_{r,o}
$$

That is the proof of the scalar GEMM formula from the row-dot-column rule.

## Python Loop Form

The math:

$$
C_{m,o} = \sum_{r=0}^{R-1} A_{m,r}B_{r,o}
$$

becomes:

```python
for m in range(M):
    for o in range(O):
        acc = 0.0
        for r in range(R):
            acc += A[m, r] * B[r, o]
        C[m, o] = acc
```

The loop meaning is:

- `m` chooses the output row.
- `o` chooses the output column.
- `r` walks across the shared feature dimension.
- `acc` is the running sum from the equation.

## PyTorch Implementation Mapping

The PyTorch file allocates the inputs:

```python
A = torch.randn((M, R), device="cuda", dtype=torch.float32)
B = torch.randn((R, O), device="cuda", dtype=torch.float32)
C = torch.empty((M, O), device="cuda", dtype=torch.float32)
```

Math meaning:

$$
A \in \mathbb{R}^{M \times R}
$$

$$
B \in \mathbb{R}^{R \times O}
$$

$$
C \in \mathbb{R}^{M \times O}
$$

Teacher wording:

```text
PyTorch owns three GPU allocations. A and B contain input values. C is a
blank output buffer that the CuTe kernel will fill.
```

Then it calls:

```python
SGemm()(
    from_dlpack(A, assumed_align=16),
    from_dlpack(B.T, assumed_align=16),
    from_dlpack(C, assumed_align=16),
)
```

`from_dlpack` means:

```text
Do not copy the tensor. Give CuTe a view of this same GPU memory.
```

## JAX Implementation Mapping

The JAX file allocates:

```python
A = jax.random.normal(key_a, (M, R), dtype=jnp.float32)
B = jax.random.normal(key_b, (R, O), dtype=jnp.float32)
C = jnp.zeros((M, O), dtype=jnp.float32)
```

Then it places them on the GPU:

```python
A = jax.device_put(A, device)
B = jax.device_put(B, device)
C = jax.device_put(C, device)
```

Teacher wording:

```text
JAX owns the arrays, but DLPack lets CuTe see the underlying GPU buffers.
This is useful for learning interop. For production JAX, a custom call/FFI
path is usually the cleaner way to represent external GPU effects.
```

The JAX version also disables JAX GPU preallocation before importing JAX:

```python
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
```

That matters because the container has about 12 GiB of GPU memory. If JAX
preallocates most of it, CuTe/CUTLASS may have less room to compile and run.

## Why The Code Passes `B.T`

The mathematical right-hand matrix has shape:

```text
B: (R, O)
```

and the equation reads:

$$
C_{m,o} = \sum_{r=0}^{R-1} A_{m,r}B_{r,o}
$$

The bundled CUTLASS `SGemm` example expects its B operand in the shape:

```text
B_for_sgemm: (O, R)
```

So the Python code passes:

```python
B.T
```

The transpose identity is:

$$
B^T_{o,r} = B_{r,o}
$$

Substitute that into the GEMM formula:

$$
C_{m,o}
= \sum_{r=0}^{R-1} A_{m,r}B^T_{o,r}
$$

Because $B^T_{o,r}$ is the same scalar as $B_{r,o}$:

$$
C_{m,o}
= \sum_{r=0}^{R-1} A_{m,r}B_{r,o}
$$

So passing `B.T` changes the layout convention for the CUTLASS example, but it
does not change the math.

## Screenshot: Matrix Copy

The screenshot formula is:

$$
B_{i,j}=A_{i,j}
$$

Symbols:

| Symbol | Meaning |
|---|---|
| $A$ | Input matrix |
| $B$ | Output matrix |
| $i$ | Row index |
| $j$ | Column index |

Teacher wording:

```text
For every cell, copy the input value at the same row and column into the
output. There is no multiplication and no reduction.
```

This is not GEMM. It is the simpler elementwise pattern:

$$
\text{one output element depends on one input element}
$$

GEMM is different:

$$
\text{one output element depends on } R \text{ pairs of input elements}
$$

## Screenshot: Sigmoid Activation

The screenshot formula is:

$$
\sigma(x)=\frac{1}{1+\exp(-x)}
$$

For a vector:

$$
Y_i=\sigma(X_i)
$$

Symbols:

| Symbol | Meaning |
|---|---|
| $X_i$ | Input scalar at index $i$ |
| $Y_i$ | Output scalar at index $i$ |
| $\exp$ | Exponential function |
| $\sigma$ | Sigmoid function |

Teacher wording:

```text
Take one input number. Negate it, exponentiate it, add one, and take the
reciprocal. The result is squeezed into the range from 0 to 1.
```

Transformer connection:

```text
Modern transformer MLPs more often use SiLU/GELU-style activations, but sigmoid
is the base gate shape behind many gating ideas.
```

This is elementwise, not GEMM.

## Screenshot: Softmax

The screenshot formula is:

$$
\operatorname{softmax}(x)_i
=
\frac{\exp(x_i)}
{\sum_{j=0}^{n-1}\exp(x_j)}
$$

Numerically stable softmax subtracts the maximum:

$$
\operatorname{softmax}(x)_i
=
\frac{\exp(x_i-a)}
{\sum_{j=0}^{n-1}\exp(x_j-a)}
$$

where:

$$
a=\max_{0 \le j < n}x_j
$$

Symbols:

| Symbol | Meaning |
|---|---|
| $x$ | Input vector |
| $n$ | Vector length |
| $i$ | Output index |
| $j$ | Reduction index over all vector entries |
| $a$ | Maximum input value |

Proof that subtracting $a$ does not change softmax:

$$
\frac{\exp(x_i-a)}
{\sum_j \exp(x_j-a)}
=
\frac{\exp(x_i)\exp(-a)}
{\sum_j \exp(x_j)\exp(-a)}
$$

Factor $\exp(-a)$ out of the denominator:

$$
=
\frac{\exp(x_i)\exp(-a)}
{\exp(-a)\sum_j \exp(x_j)}
$$

Cancel the same nonzero factor:

$$
=
\frac{\exp(x_i)}
{\sum_j \exp(x_j)}
$$

Teacher wording:

```text
Softmax turns a row of scores into probabilities. Subtracting the max keeps the
largest exponent near exp(0), which avoids overflow, and the probabilities are
unchanged because the same scale factor cancels out.
```

This is not GEMM, but it sits between two GEMMs in attention.

## Screenshot: Softmax Attention

The screenshot formula is:

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt d}
\right)V
$$

Shapes from the screenshot:

```text
Q: (M, d)
K: (N, d)
V: (N, d)
output: (M, d)
```

Symbols:

| Symbol | Meaning | Transformer wording |
|---|---|---|
| $Q$ | Query matrix | What each token is looking for |
| $K$ | Key matrix | What each token offers to be matched against |
| $V$ | Value matrix | Information to mix after attention weights are known |
| $M$ | Number of query rows | Number of output/query tokens |
| $N$ | Number of key/value rows | Number of source tokens |
| $d$ | Feature width | Head dimension |

First GEMM:

$$
S = QK^T
$$

Elementwise:

$$
S_{m,n} = \sum_{r=0}^{d-1} Q_{m,r}K_{n,r}
$$

Teacher wording:

```text
For one query token m and one key token n, take their dot product. A large dot
product means the query and key point in similar directions.
```

Scale:

$$
\tilde{S}_{m,n}=\frac{S_{m,n}}{\sqrt d}
$$

Teacher wording:

```text
Divide by square root of the feature width so the dot products do not grow too
large as d increases.
```

Row-wise softmax:

$$
P_{m,n}
=
\frac{\exp(\tilde{S}_{m,n})}
{\sum_{u=0}^{N-1}\exp(\tilde{S}_{m,u})}
$$

Teacher wording:

```text
For each query row, softmax turns all key scores into weights that sum to one.
```

Second GEMM:

$$
O = PV
$$

Elementwise:

$$
O_{m,r}
=
\sum_{n=0}^{N-1} P_{m,n}V_{n,r}
$$

Teacher wording:

```text
The output token is a weighted average of value vectors. The attention weights
choose how much information to take from each value row.
```

The implemented CuTe SGemm files directly teach the two GEMM pieces:

$$
QK^T
$$

and:

$$
PV
$$

They do not implement the softmax middle step.

## Screenshot: FP16 Batched Matrix Multiplication

The screenshot formula is:

$$
C_b = A_bB_b
$$

Elementwise:

$$
C_{b,m,n}
=
\sum_{k=0}^{K-1} A_{b,m,k}B_{b,k,n}
$$

Symbols:

| Symbol | Meaning |
|---|---|
| $b$ | Batch index |
| $M$ | Rows per matrix |
| $K$ | Reduction dimension |
| $N$ | Output columns |
| $A_b$ | Batch item $b$ of matrix $A$ |
| $B_b$ | Batch item $b$ of matrix $B$ |
| $C_b$ | Batch item $b$ of output matrix $C$ |

Teacher wording:

```text
This is the same GEMM equation, but repeated independently for each batch
index. Batch b does not mix with batch b+1.
```

The files here implement one FP32 GEMM:

$$
C = AB
$$

The batched form adds one outer batch index:

$$
C_b=A_bB_b
$$

## Screenshot: Sparse Matrix-Vector Multiplication

The screenshot formula is:

$$
y_i = \sum_{j=0}^{N-1} A_{i,j}x_j
$$

Symbols:

| Symbol | Meaning |
|---|---|
| $A$ | Sparse matrix |
| $x$ | Dense input vector |
| $y$ | Dense output vector |
| $i$ | Row index |
| $j$ | Column/reduction index |

Teacher wording:

```text
For one output row, multiply each matrix entry in that row by the matching
vector entry, then add the products.
```

This is GEMM's dot-product idea with only one output vector column:

$$
y_i = \sum_j A_{i,j}x_j
$$

compared with:

$$
C_{m,o} = \sum_r A_{m,r}B_{r,o}
$$

The extra sparse challenge is storage:

```text
skip zero entries instead of reading and multiplying every A[i,j]
```

The files here implement dense GEMM, not sparse compressed storage.

## Why This Matters For Transformers

Most transformer blocks are built from these patterns:

1. Linear layers are GEMMs.
2. Attention score computation is a GEMM: $QK^T$.
3. Attention output computation is a GEMM: $PV$.
4. Softmax normalizes attention scores row by row.
5. Activations like sigmoid, SiLU, and GELU are elementwise functions.

The two runnable files focus on the core GEMM because it is the largest and
most repeated building block.

## Final Mental Model

Say the GEMM equation like this:

$$
C_{m,o} = \sum_{r=0}^{R-1} A_{m,r}B_{r,o}
$$

Teacher wording:

```text
For output row m and output column o, walk across the shared dimension r.
At each r, multiply the value from A's row by the matching value from B's
column. Add all those products. Store the final sum in C[m,o].
```

PyTorch version:

```text
PyTorch owns A, B, C -> DLPack exposes them -> CuTe SGemm writes C
```

JAX version:

```text
JAX owns A, B, C -> DLPack exposes them -> CuTe SGemm writes C
```

The math is identical. The host framework only changes who allocates the GPU
memory and who computes the reference answer.
