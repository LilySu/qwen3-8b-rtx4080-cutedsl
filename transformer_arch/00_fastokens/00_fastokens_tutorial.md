# fastokens — Technical Tutorial

*Position in inference pipeline: step 0 — text → token IDs, before any GPU work.*

Sources: [GitHub](https://github.com/crusoecloud/fastokens) ·
[Rust API](https://docs.rs/fastokens/latest/fastokens/struct.Tokenizer.html) ·
[vLLM integration](https://docs.vllm.ai/en/latest/api/vllm/tokenizers/fastokens/) ·
[NVIDIA Dynamo guide](https://docs.nvidia.com/dynamo/dev/user-guides/fastokens-tokenizer)

---

## What fastokens is

fastokens is a high-performance BPE tokenizer with a Rust backend and Python bindings. It is
a drop-in replacement for HuggingFace `tokenizers`, loading the same `tokenizer.json` files
with **10×+ faster throughput** — particularly at long prompts where the speedup compounds.

| Attribute | Value |
|-----------|-------|
| Backend | Rust (91.6% of codebase) |
| Python bindings | 8.4% |
| License | Apache-2.0 |
| Maintained by | Crusoe AI / Atero AI |
| Algorithm | Byte-Pair Encoding (BPE), byte-level |
| Format | HuggingFace `tokenizer.json` compatible |

The library targets inference workloads and explicitly does not support features not needed
there: additional encoding outputs, some normalizer/pretokenizer variants.

---

## Installation

```bash
# From PyPI (abi3 wheel — compatible with Python 3.9+)
pip install fastokens

# Or from source
git clone https://github.com/atero-ai/fast-tokens
uv pip install fast-tokens/python
```

The `cp39-abi3-manylinux_2_17_x86_64` wheel is compatible with Python 3.12 / glibc 2.41
(the container environment for this project).

---

## Architecture: the encoding pipeline

fastokens implements the full HuggingFace tokenizer pipeline in compiled Rust:

```
raw text
    │
    ▼
[Added tokens split]     — identify special tokens like <|im_start|> before any normalization
    │
    ▼
[Normalizer]             — Unicode normalization (ICU-based), lowercasing, etc.
    │
    ▼
[Pre-tokenizer]          — split text into chunks before BPE merges
    │
    ▼
[BPE Model]              — apply merge rules to byte sequences
    │
    ▼
[Post-processor]         — add BOS/EOS, wrap with special tokens
    │
    ▼
token IDs  []int
```

The key performance gain: HuggingFace's tokenizers library does the BPE merge loop in Rust
but calls back into Python for orchestration. fastokens compiles the entire pipeline — including
the pre-tokenizer regex, merge cache, and post-processor — into a single Rust binary with no
Python roundtrips during encoding.

---

## Pre-tokenizers (`fastokens::pre_tokenizers`)

Pre-tokenization splits raw text into chunks before BPE merges are applied. fastokens provides
two compiled pre-tokenizer types:

### `ByteLevel`

Applies GPT-2-style byte-to-unicode mapping before splitting. Every byte (0–255) maps to a
printable Unicode character, so the BPE vocabulary operates on printable characters only and
the model never sees raw bytes. This is what Qwen3 uses.

```
" Hello" → "ĠHello"   (space → Ġ, then BPE on the Unicode string)
```

### `Split`

Delimiter-based splitting. Controlled by `SplitBehavior`:
- `Removed` — delimiter is consumed
- `Isolated` — delimiter becomes its own token
- `MergedWithPrevious` / `MergedWithNext` — delimiter attaches to adjacent chunk

The pre-tokenizer is compiled once at tokenizer load time. The regex used by GPT-2/Qwen3
style tokenizers (matching words, punctuation, whitespace, numbers separately) is compiled
to a native PCRE2 pattern — this is the main source of pre-tokenization speedup vs Python
regex.

---

## Decoders (`fastokens::decoders`)

### `ByteLevelDecoder`

Reverses the GPT-2 byte-to-unicode mapping. When decoding token IDs back to text:

```
token strings → Unicode characters → reverse byte mapping → UTF-8 text
```

This must be applied carefully for streaming detokenization to avoid splitting multi-byte
UTF-8 sequences across decode calls. vLLM's fastokens integration rebinds
`tokenizers.decoders.DecodeStream` to the fastokens implementation to handle this correctly.

---

## Python API

### Loading a tokenizer

```python
from fastokens import Tokenizer

# From a local tokenizer.json file
tok = Tokenizer.from_file("weights/tokenizer.json")

# From HuggingFace Hub (downloads automatically)
tok = Tokenizer.from_model("Qwen/Qwen3-8B")

# With authentication token
tok = Tokenizer.from_model_with_token("Qwen/Qwen3-8B", token="hf_...")
```

### Encoding

```python
# Single string → list[int]
ids = tok.encode("Hello, world!")

# With special tokens explicitly included
ids = tok.encode_with_special_tokens("<|im_start|>user\nHello<|im_end|>")

# Batch encoding (parallelized in Rust via rayon)
batch_ids = tok.encode_batch(["Hello", "world", "foo"])
```

### Decoding

```python
# list[int] → str
text = tok.decode([9906, 11, 1917, 0])

# Batch decoding
texts = tok.decode_batch([[9906], [1917]])

# Decode token strings directly
text = tok.decode_tokens(["Hello", ",", "Ġworld"])
```

### Vocabulary lookup

```python
tok.vocab_size()            # 151936 for Qwen3
tok.token_to_id("<|eos|>")  # 151643
tok.id_to_token(151643)     # "<|eos|>"
tok.is_special_token(151644)  # True (<|im_start|>)
```

### Component introspection

```python
tok.normalizer()       # NFC, NFKC, etc.
tok.pre_tokenizer()    # ByteLevel or Split instance
tok.post_processor()   # adds BOS/EOS sequences
tok.decoder()          # ByteLevelDecoder
tok.model()            # the BPE merge table
tok.added_tokens()     # special tokens added on top of BPE
```

---

## Drop-in HuggingFace replacement

For codebases that load tokenizers via `transformers.AutoTokenizer`:

```python
import fastokens

# Patches every HF fast tokenizer loaded in this process
# Idempotent: safe to call multiple times
fastokens.patch_transformers()

# Now this uses fastokens under the hood
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
```

What the patch does internally:
1. Swaps the inner Rust tokenizer of every HF fast tokenizer with the fastokens shim
2. Rebinds `tokenizers.decoders.DecodeStream` to fastokens's streaming decoder

---

## vLLM integration

Enable via environment variable — no code changes:

```bash
VLLM_USE_FASTOKENS=1 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B
```

The vLLM `vllm.tokenizers.fastokens` module applies `fastokens.patch_transformers()` at
process startup. Compatible tokenizer modes: `hf`, `deepseek_v32`, `deepseek_v4`, and any
mode that loads a HuggingFace fast tokenizer.

---

## NVIDIA Dynamo integration

```bash
dynamo serve Qwen/Qwen3-8B --tokenizer fastokens
```

Dynamo's frontend supports fastokens as a named tokenizer backend. It selects the fastokens
encoder/decoder pipeline instead of the HuggingFace tokenizers path, reducing
time-to-first-token for long prompts where pre-tokenization regex is a bottleneck.

---

## Qwen3 special tokens

Qwen3 uses the ChatML format. Key token IDs:

| Token | ID | Role |
|-------|----|------|
| `<\|endoftext\|>` | 151643 | EOS / padding |
| `<\|im_start\|>` | 151644 | ChatML turn start |
| `<\|im_end\|>` | 151645 | ChatML turn end |

Vocabulary size: **151,936** (151,643 BPE + control tokens).

### ChatML format

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is RoPE?<|im_end|>
<|im_start|>assistant
```

This is what `tokenizer.py` (project root) produces via `apply_chat_template()`.

---

## How this project uses fastokens

`tokenizer.py` at the project root wraps fastokens with three methods:

```python
class Qwen3Tokenizer:
    def __init__(self, path: str):
        self._tok = Tokenizer.from_file(path)   # fastokens

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)

    def apply_chat_template(self, messages: list[dict]) -> list[int]:
        # builds ChatML string, encodes with special tokens
        ...
```

The `from_file()` call loads `weights/tokenizer.json` directly — no HuggingFace Hub
dependency at inference time, no `transformers` import, no Python tokenizer overhead.

---

## Performance characteristics

- **Throughput**: 10×+ vs HuggingFace `tokenizers` at long prompts
- **Parallelism**: `encode_batch()` uses Rayon for CPU-parallel batch encoding
- **Startup**: tokenizer load is fast (compiled regex, no Python parsing)
- **Memory**: the BPE merge table for Qwen3 (~151K tokens) fits in L3 cache on modern CPUs

The performance gain concentrates in the pre-tokenization regex step (PCRE2 vs Python `re`)
and in eliminating Python overhead from the merge loop. For single short strings the
difference is smaller; for batches of long prompts it is the dominant factor in
time-to-first-token before the model even runs.

---

## BPE algorithm overview

1. **Initialize**: every byte (0–255) is a token. Map bytes to printable Unicode via GPT-2 table.
2. **Pre-tokenize**: apply regex to split text into word/punctuation/number chunks. BPE merges
   never cross pre-tokenizer boundaries ("authority zones").
3. **Apply merges**: iterate over the sorted merge priority table. Find the highest-priority
   adjacent pair in each chunk and merge. Repeat until no mergeable pairs remain.
4. **Post-process**: prepend/append special tokens, add padding if needed.
5. **Output**: integer IDs in vocabulary order.

The merge table is loaded from `tokenizer.json` at startup and compiled into a hash map for
O(1) pair lookup. The main loop is `O(n log n)` in practice due to the priority queue.

---

## Thread safety

The `Tokenizer` struct implements `Send + Sync` — safe to share across threads and use in
async contexts. `encode_batch()` internally uses Rayon's thread pool.
