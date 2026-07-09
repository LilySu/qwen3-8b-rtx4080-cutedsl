import torch
import torch.nn as nn

from .attention import KVCache
from .block import Block
from .config import Qwen3Config
from .norm import RMSNorm
from .rope import precompute_freqs_cis


class Qwen3(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        freqs = precompute_freqs_cis(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        )
        self.register_buffer("freqs_cis", freqs, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: list[KVCache] | None = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, list[KVCache]]:
        B, T = input_ids.shape
        x = self.embed_tokens(input_ids)

        freqs_cis = self.freqs_cis[start_pos : start_pos + T]

        # For single-token decode steps, is_causal inside sdpa handles masking.
        # For prefill (T > 1) with a kv_cache offset we need an explicit mask.
        mask = None
        if T > 1 and kv_caches is not None:
            past_len = kv_caches[0][0].shape[1] if kv_caches[0] is not None else 0
            full_len = past_len + T
            mask = torch.full((T, full_len), float("-inf"), device=x.device, dtype=x.dtype)
            mask = torch.triu(mask, diagonal=past_len + 1)
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, full_len)

        new_kv_caches: list[KVCache] = []
        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches else None
            x, new_kv = layer(x, freqs_cis, mask, cache)
            new_kv_caches.append(new_kv)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, new_kv_caches

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_p: float = 0.9,
        eos_token_id: int = 151643,
    ) -> torch.Tensor:
        device = input_ids.device
        generated = input_ids.clone()
        kv_caches: list[KVCache] | None = None

        # Prefill
        logits, kv_caches = self.forward(generated, kv_caches=None, start_pos=0)
        start_pos = generated.shape[1]

        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :]  # (B, vocab)

            if temperature != 1.0:
                next_logits = next_logits / temperature

            if top_p < 1.0:
                next_logits = _top_p_filter(next_logits, top_p)

            next_token = torch.multinomial(torch.softmax(next_logits, dim=-1), num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            if (next_token == eos_token_id).all():
                break

            logits, kv_caches = self.forward(next_token, kv_caches=kv_caches, start_pos=start_pos)
            start_pos += 1

        return generated


def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cumprobs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens whose cumulative prob exceeds top_p (shift right by 1 to keep the boundary token)
    remove = (cumprobs - torch.softmax(sorted_logits, dim=-1)) > top_p
    sorted_logits[remove] = float("-inf")
    return logits.scatter(-1, sorted_idx, sorted_logits)
