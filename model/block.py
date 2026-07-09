import torch
import torch.nn as nn

from .attention import Attention, KVCache
from .config import Qwen3Config
from .mlp import MLP
from .norm import RMSNorm


class Block(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.ln1  = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = Attention(config)
        self.ln2  = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp  = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache]:
        attn_out, new_kv = self.attn(self.ln1(x), freqs_cis, mask, kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_kv
