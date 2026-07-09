import torch


def precompute_freqs_cis(
    head_dim: int,
    max_seq_len: int,
    theta: float = 1_000_000.0,
) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # q, k: (B, T, H, head_dim)
    def rotate(x: torch.Tensor) -> torch.Tensor:
        x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        # freqs_cis: (T, head_dim//2) → (1, T, 1, head_dim//2)
        f = freqs_cis.unsqueeze(0).unsqueeze(2)
        return torch.view_as_real(x_c * f).flatten(3).type_as(x)

    return rotate(q), rotate(k)
