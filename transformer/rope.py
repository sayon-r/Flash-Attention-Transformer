import torch


def apply_rope(x: torch.Tensor, head_dim: int, start_pos: int = 0) -> torch.Tensor:
    """Apply Rotary Position Embeddings to x.

    Args:
        x:        (batch, heads, seq_len, head_dim)
        head_dim: dimension of each attention head
        start_pos: offset for the first token (used during cached generation)
    Returns:
        Tensor of same shape with RoPE applied to Q or K.
    """
    B, H, seq_len, d = x.shape
    assert d == head_dim

    # frequencies: theta_i = 1 / 10000^(2i/d)  for i in [0, d/2)
    half = head_dim // 2
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32) / head_dim)
    )                                                                  # (half,)

    positions = torch.arange(start_pos, start_pos + seq_len, device=x.device, dtype=torch.float32)
    # outer product → (seq_len, half)
    angles = torch.outer(positions, inv_freq)
    cos = angles.cos()[None, None, :, :]                               # (1,1,seq_len,half)
    sin = angles.sin()[None, None, :, :]

    x = x.float()
    x0, x1 = x[..., :half], x[..., half:]                             # split into two halves
    # 2D rotation: [x0, x1] → [x0·cos − x1·sin, x0·sin + x1·cos]
    rotated = torch.cat([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)
    return rotated.to(x.dtype)
