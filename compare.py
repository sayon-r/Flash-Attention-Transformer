"""
compare.py — verify FlashAttention output and gradients match standard attention.
"""

import math
import torch
import torch.nn.functional as F
from transformer.flash_attention import FlashAttention


def standard_attention(Q, K, V, causal=False):
    """Reference scaled dot-product attention (materializes full score matrix)."""
    scale = 1.0 / math.sqrt(Q.size(-1))
    scores = scale * torch.matmul(Q, K.transpose(-2, -1))             # (B,H,N,N)
    if causal:
        N = Q.size(-2)
        mask = torch.triu(torch.ones(N, N, device=Q.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, V)


def check_outputs(B=1, H=4, N=64, d=32, block_size=16, causal=False, dtype=torch.float32):
    torch.manual_seed(0)
    Q = torch.randn(B, H, N, d, dtype=dtype, requires_grad=True)
    K = torch.randn(B, H, N, d, dtype=dtype, requires_grad=True)
    V = torch.randn(B, H, N, d, dtype=dtype, requires_grad=True)

    # standard attention
    Q_std = Q.detach().requires_grad_(True)
    K_std = K.detach().requires_grad_(True)
    V_std = V.detach().requires_grad_(True)
    out_std = standard_attention(Q_std, K_std, V_std, causal=causal)
    loss_std = out_std.sum()
    loss_std.backward()

    # flash attention
    Q_fa = Q.detach().requires_grad_(True)
    K_fa = K.detach().requires_grad_(True)
    V_fa = V.detach().requires_grad_(True)
    fa = FlashAttention(block_size=block_size, causal=causal)
    out_fa = fa(Q_fa, K_fa, V_fa)
    loss_fa = out_fa.sum()
    loss_fa.backward()

    atol = 1e-4
    label = f"causal={causal}, N={N}, block={block_size}"

    out_match  = torch.allclose(out_std, out_fa,    atol=atol)
    dq_match   = torch.allclose(Q_std.grad, Q_fa.grad, atol=atol)
    dk_match   = torch.allclose(K_std.grad, K_fa.grad, atol=atol)
    dv_match   = torch.allclose(V_std.grad, V_fa.grad, atol=atol)

    ok = out_match and dq_match and dk_match and dv_match
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {label}  |  out={out_match}  dQ={dq_match}  dK={dk_match}  dV={dv_match}")

    if not out_match:
        diff = (out_std - out_fa).abs()
        print(f"       max output diff: {diff.max().item():.2e}")
    return ok


def run_gradcheck(B=1, H=2, N=16, d=8, block_size=8, causal=False):
    """torch.autograd.gradcheck on double-precision inputs."""
    torch.manual_seed(1)
    fa = FlashAttention(block_size=block_size, causal=causal)
    Q = torch.randn(B, H, N, d, dtype=torch.float64, requires_grad=True)
    K = torch.randn(B, H, N, d, dtype=torch.float64, requires_grad=True)
    V = torch.randn(B, H, N, d, dtype=torch.float64, requires_grad=True)
    ok = torch.autograd.gradcheck(fa, (Q, K, V), eps=1e-4, atol=1e-3, rtol=1e-3)
    label = f"gradcheck causal={causal}"
    print(f"[{'PASS' if ok else 'FAIL'}] {label}")
    return ok


if __name__ == "__main__":
    all_ok = True

    print("=== output + gradient correctness ===")
    all_ok &= check_outputs(N=64,  block_size=16, causal=False)
    all_ok &= check_outputs(N=64,  block_size=16, causal=True)
    all_ok &= check_outputs(N=128, block_size=32, causal=False)
    all_ok &= check_outputs(N=128, block_size=32, causal=True)
    # N not a multiple of block_size
    all_ok &= check_outputs(N=50,  block_size=16, causal=False)
    all_ok &= check_outputs(N=50,  block_size=16, causal=True)

    print("\n=== autograd gradcheck (float64) ===")
    all_ok &= run_gradcheck(causal=False)
    all_ok &= run_gradcheck(causal=True)

    print(f"\n{'All checks passed.' if all_ok else 'Some checks FAILED.'}")
