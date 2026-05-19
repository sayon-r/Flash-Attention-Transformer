import math
import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Optional


class FlashAttentionFunction(Function):
    @staticmethod
    def forward(
        ctx,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        block_size: int,
        causal: bool,
    ) -> torch.Tensor:
        B, H, N, d = Q.shape
        scale = 1.0 / math.sqrt(d)
        Br = Bc = block_size
        Tr = math.ceil(N / Br)
        Tc = math.ceil(N / Bc)

        O = torch.zeros_like(Q)
        # L[b,h,i] = log-sum-exp for query position i; saved for backward recomputation
        L = torch.empty(B, H, N, device=Q.device, dtype=Q.dtype)

        for i in range(Tr):
            qs = i * Br
            qe = min(qs + Br, N)
            Qi = Q[:, :, qs:qe, :]                                   # (B,H,Br,d)

            Oi = torch.zeros(B, H, qe - qs, d, device=Q.device, dtype=Q.dtype)
            mi = torch.full((B, H, qe - qs), float("-inf"), device=Q.device, dtype=Q.dtype)
            li = torch.zeros(B, H, qe - qs, device=Q.device, dtype=Q.dtype)

            for j in range(Tc):
                kvs = j * Bc
                kve = min(kvs + Bc, N)
                Kj = K[:, :, kvs:kve, :]                             # (B,H,Bc,d)
                Vj = V[:, :, kvs:kve, :]

                Sij = scale * torch.matmul(Qi, Kj.transpose(-2, -1)) # (B,H,Br,Bc)

                if causal:
                    q_idx = torch.arange(qs, qe, device=Q.device).unsqueeze(1)
                    k_idx = torch.arange(kvs, kve, device=Q.device).unsqueeze(0)
                    Sij = Sij.masked_fill(k_idx > q_idx, float("-inf"))

                mij = Sij.amax(dim=-1)                                # (B,H,Br)
                mi_new = torch.maximum(mi, mij)

                # rescaling factors to merge old and new partial softmax
                alpha = torch.exp(mi - mi_new)                        # (B,H,Br)
                beta  = torch.exp(mij - mi_new)

                Pij = torch.exp(Sij - mi_new.unsqueeze(-1))           # (B,H,Br,Bc)
                lij = Pij.sum(dim=-1)                                  # (B,H,Br)

                li = alpha * li + beta * lij
                Oi = alpha.unsqueeze(-1) * Oi + beta.unsqueeze(-1) * torch.matmul(Pij, Vj)
                mi = mi_new

            Oi = Oi / li.unsqueeze(-1)
            O[:, :, qs:qe, :] = Oi
            L[:, :, qs:qe] = mi + torch.log(li)                      # logsumexp

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.block_size = block_size
        ctx.causal = causal
        ctx.scale = scale
        return O

    @staticmethod
    def backward(ctx, dO: torch.Tensor):
        Q, K, V, O, L = ctx.saved_tensors
        block_size = ctx.block_size
        causal = ctx.causal
        scale = ctx.scale

        B, H, N, d = Q.shape
        Br = Bc = block_size
        Tr = math.ceil(N / Br)
        Tc = math.ceil(N / Bc)

        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        # D[b,h,i] = rowsum(dO * O); appears in softmax gradient identity
        D = (dO * O).sum(dim=-1)                                      # (B,H,N)

        for i in range(Tr):
            qs = i * Br
            qe = min(qs + Br, N)
            Qi  = Q[:, :, qs:qe, :]
            dOi = dO[:, :, qs:qe, :]
            Di  = D[:, :, qs:qe]                                      # (B,H,Br)
            Li  = L[:, :, qs:qe]                                      # (B,H,Br)

            dQi = torch.zeros_like(Qi)

            for j in range(Tc):
                kvs = j * Bc
                kve = min(kvs + Bc, N)
                Kj = K[:, :, kvs:kve, :]
                Vj = V[:, :, kvs:kve, :]

                # recompute attention weights from saved Q,K — no stored N×N matrix
                Sij = scale * torch.matmul(Qi, Kj.transpose(-2, -1))

                if causal:
                    q_idx = torch.arange(qs, qe, device=Q.device).unsqueeze(1)
                    k_idx = torch.arange(kvs, kve, device=Q.device).unsqueeze(0)
                    Sij = Sij.masked_fill(k_idx > q_idx, float("-inf"))

                Pij = torch.exp(Sij - Li.unsqueeze(-1))               # (B,H,Br,Bc)

                dV[:, :, kvs:kve, :] += torch.matmul(Pij.transpose(-2, -1), dOi)

                dPij = torch.matmul(dOi, Vj.transpose(-2, -1))        # (B,H,Br,Bc)
                # softmax Jacobian: dS = P * (dP - D) scaled by scale
                dSij = scale * Pij * (dPij - Di.unsqueeze(-1))

                dQi                    += torch.matmul(dSij, Kj)
                dK[:, :, kvs:kve, :] += torch.matmul(dSij.transpose(-2, -1), Qi)

            dQ[:, :, qs:qe, :] = dQi

        return dQ, dK, dV, None, None


class FlashAttention(nn.Module):
    """Tiled Flash Attention — forward and backward without materializing the N×N score matrix."""

    def __init__(self, block_size: int = 64, causal: bool = False):
        super().__init__()
        self.block_size = block_size
        self.causal = causal

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Q, K, V: (batch, heads, seq_len, head_dim)
        Returns:
            (batch, heads, seq_len, head_dim)
        """
        return FlashAttentionFunction.apply(Q, K, V, self.block_size, self.causal)
