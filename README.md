# Flash Attention Implementation

A from-scratch PyTorch implementation of Flash Attention with a complete transformer stack built on top of it.

Flash Attention avoids materializing the full N×N attention score matrix by processing queries and keys in tiles, computing a numerically stable online softmax as it goes. This keeps peak memory usage O(N) rather than O(N²) while producing outputs identical to standard attention.

## What's in here

| File | Description |
|---|---|
| `transformer/flash_attention.py` | Core implementation — tiled forward + backward via `torch.autograd.Function` |
| `transformer/attention.py` | Multi-head self-attention with RoPE, KV cache support |
| `transformer/rope.py` | Rotary Position Embeddings |
| `transformer/block.py` | Transformer block |
| `transformer/ffn.py` | Feed-forward network |
| `transformer/model.py` | Full transformer model |
| `transformer/tokenizer.py` | Tokenizer |
| `compare.py` | Correctness checks: output and gradient matching against standard attention |

## How Flash Attention works here

The forward pass tiles the Q, K, V matrices into blocks of size `block_size`. For each query tile it iterates over all key/value tiles, accumulating the output using an online softmax update:

```
mi_new = max(mi_old, max(Sij))
li     = exp(mi_old - mi_new) * li_old + exp(mij - mi_new) * sum(exp(Sij - mi_new))
Oi     = exp(mi_old - mi_new) * Oi_old + exp(mij - mi_new) * Pij @ Vj
```

After all key/value tiles are processed, `Oi` is normalized by `li`. The log-sum-exp `L = mi + log(li)` is saved for the backward pass instead of the full attention matrix.

The backward pass recomputes attention weights from the saved Q, K on the fly — no N×N matrix is ever stored.

## Correctness

```bash
python compare.py
```

This runs forward output and gradient checks against standard scaled dot-product attention across multiple sequence lengths and block sizes, including non-power-of-two lengths and causal masking. It also runs `torch.autograd.gradcheck` on float64 inputs.

Expected output:

```
=== output + gradient correctness ===
[PASS] causal=False, N=64,  block=16  |  out=True  dQ=True  dK=True  dV=True
[PASS] causal=True,  N=64,  block=16  |  out=True  dQ=True  dK=True  dV=True
[PASS] causal=False, N=128, block=32  |  out=True  dQ=True  dK=True  dV=True
[PASS] causal=True,  N=128, block=32  |  out=True  dQ=True  dK=True  dV=True
[PASS] causal=False, N=50,  block=16  |  out=True  dQ=True  dK=True  dV=True
[PASS] causal=True,  N=50,  block=16  |  out=True  dQ=True  dK=True  dV=True

=== autograd gradcheck (float64) ===
[PASS] gradcheck causal=False
[PASS] gradcheck causal=True
```

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.8+ and PyTorch 2.0+.

## Usage

```python
from transformer.flash_attention import FlashAttention
import torch

fa = FlashAttention(block_size=64, causal=True)

B, H, N, d = 2, 8, 512, 64
Q = torch.randn(B, H, N, d)
K = torch.randn(B, H, N, d)
V = torch.randn(B, H, N, d)

out = fa(Q, K, V)  # (B, H, N, d)
```

The multi-head attention layer wraps this with projections and RoPE:

```python
from transformer.attention import MultiHeadAttention

attn = MultiHeadAttention(d_model=512, num_heads=8)
x = torch.randn(2, 128, 512)  # (batch, seq_len, d_model)
out = attn(x, use_kv_cache=False)
```

## References

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Dao et al., 2022
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Dao, 2023
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — Su et al., 2021
