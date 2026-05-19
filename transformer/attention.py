
#Multi Head attention with Rotary Position Embeddings


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .rope import apply_rope

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with RoPE."""
    
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # linear projection for q, k, v values
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        
        # cache during generation
        self.k_cache = None
        self.v_cache = None
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_kv_cache: bool = False,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """Ffrward pass with optional caching"""
        batch_size, seq_len, _ = x.shape
        
        # project queryies, keys, value
        Q = self.Wq(x)  # (batch_size, seq_len, dmodel) -- same throughout
        K = self.Wk(x)   
        V = self.Wv(x)  
        
        if use_kv_cache and self.k_cache is not None and self.v_cache is not None:
            # add new kv cache 
            K = torch.cat([self.k_cache, K], dim=1)
            V = torch.cat([self.v_cache, V], dim=1)
            
        # Storing KV cache
        if use_kv_cache:
            self.k_cache = K
            self.v_cache = V
        
        # reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  
        
        # (batch_size, num_heads, seq_len, head_dim)

        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE to Q and K
        Q = apply_rope(Q, self.head_dim, start_pos)
        K = apply_rope(K, self.head_dim, start_pos)
        
        # scaled dot-product attention
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask if given:
        if mask is not None:
            # get current seq length from scores tensor
            seq_len = scores.size(-1)
            
            # Create a causal mask if needed
            if mask.dim() == 4:  # (batch_size, 1, seq_len, seq_len)
                mask = mask[:, :, :seq_len, :seq_len]
            elif mask.dim() == 3:  # (1, seq_len, seq_len)
                mask = mask[:, :seq_len, :seq_len]
                mask = mask.unsqueeze(1)  # add head dimension
            elif mask.dim() == 2:  # (seq_len, seq_len)
                mask = mask[:seq_len, :seq_len]
                mask = mask.unsqueeze(0).unsqueeze(0)  # add batch and head dimensions
            
            # Ensure mask on same device
            mask = mask.to(scores.device)
            
            # Apply the mask
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softma; returns weights
        attention = F.softmax(scores, dim=-1)
        output = torch.matmul(attention, V)  # (batch_size, num_heads, seq_len, head_dim)
        
        # concat heads back 
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.Wo(output) 
    
    def clear_cache(self):
        """Clear KV cache."""
        self.k_cache = None
        self.v_cache = None


