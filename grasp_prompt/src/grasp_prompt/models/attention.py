from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, dim: int, multiplier: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * multiplier),
            nn.GELU(),
            nn.Linear(dim * multiplier, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalDynamicAttention(nn.Module):
    """Self-attention with a diffusion-time-dependent attention-bias matrix."""

    def __init__(self, dim: int, num_heads: int, max_tokens: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.max_tokens = max_tokens
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.time_to_bias = nn.Linear(dim, num_heads * max_tokens * max_tokens)
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        bsz, tokens, dim = x.shape
        if tokens > self.max_tokens:
            raise ValueError(f"sequence length {tokens} exceeds max_tokens={self.max_tokens}")
        qkv = self.qkv(x).view(bsz, tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        bias = self.time_to_bias(time_emb).view(
            bsz, self.num_heads, self.max_tokens, self.max_tokens
        )
        logits = logits + self.gamma * bias[:, :, :tokens, :tokens]
        attn = F.softmax(logits, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bsz, tokens, dim)
        return self.out(out)


class TemporalEncoderBlock(nn.Module):
    """TDA plus local prompt-region aggregation."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        max_tokens: int,
        local_k: int = 4,
        use_tda: bool = True,
    ) -> None:
        super().__init__()
        self.use_tda = use_tda
        self.attn = (
            TemporalDynamicAttention(dim, num_heads, max_tokens)
            if use_tda
            else nn.MultiheadAttention(dim, num_heads, batch_first=True)
        )
        self.local_proj = nn.Linear(dim, dim)
        self.ffn = FeedForward(dim)
        self.norm_attn = nn.LayerNorm(dim)
        self.norm_local = nn.LayerNorm(dim)
        self.norm_ffn = nn.LayerNorm(dim)
        self.local_k = local_k

    def forward(self, x: torch.Tensor, coords: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        if self.use_tda:
            attn_out = self.attn(x, time_emb)
        else:
            attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm_attn(x + attn_out)
        local = self._local_region_aggregate(x, coords)
        x = self.norm_local(x + self.local_proj(local))
        x = self.norm_ffn(x + self.ffn(x))
        return x

    def _local_region_aggregate(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        bsz, tokens, dim = x.shape
        if tokens == 1:
            return x
        k = min(self.local_k + 1, tokens)
        dist = torch.cdist(coords, coords)
        idx = dist.topk(k=k, largest=False).indices
        x_expanded = x.unsqueeze(1).expand(bsz, tokens, tokens, dim)
        idx_expanded = idx.unsqueeze(-1).expand(bsz, tokens, k, dim)
        neigh = torch.gather(x_expanded, dim=2, index=idx_expanded)
        return neigh.mean(dim=2)


class CrossDimConditionInteraction(nn.Module):
    """Bidirectional prompt-condition interaction used by the decoder."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.prompt_to_cond = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.cond_to_prompt = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.condition_pool_proj = nn.Linear(dim, dim)
        self.ffn = FeedForward(dim)
        self.norm_interact = nn.LayerNorm(dim)
        self.norm_ffn = nn.LayerNorm(dim)

    def forward(self, prompt: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        p_update, _ = self.prompt_to_cond(prompt, condition, condition, need_weights=False)
        c_update, _ = self.cond_to_prompt(condition, prompt, prompt, need_weights=False)
        c_pool = c_update.mean(dim=1, keepdim=True).expand_as(prompt)
        prompt = self.norm_interact(prompt + p_update + self.condition_pool_proj(c_pool))
        prompt = self.norm_ffn(prompt + self.ffn(prompt))
        return prompt


class PromptDecoderBlock(nn.Module):
    """Decoder block used for the full model and the no-CDCI ablation."""

    def __init__(self, dim: int, num_heads: int, use_cdci: bool = True) -> None:
        super().__init__()
        self.use_cdci = use_cdci
        if use_cdci:
            self.block = CrossDimConditionInteraction(dim, num_heads)
        else:
            self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
            self.ffn = FeedForward(dim)
            self.norm_attn = nn.LayerNorm(dim)
            self.norm_ffn = nn.LayerNorm(dim)

    def forward(self, prompt: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if self.use_cdci:
            return self.block(prompt, condition)
        attn_out, _ = self.self_attn(prompt, prompt, prompt, need_weights=False)
        prompt = self.norm_attn(prompt + attn_out)
        return self.norm_ffn(prompt + self.ffn(prompt))
