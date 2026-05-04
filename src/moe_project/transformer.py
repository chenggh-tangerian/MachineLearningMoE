from __future__ import annotations

import torch
import torch.nn as nn

from .config import ProjectConfig
from .model import MoELayer


class TransformerBlock(nn.Module):
    """简化版 Transformer Block：Self-Attention + MoE 代替 FFN。"""

    def __init__(self, config: ProjectConfig, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(config.input_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(config.input_dim)
        self.norm2 = nn.LayerNorm(config.input_dim)
        self.moe = MoELayer(config)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, L, D)
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        y = self.norm1(x)

        moe_out, aux = self.moe(self.norm2(y))
        x = x + moe_out
        return x, aux


class SimpleTransformer(nn.Module):
    """一个非常小的 Transformer 分类器，演示如何把 MoE 嵌入到 FFN 位置。"""

    def __init__(self, config: ProjectConfig, seq_len: int = 1, num_layers: int = 2, num_heads: int = 4):
        super().__init__()
        self.config = config
        self.seq_len = seq_len
        self.input_proj = nn.Linear(config.input_dim, config.input_dim)
        self.blocks = nn.ModuleList([TransformerBlock(config, num_heads=num_heads) for _ in range(num_layers)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(config.input_dim, config.num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Accepts (B, L, D) or (B, D)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = self.input_proj(x)
        aux_losses = []
        for blk in self.blocks:
            x, aux = blk(x)
            aux_losses.append(aux)

        # pool over length
        pooled = x.mean(dim=1)
        logits = self.head(pooled)
        total_aux = sum(aux_losses) if aux_losses else x.new_zeros(1).sum()
        return logits, total_aux
