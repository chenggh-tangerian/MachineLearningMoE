from __future__ import annotations

import torch
import torch.nn as nn

from .config import ProjectConfig
from .experts import ExpertBank
from .router import KMeansRouter
from .model import MoELayer


class RoutedLanguageModel(nn.Module):
    def __init__(
        self,
        config: ProjectConfig,
        *,
        vocab_size: int,
        router: nn.Module,
        experts: nn.Module,
        max_positions: int = 512,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_positions = max_positions
        self.embedding = nn.Embedding(vocab_size, config.input_dim)
        self.position_embedding = nn.Embedding(max_positions, config.input_dim)
        self.norm = nn.LayerNorm(config.input_dim)
        self.moe = MoELayer(config)
        self.moe.router = router
        self.moe.experts = experts
        self.head = nn.Linear(config.input_dim, vocab_size)

    def encode_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("RoutedLanguageModel expects token tensors with shape [batch, sequence]")

        seq_len = input_ids.size(1)
        if seq_len > self.max_positions:
            raise ValueError("sequence length exceeds max_positions")

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        return self.embedding(input_ids) + self.position_embedding(positions)

    def router_features(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.encode_tokens(input_ids).mean(dim=1)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encode_tokens(input_ids)
        moe_output, aux_loss = self.moe(self.norm(hidden))
        logits = self.head(moe_output)
        return logits, aux_loss


class MoELanguageModel(RoutedLanguageModel):
    def __init__(self, config: ProjectConfig, *, vocab_size: int, max_positions: int = 512):
        super().__init__(
            config,
            vocab_size=vocab_size,
            router=KMeansRouter(
                input_dim=config.input_dim,
                num_experts=config.num_experts,
                top_k=config.top_k,
                temperature=config.router_temperature,
                balance_weight=config.router_balance_weight,
            ),
            experts=ExpertBank(config.input_dim, config.expert_hidden_dim, config.num_experts, config.drop_rate),
            max_positions=max_positions,
        )