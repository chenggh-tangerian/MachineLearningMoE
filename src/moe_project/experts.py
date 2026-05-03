from __future__ import annotations

import torch
import torch.nn as nn


class ExpertBank(nn.Module):
    """按专家存储的两层 MLP。"""

    def __init__(self, input_dim: int, hidden_dim: int, num_experts: int, drop_rate: float = 0.1):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(drop_rate),
                    nn.Linear(hidden_dim, input_dim),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, expert_inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        for expert, tokens in zip(self.experts, expert_inputs):
            if tokens.numel() == 0:
                outputs.append(tokens.new_empty((0, tokens.size(-1))))
                continue
            outputs.append(expert(tokens))
        return outputs
