from __future__ import annotations

import torch
import torch.nn as nn

from .config import ProjectConfig
from .experts import ExpertBank
from .router import KMeansRouter
from .tensor_ops import chunk_by_expert


class MoELayer(nn.Module):
    def __init__(self, config: ProjectConfig):
        super().__init__()
        self.router = KMeansRouter(
            input_dim=config.input_dim,
            num_experts=config.num_experts,
            top_k=config.top_k,
            temperature=config.router_temperature,
            balance_weight=config.router_balance_weight,
        )
        self.experts = ExpertBank(
            input_dim=config.input_dim,
            hidden_dim=config.expert_hidden_dim,
            num_experts=config.num_experts,
            drop_rate=config.drop_rate,
        )
        self.num_experts = config.num_experts

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        routing = self.router(x)

        flat_x = x.reshape(-1, x.size(-1))
        combined = torch.zeros_like(flat_x)

        for slot in range(routing.indices.size(1)):
            slot_indices = routing.indices[:, slot]
            slot_weights = routing.weights[:, slot].unsqueeze(-1)

            selected_inputs = chunk_by_expert(flat_x, slot_indices, self.num_experts)
            expert_outputs = self.experts(selected_inputs)

            routed_output = torch.zeros_like(flat_x)
            for expert_id in range(self.num_experts):
                mask = slot_indices == expert_id
                if mask.any():
                    routed_output[mask] = expert_outputs[expert_id]

            combined += routed_output * slot_weights

        return combined.reshape_as(x), routing.aux_loss


class MoEClassifier(nn.Module):
    def __init__(self, config: ProjectConfig):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.input_dim),
        )
        self.moe = MoELayer(config)
        self.norm = nn.LayerNorm(config.input_dim)
        self.head = nn.Linear(config.input_dim, config.num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input(x)
        moe_output, aux_loss = self.moe(self.norm(x))
        pooled = moe_output if moe_output.ndim == 2 else moe_output.mean(dim=1)
        logits = self.head(pooled)
        return logits, aux_loss
