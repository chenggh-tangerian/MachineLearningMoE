from __future__ import annotations

import torch
import torch.nn as nn

from .config import ProjectConfig
from .router import RandomRouter
from .tensor_ops import chunk_by_expert


class LinearExpertBank(nn.Module):
    """单层 Linear 专家，用于和 MLP 专家做消融对比。"""

    def __init__(self, input_dim: int, num_experts: int, drop_rate: float = 0.0):
        super().__init__()
        self.num_experts = num_experts
        self.dropout = nn.Dropout(drop_rate)
        self.weights = nn.ParameterList([nn.Parameter(torch.empty(input_dim, input_dim)) for _ in range(num_experts)])
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(input_dim)) for _ in range(num_experts)])
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight in self.weights:
            nn.init.xavier_uniform_(weight)

    def forward(self, expert_inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        for tokens, weight, bias in zip(expert_inputs, self.weights, self.biases):
            if tokens.numel() == 0:
                outputs.append(tokens.new_empty((0, weight.size(0))))
                continue
            outputs.append(self.dropout(tokens @ weight.t() + bias))
        return outputs


class RoutedExpertLayer(nn.Module):
    def __init__(self, router: nn.Module, experts: nn.Module, num_experts: int):
        super().__init__()
        self.router = router
        self.experts = experts
        self.num_experts = num_experts

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


class RoutedClassifier(nn.Module):
    def __init__(self, config: ProjectConfig, router: nn.Module, experts: nn.Module):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.input_dim),
        )
        self.moe = RoutedExpertLayer(router=router, experts=experts, num_experts=config.num_experts)
        self.norm = nn.LayerNorm(config.input_dim)
        self.head = nn.Linear(config.input_dim, config.num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input(x)
        moe_output, aux_loss = self.moe(self.norm(x))
        pooled = moe_output if moe_output.ndim == 2 else moe_output.mean(dim=1)
        logits = self.head(pooled)
        return logits, aux_loss
