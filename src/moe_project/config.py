from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProjectConfig:
    input_dim: int = 64
    hidden_dim: int = 128
    expert_hidden_dim: int = 256
    num_experts: int = 8
    top_k: int = 2
    num_classes: int = 10
    router_temperature: float = 1.0
    router_balance_weight: float = 0.01
    router_warmup_samples: int = 1024
    mrf_lambda: float = 0.5
    mrf_iters: int = 5
    mrf_neighbor_k: int = 4
    mrf_neighbor_sigma: float = 1.0
    learning_rate: float = 2e-3
    drop_rate: float = 0.1
