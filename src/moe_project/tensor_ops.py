from __future__ import annotations

import torch


def chunk_by_expert(tokens: torch.Tensor, expert_ids: torch.Tensor, num_experts: int) -> list[torch.Tensor]:
    chunks: list[torch.Tensor] = []
    for expert_id in range(num_experts):
        mask = expert_ids == expert_id
        chunks.append(tokens[mask])
    return chunks
