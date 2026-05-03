from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class RoutingOutput:
    indices: torch.Tensor
    weights: torch.Tensor
    aux_loss: torch.Tensor
    load: torch.Tensor


class KMeansRouter(nn.Module):
    """用 K-Means 近似专家原型的路由器。"""

    def __init__(self, input_dim: int, num_experts: int, top_k: int, temperature: float = 1.0, balance_weight: float = 0.01):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = temperature
        self.balance_weight = balance_weight
        self.register_buffer("centroids", torch.empty(num_experts, input_dim))
        self.register_buffer("fitted", torch.tensor(False))

    def fit(self, features: torch.Tensor, max_iter: int = 50) -> None:
        features = features.detach().float().cpu()
        if features.ndim != 2:
            raise ValueError("KMeansRouter.fit expects a 2D tensor")

        try:
            sklearn_cluster = importlib.import_module("sklearn.cluster")
            mini_batch_kmeans = sklearn_cluster.MiniBatchKMeans(
                n_clusters=self.num_experts,
                random_state=0,
                batch_size=min(256, len(features)),
            )
            mini_batch_kmeans.fit(features.numpy())
            centroids = torch.from_numpy(mini_batch_kmeans.cluster_centers_).to(
                self.centroids.device,
                dtype=self.centroids.dtype,
            )
        except Exception:
            indices = torch.randperm(features.size(0))[: self.num_experts]
            centroids = features[indices].clone()
            for _ in range(max_iter):
                distances = torch.cdist(features, centroids)
                assignments = distances.argmin(dim=1)
                new_centroids = []
                for expert_id in range(self.num_experts):
                    mask = assignments == expert_id
                    if mask.any():
                        new_centroids.append(features[mask].mean(dim=0))
                    else:
                        new_centroids.append(centroids[expert_id])
                centroids = torch.stack(new_centroids, dim=0)

        self.centroids.copy_(centroids.to(self.centroids.device))
        self.fitted.fill_(True)

    def forward(self, x: torch.Tensor) -> RoutingOutput:
        if not bool(self.fitted.item()):
            raise RuntimeError("KMeansRouter must be fitted before use")

        if x.ndim > 2:
            x = x.reshape(-1, x.size(-1))

        scores = -torch.cdist(x.float(), self.centroids.float()) / self.temperature
        top_scores, top_indices = scores.topk(self.top_k, dim=-1)
        weights = torch.softmax(top_scores, dim=-1)

        load = torch.bincount(top_indices.reshape(-1), minlength=self.num_experts).float()
        load_ratio = load / load.sum().clamp_min(1.0)
        balance_loss = self.balance_weight * self.num_experts * torch.sum(load_ratio * load_ratio)

        return RoutingOutput(indices=top_indices, weights=weights, aux_loss=balance_loss, load=load)


class RandomRouter(nn.Module):
    """随机 top-k 路由器，用于作为路由基线。"""

    def __init__(self, input_dim: int, num_experts: int, top_k: int, balance_weight: float = 0.01):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.balance_weight = balance_weight

    def fit(self, features: torch.Tensor, max_iter: int = 50) -> None:
        return None

    def forward(self, x: torch.Tensor) -> RoutingOutput:
        if x.ndim > 2:
            x = x.reshape(-1, x.size(-1))

        batch_size = x.size(0)
        scores = torch.rand(batch_size, self.num_experts, device=x.device, dtype=x.dtype)
        top_scores, top_indices = scores.topk(self.top_k, dim=-1)
        weights = torch.softmax(top_scores, dim=-1)

        load = torch.bincount(top_indices.reshape(-1), minlength=self.num_experts).float()
        load_ratio = load / load.sum().clamp_min(1.0)
        balance_loss = self.balance_weight * self.num_experts * torch.sum(load_ratio * load_ratio)

        return RoutingOutput(indices=top_indices, weights=weights, aux_loss=balance_loss, load=load)


def build_router_warmup_features(dataset, limit: Optional[int] = None) -> torch.Tensor:
    features = []
    for index in range(len(dataset)):
        feature, _ = dataset[index]
        features.append(feature)
        if limit is not None and len(features) >= limit:
            break
    return torch.stack(features, dim=0)
