from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch

from .router import RoutingOutput


@dataclass
class EvaluationSummary:
    loss: float
    accuracy: float
    routing_entropy: float
    load_balance: float
    load_cv: float


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=-1)
    return float((predictions == labels).float().mean().item())


def compute_router_entropy(weights: torch.Tensor, eps: float = 1e-9) -> float:
    entropy = -(weights * torch.log(weights.clamp_min(eps))).sum(dim=-1)
    return float(entropy.mean().item())


def compute_load_balance(load: torch.Tensor, eps: float = 1e-9) -> float:
    normalized = load.float() / load.sum().clamp_min(1.0)
    ideal = torch.full_like(normalized, 1.0 / max(1, normalized.numel()))
    return float((1.0 - torch.abs(normalized - ideal).sum().item() / 2.0))


def compute_load_cv(load: torch.Tensor, eps: float = 1e-9) -> float:
    mean = load.float().mean().clamp_min(eps)
    std = load.float().std(unbiased=False)
    return float((std / mean).item())


def summarize_routing(routing: RoutingOutput) -> dict[str, float]:
    return {
        "routing_entropy": compute_router_entropy(routing.weights),
        "load_balance": compute_load_balance(routing.load),
        "load_cv": compute_load_cv(routing.load),
    }


@torch.no_grad()
def evaluate_model(model, loader, device) -> EvaluationSummary:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_correct = 0
    entropies = []
    balances = []
    cvs = []

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        logits, aux_loss = model(features)
        loss = torch.nn.functional.cross_entropy(logits, labels) + aux_loss
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=-1) == labels).sum().item())
        total_samples += batch_size

        routing = model.moe.router(features)
        routing_summary = summarize_routing(routing)
        entropies.append(routing_summary["routing_entropy"])
        balances.append(routing_summary["load_balance"])
        cvs.append(routing_summary["load_cv"])

    return EvaluationSummary(
        loss=total_loss / max(1, total_samples),
        accuracy=total_correct / max(1, total_samples),
        routing_entropy=sum(entropies) / max(1, len(entropies)),
        load_balance=sum(balances) / max(1, len(balances)),
        load_cv=sum(cvs) / max(1, len(cvs)),
    )


@torch.no_grad()
def measure_throughput(model, sample_batch: torch.Tensor, device, steps: int = 50) -> float:
    model.eval()
    sample_batch = sample_batch.to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = perf_counter()
    for _ in range(steps):
        _ = model(sample_batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = perf_counter() - start
    total_samples = sample_batch.size(0) * steps
    return total_samples / max(elapsed, 1e-9)
