#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
from torch.utils.data import DataLoader

from moe_project.baselines import LinearExpertBank, RoutedClassifier
from moe_project.config import ProjectConfig
from moe_project.data import load_dataset
from moe_project.experts import ExpertBank
from moe_project.metrics import evaluate_model, measure_throughput
from moe_project.model import MoEClassifier
from moe_project.router import KMeansRouter, MRFRouter, RandomRouter, build_router_warmup_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Food101 baseline comparisons")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/food101")
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--seed-step", type=int, default=1)
    return parser.parse_args()


def build_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, aux_loss = model(features)
        loss = torch.nn.functional.cross_entropy(logits, labels) + aux_loss
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * labels.size(0)
        total_correct += int((logits.argmax(dim=-1) == labels).sum().item())
        total_samples += labels.size(0)
    return total_loss / max(1, total_samples), total_correct / max(1, total_samples)


def fit_router_if_needed(model, dataset, config, device):
    if hasattr(model.moe.router, "fit"):
        warmup = build_router_warmup_features(dataset, config.router_warmup_samples)
        try:
            model.moe.router.fit(warmup.to(device))
        except TypeError:
            model.moe.router.fit(warmup.to(device), 50)


def build_model_suite(config: ProjectConfig):
    return {
        "kmeans_moe": MoEClassifier(config),
        "mrf_moe": RoutedClassifier(
            config,
            router=MRFRouter(
                input_dim=config.input_dim,
                num_experts=config.num_experts,
                top_k=config.top_k,
                temperature=config.router_temperature,
                balance_weight=config.router_balance_weight,
                mrf_lambda=config.mrf_lambda,
                mrf_iters=config.mrf_iters,
                neighbor_k=config.mrf_neighbor_k,
                neighbor_sigma=config.mrf_neighbor_sigma,
            ),
            experts=ExpertBank(config.input_dim, config.expert_hidden_dim, config.num_experts, config.drop_rate),
        ),
        "random_router_moe": RoutedClassifier(
            config,
            router=RandomRouter(config.input_dim, config.num_experts, config.top_k, config.router_balance_weight),
            experts=ExpertBank(config.input_dim, config.expert_hidden_dim, config.num_experts, config.drop_rate),
        ),
        "linear_experts_moe": RoutedClassifier(
            config,
            router=KMeansRouter(
                input_dim=config.input_dim,
                num_experts=config.num_experts,
                top_k=config.top_k,
                temperature=config.router_temperature,
                balance_weight=config.router_balance_weight,
            ),
            experts=LinearExpertBank(config.input_dim, config.num_experts, config.drop_rate),
        ),
    }


def run_single_seed(seed: int, args: argparse.Namespace, device: torch.device) -> list[dict[str, float | int | str]]:
    torch.manual_seed(seed)

    config = ProjectConfig()
    config.num_classes = 101

    train_set = load_dataset(
        "food101",
        train=True,
        seed=seed,
        input_dim=config.input_dim,
        num_classes=config.num_classes,
        samples=args.samples,
    )
    test_set = load_dataset(
        "food101",
        train=False,
        seed=seed,
        input_dim=config.input_dim,
        num_classes=config.num_classes,
        samples=args.samples,
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    seed_dir = Path(args.output_dir) / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str]] = []
    for name, model in build_model_suite(config).items():
        model = model.to(device)
        fit_router_if_needed(model, train_set, config, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

        for _ in range(args.epochs):
            train_one_epoch(model, train_loader, optimizer, device)

        summary = evaluate_model(model, test_loader, device)
        sample_batch = torch.stack([test_set[i][0] for i in range(min(args.batch_size, len(test_set)))], dim=0)
        throughput = measure_throughput(model, sample_batch, device)

        row = {
            "seed": seed,
            "model": name,
            "loss": summary.loss,
            "accuracy": summary.accuracy,
            "routing_entropy": summary.routing_entropy,
            "load_balance": summary.load_balance,
            "load_cv": summary.load_cv,
            "throughput_samples_per_second": throughput,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    csv_path = seed_dir / "baseline_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    (seed_dir / "baseline_comparison.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return rows


def main() -> None:
    args = parse_args()
    device = build_device(args.device)

    all_rows: list[dict[str, float | int | str]] = []
    for i in range(args.num_seeds):
        seed = args.seed + i * args.seed_step
        all_rows.extend(run_single_seed(seed, args, device))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "food101_experiment_all_seeds.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)


if __name__ == "__main__":
    main()