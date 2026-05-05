#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
from torch.utils.data import DataLoader

from moe_project.baselines import LinearExpertBank
from moe_project.config import ProjectConfig
from moe_project.data import load_dataset
from moe_project.experts import ExpertBank
from moe_project.language_model import MoELanguageModel, RoutedLanguageModel
from moe_project.metrics import summarize_routing
from moe_project.router import KMeansRouter, MRFRouter, RandomRouter, build_router_warmup_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WikiText2 baseline comparisons")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/wikitext2")
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
    total_tokens = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, aux_loss = model(inputs)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)) + aux_loss
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * targets.numel()
        total_tokens += targets.numel()
    return total_loss / max(1, total_tokens)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    entropies = []
    balances = []
    cvs = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits, aux_loss = model(inputs)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)) + aux_loss
        total_loss += float(loss.item()) * targets.numel()
        total_tokens += targets.numel()

        routing = model.moe.router(model.router_features(inputs))
        routing_summary = summarize_routing(routing)
        entropies.append(routing_summary["routing_entropy"])
        balances.append(routing_summary["load_balance"])
        cvs.append(routing_summary["load_cv"])

    avg_loss = total_loss / max(1, total_tokens)
    return {
        "loss": avg_loss,
        "ppl": math.exp(min(20.0, avg_loss)),
        "routing_entropy": sum(entropies) / max(1, len(entropies)),
        "load_balance": sum(balances) / max(1, len(balances)),
        "load_cv": sum(cvs) / max(1, len(cvs)),
    }


def fit_router_if_needed(model, dataset, config, device):
    if hasattr(model.moe.router, "fit"):
        warmup = build_router_warmup_features(dataset, config.router_warmup_samples)
        router_features = model.router_features(warmup.to(device))
        try:
            model.moe.router.fit(router_features)
        except TypeError:
            model.moe.router.fit(router_features, 50)


def build_model_suite(config: ProjectConfig, vocab_size: int):
    return {
        "kmeans_moe": MoELanguageModel(config, vocab_size=vocab_size),
        "mrf_moe": RoutedLanguageModel(
            config,
            vocab_size=vocab_size,
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
        "random_router_moe": RoutedLanguageModel(
            config,
            vocab_size=vocab_size,
            router=RandomRouter(config.input_dim, config.num_experts, config.top_k, config.router_balance_weight),
            experts=ExpertBank(config.input_dim, config.expert_hidden_dim, config.num_experts, config.drop_rate),
        ),
        "linear_experts_moe": RoutedLanguageModel(
            config,
            vocab_size=vocab_size,
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
    train_set = load_dataset(
        "wikitext2",
        train=True,
        seed=seed,
        input_dim=config.input_dim,
        num_classes=config.num_classes,
        samples=args.samples,
        sequence_length=args.sequence_length,
    )
    test_set = load_dataset(
        "wikitext2",
        train=False,
        seed=seed,
        input_dim=config.input_dim,
        num_classes=config.num_classes,
        samples=args.samples,
        sequence_length=args.sequence_length,
    )

    vocab_size = int(getattr(train_set, "vocab_size"))
    config.num_classes = vocab_size

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    seed_dir = Path(args.output_dir) / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str]] = []
    for name, model in build_model_suite(config, vocab_size).items():
        model = model.to(device)
        fit_router_if_needed(model, train_set, config, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

        for _ in range(args.epochs):
            train_one_epoch(model, train_loader, optimizer, device)

        summary = evaluate(model, test_loader, device)
        sample_batch = torch.stack([test_set[i][0] for i in range(min(args.batch_size, len(test_set)))], dim=0)
        throughput = 0.0
        if sample_batch.numel() > 0:
            model.eval()
            sample_batch = sample_batch.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            if device.type == "cuda":
                start.record()
            else:
                import time

                t0 = time.perf_counter()
            steps = 50
            with torch.no_grad():
                for _ in range(steps):
                    _ = model(sample_batch)
            if device.type == "cuda":
                end.record()
                torch.cuda.synchronize()
                elapsed = start.elapsed_time(end) / 1000.0
            else:
                elapsed = time.perf_counter() - t0
            throughput = (sample_batch.size(0) * steps) / max(elapsed, 1e-9)

        row = {
            "seed": seed,
            "model": name,
            "loss": summary["loss"],
            "ppl": summary["ppl"],
            "routing_entropy": summary["routing_entropy"],
            "load_balance": summary["load_balance"],
            "load_cv": summary["load_cv"],
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
    summary_path = output_dir / "wikitext2_experiment_all_seeds.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)


if __name__ == "__main__":
    main()