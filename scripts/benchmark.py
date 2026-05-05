#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from moe_project.config import ProjectConfig
from moe_project.data import load_dataset
from moe_project.metrics import measure_throughput
from moe_project.model import MoEClassifier
from moe_project.router import build_router_warmup_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark model throughput for KMeans-MoE")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch-sizes", nargs="*", type=int, default=[1, 8, 16, 32, 64])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", type=str, default="reports/benchmark_results.csv")
    parser.add_argument("--dataset", default="digits", choices=["digits", "synthetic", "mnist", "food101", "cifar100"])
    return parser.parse_args()


def build_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    config = ProjectConfig()
    if args.dataset == "food101":
        config.num_classes = 101
    if args.dataset == "cifar100":
        config.input_dim = 1024
        config.num_classes = 100
    device = build_device(args.device)

    dataset = load_dataset(
        args.dataset,
        train=False,
        seed=args.seed,
        input_dim=config.input_dim,
        num_classes=config.num_classes,
        samples=max(args.batch_sizes) * 4,
    )
    warmup_features = build_router_warmup_features(dataset, config.router_warmup_samples)
    model = MoEClassifier(config).to(device)
    model.moe.router.fit(warmup_features.to(device))

    rows = []
    for batch_size in args.batch_sizes:
        sample_batch = torch.stack([dataset[i][0] for i in range(batch_size)], dim=0)
        throughput = measure_throughput(model, sample_batch, device)
        rows.append({"batch_size": batch_size, "throughput_samples_per_second": throughput})
        print(f"batch_size={batch_size} throughput={throughput:.2f} samples/s")

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["batch_size", "throughput_samples_per_second"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
