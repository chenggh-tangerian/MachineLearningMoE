#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
from torch.utils.data import DataLoader

from moe_project.config import ProjectConfig
from moe_project.data import load_dataset
from moe_project.metrics import evaluate_model
from moe_project.model import MoEClassifier
from moe_project.router import build_router_warmup_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the KMeans-MoE project")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, default="", help="Optional checkpoint path")
    parser.add_argument("--output-json", type=str, default="", help="Optional JSON summary path")
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
        samples=args.samples,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = MoEClassifier(config).to(device)
    warmup_features = build_router_warmup_features(dataset, config.router_warmup_samples)
    model.moe.router.fit(warmup_features.to(device))

    if args.checkpoint:
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state_dict)

    summary = evaluate_model(model, loader, device)
    payload = {
        "loss": summary.loss,
        "accuracy": summary.accuracy,
        "routing_entropy": summary.routing_entropy,
        "load_balance": summary.load_balance,
        "load_cv": summary.load_cv,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
