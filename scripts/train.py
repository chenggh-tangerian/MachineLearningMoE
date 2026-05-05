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
from moe_project.model import MoEClassifier
from moe_project.router import build_router_warmup_features
from moe_project.transformer import SimpleTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the KMeans-MoE project")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Training device")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--dataset", default="digits", choices=["digits", "synthetic", "mnist", "food101", "cifar100"], help="Benchmark dataset")
    parser.add_argument("--model", default="moe", choices=["moe", "transformer"], help="Model type")
    parser.add_argument("--seq-len", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="outputs/train")
    parser.add_argument("--save-checkpoint", action="store_true")
    return parser.parse_args()


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

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        logits, _ = model(features)
        total_correct += int((logits.argmax(dim=-1) == labels).sum().item())
        total_samples += labels.size(0)
    return total_correct / total_samples


def fit_router(model, router_features, device):
    if hasattr(model, "moe"):
        model.moe.router.fit(router_features.to(device))
        return

    if hasattr(model, "blocks"):
        for block in model.blocks:
            block.moe.router.fit(router_features.to(device))


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    config = ProjectConfig()
    if args.dataset == "food101":
        config.num_classes = 101
    if args.dataset == "cifar100":
        config.input_dim = 1024
        config.num_classes = 100
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_set = load_dataset(
        args.dataset,
        train=True,
        seed=args.seed,
        input_dim=config.input_dim,
        num_classes=config.num_classes,
        samples=args.samples,
    )
    test_set = load_dataset(
        args.dataset,
        train=False,
        seed=args.seed,
        input_dim=config.input_dim,
        num_classes=config.num_classes,
        samples=args.samples,
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    router_features = build_router_warmup_features(train_set, config.router_warmup_samples)
    if args.model == "transformer":
        model = SimpleTransformer(
            config,
            seq_len=args.seq_len,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
        ).to(device)
    else:
        model = MoEClassifier(config).to(device)

    fit_router(model, router_features, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        test_acc = evaluate(model, test_loader, device)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
        }
        history.append(record)
        print(f"epoch={epoch:02d} train_loss={train_loss:.4f} train_acc={train_acc:.4f} test_acc={test_acc:.4f}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.save_checkpoint:
        torch.save(model.state_dict(), save_dir / "model.pt")


if __name__ == "__main__":
    main()
