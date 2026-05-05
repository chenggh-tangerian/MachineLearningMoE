#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CIFAR-100 and WikiText2 routing comparisons")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cifar-epochs", type=int, default=3)
    parser.add_argument("--cifar-batch-size", type=int, default=16)
    parser.add_argument("--cifar-samples", type=int, default=2048)
    parser.add_argument("--wikitext-epochs", type=int, default=3)
    parser.add_argument("--wikitext-batch-size", type=int, default=16)
    parser.add_argument("--wikitext-samples", type=int, default=1024)
    parser.add_argument("--wikitext-sequence-length", type=int, default=64)
    parser.add_argument("--smoke", action="store_true", help="Run a very small sanity check config")
    parser.add_argument("--output-dir", type=str, default="outputs/routing_table")
    return parser.parse_args()


def run_command(args: list[str]) -> None:
    subprocess.run(args, cwd=ROOT, check=True)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json_rows(path: Path) -> list[dict[str, object]]:
    return json.loads(path.read_text(encoding="utf-8"))


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    args = parse_args()

    if args.smoke:
        args.cifar_epochs = 1
        args.cifar_batch_size = 4
        args.cifar_samples = 32
        args.wikitext_epochs = 1
        args.wikitext_batch_size = 4
        args.wikitext_samples = 32
        args.wikitext_sequence_length = 16

    if args.cifar_samples < 512 or args.cifar_epochs < 2 or args.wikitext_samples < 512 or args.wikitext_epochs < 2:
        print(
            "[warning] 当前参数偏小，更适合连通性验证（smoke），不适合做最终结论。"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cifar_dir = output_dir / "cifar100"
    wikitext_dir = output_dir / "wikitext2"

    run_command(
        [
            PYTHON,
            "scripts/compare_baselines.py",
            "--device",
            args.device,
            "--epochs",
            str(args.cifar_epochs),
            "--batch-size",
            str(args.cifar_batch_size),
            "--samples",
            str(args.cifar_samples),
            "--seed",
            str(args.seed),
            "--output-dir",
            str(cifar_dir),
            "--dataset",
            "cifar100",
        ]
    )

    run_command(
        [
            PYTHON,
            "scripts/run_wikitext2_experiment.py",
            "--device",
            args.device,
            "--epochs",
            str(args.wikitext_epochs),
            "--batch-size",
            str(args.wikitext_batch_size),
            "--samples",
            str(args.wikitext_samples),
            "--sequence-length",
            str(args.wikitext_sequence_length),
            "--seed",
            str(args.seed),
            "--num-seeds",
            "1",
            "--output-dir",
            str(wikitext_dir),
        ]
    )

    cifar_rows = load_csv_rows(cifar_dir / "baseline_comparison.csv")
    wiki_rows = load_json_rows(wikitext_dir / f"seed_{args.seed}" / "baseline_comparison.json")

    wiki_lookup = {row["model"]: row for row in wiki_rows}
    combined_rows: list[dict[str, object]] = []

    order = ["random_router_moe", "linear_experts_moe", "kmeans_moe", "mrf_moe"]
    label_map = {
        "random_router_moe": "Random 随机路由基线",
        "linear_experts_moe": "Linear 线性路由基线",
        "kmeans_moe": "K-Means 距离路由",
        "mrf_moe": "MRF 平滑路由",
    }

    for model_name in order:
        cifar_row = next(row for row in cifar_rows if row["model"] == model_name)
        wiki_row = wiki_lookup[model_name]
        combined_rows.append(
            {
                "routing_type": label_map[model_name],
                "cifar100_accuracy": float(cifar_row["accuracy"]),
                "wikitext_ppl": float(wiki_row["ppl"]),
                "load_cv": float(cifar_row["load_cv"]),
                "throughput_samples_per_second": float(cifar_row["throughput_samples_per_second"]),
            }
        )

    csv_path = output_dir / "routing_table.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "routing_type",
                "cifar100_accuracy",
                "wikitext_ppl",
                "load_cv",
                "throughput_samples_per_second",
            ],
        )
        writer.writeheader()
        writer.writerows(combined_rows)

    markdown_lines = [
        "| 路由策略 | CIFAR-100 准确率 | WikiText 困惑度 (PPL) | 负载变异系数 (Load CV) | 推理吞吐量 (Samples/s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in combined_rows:
        markdown_lines.append(
            "| {routing_type} | {acc} | {ppl:.2f} | {load_cv:.3f} | {thr:,.0f} |".format(
                routing_type=row["routing_type"],
                acc=format_percent(float(row["cifar100_accuracy"])),
                ppl=float(row["wikitext_ppl"]),
                load_cv=float(row["load_cv"]),
                thr=float(row["throughput_samples_per_second"]),
            )
        )

    md_path = output_dir / "routing_table.md"
    md_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    print(md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()