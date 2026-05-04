#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Food101 experiment results")
    parser.add_argument("--input-dir", type=str, default="outputs/food101")
    parser.add_argument("--output-dir", type=str, default="outputs/food101/figures")
    return parser.parse_args()


def load_results(input_dir: Path) -> pd.DataFrame:
    csv_files = sorted(input_dir.glob("seed_*/baseline_comparison.csv"))
    if not csv_files:
        combined = input_dir / "food101_experiment_all_seeds.csv"
        if not combined.exists():
            raise FileNotFoundError(f"No result CSV found under {input_dir}")
        csv_files = [combined]

    frames = []
    for csv_file in csv_files:
        frame = pd.read_csv(csv_file)
        if "seed" not in frame.columns:
            frame["seed"] = -1
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "loss",
        "accuracy",
        "routing_entropy",
        "load_balance",
        "load_cv",
        "throughput_samples_per_second",
    ]
    grouped = df.groupby("model", as_index=False)[metric_cols].agg(["mean", "std"])
    grouped.columns = ["_".join(col).rstrip("_") for col in grouped.columns.to_flat_index()]
    rename_map = {"model_": "model"}
    grouped = grouped.rename(columns=rename_map)
    ordered_cols = ["model"]
    for metric in metric_cols:
        ordered_cols.extend([f"{metric}_mean", f"{metric}_std"])
    return grouped[ordered_cols].sort_values("accuracy_mean", ascending=False)


def write_summary_table(summary: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "food101_summary_stats.csv", index=False)

    markdown_lines = [
        "| Model | Accuracy | Loss | Routing Entropy | Load Balance | Load CV | Throughput |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        markdown_lines.append(
            "| {model} | {accuracy:.4f} ± {accuracy_std:.4f} | {loss:.4f} ± {loss_std:.4f} | {routing_entropy:.4f} ± {routing_entropy_std:.4f} | {load_balance:.4f} ± {load_balance_std:.4f} | {load_cv:.4f} ± {load_cv_std:.4f} | {throughput_samples_per_second:.2f} ± {throughput_samples_per_second_std:.2f} |".format(
                model=row["model"],
                accuracy=row["accuracy_mean"],
                accuracy_std=row["accuracy_std"] if pd.notna(row["accuracy_std"]) else 0.0,
                loss=row["loss_mean"],
                loss_std=row["loss_std"] if pd.notna(row["loss_std"]) else 0.0,
                routing_entropy=row["routing_entropy_mean"],
                routing_entropy_std=row["routing_entropy_std"] if pd.notna(row["routing_entropy_std"]) else 0.0,
                load_balance=row["load_balance_mean"],
                load_balance_std=row["load_balance_std"] if pd.notna(row["load_balance_std"]) else 0.0,
                load_cv=row["load_cv_mean"],
                load_cv_std=row["load_cv_std"] if pd.notna(row["load_cv_std"]) else 0.0,
                throughput_samples_per_second=row["throughput_samples_per_second_mean"],
                throughput_samples_per_second_std=row["throughput_samples_per_second_std"] if pd.notna(row["throughput_samples_per_second_std"]) else 0.0,
            )
        )

    (output_dir / "food101_summary_table.md").write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")


def plot_metrics(summary: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    order = list(summary["model"])
    palette = {
        "kmeans_moe": "#1f77b4",
        "mrf_moe": "#d62728",
        "random_router_moe": "#7f7f7f",
        "linear_experts_moe": "#2ca02c",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Food101 Baseline Comparison", fontsize=16, fontweight="bold")

    metrics = [
        ("accuracy_mean", "accuracy_std", "Accuracy", axes[0, 0]),
        ("load_balance_mean", "load_balance_std", "Load Balance", axes[0, 1]),
        ("load_cv_mean", "load_cv_std", "Load CV", axes[1, 0]),
        ("throughput_samples_per_second_mean", "throughput_samples_per_second_std", "Throughput (samples/s)", axes[1, 1]),
    ]

    for mean_col, std_col, title, ax in metrics:
        values = summary.set_index("model").loc[order, mean_col]
        errors = summary.set_index("model").loc[order, std_col].fillna(0.0)
        colors = [palette.get(model, "#1f77b4") for model in order]
        ax.bar(order, values, yerr=errors, capsize=4, color=colors, alpha=0.9)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(output_dir / "food101_metrics.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "food101_metrics.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    results = load_results(input_dir)
    summary = summarize(results)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "food101_all_runs.csv", index=False)
    summary.to_csv(output_dir / "food101_summary_stats.csv", index=False)
    write_summary_table(summary, output_dir)
    plot_metrics(summary, output_dir)
    print(f"Saved plots and summaries to {output_dir}")


if __name__ == "__main__":
    main()