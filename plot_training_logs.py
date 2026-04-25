import argparse
import json
import os
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    if not os.path.isfile(path):
        return rows
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_line_plot(
    path: str,
    title: str,
    x_label: str,
    y_label: str,
    series: list[dict[str, Any]],
) -> None:
    if not series:
        return
    plt.figure(figsize=(10, 6))
    for item in series:
        xs = item["x"]
        ys = item["y"]
        if not xs or not ys:
            continue
        plt.plot(xs, ys, marker="o", linewidth=2, markersize=3, label=item["label"])
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.3)
    if len(series) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_bar_plot(
    path: str,
    title: str,
    x_label: str,
    y_label: str,
    labels: list[str],
    values: list[float],
) -> None:
    if not labels:
        return
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=15, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def extract_metric(rows: list[dict[str, Any]], key: str) -> tuple[list[float], list[float]]:
    xs = []
    ys = []
    for row in rows:
        if key in row and row.get("step") is not None:
            xs.append(row["step"])
            ys.append(row[key])
    return xs, ys


def plot_sft(run_dir: str, output_dir: str) -> list[str]:
    plot_data_dir = os.path.join(run_dir, "plot_data")
    train_rows = read_jsonl(os.path.join(plot_data_dir, "train_metrics.jsonl"))
    eval_rows = read_jsonl(os.path.join(plot_data_dir, "eval_metrics.jsonl"))
    dataset_profile_path = os.path.join(plot_data_dir, "dataset_profile.json")
    summary_path = os.path.join(plot_data_dir, "summary.json")

    created = []

    train_loss_x, train_loss_y = extract_metric(train_rows, "loss")
    if train_loss_x:
        path = os.path.join(output_dir, "sft_train_loss.png")
        save_line_plot(
            path,
            "SFT Training Loss",
            "Training Step",
            "Loss",
            [{"label": "train_loss", "x": train_loss_x, "y": train_loss_y}],
        )
        created.append(path)

    learning_rate_x, learning_rate_y = extract_metric(train_rows, "learning_rate")
    if learning_rate_x:
        path = os.path.join(output_dir, "sft_learning_rate.png")
        save_line_plot(
            path,
            "SFT Learning Rate",
            "Training Step",
            "Learning Rate",
            [{"label": "learning_rate", "x": learning_rate_x, "y": learning_rate_y}],
        )
        created.append(path)

    eval_loss_x, eval_loss_y = extract_metric(eval_rows, "eval_loss")
    if eval_loss_x:
        path = os.path.join(output_dir, "sft_eval_loss.png")
        save_line_plot(
            path,
            "SFT Validation Loss",
            "Training Step",
            "Eval Loss",
            [{"label": "eval_loss", "x": eval_loss_x, "y": eval_loss_y}],
        )
        created.append(path)

    if os.path.isfile(dataset_profile_path):
        dataset_profile = read_json(dataset_profile_path)
        labels = []
        values = []
        for split_name, stats in dataset_profile.items():
            labels.append(f"{split_name}_rows")
            values.append(stats.get("num_examples", 0))
        path = os.path.join(output_dir, "sft_dataset_sizes.png")
        save_bar_plot(path, "SFT Dataset Sizes", "Split", "Rows", labels, values)
        created.append(path)

        labels = []
        values = []
        for split_name, stats in dataset_profile.items():
            labels.append(f"{split_name}_avg_input")
            values.append(stats.get("avg_input_tokens", 0))
        path = os.path.join(output_dir, "sft_avg_input_tokens.png")
        save_bar_plot(path, "SFT Average Input Tokens", "Split", "Tokens", labels, values)
        created.append(path)

        labels = []
        values = []
        for split_name, stats in dataset_profile.items():
            labels.append(f"{split_name}_avg_target")
            values.append(stats.get("avg_target_tokens", 0))
        path = os.path.join(output_dir, "sft_avg_target_tokens.png")
        save_bar_plot(path, "SFT Average Target Tokens", "Split", "Tokens", labels, values)
        created.append(path)

    if os.path.isfile(summary_path):
        summary = read_json(summary_path)
        labels = ["train_rows", "validation_rows", "final_global_step"]
        values = [
            summary.get("train_rows", 0),
            summary.get("validation_rows", 0),
            summary.get("final_global_step", 0),
        ]
        path = os.path.join(output_dir, "sft_run_summary.png")
        save_bar_plot(path, "SFT Run Summary", "Metric", "Value", labels, values)
        created.append(path)

    return created


def plot_grpo(run_dir: str, output_dir: str) -> list[str]:
    plot_data_dir = os.path.join(run_dir, "plot_data")
    train_rows = read_jsonl(os.path.join(plot_data_dir, "train_metrics.jsonl"))
    reward_rows = read_jsonl(os.path.join(plot_data_dir, "reward_components.jsonl"))
    dataset_profile_path = os.path.join(plot_data_dir, "dataset_profile.json")
    summary_path = os.path.join(plot_data_dir, "summary.json")

    created = []

    line_specs = []
    for key in ["loss", "reward", "kl", "learning_rate"]:
        xs, ys = extract_metric(train_rows, key)
        if xs:
            line_specs.append({"label": key, "x": xs, "y": ys})
    if line_specs:
        path = os.path.join(output_dir, "grpo_training_metrics.png")
        save_line_plot(
            path,
            "GRPO Trainer Metrics",
            "Training Step",
            "Metric Value",
            line_specs,
        )
        created.append(path)

    component_series: dict[str, dict[str, list[float] | str]] = {}
    for row in reward_rows:
        step = row.get("step")
        components = row.get("components", {})
        for name, stats in components.items():
            component_series.setdefault(name, {"label": name, "x": [], "y": []})
            component_series[name]["x"].append(step)
            component_series[name]["y"].append(stats.get("mean", 0.0))
    if component_series:
        path = os.path.join(output_dir, "grpo_reward_components.png")
        save_line_plot(
            path,
            "GRPO Reward Components",
            "Training Step",
            "Mean Reward",
            list(component_series.values()),
        )
        created.append(path)

    quality_rows = read_json(os.path.join(run_dir, "metrics.json")) if os.path.isfile(os.path.join(run_dir, "metrics.json")) else []
    if quality_rows:
        xs = list(range(1, len(quality_rows) + 1))
        series = []
        for key in ["valid_json_rate", "accuracy", "agent_accuracy", "unsafe_rate"]:
            series.append({"label": key, "x": xs, "y": [row.get(key, 0.0) for row in quality_rows]})
        path = os.path.join(output_dir, "grpo_quality_metrics.png")
        save_line_plot(
            path,
            "GRPO Quality Metrics",
            "Logged Batch",
            "Rate",
            series,
        )
        created.append(path)

    if os.path.isfile(dataset_profile_path):
        dataset_profile = read_json(dataset_profile_path)
        labels = ["num_examples", "num_scenarios", "avg_prompt_chars", "avg_unsafe_actions"]
        values = [
            dataset_profile.get("num_examples", 0),
            dataset_profile.get("num_scenarios", 0),
            dataset_profile.get("avg_prompt_chars", 0),
            dataset_profile.get("avg_unsafe_actions", 0),
        ]
        path = os.path.join(output_dir, "grpo_dataset_profile.png")
        save_bar_plot(path, "GRPO Dataset Profile", "Metric", "Value", labels, values)
        created.append(path)

    if os.path.isfile(summary_path):
        summary = read_json(summary_path)
        labels = ["num_examples", "final_global_step", "log_history_entries"]
        values = [
            summary.get("num_examples", 0),
            summary.get("final_global_step", 0),
            summary.get("log_history_entries", 0),
        ]
        path = os.path.join(output_dir, "grpo_run_summary.png")
        save_bar_plot(path, "GRPO Run Summary", "Metric", "Value", labels, values)
        created.append(path)

        last_quality = summary.get("last_quality_metrics", {})
        if last_quality:
            labels = ["valid_json_rate", "accuracy", "agent_accuracy", "unsafe_rate"]
            values = [last_quality.get(label, 0.0) for label in labels]
            path = os.path.join(output_dir, "grpo_final_quality_snapshot.png")
            save_bar_plot(path, "GRPO Final Quality Snapshot", "Metric", "Value", labels, values)
            created.append(path)

    return created


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate plot PNGs from SFT/GRPO training logs.")
    parser.add_argument("--sft-dir", default="", help="Path to the SFT output directory.")
    parser.add_argument("--grpo-dir", default="", help="Path to the GRPO output directory.")
    parser.add_argument("--output-dir", default="plots/generated", help="Where to save PNG plots.")
    args = parser.parse_args()

    if not args.sft_dir and not args.grpo_dir:
        raise ValueError("Pass at least one of --sft-dir or --grpo-dir.")

    ensure_dir(args.output_dir)
    created_files = []
    if args.sft_dir:
        created_files.extend(plot_sft(args.sft_dir, args.output_dir))
    if args.grpo_dir:
        created_files.extend(plot_grpo(args.grpo_dir, args.output_dir))

    manifest = {
        "sft_dir": args.sft_dir,
        "grpo_dir": args.grpo_dir,
        "output_dir": args.output_dir,
        "created_files": created_files,
    }
    with open(os.path.join(args.output_dir, "plot_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
