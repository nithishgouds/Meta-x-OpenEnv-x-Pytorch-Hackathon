import argparse
import json
import os
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download


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


def maybe_download_file(repo_id: str, filename: str, subfolder: str = "") -> str:
    try:
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            subfolder=subfolder or None,
            repo_type="model",
        )
    except Exception:
        return ""


def resolve_local_or_hub_file(local_path: str, repo_id: str, filename: str, subfolder: str = "") -> str:
    if local_path and os.path.isfile(local_path):
        return local_path
    if repo_id:
        return maybe_download_file(repo_id, filename, subfolder)
    return ""


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


def save_dual_axis_plot(
    path: str,
    title: str,
    x_label: str,
    left_label: str,
    right_label: str,
    left_series: list[dict[str, Any]],
    right_series: list[dict[str, Any]],
) -> None:
    if not left_series and not right_series:
        return
    fig, ax_left = plt.subplots(figsize=(10, 6))
    ax_right = ax_left.twinx()
    handles = []
    labels = []

    for item in left_series:
        xs = item["x"]
        ys = item["y"]
        if not xs or not ys:
            continue
        line, = ax_left.plot(xs, ys, marker="o", linewidth=2, markersize=3, label=item["label"])
        handles.append(line)
        labels.append(item["label"])

    for item in right_series:
        xs = item["x"]
        ys = item["y"]
        if not xs or not ys:
            continue
        line, = ax_right.plot(xs, ys, marker="o", linewidth=2, markersize=3, linestyle="--", label=item["label"])
        handles.append(line)
        labels.append(item["label"])

    ax_left.set_title(title)
    ax_left.set_xlabel(x_label)
    ax_left.set_ylabel(left_label)
    ax_right.set_ylabel(right_label)
    ax_left.grid(True, alpha=0.3)
    if handles:
        ax_left.legend(handles, labels, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


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


def save_histogram(
    path: str,
    title: str,
    x_label: str,
    y_label: str,
    values: list[float],
    bins: int = 20,
) -> None:
    if not values:
        return
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=bins, edgecolor="black", alpha=0.8)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
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


def moving_average(values: list[float], window: int = 5) -> list[float]:
    if not values:
        return []
    averaged = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        chunk = values[start : index + 1]
        averaged.append(sum(chunk) / len(chunk))
    return averaged


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


def plot_sft_from_sources(run_dir: str, repo_id: str, output_dir: str) -> list[str]:
    plot_data_dir = os.path.join(run_dir, "plot_data") if run_dir else ""
    train_rows = read_jsonl(resolve_local_or_hub_file(
        os.path.join(plot_data_dir, "train_metrics.jsonl") if plot_data_dir else "",
        repo_id,
        "train_metrics.jsonl",
        "plot_data",
    ))
    eval_rows = read_jsonl(resolve_local_or_hub_file(
        os.path.join(plot_data_dir, "eval_metrics.jsonl") if plot_data_dir else "",
        repo_id,
        "eval_metrics.jsonl",
        "plot_data",
    ))
    dataset_profile_path = resolve_local_or_hub_file(
        os.path.join(plot_data_dir, "dataset_profile.json") if plot_data_dir else "",
        repo_id,
        "dataset_profile.json",
        "plot_data",
    )
    summary_path = resolve_local_or_hub_file(
        os.path.join(plot_data_dir, "summary.json") if plot_data_dir else "",
        repo_id,
        "summary.json",
        "plot_data",
    )

    created = []

    train_loss_x, train_loss_y = extract_metric(train_rows, "loss")
    if train_loss_x:
        path = os.path.join(output_dir, "sft_train_loss.png")
        save_line_plot(path, "SFT Training Loss", "Training Step", "Loss", [{"label": "train_loss", "x": train_loss_x, "y": train_loss_y}])
        created.append(path)
        path = os.path.join(output_dir, "sft_train_loss_smoothed.png")
        save_line_plot(
            path,
            "SFT Training Loss (Smoothed)",
            "Training Step",
            "Loss",
            [
                {"label": "train_loss", "x": train_loss_x, "y": train_loss_y},
                {"label": "train_loss_ma5", "x": train_loss_x, "y": moving_average(train_loss_y, window=5)},
            ],
        )
        created.append(path)

    learning_rate_x, learning_rate_y = extract_metric(train_rows, "learning_rate")
    if learning_rate_x:
        path = os.path.join(output_dir, "sft_learning_rate.png")
        save_line_plot(path, "SFT Learning Rate", "Training Step", "Learning Rate", [{"label": "learning_rate", "x": learning_rate_x, "y": learning_rate_y}])
        created.append(path)
        if train_loss_x:
            path = os.path.join(output_dir, "sft_loss_vs_lr.png")
            save_dual_axis_plot(
                path,
                "SFT Loss and Learning Rate",
                "Training Step",
                "Loss",
                "Learning Rate",
                [{"label": "train_loss", "x": train_loss_x, "y": train_loss_y}],
                [{"label": "learning_rate", "x": learning_rate_x, "y": learning_rate_y}],
            )
            created.append(path)

    eval_loss_x, eval_loss_y = extract_metric(eval_rows, "eval_loss")
    if eval_loss_x:
        path = os.path.join(output_dir, "sft_eval_loss.png")
        save_line_plot(path, "SFT Validation Loss", "Training Step", "Eval Loss", [{"label": "eval_loss", "x": eval_loss_x, "y": eval_loss_y}])
        created.append(path)
        path = os.path.join(output_dir, "sft_eval_loss_smoothed.png")
        save_line_plot(
            path,
            "SFT Validation Loss (Smoothed)",
            "Training Step",
            "Eval Loss",
            [
                {"label": "eval_loss", "x": eval_loss_x, "y": eval_loss_y},
                {"label": "eval_loss_ma3", "x": eval_loss_x, "y": moving_average(eval_loss_y, window=3)},
            ],
        )
        created.append(path)
        if train_loss_x:
            path = os.path.join(output_dir, "sft_train_vs_eval_loss.png")
            save_line_plot(
                path,
                "SFT Train vs Validation Loss",
                "Training Step",
                "Loss",
                [
                    {"label": "train_loss", "x": train_loss_x, "y": train_loss_y},
                    {"label": "eval_loss", "x": eval_loss_x, "y": eval_loss_y},
                ],
            )
            created.append(path)

    if dataset_profile_path:
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

        labels = []
        values = []
        for split_name, stats in dataset_profile.items():
            labels.extend([f"{split_name}_min_input", f"{split_name}_max_input"])
            values.extend([stats.get("min_input_tokens", 0), stats.get("max_input_tokens", 0)])
        path = os.path.join(output_dir, "sft_input_token_range.png")
        save_bar_plot(path, "SFT Input Token Range", "Metric", "Tokens", labels, values)
        created.append(path)

    if summary_path:
        summary = read_json(summary_path)
        labels = ["train_rows", "validation_rows", "final_global_step"]
        values = [summary.get("train_rows", 0), summary.get("validation_rows", 0), summary.get("final_global_step", 0)]
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


def plot_grpo_from_sources(run_dir: str, repo_id: str, output_dir: str) -> list[str]:
    plot_data_dir = os.path.join(run_dir, "plot_data") if run_dir else ""
    train_rows = read_jsonl(resolve_local_or_hub_file(
        os.path.join(plot_data_dir, "train_metrics.jsonl") if plot_data_dir else "",
        repo_id,
        "train_metrics.jsonl",
        "plot_data",
    ))
    reward_rows = read_jsonl(resolve_local_or_hub_file(
        os.path.join(plot_data_dir, "reward_components.jsonl") if plot_data_dir else "",
        repo_id,
        "reward_components.jsonl",
        "plot_data",
    ))
    dataset_profile_path = resolve_local_or_hub_file(
        os.path.join(plot_data_dir, "dataset_profile.json") if plot_data_dir else "",
        repo_id,
        "dataset_profile.json",
        "plot_data",
    )
    summary_path = resolve_local_or_hub_file(
        os.path.join(plot_data_dir, "summary.json") if plot_data_dir else "",
        repo_id,
        "summary.json",
        "plot_data",
    )
    metrics_path = resolve_local_or_hub_file(
        os.path.join(run_dir, "metrics.json") if run_dir else "",
        repo_id,
        "metrics.json",
        "",
    )

    created = []

    trainer_series = []
    for key in ["loss", "reward", "kl"]:
        xs, ys = extract_metric(train_rows, key)
        if xs:
            trainer_series.append({"label": key, "x": xs, "y": ys})
    if trainer_series:
        path = os.path.join(output_dir, "grpo_training_metrics.png")
        save_line_plot(path, "GRPO Trainer Metrics", "Training Step", "Metric Value", trainer_series)
        created.append(path)

    reward_x, reward_y = extract_metric(train_rows, "reward")
    if reward_x:
        path = os.path.join(output_dir, "grpo_reward_only.png")
        save_line_plot(
            path,
            "GRPO Reward Curve",
            "Training Step",
            "Reward",
            [{"label": "reward", "x": reward_x, "y": reward_y}],
        )
        created.append(path)
        path = os.path.join(output_dir, "grpo_reward_smoothed.png")
        save_line_plot(
            path,
            "GRPO Reward Curve (Smoothed)",
            "Training Step",
            "Reward",
            [
                {"label": "reward", "x": reward_x, "y": reward_y},
                {"label": "reward_ma5", "x": reward_x, "y": moving_average(reward_y, window=5)},
            ],
        )
        created.append(path)
        path = os.path.join(output_dir, "grpo_reward_distribution.png")
        save_histogram(path, "GRPO Reward Distribution", "Reward", "Frequency", reward_y)
        created.append(path)

    loss_x, loss_y = extract_metric(train_rows, "loss")
    if loss_x:
        path = os.path.join(output_dir, "grpo_loss_only.png")
        save_line_plot(path, "GRPO Loss Curve", "Training Step", "Loss", [{"label": "loss", "x": loss_x, "y": loss_y}])
        created.append(path)

    kl_x, kl_y = extract_metric(train_rows, "kl")
    if kl_x:
        path = os.path.join(output_dir, "grpo_kl_only.png")
        save_line_plot(path, "GRPO KL Stability", "Training Step", "KL", [{"label": "kl", "x": kl_x, "y": kl_y}])
        created.append(path)
        if reward_x:
            path = os.path.join(output_dir, "grpo_reward_vs_kl.png")
            save_dual_axis_plot(
                path,
                "GRPO Reward and KL",
                "Training Step",
                "Reward",
                "KL",
                [{"label": "reward", "x": reward_x, "y": reward_y}],
                [{"label": "kl", "x": kl_x, "y": kl_y}],
            )
            created.append(path)

    learning_rate_x, learning_rate_y = extract_metric(train_rows, "learning_rate")
    if learning_rate_x:
        path = os.path.join(output_dir, "grpo_learning_rate.png")
        save_line_plot(
            path,
            "GRPO Learning Rate",
            "Training Step",
            "Learning Rate",
            [{"label": "learning_rate", "x": learning_rate_x, "y": learning_rate_y}],
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
        save_line_plot(path, "GRPO Reward Components", "Training Step", "Mean Reward", list(component_series.values()))
        created.append(path)
        for component_name, series in component_series.items():
            path = os.path.join(output_dir, f"grpo_component_{component_name}.png")
            save_line_plot(
                path,
                f"GRPO Component: {component_name}",
                "Training Step",
                "Mean Reward",
                [series],
            )
            created.append(path)

    quality_rows = read_json(metrics_path) if metrics_path else []
    if quality_rows:
        xs = list(range(1, len(quality_rows) + 1))
        series = []
        for key in ["valid_json_rate", "accuracy", "agent_accuracy", "unsafe_rate"]:
            series.append({"label": key, "x": xs, "y": [row.get(key, 0.0) for row in quality_rows]})
        path = os.path.join(output_dir, "grpo_quality_metrics.png")
        save_line_plot(path, "GRPO Quality Metrics", "Logged Batch", "Rate", series)
        created.append(path)
        path = os.path.join(output_dir, "grpo_quality_metrics_smoothed.png")
        save_line_plot(
            path,
            "GRPO Quality Metrics (Smoothed)",
            "Logged Batch",
            "Rate",
            [
                {"label": item["label"], "x": item["x"], "y": moving_average(item["y"], window=5)}
                for item in series
            ],
        )
        created.append(path)

        path = os.path.join(output_dir, "grpo_json_accuracy.png")
        save_line_plot(
            path,
            "GRPO JSON Validity and Accuracy",
            "Logged Batch",
            "Rate",
            [
                {"label": "valid_json_rate", "x": xs, "y": [row.get("valid_json_rate", 0.0) for row in quality_rows]},
                {"label": "accuracy", "x": xs, "y": [row.get("accuracy", 0.0) for row in quality_rows]},
                {"label": "agent_accuracy", "x": xs, "y": [row.get("agent_accuracy", 0.0) for row in quality_rows]},
            ],
        )
        created.append(path)

        path = os.path.join(output_dir, "grpo_unsafe_rate.png")
        save_line_plot(
            path,
            "GRPO Unsafe Action Rate",
            "Logged Batch",
            "Rate",
            [{"label": "unsafe_rate", "x": xs, "y": [row.get("unsafe_rate", 0.0) for row in quality_rows]}],
        )
        created.append(path)

    if dataset_profile_path:
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

        labels = ["min_prompt_chars", "avg_prompt_chars", "max_prompt_chars"]
        values = [
            dataset_profile.get("min_prompt_chars", 0),
            dataset_profile.get("avg_prompt_chars", 0),
            dataset_profile.get("max_prompt_chars", 0),
        ]
        path = os.path.join(output_dir, "grpo_prompt_length_profile.png")
        save_bar_plot(path, "GRPO Prompt Length Profile", "Metric", "Characters", labels, values)
        created.append(path)

    if summary_path:
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

    if created:
        caption_path = os.path.join(output_dir, "plot_captions.json")
        captions = {
            "grpo_reward_only.png": "Raw reward over GRPO training steps.",
            "grpo_reward_smoothed.png": "Reward with moving average to show trend more clearly.",
            "grpo_loss_only.png": "GRPO loss over training steps.",
            "grpo_kl_only.png": "KL stability during GRPO optimization.",
            "grpo_reward_vs_kl.png": "Reward trend compared against KL drift.",
            "grpo_reward_components.png": "All reward components logged during GRPO training.",
            "grpo_quality_metrics.png": "JSON validity, action accuracy, agent accuracy, and unsafe rate across training.",
            "grpo_json_accuracy.png": "Core behavioral quality metrics for the trained policy.",
            "grpo_unsafe_rate.png": "Unsafe action rate across logged GRPO batches.",
            "sft_train_loss.png": "Supervised fine-tuning loss over training steps.",
            "sft_eval_loss.png": "Validation loss during SFT.",
            "sft_train_vs_eval_loss.png": "Train versus validation loss during SFT.",
        }
        with open(caption_path, "w", encoding="utf-8") as handle:
            json.dump(captions, handle, indent=2)
        created.append(caption_path)

    return created


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate plot PNGs from SFT/GRPO training logs.")
    parser.add_argument("--sft-dir", default="", help="Path to the SFT output directory.")
    parser.add_argument("--grpo-dir", default="", help="Path to the GRPO output directory.")
    parser.add_argument("--sft-repo", default="", help="HF repo id for SFT logs, e.g. user/repo.")
    parser.add_argument("--grpo-repo", default="", help="HF repo id for GRPO logs, e.g. user/repo.")
    parser.add_argument("--output-dir", default="plots/generated", help="Where to save PNG plots.")
    args = parser.parse_args()

    if not args.sft_dir and not args.grpo_dir and not args.sft_repo and not args.grpo_repo:
        raise ValueError("Pass at least one of --sft-dir, --grpo-dir, --sft-repo, or --grpo-repo.")

    ensure_dir(args.output_dir)
    created_files = []
    if args.sft_dir or args.sft_repo:
        created_files.extend(plot_sft_from_sources(args.sft_dir, args.sft_repo, args.output_dir))
    if args.grpo_dir or args.grpo_repo:
        created_files.extend(plot_grpo_from_sources(args.grpo_dir, args.grpo_repo, args.output_dir))

    manifest = {
        "sft_dir": args.sft_dir,
        "grpo_dir": args.grpo_dir,
        "sft_repo": args.sft_repo,
        "grpo_repo": args.grpo_repo,
        "output_dir": args.output_dir,
        "created_files": created_files,
    }
    with open(os.path.join(args.output_dir, "plot_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
