import argparse
import json
import os
import random
import re
import subprocess
from datetime import datetime
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, TrainerCallback, set_seed
from trl import GRPOConfig, GRPOTrainer

from generate_sft_dataset import compute_file_hash, generate_examples_for_scenario, load_scenarios

metrics_log = []
latest_reward_snapshot: dict[str, Any] = {}


def ensure_plot_dirs(output_dir: str) -> dict[str, str]:
    plot_dir = os.path.join(output_dir, "plot_data")
    os.makedirs(plot_dir, exist_ok=True)
    return {
        "root": plot_dir,
        "train": os.path.join(plot_dir, "train_metrics.jsonl"),
        "reward": os.path.join(plot_dir, "reward_components.jsonl"),
        "samples": os.path.join(plot_dir, "completion_samples.jsonl"),
        "summary": os.path.join(plot_dir, "summary.json"),
        "dataset": os.path.join(plot_dir, "dataset_profile.json"),
    }


def append_jsonl(path: str, payload: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def summarize_prompt_dataset(dataset: Dataset) -> dict[str, float | int]:
    prompt_lengths = []
    unsafe_counts = []
    scenario_ids = set()
    for row in dataset:
        prompt_lengths.append(len(row["prompt"][0]["content"]))
        unsafe_counts.append(len(row.get("unsafe_actions", [])))
        scenario_ids.add(row.get("scenario_id", "unknown"))
    return {
        "num_examples": len(dataset),
        "num_scenarios": len(scenario_ids),
        "min_prompt_chars": min(prompt_lengths) if prompt_lengths else 0,
        "max_prompt_chars": max(prompt_lengths) if prompt_lengths else 0,
        "avg_prompt_chars": round(sum(prompt_lengths) / len(prompt_lengths), 2) if prompt_lengths else 0.0,
        "avg_unsafe_actions": round(sum(unsafe_counts) / len(unsafe_counts), 2) if unsafe_counts else 0.0,
    }


def update_reward_snapshot(name: str, rewards: list[float]) -> None:
    latest_reward_snapshot[name] = {
        "mean": round(sum(rewards) / len(rewards), 6) if rewards else 0.0,
        "min": round(min(rewards), 6) if rewards else 0.0,
        "max": round(max(rewards), 6) if rewards else 0.0,
        "count": len(rewards),
    }


class GRPOPlotMetricsCallback(TrainerCallback):
    def __init__(self, paths: dict[str, str]):
        self.paths = paths

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        payload = {
            "step": state.global_step,
            "epoch": float(state.epoch) if state.epoch is not None else None,
            **logs,
        }
        append_jsonl(self.paths["train"], payload)
        if latest_reward_snapshot:
            append_jsonl(
                self.paths["reward"],
                {
                    "step": state.global_step,
                    "epoch": float(state.epoch) if state.epoch is not None else None,
                    "timestamp": datetime.now().isoformat(),
                    "components": latest_reward_snapshot,
                },
            )


def resolve_precision(precision: str) -> torch.dtype:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.float32


def build_run_metadata(args: argparse.Namespace) -> dict[str, Any]:
    metadata = {"args": vars(args)}
    try:
        metadata["git_sha"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        metadata["git_sha"] = "unknown"
    if os.path.isfile(args.input):
        metadata["dataset_hash"] = compute_file_hash(args.input)
    if os.path.isfile(args.prompt_file):
        metadata["prompt_file_hash"] = compute_file_hash(args.prompt_file)
    return metadata


def build_model_and_peft(args: argparse.Namespace):
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=resolve_precision(args.precision),
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    if args.sft_adapter:
        return PeftModel.from_pretrained(model, args.sft_adapter, is_trainable=True), None

    model.enable_input_require_grads()
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    return model, peft_config


def extract_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
    if isinstance(completion, dict):
        return str(completion.get("content", completion))
    return str(completion)


def parse_json_object(text: str) -> dict[str, Any] | None:
    text = extract_text(text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def load_grpo_prompt_dataset(prompt_file: str) -> Dataset:
    rows = []
    with open(prompt_file, "r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            metadata = payload.get("metadata", {})
            rows.append({
                "prompt": payload["prompt"],
                "gold_action": metadata["gold_action"],
                "gold_target_agent": metadata["target_agent"],
                "unsafe_actions": [item["action"] for item in metadata.get("unsafe_actions", [])],
                "scenario_id": metadata["scenario_id"],
                "step_idx": metadata["step_idx"],
            })
    return Dataset.from_list(rows)


def build_grpo_dataset(input_path: str, max_contrast_per_step: int) -> Dataset:
    scenarios = load_scenarios(input_path)
    rows = []
    for scenario in scenarios:
        examples = generate_examples_for_scenario(
            scenario,
            max_contrast_per_step=max_contrast_per_step,
            rng=random.Random(42),
        )
        for example in examples:
            metadata = example["metadata"]
            if metadata["label_type"] != "gold_next_action":
                continue
            rows.append({
                "prompt": [{"role": "user", "content": example["messages"][0]["content"]}],
                "gold_action": metadata["gold_action"],
                "gold_target_agent": metadata["target_agent"],
                "unsafe_actions": [item["action"] for item in metadata.get("unsafe_actions", [])],
                "scenario_id": metadata["scenario_id"],
                "step_idx": metadata["step_idx"],
            })
    return Dataset.from_list(rows)


def payloads_from_completions(completions) -> list[dict[str, Any] | None]:
    return [parse_json_object(extract_text(completion)) for completion in completions]


def parse_reward(completions, **kwargs):
    rewards = [0.4 if payload is not None else -1.0 for payload in payloads_from_completions(completions)]
    update_reward_snapshot("parse_reward", rewards)
    return rewards


def action_match_reward(completions, gold_action, **kwargs):
    rewards = []
    for payload, expected_action in zip(payloads_from_completions(completions), gold_action):
        if payload is None:
            rewards.append(0.0)
            continue
        rewards.append(1.0 if str(payload.get("next_action", "")) == expected_action else -0.25)
    update_reward_snapshot("action_match_reward", rewards)
    return rewards


def agent_match_reward(completions, gold_target_agent, **kwargs):
    rewards = []
    for payload, expected_agent in zip(payloads_from_completions(completions), gold_target_agent):
        if payload is None:
            rewards.append(0.0)
            continue
        rewards.append(0.35 if str(payload.get("target_agent", "")) == expected_agent else -0.1)
    update_reward_snapshot("agent_match_reward", rewards)
    return rewards


def unsafe_action_penalty(completions, unsafe_actions, **kwargs):
    rewards = []
    for payload, unsafe in zip(payloads_from_completions(completions), unsafe_actions):
        if payload is None:
            rewards.append(0.0)
            continue
        rewards.append(-1.0 if str(payload.get("next_action", "")) in set(unsafe or []) else 0.1)
    update_reward_snapshot("unsafe_action_penalty", rewards)
    return rewards


def format_reward(completions, **kwargs):
    required_keys = {"analysis", "plan", "next_action", "target_agent", "reasoning", "confidence"}
    rewards = []
    for payload in payloads_from_completions(completions):
        if payload is None:
            rewards.append(0.0)
            continue
        rewards.append(0.15 if required_keys.issubset(payload.keys()) else -0.1)
    update_reward_snapshot("format_reward", rewards)
    return rewards


def log_reward_metrics(completions, gold_action, gold_target_agent, unsafe_actions, **kwargs):
    payloads = payloads_from_completions(completions)
    step_metrics = []
    for payload, expected_action, expected_agent, unsafe in zip(
        payloads, gold_action, gold_target_agent, unsafe_actions
    ):
        step_metrics.append({
            "valid_json": payload is not None,
            "correct_action": bool(payload and str(payload.get("next_action", "")) == expected_action),
            "correct_agent": bool(payload and str(payload.get("target_agent", "")) == expected_agent),
            "unsafe_action": bool(payload and str(payload.get("next_action", "")) in set(unsafe or [])),
        })

    metrics_log.append({
        "timestamp": datetime.now().isoformat(),
        "valid_json_rate": sum(m["valid_json"] for m in step_metrics) / len(step_metrics),
        "accuracy": sum(m["correct_action"] for m in step_metrics) / len(step_metrics),
        "agent_accuracy": sum(m["correct_agent"] for m in step_metrics) / len(step_metrics),
        "unsafe_rate": sum(m["unsafe_action"] for m in step_metrics) / len(step_metrics),
    })
    update_reward_snapshot("quality_metrics", [
        metrics_log[-1]["valid_json_rate"],
        metrics_log[-1]["accuracy"],
        metrics_log[-1]["agent_accuracy"],
        1.0 - metrics_log[-1]["unsafe_rate"],
    ])
    return [0.0] * len(completions)


def main() -> None:
    parser = argparse.ArgumentParser(description="Budget-capped JSON-action GRPO refinement for OpsSim-AI.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--sft-adapter", default="", help="Path or HF repo id for the SFT LoRA adapter.")
    parser.add_argument("--input", default="tasks/cascade.json")
    parser.add_argument("--prompt-file", default="data/generated/grpo_prompts.jsonl")
    parser.add_argument("--output-dir", default="outputs/grpo-qwen2.5-1.5b")
    parser.add_argument("--hub-model-id", default="", help="Optional HF Hub repo id for pushing the GRPO adapter.")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--max-prompt-length", type=int, default=1536)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-contrast-per-step", type=int, default=2)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.batch_size % args.num_generations != 0:
        raise ValueError("--batch-size must be divisible by --num-generations for GRPOTrainer.")

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    plot_paths = ensure_plot_dirs(args.output_dir)
    with open(os.path.join(args.output_dir, "run_meta.json"), "w", encoding="utf-8") as handle:
        json.dump(build_run_metadata(args), handle, indent=2)

    if os.path.isfile(args.prompt_file):
        dataset = load_grpo_prompt_dataset(args.prompt_file)
    else:
        dataset = build_grpo_dataset(args.input, args.max_contrast_per_step)
    with open(plot_paths["dataset"], "w", encoding="utf-8") as handle:
        json.dump(summarize_prompt_dataset(dataset), handle, indent=2)

    model, peft_config = build_model_and_peft(args)

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        log_completions=True,
        bf16=args.precision == "bf16" and torch.cuda.is_available(),
        fp16=args.precision == "fp16" and torch.cuda.is_available(),
        gradient_checkpointing=True,
        report_to="none",
        push_to_hub=bool(args.hub_model_id),
        hub_model_id=args.hub_model_id or None,
        seed=args.seed,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=[
            parse_reward,
            action_match_reward,
            agent_match_reward,
            unsafe_action_penalty,
            format_reward,
            log_reward_metrics,
        ],
        peft_config=peft_config,
        callbacks=[GRPOPlotMetricsCallback(plot_paths)],
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics_log, handle, indent=2)
    summary = {
        "num_examples": len(dataset),
        "final_global_step": trainer.state.global_step,
        "final_epoch": trainer.state.epoch,
        "log_history_entries": len(trainer.state.log_history),
        "last_quality_metrics": metrics_log[-1] if metrics_log else {},
    }
    with open(plot_paths["summary"], "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    for sample in dataset.select(range(min(10, len(dataset)))):
        append_jsonl(
            plot_paths["samples"],
            {
                "scenario_id": sample.get("scenario_id"),
                "step_idx": sample.get("step_idx"),
                "gold_action": sample.get("gold_action"),
                "gold_target_agent": sample.get("gold_target_agent"),
                "unsafe_actions": sample.get("unsafe_actions", []),
                "prompt_preview": sample["prompt"][0]["content"][:500],
            },
        )
    if args.hub_model_id:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
