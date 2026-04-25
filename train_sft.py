import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)


def format_messages(tokenizer, messages: list[dict[str, str]], add_generation_prompt: bool = False) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    rendered = []
    for message in messages:
        rendered.append(f"{message['role'].upper()}:\n{message['content']}")
    if add_generation_prompt:
        rendered.append("ASSISTANT:\n")
    return "\n\n".join(rendered)


def resolve_precision(precision: str) -> tuple[torch.dtype, bool, bool]:
    if precision == "bf16":
        return torch.bfloat16, True, False
    if precision == "fp16":
        return torch.float16, False, True
    return torch.float32, False, False


def build_run_metadata(args: argparse.Namespace) -> dict[str, Any]:
    metadata = {"args": vars(args)}
    try:
        metadata["git_sha"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        metadata["git_sha"] = "unknown"
    return metadata


def ensure_plot_dirs(output_dir: str) -> dict[str, str]:
    plot_dir = os.path.join(output_dir, "plot_data")
    os.makedirs(plot_dir, exist_ok=True)
    return {
        "root": plot_dir,
        "train": os.path.join(plot_dir, "train_metrics.jsonl"),
        "eval": os.path.join(plot_dir, "eval_metrics.jsonl"),
        "summary": os.path.join(plot_dir, "summary.json"),
        "dataset": os.path.join(plot_dir, "dataset_profile.json"),
    }


def append_jsonl(path: str, payload: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def summarize_split(split) -> dict[str, float | int]:
    input_lengths = [len(row["input_ids"]) for row in split]
    target_lengths = [sum(1 for token in row["labels"] if token != -100) for row in split]
    return {
        "num_examples": len(split),
        "min_input_tokens": min(input_lengths) if input_lengths else 0,
        "max_input_tokens": max(input_lengths) if input_lengths else 0,
        "avg_input_tokens": round(sum(input_lengths) / len(input_lengths), 2) if input_lengths else 0.0,
        "min_target_tokens": min(target_lengths) if target_lengths else 0,
        "max_target_tokens": max(target_lengths) if target_lengths else 0,
        "avg_target_tokens": round(sum(target_lengths) / len(target_lengths), 2) if target_lengths else 0.0,
    }


class PlotMetricsCallback(TrainerCallback):
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
        target = self.paths["eval"] if any(key.startswith("eval_") for key in logs) else self.paths["train"]
        append_jsonl(target, payload)


def find_last_subsequence(sequence: list[int], subsequence: list[int]) -> int:
    if not subsequence or len(subsequence) > len(sequence):
        return -1
    for start in range(len(sequence) - len(subsequence), -1, -1):
        if sequence[start : start + len(subsequence)] == subsequence:
            return start
    return -1


def tokenize_example(example: dict[str, Any], tokenizer, max_seq_length: int) -> dict[str, Any]:
    messages = example["messages"]
    full_text = format_messages(tokenizer, messages, add_generation_prompt=False)
    assistant_content = messages[-1]["content"]
    assistant_char_start = full_text.rfind(assistant_content)
    if assistant_char_start < 0:
        raise ValueError("Assistant content could not be located in the rendered chat transcript.")

    full = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_length,
    )

    input_ids = list(full["input_ids"])
    attention_mask = list(full["attention_mask"])
    if tokenizer.eos_token_id is not None and (not input_ids or input_ids[-1] != tokenizer.eos_token_id):
        if len(input_ids) >= max_seq_length:
            input_ids = input_ids[: max_seq_length - 1]
            attention_mask = attention_mask[: max_seq_length - 1]
        input_ids.append(tokenizer.eos_token_id)
        attention_mask.append(1)

    assistant_ids = tokenizer(assistant_content, add_special_tokens=False)["input_ids"]
    assistant_start = find_last_subsequence(input_ids, assistant_ids)
    if assistant_start < 0:
        prefix_text = full_text[:assistant_char_start]
        prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
        assistant_start = min(len(prefix_ids), len(input_ids))

    labels = input_ids.copy()
    labels[:assistant_start] = [-100] * assistant_start

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


@dataclass
class SupervisedDataCollator:
    tokenizer: Any
    label_pad_token_id: int = -100

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_mask = []
        labels = []
        pad_id = self.tokenizer.pad_token_id

        for feature in features:
            pad_length = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [pad_id] * pad_length)
            attention_mask.append(feature["attention_mask"] + [0] * pad_length)
            labels.append(feature["labels"] + [self.label_pad_token_id] * pad_length)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def load_jsonl_dataset(train_file: str, val_file: str | None):
    data_files = {"train": train_file}
    if val_file and os.path.isfile(val_file):
        data_files["validation"] = val_file
    return load_dataset("json", data_files=data_files)


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA SFT for OpsSim-AI Incident Commander policy.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--train-file", default="data/generated/sft_train.jsonl")
    parser.add_argument("--val-file", default="data/generated/sft_val.jsonl")
    parser.add_argument("--output-dir", default="outputs/sft-qwen2.5-1.5b")
    parser.add_argument("--hub-model-id", default="", help="Optional HF Hub repo id for pushing the adapter.")
    parser.add_argument("--max-seq-length", type=int, default=1536)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.isfile(args.train_file):
        raise FileNotFoundError(f"Training file not found: {args.train_file}")

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype, use_bf16, use_fp16 = resolve_precision(args.precision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("LoRA adapter has no trainable parameters; check target_modules for this model.")
    model.print_trainable_parameters()

    dataset = load_jsonl_dataset(args.train_file, args.val_file)
    tokenized = dataset.map(
        lambda example: tokenize_example(example, tokenizer, args.max_seq_length),
        remove_columns=dataset["train"].column_names,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    plot_paths = ensure_plot_dirs(args.output_dir)
    with open(os.path.join(args.output_dir, "training_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(build_run_metadata(args), handle, indent=2)
    dataset_profile = {"train": summarize_split(tokenized["train"])}
    if "validation" in tokenized:
        dataset_profile["validation"] = summarize_split(tokenized["validation"])
    with open(plot_paths["dataset"], "w", encoding="utf-8") as handle:
        json.dump(dataset_profile, handle, indent=2)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="steps" if "validation" in tokenized else "no",
        eval_steps=args.save_steps if "validation" in tokenized else None,
        bf16=use_bf16 and torch.cuda.is_available(),
        fp16=use_fp16 and torch.cuda.is_available(),
        gradient_checkpointing=True,
        report_to="none",
        push_to_hub=bool(args.hub_model_id),
        hub_model_id=args.hub_model_id or None,
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        data_collator=SupervisedDataCollator(tokenizer),
        callbacks=[PlotMetricsCallback(plot_paths)],
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.save_state()
    summary = {
        "train_rows": len(tokenized["train"]),
        "validation_rows": len(tokenized["validation"]) if "validation" in tokenized else 0,
        "final_global_step": trainer.state.global_step,
        "final_epoch": trainer.state.epoch,
        "log_history_entries": len(trainer.state.log_history),
        "best_metric": trainer.state.best_metric,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
    }
    with open(plot_paths["summary"], "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if args.hub_model_id:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
