import argparse
import json
import os
from dataclasses import dataclass
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
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


def tokenize_example(example: dict[str, Any], tokenizer, max_seq_length: int) -> dict[str, Any]:
    messages = example["messages"]
    prompt_messages = messages[:-1]

    prompt_text = format_messages(tokenizer, prompt_messages, add_generation_prompt=True)
    full_text = format_messages(tokenizer, messages, add_generation_prompt=False)

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_length,
    )

    input_ids = full["input_ids"]
    labels = input_ids.copy()
    prompt_len = min(len(prompt_ids), len(labels))
    labels[:prompt_len] = [-100] * prompt_len

    return {
        "input_ids": input_ids,
        "attention_mask": full["attention_mask"],
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
    dataset = load_dataset("json", data_files=data_files)
    return dataset


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
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    if not os.path.isfile(args.train_file):
        raise FileNotFoundError(f"Training file not found: {args.train_file}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if args.bf16 and torch.cuda.is_available() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    dataset = load_jsonl_dataset(args.train_file, args.val_file)
    tokenized = dataset.map(
        lambda example: tokenize_example(example, tokenizer, args.max_seq_length),
        remove_columns=dataset["train"].column_names,
    )

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
        bf16=args.bf16 and torch.cuda.is_available(),
        fp16=args.fp16 and torch.cuda.is_available(),
        gradient_checkpointing=True,
        report_to="none",
        push_to_hub=bool(args.hub_model_id),
        hub_model_id=args.hub_model_id or None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        data_collator=SupervisedDataCollator(tokenizer),
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.hub_model_id:
        trainer.push_to_hub()
        with open(os.path.join(args.output_dir, "training_metadata.json"), "w", encoding="utf-8") as handle:
            json.dump(vars(args), handle, indent=2)


if __name__ == "__main__":
    main()
