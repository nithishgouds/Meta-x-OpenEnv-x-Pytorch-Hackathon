import argparse
import json
import os
import re
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

from generate_sft_dataset import generate_examples_for_scenario, load_scenarios


def build_model_or_id(model_id: str, sft_adapter: str):
    if not sft_adapter:
        return model_id

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    return PeftModel.from_pretrained(model, sft_adapter, is_trainable=True)


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


def build_grpo_dataset(input_path: str, max_contrast_per_step: int) -> Dataset:
    scenarios = load_scenarios(input_path)
    rows = []
    for scenario in scenarios:
        examples = generate_examples_for_scenario(scenario, max_contrast_per_step=max_contrast_per_step, rng=__import__("random").Random(42))
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


def action_reward(completions, gold_action, gold_target_agent, unsafe_actions, **kwargs):
    rewards = []
    for completion, expected_action, expected_agent, unsafe in zip(
        completions, gold_action, gold_target_agent, unsafe_actions
    ):
        payload = parse_json_object(extract_text(completion))
        if payload is None:
            rewards.append(-0.8)
            continue

        reward = 0.0
        action = str(payload.get("next_action", ""))
        target_agent = str(payload.get("target_agent", ""))
        plan = payload.get("plan")
        confidence = payload.get("confidence")

        reward += 0.25
        if action == expected_action:
            reward += 1.0
        elif action in set(unsafe or []):
            reward -= 1.0
        else:
            reward -= 0.2

        if target_agent == expected_agent:
            reward += 0.35
        elif target_agent:
            reward -= 0.15

        if isinstance(plan, list):
            reward += 0.15
        if payload.get("analysis"):
            reward += 0.1
        if payload.get("reasoning"):
            reward += 0.1
        try:
            confidence_value = float(confidence)
            if 0.0 <= confidence_value <= 1.0:
                reward += 0.05
        except (TypeError, ValueError):
            reward -= 0.05

        rewards.append(reward)
    return rewards


def main() -> None:
    parser = argparse.ArgumentParser(description="Budget-capped JSON-action GRPO refinement for OpsSim-AI.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--sft-adapter", default="", help="Path or HF repo id for the SFT LoRA adapter.")
    parser.add_argument("--input", default="tasks/cascade.json")
    parser.add_argument("--output-dir", default="outputs/grpo-qwen2.5-1.5b")
    parser.add_argument("--hub-model-id", default="", help="Optional HF Hub repo id for pushing the GRPO adapter.")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-contrast-per-step", type=int, default=2)
    args = parser.parse_args()

    dataset = build_grpo_dataset(args.input, args.max_contrast_per_step)
    model = build_model_or_id(args.model, args.sft_adapter)

    peft_config = None
    if not args.sft_adapter:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        log_completions=True,
        bf16=torch.cuda.is_available(),
        report_to="none",
        push_to_hub=bool(args.hub_model_id),
        hub_model_id=args.hub_model_id or None,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=action_reward,
        peft_config=peft_config,
    )

    trainer.train()
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    if args.hub_model_id:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
