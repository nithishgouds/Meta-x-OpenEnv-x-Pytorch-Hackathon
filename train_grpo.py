import argparse
import os

import torch
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

from train import MultiAgentTrainingEnv, build_dataset, reward_func


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Budget-capped GRPO refinement for OpsSim-AI.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--sft-adapter", default="", help="Path or HF repo id for the SFT LoRA adapter.")
    parser.add_argument("--output-dir", default="outputs/grpo-qwen2.5-1.5b")
    parser.add_argument("--hub-model-id", default="", help="Optional HF Hub repo id for pushing the GRPO adapter.")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--max-tool-call-iterations", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()

    dataset = build_dataset()
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
        max_tool_calling_iterations=args.max_tool_call_iterations,
        log_completions=True,
        bf16=torch.cuda.is_available(),
        report_to="none",
        push_to_hub=bool(args.hub_model_id),
        hub_model_id=args.hub_model_id or None,
        chat_template_kwargs={"enable_thinking": False},
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=reward_func,
        environment_factory=MultiAgentTrainingEnv,
        peft_config=peft_config,
    )

    trainer.train()
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    if args.hub_model_id:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
