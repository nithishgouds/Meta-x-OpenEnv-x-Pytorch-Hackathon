import argparse
import getpass
import os
import textwrap

from huggingface_hub import run_job


REPO_URL = "https://github.com/nithishgouds/Meta-x-OpenEnv-x-Pytorch-Hackathon.git"
BRANCH = "sandeep"
IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"
HF_USER = "meancodi"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def shell_prelude() -> str:
    return (
        "apt-get update && apt-get install -y git && "
        f"git clone --branch {BRANCH} {REPO_URL} app && "
        "cd app && "
        "pip install -U pip && "
        "pip install -e ."
    )


def smoke_command() -> str:
    return (
        f"{shell_prelude()} && "
        "python generate_sft_dataset.py --input tasks/cascade.json --output-dir data/generated --validate && "
        f"python train_sft.py --model {MODEL} "
        "--train-file data/generated/sft_train.jsonl "
        "--val-file data/generated/sft_val.jsonl "
        "--output-dir outputs/smoke-sft "
        f"--hub-model-id {HF_USER}/opssim-qwen25-1p5b-smoke-lora "
        "--max-steps 10 --batch-size 1 --grad-accum 1 --max-seq-length 1024"
    )


def sft_command() -> str:
    return (
        f"{shell_prelude()} && "
        "python generate_sft_dataset.py --input tasks/cascade.json --output-dir data/generated --validate && "
        f"python train_sft.py --model {MODEL} "
        "--train-file data/generated/sft_train.jsonl "
        "--val-file data/generated/sft_val.jsonl "
        "--output-dir outputs/sft-qwen2.5-1.5b "
        f"--hub-model-id {HF_USER}/opssim-qwen25-1p5b-sft-lora "
        "--epochs 1 --batch-size 1 --grad-accum 8 --max-seq-length 1536"
    )


def grpo_command() -> str:
    return (
        f"{shell_prelude()} && "
        f"python train_grpo.py --model {MODEL} "
        f"--sft-adapter {HF_USER}/opssim-qwen25-1p5b-sft-lora "
        "--output-dir outputs/grpo-qwen2.5-1.5b "
        f"--hub-model-id {HF_USER}/opssim-qwen25-1p5b-grpo-lora "
        "--max-steps 100 --batch-size 1 --grad-accum 4 "
        "--num-generations 2 --max-completion-length 256"
    )


def combined_command() -> str:
    return (
        f"{shell_prelude()} && "
        "python generate_sft_dataset.py --input tasks/cascade.json --output-dir data/generated --validate && "
        f"python train_sft.py --model {MODEL} "
        "--train-file data/generated/sft_train.jsonl "
        "--val-file data/generated/sft_val.jsonl "
        "--output-dir outputs/sft-qwen2.5-1.5b "
        f"--hub-model-id {HF_USER}/opssim-qwen25-1p5b-sft-lora "
        "--epochs 1 --batch-size 1 --grad-accum 8 --max-seq-length 1536 && "
        f"python train_grpo.py --model {MODEL} "
        f"--sft-adapter {HF_USER}/opssim-qwen25-1p5b-sft-lora "
        "--output-dir outputs/grpo-qwen2.5-1.5b "
        f"--hub-model-id {HF_USER}/opssim-qwen25-1p5b-grpo-lora "
        "--max-steps 100 --batch-size 1 --grad-accum 4 "
        "--num-generations 2 --max-completion-length 256"
    )


def get_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    return getpass.getpass("Paste HF token (hidden): ").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit OpsSim-AI training jobs to Hugging Face Jobs.")
    parser.add_argument("stage", choices=["smoke", "sft", "grpo", "all"])
    parser.add_argument("--flavor", default="l4x1")
    parser.add_argument("--timeout", default=None)
    parser.add_argument("--namespace", default=HF_USER)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    commands = {
        "smoke": smoke_command,
        "sft": sft_command,
        "grpo": grpo_command,
        "all": combined_command,
    }
    default_timeouts = {
        "smoke": "20m",
        "sft": "1h",
        "grpo": "1h",
        "all": "2h",
    }
    timeout = args.timeout or default_timeouts[args.stage]
    script = commands[args.stage]()
    command = ["bash", "-lc", script]

    print("\nSubmitting HF Job")
    print(f"  stage: {args.stage}")
    print(f"  flavor: {args.flavor}")
    print(f"  timeout: {timeout}")
    print(f"  namespace: {args.namespace}")
    print("\nCommand preview:")
    print(textwrap.shorten(script, width=500, placeholder=" ..."))

    if args.dry_run:
        return

    token = get_token()
    job = run_job(
        image=IMAGE,
        command=command,
        flavor=args.flavor,
        timeout=timeout,
        namespace=args.namespace,
        token=token,
        secrets={"HF_TOKEN": token},
        labels={"project": "opssim-ai", "stage": args.stage, "model": "qwen25-1p5b"},
    )
    print("\nJob submitted.")
    print(f"  id: {job.id}")
    print(f"  url: {job.url}")
    print("\nOpen the URL in your browser for logs/status. Avoid repeatedly polling the CLI.")


if __name__ == "__main__":
    main()
