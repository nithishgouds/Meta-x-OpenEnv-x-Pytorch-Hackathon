import argparse
import getpass
import os
import re
import textwrap

from huggingface_hub import run_job


IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"


def sanitize_tag(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9\-_=:]", "-", value.strip())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    return cleaned[:256] or "default"


def shell_prelude(repo_url: str, branch: str) -> str:
    return (
        f"git clone --branch {branch} {repo_url} app && "
        "cd app && "
        "pip install -U pip && "
        "pip install -e . && "
        "pip install -r requirements.txt"
    )


def smoke_command(repo_url: str, branch: str, model: str, hf_user: str) -> str:
    return (
        f"{shell_prelude(repo_url, branch)} && "
        "python generate_sft_dataset.py --input tasks/cascade.json --output-dir data/generated --validate && "
        f"python train_sft.py --model {model} "
        "--train-file data/generated/sft_train.jsonl "
        "--val-file data/generated/sft_val.jsonl "
        "--output-dir outputs/smoke-sft "
        f"--hub-model-id {hf_user}/opssim-qwen25-1p5b-smoke-lora "
        "--max-steps 10 --batch-size 1 --grad-accum 1 --max-seq-length 1024"
    )


def sft_command(repo_url: str, branch: str, model: str, hf_user: str) -> str:
    return (
        f"{shell_prelude(repo_url, branch)} && "
        "python generate_sft_dataset.py --input tasks/cascade.json --output-dir data/generated --validate && "
        f"python train_sft.py --model {model} "
        "--train-file data/generated/sft_train.jsonl "
        "--val-file data/generated/sft_val.jsonl "
        "--output-dir outputs/sft-qwen2.5-1.5b "
        f"--hub-model-id {hf_user}/opssim-qwen25-1p5b-sft-lora "
        "--epochs 1 --batch-size 1 --grad-accum 8 --max-seq-length 1536 --save-steps 50"
    )


def grpo_command(repo_url: str, branch: str, model: str, hf_user: str) -> str:
    return (
        f"{shell_prelude(repo_url, branch)} && "
        "python generate_sft_dataset.py --input tasks/cascade.json --output-dir data/generated --validate && "
        f"python train_grpo.py --model {model} "
        f"--sft-adapter {hf_user}/opssim-qwen25-1p5b-sft-lora "
        "--input tasks/cascade.json "
        "--prompt-file data/generated/grpo_prompts.jsonl "
        "--output-dir outputs/grpo-qwen2.5-1.5b "
        f"--hub-model-id {hf_user}/opssim-qwen25-1p5b-grpo-lora "
        "--max-steps 100 --batch-size 2 --grad-accum 2 "
        "--num-generations 2 --max-completion-length 512 --max-prompt-length 1536"
    )


def combined_command(repo_url: str, branch: str, model: str, hf_user: str) -> str:
    return (
        f"{shell_prelude(repo_url, branch)} && "
        "python generate_sft_dataset.py --input tasks/cascade.json --output-dir data/generated --validate && "
        f"python train_sft.py --model {model} "
        "--train-file data/generated/sft_train.jsonl "
        "--val-file data/generated/sft_val.jsonl "
        "--output-dir outputs/sft-qwen2.5-1.5b "
        f"--hub-model-id {hf_user}/opssim-qwen25-1p5b-sft-lora "
        "--epochs 1 --batch-size 1 --grad-accum 8 --max-seq-length 1536 --save-steps 50 && "
        "python generate_sft_dataset.py --input tasks/cascade.json --output-dir data/generated --validate && "
        f"python train_grpo.py --model {model} "
        f"--sft-adapter {hf_user}/opssim-qwen25-1p5b-sft-lora "
        "--input tasks/cascade.json "
        "--prompt-file data/generated/grpo_prompts.jsonl "
        "--output-dir outputs/grpo-qwen2.5-1.5b "
        f"--hub-model-id {hf_user}/opssim-qwen25-1p5b-grpo-lora "
        "--max-steps 100 --batch-size 2 --grad-accum 2 "
        "--num-generations 2 --max-completion-length 512 --max-prompt-length 1536"
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
    parser.add_argument("--namespace", default=os.environ.get("HF_USER", "meancodi"))
    parser.add_argument("--hf-user", default=os.environ.get("HF_USER", "meancodi"))
    parser.add_argument("--model", default=os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B-Instruct"))
    parser.add_argument("--repo-url", default=os.environ.get("REPO_URL", "https://github.com/nithishgouds/Meta-x-OpenEnv-x-Pytorch-Hackathon.git"))
    parser.add_argument("--branch", default=os.environ.get("REPO_BRANCH", "sandeep"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    commands = {
        "smoke": lambda: smoke_command(args.repo_url, args.branch, args.model, args.hf_user),
        "sft": lambda: sft_command(args.repo_url, args.branch, args.model, args.hf_user),
        "grpo": lambda: grpo_command(args.repo_url, args.branch, args.model, args.hf_user),
        "all": lambda: combined_command(args.repo_url, args.branch, args.model, args.hf_user),
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
        labels={
            "project": sanitize_tag("opssim-ai"),
            "stage": sanitize_tag(args.stage),
            "model": sanitize_tag(args.model),
        },
    )
    print("\nJob submitted.")
    print(f"  id: {job.id}")
    print(f"  url: {job.url}")
    print("\nOpen the URL in your browser for logs/status. Avoid repeatedly polling the CLI.")


if __name__ == "__main__":
    main()
