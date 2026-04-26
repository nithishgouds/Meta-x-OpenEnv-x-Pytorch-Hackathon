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


def model_slug(model: str) -> str:
    lowered = model.lower()
    match = re.search(r"(\d+(?:\.\d+)?)b", lowered)
    if match:
        size = match.group(1).replace(".", "p")
        return f"qwen25-{size}b"
    cleaned = lowered.split("/")[-1]
    cleaned = re.sub(r"[^a-z0-9]+", "-", cleaned).strip("-")
    return cleaned or "model"


def stage_paths(model: str, hf_user: str) -> dict[str, str]:
    slug = model_slug(model)
    return {
        "smoke_output": f"outputs/smoke-{slug}",
        "sft_output": f"outputs/sft-{slug}",
        "grpo_output": f"outputs/grpo-{slug}",
        "smoke_repo": f"{hf_user}/opssim-{slug}-smoke-lora",
        "sft_repo": f"{hf_user}/opssim-{slug}-sft-lora",
        "grpo_repo": f"{hf_user}/opssim-{slug}-grpo-lora",
    }


def shell_prelude(repo_url: str, branch: str) -> str:
    return (
        "(command -v git >/dev/null 2>&1 || (apt-get update && apt-get install -y git)) && "
        f"git clone --branch {branch} {repo_url} app && "
        "cd app && "
        "pip install -U pip && "
        "pip install -e . && "
        "pip install -r requirements.txt"
    )


# def smoke_command(repo_url: str, branch: str, model: str, hf_user: str) -> str:
#     paths = stage_paths(model, hf_user)
#     return (
#         f"{shell_prelude(repo_url, branch)} && "
#         "python generate_sft_dataset.py --input tasks/cascade.json --output-dir data/generated --validate && "
#         f"python train_sft.py --model {model} "
#         "--train-file data/generated/sft_train.jsonl "
#         "--val-file data/generated/sft_val.jsonl "
#         f"--output-dir {paths['smoke_output']} "
#         f"--hub-model-id {paths['smoke_repo']} "
#         "--max-steps 10 --batch-size 1 --grad-accum 1 --max-seq-length 1024"
#     )

def smoke_command(repo_url: str, branch: str, model: str, hf_user: str) -> str:
    paths = stage_paths(model, hf_user)
    return (
        f"{shell_prelude(repo_url, branch)} && "

        # Dataset (run once)
        "python generate_sft_dataset.py --input tasks/cascade.json --output-dir data/generated --validate && "

        # SFT smoke
        f"python train_sft.py --model {model} "
        "--train-file data/generated/sft_train.jsonl "
        "--val-file data/generated/sft_val.jsonl "
        f"--output-dir {paths['smoke_output']} "
        f"--hub-model-id {paths['smoke_repo']} "
        "--max-steps 10 --batch-size 1 --grad-accum 1 --max-seq-length 512 && "

        # GRPO smoke (lightweight)
        f"python train_grpo.py --model {model} "
        f"--sft-adapter {paths['smoke_repo']} "
        "--input tasks/cascade.json "
        "--prompt-file data/generated/grpo_prompts.jsonl "
        f"--output-dir {paths['smoke_output']}-grpo "
        f"--hub-model-id {paths['smoke_repo']}-grpo "
        "--max-steps 5 --batch-size 2 --grad-accum 1 "
        "--num-generations 2 --max-completion-length 128 --max-prompt-length 512 "
        "--learning-rate 1e-5 --beta 0.01 --temperature 0.9"
    )


def sft_command(repo_url: str, branch: str, model: str, hf_user: str) -> str:
    paths = stage_paths(model, hf_user)
    return (
        f"{shell_prelude(repo_url, branch)} && "
        "python generate_sft_dataset.py --input tasks/cascade.json --output-dir data/generated --validate && "
        f"python train_sft.py --model {model} "
        "--train-file data/generated/sft_train.jsonl "
        "--val-file data/generated/sft_val.jsonl "
        f"--output-dir {paths['sft_output']} "
        f"--hub-model-id {paths['sft_repo']} "
        "--epochs 1 --batch-size 1 --grad-accum 8 --max-seq-length 1536 --save-steps 50"
    )


def grpo_command(repo_url: str, branch: str, model: str, hf_user: str) -> str:
    paths = stage_paths(model, hf_user)
    # Env-grounded GRPO: batch_size=8, num_generations=8 gives 1 prompt per
    # micro-batch with 8 completions for richer GRPO advantages. With
    # grad_accum=2, each optimizer step sees 2 prompts. lr=1e-5 and beta=0.01
    # allow meaningful policy updates (prior runs had KL≈0.0008 = frozen).
    return (
        f"{shell_prelude(repo_url, branch)} && "
        "python generate_sft_dataset.py --input tasks/cascade.json --output-dir data/generated --validate && "
        f"python train_grpo.py --model {model} "
        f"--sft-adapter {paths['sft_repo']} "
        "--input tasks/cascade.json "
        "--prompt-file data/generated/grpo_prompts.jsonl "
        f"--output-dir {paths['grpo_output']} "
        f"--hub-model-id {paths['grpo_repo']} "
        "--max-steps 300 --batch-size 8 --grad-accum 2 "
        "--num-generations 8 --max-completion-length 384 --max-prompt-length 1536 "
        "--learning-rate 1e-5 --beta 0.01 --temperature 0.9"
    )


def combined_command(repo_url: str, branch: str, model: str, hf_user: str) -> str:
    paths = stage_paths(model, hf_user)
    return (
        f"{shell_prelude(repo_url, branch)} && "
        "python generate_sft_dataset.py --input tasks/cascade.json --output-dir data/generated --validate && "
        f"python train_sft.py --model {model} "
        "--train-file data/generated/sft_train.jsonl "
        "--val-file data/generated/sft_val.jsonl "
        f"--output-dir {paths['sft_output']} "
        f"--hub-model-id {paths['sft_repo']} "
        "--epochs 1 --batch-size 1 --grad-accum 8 --max-seq-length 1536 --save-steps 50 && "
        "python generate_sft_dataset.py --input tasks/cascade.json --output-dir data/generated --validate && "
        f"python train_grpo.py --model {model} "
        f"--sft-adapter {paths['sft_repo']} "
        "--input tasks/cascade.json "
        "--prompt-file data/generated/grpo_prompts.jsonl "
        f"--output-dir {paths['grpo_output']} "
        f"--hub-model-id {paths['grpo_repo']} "
        "--max-steps 300 --batch-size 8 --grad-accum 2 "
        "--num-generations 8 --max-completion-length 384 --max-prompt-length 1536 "
        "--learning-rate 1e-5 --beta 0.01 --temperature 0.9"
    )


def get_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    return getpass.getpass("Paste HF token (hidden): ").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit OpsSim-AI training jobs to Hugging Face Jobs.")
    parser.add_argument("stage", choices=["smoke", "sft", "grpo", "all"])
    # a100-large = 80GB A100 at ~$2.50/h. Required for 1.5B/3B + frozen reference
    # + bf16 activations under GRPO with batch=8 and num_generations=8.
    parser.add_argument("--flavor", default="l40sx1")
    parser.add_argument("--timeout", default=None)
    parser.add_argument("--namespace", default=os.environ.get("HF_USER", "meancodi"))
    parser.add_argument("--hf-user", default=os.environ.get("HF_USER", "meancodi"))
    parser.add_argument("--model", default=os.environ.get("MODEL", "Qwen/Qwen2.5-3B-Instruct"))
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
    # GRPO with env-grounded reward, 300 steps, num_generations=4 on A100-large
    # is realistically a 2.5-3.5h job; budget 4h to absorb container/launch
    # overhead. Combined SFT+GRPO budgeted at 6h to stay inside the $25 cap
    # (4h GRPO + ~1h SFT + setup).
    default_timeouts = {
        "smoke": "20m",
        "sft": "2h",
        "grpo": "4h",
        "all": "6h",
    }
    timeout = args.timeout or default_timeouts[args.stage]
    script = commands[args.stage]()
    command = ["bash", "-lc", script]

    print("\nSubmitting HF Job")
    print(f"  stage: {args.stage}")
    print(f"  flavor: {args.flavor}")
    print(f"  timeout: {timeout}")
    print(f"  namespace: {args.namespace}")
    print(f"  model: {args.model}")
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
