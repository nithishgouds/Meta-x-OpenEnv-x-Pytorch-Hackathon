# HF-Only SFT + GRPO Runbook

This runbook keeps `env.py` and `multi_agent.py` unchanged. The training flow is:

1. Generate SFT/GRPO data from `tasks/cascade.json`.
2. SFT `Qwen/Qwen2.5-1.5B-Instruct` with LoRA.
3. Run a short GRPO refinement from the SFT adapter.
4. Push adapters to Hugging Face Hub before the job exits.

## Budget Target

Use one `L4 24GB` job where possible. For a `$10` ceiling:

- Smoke/data generation: `$1-2`
- SFT: `$4-5`
- GRPO: `$2-3`
- Eval/rerun buffer: `$1`

Stop early if setup burns more than `$2`, SFT cannot emit valid JSON, or GRPO reward is flat.

## Required Inputs

Set these as HF Job secrets/env vars, not in git:

- `HF_TOKEN`: token with read/write access to your account.
- `HF_USER`: your Hugging Face username or org.

Choose repo names:

- SFT adapter: `$HF_USER/opssim-qwen25-1p5b-sft-lora`
- GRPO adapter: `$HF_USER/opssim-qwen25-1p5b-grpo-lora`

## One-Shot Budget Command

Preferred: use the Python submitter instead of raw CLI quoting:

```powershell
$env:HF_TOKEN = Read-Host "Paste HF token"
python submit_hf_job.py smoke
```

If the smoke job succeeds:

```powershell
python submit_hf_job.py sft
python submit_hf_job.py grpo
```

The script prints a job URL. Open that URL in the browser for logs/status instead of repeatedly polling the CLI.

Raw CLI equivalent, if needed:

```powershell
hf jobs run --flavor l4x1 --timeout 2h --secrets HF_TOKEN `
  pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel -- bash -lc 'git clone --branch sandeep https://github.com/nithishgouds/Meta-x-OpenEnv-x-Pytorch-Hackathon.git app && cd app && pip install -U pip && pip install -e . && python generate_sft_dataset.py --input tasks/cascade.json --output-dir data/generated --validate && python train_sft.py --model Qwen/Qwen2.5-1.5B-Instruct --train-file data/generated/sft_train.jsonl --val-file data/generated/sft_val.jsonl --output-dir outputs/sft-qwen2.5-1.5b --hub-model-id meancodi/opssim-qwen25-1p5b-sft-lora --epochs 1 --batch-size 1 --grad-accum 8 --max-seq-length 1536 --save-steps 50 && python train_grpo.py --model Qwen/Qwen2.5-1.5B-Instruct --sft-adapter meancodi/opssim-qwen25-1p5b-sft-lora --output-dir outputs/grpo-qwen2.5-1.5b --hub-model-id meancodi/opssim-qwen25-1p5b-grpo-lora --max-steps 100 --batch-size 1 --grad-accum 4 --num-generations 2 --max-completion-length 256'
```

## Safer Two-Stage Commands

If you want a hard checkpoint between SFT and GRPO, run SFT first:

```powershell
hf jobs run --flavor l4x1 --timeout 1h --secrets HF_TOKEN `
  pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel -- bash -lc 'git clone --branch sandeep https://github.com/nithishgouds/Meta-x-OpenEnv-x-Pytorch-Hackathon.git app && cd app && pip install -U pip && pip install -e . && python generate_sft_dataset.py --input tasks/cascade.json --output-dir data/generated --validate && python train_sft.py --model Qwen/Qwen2.5-1.5B-Instruct --train-file data/generated/sft_train.jsonl --val-file data/generated/sft_val.jsonl --output-dir outputs/sft-qwen2.5-1.5b --hub-model-id meancodi/opssim-qwen25-1p5b-sft-lora --epochs 1 --batch-size 1 --grad-accum 8 --max-seq-length 1536'
```

Then run the short GRPO job:

```powershell
hf jobs run --flavor l4x1 --timeout 1h --secrets HF_TOKEN `
  pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel -- bash -lc 'git clone --branch sandeep https://github.com/nithishgouds/Meta-x-OpenEnv-x-Pytorch-Hackathon.git app && cd app && pip install -U pip && pip install -e . && python train_grpo.py --model Qwen/Qwen2.5-1.5B-Instruct --sft-adapter meancodi/opssim-qwen25-1p5b-sft-lora --output-dir outputs/grpo-qwen2.5-1.5b --hub-model-id meancodi/opssim-qwen25-1p5b-grpo-lora --max-steps 100 --batch-size 1 --grad-accum 4 --num-generations 2 --max-completion-length 256'
```

## Output Locations

- Ephemeral HF job disk:
  - `data/generated/sft_train.jsonl`
  - `data/generated/sft_val.jsonl`
  - `data/generated/grpo_prompts.jsonl`
  - `outputs/...`
- Persistent HF Hub:
  - SFT LoRA adapter repo
  - GRPO LoRA adapter repo

If an artifact is not pushed to Hub, assume it can disappear when the job ends.
