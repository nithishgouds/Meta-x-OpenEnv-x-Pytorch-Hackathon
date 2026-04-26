# OpsSim-AI Training Pipeline — End-to-End Guide

This document walks you through running the full **SFT → GRPO** training pipeline, verifying it works, and interpreting the output plots.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Pipeline Overview](#2-pipeline-overview)
3. [Quick Smoke Test (verify E2E works)](#3-quick-smoke-test)
4. [Full Training Run](#4-full-training-run)
5. [Running on Hugging Face Jobs (cloud GPU)](#5-running-on-hugging-face-jobs)
6. [Output Artifacts](#6-output-artifacts)
7. [How to Read the Training Plots](#7-how-to-read-the-training-plots)
8. [Inference with Trained Model](#8-inference-with-trained-model)
9. [Training Run Analysis (Qwen2.5-3B, 300 steps)](#9-training-run-analysis-qwen25-3b-300-steps)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Prerequisites

### Hardware

| Stage | Minimum GPU | Recommended |
|-------|------------|-------------|
| SFT | 16 GB VRAM (T4) | 24 GB (A10/L4) |
| GRPO | 40 GB VRAM (A100-40G) | 80 GB (A100-80G / L40S) |

> GRPO requires more VRAM because it holds the policy model + frozen reference model + 8 completions per prompt simultaneously.

### Software

```bash
# Python 3.11+
pip install -e .
pip install -r requirements.txt
```

Key dependencies: `torch`, `transformers==4.51.3`, `trl==0.17.0`, `peft==0.15.2`, `accelerate==1.6.0`, `datasets==3.5.0`, `matplotlib==3.9.2`.

### Environment Variables (for HF Hub push)

```bash
export HF_TOKEN="hf_..."          # Hugging Face write token
export HF_USER="your-username"    # Your HF username (for repo naming)
```

---

## 2. Pipeline Overview

The training pipeline has **3 stages** that run sequentially:

```
┌─────────────────────┐     ┌──────────────────┐     ┌──────────────────────┐
│ 1. Dataset Gen      │────▶│ 2. SFT Training  │────▶│ 3. GRPO Training     │
│ generate_sft_dataset│     │ train_sft.py     │     │ train_grpo.py        │
│                     │     │                  │     │                      │
│ cascade.json ──▶    │     │ Supervised fine- │     │ RL fine-tuning with  │
│ sft_train.jsonl     │     │ tuning: teaches  │     │ env-grounded reward: │
│ sft_val.jsonl       │     │ the model the    │     │ teaches the model to │
│ grpo_prompts.jsonl  │     │ output format &  │     │ maximize the DevOps  │
│                     │     │ basic competence │     │ environment's score  │
└─────────────────────┘     └──────────────────┘     └──────────────────────┘
```

**SFT** (Supervised Fine-Tuning) teaches the model to produce valid JSON responses in the correct format. **GRPO** (Group Relative Policy Optimization) then improves the model's *decisions* by running each predicted action through the DevOps simulator and using the resulting reward signal.

---

## 3. Quick Smoke Test

**Goal:** Verify the entire pipeline runs end-to-end without errors, loss decreases, and reward signals are positive for correct actions. This uses minimal steps so it finishes in ~5-10 minutes on a single GPU.

### Step 1: Generate the dataset

```bash
python generate_sft_dataset.py \
  --input tasks/cascade.json \
  --output-dir data/generated \
  --validate
```

This creates `data/generated/sft_train.jsonl`, `sft_val.jsonl`, and `grpo_prompts.jsonl`.

### Step 2: SFT smoke test (10 steps)

```bash
python train_sft.py 
  --model Qwen/Qwen2.5-3B-Instruct 
  --train-file data/generated/sft_train.jsonl 
  --val-file data/generated/sft_val.jsonl 
  --output-dir outputs/smoke-sft 
  --max-steps 10 
  --batch-size 1
  --grad-accum 1 
  --max-seq-length 1024 
  --logging-steps 1 
  --save-steps 10
```

**What to check:**
- It runs without errors
- Console shows loss values that decrease (even slightly) over 10 steps
- `outputs/smoke-sft/` contains `adapter_model.safetensors`

### Step 3: GRPO smoke test (10 steps)

```bash
python train_grpo.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --sft-adapter outputs/smoke-sft \
  --input tasks/cascade.json \
  --prompt-file data/generated/grpo_prompts.jsonl \
  --output-dir outputs/smoke-grpo \
  --max-steps 10 \
  --batch-size 8 \
  --grad-accum 1 \
  --num-generations 8 \
  --max-completion-length 384 \
  --max-prompt-length 1536 \
  --learning-rate 1e-5 \
  --beta 0.01 \
  --temperature 0.9 \
  --logging-steps 1 \
  --save-steps 10
```

**What to check:**
- Runs without errors
- Console logs show `reward` values (some should be positive)
- `loss` values appear (may fluctuate initially — this is fine for 10 steps)
- `outputs/smoke-grpo/` contains `adapter_model.safetensors` and a `plots/` folder with PNGs

### Verifying reward signal correctness

If you want to independently verify the reward design without running training, you can run this quick Python snippet:

```python
import json
from env import DevOpsEnv, AGENT_DOMAIN_MAP
from models import OpsSIMAction

DOMAIN_TO_AGENT = {d: a for a, d in AGENT_DOMAIN_MAP.items()}

env = DevOpsEnv(seed=42, max_steps=15)
env.reset()

# Gold action for cascade_001 step 1
_, reward, _, _ = env.step(OpsSIMAction(
    action_type="investigate_payment_service",
    agent="AppOps", target_agent="AppOps",
    ic_directive=True, supervisor_approved=True,
))
aq = reward.action_quality  # should be +0.2
print(f"Gold action quality: {aq:+.2f}")  # expect +0.20
```

---

## 4. Full Training Run

These are the production-quality commands for a complete training run.

### Step 1: Generate dataset

```bash
python generate_sft_dataset.py \
  --input tasks/cascade.json \
  --output-dir data/generated \
  --validate
```

### Step 2: SFT (full)

```bash
python train_sft.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --train-file data/generated/sft_train.jsonl \
  --val-file data/generated/sft_val.jsonl \
  --output-dir outputs/sft-qwen25-3b \
  --hub-model-id <your-hf-user>/opssim-qwen25-3b-sft-lora \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 8 \
  --max-seq-length 1536 \
  --save-steps 50 \
  --learning-rate 2e-4
```

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| `epochs` | 1 | Single pass is enough — SFT just teaches format, not decision-making |
| `batch-size × grad-accum` | 1 × 8 = 8 | Effective batch of 8 for stable gradients within VRAM limits |
| `learning-rate` | 2e-4 | Standard for LoRA SFT |
| `max-seq-length` | 1536 | Covers full prompt + response (prompts are ~800-1200 chars) |

**Duration:** ~30-60 min on A100, ~60-90 min on L40S.

### Step 3: GRPO (full)

```bash
python train_grpo.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --sft-adapter <your-hf-user>/opssim-qwen25-3b-sft-lora \
  --input tasks/cascade.json \
  --prompt-file data/generated/grpo_prompts.jsonl \
  --output-dir outputs/grpo-qwen25-3b \
  --hub-model-id <your-hf-user>/opssim-qwen25-3b-grpo-lora \
  --max-steps 500 \
  --batch-size 8 \
  --grad-accum 2 \
  --num-generations 8 \
  --max-completion-length 384 \
  --max-prompt-length 1536 \
  --learning-rate 2e-5 \
  --beta 0.005 \
  --temperature 0.9 \
  --warmup-ratio 0.05 \
  --save-steps 100
```

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| `max-steps` | 500 | 500 × 8 completions = 4000 graded completions; reward still climbing at 300 in prior runs |
| `batch-size` | 8 | Must equal `num-generations` (all 8 completions for 1 prompt per micro-batch) |
| `num-generations` | 8 | 8 completions per prompt for rich GRPO advantages |
| `grad-accum` | 2 | Each optimizer step sees 2 prompts × 8 completions = 16 graded samples |
| `learning-rate` | 2e-5 | Higher than typical GRPO — prior run showed KL≈0.005 with lr=1e-5 (too conservative) |
| `beta` | 0.005 | Reduced from 0.01 to allow more policy divergence from SFT |
| `temperature` | 0.9 | Encourages diverse completions (diverse actions = meaningful GRPO advantages) |
| `warmup-ratio` | 0.05 | Smooths noisy early gradients for first 5% of training |

**Duration:** ~3-5 hours on L40S, ~2-3 hours on A100.

> **Note:** All commands above use the 3B model. For a lighter run on smaller GPUs, replace `3B` with `1.5B` in model names and repo slugs.

---

## 5. Running on Hugging Face Jobs

The simplest way to run the full pipeline with a cloud GPU:

### Smoke test (verify E2E)

```bash
python submit_hf_job.py smoke --dry-run            # preview the command
python submit_hf_job.py smoke --flavor l40sx1       # submit
```

### Full pipeline (SFT + GRPO combined)

```bash
python submit_hf_job.py all --flavor l40sx1
```

This runs SFT + GRPO end-to-end with the updated params (`lr=2e-5`, `beta=0.005`, 500 steps, `warmup=0.05`, `unsafe_penalty` re-added). Timeout is 8h, estimated cost **~$10-14** on L40S.

### GRPO only (skip SFT)

If you already have a working SFT adapter from a prior run (e.g. `meancodi/opssim-qwen25-3b-sft-lora`):

```bash
python submit_hf_job.py grpo --flavor l40sx1
```

This runs only GRPO, reusing the existing SFT adapter. Timeout 6h, estimated cost **~$7-9** on L40S.

### Individual stages

```bash
python submit_hf_job.py sft  --flavor l40sx1 --hf-user <your-username>
python submit_hf_job.py grpo --flavor l40sx1 --hf-user <your-username>
```

Available stages: `smoke`, `sft`, `grpo`, `all` (combined SFT + GRPO).

| Flag | Default | Description |
|------|---------|-------------|
| `--flavor` | `l40sx1` | GPU type (L40S 48GB — sufficient for 3B) |
| `--model` | `Qwen/Qwen2.5-3B-Instruct` | Base model |
| `--hf-user` | `meancodi` | Your HF username (repos are created as `<user>/opssim-...`) |
| `--branch` | `sandeep` | Git branch to clone |
| `--timeout` | auto (20m/2h/6h/8h) | Max runtime |
| `--dry-run` | off | Print command without submitting |

### Cost estimates

| Stage | GPU | Time | Cost |
|-------|-----|------|------|
| Smoke | L40S | ~10 min | ~$0.30 |
| SFT only | L40S | ~1-1.5h | ~$2-3 |
| GRPO only (500 steps) | L40S | ~4-5h | ~$7-9 |
| All (SFT + GRPO) | L40S | ~5-6.5h | ~$10-14 |

> L40S is $1.80/h on HF Jobs.

The job automatically:
1. Clones the repo & installs dependencies
2. Generates the dataset
3. Runs training
4. Uploads the adapter + plots to Hugging Face Hub

---

## 6. Output Artifacts

After training completes, the output directory contains:

```
outputs/grpo-qwen25-3b/
├── adapter_model.safetensors    # LoRA weights (the trained model)
├── adapter_config.json          # LoRA config
├── tokenizer.json               # Tokenizer
├── metrics.json                 # Per-step quality metrics
├── run_meta.json                # Run configuration
├── final_metrics.json           # Summary numbers
├── train.log                    # Full console log
├── plot_data/
│   ├── train_metrics.jsonl      # Raw training metrics per step
│   ├── reward_components.jsonl  # Env reward component breakdown
│   ├── quality_metrics.jsonl    # Accuracy, parse rate, etc.
│   └── summary.json             # Run summary
└── plots/                       # Auto-generated PNG plots (see §7)
    ├── grpo_training_metrics.png
    ├── grpo_reward_only.png
    ├── grpo_loss_only.png
    ├── grpo_kl_only.png
    ├── grpo_reward_smoothed.png
    ├── grpo_reward_components.png
    ├── grpo_reward_vs_kl.png
    ├── grpo_component_env_reward.png
    ├── grpo_component_env_action_quality.png
    ├── grpo_component_env_sequencing_reward.png
    ├── grpo_component_env_coordination_reward.png
    ├── grpo_component_env_success_reward.png
    ├── grpo_component_parse_penalty.png
    ├── grpo_component_format_penalty.png
    └── grpo_quality_metrics.png
```

---

## 7. How to Read the Training Plots

After training, the `plots/` directory contains ~15-20 PNG files. Here's what each one tells you and what to look for.

### 7.1 Primary Plots (check these first)

#### `grpo_training_metrics.png`

**What it shows:** Loss, reward, and KL divergence on a single chart.

**Healthy training looks like:**
- **Loss (blue):** Starts noisy, trends downward over 250 steps. May not be smooth — that's normal for RL.
- **Reward (orange):** Starts around -0.3 to 0.0, trends **upward** toward +0.3 to +0.6 by step 250. **This is the most important signal.**
- **KL (green):** Starts near 0, rises to 0.01-0.10 range. If KL stays flat at ~0.0 (≤0.001) the entire run, the policy is frozen and not learning — something is wrong.

**Red flags:**
- Reward is flat or trending downward → reward design issue
- KL is flat at ~0.0 → learning rate too low or beta too high
- Loss is NaN or exploding → precision or gradient issue

#### `grpo_reward_only.png`

**What it shows:** Just the mean reward per step, easier to read than the combined plot.

**Healthy:** Noisy but with a clear upward trend. The smoothed version (`grpo_reward_smoothed.png`) is easier to judge visually.

**Expected range:** Starts around -0.5 to 0.0, should reach +0.2 to +0.5 by end.

#### `grpo_loss_only.png`

**What it shows:** GRPO policy loss per step.

**Important:** GRPO loss behaves **differently from supervised loss**. In GRPO, loss = $-\mathbb{E}[\text{advantage} \times \log\pi_\theta(a|s)]$. Advantages are computed GROUP-relative (zero-mean within each group of `num_generations` completions). This means:
- Loss oscillates around 0 and does **NOT monotonically decrease** — this is expected!
- As the policy improves, all completions become similar → advantages shrink → loss stays near 0
- When curriculum shifts to harder scenarios, loss temporarily spikes
- The sign alternates: negative loss = increasing probability of good actions, positive = decreasing probability of bad actions

**The correct metric to judge GRPO training is the REWARD curve, not the loss curve.** If reward trends upward, training is working — regardless of what loss does.

#### `grpo_kl_only.png`

**What it shows:** KL divergence between current policy and the SFT reference.

**Healthy:** Gradually rises from ~0 to 0.01-0.10. This means the policy is exploring away from SFT.

**Concerning:**
- KL > 0.5 → policy has drifted too far, may be producing gibberish. Lower the learning rate or raise beta.
- KL ≈ 0.0 → policy is not moving at all. Raise the learning rate or lower beta.

### 7.2 Reward Breakdown Plots

#### `grpo_reward_smoothed.png`

Smoothed (exponential moving average) version of reward. **Best plot for judging overall training trend.** If this trends up, training is working.

#### `grpo_reward_vs_kl.png`

**What it shows:** Reward on Y-axis vs KL on X-axis. Shows the quality-exploration tradeoff.

**Healthy:** Moving right (more KL) and up (more reward) simultaneously. The ideal shape is a curve going up and to the right.

**Concerning:** Moving right without going up means the model is exploring but not finding better actions.

#### `grpo_reward_components.png`

**What it shows:** Breakdown of reward into env_reward, parse_penalty, format_penalty.

**Healthy:**
- `env_reward` (the main signal) should trend upward
- `parse_penalty` should trend toward 0 (fewer parse failures)
- `format_penalty` should be near 0 throughout

### 7.3 Environment Component Plots

These show individual components of the env reward. Each has a separate `grpo_component_env_*.png`:

| Plot | What it measures | Healthy trend |
|------|-----------------|---------------|
| `env_action_quality` | Was the action appropriate? | Trending up (toward +0.15 to +0.30) |
| `env_sequencing_reward` | Was the action in the right order? | Trending up (toward +0.10 to +0.15) |
| `env_coordination_reward` | Did the right domain-agent execute? | Trending up (toward +0.10 to +0.15) |
| `env_success_reward` | SLA pass (+2.0) or fail (-2.0) | Occasional spikes at +2.0 = great |
| `env_delta_health` | Did system health improve? | Should hover near 0 or slightly positive |
| `env_responsibility_penalty` | Wrong-agent violations | Should decrease (fewer violations over time) |

### 7.4 Quality Metrics Plot

#### `grpo_quality_metrics.png` / `grpo_component_quality_metrics.png`

**What it shows:** Accuracy, parse rate, agent accuracy, unsafe rate, env success rate.

**Healthy:**
- `valid_json_rate`: Should be >0.8 and trending toward 1.0
- `accuracy` (correct action): Should increase from ~0.1-0.3 toward 0.4-0.6
- `agent_accuracy`: Should increase
- `unsafe_rate`: Should decrease toward 0

### 7.5 Summary: Quick Health Check

After training finishes, check these plots in order:

| # | Plot | What to verify |
|---|------|---------------|
| 1 | `grpo_reward_smoothed.png` | **Reward trends upward** (most important signal) |
| 2 | `grpo_kl_only.png` | **KL rises to 0.01-0.10** (not stuck at 0) |
| 3 | `grpo_loss_only.png` | **Loss oscillates near 0** (GRPO loss does NOT need to decline — see §7.1) |
| 4 | `grpo_component_env_action_quality.png` | Action quality improving |
| 5 | `grpo_quality_metrics.png` | Accuracy increasing, unsafe_rate decreasing |
| 6 | `grpo_unsafe_rate.png` | Unsafe action rate declining over training |

If plots 1-2 look healthy, training is working correctly.

---

## 8. Inference with Trained Model

After training, test the model on a scenario:

```bash
python run_trained_inference.py \
  --base-model Qwen/Qwen2.5-3B-Instruct \
  --adapter <your-hf-user>/opssim-qwen25-3b-grpo-lora \
  --scenario cascade_001_checkout_meltdown \
  --step-idx 1
```

Or with a local adapter:

```bash
python run_trained_inference.py \
  --base-model Qwen/Qwen2.5-3B-Instruct \
  --adapter outputs/grpo-qwen25-3b \
  --scenario cascade_001_checkout_meltdown \
  --step-idx 1
```

Expected output: a JSON object with `next_action`, `target_agent`, `analysis`, `plan`, `reasoning`, and `confidence` fields. The model should predict `investigate_payment_service` for cascade_001 step 1.

---

## 9. Training Run Analysis (Qwen2.5-3B, 300 steps)

This section documents findings from the first full 3B GRPO run and the fixes applied.

### 9.1 What went well

| Metric | Observation |
|--------|-------------|
| **Reward (smoothed)** | Clear upward trend: 0.22 → 0.40 over 300 steps |
| **KL divergence** | Steadily rising: 0 → 0.005-0.008 (policy is learning) |
| **valid_json_rate** | ~97% throughout (model produces well-formed JSON) |
| **format_penalty** | Near-zero after step 200 (learned output schema) |
| **parse_penalty** | Rare spikes only (model consistently parses) |
| **env_action_quality** | Upward trend: 0.05 → 0.10 |
| **env_delta_health** | Upward trend: 0.03 → 0.07 |
| **env_coordination_reward** | Stabilized near ceiling (~0.14 of 0.15 max) |

### 9.2 Issues diagnosed

| Issue | Root Cause | Fix Applied |
|-------|-----------|-------------|
| **Loss not declining** | GRPO loss oscillates near 0 by design (group-relative advantages are zero-mean). This is NOT a bug. | Updated docs (§7) to explain this is expected. Reward is the correct metric. |
| **env_success_rate = 0.0** | `final_metrics.json` reports only the LAST batch's metrics. The `success_reward` plot shows periodic non-zero values, proving SLA passes DO happen. | Added cumulative quality metrics to `final_metrics.json` so the aggregate picture is visible. |
| **unsafe_rate ~15% (flat)** | `unsafe_penalty` was removed from `reward_funcs` in a prior fix to prevent double-counting. But the env's `action_quality` does NOT specifically penalize unsafe actions — it only rewards matching transition rules. The model had **zero direct gradient signal** to avoid unsafe actions. | Re-added `unsafe_penalty` at -0.3 (lighter than old -0.5). |
| **KL too low (~0.005)** | `lr=1e-5` and `beta=0.01` were too conservative — the KL penalty dominated small gradient updates. | Increased LR to `2e-5`, reduced beta to `0.005`, added `warmup_ratio=0.05`. |
| **accuracy plateaued ~55-60%** | 300 steps may not be enough; reward was still climbing. | Increased default to 500 steps. |

### 9.3 Reward function composition (current)

```
Total Reward = env_reward + parse_penalty + format_penalty + unsafe_penalty

env_reward:      [-1.0, +1.5]  ← controllable env components only
parse_penalty:   -0.5 or 0.0   ← invalid JSON
format_penalty:  -0.25 or 0.0  ← missing required keys
unsafe_penalty:  -0.3 or 0.0   ← predicted action in unsafe list

Total range:     [-2.05, +1.5]
```

The `env_reward` uses only controllable components: `action_quality + sequencing_reward + coordination_reward + success_reward + delta_health + conflict_penalty`. Uncontrollable penalties (urgency, bleed, stagnation) are stripped. Responsibility violations get a fixed -0.5 instead of the env's catastrophic -5.0.

---

## 10. Troubleshooting

### Training crashes / OOM

- Reduce `--batch-size` (must still equal `--num-generations`)
- Add `--precision fp16` if bf16 is not supported
- For GRPO, try `--batch-size 4 --num-generations 4` (reduces VRAM at the cost of weaker GRPO advantages)

### KL stays flat at ~0.0

The policy is not learning. Try:
- Increase `--learning-rate` (e.g., 2e-5)
- Decrease `--beta` (e.g., 0.005)
- Verify the SFT adapter loaded correctly (check console for `[GRPO] merge_and_unload` messages)

### Reward stays negative / doesn't improve

- Check `grpo_component_parse_penalty.png` — if parse failures are >50%, the model isn't producing valid JSON. Re-run SFT for more steps first.
- Check `grpo_component_env_responsibility_penalty.png` — if responsibility violations are frequent, the model is routing to wrong domain agents. This should decrease over training.

### LFS pointer error on HF Hub push

This is already fixed in the codebase. The training scripts use `HfApi.upload_folder()` with explicit `allow_patterns` instead of `trainer.push_to_hub()`. If you see this error in an older version, update to the latest code.

### `cp1252` / encoding error on Windows

This is a known issue with `transformers` on Python 3.13 + Windows. The training pipeline is designed to run on Linux (HF Jobs containers). For local testing on Windows, use the smoke test commands with `--max-steps 1`.

---

## Appendix: Full Configuration Reference

### train_sft.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `Qwen/Qwen2.5-3B-Instruct` | Base model |
| `--train-file` | `data/generated/sft_train.jsonl` | Training data |
| `--val-file` | `data/generated/sft_val.jsonl` | Validation data |
| `--output-dir` | `outputs/sft-qwen2.5-3b` | Output directory |
| `--hub-model-id` | `""` | HF Hub repo (empty = don't push) |
| `--max-seq-length` | `1536` | Max sequence length |
| `--epochs` | `1.0` | Training epochs |
| `--learning-rate` | `2e-4` | Learning rate |
| `--batch-size` | `1` | Per-device batch size |
| `--grad-accum` | `8` | Gradient accumulation steps |
| `--max-steps` | `-1` | Max steps (-1 = use epochs) |
| `--lora-r` | `16` | LoRA rank |
| `--lora-alpha` | `32` | LoRA alpha |
| `--precision` | `bf16` | `bf16`, `fp16`, or `fp32` |

### train_grpo.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `Qwen/Qwen2.5-3B-Instruct` | Base model |
| `--sft-adapter` | `""` | Path or HF repo for SFT LoRA adapter |
| `--input` | `tasks/cascade.json` | Scenario definitions |
| `--prompt-file` | `data/generated/grpo_prompts.jsonl` | GRPO prompts |
| `--output-dir` | `outputs/grpo-qwen2.5-3b` | Output directory |
| `--hub-model-id` | `""` | HF Hub repo (empty = don't push) |
| `--max-steps` | `300` | Training steps |
| `--batch-size` | `8` | Per-device batch size (must = num-generations) |
| `--grad-accum` | `2` | Gradient accumulation |
| `--num-generations` | `8` | Completions per prompt for GRPO |
| `--max-completion-length` | `384` | Max tokens per completion |
| `--max-prompt-length` | `1536` | Max prompt tokens |
| `--learning-rate` | `2e-5` | Learning rate |
| `--beta` | `0.005` | KL penalty coefficient |
| `--temperature` | `0.9` | Sampling temperature |
| `--warmup-ratio` | `0.05` | Fraction of steps for LR warmup |
| `--curriculum / --no-curriculum` | on | Easy→hard ordering |
| `--precision` | `bf16` | `bf16`, `fp16`, or `fp32` |

### submit_hf_job.py

| Argument | Default | Description |
|----------|---------|-------------|
| stage (positional) | required | `smoke`, `sft`, `grpo`, or `all` |
| `--flavor` | `l40sx1` | GPU type |
| `--model` | `Qwen/Qwen2.5-3B-Instruct` | Base model |
| `--hf-user` | `meancodi` | HF username |
| `--branch` | `sandeep` | Git branch |
| `--timeout` | auto | Max runtime |
| `--dry-run` | off | Preview only |
