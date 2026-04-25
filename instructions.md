# OpsSim-AI — Operator's Guide

End-to-end instructions for running every script in this repo locally and on Hugging Face Jobs. Read top to bottom the first time; later you can jump straight to **Common Workflows**.

---

## 0. Why the next run should perform well

After this session's refactor, the GRPO stage is finally set up to actually improve the policy. The short version:

- **Reward signal is real.** `EnvScorer` calls `DevOpsEnv.step` per completion, so the gradient follows actual environment outcomes (`success_reward`, `delta_health`, `urgency_penalty`, …) — not gold-label match.
- **Per-scenario normalization** keeps the reward in `[-1.5, +1.5]` regardless of scenario length, so easy and hard scenarios contribute comparable gradients.
- **Curriculum + `SequentialSampler`** feeds easy → medium → all, so the policy doesn't get crushed by hard scenarios early.
- **KL anchor is correct.** SFT is merged into the base before a *fresh* GRPO LoRA is attached — KL is anchored to the SFT policy as intended, not to the base model.
- **Hyperparameters fit a100-large + $25.** `lr=2e-6`, `beta=0.05`, `batch=4`, `G=4`, `max_steps=300`. Conservative LR + KL keeps drift tame; 300 steps × 8 generations gives enough exploration.
- **Crash-safe observability.** Every step's quality metrics and reward components flush to JSONL immediately; `train.log` mirrors stdout; plots auto-generate at end. If something goes sideways, you'll see exactly where.

What to watch on a live run (full detail in §8.1):

- KL stays under ~1.0 throughout — if it shoots past 5 in <30 steps, drop `--learning-rate` to `1e-6` or raise `--beta` to `0.1`.
- Unsafe rate falls below 5% within ~50 steps.
- Smoothed reward drifts up through stage 1, plateaus on stage 2, climbs again on stage 3.
- `env_success_rate` is the north star — anything above the 0.30 baseline is a real win.

If the curves look off, §11 maps every realistic symptom to a one-line fix.

---

## 1. What this project is

OpsSim-AI trains a Qwen2.5 model to act as an incident-response decision agent over a custom **DevOpsEnv** (`env.py`). Pipeline:

1. **Scenarios** in [tasks/cascade.json](tasks/cascade.json) — 10 multi-step incidents (cascade failures, breaches, perf degradation, etc.).
2. **SFT dataset** generated from gold trajectories — supervised fine-tune teaches valid JSON action format and basic action selection.
3. **GRPO refinement** — env-grounded reward (uses real `DevOpsEnv.step` outcomes, not just label match), per-scenario normalization, curriculum, KL anchored to SFT.
4. **Inference** — local eval, multi-agent eval, or single-shot run against the trained adapter.

Hardware target: **single 80 GB A100** (HF Jobs `a100-large`, ~$2.50/h, $25 budget).

---

## 2. Repo layout (what each file does)

| File | Purpose |
|------|---------|
| [env.py](env.py) | DevOpsEnv: scenario loader, `step`, reward computation, `Reward` model. |
| [models.py](models.py) | Pydantic models: `OpsSIMAction`, `Reward`, agent enums. |
| [tasks/cascade.json](tasks/cascade.json) | 10 incident scenarios, each with optimal trajectory + harmful contrast actions. |
| [generate_sft_dataset.py](generate_sft_dataset.py) | Builds SFT train/val JSONL + GRPO prompt JSONL from scenarios. |
| [train_sft.py](train_sft.py) | Stage 1 — SFT LoRA on Qwen2.5. |
| [train_grpo.py](train_grpo.py) | Stage 2 — env-grounded GRPO with curriculum + KL anchor to SFT. |
| [training_logging.py](training_logging.py) | Shared observability (tee logger, JSONL append, final_metrics, auto-plot). |
| [plot_training_logs.py](plot_training_logs.py) | Generates PNG plots from `plot_data/*.jsonl`. |
| [submit_hf_job.py](submit_hf_job.py) | One-shot submitter for HF Jobs (`smoke` / `sft` / `grpo` / `all`). |
| [inference.py](inference.py) | Single-agent eval against scenarios. |
| [multi_agent.py](multi_agent.py) | Multi-agent rollout/eval. |
| [run_trained_inference.py](run_trained_inference.py) | Quick smoke against a trained LoRA. |
| [test_inference_fix.py](test_inference_fix.py) | Lightweight regression for the inference path. |
| [server/app.py](server/app.py) | OpenEnv FastAPI server wrapping `DevOpsEnv`. |
| [Dockerfile](Dockerfile), [openenv.yaml](openenv.yaml) | Container + OpenEnv manifest for the env server. |
| [pyproject.toml](pyproject.toml), [requirements.txt](requirements.txt) | Local install (editable). |
| [submit_hf_job.py](submit_hf_job.py) | HF Jobs orchestrator (a100-large defaults). |
| [grpo_reward_review.md](grpo_reward_review.md) | Design notes for the env-grounded reward. |
| [HF_TRAINING.md](HF_TRAINING.md), [train_details.md](train_details.md), [training_analysis.md](training_analysis.md) | Historical context — read after this file. |

Output of each run lives under `outputs/<stage>-<slug>/` and contains `train.log`, `final_metrics.json`, `plot_data/`, and `plots/`.

---

## 3. Prerequisites

### 3.1 Python environment

- Python 3.10 or 3.11 (3.12 will likely work for inference but TRL 0.17 wheels are most reliable on 3.10/3.11).
- CUDA 12.1+ for local training; CPU-only is fine for dataset generation, plotting, and dry-run smoke tests.

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -e .
pip install -r requirements.txt
```

`requirements.txt` pins `trl==0.17.0`, `peft`, `transformers`, `datasets`, `accelerate`, `torch`, `matplotlib`, `huggingface_hub`.

### 3.2 Environment variables

| Variable | Purpose | Required for |
|----------|---------|--------------|
| `HF_TOKEN` | Hugging Face token (write scope) | pushing adapters, running HF Jobs |
| `HF_USER` | Your HF namespace (e.g. `meancodi`) | `submit_hf_job.py` |
| `OPENAI_API_KEY` | Only if you call OpenAI baselines in `inference.py` | optional |
| `MODEL` | Override default model id | optional |
| `REPO_URL`, `REPO_BRANCH` | Override the git source HF Jobs clones | optional |

```powershell
$env:HF_TOKEN = "hf_xxx"
$env:HF_USER  = "meancodi"
```

> **Windows note.** TRL 0.17.0 has a `cp1252` decode bug in `chat_template_utils.py` line 309 when reading templates with non-ASCII chars. If you hit it locally, set `PYTHONUTF8=1` before running, or train on Linux/HF Jobs.

---

## 4. Generate datasets

Always run this before SFT or GRPO — it produces the JSONL files that both training scripts consume:

```powershell
python generate_sft_dataset.py `
  --input tasks/cascade.json `
  --output-dir data/generated `
  --validate
```

Outputs:

- `data/generated/sft_train.jsonl` — SFT training pairs
- `data/generated/sft_val.jsonl` — SFT validation
- `data/generated/grpo_prompts.jsonl` — prompt-only file consumed by GRPO
- `data/generated/dataset_profile.json` — token stats for sanity checking

`--validate` re-loads each row through `OpsSIMAction` to guarantee parsable JSON.

---

## 5. Stage 1 — SFT

### 5.1 Local run (1.5B for fast iteration)

```powershell
python train_sft.py `
  --model Qwen/Qwen2.5-1.5B-Instruct `
  --train-file data/generated/sft_train.jsonl `
  --val-file   data/generated/sft_val.jsonl `
  --output-dir outputs/sft-qwen25-1p5b `
  --epochs 1 --batch-size 1 --grad-accum 8 --max-seq-length 1536
```

### 5.2 Production run (3B, matches HF Jobs defaults)

```powershell
python train_sft.py `
  --model Qwen/Qwen2.5-3B-Instruct `
  --train-file data/generated/sft_train.jsonl `
  --val-file   data/generated/sft_val.jsonl `
  --output-dir outputs/sft-qwen25-3b `
  --hub-model-id $env:HF_USER/opssim-qwen25-3b-sft-lora `
  --epochs 1 --batch-size 1 --grad-accum 8 --max-seq-length 1536 --save-steps 50
```

### 5.3 Key SFT flags

| Flag | Default | Notes |
|------|---------|-------|
| `--model` | `Qwen/Qwen2.5-1.5B-Instruct` | Override to `Qwen/Qwen2.5-3B-Instruct` for prod. |
| `--epochs` | `1.0` | One epoch is enough on this dataset. |
| `--batch-size` / `--grad-accum` | `1 / 8` | Effective batch 8. Bump on bigger GPUs. |
| `--max-seq-length` | `1536` | Long enough for full prompts + actions. |
| `--learning-rate` | `2e-4` | LoRA-friendly. |
| `--lora-r / --lora-alpha / --lora-dropout` | `16 / 32 / 0.05` | Same shape used by GRPO (must match for KL anchor reuse). |
| `--save-steps` | `50` | Checkpoint cadence. |
| `--precision` | `bf16` | A100 native. Use `fp16` only on T4/V100. |
| `--hub-model-id` | empty | When set, adapter is pushed at end. |

### 5.4 Artifacts produced

```
outputs/sft-qwen25-3b/
├── train.log                  # full stdout/stderr tee
├── final_metrics.json         # headline numbers (loss, best_metric, steps)
├── adapter_config.json        # LoRA config
├── adapter_model.safetensors  # LoRA weights (push-ready)
├── plots/                     # PNGs auto-generated
└── plot_data/
    ├── train_metrics.jsonl    # per-step train loss / lr
    ├── eval_metrics.jsonl     # per-eval validation loss
    └── summary.json
```

---

## 6. Stage 2 — GRPO refinement

GRPO consumes the SFT adapter as both **start point and KL anchor**. Architecturally:

1. Base model is loaded.
2. SFT LoRA is applied with `is_trainable=False`, then `merge_and_unload()` folds it into the base. The merged weights become the **frozen reference** for KL.
3. A **fresh** GRPO LoRA (same shape) is attached on top — this is the only thing that trains.
4. For every completion, `EnvScorer` replays the optimal prefix of the scenario, then steps `DevOpsEnv` once with the model's predicted action and reads the actual `Reward`. Per-scenario min/max normalization (see `EnvScorer.norm`) maps it to roughly `[-1.5, +1.5]`.
5. A curriculum sampler (`CurriculumGRPOTrainer._get_train_sampler`) feeds easy → medium → all scenarios in order — `SequentialSampler`, never reshuffled.

### 6.1 Local run (small)

```powershell
python train_grpo.py `
  --model Qwen/Qwen2.5-3B-Instruct `
  --sft-adapter outputs/sft-qwen25-3b `
  --input tasks/cascade.json `
  --prompt-file data/generated/grpo_prompts.jsonl `
  --output-dir outputs/grpo-qwen25-3b `
  --max-steps 50 --batch-size 2 --num-generations 2
```

### 6.2 Production run (matches HF Jobs `grpo` defaults)

```powershell
python train_grpo.py `
  --model Qwen/Qwen2.5-3B-Instruct `
  --sft-adapter $env:HF_USER/opssim-qwen25-3b-sft-lora `
  --input tasks/cascade.json `
  --prompt-file data/generated/grpo_prompts.jsonl `
  --output-dir outputs/grpo-qwen25-3b `
  --hub-model-id $env:HF_USER/opssim-qwen25-3b-grpo-lora `
  --max-steps 300 --batch-size 4 --grad-accum 2 `
  --num-generations 4 --max-completion-length 384 --max-prompt-length 1536 `
  --learning-rate 2e-6 --beta 0.05
```

### 6.3 Key GRPO flags (and why they are what they are)

| Flag | Default | Why |
|------|---------|-----|
| `--sft-adapter` | empty | **Required** for KL anchor. Path or HF repo. |
| `--max-steps` | `300` | 300 × (batch 4 × G 4 × ga 2) = 9 600 completions. Fits $25 budget. |
| `--batch-size` | `4` | Prompts per device step. |
| `--grad-accum` | `2` | Effective optimizer batch = 8 prompts. |
| `--num-generations` | `4` | Group size for GRPO advantage estimation. Lower = noisier reward. |
| `--max-completion-length` | `384` | Action JSON fits in <200 tokens; 384 leaves headroom. |
| `--max-prompt-length` | `1536` | Matches SFT seq length. |
| `--learning-rate` | `2e-6` | RL is ~100× more sensitive than SFT. Do **not** raise to 2e-4. |
| `--beta` | `0.05` | KL coefficient. Lower = more drift; higher = stays near SFT. |
| `--curriculum` | on | Easy(3) → Medium(6) → All(10). Disable with `--no-curriculum` if you have one. |
| `--lora-r/alpha/dropout` | `16 / 32 / 0.05` | Must match SFT LoRA shape. |

### 6.4 Reward design summary

- Uses real `DevOpsEnv.step` reward, not gold-label match.
- Per-scenario normalization: `raw / ((max_full - min_full) / 2 / max_steps)`, clipped to `[-1.5, +1.5]`.
- Penalties (one-sided): `-1.0` for unparsable JSON, `-0.25` for malformed format, `-0.5` for unsafe action lookup hit.
- No bonuses for "well-formed JSON" — that's table stakes and saturated old runs.

See [grpo_reward_review.md](grpo_reward_review.md) for full numeric design.

### 6.5 GRPO artifacts

```
outputs/grpo-qwen25-3b/
├── train.log
├── final_metrics.json
├── metrics.json                 # full quality history
├── run_meta.json                # args + git_sha + dataset hashes
├── plots/
│   ├── grpo_reward_smoothed.png
│   ├── grpo_reward_components.png
│   ├── grpo_quality_metrics_smoothed.png
│   ├── grpo_unsafe_rate.png
│   ├── grpo_kl_only.png
│   └── grpo_final_quality_snapshot.png
└── plot_data/
    ├── train_metrics.jsonl       # one line per trainer log step
    ├── reward_components.jsonl   # per-step component mean/min/max
    ├── quality_metrics.jsonl     # per-step env-grounded quality (incremental)
    ├── completion_samples.jsonl
    ├── dataset_profile.json
    └── summary.json
```

---

## 7. Run on Hugging Face Jobs

[submit_hf_job.py](submit_hf_job.py) wraps everything into a single command.

```powershell
$env:HF_TOKEN = "hf_xxx"
$env:HF_USER  = "meancodi"

# preview without launching
python submit_hf_job.py grpo --dry-run

# fast smoke (10 SFT steps, ~20 min)
python submit_hf_job.py smoke

# full SFT only (~1 h)
python submit_hf_job.py sft

# full GRPO only — assumes SFT adapter already at meancodi/opssim-qwen25-3b-sft-lora
python submit_hf_job.py grpo

# end-to-end (SFT then GRPO) — budget 6h
python submit_hf_job.py all
```

Defaults: flavor `a100-large`, model `Qwen/Qwen2.5-3B-Instruct`, branch `sandeep`. Override with `--flavor`, `--model`, `--branch`, `--namespace`, `--timeout`.

After submission, follow the printed URL for live logs. Avoid CLI polling — the web UI is faster and doesn't rate-limit you.

### 7.1 Cost cheat sheet

| Stage | Flavor | Time | Cost |
|-------|--------|------|------|
| smoke | a100-large | ~20 min | ~$0.85 |
| sft   | a100-large | ~1 h | ~$2.50 |
| grpo  | a100-large | ~3 h | ~$7.50 |
| all   | a100-large | ~4 h | ~$10 |

Stays well inside the $25 hackathon cap with room for one full re-run.

---

## 8. Observability — what to look at after a run

Every training run drops the same flat layout under `--output-dir`:

1. **`train.log`** — full stdout/stderr. First place to look if something went sideways.
2. **`final_metrics.json`** — one-liner summary; ideal for dashboards / commit messages.
3. **`plots/*.png`** — auto-generated by [plot_training_logs.py](plot_training_logs.py) at end of run.
4. **`plot_data/*.jsonl`** — every JSONL is appended **per step** with explicit flush, so a SIGKILL mid-run still leaves you with valid data.

To regenerate plots from an existing run (or a downloaded HF Jobs artifact):

```powershell
python plot_training_logs.py `
  --grpo-dir outputs/grpo-qwen25-3b `
  --output-dir outputs/grpo-qwen25-3b/plots
```

It also accepts `--sft-dir` and HF Hub repos via `--grpo-repo / --sft-repo`.

### 8.1 What "healthy" looks like

- **GRPO reward (smoothed)** — should drift up over the first 100 steps, plateau around stage 2 (medium curriculum), then keep improving slightly through stage 3.
- **KL** — should stay well under 1.0 with `--beta 0.05`. A KL spike past 5 means your `--beta` is too low or `--learning-rate` too high.
- **Unsafe rate** — should fall below 5% within ~50 steps and stay there.
- **Valid JSON rate** — should hit 100% within ~20 steps (SFT already taught format).
- **env_success_rate** — primary north star metric. Anything above the 0.30 baseline is a real win.

If reward goes flat and KL is also flat, your SFT adapter likely wasn't loaded — check `train.log` for `merge_and_unload` lines.

---

## 9. Inference / evaluation

### 9.1 Quick run against a trained adapter

```powershell
python run_trained_inference.py `
  --model Qwen/Qwen2.5-3B-Instruct `
  --adapter outputs/grpo-qwen25-3b `
  --scenario cascade-001
```

### 9.2 Full eval suite

```powershell
python inference.py `
  --model Qwen/Qwen2.5-3B-Instruct `
  --adapter outputs/grpo-qwen25-3b `
  --tasks tasks/cascade.json `
  --output eval_results/results.json
```

### 9.3 Multi-agent rollout

```powershell
python multi_agent.py --tasks tasks/cascade.json
```

### 9.4 OpenEnv server (optional)

```powershell
cd server
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8080
```

[Dockerfile](Dockerfile) + [openenv.yaml](openenv.yaml) wrap the same server for OpenEnv submission.

---

## 10. Common workflows (copy-paste)

### 10.1 Full pipeline, locally on A100

```powershell
python generate_sft_dataset.py --input tasks/cascade.json --output-dir data/generated --validate

python train_sft.py --model Qwen/Qwen2.5-3B-Instruct `
  --train-file data/generated/sft_train.jsonl --val-file data/generated/sft_val.jsonl `
  --output-dir outputs/sft-qwen25-3b --epochs 1 --batch-size 1 --grad-accum 8

python train_grpo.py --model Qwen/Qwen2.5-3B-Instruct `
  --sft-adapter outputs/sft-qwen25-3b `
  --input tasks/cascade.json --prompt-file data/generated/grpo_prompts.jsonl `
  --output-dir outputs/grpo-qwen25-3b `
  --max-steps 300 --batch-size 4 --grad-accum 2 --num-generations 4 `
  --learning-rate 2e-6 --beta 0.05

python plot_training_logs.py --sft-dir outputs/sft-qwen25-3b --grpo-dir outputs/grpo-qwen25-3b `
  --output-dir plots/qwen25-3b-full
```

### 10.2 Full pipeline on HF Jobs

```powershell
$env:HF_TOKEN = "hf_xxx"; $env:HF_USER = "meancodi"
python submit_hf_job.py all
```

### 10.3 Retrain GRPO only with different KL beta

```powershell
python submit_hf_job.py grpo --dry-run   # confirm command
python submit_hf_job.py grpo             # uses --beta 0.05 default
# Or locally:
python train_grpo.py --model Qwen/Qwen2.5-3B-Instruct `
  --sft-adapter meancodi/opssim-qwen25-3b-sft-lora `
  --input tasks/cascade.json --prompt-file data/generated/grpo_prompts.jsonl `
  --output-dir outputs/grpo-beta-0p1 `
  --max-steps 300 --beta 0.1
```

---

## 11. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `UnicodeDecodeError` reading `chat_template_utils.py` | Windows cp1252 + TRL 0.17 bug | `setx PYTHONUTF8 1`, restart shell, or run on Linux. |
| GRPO reward stays flat near zero | SFT adapter not merged in | Confirm `--sft-adapter` path/repo is reachable; check `train.log` for `merge_and_unload`. |
| KL explodes past 5 within 30 steps | `--learning-rate` too high or `--beta` too low | Set `--learning-rate 1e-6` and `--beta 0.1`. |
| `ValueError: num_generations must divide global batch` | TRL group constraint | Make sure `batch_size * grad_accum % num_generations == 0`. |
| `OOM` on a100-large | `max-prompt-length` + `max-completion-length` too high | Lower `--max-completion-length` to 256. |
| HF Job dies at clone step | Bad branch name / private repo | Re-check `--repo-url --branch`; ensure HF_TOKEN has read scope. |
| Plots empty | `plot_data/*.jsonl` empty | Means callbacks never fired — check `train.log` for early crash before `trainer.train()`. |
| `OpsSIMAction.parse_raw` keeps failing in eval | Adapter not actually loaded | Pass `--adapter` (not `--lora-path`) and confirm `adapter_config.json` exists. |

---

## 12. Reproducibility checklist

Before publishing results, verify:

- [ ] `data/generated/dataset_profile.json` matches across runs (same seed, same cascade.json hash).
- [ ] `run_meta.json` git_sha matches the commit you tagged.
- [ ] `final_metrics.json` `last_quality_metrics.env_success_rate` is recorded.
- [ ] `plots/` PNGs ship inside the pushed adapter repo (auto-handled).
- [ ] `train.log` doesn't show any `WARNING: peft is overwriting` messages.

---

## 13. Where to read next

- [grpo_reward_review.md](grpo_reward_review.md) — reward design rationale + numbers.
- [HF_TRAINING.md](HF_TRAINING.md) — original HF Jobs notes.
- [train_details.md](train_details.md) — historical experiment log.
- [training_analysis.md](training_analysis.md) — earlier failed-run post-mortem.
