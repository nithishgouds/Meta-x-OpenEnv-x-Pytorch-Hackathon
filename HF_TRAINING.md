# HF Training Runbook

This is the canonical Hugging Face Jobs flow for the current `sandeep` branch.

## Assumptions

- GPU target: `L4 24GB`
- Base model default: `Qwen/Qwen2.5-1.5B-Instruct`
- Seed default: `42`
- Training dependencies are pinned in `pyproject.toml` and `requirements.txt`
- The submitter clones branch `sandeep` unless overridden with `--branch`

## Required Secrets

- `HF_TOKEN`: read/write token for model repos
- `HF_USER`: your HF username or org

Optional environment overrides:

- `MODEL`
- `REPO_URL`
- `REPO_BRANCH`

## Recommended Flow

Use the Python submitter rather than raw `hf jobs run` quoting.

Smoke:

```powershell
$env:HF_TOKEN = Read-Host "Paste HF token"
$env:HF_USER = "your-hf-user"
python submit_hf_job.py smoke
```

SFT:

```powershell
python submit_hf_job.py sft
```

GRPO:

```powershell
python submit_hf_job.py grpo
```

Combined run, only if you are comfortable with a longer job:

```powershell
python submit_hf_job.py all
```

## What Each Stage Does

- `smoke`
  - generates `data/generated/*`
  - runs a very short SFT job
- `sft`
  - regenerates datasets
  - trains the LoRA SFT adapter
  - writes `outputs/sft-qwen2.5-1.5b/training_metadata.json`
- `grpo`
  - regenerates datasets
  - reads `data/generated/grpo_prompts.jsonl`
  - trains GRPO with:
    - `batch-size=2`
    - `num-generations=2`
    - `max-completion-length=512`
    - `max-prompt-length=1536`
  - writes:
    - `outputs/grpo-qwen2.5-1.5b/run_meta.json`
    - `outputs/grpo-qwen2.5-1.5b/metrics.json`

## Reproducibility

- SFT and GRPO both accept `--seed`
- Run metadata records:
  - CLI args
  - git SHA
  - dataset hash
  - prompt-file hash where applicable

## Evaluation

After training, verify:

1. the adapter loads with `PeftModel.from_pretrained(...)`
2. SFT emits valid JSON with the required keys
3. GRPO metrics show non-flat `valid_json_rate` and `accuracy`
4. downstream eval artifacts in `eval_results/` still look healthy

## Failure Modes

- `push_to_hub` 401:
  - verify `HF_TOKEN`
  - confirm repo namespace and write access
- OOM:
  - reduce `--max-prompt-length`
  - reduce `--max-completion-length`
  - reduce `--grad-accum` or training length
- Missing adapter during GRPO:
  - prefer running `sft` and `grpo` as separate jobs
- Dependency mismatch:
  - reinstall from pinned `requirements.txt`

## Rollback

If a pushed adapter is bad, revert by changing the Hub revision you consume, or delete the faulty repo/version with the Hugging Face CLI or web UI.
