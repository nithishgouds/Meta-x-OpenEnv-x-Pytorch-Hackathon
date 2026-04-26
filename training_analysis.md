# Training Pipeline Critical Review (`sandeep` branch)

Scope: [train_grpo.py](train_grpo.py), [train_sft.py](train_sft.py), [generate_sft_dataset.py](generate_sft_dataset.py), [submit_hf_job.py](submit_hf_job.py), [HF_TRAINING.md](HF_TRAINING.md).

---

## 1. [train_sft.py](train_sft.py)

### Critical correctness issues

- **Label-masking misalignment with chat template.** `tokenize_example` builds two separate tokenizations:
  - `prompt_text = apply_chat_template(prompt_messages, add_generation_prompt=True)`
  - `full_text  = apply_chat_template(messages, add_generation_prompt=False)`

  These two strings are **not guaranteed to share a prefix**. For Qwen2.5's template, the prompt with `add_generation_prompt=True` ends with `<|im_start|>assistant\n`, while `full_text` (no generation prompt) emits the assistant turn as `<|im_start|>assistant\n{content}<|im_end|>\n`. They happen to align at the boundary, but this is a coincidence per-template — for many templates it isn't true. Then `prompt_len = min(len(prompt_ids), len(labels))` silently masks/un-masks the wrong tokens. Robust fix: tokenize once and locate the assistant span via the assistant header token sequence, or use `trl.SFTTrainer` with `DataCollatorForCompletionOnlyLM` and an explicit response template.

- **No EOS at end of `full_text`.** If the chat template doesn't append `eos_token` after `<|im_end|>` (Qwen does emit `<|im_end|>` but training without re-checking that the EOS id is included in `input_ids` and is unmasked in `labels` is fragile), the model can fail to terminate at inference. Verify that `tokenizer.eos_token_id` (or `<|im_end|>` id) is present and unmasked in the last position; if not, append it.

- **`bf16` flag is non-disablable from CLI.** `--bf16 action="store_true", default=True` makes `args.bf16` always `True` regardless of CLI input. Passing `--fp16` does not turn off bf16; the resulting `TrainingArguments(bf16=True, fp16=True)` raises in newer Transformers. Use `BooleanOptionalAction` or split into `--precision {bf16,fp16,fp32}`.

- **`device_map="auto"` + `Trainer` is unsafe.** With `Trainer` (which expects a single accelerator placement), `device_map="auto"` will dispatch shards across visible GPUs and break gradient updates / DDP. For a 1.5B model on L4 24GB you do not need sharding — drop `device_map` or set `device_map={"": 0}`.

- **PEFT + gradient checkpointing without `enable_input_require_grads`.** With `gradient_checkpointing=True`, LoRA gradients are frequently silently zeroed because the input embeddings have `requires_grad=False`. Standard pattern:
  ```python
  model.gradient_checkpointing_enable()
  model.enable_input_require_grads()
  model = get_peft_model(model, peft_config)
  ```

- **`training_metadata.json` is written after `push_to_hub()`.** The metadata file is created *after* the push call, so it is never uploaded. Also it's only written when `hub_model_id` is set — should always be saved for reproducibility.

### Robustness / reproducibility

- No `set_seed(seed)` and no `--seed` flag. Two SFT runs are not bit-identical.
- `dataset.map(...)` is not `batched=True`; tokenization is slow per-row.
- `tokenizer.pad_token = tokenizer.eos_token` — fine, but you should also set `model.config.pad_token_id = tokenizer.pad_token_id` so generation/eval honors padding.
- No `remove_unused_columns=False` setting; OK because `map` removes them, but explicit is safer.
- `save_total_limit=2` combined with `push_to_hub=True` only pushes whatever happens to remain locally — combine with `save_strategy="steps"` and explicit final-checkpoint push.
- Validation is enabled silently only if a val file is present; no guard that val rows tokenize within `max_seq_length` (truncation will happen and skew eval loss).

---

## 2. [train_grpo.py](train_grpo.py)

### Critical correctness issues

- **`num_generations=2` with `per_device_train_batch_size=1` is invalid for `GRPOTrainer`.** TRL requires `(per_device_train_batch_size * num_processes) % num_generations == 0` and that each prompt has `num_generations` completions in the same micro-batch. With `1 % 2 != 0`, TRL raises an assertion or silently drops samples depending on version. Either set `per_device_train_batch_size=num_generations` (e.g., 2) or `num_generations` to a divisor of the batch.

- **`max_completion_length=256` cliff against the reward.** The required JSON has six keys (`analysis`, `plan`, `next_action`, `target_agent`, `reasoning`, `confidence`). Synthetic assistant outputs from `generate_sft_dataset.py` are commonly >256 tokens once the optimal plan list is non-trivial. Truncated completions fail `parse_json_object` and receive a flat `-0.8`. Many groups will then have zero reward variance → GRPO advantage = 0 → no learning signal. Raise to ≥512 or, better, prompt the model for a constrained-format JSON and add a length-aware reward.

- **`build_model_or_id` returns a `PeftModel` but `peft_config=None` is also passed to `GRPOTrainer`.** Newer TRL (`>=0.10`) detects an existing `PeftModel` and reuses it as policy + ref-policy with adapter disabled, but earlier versions try to wrap it again. Pin TRL or assert the version — and pass `peft_config=None` *only* when `model` is already a `PeftModel`. Currently `peft_config=None` also fires when `sft_adapter==""` and `model` is a string id, which is incorrect: you'd train the full base model. The current code does set a `LoraConfig` in that case, so behavior is OK, but the conditional is convoluted; refactor.

- **`build_grpo_dataset` re-runs the full SFT generator for every call.** It calls `generate_examples_for_scenario` (which executes the playbook simulator) and then throws away every row whose `label_type != "gold_next_action"`. The contrast rows are computed and discarded — pure waste, plus `__import__("random").Random(42)` is called in a loop. Use `write_grpo_prompts` output (`data/generated/grpo_prompts.jsonl`) directly instead of re-deriving.

- **Reward shaping mixes binary and dense terms additively.** Range is roughly `[-1.55, +1.95]` with discontinuities (e.g., `-0.8` for unparseable). GRPO normalizes within a group of `num_generations` samples; with only 2 generations per prompt and the JSON-parse cliff, std collapses and the per-token advantage becomes either `±1.0` or `0`. Result: training is dominated by parse-success vs parse-failure, not by action quality. Recommend: split into separate `reward_funcs` (parse, action, agent, format) — TRL averages them and reports them individually, which also improves diagnostics.

- **No KL/beta tuning, no `max_prompt_length`.** With prompts that include playbook text + flattened state + completed actions, prompts can exceed 1.5k tokens. Without `max_prompt_length`, OOM at high steps is likely on L4. Set `max_prompt_length` and confirm `max_completion_length + max_prompt_length` fits the model context.

- **`gradient_checkpointing` not enabled.** Generation rollouts already double memory; without checkpointing on a 1.5B model with 2 generations and ~1.5k prompts you are very close to L4 24GB OOM.

- **`save_model(final_dir)` then `trainer.push_to_hub()` push different paths.** `push_to_hub` pushes from `output_dir`, not from `final_dir`, so the "final" copy isn't what ends up on the Hub.

### Robustness

- No `seed` argument or `set_seed()` → GRPO trajectories non-reproducible.
- `extract_text` loses information when completion is a list of dict messages but more than one assistant turn — only the last is read; the trainer normally produces a single completion, so this is fine but should be asserted.
- `confidence` parsing penalizes valid `null` (model often emits `null`); minor, but biases shaping.

---

## 3. [generate_sft_dataset.py](generate_sft_dataset.py)

### Correctness

- **Boolean expression parser is naive.** `evaluate_condition` splits on `" OR "` and `" AND "` purely textually. It does **not** support parentheses, mixed precedence, or the `NOT` operator. If `tasks/cascade.json` ever contains `(A OR B) AND C`, this evaluator returns the wrong truth value, silently corrupting "feasible" labels and `unsafe_actions`. Either restrict the schema (and assert it) or use a real parser (`pyparsing`, `ast.parse` on a sanitized expression).

- **`apply_effects` numeric coercion swallows bugs.** When `current.get(target_key, 0)` is a string like `"degraded"` and effect is `"+1"`, `parse_value("degraded")` returns `"degraded"`, then `float(previous) if isinstance(previous, (int,float)) else 0.0` resets it to 0 and adds — silently losing state. At minimum log a warning when the type doesn't match the expected operator.

- **Anomaly detection via substring on `str(value).lower()` is leaky.** A field value of `"no_errors"` or numeric `0` gets matched against `"error"`/`"down"` because `"errors_per_minute: 0"` flattens to value `0` (OK) but a string field `"errors": "none_observed"` matches `"error"`. False-positive anomalies go into the prompt and into `analysis`, training the model on noise.

- **Investigation prefix logic in `compute_confidence`** is brittle: any future action named `correlate_logs` counts; a `verify_state` action does not. The whole `compute_confidence` heuristic produces a synthetic label the model learns to mimic — this teaches a spurious confidence value, not a calibrated one. Either drop `confidence` from the schema or emit a constant.

- **Scenario-level split with very few scenarios.** `val_count = max(1, round(N * 0.2))`. If `cascade.json` has 5 scenarios, val set is one scenario's deterministic walk — eval loss is noise. Better: stratified step-level holdout, or k-fold across scenarios.

- **`target_modules` hardcoded to Qwen MLP/attn names** (also in `train_sft.py` / `train_grpo.py`). Pass a non-Qwen `--model` and PEFT silently matches nothing → no trainable params. Add an assertion that `print_trainable_parameters()` is non-zero.

- `INVESTIGATE_PREFIXES` and `CRITICAL_TERMS` are coupled to scenario authoring conventions but live in a separate module — any drift breaks dataset semantics with no error. Move these constants into `tasks/cascade.json` or unit-test them.

### Reproducibility

- `rng` is used but `random.shuffle(scenarios)` happens *before* `train`/`val` split — different seeds produce different splits, fine, but the split is not logged with a hash of `tasks/cascade.json`. Add input hash and tool versions to `summary.json`.

---

## 4. [submit_hf_job.py](submit_hf_job.py)

- **Hard-coded `REPO_URL`, `BRANCH`, `HF_USER`, `MODEL`.** Any fork/branch needs a code edit. Promote to CLI args / env vars.
- **`pip install -e .` only.** Relies on `pyproject.toml` declaring all training dependencies (`trl`, `peft`, `datasets`, `bitsandbytes` if needed, `huggingface_hub` of compatible version). If `pyproject.toml` is leaner than `requirements.txt`, the job will run and fail late. Either consolidate into `pyproject.toml` or run `pip install -r requirements.txt` after `-e .`.
- **No version pins for `transformers`/`trl`/`peft` visible from this script.** GRPO API changed across TRL `0.9 → 0.11` (`num_generations` semantics, reward signature). Pin in `pyproject.toml`.
- **Token leakage surface.** `secrets={"HF_TOKEN": token}` passes the token into the job environment — fine — but also `token=token` is passed to `run_job`, and `getpass` falls back to terminal input. Make sure the script never echoes the token to stdout (it doesn't currently, OK), and avoid passing the token both as `token=` and as a secret unless the SDK requires both.
- **`combined_command()` chains SFT and GRPO with `&&`.** If SFT finishes but the Hub push silently rate-limits, GRPO will pull a stale or missing adapter. Add an explicit `huggingface-cli repo create … --exist-ok` and an `hf upload` verification step before the GRPO stage, or use the two-stage flow as the default.
- **`--save-steps 50` is in the raw CLI runbook but missing from `sft_command()`** — runbook and submitter diverge.
- **`shell_prelude` runs `apt-get update && apt-get install -y git`** unconditionally; the `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel` image already has git. This is wasted ~30s every job and breaks if the apt mirror is offline.
- **No log capture path.** The job ID/URL is printed but nothing is persisted to `eval_results/` or similar. After the job dies the CLI artifact disappears.
- **Hard-coded `flavor="l4x1"` default with `timeout="2h"` for `all`** — if SFT alone exceeds 1h, GRPO is killed mid-step, producing a partially-trained adapter that nevertheless gets pushed. Add a `--no-push-on-timeout` guard or split jobs.

---

## 5. [HF_TRAINING.md](HF_TRAINING.md)

- Cost figures (`$1-2`, `$4-5`) are unsubstantiated; no per-step throughput numbers, no cite for L4 pricing.
- Documents two ways (`submit_hf_job.py` and raw `hf jobs run`) that are not byte-identical (`--save-steps 50` only in raw). Pick one.
- Does not document the **branch invariant** that `submit_hf_job.py` clones (`BRANCH = "sandeep"`). Anyone running from `main` will silently train on stale code.
- No section on how to **evaluate** the resulting adapters or how to roll back a bad push (`huggingface-cli repo delete` / revisions). `eval_results/results.json` exists in the repo but is unmentioned here.
- No mention of seeding, GPU type assumptions, dependency pinning, or how to verify the adapter loads (`PeftModel.from_pretrained`) before trusting it.
- No failure modes section: what to do if `push_to_hub` 401s mid-run, if `bitsandbytes` is unavailable, etc.

---

## Concrete improvement checklist (prioritized)

1. **Fix label masking in [train_sft.py](train_sft.py)** by using `DataCollatorForCompletionOnlyLM` with the Qwen `<|im_start|>assistant\n` response template, or by locating the assistant span via token-id search rather than two separate `apply_chat_template` calls.
2. **Make GRPO batching valid**: set `per_device_train_batch_size = num_generations` (e.g., both 2), enable `gradient_checkpointing`, set `max_prompt_length`, raise `max_completion_length` to ≥512.
3. **Split GRPO reward into multiple `reward_funcs`** (`parse_ok`, `action_match`, `agent_match`, `format_bonus`, `unsafe_penalty`) so TRL logs each separately and group variance is meaningful.
4. **Stop re-generating contrast rows** in `train_grpo.py`'s dataset builder — read `data/generated/grpo_prompts.jsonl`.
5. **Add `set_seed(args.seed)`** and `--seed` flag to both training scripts; record `args` + git SHA + dataset hash to `output_dir/run_meta.json` *before* training.
6. **Replace `--bf16 default=True` argparse pattern** with `--precision {bf16,fp16,fp32}` and assert mutual exclusivity.
7. **Drop `device_map="auto"`** from both SFT and GRPO model loaders.
8. **Add `model.enable_input_require_grads()`** before `get_peft_model` when gradient checkpointing is on.
9. **Replace `evaluate_condition`** with a real expression parser, or constrain `tasks/cascade.json` schema and validate it explicitly.
10. **Remove the `compute_confidence` heuristic from synthetic targets** — it teaches a non-grounded label the model can't verify. Fix to a constant or omit from the schema.
11. **Pin `transformers`, `trl`, `peft`, `accelerate`, `datasets`** versions in `pyproject.toml`; remove `apt-get install -y git` from `shell_prelude` and run `pip install -r requirements.txt`.
12. **Parameterize** `BRANCH`, `HF_USER`, `MODEL`, `REPO_URL` in [submit_hf_job.py](submit_hf_job.py) via CLI flags / env vars.
13. **Move `training_metadata.json` write before `push_to_hub`** in [train_sft.py](train_sft.py), and write it unconditionally.
14. **Align `push_to_hub` source path with `save_model` final path** in [train_grpo.py](train_grpo.py) (or pass `final_dir` as `output_dir`).
15. **Update [HF_TRAINING.md](HF_TRAINING.md)** to a single canonical command flow, document seeds/GPU/version pins, and add an evaluation + rollback section that references `eval_results/results.json`.
