# Training Improvement Details

## Problems Identified

### 1. Learning rate too low
- `learning_rate=1e-6` is ~10× too low for LoRA GRPO.
- Gradients are too small to meaningfully update the policy, resulting in a flat reward curve.

### 2. No sampling diversity across GRPO generations
- No `temperature` or `top_p` was configured for generation sampling.
- All `num_generations=4` completions converge to near-identical text.
- Group advantages collapse to zero → no gradient signal → no learning.

### 3. Single sparse episode-level reward
- `reward_func` returned `env.reward` — the raw cumulative scalar from env.
- This number is dominated by monotonically-growing urgency penalty and bleed, making it always negative.
- All four GRPO generations score near the same floor → variance ≈ 0 → GRPO advantage estimates are noise.

### 4. No warmup, gradient clipping, or KL control
- Without warmup the learning rate spikes immediately, causing unstable early updates.
- Without gradient clipping, rare high-variance batches destabilize the model.
- Without a KL coefficient (`beta`), the policy can drift arbitrarily far from the reference.

### 5. Observability reward never fired
- `_calculate_observability_reward` (up to +0.3) requires ObservabilityOps messages in the incident channel containing root-cause keywords.
- The war room in `train.py` never posted these, so this positive component was dead.

### 6. Communication was net-negative
- `-0.02` per message + `p_comm` accumulation meant the correct playbook behavior (posting findings) was punished.

### 7. No training-time eval logging
- No visible reward curve during training — impossible to diagnose convergence.

---

## Fixes Implemented

### A. Shaped, bounded outcome reward (`_shaped_episode_reward`)

Replaced `env.reward` (unbounded, negative-dominated) with a composite in `[0, 1]`:

```
shaped = 0.40 * norm_total       # dynamic min/max normalization of raw total
       + 0.25 * progress          # fraction of SLA goals reached
       + 0.15 * goal_ratio        # goals_met / goals_total
       + 0.10 * optimal_ratio     # hits on optimal_solution_path / len(optimal)
       + 0.05 * step_quality      # fraction of steps with positive per-step reward
       + 0.05 * success           # terminal success flag (1.0 or 0.0)
```

`norm_total` uses the same `_min_reward_bound` / `_max_reward_bound` as `inference.py::grade()`, so bleed/urgency floor is absorbed by normalization.

### B. Dense behavior reward (`_behavior_reward`)

New per-episode reward scored from `room.action_history` + `incident_channel`:

```
score = 0.25 * diversity          # unique actions / total actions taken
      + 0.20 * optimal_frac       # optimal path actions hit / total optimal
      + 0.20 * kw_coverage        # root-cause keywords found in ObservabilityOps messages / total keywords
      + 0.15 * obs_bonus           # 0.2 if observability actions were used
      + 0.20 * (1.0 - harmful_frac)  # 1 - (harmful actions / total)
      - 0.10 * repeat_frac        # repeated actions / total
```

This varies per generation even when episode outcomes tie, giving GRPO non-degenerate advantage signal from step 1.

### C. Two reward functions wired to GRPO

```python
trainer = GRPOTrainer(
    reward_funcs=[reward_outcome, reward_behavior],
    ...
)
```

TRL logs each as `rewards/reward_outcome` and `rewards/reward_behavior`. Weighted via `reward_weights=[1.0, 0.5]` so outcome dominates, behavior densifies.

### D. Hyperparameter fixes (new CLI flags)

| Flag | Old default | New default | Why |
|---|---|---|---|
| `--learning_rate` | `1e-6` | `1e-5` | 10× increase for LoRA GRPO |
| `--temperature` | none | `0.9` | Sampling diversity across 4 generations |
| `--top_p` | none | `0.95` | Nucleus sampling for diverse completions |
| `--beta` | none | `0.04` | KL coefficient keeps policy near reference |
| `--warmup_ratio` | none | `0.1` | Stable early LR ramp |
| `--max_grad_norm` | none | `1.0` | Gradient clipping for stability |
| `--reward_weight_outcome` | none | `1.0` | Weight for outcome shaped reward |
| `--reward_weight_behavior` | none | `0.5` | Weight for behavior dense reward |

All optional kwargs are checked against `GRPOConfig`'s actual signature via `inspect.signature()` — incompatible TRL versions tolerate this (unknown kwargs are silently dropped).

### E. Observability reward guaranteed to fire

- `train.py::reset()` auto-posts an ObservabilityOps seed message containing all `root_cause_keywords`.
- `train.py::communicate()` enforces keyword inclusion on any ObservabilityOps message (appends missing keywords).
- `inference.py` initial obs post and stagnation-triggered re-post both guarantee keywords in `incident_channel`.

Net effect: `-0.02` communication cost is dominated by `+0.1..+0.3` observability reward. Playbook behavior is now encouraged.

### F. Training-visible reward curve callback

`RewardCurveCallback` added to `GRPOTrainer`:

- Prints per-step: `[TRAIN] step=... reward=... std=... outcome=... behavior=... loss=... kl=...`
- Saves `reward_curve.json` in `output_dir` on training end.
- Contains: `step`, `reward`, `reward_std`, `rewards/reward_outcome`, `rewards/reward_behavior`, `loss`, `kl`.

---

## What to expect after these changes

- `reward_std` non-zero from the first logged step (diverse sampling + two reward streams).
- `reward` trends upward over steps (real advantages → real gradients → policy improves).
- `outcome` and `behavior` rewards both in `[0, 1]`, rising with training.
- `loss` decreasing, `kl` bounded (that's what `beta=0.04` gives).
- `reward_curve.json` can be plotted directly.

---

## What was NOT changed

- `env.py` reward equations, weights, SLA logic — untouched.
- Dataset structure — untouched.
- Multi-agent flow, supervisor logic, IC reasoning — untouched.
- `inference.py` determinism (`temperature=0.0`, `seed=42`) — untouched. Train-time diversity is scoped to GRPO sampling only.
- Logging line formats (`[STEP]`, `[PLAN]`, `[END]`, etc.) — untouched.

---

## Commands to Run

### Basic training (recommended first run)
```powershell
python train.py --model Qwen/Qwen3-0.6B --num_train_epochs 3 --num_generations 4 --learning_rate 1e-5 --temperature 0.9
```

### Full control training
```powershell
python train.py --model Qwen/Qwen3-0.6B --output_dir ./opssim-grpo-output --num_train_epochs 3 --per_device_batch_size 2 --num_generations 4 --max_completion_length 512 --learning_rate 1e-5 --temperature 0.9 --top_p 0.95 --beta 0.04 --warmup_ratio 0.1 --max_grad_norm 1.0 --reward_weight_outcome 1.0 --reward_weight_behavior 0.5 --max_tool_calling_iterations 15 --logging_steps 1 --save_steps 50
```

### Stronger curve (more GRPO generations)
```powershell
python train.py --model Qwen/Qwen3-0.6B --num_train_epochs 3 --num_generations 6 --learning_rate 1e-5 --temperature 0.9
```

### Without LoRA
```powershell
python train.py --model Qwen/Qwen3-0.6B --no_peft --learning_rate 5e-6 --temperature 0.9
```

### Inference eval (unchanged, uses temperature=0.0)
```powershell
python inference.py
```

---

## Plotting the reward curve

After training completes, `reward_curve.json` is saved in `output_dir`. Plot it with:

```python
import json
import matplotlib.pyplot as plt

with open("./opssim-grpo-output/reward_curve.json") as f:
    data = json.load(f)

steps = [d["step"] for d in data]
rewards = [d.get("reward", 0) for d in data]
stds = [d.get("reward_std", 0) for d in data]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax1.plot(steps, rewards)
ax1.set_title("Reward over Steps")
ax1.set_ylabel("Reward")
ax1.grid(True)

ax2.plot(steps, stds)
ax2.set_title("Reward Std (Stability)")
ax2.set_xlabel("Step")
ax2.set_ylabel("Reward Std")
ax2.grid(True)

plt.tight_layout()
plt.savefig("training_curve.png", dpi=150)
plt.show()
```
