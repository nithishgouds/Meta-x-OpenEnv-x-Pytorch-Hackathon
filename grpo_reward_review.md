# GRPO Training Review — `qwen25-3b`

Source plots: [plots/qwen25-3b-grpo](plots/qwen25-3b-grpo) and [plots/qwen25-3b-full](plots/qwen25-3b-full).
Reward implementation reviewed: [train_grpo.py](train_grpo.py#L210-L257).

---

## Findings

### What the plots show
- **[grpo_reward_components.png](plots/qwen25-3b-grpo/grpo_reward_components.png)**: `parse_reward` is pinned at **+0.4** for nearly every step, `format_reward` is pinned at **+0.15**. These two together are a flat ~0.55 baseline that contributes **no gradient signal**. `action_match_reward` swings between ~−0.25 and +1.0, `agent_match_reward` lives between 0.0 and +0.35, and `unsafe_action_penalty` oscillates between +0.1 and −0.7+ with no downward trend.
- **[grpo_reward_smoothed.png](plots/qwen25-3b-grpo/grpo_reward_smoothed.png)**: MA(5) reward rises from ~0.45 to a peak of ~1.4 around step 47, then **drifts back down to ~0.6–0.7** by step 100. The run is regressing in the second half.
- **[grpo_quality_metrics_smoothed.png](plots/qwen25-3b-grpo/grpo_quality_metrics_smoothed.png)**: `valid_json_rate` hits **0.9–1.0 within ~5 steps and stays there**. Action `accuracy` peaks at ~0.65 around step 47 and drifts down to ~0.35. `agent_accuracy` peaks at ~0.85 and drifts to ~0.5. `unsafe_rate` is noisy around 0.15–0.25 with spikes to 0.35, no learning trend.
- **[grpo_unsafe_rate.png](plots/qwen25-3b-grpo/grpo_unsafe_rate.png)**: per-batch unsafe rate frequently spikes to **0.5 and 0.75** even late in training (steps 70–100). Safety is not being learned.
- **[grpo_kl_only.png](plots/qwen25-3b-grpo/grpo_kl_only.png)**: KL averages ~1.2–1.5 with spikes to **2–3**. High and unstable for a 100-step LoRA refinement.
- **[grpo_final_quality_snapshot.png](plots/qwen25-3b-grpo/grpo_final_quality_snapshot.png)**: end of run = **valid_json 1.0, accuracy 0.25, agent_accuracy 0.25, unsafe_rate 0**. The very last batch happened to be safe but action/agent accuracy collapsed to 25%.
- **SFT plots** ([sft_train_loss_smoothed.png](plots/qwen25-3b-full/sft_train_loss_smoothed.png)): only ~15 logged steps, loss falling 1.9 → 0.5. SFT is short but healthy. Not the problem.

### What is working
- JSON parsing / schema compliance: solved within the first few steps and stays solved.
- KL is not exploding catastrophically (stays under ~3).
- The infrastructure (logging, generation, training loop) is fine.

### What is not working
- **The model is learning formatting, not decisions.** Valid-JSON ≈ 1.0 while action accuracy degrades from 0.65 → 0.25 in the second half. Textbook reward hacking on the easy component.
- Unsafe-action rate is not decreasing — late-run spikes to 0.5–0.75 are not penalized hard enough to be eliminated.
- Reward curve is regressing, not converging. The "best" policy was around step ~47, the final policy is worse.

---

## Reward Problems

### Components that are too strong (and saturated)
- **`parse_reward = +0.4 / −1.0`**: dominates the average reward once JSON parsing is solved. Contributes a flat +0.4 to nearly every sample → zero gradient. The +0.4 valid bonus is essentially **free reward** that crowds out the action signal in the GRPO group baseline.
- **`format_reward = +0.15 / −0.1`**: also saturated at +0.15 because once SFT keys are emitted they are always present. Pure noise contribution.
- **`unsafe_action_penalty: +0.1 safe / −1.0 unsafe`**: the **+0.1 safe bonus is wrong** — every safe completion (correct or incorrect) gets paid. Wrong-but-not-in-unsafe-set actions collect this, blunting the action-correctness signal.

### Components that are too weak
- **Action correctness**: swing of +1.0 vs −0.25 = 1.25. Dominant in theory, but with `num_generations=2` the advantage is computed against a single peer; combined with ~0.65 of saturated free reward, the relative SNR per group is poor.
- **Agent correctness**: swing only +0.35 vs −0.1 = 0.45. Routing matters as much as action choice in this multi-agent task; under-weighted.
- **Unsafe penalty as actually realized**: triggers only when the (often wrong) action is in the small `unsafe_actions` set. Most wrong actions are not "unsafe" and slip through with a +0.1 bonus.

### Net effect
The reward mix is **~0.55 in saturated formatting credit + ~0.1 free safety bonus = ~0.65 of decision-independent reward** out of a max of ~2.0. About a third of the reward signal carries no learning information. The optimizer chooses low-KL formatting moves and does not commit to action/agent correctness, which is why accuracy *decays* after the format is locked in.

**Conclusion: yes, the current GRPO design is over-indexed on JSON/format compliance and under-indexed on action quality, agent routing, and safety. The run is not good enough — final accuracy 0.25 with regressing trend says ship a redesigned reward and re-run.**

---

## Suggested Reward Changes

Goal: remove saturated free reward, make decision quality dominant, make safety strict.

| Component | Current | Proposed | Why |
|---|---|---|---|
| `parse_reward` (valid JSON bonus) | **+0.4** | **+0.05** | Schema is already solved; keep tiny anchor only. Eliminates ~0.35 of dead signal. |
| Invalid-JSON penalty | **−1.0** | **−1.5** | Sharper failure cliff so policy never drifts back into malformed output. |
| `action_match_reward` (correct) | **+1.0** | **+1.5** | Make decision quality clearly the biggest gradient component. |
| Wrong-action penalty | **−0.25** | **−0.75** | 6× stronger relative contrast vs. current; punishes "valid JSON, wrong answer" hacking. |
| `agent_match_reward` (correct) | **+0.35** | **+0.75** | Doubles routing signal; agent_accuracy was unstable and collapsing. |
| Wrong-agent penalty | **−0.1** | **−0.5** | Currently almost free to mis-route; raise the cost. |
| `unsafe_action_penalty` (unsafe) | **−1.0** | **−2.0** | Safety should be the single biggest single-step signal. Hard veto. |
| Safe-bonus | **+0.1** | **0.0** (remove) | Stops paying every harmless-but-wrong answer. |
| `format_reward` (all keys) | **+0.15 / −0.1** | **+0.0 / −0.25** | One-sided: no bonus for satisfying schema (saturated), only a penalty if keys are dropped. |

Net per-sample range becomes roughly **[−4.75, +2.30]** instead of **[−2.05, +2.0]**. The decision components (action 1.5 + agent 0.75 = **2.25**) make up >95% of the upside; safety is the dominant downside.

### Two extra implementation rules
1. When JSON fails to parse, return `−0.75` from `action_match_reward` and `agent_match_reward` (instead of `0.0`). Right now invalid JSON zeros these out — partially shielding bad outputs.
2. If `next_action` is unsafe, zero out the `action_match_reward` positive (don't pay correctness on an unsafe action even if it matches the gold label).

---

## Recommended Next Run

Given a hackathon-scale 3B model with LoRA r=16 and the redesigned reward:

| Hyperparameter | Current | Proposed | Reason |
|---|---|---|---|
| `--max-steps` | 100 | **300** | Peak was at step ~47 then degraded. Need more steps under a *stable* reward. |
| `--batch-size` (per-device) | 2 | **8** | Per-sample reward signal is noisy; bigger batches stabilize the GRPO group baseline. |
| `--grad-accum` | 4 | **2** | Compensates for the larger batch (effective batch ~16, similar order). |
| `--num-generations` | 2 | **4** | Single biggest stability win. With G=2 the advantage is just (a−b)/std of two samples — extreme variance. G=4 quadruples contrasts per prompt without quadrupling cost. |
| `--learning-rate` | 5e-6 | **2e-6** | KL spikes to 2–3 say the LR is too hot for LoRA refinement. |
| `--max-completion-length` | 512 | **384** | JSON outputs are short; freeing budget allows G=4. |
| `beta` (KL coefficient) | default | **0.05–0.1** | Explicit KL anchor; current run shows KL drift correlates with reward regression. |

### How these interact with the reward redesign
- Removing the saturated +0.4 parse and +0.15 format bonuses shrinks the reward floor, so the GRPO group advantage is no longer dominated by formatting noise. Smaller LR and bigger groups will actually move action/agent correctness instead of bouncing off a saturated baseline.
- Stronger wrong-action / wrong-agent / unsafe penalties widen the reward gap between siblings in a group — exactly what GRPO needs for clean advantages — so `num_generations=4` will yield meaningful contrasts per prompt instead of two near-identical formatted JSONs.
- 300 steps × G=4 × batch 8 / grad_accum 2 gives ~5–6× the effective gradient updates of the current run — what you need to push action accuracy past the current ~0.5 ceiling once formatting credit is removed.
- Lower LR + the one-sided format penalty keeps the model from forgetting the SFT-learned schema while it explores different `next_action` choices — addresses the late-run quality regression visible in [grpo_reward_smoothed.png](plots/qwen25-3b-grpo/grpo_reward_smoothed.png).

---

## Bluntly
The current run is **not good enough to ship**. It learned to write JSON and forgot to make decisions. Final action accuracy 0.25 with unsafe-rate spikes to 0.75 says: redesign the rewards as above, re-run with G=4 / LR=2e-6 / 300 steps, and only then evaluate. Reuse the existing plotting script — if the new run shows `parse_reward`/`format_reward` lines no longer flat-topping the components chart, and `accuracy`/`agent_accuracy` rising past 0.7 while `unsafe_rate` trends to <0.05, the redesign worked.
