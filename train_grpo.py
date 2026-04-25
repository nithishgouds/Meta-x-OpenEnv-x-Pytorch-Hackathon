import argparse
import copy
import json
import os
import random
import re
import subprocess
from datetime import datetime
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from torch.utils.data import SequentialSampler
from transformers import AutoModelForCausalLM, TrainerCallback, set_seed
from trl import GRPOConfig, GRPOTrainer

from generate_sft_dataset import compute_file_hash, generate_examples_for_scenario, load_scenarios
from env import DevOpsEnv, AGENT_DOMAIN_MAP, EXECUTION_AGENTS
from models import OpsSIMAction
from training_logging import (
    append_jsonl as _append_jsonl,
    auto_generate_plots,
    install_console_logger,
    write_final_metrics,
)

DOMAIN_TO_AGENT = {domain: agent for agent, domain in AGENT_DOMAIN_MAP.items()}

metrics_log = []
latest_reward_snapshot: dict[str, Any] = {}
ENV_SCORER: "EnvScorer | None" = None
LAST_RAW_REWARDS: list[float] = []
# Path of the per-step quality_metrics JSONL file. Set in main() so the
# env_reward callback can flush each step's metrics incrementally and we
# never lose data on a mid-run crash.
QUALITY_LOG_PATH: str | None = None


def ensure_plot_dirs(output_dir: str) -> dict[str, str]:
    plot_dir = os.path.join(output_dir, "plot_data")
    os.makedirs(plot_dir, exist_ok=True)
    return {
        "root": plot_dir,
        "train": os.path.join(plot_dir, "train_metrics.jsonl"),
        "reward": os.path.join(plot_dir, "reward_components.jsonl"),
        "quality": os.path.join(plot_dir, "quality_metrics.jsonl"),
        "samples": os.path.join(plot_dir, "completion_samples.jsonl"),
        "summary": os.path.join(plot_dir, "summary.json"),
        "dataset": os.path.join(plot_dir, "dataset_profile.json"),
    }


def append_jsonl(path: str, payload: dict[str, Any]) -> None:
    # Delegate to the shared helper so incremental writes are flushed and
    # parent directories are created consistently across train_sft.py and
    # train_grpo.py.
    _append_jsonl(path, payload)


def summarize_prompt_dataset(dataset: Dataset) -> dict[str, float | int]:
    prompt_lengths = []
    unsafe_counts = []
    scenario_ids = set()
    for row in dataset:
        prompt_lengths.append(len(row["prompt"][0]["content"]))
        unsafe_counts.append(len(row.get("unsafe_actions", [])))
        scenario_ids.add(row.get("scenario_id", "unknown"))
    return {
        "num_examples": len(dataset),
        "num_scenarios": len(scenario_ids),
        "min_prompt_chars": min(prompt_lengths) if prompt_lengths else 0,
        "max_prompt_chars": max(prompt_lengths) if prompt_lengths else 0,
        "avg_prompt_chars": round(sum(prompt_lengths) / len(prompt_lengths), 2) if prompt_lengths else 0.0,
        "avg_unsafe_actions": round(sum(unsafe_counts) / len(unsafe_counts), 2) if unsafe_counts else 0.0,
    }


def update_reward_snapshot(name: str, rewards: list[float]) -> None:
    latest_reward_snapshot[name] = {
        "mean": round(sum(rewards) / len(rewards), 6) if rewards else 0.0,
        "min": round(min(rewards), 6) if rewards else 0.0,
        "max": round(max(rewards), 6) if rewards else 0.0,
        "count": len(rewards),
    }


class GRPOPlotMetricsCallback(TrainerCallback):
    def __init__(self, paths: dict[str, str]):
        self.paths = paths

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        payload = {
            "step": state.global_step,
            "epoch": float(state.epoch) if state.epoch is not None else None,
            **logs,
        }
        append_jsonl(self.paths["train"], payload)
        if latest_reward_snapshot:
            append_jsonl(
                self.paths["reward"],
                {
                    "step": state.global_step,
                    "epoch": float(state.epoch) if state.epoch is not None else None,
                    "timestamp": datetime.now().isoformat(),
                    "components": latest_reward_snapshot,
                },
            )


def resolve_precision(precision: str) -> torch.dtype:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.float32


def build_run_metadata(args: argparse.Namespace) -> dict[str, Any]:
    metadata = {"args": vars(args)}
    try:
        metadata["git_sha"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        metadata["git_sha"] = "unknown"
    if os.path.isfile(args.input):
        metadata["dataset_hash"] = compute_file_hash(args.input)
    if os.path.isfile(args.prompt_file):
        metadata["prompt_file_hash"] = compute_file_hash(args.prompt_file)
    return metadata


def build_model_and_peft(args: argparse.Namespace):
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=resolve_precision(args.precision),
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # When an SFT adapter is supplied we MERGE it into the base weights and then
    # attach a fresh GRPO LoRA on top. With this layout, GRPOTrainer's reference
    # model (computed by disabling the active adapter) is exactly the SFT
    # checkpoint, so the KL term anchors the policy to SFT — not to the raw
    # base model. The previous behavior loaded the SFT adapter as trainable,
    # which both anchored KL to the base model AND let GRPO overwrite the SFT
    # weights directly, causing unbounded drift away from SFT.
    if args.sft_adapter:
        peft_loaded = PeftModel.from_pretrained(model, args.sft_adapter, is_trainable=False)
        try:
            merged = peft_loaded.merge_and_unload()
        except Exception as exc:
            print(f"[GRPO][WARN] merge_and_unload failed ({exc}); falling back to trainable SFT adapter.")
            return PeftModel.from_pretrained(model, args.sft_adapter, is_trainable=True), None
        merged.config.use_cache = False
        merged.gradient_checkpointing_enable()
        merged.enable_input_require_grads()
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        return merged, peft_config

    model.enable_input_require_grads()
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    return model, peft_config


def extract_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
    if isinstance(completion, dict):
        return str(completion.get("content", completion))
    return str(completion)


def parse_json_object(text: str) -> dict[str, Any] | None:
    text = extract_text(text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def load_grpo_prompt_dataset(prompt_file: str) -> Dataset:
    rows = []
    with open(prompt_file, "r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            metadata = payload.get("metadata", {})
            rows.append({
                "prompt": payload["prompt"],
                "gold_action": metadata["gold_action"],
                "gold_target_agent": metadata["target_agent"],
                "unsafe_actions": [item["action"] for item in metadata.get("unsafe_actions", [])],
                "scenario_id": metadata["scenario_id"],
                "step_idx": metadata["step_idx"],
            })
    return Dataset.from_list(rows)


def build_grpo_dataset(input_path: str, max_contrast_per_step: int) -> Dataset:
    scenarios = load_scenarios(input_path)
    rows = []
    for scenario in scenarios:
        examples = generate_examples_for_scenario(
            scenario,
            max_contrast_per_step=max_contrast_per_step,
            rng=random.Random(42),
        )
        for example in examples:
            metadata = example["metadata"]
            if metadata["label_type"] != "gold_next_action":
                continue
            rows.append({
                "prompt": [{"role": "user", "content": example["messages"][0]["content"]}],
                "gold_action": metadata["gold_action"],
                "gold_target_agent": metadata["target_agent"],
                "unsafe_actions": [item["action"] for item in metadata.get("unsafe_actions", [])],
                "scenario_id": metadata["scenario_id"],
                "step_idx": metadata["step_idx"],
            })
    return Dataset.from_list(rows)


def payloads_from_completions(completions) -> list[dict[str, Any] | None]:
    return [parse_json_object(extract_text(completion)) for completion in completions]


# ---------------------------------------------------------------------------
# Environment-grounded reward
# ---------------------------------------------------------------------------
#
# The previous reward design scored completions only against static gold labels
# (parse, action_match, agent_match, unsafe_action, format). The DevOps
# environment defined in env.py — which actually models preconditions, SLA,
# bleed, sequencing, conflicts, urgency, and coordination — was never queried.
# That meant the model could only learn to imitate dataset labels, never to
# optimize the underlying simulator's reward.
#
# This module replays the scenario's prior optimal steps to reach the prompt's
# step_idx, then steps the env once with the model's predicted (action,
# target_agent) and reads the env's full Reward object. We then normalize per
# scenario using the same dynamic min/max formula already used in inference.py
# so that all 10 scenarios contribute on a comparable scale to the gradient.

def _calc_dynamic_min_reward(scenario: dict[str, Any], max_steps: int) -> float:
    worst_bleed = 0.0
    for sw in scenario.get("severity_weights", []) or []:
        worst_bleed += float(sw.get("weight", 0.0))
    for _domain, rules in (scenario.get("local_bleed_rules", {}) or {}).items():
        for rule in rules:
            worst_bleed += abs(float(rule.get("penalty", 0.0)))

    lambda_val = 1.0 / max(max_steps, 1)
    worst_urgency = sum(lambda_val * t for t in range(1, max_steps + 1))

    tr = scenario.get("transition_rules", {}) or {}
    worst_else = min((float(r.get("else_reward", 0)) for r in tr.values() if "else_reward" in r), default=-0.5)
    worst_q_act = min(worst_else, -0.5)

    worst_seq = -0.15
    conflict_pairs = scenario.get("conflict_pairs", []) or []
    worst_conf = 0.3 if conflict_pairs else 0.1

    sla_penalty = float(scenario.get("sla_violation_penalty", -2.0))
    worst_per_step = -worst_bleed - worst_urgency / max_steps + worst_q_act + worst_seq - worst_conf
    return (max_steps * worst_per_step) + sla_penalty


def _calc_dynamic_max_reward(scenario: dict[str, Any], max_steps: int) -> float:
    tr = scenario.get("transition_rules", {}) or {}
    total_action_quality = sum(max(0.0, float(r.get("reward", 0))) for r in tr.values())
    optimal_len = len(scenario.get("optimal_solution_path", []) or [])
    max_sequencing = 0.15 * optimal_len
    max_coordination = 0.15 * min(max_steps, len(tr))
    max_observability = 0.3
    max_supervisor = 0.2
    success_reward = 2.0
    return total_action_quality + max_sequencing + max_coordination + max_observability + max_supervisor + success_reward


def _domain_agent_for_action(action: str, scenario: dict[str, Any]) -> str:
    action_domains = scenario.get("action_domains", {}) or {}
    for domain, actions in action_domains.items():
        if action in actions:
            return DOMAIN_TO_AGENT.get(domain, "AppOps")
    return "AppOps"


class EnvScorer:
    """Replays each scenario up to step_idx and grades a candidate (action, agent)
    using DevOpsEnv.step. Returns per-scenario normalized reward in [-1.5, 1.5]
    plus the full Reward component breakdown for logging.
    """

    def __init__(self, scenarios_path: str, max_steps: int = 15) -> None:
        with open(scenarios_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        scenarios = data.get("cascade_tasks_dataset") or data.get("scenarios") or []
        self.scenarios: dict[str, dict[str, Any]] = {s["scenario_id"]: s for s in scenarios}
        self.scenario_order: list[str] = [s["scenario_id"] for s in scenarios]
        self.max_steps = max_steps
        self.norm: dict[str, tuple[float, float]] = {}
        for sid, sc in self.scenarios.items():
            mn = _calc_dynamic_min_reward(sc, max_steps)
            mx = _calc_dynamic_max_reward(sc, max_steps)
            if mx - mn < 1e-3:
                mx = mn + 1.0
            self.norm[sid] = (mn, mx)

    def difficulty(self, scenario_id: str) -> float:
        sc = self.scenarios.get(scenario_id, {})
        opt_len = len(sc.get("optimal_solution_path", []) or [])
        rule_count = len(sc.get("transition_rules", {}) or {})
        sev_count = len(sc.get("severity_weights", []) or [])
        conflict = len(sc.get("conflict_pairs", []) or [])
        return opt_len * 1.0 + rule_count * 0.1 + sev_count * 0.2 + conflict * 0.3

    def _make_env(self, scenario_id: str) -> DevOpsEnv | None:
        if scenario_id not in self.scenarios:
            return None
        env = DevOpsEnv(seed=42, max_steps=self.max_steps)
        # Pin env to the requested scenario rather than relying on round-robin order.
        try:
            env.scenario_index = self.scenario_order.index(scenario_id)
        except ValueError:
            return None
        env.reset()
        return env

    def _replay_prefix(self, env: DevOpsEnv, scenario: dict[str, Any], step_idx: int) -> bool:
        optimal = scenario.get("optimal_solution_path", []) or []
        prior = optimal[: max(0, int(step_idx) - 1)]
        for action in prior:
            agent = _domain_agent_for_action(action, scenario)
            try:
                _, _, done, _ = env.step(OpsSIMAction(
                    action_type=action,
                    agent=agent,
                    target_agent=agent,
                    ic_directive=True,
                    supervisor_approved=True,
                ))
            except Exception:
                return False
            if done:
                return False
        return True

    def score(
        self,
        scenario_id: str,
        step_idx: int,
        predicted_action: str,
        predicted_target_agent: str,
    ) -> tuple[float, dict[str, float], dict[str, bool]]:
        if not predicted_action:
            return -1.0, {}, {"valid": False, "stepped": False}

        scenario = self.scenarios.get(scenario_id)
        if scenario is None:
            return 0.0, {}, {"valid": False, "stepped": False}

        env = self._make_env(scenario_id)
        if env is None:
            return 0.0, {}, {"valid": False, "stepped": False}

        if not self._replay_prefix(env, scenario, step_idx):
            # Episode already terminated during prefix replay (e.g. SLA passed
            # mid-path). The candidate action gets no signal — return 0 so it
            # neither helps nor hurts the GRPO group baseline.
            return 0.0, {}, {"valid": True, "stepped": False}

        target = predicted_target_agent if predicted_target_agent in EXECUTION_AGENTS else None
        action_obj = OpsSIMAction(
            action_type=predicted_action,
            agent=target or _domain_agent_for_action(predicted_action, scenario) or "AppOps",
            target_agent=target or _domain_agent_for_action(predicted_action, scenario) or "AppOps",
            ic_directive=True,
            supervisor_approved=True,
        )
        try:
            _, reward, done, _info = env.step(action_obj)
        except Exception:
            return -1.0, {}, {"valid": False, "stepped": False}

        raw = float(getattr(reward, "value", 0.0))
        mn, mx = self.norm.get(scenario_id, (-5.0, 5.0))
        # The min/max from inference.py are FULL-EPISODE ranges (max_steps
        # accumulated). For a single env step we need to rescale by max_steps
        # so the per-step signal isn't squashed to a thin slice of [-1, 1].
        # This keeps cross-scenario equalization (different scenarios still
        # share a comparable per-step half-range) while giving GRPO enough
        # spread between good and bad single-step actions to learn from.
        half_full = max((mx - mn) / 2.0, 1.0)
        half_per_step = max(half_full / max(self.max_steps, 1), 0.5)
        norm = max(-1.5, min(1.5, raw / half_per_step))

        components = {
            "value_raw": raw,
            "action_quality": float(getattr(reward, "action_quality", 0.0)),
            "sequencing_reward": float(getattr(reward, "sequencing_reward", 0.0)),
            "coordination_reward": float(getattr(reward, "coordination_reward", 0.0)),
            "success_reward": float(getattr(reward, "success_reward", 0.0)),
            "global_bleed": float(getattr(reward, "global_bleed", 0.0)),
            "responsibility_penalty": float(getattr(reward, "responsibility_penalty", 0.0)),
            "urgency_penalty": float(getattr(reward, "urgency_penalty", 0.0)),
            "delta_health": float(getattr(reward, "delta_health", 0.0)),
        }
        flags = {
            "valid": True,
            "stepped": True,
            "done": bool(done),
            "success": float(getattr(reward, "success_reward", 0.0)) > 0.0,
            "sla_fail": float(getattr(reward, "success_reward", 0.0)) < 0.0,
        }
        return norm, components, flags


# ---------------------------------------------------------------------------
# Reward functions registered with the GRPOTrainer
# ---------------------------------------------------------------------------
#
# Reward decomposition (all components are added by GRPOTrainer):
#   env_reward          : main signal in roughly [-1.5, +1.5], normalized per
#                         scenario so easy and hard scenarios contribute on
#                         comparable scale.
#   parse_penalty       : -1.0 if completion is not valid JSON, else 0.0.
#                         Schema is already solved by SFT, so we no longer pay
#                         a saturated bonus for valid JSON; we only punish
#                         regression.
#   format_penalty      : -0.25 if required keys are missing, else 0.0.
#   unsafe_penalty      : -1.0 if predicted action is in the dataset's unsafe
#                         set (kept as an extra strict guardrail on top of
#                         whatever the env already says about the action).
# Total reward range ~ [-3.75, +1.5]. Decision quality (env signal) dominates
# the upside; safety/format dominate the downside.

REQUIRED_KEYS = {"analysis", "plan", "next_action", "target_agent", "reasoning", "confidence"}


def env_reward(completions, scenario_id, step_idx, gold_action, gold_target_agent, unsafe_actions, **_kwargs):
    if ENV_SCORER is None:
        raise RuntimeError("ENV_SCORER not initialized before reward call.")
    rewards: list[float] = []
    component_logs: dict[str, list[float]] = {
        "action_quality": [],
        "sequencing_reward": [],
        "coordination_reward": [],
        "success_reward": [],
        "global_bleed": [],
        "responsibility_penalty": [],
        "urgency_penalty": [],
        "delta_health": [],
        "value_raw": [],
    }
    quality = {"valid_json": 0, "correct_action": 0, "correct_agent": 0, "unsafe_action": 0,
               "env_success": 0, "env_sla_fail": 0, "stepped": 0, "n": 0}

    payloads = payloads_from_completions(completions)
    raw_scores: list[float] = []
    for payload, sid, sidx, gold_a, gold_ag, unsafe in zip(
        payloads, scenario_id, step_idx, gold_action, gold_target_agent, unsafe_actions
    ):
        quality["n"] += 1
        if payload is None:
            rewards.append(0.0)  # parse penalty applied separately; env signal is 0 if unparsable
            raw_scores.append(0.0)
            continue
        quality["valid_json"] += 1
        pred_a = str(payload.get("next_action", ""))
        pred_ag = str(payload.get("target_agent", ""))
        if pred_a == gold_a:
            quality["correct_action"] += 1
        if pred_ag == gold_ag:
            quality["correct_agent"] += 1
        if pred_a in set(unsafe or []):
            quality["unsafe_action"] += 1
        norm, comps, flags = ENV_SCORER.score(str(sid), int(sidx), pred_a, pred_ag)
        if flags.get("stepped"):
            quality["stepped"] += 1
        if flags.get("success"):
            quality["env_success"] += 1
        if flags.get("sla_fail"):
            quality["env_sla_fail"] += 1
        rewards.append(norm)
        raw_scores.append(comps.get("value_raw", 0.0))
        for k, lst in component_logs.items():
            lst.append(float(comps.get(k, 0.0)))

    update_reward_snapshot("env_reward", rewards)
    for name, vals in component_logs.items():
        if vals:
            update_reward_snapshot(f"env_{name}", vals)

    if quality["n"]:
        metrics_log.append({
            "timestamp": datetime.now().isoformat(),
            "valid_json_rate": quality["valid_json"] / quality["n"],
            "accuracy": quality["correct_action"] / quality["n"],
            "agent_accuracy": quality["correct_agent"] / quality["n"],
            "unsafe_rate": quality["unsafe_action"] / quality["n"],
            "env_success_rate": quality["env_success"] / quality["n"],
            "env_sla_fail_rate": quality["env_sla_fail"] / quality["n"],
            "env_stepped_rate": quality["stepped"] / quality["n"],
            "raw_reward_mean": (sum(raw_scores) / quality["n"]) if raw_scores else 0.0,
        })
        # Flush incrementally so quality_metrics.jsonl is recoverable mid-run.
        if QUALITY_LOG_PATH is not None:
            _append_jsonl(QUALITY_LOG_PATH, {
                "step": len(metrics_log),
                **metrics_log[-1],
            })
        update_reward_snapshot("quality_metrics", [
            metrics_log[-1]["valid_json_rate"],
            metrics_log[-1]["accuracy"],
            metrics_log[-1]["agent_accuracy"],
            1.0 - metrics_log[-1]["unsafe_rate"],
            metrics_log[-1]["env_success_rate"],
        ])

    LAST_RAW_REWARDS.clear()
    LAST_RAW_REWARDS.extend(raw_scores)
    return rewards


def parse_penalty(completions, **_kwargs):
    rewards = [0.0 if payload is not None else -1.0 for payload in payloads_from_completions(completions)]
    update_reward_snapshot("parse_penalty", rewards)
    return rewards


def format_penalty(completions, **_kwargs):
    rewards = []
    for payload in payloads_from_completions(completions):
        if payload is None:
            rewards.append(0.0)
            continue
        rewards.append(0.0 if REQUIRED_KEYS.issubset(payload.keys()) else -0.25)
    update_reward_snapshot("format_penalty", rewards)
    return rewards


def unsafe_penalty(completions, unsafe_actions, **_kwargs):
    rewards = []
    for payload, unsafe in zip(payloads_from_completions(completions), unsafe_actions):
        if payload is None:
            rewards.append(0.0)
            continue
        pred = str(payload.get("next_action", ""))
        rewards.append(-1.0 if pred in set(unsafe or []) else 0.0)
    update_reward_snapshot("unsafe_penalty", rewards)
    return rewards


# ---------------------------------------------------------------------------
# Curriculum: order training prompts easy -> hard so early gradients are clean
# ---------------------------------------------------------------------------

def build_curriculum_dataset(
    base_rows: list[dict[str, Any]],
    scorer: EnvScorer,
    target_steps: int,
    effective_batch: int,
    stage_fractions: tuple[float, float, float] = (0.25, 0.35, 0.40),
    seed: int = 42,
) -> Dataset:
    """Concatenate curriculum stages (easy -> medium -> all) into a single
    sequential dataset sized so that the trainer can run target_steps
    optimizer steps without re-shuffling difficulty in.
    """
    rng = random.Random(seed)
    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for row in base_rows:
        by_scenario.setdefault(row["scenario_id"], []).append(row)

    sids_by_difficulty = sorted(by_scenario.keys(), key=scorer.difficulty)
    n = len(sids_by_difficulty)
    if n == 0:
        return Dataset.from_list([])
    easy = sids_by_difficulty[: max(3, n // 3)]
    medium = sids_by_difficulty[: max(6, (2 * n) // 3)]
    full = sids_by_difficulty[:]

    rows_per_step = max(1, effective_batch)
    total_rows_needed = max(target_steps, 1) * rows_per_step
    stage_targets = [max(1, int(round(total_rows_needed * f))) for f in stage_fractions]
    # adjust last stage so totals match exactly
    stage_targets[-1] = max(1, total_rows_needed - sum(stage_targets[:-1]))

    def stage_rows(sids: list[str], target: int) -> list[dict[str, Any]]:
        pool = []
        for sid in sids:
            pool.extend(by_scenario.get(sid, []))
        if not pool:
            return []
        rng.shuffle(pool)
        out: list[dict[str, Any]] = []
        i = 0
        while len(out) < target:
            out.append(pool[i % len(pool)])
            i += 1
        return out

    ordered: list[dict[str, Any]] = []
    ordered.extend(stage_rows(easy, stage_targets[0]))
    ordered.extend(stage_rows(medium, stage_targets[1]))
    ordered.extend(stage_rows(full, stage_targets[2]))
    return Dataset.from_list(ordered)


class CurriculumGRPOTrainer(GRPOTrainer):
    """GRPOTrainer that walks the dataset sequentially so the curriculum order
    survives the dataloader. Without this override the default RandomSampler
    re-mixes hard scenarios into stage 1 and the curriculum is a no-op.
    """

    def _get_train_sampler(self, train_dataset=None):  # type: ignore[override]
        ds = train_dataset if train_dataset is not None else self.train_dataset
        return SequentialSampler(ds)


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Env-grounded GRPO refinement for OpsSim-AI.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--sft-adapter", default="", help="Path or HF repo id for the SFT LoRA adapter.")
    parser.add_argument("--input", default="tasks/cascade.json")
    parser.add_argument("--prompt-file", default="data/generated/grpo_prompts.jsonl")
    parser.add_argument("--output-dir", default="outputs/grpo-qwen2.5-3b")
    parser.add_argument("--hub-model-id", default="", help="Optional HF Hub repo id for pushing the GRPO adapter.")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=384)
    parser.add_argument("--max-prompt-length", type=int, default=1536)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--beta", type=float, default=0.05,
                        help="KL coefficient anchoring the policy to the SFT reference.")
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-contrast-per-step", type=int, default=2)
    parser.add_argument("--env-max-steps", type=int, default=15,
                        help="max_steps used when constructing the scoring DevOpsEnv.")
    parser.add_argument("--curriculum", action="store_true", default=True,
                        help="Order the dataset easy->medium->all (on by default).")
    parser.add_argument("--no-curriculum", dest="curriculum", action="store_false")
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.batch_size % args.num_generations != 0:
        raise ValueError("--batch-size must be divisible by --num-generations for GRPOTrainer.")

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = install_console_logger(args.output_dir, stage="grpo")
    print(f"[grpo] console mirrored to {log_path}", flush=True)
    plot_paths = ensure_plot_dirs(args.output_dir)
    global QUALITY_LOG_PATH
    QUALITY_LOG_PATH = plot_paths["quality"]
    with open(os.path.join(args.output_dir, "run_meta.json"), "w", encoding="utf-8") as handle:
        json.dump(build_run_metadata(args), handle, indent=2)

    if os.path.isfile(args.prompt_file):
        base_dataset = load_grpo_prompt_dataset(args.prompt_file)
    else:
        base_dataset = build_grpo_dataset(args.input, args.max_contrast_per_step)

    # Initialize env scorer once. Loading + caching all 10 normalization
    # ranges costs <1s and makes per-completion scoring just an env replay.
    global ENV_SCORER
    ENV_SCORER = EnvScorer(args.input, max_steps=args.env_max_steps)

    if args.curriculum:
        effective_batch = args.batch_size * args.grad_accum
        dataset = build_curriculum_dataset(
            base_rows=[base_dataset[i] for i in range(len(base_dataset))],
            scorer=ENV_SCORER,
            target_steps=args.max_steps,
            effective_batch=effective_batch,
            seed=args.seed,
        )
    else:
        dataset = base_dataset

    with open(plot_paths["dataset"], "w", encoding="utf-8") as handle:
        json.dump(summarize_prompt_dataset(dataset), handle, indent=2)

    model, peft_config = build_model_and_peft(args)

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        learning_rate=args.learning_rate,
        beta=args.beta,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        log_completions=True,
        bf16=args.precision == "bf16" and torch.cuda.is_available(),
        fp16=args.precision == "fp16" and torch.cuda.is_available(),
        gradient_checkpointing=True,
        report_to="none",
        push_to_hub=bool(args.hub_model_id),
        hub_model_id=args.hub_model_id or None,
        seed=args.seed,
    )

    trainer_cls = CurriculumGRPOTrainer if args.curriculum else GRPOTrainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=[
            env_reward,
            parse_penalty,
            format_penalty,
            unsafe_penalty,
        ],
        peft_config=peft_config,
        callbacks=[GRPOPlotMetricsCallback(plot_paths)],
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics_log, handle, indent=2)
    summary = {
        "num_examples": len(dataset),
        "final_global_step": trainer.state.global_step,
        "final_epoch": trainer.state.epoch,
        "log_history_entries": len(trainer.state.log_history),
        "last_quality_metrics": metrics_log[-1] if metrics_log else {},
        "scenario_normalization": ENV_SCORER.norm if ENV_SCORER else {},
        "curriculum_used": bool(args.curriculum),
        "kl_beta": args.beta,
    }
    with open(plot_paths["summary"], "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    for sample in dataset.select(range(min(10, len(dataset)))):
        append_jsonl(
            plot_paths["samples"],
            {
                "scenario_id": sample.get("scenario_id"),
                "step_idx": sample.get("step_idx"),
                "gold_action": sample.get("gold_action"),
                "gold_target_agent": sample.get("gold_target_agent"),
                "unsafe_actions": sample.get("unsafe_actions", []),
                "prompt_preview": sample["prompt"][0]["content"][:500],
            },
        )

    # Headline numbers for quick post-run inspection (and dashboards).
    final = {
        "stage": "grpo",
        "model": args.model,
        "sft_adapter": args.sft_adapter,
        "final_global_step": trainer.state.global_step,
        "final_epoch": trainer.state.epoch,
        "num_examples": len(dataset),
        "kl_beta": args.beta,
        "curriculum_used": bool(args.curriculum),
        "last_quality_metrics": metrics_log[-1] if metrics_log else {},
    }
    write_final_metrics(args.output_dir, final)

    # Auto-generate plots so HF Job artifacts include PNGs without a manual
    # step. Failures are logged but never raised — plots are not critical.
    auto_generate_plots(args.output_dir, stage="grpo")

    if args.hub_model_id:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
