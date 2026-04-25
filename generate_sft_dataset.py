import argparse
import copy
import json
import os
import random
import re
import sys
from collections import OrderedDict
from typing import Any


AGENT_DOMAIN_MAP = {
    "AppOps": "app",
    "InfraOps": "infra",
    "DatabaseOps": "database",
    "NetworkOps": "network",
    "SecOps": "security",
    "MiddlewareOps": "middleware",
    "ObservabilityOps": "observability",
}
DOMAIN_TO_AGENT = {domain: agent for agent, domain in AGENT_DOMAIN_MAP.items()}
VALID_AGENTS = set(AGENT_DOMAIN_MAP.keys())

CRITICAL_TERMS = [
    "failing", "offline", "dead", "down", "failed", "error", "crash",
    "crash_loop", "corrupted", "timeout", "broken", "compromised",
    "exhausted", "severed", "oom_killed", "exposed", "critical",
]
MODERATE_TERMS = [
    "degraded", "overloaded", "stressed", "slow", "partial", "stale",
    "flapping", "at_risk", "unknown", "high", "dropping", "draining",
    "rerouting", "rotating", "backed_up", "stalled", "pressure",
]
INVESTIGATE_PREFIXES = ("investigate_", "check_", "diagnose_", "inspect_", "analyze_", "trace_", "correlate_")


def parse_value(value: Any) -> Any:
    if isinstance(value, str):
        cleaned = value.replace("$", "").replace(",", "").strip("'").strip('"').strip()
        if cleaned.endswith("%"):
            cleaned = cleaned[:-1]
        try:
            return float(cleaned)
        except ValueError:
            if cleaned.lower() == "true":
                return True
            if cleaned.lower() == "false":
                return False
            return cleaned
    return value


def get_nested(state: dict[str, Any], key: str) -> Any:
    current = state
    for part in key.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def evaluate_condition(state: dict[str, Any], condition: str) -> bool:
    if not condition:
        return True
    expression = condition.strip()
    if expression in {"1 == 1", "true", "True"}:
        return True
    if " OR " in expression:
        return any(evaluate_condition(state, part.strip()) for part in expression.split(" OR "))
    if " AND " in expression:
        return all(evaluate_condition(state, part.strip()) for part in expression.split(" AND "))

    match = re.match(r"([\w\.]+)\s*(==|!=|<=|>=|<|>|IN)\s*(.+)", expression)
    if not match:
        return False
    key, operator, raw_target = match.groups()
    current = get_nested(state, key)
    if current is None:
        return False

    current_value = parse_value(current)
    if operator == "IN":
        candidates = [parse_value(item.strip().strip("[]'\"")) for item in raw_target.split(",")]
        return current_value in candidates

    target_value = parse_value(raw_target)
    if operator == "==":
        return str(current_value).lower() == str(target_value).lower()
    if operator == "!=":
        return str(current_value).lower() != str(target_value).lower()

    try:
        left = float(current_value)
        right = float(target_value)
    except (TypeError, ValueError):
        return False

    if operator == "<=":
        return left <= right
    if operator == ">=":
        return left >= right
    if operator == "<":
        return left < right
    if operator == ">":
        return left > right
    return False


def apply_effects(state: dict[str, Any], effects: dict[str, Any]) -> None:
    for key, effect in (effects or {}).items():
        current = state
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]

        target_key = parts[-1]
        if isinstance(effect, str) and effect[:1] in {"+", "-"} and effect[1:].replace(".", "").isdigit():
            previous = parse_value(current.get(target_key, 0))
            previous_number = float(previous) if isinstance(previous, (int, float)) else 0.0
            delta = float(effect[1:])
            current[target_key] = max(0.0, previous_number - delta) if effect[0] == "-" else previous_number + delta
        else:
            current[target_key] = effect


def flatten_state(state: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened = {}
    for key, value in (state or {}).items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten_state(value, full_key))
        else:
            flattened[full_key] = value
    return flattened


def detect_anomalies(state: dict[str, Any]) -> list[tuple[str, Any, str]]:
    anomalies = []
    for key, value in flatten_state(state).items():
        text = str(value).lower()
        if any(term in text for term in CRITICAL_TERMS):
            anomalies.append((key, value, "CRITICAL"))
        elif any(term in text for term in MODERATE_TERMS):
            anomalies.append((key, value, "DEGRADED"))
    return anomalies


def find_agent_for_action(action: str, action_domains: dict[str, list[str]]) -> tuple[str | None, str | None]:
    for domain, actions in (action_domains or {}).items():
        if action in actions:
            return DOMAIN_TO_AGENT.get(domain), domain
    return None, None


def is_feasible(state: dict[str, Any], transition_rules: dict[str, Any], action: str) -> bool:
    rule = (transition_rules or {}).get(action, {})
    return evaluate_condition(state, rule.get("condition", "1 == 1"))


def summarize_state(state: dict[str, Any]) -> str:
    rows = []
    for key, value in flatten_state(state).items():
        if key.startswith("discovered."):
            continue
        rows.append(f"  {key}: {value}")
    return "\n".join(rows[:80]) if rows else "  (empty)"


def summarize_actions_by_agent(
    available_actions: list[str],
    action_domains: dict[str, list[str]],
    transition_rules: dict[str, Any],
    state: dict[str, Any],
    completed: set[str],
) -> str:
    lines = []
    for agent, domain in AGENT_DOMAIN_MAP.items():
        rows = []
        for action in action_domains.get(domain, []):
            if action not in available_actions or action in completed or action == "do_nothing":
                continue
            tag = "FEASIBLE" if is_feasible(state, transition_rules, action) else "WAIT"
            rows.append(f"    - {action} [{tag}]")
        if rows:
            lines.append(f"  {agent}:")
            lines.extend(rows)
    return "\n".join(lines) if lines else "  (none)"


def classify_unsafe_actions(
    scenario: dict[str, Any],
    state: dict[str, Any],
    completed: set[str],
    gold_action: str,
) -> list[dict[str, str]]:
    penalties = scenario.get("penalties", {}) or {}
    transition_rules = scenario.get("transition_rules", {}) or {}
    available_actions = scenario.get("available_actions", []) or []
    optimal = scenario.get("optimal_solution_path", []) or []

    unsafe = []
    for action in available_actions:
        if action == gold_action or action in completed or action == "do_nothing":
            continue
        reason = ""
        penalty = float(penalties.get(action, 0.0))
        if penalty <= -0.3:
            reason = f"penalized action ({penalty:+.2f})"
        elif action in optimal and not is_feasible(state, transition_rules, action):
            reason = "premature optimal-path action; preconditions are not met"
        elif not is_feasible(state, transition_rules, action):
            reason = "preconditions are not met in the current state"

        if reason:
            agent, domain = find_agent_for_action(action, scenario.get("action_domains", {}) or {})
            unsafe.append({
                "action": action,
                "target_agent": agent or "AppOps",
                "domain": domain or "app",
                "reason": reason,
            })
    return unsafe


def build_prompt(
    scenario: dict[str, Any],
    state: dict[str, Any],
    step_idx: int,
    total_steps: int,
    completed: list[str],
    completed_rewards: list[float],
    candidate: dict[str, str] | None = None,
) -> str:
    available_actions = scenario.get("available_actions", []) or []
    action_domains = scenario.get("action_domains", {}) or {}
    transition_rules = scenario.get("transition_rules", {}) or {}
    sla_required = (scenario.get("sla_rules", {}) or {}).get("required", []) or []
    root_keywords = scenario.get("root_cause_keywords", []) or []

    anomalies = detect_anomalies(state)
    anomaly_text = "\n".join(f"  {key}: {value} [{severity}]" for key, value, severity in anomalies)
    if not anomaly_text:
        anomaly_text = "  None detected."

    unmet = [condition for condition in sla_required if not evaluate_condition(state, condition)]
    unmet_text = "\n".join(f"  - {condition}" for condition in unmet) if unmet else "  All met."

    completed_text = "  None yet."
    if completed:
        completed_text = "\n".join(
            f"  {index + 1}. {action} (reward={reward:+.2f})"
            for index, (action, reward) in enumerate(zip(completed, completed_rewards))
        )

    candidate_text = ""
    if candidate:
        candidate_text = (
            "\n\nCANDIDATE DIRECTIVE UNDER REVIEW:\n"
            f"  target_agent: {candidate['target_agent']}\n"
            f"  action: {candidate['action']}\n"
            f"  safety_note: {candidate['reason']}\n"
            "If this directive is unsafe or premature, choose a safer next_action instead."
        )

    return (
        "You are the Incident Commander for a multi-agent DevOps war room.\n"
        "Return ONLY one valid JSON object with exactly these keys: "
        "analysis, plan, next_action, target_agent, reasoning, confidence.\n"
        "Investigate before remediation, target the correct domain agent, avoid penalized actions, "
        "and prefer root-cause fixes over symptom chasing.\n\n"
        f"INCIDENT [{scenario.get('scenario_id', 'unknown')}]: {scenario.get('description', '')}\n\n"
        f"PLAYBOOK:\n{scenario.get('playbook_text', '')}\n\n"
        f"SYSTEM STATE:\n{summarize_state(state)}\n\n"
        f"ROOT CAUSE KEYWORDS: {', '.join(root_keywords) if root_keywords else '(none)'}\n\n"
        f"ANOMALIES:\n{anomaly_text}\n\n"
        f"AVAILABLE ACTIONS BY AGENT:\n"
        f"{summarize_actions_by_agent(available_actions, action_domains, transition_rules, state, set(completed))}\n\n"
        f"COMPLETED ACTIONS:\n{completed_text}\n\n"
        f"UNMET SLA GOALS:\n{unmet_text}\n\n"
        f"Step {step_idx}/{total_steps}."
        f"{candidate_text}"
    )


def build_analysis(state: dict[str, Any], scenario: dict[str, Any], step_idx: int, total_steps: int) -> str:
    anomalies = detect_anomalies(state)
    critical = [f"{key}={value}" for key, value, severity in anomalies if severity == "CRITICAL"]
    degraded = [f"{key}={value}" for key, value, severity in anomalies if severity == "DEGRADED"]
    evidence = ", ".join(critical[:3] or degraded[:3]) or "no obvious anomaly"
    keywords = scenario.get("root_cause_keywords", []) or []
    state_text = " ".join(str(value).lower() for value in flatten_state(state).values())
    matched = [keyword for keyword in keywords if keyword.lower() in state_text]
    matched_text = ", ".join(matched[:5]) if matched else "no root-cause keyword confirmed yet"
    phase = "investigation" if step_idx <= min(3, total_steps) else "remediation"
    return f"{phase.capitalize()} phase: current evidence shows {evidence}. Root-cause signal: {matched_text}."


def compute_confidence(step_idx: int, completed: list[str], completed_rewards: list[float], candidate: dict[str, str] | None) -> float:
    confidence = 0.52 + min(0.18, 0.04 * len([action for action in completed if action.startswith(INVESTIGATE_PREFIXES)]))
    confidence += min(0.15, 0.03 * len([reward for reward in completed_rewards if reward > 0]))
    if candidate:
        confidence += 0.08
    if step_idx == 1:
        confidence = min(confidence, 0.62)
    return round(min(0.94, confidence), 2)


def build_reasoning(
    action: str,
    target_agent: str,
    domain: str,
    scenario: dict[str, Any],
    state: dict[str, Any],
    completed: list[str],
    candidate: dict[str, str] | None,
) -> str:
    transition_rules = scenario.get("transition_rules", {}) or {}
    rule = transition_rules.get(action, {}) or {}
    condition = rule.get("condition", "1 == 1")
    effects = rule.get("effects", {}) or {}
    effect_text = ", ".join(f"{key}={value}" for key, value in effects.items()) if effects else "state verification"
    prefix = ""
    if candidate:
        prefix = f"Reject {candidate['action']} because {candidate['reason']}. "
    feasibility = "met" if evaluate_condition(state, condition) else "not met"
    completed_text = ", ".join(completed) if completed else "none"
    return (
        f"{prefix}Choose {action} for {target_agent} ({domain}) because condition '{condition}' is {feasibility}. "
        f"Expected effect: {effect_text}. Completed prerequisites: {completed_text}."
    )


def build_assistant(
    scenario: dict[str, Any],
    state: dict[str, Any],
    step_idx: int,
    total_steps: int,
    completed: list[str],
    completed_rewards: list[float],
    action: str,
    candidate: dict[str, str] | None = None,
) -> str:
    target_agent, domain = find_agent_for_action(action, scenario.get("action_domains", {}) or {})
    target_agent = target_agent or "AppOps"
    domain = domain or "app"
    optimal = scenario.get("optimal_solution_path", []) or []
    plan = [future for future in optimal[step_idx:] if future not in completed]
    payload = OrderedDict([
        ("analysis", build_analysis(state, scenario, step_idx, total_steps)),
        ("plan", plan),
        ("next_action", action),
        ("target_agent", target_agent),
        ("reasoning", build_reasoning(action, target_agent, domain, scenario, state, completed, candidate)),
        ("confidence", compute_confidence(step_idx, completed, completed_rewards, candidate)),
    ])
    return json.dumps(payload, ensure_ascii=False)


def build_example(
    scenario: dict[str, Any],
    state: dict[str, Any],
    step_idx: int,
    total_steps: int,
    completed: list[str],
    completed_rewards: list[float],
    gold_action: str,
    label_type: str,
    candidate: dict[str, str] | None = None,
) -> dict[str, Any]:
    prompt = build_prompt(scenario, state, step_idx, total_steps, completed, completed_rewards, candidate)
    unsafe_actions = classify_unsafe_actions(scenario, state, set(completed), gold_action)
    target_agent, domain = find_agent_for_action(gold_action, scenario.get("action_domains", {}) or {})
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": build_assistant(
                scenario, state, step_idx, total_steps, completed, completed_rewards, gold_action, candidate
            )},
        ],
        "metadata": {
            "scenario_id": scenario.get("scenario_id", "unknown"),
            "step_idx": step_idx,
            "total_steps": total_steps,
            "label_type": label_type,
            "gold_action": gold_action,
            "target_agent": target_agent or "AppOps",
            "domain": domain or "app",
            "candidate_action": candidate["action"] if candidate else "",
            "unsafe_actions": unsafe_actions[:8],
        },
    }


def generate_examples_for_scenario(scenario: dict[str, Any], max_contrast_per_step: int, rng: random.Random) -> list[dict[str, Any]]:
    examples = []
    state = copy.deepcopy(scenario.get("initial_state", {}) or {})
    optimal = scenario.get("optimal_solution_path", []) or []
    transition_rules = scenario.get("transition_rules", {}) or {}
    completed: list[str] = []
    completed_rewards: list[float] = []

    for index, action in enumerate(optimal):
        step_idx = index + 1
        total_steps = len(optimal)
        examples.append(build_example(
            scenario, state, step_idx, total_steps, completed, completed_rewards, action, "gold_next_action"
        ))

        unsafe = classify_unsafe_actions(scenario, state, set(completed), action)
        rng.shuffle(unsafe)
        for candidate in unsafe[:max_contrast_per_step]:
            examples.append(build_example(
                scenario, state, step_idx, total_steps, completed, completed_rewards, action, "contrast_rejection", candidate
            ))

        rule = transition_rules.get(action, {}) or {}
        reward = float(rule.get("reward", 0.1))
        if rule.get("effects"):
            apply_effects(state, rule["effects"])
        completed.append(action)
        completed_rewards.append(reward)

    return examples


def validate_examples(examples: list[dict[str, Any]]) -> bool:
    required = {"analysis", "plan", "next_action", "target_agent", "reasoning", "confidence"}
    failures = 0
    for index, example in enumerate(examples):
        try:
            messages = example["messages"]
            assert messages[0]["role"] == "user"
            assert messages[1]["role"] == "assistant"
            payload = json.loads(messages[1]["content"])
            missing = required - set(payload)
            assert not missing, f"missing {missing}"
            assert set(payload) == required, f"extra keys {set(payload) - required}"
            assert payload["target_agent"] in VALID_AGENTS
            assert isinstance(payload["plan"], list)
            confidence = float(payload["confidence"])
            assert 0.0 <= confidence <= 1.0
        except Exception as exc:
            failures += 1
            print(f"[FAIL] example={index} scenario={example.get('metadata', {}).get('scenario_id')} error={exc}")
    print(f"Validation: {len(examples) - failures} passed, {failures} failed")
    return failures == 0


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_grpo_prompts(path: str, examples: list[dict[str, Any]]) -> None:
    prompts = []
    seen = set()
    for example in examples:
        metadata = example["metadata"]
        if metadata["label_type"] != "gold_next_action":
            continue
        key = (metadata["scenario_id"], metadata["step_idx"])
        if key in seen:
            continue
        seen.add(key)
        prompts.append({
            "prompt": [{"role": "user", "content": example["messages"][0]["content"]}],
            "metadata": metadata,
        })
    write_jsonl(path, prompts)


def load_scenarios(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    scenarios = data.get("cascade_tasks_dataset") or data.get("scenarios") or []
    if not scenarios:
        raise ValueError("No scenarios found under 'cascade_tasks_dataset' or 'scenarios'.")
    return scenarios


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SFT and GRPO datasets for OpsSim-AI.")
    parser.add_argument("--input", default="tasks/cascade.json")
    parser.add_argument("--output-dir", default="data/generated")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-contrast-per-step", type=int, default=2)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--preview", type=int, default=0)
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(args.seed)
    scenarios = load_scenarios(args.input)
    rng.shuffle(scenarios)

    val_count = max(1, int(round(len(scenarios) * args.val_ratio))) if len(scenarios) > 1 else 0
    val_scenarios = scenarios[:val_count]
    train_scenarios = scenarios[val_count:]

    train_examples = []
    val_examples = []
    for scenario in train_scenarios:
        train_examples.extend(generate_examples_for_scenario(scenario, args.max_contrast_per_step, rng))
    for scenario in val_scenarios:
        val_examples.extend(generate_examples_for_scenario(scenario, args.max_contrast_per_step, rng))

    rng.shuffle(train_examples)
    rng.shuffle(val_examples)

    write_jsonl(os.path.join(args.output_dir, "sft_train.jsonl"), train_examples)
    write_jsonl(os.path.join(args.output_dir, "sft_val.jsonl"), val_examples)
    write_grpo_prompts(os.path.join(args.output_dir, "grpo_prompts.jsonl"), train_examples + val_examples)

    summary = {
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "train_scenarios": [scenario.get("scenario_id", "unknown") for scenario in train_scenarios],
        "val_scenarios": [scenario.get("scenario_id", "unknown") for scenario in val_scenarios],
        "max_contrast_per_step": args.max_contrast_per_step,
        "seed": args.seed,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, indent=2))
    if args.validate:
        ok = validate_examples(train_examples + val_examples)
        if not ok:
            sys.exit(1)

    if args.preview:
        for example in (train_examples + val_examples)[: args.preview]:
            print("\n--- USER ---")
            print(example["messages"][0]["content"])
            print("\n--- ASSISTANT ---")
            print(example["messages"][1]["content"])
            print("\n--- METADATA ---")
            print(json.dumps(example["metadata"], indent=2))


if __name__ == "__main__":
    main()
