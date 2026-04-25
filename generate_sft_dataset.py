import argparse
import copy
import json
import os
import re
import sys
from collections import OrderedDict


AGENT_DOMAIN_MAP = {
    "AppOps": "app",
    "InfraOps": "infra",
    "DatabaseOps": "database",
    "NetworkOps": "network",
    "SecOps": "security",
    "MiddlewareOps": "middleware",
    "ObservabilityOps": "observability",
}
DOMAIN_TO_AGENT = {v: k for k, v in AGENT_DOMAIN_MAP.items()}
VALID_AGENTS = set(AGENT_DOMAIN_MAP.keys())

CRITICAL_TERMS = ["failing", "offline", "dead", "down", "failed", "error", "crash",
                  "crash_loop", "corrupted", "timeout", "broken", "compromised",
                  "exhausted", "severed", "oom_killed", "exposed"]
MODERATE_TERMS = ["degraded", "overloaded", "stressed", "slow", "partial",
                  "stale", "flapping", "at_risk", "unknown", "high", "dropping",
                  "draining", "rerouting", "rotating", "backed_up", "stalled", "pressure"]

INVESTIGATE_PREFIXES = ("investigate_", "check_", "diagnose_", "inspect_")


def flatten_state(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten_state(v, key))
        else:
            out[key] = v
    return out


def detect_anomalies(state):
    out = []
    flat = flatten_state(state)
    for k, v in flat.items():
        if not isinstance(v, str):
            continue
        vl = v.lower()
        if any(t in vl for t in CRITICAL_TERMS):
            out.append((k, v, "CRITICAL"))
        elif any(t in vl for t in MODERATE_TERMS):
            out.append((k, v, "DEGRADED"))
    return out


def parse_value(v):
    if isinstance(v, str):
        s = v.replace("$", "").replace(",", "").strip("'").strip('"').strip()
        try:
            return float(s)
        except ValueError:
            if s.lower() == "true":
                return True
            if s.lower() == "false":
                return False
            return s
    return v


def get_nested(state, key):
    parts = key.split(".")
    curr = state
    for p in parts:
        if isinstance(curr, dict) and p in curr:
            curr = curr[p]
        else:
            return None
    return curr


def evaluate_condition(state, condition):
    if not condition:
        return True
    cs = condition.strip()
    if cs == "1 == 1" or cs.lower() == "true":
        return True
    if " OR " in cs:
        return any(evaluate_condition(state, c.strip()) for c in cs.split(" OR "))
    if " AND " in cs:
        return all(evaluate_condition(state, c.strip()) for c in cs.split(" AND "))
    m = re.match(r"([\w\.]+)\s*(==|!=|<=|>=|<|>|IN)\s*(.+)", cs)
    if not m:
        return False
    key, op, raw = m.groups()
    cur = get_nested(state, key)
    if cur is None:
        return False
    cur_v = parse_value(cur if not isinstance(cur, str) else cur)
    if op == "IN":
        items = [parse_value(x.strip().strip("[]'\"")) for x in raw.split(",")]
        return cur_v in items
    target = parse_value(raw)
    if op == "==":
        return str(cur_v).lower() == str(target).lower()
    if op == "!=":
        return str(cur_v).lower() != str(target).lower()
    try:
        a = float(cur_v); b = float(target)
        if op == "<=": return a <= b
        if op == ">=": return a >= b
        if op == "<":  return a <  b
        if op == ">":  return a >  b
    except (ValueError, TypeError):
        return False
    return False


def apply_effects(state, effects_dict):
    if not effects_dict:
        return
    for key, effect in effects_dict.items():
        parts = key.split(".")
        curr = state
        for p in parts[:-1]:
            if p not in curr or not isinstance(curr[p], dict):
                curr[p] = {}
            curr = curr[p]
        tk = parts[-1]
        if isinstance(effect, str) and effect.startswith("-") and effect[1:].replace(".", "").isdigit():
            cv = float(str(curr.get(tk, 0)).replace("%", "").replace("$", ""))
            curr[tk] = max(0.0, cv - float(effect[1:]))
        elif isinstance(effect, str) and effect.startswith("+") and effect[1:].replace(".", "").isdigit():
            cv = float(str(curr.get(tk, 0)).replace("%", "").replace("$", ""))
            curr[tk] = cv + float(effect[1:])
        else:
            curr[tk] = effect


def is_feasible(state, transition_rules, action):
    rule = (transition_rules or {}).get(action)
    if not rule:
        return True
    cond = rule.get("condition", "")
    if not cond:
        return True
    return evaluate_condition(state, cond)


def find_domain_for_action(action, action_domains):
    for domain, actions in (action_domains or {}).items():
        if action in actions:
            return domain
    return None


def find_agent_for_action(action, action_domains):
    domain = find_domain_for_action(action, action_domains)
    if domain is None:
        return None, None
    return DOMAIN_TO_AGENT.get(domain), domain


def build_user_prompt(scenario, state, step_idx, total_steps, completed, completed_rewards):
    sid = scenario.get("scenario_id", "unknown")
    desc = scenario.get("description", "")
    playbook = scenario.get("playbook_text", "")
    keywords = scenario.get("root_cause_keywords", []) or []
    available = scenario.get("available_actions", []) or []
    action_domains = scenario.get("action_domains", {}) or {}
    transition_rules = scenario.get("transition_rules", {}) or {}
    optimal = scenario.get("optimal_solution_path", []) or []
    sla_required = (scenario.get("sla_rules", {}) or {}).get("required", []) or []

    state_lines = []
    for k, v in (state or {}).items():
        if isinstance(v, dict):
            for sk, sv in v.items():
                state_lines.append(f"  {k}.{sk}: {sv}")
        else:
            state_lines.append(f"  {k}: {v}")
    state_text = "\n".join(state_lines) if state_lines else "  (empty)"

    anomalies = detect_anomalies(state)
    if anomalies:
        anomaly_text = "\n".join(f"  {k}: {v} [{tag}]" for k, v, tag in anomalies)
    else:
        anomaly_text = "  None detected."

    completed_set = set(completed)
    by_agent_lines = []
    for agent in AGENT_DOMAIN_MAP.keys():
        domain = AGENT_DOMAIN_MAP[agent]
        domain_actions = action_domains.get(domain, []) or []
        rows = []
        for a in domain_actions:
            if a in completed_set or a == "do_nothing":
                continue
            if a not in available:
                continue
            tag = "[FEASIBLE NOW]" if is_feasible(state, transition_rules, a) else "[PRECONDITIONS NOT MET]"
            rows.append(f"    - {a} {tag}")
        if rows:
            by_agent_lines.append(f"  {agent}:")
            by_agent_lines.extend(rows)
    actions_text = "\n".join(by_agent_lines) if by_agent_lines else "  (none)"

    if completed:
        comp_text = "\n".join(
            f"  {i+1}. {a} (reward={r:+.2f})" for i, (a, r) in enumerate(zip(completed, completed_rewards))
        )
    else:
        comp_text = "  None yet."

    unmet = [c for c in sla_required if not evaluate_condition(state, c)]
    unmet_text = "\n".join(f"  - {c}" for c in unmet) if unmet else "  All met."

    remaining = [a for a in optimal if a not in completed_set]
    suggested_text = " -> ".join(remaining) if remaining else "(none remaining)"

    progress = 0
    if sla_required:
        met = sum(1 for c in sla_required if evaluate_condition(state, c))
        progress = int(round(100 * met / len(sla_required)))

    instructions = (
        "Instructions: You are the Incident Commander. Output ONLY valid JSON. "
        "Steps 1-3: investigate first. Steps 4+: execute fixes targeting root cause. "
        "Follow upstream -> root cause -> downstream order."
    )

    prompt = (
        f"INCIDENT [{sid}]: {desc}\n\n"
        f"PLAYBOOK: {playbook}\n\n"
        f"SYSTEM STATE:\n{state_text}\n\n"
        f"ROOT CAUSE KEYWORDS: {', '.join(keywords) if keywords else '(none)'}\n\n"
        f"ANOMALIES:\n{anomaly_text}\n\n"
        f"AVAILABLE ACTIONS BY AGENT:\n{actions_text}\n\n"
        f"COMPLETED ACTIONS:\n{comp_text}\n\n"
        f"UNMET SLA GOALS:\n{unmet_text}\n\n"
        f"SUGGESTED ORDER: {suggested_text}\n\n"
        f"Step {step_idx}/{total_steps} | Progress: {progress}%\n\n"
        f"{instructions}"
    )
    return prompt


def compute_confidence(step_idx, completed, completed_rewards):
    if step_idx == 1:
        return 0.5
    conf = 0.5
    for a in completed:
        if a.startswith(INVESTIGATE_PREFIXES):
            conf += 0.08
    for r in completed_rewards:
        if r > 0:
            conf += 0.05
    return round(min(0.95, conf), 2)


def build_analysis(state, scenario, step_idx, total_steps, completed):
    anomalies = detect_anomalies(state)
    crit_summary = ", ".join(f"{k}={v}" for k, v, tag in anomalies if tag == "CRITICAL")[:240]
    if not crit_summary:
        crit_summary = "no remaining critical anomalies"
    keywords = scenario.get("root_cause_keywords", []) or []
    flat = flatten_state(state)
    text_blob = " ".join(str(v).lower() for v in flat.values() if isinstance(v, (str, int, float)))
    matched = [k for k in keywords if k.lower() in text_blob]
    matched_text = ", ".join(matched) if matched else "none yet"
    phase = "investigation" if step_idx <= 3 else "execution"
    if completed:
        comp_summary = ", ".join(completed)
    else:
        comp_summary = "none"
    return (
        f"State shows {crit_summary}. Root cause keywords {matched_text} detected. "
        f"Step {step_idx}/{total_steps}: {phase} phase. Previously completed: {comp_summary}."
    )


def build_reasoning(action, scenario, state, step_idx, completed, target_agent, domain):
    tr = (scenario.get("transition_rules", {}) or {}).get(action, {})
    optimal = scenario.get("optimal_solution_path", []) or []
    position = optimal.index(action) + 1 if action in optimal else step_idx
    prereqs = optimal[:position - 1]
    prereqs_done = [p for p in prereqs if p in completed]
    if tr and tr.get("condition"):
        cond = tr.get("condition", "")
        effects = tr.get("effects", {}) or {}
        reward = tr.get("reward", 0.0)
        eff_str = ", ".join(f"{k}={v}" for k, v in effects.items()) if effects else "(none)"
        return (
            f"{action} selected because condition '{cond}' is currently met. "
            f"Expected effect: {eff_str}. Expected reward: {reward}. "
            f"This is step {position} in the optimal path, prerequisites "
            f"{prereqs_done if prereqs_done else 'none'} are completed."
        )
    return (
        f"{action} is the next step in the optimal solution path. "
        f"Domain: {domain}. Agent: {target_agent}. "
        f"No precondition required."
    )


def generate_examples_for_scenario(scenario):
    examples = []
    state = copy.deepcopy(scenario.get("initial_state", {}) or {})
    optimal = scenario.get("optimal_solution_path", []) or []
    transition_rules = scenario.get("transition_rules", {}) or {}
    action_domains = scenario.get("action_domains", {}) or {}
    total_steps = len(optimal)
    completed = []
    completed_rewards = []

    feasible_count = 0
    total_count = 0

    for i, action in enumerate(optimal):
        step_idx = i + 1
        prompt = build_user_prompt(scenario, state, step_idx, total_steps, completed, completed_rewards)

        target_agent, domain = find_agent_for_action(action, action_domains)
        if target_agent is None:
            target_agent = "AppOps"
            domain = "app"

        plan = optimal[step_idx:]
        confidence = compute_confidence(step_idx, completed, completed_rewards)
        analysis = build_analysis(state, scenario, step_idx, total_steps, completed)
        reasoning = build_reasoning(action, scenario, state, step_idx, completed, target_agent, domain)

        feasible_now = is_feasible(state, transition_rules, action)
        total_count += 1
        if feasible_now:
            feasible_count += 1

        assistant_obj = OrderedDict([
            ("analysis", analysis),
            ("plan", plan),
            ("next_action", action),
            ("target_agent", target_agent),
            ("reasoning", reasoning),
            ("confidence", confidence),
        ])
        assistant_str = json.dumps(assistant_obj, ensure_ascii=False)

        examples.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": assistant_str},
            ]
        })

        rule = transition_rules.get(action, {}) or {}
        reward = float(rule.get("reward", 0.1))
        if rule.get("effects"):
            apply_effects(state, rule["effects"])
        completed.append(action)
        completed_rewards.append(reward)

    return examples, feasible_count, total_count


def validate_examples(examples, scenarios_by_index):
    required = ["analysis", "plan", "next_action", "target_agent", "reasoning", "confidence"]
    passed = 0
    failed = 0
    idx = 0
    for sc_i, scenario in enumerate(scenarios_by_index):
        optimal = scenario.get("optimal_solution_path", []) or []
        available = set(scenario.get("available_actions", []) or [])
        for _ in optimal:
            ex = examples[idx]
            ok = True
            err = ""
            try:
                msgs = ex["messages"]
                assert msgs[0]["role"] == "user"
                assert msgs[1]["role"] == "assistant"
                obj = json.loads(msgs[1]["content"])
                for f in required:
                    if f not in obj:
                        ok = False; err = f"missing field {f}"; break
                if ok:
                    if obj["next_action"] not in available and obj["next_action"] not in optimal:
                        ok = False; err = f"next_action {obj['next_action']} not in available_actions"
                if ok and obj["target_agent"] not in VALID_AGENTS:
                    ok = False; err = f"invalid target_agent {obj['target_agent']}"
                if ok and not isinstance(obj["plan"], list):
                    ok = False; err = "plan not a list"
                if ok and not (0.0 <= float(obj["confidence"]) <= 1.0):
                    ok = False; err = "confidence out of range"
            except Exception as e:
                ok = False; err = f"exception: {e}"
            tag = "PASS" if ok else "FAIL"
            print(f"  [{tag}] scenario={scenario.get('scenario_id','?')} step={idx} {('('+err+')') if not ok else ''}")
            if ok:
                passed += 1
            else:
                failed += 1
            idx += 1
    print(f"\nValidation: {passed} passed, {failed} failed (total {passed+failed})")
    return failed == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="./tasks/cascade.json")
    ap.add_argument("--out-jsonl", default="sft_dataset.jsonl")
    ap.add_argument("--out-hf", default="sft_dataset_hf.json")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--preview", type=int, default=0)
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    scenarios = data.get("cascade_tasks_dataset") or data.get("scenarios") or []
    if not scenarios:
        print("ERROR: no scenarios found under 'cascade_tasks_dataset'", file=sys.stderr)
        sys.exit(1)

    all_examples = []
    per_scenario_counts = []
    total_feasible = 0
    total_actions = 0
    confidences = []

    for scenario in scenarios:
        examples, feas, tot = generate_examples_for_scenario(scenario)
        all_examples.extend(examples)
        per_scenario_counts.append((scenario.get("scenario_id", "unknown"), len(examples)))
        total_feasible += feas
        total_actions += tot
        for ex in examples:
            try:
                confidences.append(json.loads(ex["messages"][1]["content"])["confidence"])
            except Exception:
                pass

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(args.out_hf, "w", encoding="utf-8") as f:
        json.dump({"train": all_examples}, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(all_examples)} examples across {len(scenarios)} scenarios")
    for sid, c in per_scenario_counts:
        print(f"Scenario {sid}: {c} examples")
    if confidences:
        print(f"Average confidence progression: {confidences[0]:.2f} -> {confidences[-1]:.2f}")
    print(f"Actions covered: {total_feasible}/{total_actions} have [FEASIBLE NOW] annotation")

    if args.preview > 0:
        print("\n===== PREVIEW =====")
        for ex in all_examples[: args.preview]:
            print("\n--- USER ---")
            print(ex["messages"][0]["content"])
            print("\n--- ASSISTANT ---")
            print(json.dumps(json.loads(ex["messages"][1]["content"]), indent=2))
            print("--- END ---")

    if args.validate:
        print("\n===== VALIDATION =====")
        validate_examples(all_examples, scenarios)


if __name__ == "__main__":
    main()
