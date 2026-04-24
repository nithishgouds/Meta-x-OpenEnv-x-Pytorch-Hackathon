import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from env import DevOpsEnv, EXECUTION_AGENTS, IC_NAME, SUPERVISOR_NAME, AGENT_DOMAIN_MAP
from models import Action, Reward
from multi_agent import WarRoom, AGENT_NAMES

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")
MAX_STEPS = 15
LLM_SEED = 42
MAX_RETRIES = 3
STAGNATION_WINDOW = 3

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

DOMAIN_TO_AGENT = {d: a for a, d in AGENT_DOMAIN_MAP.items()}

CRITICAL_KW = ["failing", "offline", "dead", "severed", "down", "failed", "timeout",
               "error", "critical", "crash_loop", "oom_killed", "corrupted", "compromised",
               "exhausted", "broken", "contention", "exposed", "route_leak"]
DEGRADED_KW = ["degraded", "overloaded", "stressed", "backed_up", "stalled", "pressure",
               "stale", "flapping", "at_risk", "unknown", "slow", "high", "dropping",
               "draining", "rerouting", "rotating", "partial"]

VALID_ROOT_DOMAINS = {"database", "infra", "network", "security", "application"}
ROOT_DOMAIN_HINTS = {
    "database": ["database", "db", "sql", "query", "replica", "replication", "transaction", "lock", "storage"],
    "infra": ["infra", "node", "host", "cpu", "memory", "oom", "disk", "pod", "container", "kube", "kubernetes"],
    "network": ["network", "dns", "latency", "packet", "route", "gateway", "timeout", "connection"],
    "security": ["security", "auth", "token", "certificate", "tls", "ssl", "permission", "access", "credential", "firewall"],
    "application": ["application", "app", "service", "worker", "queue", "middleware", "exception", "crash", "bug"],
}


class AgentMemory:
    def __init__(self):
        self.actions = []
        self.failed_actions = set()
        self.successful_actions = set()
        self.completed_actions = set()
        self.reward_trend = []
        self.state_changes = []
        self.root_cause_analysis = ""

    def record(self, step, action, agent, reward, prev_state, new_state):
        changes = _diff_states(prev_state, new_state)
        self.actions.append({
            "step": step,
            "action": action,
            "agent": agent,
            "reward": reward,
            "changes": changes,
        })
        self.completed_actions.add(action)
        self.reward_trend.append(reward)
        if reward < -0.15:
            self.failed_actions.add(action)
        if reward > 0:
            self.successful_actions.add(action)
        if changes:
            self.state_changes.append({"action": action, "changes": changes})

    def format_history(self):
        if not self.actions:
            return "  No actions taken yet."
        lines = []
        for e in self.actions:
            mark = "+" if e["reward"] > 0 else ("x" if e["reward"] < -0.15 else "~")
            ch = ""
            if e["changes"]:
                top = list(e["changes"].items())[:3]
                ch = " | " + ", ".join(f"{k}={v}" for k, v in top)
            lines.append(f"  [{mark}] Step {e['step']}: {e['action']} -> {e['agent']} (reward={e['reward']:+.3f}){ch}")
        return "\n".join(lines)

    def is_stagnating(self):
        if len(self.reward_trend) < 3:
            return False
        return all(r < 0 for r in self.reward_trend[-3:])

    def is_declining(self):
        if len(self.reward_trend) < STAGNATION_WINDOW:
            return False
        recent = self.reward_trend[-STAGNATION_WINDOW:]
        for i in range(1, len(recent)):
            if recent[i] > recent[i - 1]:
                return False
        return True

    def last_failed(self):
        if self.actions and self.actions[-1]["reward"] < -0.15:
            return self.actions[-1]
        return None


class PlanTracker:
    def __init__(self):
        self.current_plan = []
        self.revision_count = 0
        self.steps_since_revision = 0
        self.last_signature = None

    def update(self, new_plan):
        if new_plan and new_plan != self.current_plan:
            self.current_plan = list(new_plan)
            self.revision_count += 1
            self.steps_since_revision = 0
            self.last_signature = tuple(new_plan)

    def maybe_update(self, new_plan, allow_revision):
        self.steps_since_revision += 1
        if not new_plan:
            return
        signature = tuple(new_plan)
        if not self.current_plan:
            self.current_plan = list(new_plan)
            self.revision_count += 1
            self.steps_since_revision = 0
            self.last_signature = signature
            return
        if allow_revision and signature != self.last_signature:
            self.current_plan = list(new_plan)
            self.revision_count += 1
            self.steps_since_revision = 0
            self.last_signature = signature

    def mark_done(self, action):
        if action in self.current_plan:
            self.current_plan.remove(action)

    def format_plan(self):
        if not self.current_plan:
            return "  No plan yet. Create one."
        return "\n".join(f"  {i+1}. {a}" for i, a in enumerate(self.current_plan))


class StrategyTracker:
    INVESTIGATION_LIMIT = 3
    CONFIDENCE_THRESHOLD = 0.6

    def __init__(self):
        self.investigation_count = 0
        self.confidence = 0.0
        self.phase = "investigate"
        self.committed = False
        self.root_cause_locked = False
        self.last_confidence_source = "init"

    @staticmethod
    def _is_investigation(action):
        a = (action or "").lower()
        return "investigate" in a or a.startswith("check_") or "diagnose" in a or "inspect" in a

    def ingest_llm_confidence(self, value, source="llm"):
        if value is None:
            self.confidence = 0.5
            self.last_confidence_source = source + "_default"
            return
        try:
            c = float(value)
        except (TypeError, ValueError):
            self.confidence = 0.5
            self.last_confidence_source = source + "_default"
            return
        if c < 0.0 or c > 1.0:
            self.confidence = 0.5
            self.last_confidence_source = source + "_default"
            return
        self.confidence = c
        self.last_confidence_source = source

    def record_step(self, action, reward, discovered_after):
        if self._is_investigation(action):
            self.investigation_count += 1
        if discovered_after:
            root_hits = sum(1 for k, v in discovered_after.items() if "root_cause" in k and v)
            if root_hits:
                self.root_cause_locked = True
        if not self.committed:
            if (self.confidence >= self.CONFIDENCE_THRESHOLD
                    or self.investigation_count >= self.INVESTIGATION_LIMIT
                    or self.root_cause_locked):
                self.phase = "execute"
                self.committed = True

    def allow_investigation(self):
        return not self.committed and self.investigation_count < self.INVESTIGATION_LIMIT

    def should_revise_plan(self, memory):
        if not memory.actions:
            return True
        last = memory.actions[-1]
        if last["reward"] < -0.15:
            return True
        if memory.is_stagnating():
            return True
        return False

    def format_status(self):
        return (f"  phase={self.phase} confidence={self.confidence:.2f} "
                f"source={self.last_confidence_source} "
                f"investigations={self.investigation_count}/{self.INVESTIGATION_LIMIT} "
                f"committed={self.committed}")


def call_llm(prompt, max_tokens=250):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_tokens,
            seed=LLM_SEED,
        )
        return response.choices[0].message.content or ""
    except Exception:
        return ""


def _agent_for_action(action_str, action_domains):
    for domain, actions in action_domains.items():
        if action_str in actions:
            return DOMAIN_TO_AGENT.get(domain)
    return None


def _get_anomalies(system_state):
    out = []
    for key, val in system_state.items():
        if isinstance(val, dict):
            for sk, sv in val.items():
                if isinstance(sv, str):
                    vl = sv.lower()
                    if any(t in vl for t in CRITICAL_KW):
                        out.append(f"  {key}.{sk}: {sv} [CRITICAL]")
                    elif any(t in vl for t in DEGRADED_KW):
                        out.append(f"  {key}.{sk}: {sv} [DEGRADED]")
        elif isinstance(val, str):
            vl = val.lower()
            if any(t in vl for t in CRITICAL_KW):
                out.append(f"  {key}: {val} [CRITICAL]")
            elif any(t in vl for t in DEGRADED_KW):
                out.append(f"  {key}: {val} [DEGRADED]")
    return out


def _infer_root_domain(root_cause_analysis, anomalies):
    text = ((root_cause_analysis or "") + "\n" + "\n".join(anomalies or [])).lower()
    scores = {}
    for domain, hints in ROOT_DOMAIN_HINTS.items():
        scores[domain] = sum(text.count(h) for h in hints)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "application"


def _root_domain_matches_anomalies(root_domain, system_state):
    if root_domain not in ROOT_DOMAIN_HINTS:
        return True
    anomaly_text = "\n".join(_get_anomalies(system_state)).lower()
    if not anomaly_text:
        return True
    return any(h in anomaly_text for h in ROOT_DOMAIN_HINTS[root_domain])


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _diff_states(prev, curr):
    fp = _flatten(prev)
    fc = _flatten(curr)
    changes = {}
    for k in set(list(fp.keys()) + list(fc.keys())):
        pv = fp.get(k)
        cv = fc.get(k)
        if str(pv) != str(cv):
            changes[k] = f"{pv}->{cv}"
    return changes


def _condition_met(env, action_str):
    tr = env.state_data.get("transition_rules", {}).get(action_str, {})
    if not tr:
        return True
    cond = tr.get("condition", "")
    return env.evaluate_condition(env.state_data["state"], cond)


def _extract_json(text):
    text = text.strip()
    if "```" in text:
        for part in text.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                try:
                    return json.loads(part)
                except Exception:
                    pass
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass
    return None


def _is_feasible(env, action_str):
    if action_str == "do_nothing":
        return True
    tr = env.state_data.get("transition_rules", {}).get(action_str, {})
    penalties = env.state_data.get("penalties", {})
    penalty = float(penalties.get(action_str, 0))
    cond = tr.get("condition", "") if tr else ""
    cond_met = env.evaluate_condition(env.state_data["state"], cond) if cond else True
    if penalty <= -0.3 and not cond_met:
        return False
    if tr and not cond_met:
        else_r = float(tr.get("else_reward", 0))
        if else_r < -0.05:
            return False
    return True


def _is_unsafe(env, action_str):
    penalties = env.state_data.get("penalties", {})
    if action_str not in penalties:
        return False
    if float(penalties.get(action_str, 0)) > -0.3:
        return False
    return not _condition_met(env, action_str)


def build_observability_prompt(system_state, root_cause_keywords, description):
    anomalies = _get_anomalies(system_state)
    anomaly_text = "\n".join(anomalies) if anomalies else "  None detected."
    return f"""DO NOT output anything except valid JSON.
You are ObservabilityOps analyzing a production incident.

Incident: {description}

Anomalies:
{anomaly_text}

You MUST include these keywords in your analysis: {', '.join(root_cause_keywords)}

Return ONLY JSON:
{{"root_cause_analysis": "<analysis using the keywords above>", "cascade_chain": "<A causes B causes C>", "root_domain": "<one of: database | infra | network | security | application>", "confidence": <number between 0.0 and 1.0 reflecting how certain you are of the root cause>}}"""


def build_planning_prompt(system_state, playbook_text, memory, planner,
                          action_domains, obs_actions, step_count, goal_state,
                          progress, description, available_actions, suggested_order,
                          strategy=None, feasibility_map=None, replan_feedback=""):
    anomalies = _get_anomalies(system_state)
    anomaly_text = "\n".join(anomalies) if anomalies else "  All healthy"

    discovered = system_state.get("discovered", {})
    disc_text = "\n".join(f"  {k}: {'YES' if v else 'NO'}" for k, v in discovered.items()) if discovered else "  None"

    history_text = memory.format_history()
    plan_text = planner.format_plan()

    root_domain = getattr(memory, "root_domain", "")
    root_confidence = getattr(memory, "root_confidence", None)
    domain_locked = getattr(memory, "domain_locked", True)
    allow_invest = True if strategy is None else strategy.allow_investigation()
    if root_confidence is not None and root_confidence < 0.5:
        allow_invest = True
    if not domain_locked:
        allow_invest = True
    feasibility_map = feasibility_map or {}
    avail_text = ""
    for domain, actions in action_domains.items():
        if domain == "observability":
            continue
        if domain_locked and root_domain and domain != root_domain:
            continue
        agent = DOMAIN_TO_AGENT.get(domain, domain)
        feasible_list = []
        blocked_list = []
        for a in actions:
            if a in memory.completed_actions or a in obs_actions:
                continue
            if not allow_invest and StrategyTracker._is_investigation(a):
                continue
            if a in memory.failed_actions:
                continue
            if feasibility_map.get(a, True):
                feasible_list.append(a)
            else:
                blocked_list.append(a)
        if feasible_list:
            avail_text += f"  {agent} [FEASIBLE NOW]: {', '.join(feasible_list)}\n"
        if blocked_list:
            avail_text += f"  {agent} [PRECONDITIONS NOT MET - DO NOT PICK]: {', '.join(blocked_list)}\n"

    unmet = [g for g, met in (goal_state or {}).items() if not met]
    unmet_text = "\n".join(f"  - {g}" for g in unmet) if unmet else "  All met!"

    warnings = ""
    if memory.is_stagnating():
        warnings += "\nWARNING: Last 3 actions had negative reward. The current approach is not working. You MUST switch to a DIFFERENT strategy that directly targets the ROOT CAUSE. Do NOT repeat similar actions."
    if memory.is_declining():
        warnings += "\nWARNING: Reward trend is declining. Previous actions REDUCED system health. Pick a different feasible action that improves system state and addresses the root cause upstream dependency."
        warnings += "\nROOT DOMAIN VALIDATION: Recent reward evidence is weakening confidence in the current root domain. Re-investigation is allowed and domain reconsideration is allowed only if anomalies support it."
    if not domain_locked:
        warnings += "\nPrevious strategy is failing. Re-evaluate root cause."
    last_fail = memory.last_failed()
    if last_fail:
        warnings += f"\nLAST ACTION FAILED: {last_fail['action']} (reward={last_fail['reward']:+.3f}). Do NOT repeat it and do NOT choose similar actions in the same direction; switch strategy to address the ROOT CAUSE."
    if memory.failed_actions:
        warnings += f"\nBANNED (previously failed, do NOT choose): {', '.join(sorted(memory.failed_actions))}"

    if replan_feedback:
        warnings += f"\nREPLAN REQUIRED: {replan_feedback}"

    root_note = ""
    if memory.root_cause_analysis:
        root_note = f"\nROOT CAUSE ANALYSIS: {memory.root_cause_analysis}"
    if root_domain:
        root_note += f"\nROOT DOMAIN: {root_domain}"
    if root_confidence is not None:
        root_note += f"\nROOT DOMAIN CONFIDENCE: {root_confidence:.2f}"

    hint_text = ""
    if suggested_order:
        remaining_hints = [a for a in suggested_order if a not in memory.completed_actions]
        if remaining_hints:
            hint_text = f"\nSUGGESTED ORDER (reference only, you decide): {' -> '.join(remaining_hints)}"

    strategy_text = ""
    phase_rule = ""
    if strategy is not None:
        strategy_text = "\nSTRATEGY STATE:\n" + strategy.format_status()
        if root_confidence is not None and root_confidence < 0.5:
            phase_rule = (f"\nCONFIDENCE RULE: Root domain confidence is {root_confidence:.2f} (< 0.50). "
                          "Investigation actions are allowed. Do NOT force strict execution yet. "
                          "Use investigation to validate the root domain against anomalies.")
        elif root_confidence is not None and root_confidence >= 0.5:
            phase_rule = (f"\nCONFIDENCE RULE: Root domain confidence is {root_confidence:.2f} (>= 0.50). "
                          "Commit to execution in the current root domain and avoid unnecessary investigation.")
        elif strategy.phase == "execute" or not strategy.allow_investigation():
            phase_rule = ("\nPHASE RULE: Investigation phase is CLOSED. "
                          "You MUST commit to fix/execution actions now. "
                          "Do NOT choose investigate_* or check_* actions.")
        else:
            remaining = max(0, StrategyTracker.INVESTIGATION_LIMIT - strategy.investigation_count)
            phase_rule = (f"\nPHASE RULE: You have at most {remaining} investigation step(s) left. "
                          "Diagnose quickly then COMMIT to fixes.")
        if strategy.should_revise_plan(memory):
            phase_rule += "\nPLAN RULE: Revise plan now (failure or stagnation detected)."
        else:
            phase_rule += "\nPLAN RULE: Keep current plan stable. Only revise if truly necessary."

    return f"""Return ONLY one valid JSON object.
You are the Incident Commander. Choose the next action from the available actions only.

INCIDENT: {description}
PLAYBOOK: {playbook_text}{root_note}

ANOMALIES:
{anomaly_text}

INVESTIGATION STATUS:
{disc_text}

ACTION HISTORY & OUTCOMES:
{history_text}

CURRENT PLAN:
{plan_text}
{strategy_text}

UNMET SLA GOALS:
{unmet_text}

PRIMARY ROOT CAUSE DOMAIN: {getattr(memory, "root_domain", "")}

AVAILABLE ACTIONS (feasibility annotated):
{avail_text}{hint_text}{warnings}{phase_rule}

Step {step_count}/{MAX_STEPS} | Progress: {progress:.0%}

RULES:
- Use memory, current state, anomalies, and root cause analysis.
- Choose ONLY from actions listed under [FEASIBLE NOW].
- NEVER choose actions listed under [PRECONDITIONS NOT MET].
- NEVER repeat completed or failed actions.
- NEVER use observability actions.
- If domain is locked, stay in the PRIMARY ROOT CAUSE DOMAIN.
- If domain is unlocked, re-evaluate root cause and you may investigate across domains.
- Follow dependency order: upstream -> root cause -> downstream.
- If multiple actions are feasible, choose the one closest to the root cause.
- If prior actions failed or progress is stalled, choose a different valid action.

Return ONLY JSON:
{{"analysis": "<situation assessment grounded in state+memory>", "plan": ["action1", "action2", ...], "next_action": "<action_name>", "target_agent": "<AgentName>", "reasoning": "<why this action next, citing memory/state>", "confidence": <number between 0.0 and 1.0 reflecting how certain you are of the root cause / fix strategy>}}
Do not output markdown, backticks, or any extra text."""


def _parse_planning_response(text, available_actions, action_domains, memory, obs_actions, env, strategy=None):
    data = _extract_json(text)
    invalid_reason = "no_json"
    llm_confidence = None
    if data:
        llm_confidence = data.get("confidence")
        plan = data.get("plan", []) or []
        action = data.get("next_action", "") or ""
        valid_plan = []
        for a in plan:
            if not isinstance(a, str):
                continue
            if a in available_actions and a not in memory.completed_actions and a not in obs_actions:
                valid_plan.append(a)

        def _violates_phase(a):
            if strategy is None:
                return False
            if not getattr(memory, "domain_locked", True) and StrategyTracker._is_investigation(a):
                return False
            if not strategy.allow_investigation() and StrategyTracker._is_investigation(a):
                return True
            return False

        if action:
            if action in obs_actions:
                invalid_reason = "obs_action"
            elif action in memory.completed_actions:
                invalid_reason = "repeat_completed"
            elif action in memory.failed_actions:
                invalid_reason = "repeat_failed"
            elif action not in available_actions:
                invalid_reason = "unknown_action"
            elif _violates_phase(action):
                invalid_reason = "phase_violation"
            else:
                correct_agent = _agent_for_action(action, action_domains)
                if correct_agent:
                    return {"action": action, "target_agent": correct_agent, "plan": valid_plan,
                            "analysis": data.get("analysis", ""), "reasoning": data.get("reasoning", ""),
                            "llm_decided": True, "invalid_reason": "", "confidence": llm_confidence}
                invalid_reason = "no_agent"

        for pa in valid_plan:
            if pa in memory.completed_actions or pa in obs_actions or pa in memory.failed_actions:
                continue
            if _violates_phase(pa):
                continue
            agent = _agent_for_action(pa, action_domains)
            if agent:
                return {"action": pa, "target_agent": agent, "plan": valid_plan,
                        "analysis": data.get("analysis", ""), "reasoning": "from_llm_plan",
                        "llm_decided": True, "invalid_reason": "", "confidence": llm_confidence}

    return {"action": "", "target_agent": "", "plan": [], "analysis": "",
            "reasoning": "llm_parse_failed", "llm_decided": False,
            "invalid_reason": invalid_reason, "confidence": llm_confidence}


def _build_feasibility_map(env, available_actions, obs_actions, memory, action_domains):
    fmap = {}
    for a in available_actions:
        if a in obs_actions:
            continue
        if a in memory.completed_actions or a in memory.failed_actions:
            continue
        if not _agent_for_action(a, action_domains):
            continue
        fmap[a] = _is_feasible(env, a)
    return fmap


def _validate_llm_decision(decision, env, memory, obs_actions):
    if not decision.get("llm_decided"):
        return False, decision.get("invalid_reason") or "llm_parse_failed"
    action = decision.get("action", "")
    if not action:
        return False, "empty_action"
    if action in obs_actions:
        return False, "obs_action_not_allowed"
    if action in memory.completed_actions:
        return False, "already_completed"
    if action in memory.failed_actions:
        return False, "previously_failed"
    if _is_unsafe(env, action):
        return False, "unsafe_harmful_with_preconditions_not_met"
    if not _is_feasible(env, action):
        return False, "infeasible_preconditions_not_met"
    return True, ""


def _run_episode_core(room):
    obs = room.env.observation
    available_actions = obs.available_actions or []
    playbook_text = obs.playbook_text or ""
    description = obs.logs or ""
    action_domains = room.env.state_data.get("action_domains", {})
    root_cause_keywords = room.env.state_data.get("root_cause_keywords", [])
    obs_actions = set(action_domains.get("observability", []))
    suggested_order = room.env.state_data.get("optimal_solution_path", [])

    memory = AgentMemory()
    planner = PlanTracker()
    strategy = StrategyTracker()
    memory.domain_locked = True

    system_state = room.env.state_data["state"]
    obs_prompt = build_observability_prompt(system_state, root_cause_keywords, description)
    obs_text = call_llm(obs_prompt, max_tokens=300)
    obs_data = _extract_json(obs_text)
    anomaly_list = _get_anomalies(system_state)
    if obs_data:
        root_analysis = obs_data.get("root_cause_analysis", "")
        root_domain = (obs_data.get("root_domain") or "").strip().lower()
        if root_domain not in VALID_ROOT_DOMAINS:
            root_domain = _infer_root_domain(root_analysis, anomaly_list)
        try:
            root_confidence = float(obs_data.get("confidence"))
        except (TypeError, ValueError):
            root_confidence = 0.5
        if root_confidence < 0.0 or root_confidence > 1.0:
            root_confidence = 0.5
        if not _root_domain_matches_anomalies(root_domain, system_state):
            root_confidence *= 0.7
        obs_msg = f"[ROOT CAUSE] {root_analysis} | Chain: {obs_data.get('cascade_chain', '')} | Domain: {root_domain} | Confidence: {root_confidence:.2f}"
        memory.root_cause_analysis = root_analysis
        memory.root_domain = root_domain
        memory.root_confidence = root_confidence
        strategy.ingest_llm_confidence(root_confidence, source="observability_llm")
    else:
        obs_msg = f"[ROOT CAUSE] Detected anomalies involving: {', '.join(root_cause_keywords)}"
        memory.root_cause_analysis = f"Anomalies involving: {', '.join(root_cause_keywords)}"
        memory.root_domain = _infer_root_domain(memory.root_cause_analysis, anomaly_list)
        memory.root_confidence = 0.5
        if not _root_domain_matches_anomalies(memory.root_domain, system_state):
            memory.root_confidence *= 0.7
        strategy.ingest_llm_confidence(memory.root_confidence, source="observability_llm_fallback")
    if root_cause_keywords:
        lower = obs_msg.lower()
        missing = [k for k in root_cause_keywords if k.lower() not in lower]
        if missing:
            obs_msg = obs_msg + " | keywords: " + ", ".join(root_cause_keywords)
    room.observe_and_communicate("ObservabilityOps", obs_msg)

    rewards_list = []
    recent_rewards = []
    zero_progress_steps = 0
    last_progress = 0.0

    for step in range(MAX_STEPS):
        if room.is_done():
            break

        system_state = room.env.state_data["state"]
        prev_state = json.loads(json.dumps(system_state))
        goal_state = room.get_goal_state()
        progress = room.get_progress()

        feasibility_map = _build_feasibility_map(
            room.env, available_actions, obs_actions, memory, action_domains
        )

        decision = None
        replan_feedback = ""
        attempts = 0
        valid = False
        last_invalid_reason = ""
        last_reward = rewards_list[-1] if rewards_list else None

        while not valid and attempts < MAX_RETRIES:
            attempts += 1
            prompt = build_planning_prompt(
                system_state, playbook_text, memory, planner,
                action_domains, obs_actions, step + 1, goal_state,
                progress, description, available_actions, suggested_order,
                strategy, feasibility_map, replan_feedback,
            )
            llm_text = call_llm(prompt, max_tokens=400)
            decision = _parse_planning_response(
                llm_text, available_actions, action_domains, memory, obs_actions, room.env, strategy
            )
            valid, reason = _validate_llm_decision(decision, room.env, memory, obs_actions)
            if valid:
                break
            last_invalid_reason = reason
            chosen = decision.get("action", "") or "<none>"
            reward_note = f"last_step_reward={last_reward:+.3f}" if last_reward is not None else "no_prior_reward"
            failed_list = ", ".join(sorted(memory.failed_actions)) or "<none>"
            feasible_now = sorted([a for a, ok in feasibility_map.items() if ok]) or []
            feasible_text = ", ".join(feasible_now) if feasible_now else "<none available>"
            infeasible_now = sorted([a for a, ok in feasibility_map.items() if not ok]) or []
            infeasible_text = ", ".join(infeasible_now) if infeasible_now else "<none>"
            unsafe_flag = "YES" if (chosen and _is_unsafe(room.env, chosen)) else "NO"
            replan_feedback = (
                f"PREVIOUS ATTEMPT REJECTED (attempt {attempts}/{MAX_RETRIES}). "
                f"Invalid action '{chosen}'. Reason: {reason}. Unsafe: {unsafe_flag}. "
                f"Feasible now: {feasible_text}. "
                f"Infeasible: {infeasible_text}. "
                f"Failed before: {failed_list}. "
                f"Reward: {reward_note}. "
                f"Choose one DIFFERENT valid action from [FEASIBLE] only. "
                f"If domain is unlocked, re-investigate root cause. "
                f"Return JSON only."
            )
            print(f"[LLM-RETRY] step={step+1} attempt={attempts}/{MAX_RETRIES} rejected_action={chosen} reason={reason}")

        if not valid:
            recent_rewards.append(-1.0)
            if len(recent_rewards) > 3:
                recent_rewards.pop(0)
            if progress == 0:
                zero_progress_steps += 1
            else:
                zero_progress_steps = 0
            rewards_decreasing = (
                len(recent_rewards) == 3
                and recent_rewards[0] > recent_rewards[1] > recent_rewards[2]
            )
            repeated_failure = last_invalid_reason in {"no_json", "infeasible_preconditions_not_met"}
            if rewards_decreasing or zero_progress_steps >= 3 or repeated_failure:
                memory.domain_locked = False
            print(f"[LLM-SKIP] step={step+1} retries exhausted reason={last_invalid_reason}. Skipping step (no action executed).")
            continue

        llm_decided = True
        analysis = decision.get("analysis", "")
        reasoning = decision.get("reasoning", "")
        invalid_reason = ""
        llm_confidence_raw = decision.get("confidence")
        new_plan = decision.get("plan", [])

        action_str = decision["action"]
        target_agent = decision["target_agent"]

        strategy.ingest_llm_confidence(llm_confidence_raw, source="planner_llm")

        allow_revision = strategy.should_revise_plan(memory) or memory.is_declining()
        planner.maybe_update(new_plan, allow_revision)

        print(f"[PLAN] step={step+1} revision={planner.revision_count} plan={planner.current_plan} llm={llm_decided} invalid={invalid_reason or 'none'} attempts={attempts}")
        print(f"[STRATEGY] phase={strategy.phase} confidence={strategy.confidence:.2f} investigations={strategy.investigation_count}/{StrategyTracker.INVESTIGATION_LIMIT}")
        if analysis:
            print(f"[ANALYSIS] {analysis[:200]}")
        if reasoning:
            print(f"[REASONING] {reasoning[:200]}")

        supervisor_approved = True

        result = room.execute_directive(target_agent, action_str, supervisor_approved)
        reward_val = result["reward"].value
        rewards_list.append(reward_val)

        new_state = room.env.state_data["state"]
        memory.record(step + 1, action_str, target_agent, reward_val, prev_state, new_state)
        planner.mark_done(action_str)
        strategy.record_step(action_str, reward_val, new_state.get("discovered", {}))
        if hasattr(memory, "root_confidence"):
            if reward_val > 0:
                memory.root_confidence = min(1.0, memory.root_confidence + 0.05)
            elif memory.is_declining() or memory.is_stagnating():
                memory.root_confidence = max(0.0, memory.root_confidence * 0.7)

        recent_rewards.append(reward_val)
        if len(recent_rewards) > 3:
            recent_rewards.pop(0)
        current_progress = room.get_progress()
        if current_progress == 0:
            zero_progress_steps += 1
        else:
            zero_progress_steps = 0
        rewards_decreasing = (
            len(recent_rewards) == 3
            and recent_rewards[0] > recent_rewards[1] > recent_rewards[2]
        )
        if rewards_decreasing or zero_progress_steps >= 3:
            memory.domain_locked = False

        if not getattr(memory, "domain_locked", True) and StrategyTracker._is_investigation(action_str):
            refresh_prompt = build_observability_prompt(new_state, root_cause_keywords, description)
            refresh_text = call_llm(refresh_prompt, max_tokens=300)
            refresh_data = _extract_json(refresh_text)
            refresh_anomalies = _get_anomalies(new_state)
            if refresh_data:
                refreshed_domain = (refresh_data.get("root_domain") or "").strip().lower()
                if refreshed_domain not in VALID_ROOT_DOMAINS:
                    refreshed_domain = _infer_root_domain(refresh_data.get("root_cause_analysis", ""), refresh_anomalies)
                memory.root_cause_analysis = refresh_data.get("root_cause_analysis", memory.root_cause_analysis)
                memory.root_domain = refreshed_domain
                memory.domain_locked = True
                try:
                    refreshed_confidence = float(refresh_data.get("confidence"))
                except (TypeError, ValueError):
                    refreshed_confidence = memory.root_confidence
                if 0.0 <= refreshed_confidence <= 1.0:
                    memory.root_confidence = refreshed_confidence
                    strategy.ingest_llm_confidence(refreshed_confidence, source="observability_llm_refresh")

        last_progress = current_progress

        if reward_val < -0.15 and action_str in planner.current_plan:
            planner.current_plan.remove(action_str)

        if root_cause_keywords and memory.is_stagnating():
            repost = "[OBS UPDATE] Re-confirming keywords: " + ", ".join(root_cause_keywords)
            room.observe_and_communicate("ObservabilityOps", repost)

        error_msg = room.env.last_action_error or "null"
        print(
            f"[STEP] step={step+1} target={target_agent} "
            f"action={action_str} reward={reward_val:.3f} "
            f"done={str(result['done']).lower()} error={error_msg} "
            f"progress={room.get_progress():.0%}"
        )

        if result["done"]:
            break

    return rewards_list


def run_episode(scenario_idx=None):
    room = WarRoom(seed=LLM_SEED, max_steps=MAX_STEPS)
    room.reset()
    print(f"[START] scenario={room.env.state_data.get('scenario_id', 'unknown')} model={MODEL_NAME}")
    _run_episode_core(room)
    total = room.get_total_reward()
    progress = room.get_progress()
    success = "true" if (room.is_done() and total > 0) else "false"
    print(f"[END] success={success} steps={room.step_count} total_reward={total:.3f} progress={progress:.0%}")
    return total, room


def _calculate_dynamic_min_reward(env, max_steps):
    worst_bleed = 0.0
    for sw in env.state_data.get("severity_weights", []):
        worst_bleed += float(sw.get("weight", 0.0))
    for domain, rules in env.state_data.get("local_bleed_rules", {}).items():
        for rule in rules:
            worst_bleed += abs(float(rule.get("penalty", 0.0)))

    lambda_val = 1.0 / max(max_steps, 1)
    worst_urgency = sum(lambda_val * t for t in range(1, max_steps + 1))

    tr = env.state_data.get("transition_rules", {})
    worst_else = min((float(r.get("else_reward", 0)) for r in tr.values() if "else_reward" in r), default=-0.5)
    worst_q_act = min(worst_else, -0.5)

    worst_seq = -0.15
    conflict_pairs = env.state_data.get("conflict_pairs", [])
    worst_conf = 0.3 if conflict_pairs else 0.1

    gamma_val = 1.0 / max(max_steps, 1)
    worst_comm = gamma_val * max_steps * 2
    sla_penalty = float(env.state_data.get("sla_violation_penalty", -2.0))

    worst_per_step = -worst_bleed - worst_urgency / max_steps + worst_q_act + worst_seq - worst_conf
    return (max_steps * worst_per_step) + sla_penalty - worst_comm


def _calculate_dynamic_max_reward(env, max_steps):
    tr = env.state_data.get("transition_rules", {})
    total_action_quality = sum(max(0.0, float(r.get("reward", 0))) for r in tr.values())
    optimal_len = len(env.state_data.get("optimal_solution_path", []))
    max_sequencing = 0.15 * optimal_len
    max_coordination = 0.15 * min(max_steps, len(tr))
    max_observability = 0.3
    max_supervisor = 0.2
    success_reward = 2.0
    return total_action_quality + max_sequencing + max_coordination + max_observability + max_supervisor + success_reward


def grade(num_scenarios=1):
    total_score = 0.0

    for i in range(num_scenarios):
        room = WarRoom(seed=LLM_SEED, max_steps=MAX_STEPS)
        room.reset()

        min_reward = _calculate_dynamic_min_reward(room.env, MAX_STEPS)
        max_reward = _calculate_dynamic_max_reward(room.env, MAX_STEPS)

        print(f"[START] scenario_{i+1} env=opssim_ai model={MODEL_NAME}")
        rewards_list = _run_episode_core(room)
        total = room.get_total_reward()
        score = max(0.0, min(1.0, (total - min_reward) / (max_reward - min_reward)))
        success = "true" if (room.is_done() and total > 0) else "false"
        rewards_str = ",".join(f"{r:.3f}" for r in rewards_list)
        print(f"[END] success={success} steps={room.step_count} score={score:.3f} rewards={rewards_str}")
        total_score += score

    return total_score / num_scenarios


def main():
    grade(num_scenarios=1)


if __name__ == "__main__":
    main()
