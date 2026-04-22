import json
import os
import random
import re

try:
    import importlib.util as _ilu
    _pkg = os.path.dirname(
        _ilu.find_spec("openenv").submodule_search_locations[0]  # type: ignore
    )
    _iface_path = os.path.join(_pkg, "openenv", "core", "env_server", "interfaces.py")
    _types_path = os.path.join(_pkg, "openenv", "core", "env_server", "types.py")
    _tspec = _ilu.spec_from_file_location("_oe_types", _types_path)
    _tmod = _ilu.module_from_spec(_tspec)  # type: ignore
    _tspec.loader.exec_module(_tmod)  # type: ignore
    import sys
    sys.modules["openenv.core.env_server.types"] = _tmod
    _ispec = _ilu.spec_from_file_location("_oe_iface", _iface_path)
    _imod = _ilu.module_from_spec(_ispec)  # type: ignore
    _ispec.loader.exec_module(_imod)  # type: ignore
    OpenEnvEnvironment = _imod.Environment
except Exception:
    from typing import TypeVar, Generic
    from pydantic import BaseModel
    _A = TypeVar("_A")
    _O = TypeVar("_O")
    _S = TypeVar("_S")
    class OpenEnvEnvironment(Generic[_A, _O, _S]):  # type: ignore[no-redef]
        def __init__(self): pass

from models import Observation, Action, Reward, OpsSIMObservation, OpsSIMAction, OpsSIMState

AGENT_DOMAIN_MAP = {
    "AppOps": "app",
    "InfraOps": "infra",
    "DatabaseOps": "database",
    "NetworkOps": "network",
    "SecOps": "security",
    "MiddlewareOps": "middleware",
    "ObservabilityOps": "observability",
}

EXECUTION_AGENTS = list(AGENT_DOMAIN_MAP.keys())
IC_NAME = "IncidentCommander"
SUPERVISOR_NAME = "Supervisor"

CRITICAL_TERMS = ["failing", "offline", "dead", "severed", "down", "failed", "timeout",
                   "error", "critical", "crash_loop", "oom_killed", "corrupted", "compromised",
                   "exhausted", "broken", "contention", "exposed", "route_leak"]
MODERATE_TERMS = ["degraded", "overloaded", "stressed", "backed_up", "stalled", "pressure",
                  "stale", "flapping", "at_risk", "unknown", "slow", "high", "dropping",
                  "draining", "rerouting", "rotating", "partial"]
POSITIVE_TERMS = ["online", "healthy", "stable", "normal", "active", "serving", "restored",
                  "operational", "processing", "complete", "routing", "valid", "enforced",
                  "closed", "flowing", "passing", "resolved", "recovered"]


class DevOpsEnv(OpenEnvEnvironment[OpsSIMAction, OpsSIMObservation, OpsSIMState]):

    _DATA_CACHE = None
    _BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, seed=42, max_steps=15):
        super().__init__()
        self.rng = random.Random(seed)
        self.max_steps = max_steps
        self.scenario_index = 0
        self.state_data = {}
        self.observation = None
        self.last_action_error = None
        self.step_count = 0
        self.incident_channel = []
        self.communication_count = 0
        self.health_history = []
        self.consecutive_do_nothing = 0
        self.reward = None

        if DevOpsEnv._DATA_CACHE is None:
            dataset_path = os.path.join(DevOpsEnv._BASE_DIR, "tasks", "cascade.json")
            with open(dataset_path, "r") as f:
                DevOpsEnv._DATA_CACHE = json.load(f)["cascade_tasks_dataset"]

    def reset(self, seed=None, episode_id=None, **kwargs) -> Observation:
        self.step_count = 0
        self.last_action_error = None
        self.incident_channel = []
        self.communication_count = 0
        self.health_history = []
        self.consecutive_do_nothing = 0
        self.reward = None

        if seed is not None:
            self.rng = random.Random(seed)

        dataset = DevOpsEnv._DATA_CACHE
        scenario = dataset[self.scenario_index % len(dataset)]
        self.scenario_index += 1

        self.state_data = {
            "scenario_id": scenario.get("scenario_id", ""),
            "state": json.loads(json.dumps(scenario.get("initial_state", {}))),
            "penalties": scenario.get("penalties", {}).copy(),
            "optimal_solution_path": scenario.get("optimal_solution_path", []),
            "transition_rules": scenario.get("transition_rules", {}),
            "bleed_rules": scenario.get("bleed_rules", []),
            "sla_rules": scenario.get("sla_rules", {"required": [], "forbidden": []}),
            "sla_violation_penalty": scenario.get("sla_violation_penalty", -1.0),
            "domain_visibility": scenario.get("domain_visibility", {}),
            "action_domains": scenario.get("action_domains", {}),
            "root_cause": scenario.get("root_cause", ""),
            "root_cause_keywords": scenario.get("root_cause_keywords", []),
            "conflict_pairs": scenario.get("conflict_pairs", []),
            "severity_weights": scenario.get("severity_weights", []),
            "local_bleed_rules": scenario.get("local_bleed_rules", {}),
            "history": [],
            "action_outcomes": [],
        }

        available_actions = scenario.get("available_actions", [])
        if not available_actions:
            available_actions = list(self.state_data["optimal_solution_path"])
            available_actions.extend(list(self.state_data["penalties"].keys()))
        available_actions = sorted(list(set(a for a in available_actions if a != "do_nothing")))

        initial_health = self._compute_health_score(self.state_data["state"])
        self.health_history.append(initial_health)

        goal_state = self._compute_goal_state()
        progress = self._compute_progress()

        self.observation = Observation(
            available_actions=available_actions,
            system_state=self.state_data["state"],
            playbook_text=scenario.get("playbook_text", ""),
            logs=scenario.get("description", ""),
            step_count=self.step_count,
            incident_channel=[],
            goal_state=goal_state,
            progress=progress,
        )
        return self.observation

    def step(self, action: Action, timeout_s=None, **kwargs):
        action_str = action.action_type
        self.last_action_error = None

        if action_str == "communicate":
            return self._handle_communication(action)

        self.step_count += 1
        state = self.state_data["state"]
        prev_state = json.loads(json.dumps(state))
        agent = getattr(action, "agent", None)

        if action_str == "do_nothing":
            self.consecutive_do_nothing += 1
        else:
            self.consecutive_do_nothing = 0

        p_resp = self._calculate_responsibility_penalty(action_str, agent)
        if p_resp > 0:
            self.state_data["history"].append(action_str)
            self.state_data["action_outcomes"].append({"action": action_str, "result": "responsibility_violation"})
            self.last_action_error = f"responsibility_violation_{agent}_{action_str}"
            total = -p_resp
            self._update_observation(state, action_str, total)
            self.reward = Reward(value=total, responsibility_penalty=-p_resp)
            return self.observation, self.reward, True, {"reason": "responsibility_violation"}

        self._apply_state_transition(state, action_str)
        self.state_data["history"].append(action_str)

        delta_h = self._calculate_delta_health(state, prev_state)
        b_sys = self._calculate_global_bleed(state)
        b_loc = self._calculate_local_bleed(state)
        lambda_val = 1.0 / max(self.max_steps, 1)
        p_urg = lambda_val * self.step_count

        q_act = self._calculate_action_quality(action_str, state, prev_state)
        r_seq = self._calculate_sequencing_reward(action_str)
        p_conf = self._calculate_conflict_penalty(action_str)

        r_coord = self._calculate_coordination_reward(action)
        r_obs = self._calculate_observability_reward()
        r_sup = self._calculate_supervisor_reward(action)
        gamma_val = 1.0 / max(self.max_steps, 1)
        p_comm = gamma_val * self.communication_count

        sla_status = self._check_sla_compliance(state)
        r_succ = 0.0
        done = False

        if sla_status == "FAIL":
            r_succ = float(self.state_data.get("sla_violation_penalty", -2.0))
            done = True
        elif sla_status == "PASS":
            r_succ = 2.0
            done = True

        if self.step_count >= self.max_steps and not done:
            done = True

        stagnation = self._check_stagnation()

        total = (
            delta_h
            - (b_sys + b_loc)
            - p_urg
            + q_act
            + r_seq
            - p_resp
            - p_conf
            + r_coord
            + r_obs
            + r_sup
            - p_comm
            + r_succ
            + stagnation
        )

        outcome = "success" if r_succ > 0 else ("fail" if done else "step")
        self.state_data["action_outcomes"].append({"action": action_str, "result": outcome, "reward": total})

        self._update_observation(state, action_str, total)

        self.reward = Reward(
            value=total,
            delta_health=delta_h,
            global_bleed=-b_sys,
            local_bleed=-b_loc,
            urgency_penalty=-p_urg,
            action_quality=q_act,
            sequencing_reward=r_seq,
            responsibility_penalty=-p_resp,
            conflict_penalty=-p_conf,
            coordination_reward=r_coord,
            observability_reward=r_obs,
            supervisor_reward=r_sup,
            communication_cost=-p_comm,
            success_reward=r_succ,
        )
        return self.observation, self.reward, done, {"sla_status": sla_status}

    def _handle_communication(self, action):
        self.communication_count += 1
        message = getattr(action, "message", "") or ""
        agent = getattr(action, "agent", "unknown") or "unknown"
        self.incident_channel.append({"from": agent, "message": message})
        cost = -0.02
        self.observation.incident_channel = list(self.incident_channel)
        self.observation.step_count = self.step_count
        self.reward = Reward(value=cost, communication_cost=cost)
        return self.observation, self.reward, False, {"type": "communication"}

    def _compute_health_score(self, state):
        score = 0.0
        flat = self._flatten_state(state)
        for _, val in flat.items():
            if not isinstance(val, str):
                continue
            vl = val.lower()
            if any(t in vl for t in CRITICAL_TERMS):
                score -= 0.3
            elif any(t in vl for t in MODERATE_TERMS):
                score -= 0.1
            elif any(t in vl for t in POSITIVE_TERMS):
                score += 0.1
        return score

    def _calculate_delta_health(self, state, prev_state):
        curr = self._compute_health_score(state)
        prev = self._compute_health_score(prev_state)
        self.health_history.append(curr)
        return curr - prev

    def _calculate_global_bleed(self, state):
        severity_weights = self.state_data.get("severity_weights", [])
        if severity_weights:
            total = 0.0
            for sw in severity_weights:
                if self.evaluate_condition(state, sw.get("condition", "")):
                    total += float(sw.get("weight", 0.0))
            return total
        bleed = 0.0
        for rule in self.state_data.get("bleed_rules", []):
            if self.evaluate_condition(state, rule.get("condition", "")):
                bleed += abs(float(rule.get("penalty", 0.0)))
        return bleed

    def _calculate_local_bleed(self, state):
        total = 0.0
        local_rules = self.state_data.get("local_bleed_rules", {})
        for domain, rules in local_rules.items():
            for rule in rules:
                if self.evaluate_condition(state, rule.get("condition", "")):
                    total += abs(float(rule.get("penalty", 0.0)))
        return total

    def _calculate_action_quality(self, action_str, state, prev_state):
        if action_str == "do_nothing":
            return -0.3 * (self.consecutive_do_nothing ** 1.5)

        if action_str not in (self.observation.available_actions or []):
            self.last_action_error = f"invalid_action_{action_str}"
            return -0.5

        penalties = self.state_data.get("penalties", {})
        if action_str in penalties:
            pval = float(penalties[action_str])
            if pval <= -0.5:
                return pval

        tr = self.state_data.get("transition_rules", {}).get(action_str, {})
        if "reward" in tr:
            cond = tr.get("condition", "")
            if self.evaluate_condition(prev_state, cond):
                return float(tr.get("reward", 0.0))
            else:
                return float(tr.get("else_reward", -0.1))

        if json.dumps(prev_state, sort_keys=True) == json.dumps(state, sort_keys=True):
            return -0.2

        optimal = self.state_data.get("optimal_solution_path", [])
        if action_str in optimal:
            return 0.15

        return 0.0

    def _calculate_sequencing_reward(self, action_str):
        optimal = self.state_data.get("optimal_solution_path", [])
        if action_str not in optimal:
            return 0.0

        history = self.state_data.get("history", [])
        action_idx = optimal.index(action_str)

        prerequisites_met = 0
        prerequisites_total = action_idx
        for i in range(action_idx):
            if optimal[i] in history:
                prerequisites_met += 1

        if prerequisites_total == 0:
            return 0.15

        ratio = prerequisites_met / prerequisites_total
        if ratio >= 1.0:
            return 0.15
        elif ratio >= 0.5:
            return 0.05
        else:
            return -0.15

    def _calculate_responsibility_penalty(self, action_str, agent):
        if agent is None or agent in (IC_NAME, SUPERVISOR_NAME):
            return 0.0
        if action_str in ("do_nothing", "communicate"):
            return 0.0

        action_domains = self.state_data.get("action_domains", {})
        agent_domain = AGENT_DOMAIN_MAP.get(agent, "")
        domain_actions = action_domains.get(agent_domain, [])

        if action_str in domain_actions:
            return 0.0

        for domain, actions in action_domains.items():
            if action_str in actions and domain != agent_domain:
                return 5.0

        return 0.0

    def _calculate_conflict_penalty(self, action_str):
        history = self.state_data.get("history", [])
        if len(history) < 2:
            return 0.0

        prev_action = history[-2]
        conflict_pairs = self.state_data.get("conflict_pairs", [])

        for pair in conflict_pairs:
            if action_str in pair and prev_action in pair and action_str != prev_action:
                return 0.3

        if prev_action == action_str and action_str != "do_nothing":
            return 0.1

        return 0.0

    def _calculate_coordination_reward(self, action):
        if not getattr(action, "ic_directive", False):
            return 0.0

        target = getattr(action, "target_agent", None)
        action_str = action.action_type

        if not target or target not in EXECUTION_AGENTS:
            return -0.1

        action_domains = self.state_data.get("action_domains", {})
        target_domain = AGENT_DOMAIN_MAP.get(target, "")
        domain_actions = action_domains.get(target_domain, [])

        if action_str in domain_actions:
            return 0.15
        return -0.1

    def _calculate_observability_reward(self):
        root_cause_keywords = self.state_data.get("root_cause_keywords", [])
        if not root_cause_keywords:
            return 0.0

        obs_messages = [m for m in self.incident_channel if m.get("from") == "ObservabilityOps"]
        if not obs_messages:
            return 0.0

        best_score = 0.0
        for msg in obs_messages:
            text = msg.get("message", "").lower()
            matches = sum(1 for kw in root_cause_keywords if kw.lower() in text)
            if matches > best_score:
                best_score = matches

        if best_score >= 3:
            return 0.3
        elif best_score >= 2:
            return 0.2
        elif best_score >= 1:
            return 0.1
        return -0.05

    def _calculate_supervisor_reward(self, action):
        supervisor_approved = getattr(action, "supervisor_approved", None)
        if supervisor_approved is None:
            return 0.0

        action_str = action.action_type
        penalties = self.state_data.get("penalties", {})
        is_harmful = action_str in penalties and float(penalties.get(action_str, 0)) <= -0.3

        if not supervisor_approved and is_harmful:
            return 0.2
        elif supervisor_approved and is_harmful:
            return -0.2
        elif not supervisor_approved and not is_harmful:
            return -0.1
        return 0.0

    def _check_stagnation(self):
        n = 3
        if len(self.health_history) < n + 1:
            return 0.0
        recent = self.health_history[-(n + 1):]
        if all(h <= recent[0] + 0.01 for h in recent[1:]):
            streak = len(self.health_history) - n
            return -0.05 * min(streak, 5)
        return 0.0

    def _compute_goal_state(self):
        goals = {}
        required = self.state_data.get("sla_rules", {}).get("required", [])
        state = self.state_data.get("state", {})
        for cond in required:
            goals[cond] = self.evaluate_condition(state, cond)
        return goals

    def _compute_progress(self):
        required = self.state_data.get("sla_rules", {}).get("required", [])
        if not required:
            return 0.0
        state = self.state_data.get("state", {})
        met = sum(1 for c in required if self.evaluate_condition(state, c))
        return met / len(required)

    def _update_observation(self, state, action_str, step_reward):
        self.observation.system_state = state
        self.observation.step_count = self.step_count
        self.observation.incident_channel = list(self.incident_channel)
        self.observation.goal_state = self._compute_goal_state()
        self.observation.progress = self._compute_progress()
        if self.observation.logs is None:
            self.observation.logs = ""
        self.observation.logs += f"\nStep {self.step_count}: {action_str} -> Reward: {step_reward:.3f}"

    def _apply_state_transition(self, state, action_str):
        rules = self.state_data.get("transition_rules", {})
        if action_str in rules:
            rule = rules[action_str]
            condition = rule.get("condition", "")
            if self.evaluate_condition(state, condition):
                self.apply_effects(state, rule.get("effects", {}))
            elif "else_effects" in rule:
                self.apply_effects(state, rule["else_effects"])

    def _check_sla_compliance(self, state):
        sla_rules = self.state_data.get("sla_rules", {})
        forbidden_rules = sla_rules.get("forbidden", [])
        required_rules = sla_rules.get("required", [])

        for cond in forbidden_rules:
            if self.evaluate_condition(state, cond):
                return "FAIL"

        if required_rules:
            if all(self.evaluate_condition(state, cond) for cond in required_rules):
                return "PASS"

        return "INCOMPLETE"

    def _flatten_state(self, d, prefix=""):
        vals = {}
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                vals.update(self._flatten_state(v, full_key))
            else:
                vals[full_key] = v
        return vals

    def evaluate_condition(self, state, condition_string):
        if not condition_string:
            return True

        if " OR " in condition_string:
            return any(self.evaluate_condition(state, c.strip()) for c in condition_string.split(" OR "))
        if " AND " in condition_string:
            return all(self.evaluate_condition(state, c.strip()) for c in condition_string.split(" AND "))

        match = re.match(r"([\w\.]+)\s*(==|!=|<=|>=|<|>|IN)\s*(.+)", condition_string)
        if not match:
            if condition_string.strip() == "1 == 1":
                return True
            if condition_string.strip().lower() == "true":
                return True
            return False

        key, op, val_str = match.groups()

        parts = key.split(".")
        curr = state
        for p in parts:
            if isinstance(curr, dict) and p in curr:
                curr = curr[p]
            else:
                curr = None
                break

        if curr is None:
            return False

        curr_val = self._parse_condition_val(curr)

        if op == "IN":
            target_list = [self._parse_condition_val(v.strip().strip("[]'\"")) for v in val_str.split(",")]
            return curr_val in target_list

        target_val = self._parse_condition_val(val_str)

        if op == "==":
            return str(curr_val).lower() == str(target_val).lower()
        if op == "!=":
            return str(curr_val).lower() != str(target_val).lower()

        try:
            curr_f = float(curr_val)
            target_f = float(target_val)
            if op == "<=":
                return curr_f <= target_f
            if op == ">=":
                return curr_f >= target_f
            if op == "<":
                return curr_f < target_f
            if op == ">":
                return curr_f > target_f
        except (ValueError, TypeError):
            pass

        return False

    @staticmethod
    def _parse_condition_val(v):
        if isinstance(v, str):
            v = v.replace("$", "").replace(",", "").strip("'").strip('"')
            try:
                return float(v)
            except ValueError:
                if v.lower() == "true":
                    return True
                if v.lower() == "false":
                    return False
                return v
        return v

    def apply_effects(self, state, effects_dict):
        if not effects_dict:
            return
        for key, effect in effects_dict.items():
            parts = key.split(".")
            curr = state
            for p in parts[:-1]:
                if p not in curr:
                    curr[p] = {}
                curr = curr[p]
            target_key = parts[-1]

            if isinstance(effect, str) and effect.startswith("-") and effect[1:].replace(".", "").isdigit():
                current_val = float(str(curr.get(target_key, 0)).replace("%", "").replace("$", ""))
                curr[target_key] = max(0.0, current_val - float(effect[1:]))
            elif isinstance(effect, str) and effect.startswith("+") and effect[1:].replace(".", "").isdigit():
                current_val = float(str(curr.get(target_key, 0)).replace("%", "").replace("$", ""))
                curr[target_key] = current_val + float(effect[1:])
            else:
                curr[target_key] = effect

    def get_domain_observation(self, agent_name):
        domain = AGENT_DOMAIN_MAP.get(agent_name, "")
        visibility = self.state_data.get("domain_visibility", {})
        visible_keys = visibility.get(domain, [])
        state = self.state_data["state"]
        domain_state = {}
        for k in visible_keys:
            if k in state:
                domain_state[k] = state[k]
        action_domains = self.state_data.get("action_domains", {})
        agent_actions = action_domains.get(domain, [])

        filtered_actions = [a for a in agent_actions if a != "do_nothing"]

        return Observation(
            available_actions=filtered_actions if filtered_actions else [],
            domain_state=domain_state,
            playbook_text=self.observation.playbook_text if self.step_count == 0 else None,
            logs=self.observation.logs,
            step_count=self.step_count,
            agent=agent_name,
            incident_channel=list(self.incident_channel),
            goal_state=self._compute_goal_state(),
            progress=self._compute_progress(),
        )

    @property
    def state(self):
        return OpsSIMState(
            state_data=self.state_data,
            step_count=self.step_count,
            incident_channel=list(self.incident_channel),
        )

    def get_state(self):
        return {
            "state": self.state_data,
            "step_count": self.step_count,
            "incident_channel": list(self.incident_channel),
        }

    def close(self):
        return None
