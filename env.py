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


AGENT_DOMAIN_MAP = {"AppOps": "app", "InfraOps": "infra", "DatabaseOps": "database"}


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

        if DevOpsEnv._DATA_CACHE is None:
            dataset_path = os.path.join(DevOpsEnv._BASE_DIR, "tasks", "cascade.json")
            with open(dataset_path, "r") as f:
                DevOpsEnv._DATA_CACHE = json.load(f)["cascade_tasks_dataset"]

    def reset(self, seed=None, episode_id=None, **kwargs) -> Observation:
        self.step_count = 0
        self.last_action_error = None
        self.incident_channel = []
        self.communication_count = 0

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
            "history": [],
        }

        available_actions = scenario.get("available_actions", [])
        if not available_actions:
            available_actions = list(self.state_data["optimal_solution_path"])
            available_actions.extend(list(self.state_data["penalties"].keys()))
            available_actions.append("do_nothing")
        available_actions = sorted(list(set(available_actions)))

        self.observation = Observation(
            available_actions=available_actions,
            system_state=self.state_data["state"],
            playbook_text=scenario.get("playbook_text", ""),
            logs=scenario.get("description", ""),
            step_count=self.step_count,
            incident_channel=[],
        )
        return self.observation

    def step(self, action: Action, timeout_s=None, **kwargs):
        action_str = action.action_type
        self.last_action_error = None

        if action_str == "communicate":
            return self._handle_communication(action)

        self.step_count += 1
        state = self.state_data["state"]
        done = False

        action_penalty = 0.0
        if action_str not in self.observation.available_actions and action_str != "do_nothing":
            action_penalty -= 0.2
            self.last_action_error = f"invalid_action_{action_str}"

        if action_str == "do_nothing":
            action_penalty -= 0.3

        repeat_count = self.state_data["history"].count(action_str)
        repeat_penalty = 0.0
        if repeat_count > 0:
            repeat_penalty = -0.15 * repeat_count

        self.state_data["history"].append(action_str)

        if action_str in self.state_data["penalties"]:
            penalty_val = float(self.state_data["penalties"][action_str])
            action_penalty += penalty_val

            if penalty_val <= -0.8:
                bleed_loss = self._calculate_dynamic_bleed(state)
                urgency_penalty = -0.05 * self.step_count
                coordination_reward = self._calculate_coordination_reward(action)
                communication_cost = self._calculate_communication_cost()
                conflict_penalty = self._calculate_conflict_penalty(action_str)
                step_reward = (
                    bleed_loss + action_penalty + repeat_penalty + urgency_penalty
                    + coordination_reward + communication_cost + conflict_penalty
                    + self.state_data.get("sla_violation_penalty", -1.0)
                )
                self._update_observation(state, action_str, step_reward)
                reward = Reward(
                    value=step_reward, bleed=bleed_loss, action_penalty=action_penalty,
                    repeat_penalty=repeat_penalty, urgency_penalty=urgency_penalty,
                    communication_cost=communication_cost,
                    coordination_reward=coordination_reward,
                    conflict_penalty=conflict_penalty,
                )
                return self.observation, reward, True, {"reason": "guardrail_violation"}

        prev_state = json.loads(json.dumps(state))
        self._apply_state_transition(state, action_str)

        if json.dumps(prev_state, sort_keys=True) == json.dumps(state, sort_keys=True):
            action_penalty -= 0.2

        progress_reward = 0.0
        improvement_level = self._detect_positive_progress(prev_state, state)
        if improvement_level == "critical":
            progress_reward += 0.3
        elif improvement_level == "moderate":
            progress_reward += 0.1
        elif improvement_level == "minor":
            progress_reward += 0.05

        if self._detect_sla_improvement(prev_state, state):
            progress_reward += 0.2

        bleed_loss = self._calculate_dynamic_bleed(state)
        urgency_penalty = -0.05 * self.step_count
        coordination_reward = self._calculate_coordination_reward(action)
        conflict_penalty = self._calculate_conflict_penalty(action_str)
        communication_cost = self._calculate_communication_cost()
        success_reward = 0.0

        sla_status = self._check_sla_compliance(state)

        if sla_status == "FAIL":
            step_reward = (
                bleed_loss + action_penalty + repeat_penalty + urgency_penalty
                + progress_reward + coordination_reward + conflict_penalty
                + communication_cost + self.state_data.get("sla_violation_penalty", -1.0)
            )
            self._update_observation(state, action_str, step_reward)
            reward = Reward(
                value=step_reward, bleed=bleed_loss, action_penalty=action_penalty,
                repeat_penalty=repeat_penalty, urgency_penalty=urgency_penalty,
                progress_reward=progress_reward, communication_cost=communication_cost,
                coordination_reward=coordination_reward, conflict_penalty=conflict_penalty,
            )
            return self.observation, reward, True, {"reason": "sla_violation"}

        if sla_status == "PASS":
            success_reward = 1.0
            done = True

        if self.step_count >= self.max_steps:
            done = True

        step_reward = (
            bleed_loss + action_penalty + repeat_penalty + urgency_penalty
            + progress_reward + success_reward + coordination_reward
            + conflict_penalty + communication_cost
        )

        self._update_observation(state, action_str, step_reward)
        reward = Reward(
            value=step_reward, bleed=bleed_loss, action_penalty=action_penalty,
            repeat_penalty=repeat_penalty, urgency_penalty=urgency_penalty,
            progress_reward=progress_reward + success_reward,
            communication_cost=communication_cost,
            coordination_reward=coordination_reward, conflict_penalty=conflict_penalty,
        )
        return self.observation, reward, done, {}

    def _handle_communication(self, action):
        self.communication_count += 1
        message = getattr(action, "message", "") or ""
        agent = getattr(action, "agent", "unknown") or "unknown"
        self.incident_channel.append({"from": agent, "message": message})
        communication_cost = -0.02
        self.observation.incident_channel = list(self.incident_channel)
        self.observation.step_count = self.step_count
        reward = Reward(value=communication_cost, communication_cost=communication_cost)
        return self.observation, reward, False, {"type": "communication"}

    def _calculate_coordination_reward(self, action):
        agent = getattr(action, "agent", None)
        if not agent:
            return 0.0
        action_domains = self.state_data.get("action_domains", {})
        agent_domain = AGENT_DOMAIN_MAP.get(agent, "")
        domain_actions = action_domains.get(agent_domain, [])
        if action.action_type in domain_actions:
            return 0.1
        elif domain_actions:
            return -0.1
        return 0.0

    def _calculate_conflict_penalty(self, action_str):
        history = self.state_data["history"]
        if len(history) < 2:
            return 0.0
        prev = history[-2]
        if prev == action_str and action_str != "do_nothing":
            return -0.05
        return 0.0

    def _calculate_communication_cost(self):
        if self.communication_count > 0:
            return -0.01 * self.communication_count
        return 0.0

    def _update_observation(self, state, action_str, step_reward):
        self.observation.system_state = state
        self.observation.step_count = self.step_count
        self.observation.incident_channel = list(self.incident_channel)
        if self.observation.logs is None:
            self.observation.logs = ""
        self.observation.logs += f"\nStep {self.step_count}: {action_str} -> Reward: {step_reward:.2f}"

    def _apply_state_transition(self, state, action_str):
        rules = self.state_data.get("transition_rules", {})
        if action_str in rules:
            rule = rules[action_str]
            condition = rule.get("condition", "")
            if self.evaluate_condition(state, condition):
                self.apply_effects(state, rule.get("effects", {}))
            elif "else_effects" in rule:
                self.apply_effects(state, rule["else_effects"])

    def _calculate_dynamic_bleed(self, state):
        bleed = 0.0
        for rule in self.state_data.get("bleed_rules", []):
            if self.evaluate_condition(state, rule.get("condition", "")):
                bleed += float(rule.get("penalty", 0.0))
        return bleed

    def _detect_sla_improvement(self, prev_state, new_state):
        required_rules = self.state_data.get("sla_rules", {}).get("required", [])
        if not required_rules:
            return False
        prev_passed = sum(1 for c in required_rules if self.evaluate_condition(prev_state, c))
        new_passed = sum(1 for c in required_rules if self.evaluate_condition(new_state, c))
        return new_passed > prev_passed

    def _detect_positive_progress(self, prev_state, new_state):
        def extract_values(d, prefix=""):
            vals = {}
            for k, v in d.items():
                full_key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    vals.update(extract_values(v, full_key))
                else:
                    vals[full_key] = v
            return vals

        prev_vals = extract_values(prev_state)
        new_vals = extract_values(new_state)

        critical_terms = ["failing", "offline", "dead", "severed", "down", "failed", "timeout"]
        moderate_terms = ["degraded", "overloaded", "maxed", "dropped", "stalled", "pressure", "stressed"]
        positive_terms = ["online", "healthy", "stable", "complete", "normal", "routed", "restored", "passing", "active", "serving"]

        improvement = "none"

        for key, new_val in new_vals.items():
            prev_val = prev_vals.get(key)
            if prev_val == new_val or prev_val is None:
                continue

            prev_num = self._parse_num(prev_val)
            new_num = self._parse_num(new_val)

            if prev_num is not None and new_num is not None:
                if new_num < prev_num:
                    improvement = "moderate"

            if isinstance(prev_val, str) and isinstance(new_val, str):
                prev_str = prev_val.lower()
                new_str = new_val.lower()

                was_critical = any(t in prev_str for t in critical_terms)
                was_moderate = any(t in prev_str for t in moderate_terms)
                is_good_now = any(t in new_str for t in positive_terms)

                if was_critical and is_good_now:
                    return "critical"
                elif was_critical and not any(t in new_str for t in critical_terms):
                    return "critical"
                elif was_moderate and is_good_now:
                    if improvement != "critical":
                        improvement = "moderate"
                elif is_good_now and not was_critical and not was_moderate:
                    if improvement == "none":
                        improvement = "minor"

        return improvement

    @staticmethod
    def _parse_num(v):
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            try:
                return float(v.replace("%", "").replace("$", "").replace(",", "").replace("ms", "").strip())
            except ValueError:
                return None
        return None

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
        return Observation(
            available_actions=agent_actions + (["do_nothing"] if agent_actions else []),
            domain_state=domain_state,
            playbook_text=self.observation.playbook_text if self.step_count == 0 else None,
            logs=self.observation.logs,
            step_count=self.step_count,
            agent=agent_name,
            incident_channel=list(self.incident_channel),
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
