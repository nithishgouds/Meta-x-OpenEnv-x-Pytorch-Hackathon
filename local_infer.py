import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from env import DevOpsEnv
from models import Action

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
MAX_STEPS = 8
LLM_SEED = 42

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

class LLMParser:
    def __init__(self):
        self.client = client
        self.model = MODEL_NAME
        self.cache = {}

    def parse(self, observation, action_history):
        key = self._build_cache_key(observation, action_history)
        if key in self.cache:
            return self.cache[key]

        if observation.task_type == "easy":
            result = self._parse_easy(observation)
        elif observation.task_type == "medium":
            result = self._parse_medium(observation)
        else:
            result = self._parse_hard(observation, action_history)

        if result[1] >= 0.7:
             self.cache[key] = result
        return result

    def _parse_easy(self, obs):
        prompt = self._build_easy_prompt(obs)
        response = self._call_llm(prompt)
        return self._parse_easy_response(response, obs)

    def _build_easy_prompt(self, obs) -> str:
        return f"""
DO NOT output anything except valid JSON.

You are an automated DevOps recovery agent. Your objective is to resolve a system alert by executing a single configuration fix.

1. Analyze the panicked user message and the specific error logs.
2. Review the current config state.
3. Identify and execute the single correct configuration action that resolves the root cause.
4. BEWARE of "red herring" actions: The user might blame the wrong system (e.g., UI changes) when the logs indicate a deeper infrastructure failure.

User Message: "{obs.user_message}"
Logs: {obs.logs}

Current Config State:
{json.dumps(obs.config, indent=2)}

Available Actions:
{json.dumps(obs.available_actions)}

Return ONLY JSON:
{{
  "action": "<exact_string_from_available_actions>"
}}
"""

    def _parse_easy_response(self, text, obs):
        try:
            data = json.loads(text)
            action = data.get("action", "")
            if action in obs.available_actions:
                return action, 1.0, None
            else:
                return "do_nothing", 0.0, None
        except Exception:
            return "do_nothing", 0.0, None

    def _parse_medium(self, obs):
        prompt = self._build_medium_prompt(obs)
        response = self._call_llm(prompt)
        return self._parse_medium_response(response, obs)

    def _build_medium_prompt(self, obs):
        return f"""
DO NOT output anything except valid JSON.

You are an AI agent diagnosing a production issue in a dynamic environment.

- **Logs (Snapshot):** The Logs section now only shows the result of your **very last action**. Use the [IMPACT] and [HINT] provided to understand how the system state changed.
- **User Complaints:** These are your primary signal for overall system health. If complaints remain "unchanged" or "bad," your last action likely didn't address the root cause.
- **Decision Making:** Since you only see the most recent result, maintain an internal logic of your progress. Avoid repeating actions that previously yielded "No meaningful change" or "Analysis already complete."

User complaints:
{obs.user_messages}

Last Action Result:
{obs.logs}

Available actions:
{obs.available_actions}

Return ONLY JSON:
{{
  "action": "<one action>",
  "confidence": 0.9
}}
"""

    def _parse_medium_response(self, text, obs):
        try:
            data = json.loads(text)
            action = data.get("action", "do_nothing")
            confidence = float(data.get("confidence", 0.3))
            confidence = max(0.0, min(1.0, confidence))
            return action, confidence, None
        except Exception:
            return "do_nothing", 0.3, None

    def _parse_hard(self, obs, action_history):
        prompt = self._build_hard_prompt(obs, action_history)
        response = self._call_llm(prompt)
        return self._parse_hard_response(response, obs)

    def _build_hard_prompt(self, obs, action_history):
        history_str = "\n".join([f"{i+1}. {act}" for i, act in enumerate(action_history)]) if action_history else "None"
        
        return f"""
DO NOT output anything except valid JSON.

You are managing a catastrophic system failure. You must follow the SLA Playbook rules exactly.

IMPORTANT RULES:
- Do NOT repeat actions that resulted in negative reward.
- Avoid "do_nothing" unless absolutely necessary.
- If an action caused no improvement, try a different strategy.
- Always aim to reduce system failure quickly.
- Choose ONLY from Available Actions.
- Do NOT hallucinate new actions.
- If an action caused negative reward, avoid it

Goal:
- Minimize total penalty
- Follow SLA rules strictly
- Stabilize system as fast as possible

Playbook:
{obs.playbook_text}

System State:
{json.dumps(obs.system_state, indent=2)}

Logs (Previous Steps Details):
{obs.logs}

Previous Actions Taken:
{history_str}

Available Actions:
{json.dumps(obs.available_actions)}

Step Count: {obs.step_count}

Return ONLY JSON:
{{
    "action": "<exact_string_from_available_actions>",
    "target": null,
    "confidence": 0.0-1.0
}}
"""

    def _parse_hard_response(self, text, obs):
        try:
            data = json.loads(text)
            action = data.get("action", "")
            target = data.get("target", None)
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
            
            if not action:
                 return sorted(obs.available_actions)[0], 0.0, None
            
            return action, confidence, target
        except Exception:
            return sorted(obs.available_actions)[0], 0.0, None

    def _call_llm(self, prompt):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                top_p=1.0,
                max_tokens=150,
                seed=LLM_SEED
            )
            return response.choices[0].message.content or ""
        except Exception:
            return ""

    def _build_cache_key(self, obs, action_history):
        if obs.task_type == "hard":
            return json.dumps({
                "state": obs.system_state,
                "step": obs.step_count,
                "actions": obs.available_actions,
                "history": action_history
            }, sort_keys=True)
        else:
            return json.dumps({
                "task": obs.task_type,
                "messages": getattr(obs, "user_messages", None),
                "logs": getattr(obs, "logs", ""),
                "metrics": getattr(obs, "system_metrics", None),
                "state": getattr(obs, "system_state", None)
            }, sort_keys=True)
        
def _calculate_easy_bounds(env: DevOpsEnv, max_steps: int) -> tuple[float, float]:
    """
    Calculates theoretical min and max rewards for the Easy task based on env.py logic.
    """
    # Base step costs defined in _step_easy
    base_step_cost = -0.05
    correct_reward = 1.0 + base_step_cost        # +0.95 net
    new_herring_reward = 0.2 + base_step_cost    # +0.15 net
    repeat_herring_penalty = -0.4 + base_step_cost # -0.45 net
    invalid_action_penalty = -0.2 + base_step_cost # -0.25 net

    red_herrings_count = len(env.state_data.get("red_herrings", {}))

    # MAX REWARD: The optimal "score-farming" path is to discover all available red herrings 
    # (up to max_steps - 1) for the +0.2 bonus, then execute the correct action on the final step.
    usable_herrings = min(red_herrings_count, max_steps - 1)
    max_reward = (usable_herrings * new_herring_reward) + correct_reward

    # MIN REWARD: The worst-case scenario is either spamming invalid actions for all steps,
    # OR discovering a red herring once, then getting penalized for repeating it on all remaining steps.
    worst_invalid_spam = max_steps * invalid_action_penalty
    
    if red_herrings_count > 0:
        worst_herring_spam = new_herring_reward + ((max_steps - 1) * repeat_herring_penalty)
    else:
        worst_herring_spam = float('inf')

    min_reward = min(worst_herring_spam, worst_invalid_spam)

    return min_reward, max_reward


def _calculate_medium_bounds(env: DevOpsEnv, max_steps: int) -> tuple[float, float]:
    """
    Calculates theoretical min and max rewards for the Medium task based on env.py logic.
    """
    invalid_pen = env.state_data.get("invalid_action_penalty", -0.2)
    repeat_pen = env.state_data.get("repeat_penalty", -0.15)
    step_pen = env.state_data.get("step_penalty", -0.03)
    rules = env.state_data.get("transition_rules", {})

    # MIN REWARD: Repeating an invalid action for all max_steps.
    # According to _step_medium, an invalid action converts to "do_nothing", meaning the agent 
    # gets hit with the invalid penalty, the repeat penalty (for repeating "do_nothing"), and the step penalty.
    min_reward = 0.0
    for step in range(1, max_steps + 1):
        repeat_count = step - 1 
        step_worst = invalid_pen + (repeat_pen * repeat_count) + (step_pen * step)
        min_reward += step_worst

    # MAX REWARD: Finding the highest rewarding rule and hitting it optimally.
    max_rule_reward = 0.0
    for rule in rules.values():
        r1 = float(rule.get("reward", 0.0))
        r2 = float(rule.get("else_reward", 0.0))
        max_rule_reward = max(max_rule_reward, r1, r2)
    
    # We aggregate the maximum possible rule reward minus the compounding step penalty over the maximum steps.
    max_reward = 0.0
    for step in range(1, max_steps + 1):
        max_reward += max_rule_reward + (step_pen * step)

    # Establish a minimum floor for max_reward to prevent zero-division during normalization 
    # if the scenario relies purely on condition-clearing rather than positive rule rewards.
    max_reward = max(max_reward, 1.0)

    return min_reward, max_reward
        
def grade_easy(num_scenarios=1):
    total_score = 0.0
    
    env = DevOpsEnv(seed = 42)
    parser = LLMParser()
    
    for i in range(num_scenarios):
        total_reward = 0.0
        done = False
        rewards_list = []
        action_history = []
        start_printed = False
        try:
            obs = env.reset(task = "easy")
            min_reward, max_reward = _calculate_easy_bounds(env, MAX_STEPS)
            print(f"[START] task=easy_scenario_{i+1} env=opssim_ai model={MODEL_NAME}")
            start_printed = True
            
            for step in range(MAX_STEPS):
                action_str, _, target = parser.parse(obs, action_history)
                
                obs, reward, done, info = env.step(Action(action_type=action_str, target=target))
                
                log_action = f"{action_str}({target})" if target else action_str
                action_history.append(log_action)
                
                total_reward += reward
                rewards_list.append(f"{reward:.2f}")
                error_msg = env.last_action_error if env.last_action_error is not None else "null"

                print(f"[STEP] step={step+1} action={action_str} reward={reward:.2f} done={str(done).lower()} error={error_msg}")
                
                if done:
                    break
        finally:
            env.close()
            scenario_score = max(0.0, min(1.0, (total_reward - min_reward) / (max_reward - min_reward)))
            if start_printed:
                success = "true" if (done and total_reward > 0) else "false"
                rewards_str = ",".join(rewards_list)
                print(f"[END] success={success} steps={len(rewards_list)} score={scenario_score:.2f} rewards={rewards_str}")

        total_score += scenario_score

    return total_score / num_scenarios

def grade_medium(num_scenarios = 1):
    total_score = 0.0
    
    env = DevOpsEnv(seed = 42)
    parser = LLMParser()
    
    for i in range(num_scenarios):
        total_reward = 0.0
        done = False
        rewards_list = []
        action_history = []
        start_printed = False
        try:
            obs = env.reset(task = "medium")
            min_reward, max_reward = _calculate_medium_bounds(env, MAX_STEPS)
            print(f"[START] task=medium_scenario_{i+1} env=opssim_ai model={MODEL_NAME}")
            start_printed = True
            
            for step in range(MAX_STEPS):
                action_str, _, target = parser.parse(obs, action_history)
                
                obs, reward, done, info = env.step(Action(action_type=action_str, target=target))
                
                log_action = f"{action_str}({target})" if target else action_str
                action_history.append(log_action)
                
                total_reward += reward
                rewards_list.append(f"{reward:.2f}")
                error_msg = env.last_action_error if env.last_action_error is not None else "null"

                print(f"[STEP] step={step+1} action={action_str} reward={reward:.2f} done={str(done).lower()} error={error_msg}")
                
                if done:
                    break
        finally:
            env.close()
            scenario_score = max(0.0, min(1.0, (total_reward - min_reward) / (max_reward - min_reward)))
            if start_printed:
                success = "true" if (done and total_reward > 0) else "false"
                rewards_str = ",".join(rewards_list)
                print(f"[END] success={success} steps={len(rewards_list)} score={scenario_score:.2f} rewards={rewards_str}")

        total_score += scenario_score

    return total_score / num_scenarios

def _calculate_dynamic_min_reward(env: DevOpsEnv, max_steps: int) -> float:
    worst_bleed = 0.0
    for rule in env.state_data.get("bleed_rules", []):
        penalty = rule.get("penalty", 0.0)
        if penalty < 0:
            worst_bleed += penalty
    worst_penalty = min(env.state_data.get("penalties", {}).values(), default=0)
    worst_urgency_total = -0.05 * (max_steps * (max_steps + 1) / 2)
    worst_urgency_penalty_per_step = worst_urgency_total / max_steps
    worst_case_per_step = worst_bleed + worst_penalty + worst_urgency_penalty_per_step
    sla_violation_penalty = env.state_data.get("sla_violation_penalty", -1.0)
    min_reward = (max_steps * worst_case_per_step) + sla_violation_penalty
    return min_reward

def grade_hard(num_scenarios=1):
    total_score = 0.0
    
    env = DevOpsEnv(seed=42)
    parser = LLMParser()
    
    for i in range(num_scenarios):
        total_reward = 0.0
        done = False
        action_history = []
        rewards_list = []
        start_printed = False
        min_reward = 0.0
        max_reward = 1.0
        try:
            obs = env.reset(task = "hard")
            min_reward = _calculate_dynamic_min_reward(env, MAX_STEPS)
            max_success = 1.0 * MAX_STEPS
            max_progress = 0.3 * MAX_STEPS
            max_reward = max_success + max_progress
            
            print(f"[START] task=hard_scenario_{i+1} env=opssim_ai model={MODEL_NAME}")
            start_printed = True
            
            for step in range(MAX_STEPS):
                action_str, confidence, target = parser.parse(obs, action_history)
                log_action = f"{action_str}({target})" if target else action_str
                
                action = Action(action_type=action_str, target=target)
                
                obs, reward, done, info = env.step(action)
                total_reward += reward
                rewards_list.append(f"{reward:.2f}")
                action_history.append(log_action)
                error_msg = env.last_action_error if env.last_action_error is not None else "null"
                
                print(f"[STEP] step={step+1} action={log_action} reward={reward:.2f} done={str(done).lower()} error={error_msg}")
                
                if done:
                    break
        finally:
            env.close()
            final_score = (total_reward - min_reward) / (max_reward - min_reward)
            final_score = max(0.0, min(1.0, final_score))
            if start_printed:
                success = "true" if (done and total_reward > 0) else "false"
                rewards_str = ",".join(rewards_list)
                print(f"[END] success={success} steps={len(rewards_list)} score={final_score:.2f} rewards={rewards_str}")

        total_score += final_score
        
    return total_score / num_scenarios

def main() -> None:
    grade_easy()
    grade_medium()
    grade_hard()

if __name__ == "__main__":
    main()
