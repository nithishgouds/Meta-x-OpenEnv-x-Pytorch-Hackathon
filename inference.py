import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from env import DevOpsEnv
from models import Action, Reward
from multi_agent import WarRoom, AGENT_NAMES, IC_NAME

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
MAX_STEPS = 15
LLM_SEED = 42

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def call_llm(prompt, max_tokens=150):
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


def build_domain_agent_prompt(agent_name, domain_obs, incident_channel):
    channel_text = ""
    if incident_channel:
        channel_text = "\n".join(
            f"  [{m['from']}]: {m['message']}" for m in incident_channel
        )
    else:
        channel_text = "  (empty)"

    domain_state_text = json.dumps(domain_obs.domain_state, indent=2, default=str) if domain_obs.domain_state else "No data visible in your domain."

    return f"""DO NOT output anything except valid JSON.

You are {agent_name}, a domain specialist in an incident war room.
Your role: OBSERVE your domain and REPORT findings to the Incident Commander.
You can ONLY see your own domain's data. Other domains are invisible to you.

Your Domain State:
{domain_state_text}

Recent Logs:
{domain_obs.logs or "None"}

Incident Channel (shared communication):
{channel_text}

Your Available Actions (for reference — you do NOT execute, only observe and report):
{json.dumps(domain_obs.available_actions)}

Instructions:
- Analyze your domain state for anomalies, failures, or degradation.
- Report what you see and what you recommend to the Incident Commander.
- Be specific about which metrics are abnormal and why.
- If you have no useful data, say so.

Return ONLY JSON:
{{
  "observation": "<what you see in your domain>",
  "recommendation": "<what action you recommend, if any>",
  "severity": "critical|warning|normal"
}}"""


def build_ic_prompt(incident_channel, available_actions, playbook_text, action_history, step_count):
    channel_text = "\n".join(
        f"  [{m['from']}]: {m['message']}" for m in incident_channel
    ) if incident_channel else "  (empty)"

    history_text = "\n".join(
        f"  {i+1}. [{h['target']}] {h['action']} -> reward={h['reward']:.3f}"
        for i, h in enumerate(action_history)
    ) if action_history else "  None"

    action_domain_hint = ""
    for a in available_actions:
        action_domain_hint += f"  - {a}\n"

    return f"""DO NOT output anything except valid JSON.

You are the Incident Commander (IC) in a DevOps war room.
You coordinate 3 domain agents: AppOps, InfraOps, DatabaseOps.
You do NOT investigate directly. You read their reports and issue directives.

Playbook:
{playbook_text or "No playbook available."}

Incident Channel (all agent reports and past directives):
{channel_text}

Previous Actions Taken:
{history_text}

Available Actions:
{action_domain_hint}

Step: {step_count} / {MAX_STEPS}

Rules:
- Issue ONE directive: choose which agent should execute which action.
- Do NOT repeat actions that had negative rewards.
- Prioritize critical findings reported by domain agents.
- Follow the playbook strictly.
- If insufficient information, direct an investigation action.

Return ONLY JSON:
{{
  "target_agent": "AppOps|InfraOps|DatabaseOps",
  "action": "<exact_action_from_available_actions>",
  "reasoning": "<brief explanation>"
}}"""


def parse_domain_response(text):
    try:
        data = json.loads(text)
        return {
            "observation": data.get("observation", ""),
            "recommendation": data.get("recommendation", ""),
            "severity": data.get("severity", "normal"),
        }
    except Exception:
        return {"observation": text[:200], "recommendation": "", "severity": "normal"}


def parse_ic_response(text, available_actions):
    try:
        data = json.loads(text)
        target = data.get("target_agent", "AppOps")
        action = data.get("action", "do_nothing")
        if target not in AGENT_NAMES:
            target = "AppOps"
        if action not in available_actions and action != "do_nothing":
            action = "do_nothing"
        return {"target_agent": target, "action": action}
    except Exception:
        return {"target_agent": "AppOps", "action": "do_nothing"}


def run_episode(scenario_idx=None):
    room = WarRoom(seed=LLM_SEED, max_steps=MAX_STEPS)
    obs, domain_observations = room.reset()

    available_actions = obs.available_actions or []
    playbook_text = obs.playbook_text or ""

    print(f"[START] scenario={room.env.state_data.get('scenario_id', 'unknown')} model={MODEL_NAME}")

    for step in range(MAX_STEPS):
        if room.is_done():
            break

        agent_messages = {}
        for agent_name in AGENT_NAMES:
            domain_obs = domain_observations.get(agent_name)
            if domain_obs is None:
                domain_obs = room.env.get_domain_observation(agent_name)
            prompt = build_domain_agent_prompt(agent_name, domain_obs, room.get_incident_channel())
            response_text = call_llm(prompt, max_tokens=200)
            parsed = parse_domain_response(response_text)
            message = f"[{parsed['severity'].upper()}] {parsed['observation']}"
            if parsed["recommendation"]:
                message += f" | Recommend: {parsed['recommendation']}"
            agent_messages[agent_name] = message

        ic_prompt = build_ic_prompt(
            room.get_incident_channel(),
            available_actions,
            playbook_text,
            room.action_history,
            step + 1,
        )
        ic_text = call_llm(ic_prompt, max_tokens=200)
        directive = parse_ic_response(ic_text, available_actions)

        result = room.run_step(agent_messages, directive)

        reward_val = result["reward"].value
        done = result["done"]

        error_msg = room.env.last_action_error if room.env.last_action_error else "null"
        print(
            f"[STEP] step={step+1} target={directive['target_agent']} "
            f"action={directive['action']} reward={reward_val:.2f} "
            f"done={str(done).lower()} error={error_msg}"
        )

        if done:
            break

        domain_observations = result.get("domain_observations", {})

    total = room.get_total_reward()
    success = "true" if (room.is_done() and total > 0) else "false"
    print(f"[END] success={success} steps={room.step_count} total_reward={total:.2f}")
    return total, room


def _calculate_dynamic_min_reward(env, max_steps):
    worst_bleed = 0.0
    for rule in env.state_data.get("bleed_rules", []):
        penalty = rule.get("penalty", 0.0)
        if penalty < 0:
            worst_bleed += penalty
    worst_penalty = min(env.state_data.get("penalties", {}).values(), default=0)
    worst_urgency_total = -0.05 * (max_steps * (max_steps + 1) / 2)
    worst_urgency_per_step = worst_urgency_total / max_steps
    worst_per_step = worst_bleed + worst_penalty + worst_urgency_per_step
    sla_penalty = env.state_data.get("sla_violation_penalty", -1.0)
    comm_cost = -0.02 * max_steps * 3
    return (max_steps * worst_per_step) + sla_penalty + comm_cost


def grade(num_scenarios=1):
    total_score = 0.0

    for i in range(num_scenarios):
        room = WarRoom(seed=LLM_SEED, max_steps=MAX_STEPS)
        obs, domain_observations = room.reset()
        available_actions = obs.available_actions or []
        playbook_text = obs.playbook_text or ""

        min_reward = _calculate_dynamic_min_reward(room.env, MAX_STEPS)
        max_success = 1.0 * MAX_STEPS
        max_progress = 0.3 * MAX_STEPS
        max_reward = max_success + max_progress

        print(f"[START] scenario_{i+1} env=opssim_ai model={MODEL_NAME}")

        rewards_list = []
        for step in range(MAX_STEPS):
            if room.is_done():
                break

            agent_messages = {}
            for agent_name in AGENT_NAMES:
                domain_obs = domain_observations.get(agent_name)
                if domain_obs is None:
                    domain_obs = room.env.get_domain_observation(agent_name)
                prompt = build_domain_agent_prompt(agent_name, domain_obs, room.get_incident_channel())
                text = call_llm(prompt, max_tokens=200)
                parsed = parse_domain_response(text)
                msg = f"[{parsed['severity'].upper()}] {parsed['observation']}"
                if parsed["recommendation"]:
                    msg += f" | Recommend: {parsed['recommendation']}"
                agent_messages[agent_name] = msg

            ic_prompt = build_ic_prompt(
                room.get_incident_channel(),
                available_actions,
                playbook_text,
                room.action_history,
                step + 1,
            )
            ic_text = call_llm(ic_prompt, max_tokens=200)
            directive = parse_ic_response(ic_text, available_actions)

            result = room.run_step(agent_messages, directive)
            reward_val = result["reward"].value
            rewards_list.append(f"{reward_val:.2f}")

            error_msg = room.env.last_action_error if room.env.last_action_error else "null"
            print(
                f"[STEP] step={step+1} target={directive['target_agent']} "
                f"action={directive['action']} reward={reward_val:.2f} "
                f"done={str(result['done']).lower()} error={error_msg}"
            )

            if result["done"]:
                break

            domain_observations = result.get("domain_observations", {})

        total = room.get_total_reward()
        score = max(0.0, min(1.0, (total - min_reward) / (max_reward - min_reward)))
        success = "true" if (room.is_done() and total > 0) else "false"
        print(f"[END] success={success} steps={room.step_count} score={score:.2f} rewards={','.join(rewards_list)}")
        total_score += score

    return total_score / num_scenarios


def main():
    grade(num_scenarios=1)


if __name__ == "__main__":
    main()
