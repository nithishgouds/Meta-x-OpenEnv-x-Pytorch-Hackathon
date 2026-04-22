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
MAX_STEPS = 30
LLM_SEED = 42

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


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


def build_observability_prompt(domain_obs, incident_channel, root_cause_keywords):
    channel_text = _format_channel(incident_channel)
    domain_state_text = json.dumps(domain_obs.domain_state, indent=2, default=str) if domain_obs.domain_state else "No monitoring data."

    return f"""DO NOT output anything except valid JSON.

You are ObservabilityOps, the monitoring and metrics specialist in a DevOps war room.
Your PRIMARY role: Analyze metrics, detect anomalies, and surface the ROOT CAUSE before the IC decides.

Your Monitoring Data:
{domain_state_text}

Recent Logs:
{domain_obs.logs or "None"}

Incident Channel:
{channel_text}

Root Cause Hint Keywords (use these to guide analysis): {json.dumps(root_cause_keywords)}

Instructions:
- Correlate metrics across your monitoring data to identify the root cause.
- Be SPECIFIC about which metrics are anomalous and why.
- Include relevant keywords from the hint list in your analysis.
- If you detect cascading patterns, describe the chain.
- Your report DIRECTLY affects reward — accurate root cause identification is critical.

Return ONLY JSON:
{{
  "root_cause_analysis": "<detailed root cause with specific metrics>",
  "anomalous_metrics": ["<metric1>", "<metric2>"],
  "severity": "critical|warning|normal",
  "cascade_chain": "<A causes B causes C>"
}}"""


def build_domain_agent_prompt(agent_name, domain_obs, incident_channel, goal_state, progress):
    channel_text = _format_channel(incident_channel)
    domain_state_text = json.dumps(domain_obs.domain_state, indent=2, default=str) if domain_obs.domain_state else "No data visible in your domain."
    goal_text = json.dumps(goal_state, indent=2) if goal_state else "Unknown"
    actions_text = json.dumps(domain_obs.available_actions) if domain_obs.available_actions else "[]"

    return f"""DO NOT output anything except valid JSON.

You are {agent_name}, a domain specialist in an incident war room.
You can ONLY see your own domain's data. Other domains are invisible to you.

Your Domain State:
{domain_state_text}

Incident Channel (shared communication):
{channel_text}

Your Available Actions (for reference — you do NOT execute, only observe and report):
{actions_text}

Current Goal State (SLA requirements):
{goal_text}

Resolution Progress: {progress:.0%}

Instructions:
- Analyze your domain state for anomalies, failures, or degradation.
- Report what you see and what action you recommend.
- Reference specific metrics and their current values.
- If another agent's report in the channel relates to your domain, acknowledge it.
- Do NOT recommend actions outside your domain.

Return ONLY JSON:
{{
  "observation": "<what you see in your domain>",
  "recommendation": "<specific action from your available actions, if any>",
  "severity": "critical|warning|normal",
  "dependencies": "<upstream or downstream issues you suspect>"
}}"""


def build_ic_prompt(incident_channel, available_actions, playbook_text, action_history, step_count, goal_state, progress, action_domains):
    channel_text = _format_channel(incident_channel)
    history_text = _format_history(action_history)
    goal_text = json.dumps(goal_state, indent=2) if goal_state else "Unknown"

    domain_action_text = ""
    for domain, actions in action_domains.items():
        agent = next((a for a, d in AGENT_DOMAIN_MAP.items() if d == domain), domain)
        domain_action_text += f"  {agent} ({domain}): {', '.join(actions)}\n"

    unmet_goals = [g for g, met in (goal_state or {}).items() if not met]
    unmet_text = "\n".join(f"  - {g}" for g in unmet_goals) if unmet_goals else "  All goals met!"

    return f"""DO NOT output anything except valid JSON.

You are the Incident Commander (IC) in a DevOps war room.
You coordinate 7 domain agents: AppOps, InfraOps, DatabaseOps, NetworkOps, SecOps, MiddlewareOps, ObservabilityOps.
You do NOT investigate directly. You read their reports and issue ONE directive per step.

Playbook:
{playbook_text or "No playbook available."}

Incident Channel (all reports and past directives):
{channel_text}

Previous Actions:
{history_text}

Actions by Domain Agent:
{domain_action_text}

Unmet SLA Goals:
{unmet_text}

Resolution Progress: {progress:.0%}
Step: {step_count} / {MAX_STEPS}

CRITICAL RULES:
- Issue ONE directive: choose which agent should execute which action.
- NEVER repeat an action that already failed or had negative reward.
- NEVER assign an action to the wrong domain agent.
- Follow the playbook ordering strictly.
- Prioritize critical findings from ObservabilityOps.
- Consider the dependency chain: fix root causes before symptoms.
- Do NOT use do_nothing unless absolutely no valid action exists.

PLANNING (think ahead):
- What sub-goals remain? Which order to address them?
- What is the dependency chain? (e.g., fix DB before restarting App)
- Are there any recovery steps needed from past mistakes?

Return ONLY JSON:
{{
  "target_agent": "AppOps|InfraOps|DatabaseOps|NetworkOps|SecOps|MiddlewareOps|ObservabilityOps",
  "action": "<exact_action_name>",
  "reasoning": "<brief explanation referencing agent reports>",
  "plan": ["<next_step_1>", "<next_step_2>"]
}}"""


def build_supervisor_prompt(directive, incident_channel, penalties, action_history, step_count):
    channel_text = _format_channel(incident_channel[-5:] if len(incident_channel) > 5 else incident_channel)
    history_text = _format_history(action_history)
    penalty_text = json.dumps(penalties, indent=2)

    return f"""DO NOT output anything except valid JSON.

You are the Supervisor (Fleet AI), the safety oversight layer in a DevOps war room.
Your role: Evaluate the IC's directive and VETO if it would be catastrophic.

IC Directive:
  Target Agent: {directive.get('target_agent', 'unknown')}
  Action: {directive.get('action', 'unknown')}
  Reasoning: {directive.get('reasoning', 'none provided')}

Known Dangerous Actions (penalties):
{penalty_text}

Recent Channel:
{channel_text}

Previous Actions:
{history_text}

Step: {step_count} / {MAX_STEPS}

RULES:
- APPROVE if the action is safe and follows the playbook.
- VETO if the action is in the penalty list with severe penalty (<= -0.3).
- VETO if the action was already tried and failed.
- VETO if the action targets the wrong domain.
- Be CONSERVATIVE — when in doubt, approve investigation actions.
- Do NOT veto without strong evidence of harm.

Return ONLY JSON:
{{
  "approved": true|false,
  "reasoning": "<why you approve or veto>",
  "alternative": "<suggested alternative action if vetoed, or null>"
}}"""


def _format_channel(channel):
    if not channel:
        return "  (empty)"
    return "\n".join(f"  [{m.get('from', '?')}]: {m.get('message', '')}" for m in channel)


def _format_history(history):
    if not history:
        return "  None"
    return "\n".join(
        f"  {i+1}. [{h.get('target', '?')}] {h.get('action', '?')} -> reward={h.get('reward', 0):.3f}"
        for i, h in enumerate(history)
    )


def parse_observability_response(text):
    try:
        data = json.loads(text)
        return {
            "root_cause_analysis": data.get("root_cause_analysis", ""),
            "anomalous_metrics": data.get("anomalous_metrics", []),
            "severity": data.get("severity", "normal"),
            "cascade_chain": data.get("cascade_chain", ""),
        }
    except Exception:
        return {"root_cause_analysis": text[:300], "anomalous_metrics": [], "severity": "normal", "cascade_chain": ""}


def parse_domain_response(text):
    try:
        data = json.loads(text)
        return {
            "observation": data.get("observation", ""),
            "recommendation": data.get("recommendation", ""),
            "severity": data.get("severity", "normal"),
            "dependencies": data.get("dependencies", ""),
        }
    except Exception:
        return {"observation": text[:200], "recommendation": "", "severity": "normal", "dependencies": ""}


def parse_ic_response(text, available_actions, action_domains):
    try:
        data = json.loads(text)
        target = data.get("target_agent", "AppOps")
        action = data.get("action", "")
        reasoning = data.get("reasoning", "")

        if target not in AGENT_NAMES:
            target = "AppOps"

        if action not in available_actions:
            target_domain = AGENT_DOMAIN_MAP.get(target, "")
            domain_acts = action_domains.get(target_domain, [])
            if domain_acts:
                action = domain_acts[0]
            else:
                action = available_actions[0] if available_actions else "analyze_metrics"

        return {"target_agent": target, "action": action, "reasoning": reasoning}
    except Exception:
        return {"target_agent": "ObservabilityOps", "action": "analyze_metrics", "reasoning": "fallback"}


def parse_supervisor_response(text):
    try:
        data = json.loads(text)
        return {
            "approved": data.get("approved", True),
            "reasoning": data.get("reasoning", ""),
            "alternative": data.get("alternative"),
        }
    except Exception:
        return {"approved": True, "reasoning": "parse_error_default_approve", "alternative": None}


def run_episode(scenario_idx=None):
    room = WarRoom(seed=LLM_SEED, max_steps=MAX_STEPS)
    obs, domain_observations = room.reset()

    available_actions = obs.available_actions or []
    playbook_text = obs.playbook_text or ""
    action_domains = room.env.state_data.get("action_domains", {})
    root_cause_keywords = room.env.state_data.get("root_cause_keywords", [])
    penalties = room.get_penalties()

    print(f"[START] scenario={room.env.state_data.get('scenario_id', 'unknown')} model={MODEL_NAME}")

    for step in range(MAX_STEPS):
        if room.is_done():
            break

        goal_state = room.get_goal_state()
        progress = room.get_progress()

        obs_agent_obs = domain_observations.get("ObservabilityOps")
        if obs_agent_obs is None:
            obs_agent_obs = room.env.get_domain_observation("ObservabilityOps")
        obs_prompt = build_observability_prompt(obs_agent_obs, room.get_incident_channel(), root_cause_keywords)
        obs_text = call_llm(obs_prompt, max_tokens=300)
        obs_parsed = parse_observability_response(obs_text)
        obs_message = f"[ROOT CAUSE] {obs_parsed['root_cause_analysis']}"
        if obs_parsed["anomalous_metrics"]:
            obs_message += f" | Anomalies: {', '.join(obs_parsed['anomalous_metrics'])}"
        if obs_parsed["cascade_chain"]:
            obs_message += f" | Chain: {obs_parsed['cascade_chain']}"
        room.observe_and_communicate("ObservabilityOps", obs_message)

        agent_messages = {}
        for agent_name in AGENT_NAMES:
            if agent_name == "ObservabilityOps":
                continue
            domain_obs = domain_observations.get(agent_name)
            if domain_obs is None:
                domain_obs = room.env.get_domain_observation(agent_name)
            prompt = build_domain_agent_prompt(agent_name, domain_obs, room.get_incident_channel(), goal_state, progress)
            response_text = call_llm(prompt, max_tokens=200)
            parsed = parse_domain_response(response_text)
            message = f"[{parsed['severity'].upper()}] {parsed['observation']}"
            if parsed["recommendation"]:
                message += f" | Recommend: {parsed['recommendation']}"
            if parsed["dependencies"]:
                message += f" | Deps: {parsed['dependencies']}"
            agent_messages[agent_name] = message

        for agent_name, message in agent_messages.items():
            room.observe_and_communicate(agent_name, message)

        ic_prompt = build_ic_prompt(
            room.get_incident_channel(),
            available_actions,
            playbook_text,
            room.action_history,
            step + 1,
            goal_state,
            progress,
            action_domains,
        )
        ic_text = call_llm(ic_prompt, max_tokens=300)
        directive = parse_ic_response(ic_text, available_actions, action_domains)

        sup_prompt = build_supervisor_prompt(
            directive,
            room.get_incident_channel(),
            penalties,
            room.action_history,
            step + 1,
        )
        sup_text = call_llm(sup_prompt, max_tokens=200)
        sup_decision = parse_supervisor_response(sup_text)

        supervisor_approved = sup_decision["approved"]
        final_action = directive["action"]
        final_target = directive["target_agent"]

        if not supervisor_approved:
            alt = sup_decision.get("alternative")
            if alt and alt in available_actions:
                final_action = alt
                target_domain = None
                for domain, actions in action_domains.items():
                    if alt in actions:
                        target_domain = domain
                        break
                if target_domain:
                    final_target = next((a for a, d in AGENT_DOMAIN_MAP.items() if d == target_domain), final_target)
            print(f"  [SUPERVISOR] VETOED: {directive['action']} | Reason: {sup_decision['reasoning']}")

        result = room.execute_directive(final_target, final_action, supervisor_approved)

        reward_val = result["reward"].value
        done = result["done"]

        error_msg = room.env.last_action_error if room.env.last_action_error else "null"
        print(
            f"[STEP] step={step+1} target={final_target} "
            f"action={final_action} reward={reward_val:.3f} "
            f"done={str(done).lower()} error={error_msg} "
            f"progress={room.get_progress():.0%}"
        )

        if done:
            break

        domain_observations = result.get("domain_observations", {})

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

    worst_q_act = -0.5
    worst_seq = -0.15
    worst_resp = 0.0
    worst_conf = 0.3

    gamma_val = 1.0 / max(max_steps, 1)
    worst_comm = gamma_val * max_steps * 8
    sla_penalty = float(env.state_data.get("sla_violation_penalty", -2.0))

    worst_per_step = -worst_bleed - worst_urgency / max_steps + worst_q_act + worst_seq - worst_conf
    return (max_steps * worst_per_step) + sla_penalty - worst_comm


def grade(num_scenarios=1):
    total_score = 0.0

    for i in range(num_scenarios):
        room = WarRoom(seed=LLM_SEED, max_steps=MAX_STEPS)
        obs, domain_observations = room.reset()
        available_actions = obs.available_actions or []
        playbook_text = obs.playbook_text or ""
        action_domains = room.env.state_data.get("action_domains", {})
        root_cause_keywords = room.env.state_data.get("root_cause_keywords", [])
        penalties = room.get_penalties()

        min_reward = _calculate_dynamic_min_reward(room.env, MAX_STEPS)
        max_reward = 2.0 * MAX_STEPS + 0.3 * MAX_STEPS

        print(f"[START] scenario_{i+1} env=opssim_ai model={MODEL_NAME}")

        rewards_list = []
        for step in range(MAX_STEPS):
            if room.is_done():
                break

            goal_state = room.get_goal_state()
            progress = room.get_progress()

            obs_agent_obs = domain_observations.get("ObservabilityOps")
            if obs_agent_obs is None:
                obs_agent_obs = room.env.get_domain_observation("ObservabilityOps")
            obs_prompt = build_observability_prompt(obs_agent_obs, room.get_incident_channel(), root_cause_keywords)
            obs_text = call_llm(obs_prompt, max_tokens=300)
            obs_parsed = parse_observability_response(obs_text)
            obs_msg = f"[ROOT CAUSE] {obs_parsed['root_cause_analysis']}"
            if obs_parsed["anomalous_metrics"]:
                obs_msg += f" | Anomalies: {', '.join(obs_parsed['anomalous_metrics'])}"
            room.observe_and_communicate("ObservabilityOps", obs_msg)

            agent_messages = {}
            for agent_name in AGENT_NAMES:
                if agent_name == "ObservabilityOps":
                    continue
                domain_obs = domain_observations.get(agent_name)
                if domain_obs is None:
                    domain_obs = room.env.get_domain_observation(agent_name)
                prompt = build_domain_agent_prompt(agent_name, domain_obs, room.get_incident_channel(), goal_state, progress)
                text = call_llm(prompt, max_tokens=200)
                parsed = parse_domain_response(text)
                msg = f"[{parsed['severity'].upper()}] {parsed['observation']}"
                if parsed["recommendation"]:
                    msg += f" | Recommend: {parsed['recommendation']}"
                agent_messages[agent_name] = msg

            for agent_name, message in agent_messages.items():
                room.observe_and_communicate(agent_name, message)

            ic_prompt = build_ic_prompt(
                room.get_incident_channel(), available_actions, playbook_text,
                room.action_history, step + 1, goal_state, progress, action_domains,
            )
            ic_text = call_llm(ic_prompt, max_tokens=300)
            directive = parse_ic_response(ic_text, available_actions, action_domains)

            sup_prompt = build_supervisor_prompt(
                directive, room.get_incident_channel(), penalties, room.action_history, step + 1,
            )
            sup_text = call_llm(sup_prompt, max_tokens=200)
            sup_decision = parse_supervisor_response(sup_text)

            supervisor_approved = sup_decision["approved"]
            final_action = directive["action"]
            final_target = directive["target_agent"]

            if not supervisor_approved:
                alt = sup_decision.get("alternative")
                if alt and alt in available_actions:
                    final_action = alt
                    for domain, actions in action_domains.items():
                        if alt in actions:
                            final_target = next((a for a, d in AGENT_DOMAIN_MAP.items() if d == domain), final_target)
                            break

            result = room.execute_directive(final_target, final_action, supervisor_approved)
            reward_val = result["reward"].value
            rewards_list.append(f"{reward_val:.3f}")

            error_msg = room.env.last_action_error if room.env.last_action_error else "null"
            print(
                f"[STEP] step={step+1} target={final_target} "
                f"action={final_action} reward={reward_val:.3f} "
                f"done={str(result['done']).lower()} error={error_msg}"
            )

            if result["done"]:
                break

            domain_observations = result.get("domain_observations", {})

        total = room.get_total_reward()
        score = max(0.0, min(1.0, (total - min_reward) / (max_reward - min_reward)))
        success = "true" if (room.is_done() and total > 0) else "false"
        print(f"[END] success={success} steps={room.step_count} score={score:.3f} rewards={','.join(rewards_list)}")
        total_score += score

    return total_score / num_scenarios


def main():
    grade(num_scenarios=1)


if __name__ == "__main__":
    main()
