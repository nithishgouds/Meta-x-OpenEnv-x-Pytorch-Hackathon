import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import DevOpsEnv, AGENT_DOMAIN_MAP, EXECUTION_AGENTS, IC_NAME, SUPERVISOR_NAME
from models import Action, Observation, Reward

AGENT_NAMES = EXECUTION_AGENTS
ALL_AGENTS = EXECUTION_AGENTS + [IC_NAME, SUPERVISOR_NAME]


class DomainAgent:
    def __init__(self, name):
        self.name = name
        self.domain = AGENT_DOMAIN_MAP.get(name, "")

    def observe(self, domain_obs):
        parts = [f"[{self.name}]"]
        if domain_obs.domain_state:
            for k, v in domain_obs.domain_state.items():
                parts.append(f"  {k}: {v}")
        else:
            parts.append("  No domain-specific data visible.")
        return "\n".join(parts)


class ObservabilityAgent(DomainAgent):
    def __init__(self):
        super().__init__("ObservabilityOps")

    def observe(self, domain_obs):
        parts = [f"[{self.name}]"]
        if domain_obs.domain_state:
            for k, v in domain_obs.domain_state.items():
                parts.append(f"  {k}: {v}")
        if domain_obs.incident_channel:
            parts.append("  Recent channel activity detected.")
        return "\n".join(parts)


class IncidentCommander:
    def __init__(self):
        self.name = IC_NAME

    def decide(self, incident_channel, available_actions, history):
        return {"target_agent": None, "action": "do_nothing"}


class SupervisorAgent:
    def __init__(self):
        self.name = SUPERVISOR_NAME

    def evaluate(self, directive, penalties):
        action = directive.get("action", "")
        if action in penalties and float(penalties.get(action, 0)) <= -0.3:
            return False
        return True


class WarRoom:
    def __init__(self, seed=42, max_steps=15):
        self.env = DevOpsEnv(seed=seed, max_steps=max_steps)
        self.incident_channel = []
        self.agents = {}
        for name in AGENT_NAMES:
            if name == "ObservabilityOps":
                self.agents[name] = ObservabilityAgent()
            else:
                self.agents[name] = DomainAgent(name)
        self.ic = IncidentCommander()
        self.supervisor = SupervisorAgent()
        self.total_reward = 0.0
        self.done = False
        self.step_count = 0
        self.action_history = []
        self.communication_rewards = 0.0

    def reset(self):
        obs = self.env.reset()
        self.incident_channel = []
        self.total_reward = 0.0
        self.done = False
        self.step_count = 0
        self.action_history = []
        self.communication_rewards = 0.0
        domain_observations = {}
        for name in AGENT_NAMES:
            domain_observations[name] = self.env.get_domain_observation(name)
        return obs, domain_observations

    def observe_and_communicate(self, agent_name, message):
        if self.done:
            return
        self.incident_channel.append({"from": agent_name, "message": message})
        comm_action = Action(
            action_type="communicate",
            agent=agent_name,
            message=message,
        )
        _, reward, _, _ = self.env.step(comm_action)
        self.communication_rewards += reward.value

    def execute_directive(self, target_agent, action_str, supervisor_approved=True):
        if self.done:
            return self._make_done_result()
        self.step_count += 1
        action = Action(
            action_type=action_str,
            agent=target_agent,
            target_agent=target_agent,
            ic_directive=True,
            supervisor_approved=supervisor_approved,
        )
        obs, reward, done, info = self.env.step(action)
        self.total_reward += reward.value
        self.done = done
        self.action_history.append({
            "step": self.step_count,
            "target": target_agent,
            "action": action_str,
            "reward": reward.value,
            "supervisor_approved": supervisor_approved,
        })
        self.incident_channel.append({
            "from": IC_NAME,
            "message": f"DIRECTIVE: {target_agent} execute {action_str} -> reward={reward.value:.3f}",
        })
        domain_observations = {}
        for name in AGENT_NAMES:
            domain_observations[name] = self.env.get_domain_observation(name)
        return {
            "observation": obs,
            "reward": reward,
            "done": done,
            "info": info,
            "domain_observations": domain_observations,
            "incident_channel": list(self.incident_channel),
        }

    def run_step(self, agent_messages, ic_directive, supervisor_approved=True):
        for agent_name, message in agent_messages.items():
            if agent_name in AGENT_NAMES and message:
                self.observe_and_communicate(agent_name, message)
        target = ic_directive.get("target_agent", "AppOps")
        action_str = ic_directive.get("action", "do_nothing")
        return self.execute_directive(target, action_str, supervisor_approved)

    def get_total_reward(self):
        return self.total_reward + self.communication_rewards

    def get_incident_channel(self):
        return list(self.incident_channel)

    def is_done(self):
        return self.done

    def get_penalties(self):
        return self.env.state_data.get("penalties", {})

    def get_progress(self):
        return self.env._compute_progress()

    def get_goal_state(self):
        return self.env._compute_goal_state()

    def _make_done_result(self):
        domain_observations = {}
        for name in AGENT_NAMES:
            domain_observations[name] = self.env.get_domain_observation(name)
        return {
            "observation": self.env.observation,
            "reward": Reward(value=0.0),
            "done": True,
            "info": {"reason": "already_done"},
            "domain_observations": domain_observations,
            "incident_channel": list(self.incident_channel),
        }
