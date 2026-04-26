import argparse
import json
import os
import sys

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer


class MultiAgentTrainingEnv:
    def __init__(self):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from env import DevOpsEnv, EXECUTION_AGENTS, AGENT_DOMAIN_MAP, IC_NAME, SUPERVISOR_NAME
        from multi_agent import WarRoom

        self.room = WarRoom(seed=42, max_steps=15)
        self.reward = 0.0
        self.done = False
        self._available_actions = []
        self._agent_names = EXECUTION_AGENTS
        self._agent_domain_map = AGENT_DOMAIN_MAP
        self._domain_observations = {}
        self._playbook_text = ""
        self._action_domains = {}
        self._root_cause_keywords = []
        self._penalties = {}

    def reset(self, prompt=None, **kwargs):
        obs, domain_obs = self.room.reset()
        self.reward = 0.0
        self.done = False
        self._available_actions = obs.available_actions or []
        self._domain_observations = domain_obs
        self._playbook_text = obs.playbook_text or ""
        self._action_domains = self.room.env.state_data.get("action_domains", {})
        self._root_cause_keywords = self.room.env.state_data.get("root_cause_keywords", [])
        self._penalties = self.room.env.state_data.get("penalties", {})

        goal_state = self.room.get_goal_state()
        unmet = [g for g, met in goal_state.items() if not met]

        parts = [
            f"INCIDENT: {obs.logs or 'Unknown incident'}",
            f"\nPlaybook:\n{self._playbook_text}",
            f"\nSystem State:\n{json.dumps(obs.system_state, indent=2, default=str)}",
            f"\nAvailable Actions: {', '.join(self._available_actions)}",
            f"\nUnmet SLA Goals:\n" + "\n".join(f"  - {g}" for g in unmet),
            f"\nRoot Cause Keywords: {', '.join(self._root_cause_keywords)}",
            "\nYou are the Incident Commander in a war room with 7 domain agents:",
            "  AppOps, InfraOps, DatabaseOps, NetworkOps, SecOps, MiddlewareOps, ObservabilityOps",
            "Plus a Supervisor (Fleet AI) who approves or vetoes your directives.",
            "\nTools available:",
            "  observe_domain(agent_name) - Check a domain agent's view",
            "  communicate(agent_name, message) - Post to incident channel",
            "  execute_directive(target_agent, action) - Issue action to a domain agent",
            "\nFollow the playbook. Investigate before acting. Fix root causes before symptoms.",
        ]
        return "\n".join(parts)

    def observe_domain(self, agent_name: str) -> str:
        """Get the domain-specific observation for a specialist agent."""
        if self.done:
            return "Incident resolved or episode ended."
        if agent_name not in self._agent_names:
            return f"Unknown agent: {agent_name}. Use one of: {', '.join(self._agent_names)}"
        domain_obs = self._domain_observations.get(agent_name)
        if domain_obs is None:
            domain_obs = self.room.env.get_domain_observation(agent_name)
        parts = [f"[{agent_name} Domain View]"]
        if domain_obs.domain_state:
            parts.append(f"State: {json.dumps(domain_obs.domain_state, indent=2, default=str)}")
        else:
            parts.append("State: No domain-specific data visible.")
        parts.append(f"Actions: {json.dumps(domain_obs.available_actions)}")
        if domain_obs.goal_state:
            parts.append(f"Goal Progress: {domain_obs.progress:.0%}")
        return "\n".join(parts)

    def communicate(self, agent_name: str, message: str) -> str:
        """Post a message to the shared incident channel as a domain agent."""
        if self.done:
            return "Incident resolved or episode ended."
        self.room.observe_and_communicate(agent_name, message)
        return f"[{agent_name}] message posted to incident channel."

    def execute_directive(self, target_agent: str, action: str) -> str:
        """As Incident Commander, issue an action directive to a domain agent."""
        if self.done:
            return "Incident resolved or episode ended."

        supervisor_approved = True
        if action in self._penalties and float(self._penalties.get(action, 0)) <= -0.3:
            supervisor_approved = False

        result = self.room.execute_directive(target_agent, action, supervisor_approved)
        self.reward = self.room.get_total_reward()
        self.done = result["done"]

        obs = result["observation"]
        self._domain_observations = result.get("domain_observations", {})

        parts = []
        if obs.logs:
            parts.append(f"Logs: {obs.logs}")
        if obs.system_state:
            parts.append(f"State: {json.dumps(obs.system_state, indent=2, default=str)}")
        parts.append(f"Step Reward: {result['reward'].value:.3f}")
        parts.append(f"Total Reward: {self.reward:.3f}")
        parts.append(f"Progress: {self.room.get_progress():.0%}")
        parts.append(f"Done: {self.done}")
        if not self.done:
            parts.append(f"\nAvailable Actions: {', '.join(self._available_actions)}")
        return "\n".join(parts)


def reward_func(environments, **kwargs):
    return [env.reward for env in environments]


def build_dataset():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(base_dir, "tasks", "cascade.json")
    prompts = []

    with open(filepath, "r") as f:
        scenarios = json.load(f)["cascade_tasks_dataset"]

    for scenario in scenarios:
        scenario_id = scenario.get("scenario_id", "unknown")
        description = scenario.get("description", "")
        playbook = scenario.get("playbook_text", "")
        actions = scenario.get("available_actions", [])
        root_cause_kw = scenario.get("root_cause_keywords", [])
        action_domains = scenario.get("action_domains", {})

        domain_text = ""
        for domain, acts in action_domains.items():
            domain_text += f"  {domain}: {', '.join(acts)}\n"

        parts = [
            f"DevOps Incident: {scenario_id}",
            f"\nDescription: {description}",
            f"\nPlaybook:\n{playbook}",
            f"\nAvailable Actions: {', '.join(actions)}",
            f"\nActions by Domain:\n{domain_text}",
            f"\nRoot Cause Keywords: {', '.join(root_cause_kw)}",
            "\nYou are the Incident Commander. Use tools to observe domains, "
            "communicate findings, and execute directives to resolve this incident.",
            "You have 7 domain agents: AppOps, InfraOps, DatabaseOps, NetworkOps, SecOps, MiddlewareOps, ObservabilityOps.",
            "A Supervisor reviews your directives for safety.",
        ]

        if "initial_state" in scenario:
            parts.insert(2, f"\nInitial State:\n{json.dumps(scenario['initial_state'], indent=2)}")

        prompt_text = "\n".join(parts)
        prompts.append({"prompt": [{"role": "user", "content": prompt_text}]})

    if len(prompts) < 32:
        repeat_factor = max(1, 32 // len(prompts))
        prompts = prompts * repeat_factor

    return Dataset.from_list(prompts)


def main():
    parser = argparse.ArgumentParser(description="OpsSim-AI Multi-Agent GRPO Training")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--output_dir", type=str, default="./opssim-grpo-output",
                        help="Directory to save model checkpoints")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--per_device_batch_size", type=int, default=2,
                        help="Batch size per device")
    parser.add_argument("--num_generations", type=int, default=4,
                        help="Number of completions per prompt for GRPO")
    parser.add_argument("--max_completion_length", type=int, default=512,
                        help="Maximum completion token length")
    parser.add_argument("--learning_rate", type=float, default=1e-6,
                        help="Learning rate")
    parser.add_argument("--max_tool_calling_iterations", type=int, default=15,
                        help="Max tool-calling rounds per episode")
    parser.add_argument("--use_peft", action="store_true", default=True,
                        help="Use LoRA/PEFT for memory efficiency")
    parser.add_argument("--no_peft", action="store_true",
                        help="Disable PEFT/LoRA")
    parser.add_argument("--logging_steps", type=int, default=1,
                        help="Log every N steps")
    parser.add_argument("--save_steps", type=int, default=50,
                        help="Save checkpoint every N steps")
    args = parser.parse_args()

    print("Building multi-agent training dataset...")
    dataset = build_dataset()
    print(f"Dataset size: {len(dataset)} prompts")

    peft_config = None
    if args.use_peft and not args.no_peft:
        try:
            from peft import LoraConfig
            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                task_type="CAUSAL_LM",
            )
            print("Using LoRA with r=16, alpha=32")
        except ImportError:
            print("Warning: peft not installed, training without LoRA")

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        max_tool_calling_iterations=args.max_tool_calling_iterations,
        log_completions=True,
        bf16=True,
        chat_template_kwargs={"enable_thinking": False},
    )

    print(f"Training model: {args.model}")
    print(f"Output dir: {args.output_dir}")

    trainer = GRPOTrainer(
        model=args.model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=reward_func,
        environment_factory=MultiAgentTrainingEnv,
        peft_config=peft_config,
    )

    trainer.train()

    trainer.save_model(os.path.join(args.output_dir, "final"))
    print(f"Training complete. Model saved to {args.output_dir}/final")


if __name__ == "__main__":
    main()
