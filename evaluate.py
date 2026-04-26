"""
OpsSim-AI Evaluation Script
=============================
Evaluates a trained (or baseline) model on the DevOpsEnv environment.
Generates reward curves, per-task metrics, and comparison plots.

Usage:
    # Evaluate a model
    python evaluate.py --model Qwen/Qwen3-0.6B --task all

    # Evaluate trained model
    python evaluate.py --model ./opssim-grpo-output/final --task all

    # Compare before and after training
    python evaluate.py --compare Qwen/Qwen3-0.6B ./opssim-grpo-output/final --task easy
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import DevOpsEnv
from models import Action


# ---------------------------------------------------------------------------
# Evaluation runner (rule-based baseline or LLM)
# ---------------------------------------------------------------------------

def run_episode(env, task_type, agent_fn, max_steps=8):
    """Run a single episode and return metrics."""
    obs = env.reset(task=task_type)
    total_reward = 0.0
    step_rewards = []
    actions_taken = []
    done = False
    steps = 0

    while not done and steps < max_steps:
        action_type = agent_fn(obs, task_type)
        actions_taken.append(action_type)

        action = Action(action_type=action_type)
        obs, reward_obj, done, info = env.step(action)
        total_reward += reward_obj.value
        step_rewards.append(reward_obj.value)
        steps += 1

    return {
        "total_reward": total_reward,
        "steps": steps,
        "done": done,
        "step_rewards": step_rewards,
        "actions": actions_taken,
    }


def llm_agent_factory(model_path):
    """
    Create an LLM-based agent using transformers.
    Returns a function (obs, task_type) -> action_type.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    def agent_fn(obs, task_type):
        # Build prompt
        parts = []
        if obs.user_message:
            parts.append(f"Incident Report: {obs.user_message}")
        if obs.user_messages:
            parts.append("User Reports:\n" + "\n".join(f"  - {m}" for m in obs.user_messages))
        if obs.logs:
            parts.append(f"System Logs:\n{obs.logs}")
        if obs.config:
            parts.append(f"Config: {json.dumps(obs.config)}")
        if obs.system_metrics:
            parts.append(f"Metrics: {json.dumps(obs.system_metrics)}")
        if obs.system_state:
            parts.append(f"State: {json.dumps(obs.system_state, default=str)}")

        available = obs.available_actions or []
        parts.append(f"\nAvailable Actions: {', '.join(available)}")
        parts.append(
            "\nChoose exactly ONE action from the available actions above. "
            "Respond with only the action name, nothing else."
        )

        prompt = "\n\n".join(parts)
        messages = [{"role": "user", "content": prompt}]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                temperature=0.1,
                do_sample=True,
                top_p=0.9,
            )

        response = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

        # Extract action from response
        for action in available:
            if action in response:
                return action

        # Fallback: return first action
        return available[0] if available else "do_nothing"

    return agent_fn


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate_agent(agent_fn, task_types, num_episodes=20, seed=42):
    """Evaluate an agent across task types and return structured metrics."""
    results = defaultdict(list)

    for task_type in task_types:
        env = DevOpsEnv(seed=seed, max_steps=8)
        for ep in range(num_episodes):
            episode_result = run_episode(env, task_type, agent_fn)
            episode_result["episode"] = ep
            episode_result["task_type"] = task_type
            results[task_type].append(episode_result)

    return dict(results)


def compute_summary(results):
    """Compute summary statistics from evaluation results."""
    summary = {}
    for task_type, episodes in results.items():
        rewards = [ep["total_reward"] for ep in episodes]
        steps = [ep["steps"] for ep in episodes]
        success = [1.0 if ep["done"] and ep["total_reward"] > 0 else 0.0 for ep in episodes]

        summary[task_type] = {
            "mean_reward": sum(rewards) / len(rewards),
            "max_reward": max(rewards),
            "min_reward": min(rewards),
            "mean_steps": sum(steps) / len(steps),
            "success_rate": sum(success) / len(success),
            "num_episodes": len(episodes),
        }
    return summary


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_rewards(results, title, output_path):
    """Plot reward distribution per task type."""
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]

    for ax, (task_type, episodes) in zip(axes, results.items()):
        rewards = [ep["total_reward"] for ep in episodes]
        episodes_x = list(range(len(rewards)))

        ax.bar(episodes_x, rewards, alpha=0.7, color="steelblue")
        ax.axhline(y=sum(rewards) / len(rewards), color="red", linestyle="--", label="Mean")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.set_title(f"{task_type.capitalize()} Tasks")
        ax.legend()

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved reward plot to {output_path}")


def plot_comparison(results_before, results_after, label_before, label_after, output_path):
    """Plot comparison of two agents side by side."""
    task_types = sorted(set(list(results_before.keys()) + list(results_after.keys())))

    fig, axes = plt.subplots(1, len(task_types), figsize=(6 * len(task_types), 5))
    if len(task_types) == 1:
        axes = [axes]

    for ax, task_type in zip(axes, task_types):
        means = []
        labels_list = []
        colors = []

        if task_type in results_before:
            rewards = [ep["total_reward"] for ep in results_before[task_type]]
            means.append(sum(rewards) / len(rewards))
            labels_list.append(label_before)
            colors.append("salmon")

        if task_type in results_after:
            rewards = [ep["total_reward"] for ep in results_after[task_type]]
            means.append(sum(rewards) / len(rewards))
            labels_list.append(label_after)
            colors.append("steelblue")

        ax.bar(labels_list, means, color=colors, alpha=0.8)
        ax.set_ylabel("Mean Reward")
        ax.set_title(f"{task_type.capitalize()} Tasks")

    fig.suptitle("Before vs After Training", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison plot to {output_path}")


def plot_reward_curves(results, title, output_path):
    """Plot cumulative reward curves across episodes."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for task_type, episodes in results.items():
        cumulative = []
        running = 0.0
        for ep in episodes:
            running += ep["total_reward"]
            cumulative.append(running / (len(cumulative) + 1))
        ax.plot(cumulative, label=f"{task_type.capitalize()}", linewidth=2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Running Mean Reward")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved reward curves to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OpsSim-AI Evaluation")
    parser.add_argument("--model", type=str, default=None,
                        help="Model to evaluate (HF model ID or local path)")
    parser.add_argument("--task", type=str, default="all",
                        choices=["easy", "medium", "hard", "all"],
                        help="Task difficulty to evaluate")
    parser.add_argument("--num_episodes", type=int, default=20,
                        help="Number of episodes per task type")
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                        help="Directory to save evaluation results and plots")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"),
                        help="Compare two models: --compare <baseline> <trained>")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    task_types = ["easy", "medium", "hard"] if args.task == "all" else [args.task]

    if args.compare:
        # Comparison mode
        model_before, model_after = args.compare
        print(f"Comparing: {model_before} vs {model_after}")

        agent_before = llm_agent_factory(model_before)
        agent_after = llm_agent_factory(model_after)

        print(f"\nEvaluating BEFORE model: {model_before}")
        results_before = evaluate_agent(agent_before, task_types, args.num_episodes, args.seed)
        summary_before = compute_summary(results_before)

        print(f"\nEvaluating AFTER model: {model_after}")
        results_after = evaluate_agent(agent_after, task_types, args.num_episodes, args.seed)
        summary_after = compute_summary(results_after)

        # Print comparison
        print("\n" + "=" * 60)
        print("COMPARISON RESULTS")
        print("=" * 60)
        for task_type in task_types:
            s_before = summary_before.get(task_type, {})
            s_after = summary_after.get(task_type, {})
            print(f"\n{task_type.upper()}:")
            print(f"  Before: mean_reward={s_before.get('mean_reward', 0):.3f}, "
                  f"success_rate={s_before.get('success_rate', 0):.1%}")
            print(f"  After:  mean_reward={s_after.get('mean_reward', 0):.3f}, "
                  f"success_rate={s_after.get('success_rate', 0):.1%}")
            improvement = s_after.get("mean_reward", 0) - s_before.get("mean_reward", 0)
            print(f"  Improvement: {improvement:+.3f}")

        # Save plots
        plot_comparison(
            results_before, results_after,
            os.path.basename(model_before), os.path.basename(model_after),
            os.path.join(args.output_dir, "comparison.png"),
        )

        # Save raw results
        output = {
            "before": {"model": model_before, "summary": summary_before},
            "after": {"model": model_after, "summary": summary_after},
        }
        with open(os.path.join(args.output_dir, "comparison.json"), "w") as f:
            json.dump(output, f, indent=2)

    else:
        # Single model evaluation
        if not args.model:
            parser.error("--model is required (HF model ID or local path)")
        agent_fn = llm_agent_factory(args.model)
        agent_name = os.path.basename(args.model)

        print(f"Evaluating agent: {agent_name}")
        print(f"Task types: {task_types}")
        print(f"Episodes per task: {args.num_episodes}")

        start = time.time()
        results = evaluate_agent(agent_fn, task_types, args.num_episodes, args.seed)
        elapsed = time.time() - start

        summary = compute_summary(results)

        # Print results
        print("\n" + "=" * 60)
        print(f"EVALUATION RESULTS ({agent_name})")
        print("=" * 60)
        for task_type, stats in summary.items():
            print(f"\n{task_type.upper()}:")
            print(f"  Mean Reward:  {stats['mean_reward']:.3f}")
            print(f"  Max Reward:   {stats['max_reward']:.3f}")
            print(f"  Min Reward:   {stats['min_reward']:.3f}")
            print(f"  Mean Steps:   {stats['mean_steps']:.1f}")
            print(f"  Success Rate: {stats['success_rate']:.1%}")
        print(f"\nTime: {elapsed:.1f}s")

        # Save plots
        plot_rewards(results, f"{agent_name} Evaluation", os.path.join(args.output_dir, "rewards.png"))
        plot_reward_curves(results, f"{agent_name} Reward Curves", os.path.join(args.output_dir, "reward_curves.png"))

        # Save raw results
        output = {
            "agent": agent_name,
            "model": args.model,
            "summary": summary,
            "elapsed_seconds": elapsed,
        }
        with open(os.path.join(args.output_dir, "results.json"), "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
