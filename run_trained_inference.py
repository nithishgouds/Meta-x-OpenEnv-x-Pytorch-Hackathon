import argparse
import copy
import json
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from generate_sft_dataset import apply_effects, build_prompt, load_scenarios


def extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def load_policy(base_model: str, adapter: str, precision: str):
    if precision == "bf16":
        torch_dtype = torch.bfloat16
    elif precision == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tokenizer, model


def find_scenario(scenarios: list[dict[str, Any]], scenario_id: str) -> dict[str, Any]:
    for scenario in scenarios:
        if scenario.get("scenario_id") == scenario_id:
            return scenario
    raise ValueError(f"Scenario not found: {scenario_id}")


def resolve_completed_actions(scenario: dict[str, Any], completed_actions_arg: str, step_idx: int) -> list[str]:
    if completed_actions_arg.strip():
        return [part.strip() for part in completed_actions_arg.split(",") if part.strip()]
    optimal = scenario.get("optimal_solution_path", []) or []
    if step_idx <= 1:
        return []
    return optimal[: step_idx - 1]


def replay_state(scenario: dict[str, Any], completed_actions: list[str]) -> tuple[dict[str, Any], list[float]]:
    state = copy.deepcopy(scenario.get("initial_state", {}) or {})
    transition_rules = scenario.get("transition_rules", {}) or {}
    rewards = []
    for action in completed_actions:
        rule = transition_rules.get(action, {}) or {}
        if rule.get("effects"):
            apply_effects(state, rule["effects"])
        rewards.append(float(rule.get("reward", 0.1)))
    return state, rewards


def render_chat(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"USER:\n{prompt}\n\nASSISTANT:\n"


def generate_response(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    rendered = render_chat(tokenizer, prompt)
    inputs = tokenizer(rendered, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one-step inference with the trained OpsSim LoRA policy.")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", default="meancodi/opssim-qwen25-1p5b-sft-lora")
    parser.add_argument("--input", default="tasks/cascade.json")
    parser.add_argument("--scenario-id", default="cascade_001_checkout_meltdown")
    parser.add_argument("--step-idx", type=int, default=1)
    parser.add_argument("--completed-actions", default="", help="Comma-separated completed actions. If omitted, replay optimal path up to step-1.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--show-prompt", action="store_true")
    args = parser.parse_args()

    scenarios = load_scenarios(args.input)
    scenario = find_scenario(scenarios, args.scenario_id)
    total_steps = len(scenario.get("optimal_solution_path", []) or []) or 1
    step_idx = max(1, min(args.step_idx, total_steps))
    completed_actions = resolve_completed_actions(scenario, args.completed_actions, step_idx)
    state, completed_rewards = replay_state(scenario, completed_actions)

    prompt = build_prompt(
        scenario=scenario,
        state=state,
        step_idx=step_idx,
        total_steps=total_steps,
        completed=completed_actions,
        completed_rewards=completed_rewards,
        candidate=None,
    )

    tokenizer, model = load_policy(args.base_model, args.adapter, args.precision)
    raw_output = generate_response(
        tokenizer=tokenizer,
        model=model,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    parsed = extract_json(raw_output)

    result = {
        "scenario_id": scenario.get("scenario_id"),
        "step_idx": step_idx,
        "completed_actions": completed_actions,
        "adapter": args.adapter,
        "base_model": args.base_model,
        "parsed_json": parsed,
        "raw_output": raw_output,
    }

    if args.show_prompt:
        result["prompt"] = prompt

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
