import os
import sys
import json
from unittest.mock import patch, MagicMock
os.environ.setdefault("HF_TOKEN", "test_token_dummy")
sys.path.insert(0, r"c:\Users\srimanish.surago\Downloads\github\Hackathon")

from env import DevOpsEnv, AGENT_DOMAIN_MAP
from multi_agent import WarRoom
from models import Reward


def test_agent_memory():
    from inference import AgentMemory, _diff_states

    mem = AgentMemory()
    assert len(mem.actions) == 0
    assert len(mem.completed_actions) == 0

    prev = {"checkout_status": "error_500", "discovered": {"payment_checked": False}}
    curr = {"checkout_status": "error_500", "discovered": {"payment_checked": True}}
    mem.record(1, "investigate_payment_service", "AppOps", 0.2, prev, curr)

    assert "investigate_payment_service" in mem.completed_actions
    assert "investigate_payment_service" in mem.successful_actions
    assert len(mem.reward_trend) == 1
    assert mem.reward_trend[0] == 0.2
    assert not mem.is_stagnating()
    print("[PASS] AgentMemory basic recording")

    mem.record(2, "bad_action", "InfraOps", -0.5, curr, curr)
    assert "bad_action" in mem.failed_actions
    assert mem.last_failed()["action"] == "bad_action"
    print("[PASS] AgentMemory failure tracking")

    mem.record(3, "another_bad", "InfraOps", -0.3, curr, curr)
    mem.record(4, "third_bad", "InfraOps", -0.2, curr, curr)
    assert mem.is_stagnating()
    print("[PASS] AgentMemory stagnation detection")

    history = mem.format_history()
    assert "[+]" in history
    assert "[x]" in history
    print("[PASS] AgentMemory format_history")


def test_plan_tracker():
    from inference import PlanTracker

    pt = PlanTracker()
    assert "No plan yet" in pt.format_plan()

    pt.update(["investigate_cache", "reboot_redis", "flush_db_connections"])
    assert pt.revision_count == 1
    assert len(pt.current_plan) == 3

    pt.mark_done("investigate_cache")
    assert "investigate_cache" not in pt.current_plan
    assert len(pt.current_plan) == 2

    pt.update(["reboot_redis", "investigate_database", "flush_db_connections", "restart_checkout"])
    assert pt.revision_count == 2
    print("[PASS] PlanTracker plan evolution")


def test_diff_states():
    from inference import _diff_states

    prev = {"checkout_status": "error_500", "discovered": {"payment_checked": False}}
    curr = {"checkout_status": "error_500", "discovered": {"payment_checked": True}}
    diff = _diff_states(prev, curr)
    assert "discovered.payment_checked" in diff
    assert "False->True" in diff["discovered.payment_checked"]
    print("[PASS] _diff_states")


def test_helpers():
    from inference import _agent_for_action, _get_anomalies, _extract_json, DOMAIN_TO_AGENT

    action_domains = {
        "app": ["investigate_payment_service", "restart_checkout"],
        "infra": ["investigate_cache", "reboot_redis"],
        "database": ["investigate_database", "flush_db_connections"],
        "observability": ["analyze_metrics"],
    }

    assert _agent_for_action("investigate_payment_service", action_domains) == "AppOps"
    assert _agent_for_action("reboot_redis", action_domains) == "InfraOps"
    assert _agent_for_action("flush_db_connections", action_domains) == "DatabaseOps"
    assert _agent_for_action("nonexistent", action_domains) is None
    print("[PASS] _agent_for_action")

    state = {
        "checkout_status": "error_500",
        "payment_service": "timeout_upstream",
        "network_connectivity": "healthy",
        "discovered": {"payment_checked": False},
    }
    anomalies = _get_anomalies(state)
    assert any("error" in a.lower() or "CRITICAL" in a for a in anomalies)
    assert any("timeout" in a.lower() for a in anomalies)
    assert not any("healthy" in a.lower() for a in anomalies)
    print(f"[PASS] _get_anomalies found {len(anomalies)} anomalies")

    assert _extract_json('{"a": 1}') == {"a": 1}
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _extract_json('Sure! Here is: {"a": 1} done.') == {"a": 1}
    assert _extract_json('garbage') is None
    print("[PASS] _extract_json")


def test_state_aware_select():
    from inference import _state_aware_select, AgentMemory

    room = WarRoom(seed=42, max_steps=15)
    room.reset()

    mem = AgentMemory()
    action_domains = room.env.state_data.get("action_domains", {})
    obs_actions = set(action_domains.get("observability", []))
    available = room.env.observation.available_actions or []

    a, ag = _state_aware_select(room.env, available, mem, obs_actions, action_domains)
    assert a is not None
    assert ag is not None
    assert a not in obs_actions
    assert "investigate" in a or "check" in a
    print(f"[PASS] _state_aware_select initial: {a} -> {ag}")

    prev_state = json.loads(json.dumps(room.env.state_data["state"]))
    mem.record(1, a, ag, 0.2, prev_state, room.env.state_data["state"])
    a2, ag2 = _state_aware_select(room.env, available, mem, obs_actions, action_domains)
    assert a2 != a
    assert a2 not in obs_actions
    print(f"[PASS] _state_aware_select after first action: {a2} -> {ag2}")


def test_state_aware_no_optimal_dependency():
    from inference import _state_aware_select, AgentMemory

    room = WarRoom(seed=42, max_steps=15)
    room.reset()

    room.env.state_data["optimal_solution_path"] = []

    mem = AgentMemory()
    action_domains = room.env.state_data.get("action_domains", {})
    obs_actions = set(action_domains.get("observability", []))
    available = room.env.observation.available_actions or []

    a, ag = _state_aware_select(room.env, available, mem, obs_actions, action_domains)
    assert a is not None
    assert a not in obs_actions
    print(f"[PASS] _state_aware_select works without optimal_solution_path: {a} -> {ag}")


def test_parse_planning_response():
    from inference import _parse_planning_response, AgentMemory

    room = WarRoom(seed=42, max_steps=15)
    room.reset()

    mem = AgentMemory()
    action_domains = room.env.state_data.get("action_domains", {})
    obs_actions = set(action_domains.get("observability", []))
    available = room.env.observation.available_actions or []

    valid_response = json.dumps({
        "analysis": "Redis cache is down causing payment timeouts",
        "plan": ["investigate_payment_service", "investigate_cache", "reboot_redis"],
        "next_action": "investigate_payment_service",
        "target_agent": "AppOps",
        "reasoning": "Need to investigate payment service first"
    })
    result = _parse_planning_response(valid_response, available, action_domains, mem, obs_actions, room.env)
    assert result["action"] == "investigate_payment_service"
    assert result["target_agent"] == "AppOps"
    assert result["llm_decided"] == True
    assert len(result["plan"]) == 3
    print("[PASS] parse_planning_response with valid LLM output")

    obs_response = json.dumps({
        "analysis": "Need metrics",
        "plan": ["analyze_metrics"],
        "next_action": "analyze_metrics",
        "target_agent": "ObservabilityOps",
        "reasoning": "Check metrics"
    })
    result2 = _parse_planning_response(obs_response, available, action_domains, mem, obs_actions, room.env)
    assert result2["action"] != "analyze_metrics"
    assert result2["action"] not in obs_actions
    print(f"[PASS] parse_planning_response rejects obs actions, falls to state-aware: {result2['action']}")

    result3 = _parse_planning_response("GARBAGE", available, action_domains, mem, obs_actions, room.env)
    assert result3["action"] not in obs_actions
    assert result3["llm_decided"] == False
    print(f"[PASS] parse_planning_response garbage -> state-aware fallback: {result3['action']}")

    wrong_agent = json.dumps({
        "analysis": "Fix cache",
        "plan": ["reboot_redis"],
        "next_action": "reboot_redis",
        "target_agent": "AppOps",
        "reasoning": "Fix redis"
    })
    result4 = _parse_planning_response(wrong_agent, available, action_domains, mem, obs_actions, room.env)
    assert result4["action"] == "reboot_redis"
    assert result4["target_agent"] == "InfraOps"
    print("[PASS] parse_planning_response corrects wrong agent assignment")


def test_conflict_avoidance():
    from inference import _state_aware_select, AgentMemory

    room = WarRoom(seed=42, max_steps=15)
    room.reset()

    room.env.state_data["state"]["discovered"]["cache_checked"] = True
    room.env.state_data["state"]["redis_cache"] = "connection_refused"
    room.env.state_data["state"]["discovered"]["root_cause_found"] = True

    mem = AgentMemory()
    prev = json.loads(json.dumps(room.env.state_data["state"]))
    mem.record(1, "investigate_cache", "InfraOps", 0.25, prev, room.env.state_data["state"])

    action_domains = room.env.state_data.get("action_domains", {})
    obs_actions = set(action_domains.get("observability", []))
    available = room.env.observation.available_actions or []

    a, ag = _state_aware_select(room.env, available, mem, obs_actions, action_domains)
    assert a != "reboot_redis" or ag == "InfraOps"
    print(f"[PASS] conflict avoidance after investigate_cache: picked {a} -> {ag} (avoids immediate reboot_redis conflict)")


def test_full_episode_state_aware_fallback():
    call_count = [0]

    def mock_call_llm(prompt, max_tokens=250):
        call_count[0] += 1
        if "ObservabilityOps" in prompt:
            return json.dumps({
                "root_cause_analysis": "redis cache connection_refused causing payment timeout and checkout error",
                "cascade_chain": "redis failure -> payment timeout -> checkout error_500"
            })
        return "INVALID LLM OUTPUT FORCING STATE-AWARE FALLBACK"

    with patch("inference.call_llm", side_effect=mock_call_llm):
        from inference import _run_episode_core

        room = WarRoom(seed=42, max_steps=15)
        room.reset()

        rewards_list = _run_episode_core(room)

    total = room.get_total_reward()
    progress = room.get_progress()
    comm_count = room.env.communication_count

    print(f"\n--- STATE-AWARE FALLBACK EPISODE ---")
    print(f"Total reward: {total:.3f}")
    print(f"Progress: {progress:.0%}")
    print(f"Steps: {room.step_count}")
    print(f"Communication count: {comm_count}")
    print(f"LLM calls: {call_count[0]}")
    print(f"Done: {room.is_done()}")

    executed = [h["action"] for h in room.action_history]
    print(f"Actions executed: {executed}")
    print(f"Per-step rewards: {[f'{r:.3f}' for r in rewards_list]}")

    assert comm_count == 1, f"Expected 1 communication, got {comm_count}"
    print("[PASS] Communication minimized to 1")

    assert room.is_done(), "Episode should complete (SLA met)"
    print("[PASS] Episode completed successfully")

    assert total > 0, f"Total reward should be positive, got {total:.3f}"
    print(f"[PASS] Positive total reward: {total:.3f}")

    assert progress == 1.0, f"Expected 100% progress, got {progress:.0%}"
    print("[PASS] 100% SLA progress achieved")

    assert len(set(executed)) == len(executed), f"Duplicate actions detected: {executed}"
    print("[PASS] No duplicate actions")

    for a in executed:
        assert a not in {"analyze_metrics", "check_alerts", "trace_requests", "correlate_logs"}, f"Observability action in execution: {a}"
    print("[PASS] No observability actions in execution")


def test_full_episode_llm_decides():
    call_count = [0]
    step_actions = [
        "investigate_payment_service",
        "investigate_cache",
        "reboot_redis",
        "investigate_database",
        "flush_db_connections",
        "restart_checkout",
    ]
    step_plans = [
        ["investigate_payment_service", "investigate_cache", "reboot_redis", "investigate_database", "flush_db_connections", "restart_checkout"],
        ["investigate_cache", "reboot_redis", "investigate_database", "flush_db_connections", "restart_checkout"],
        ["reboot_redis", "investigate_database", "flush_db_connections", "restart_checkout"],
        ["investigate_database", "flush_db_connections", "restart_checkout"],
        ["flush_db_connections", "restart_checkout"],
        ["restart_checkout"],
    ]

    agent_map = {
        "investigate_payment_service": "AppOps",
        "investigate_cache": "InfraOps",
        "reboot_redis": "InfraOps",
        "investigate_database": "DatabaseOps",
        "flush_db_connections": "DatabaseOps",
        "restart_checkout": "AppOps",
    }

    def mock_call_llm(prompt, max_tokens=250):
        call_count[0] += 1
        if "ObservabilityOps" in prompt:
            return json.dumps({
                "root_cause_analysis": "redis cache connection_refused causing payment timeout and checkout error",
                "cascade_chain": "redis failure -> payment timeout -> checkout error_500"
            })

        ic_call = call_count[0] - 2
        if ic_call < 0:
            ic_call = 0
        idx = min(ic_call, len(step_actions) - 1)
        return json.dumps({
            "analysis": f"Step {idx+1}: addressing {step_actions[idx]}",
            "plan": step_plans[idx],
            "next_action": step_actions[idx],
            "target_agent": agent_map[step_actions[idx]],
            "reasoning": f"Following investigation-first approach for {step_actions[idx]}"
        })

    with patch("inference.call_llm", side_effect=mock_call_llm):
        from inference import _run_episode_core

        room = WarRoom(seed=42, max_steps=15)
        room.reset()

        rewards_list = _run_episode_core(room)

    total = room.get_total_reward()
    progress = room.get_progress()
    executed = [h["action"] for h in room.action_history]

    print(f"\n--- LLM-DRIVEN EPISODE ---")
    print(f"Total reward: {total:.3f}")
    print(f"Progress: {progress:.0%}")
    print(f"Steps: {room.step_count}")
    print(f"Actions executed: {executed}")
    print(f"Per-step rewards: {[f'{r:.3f}' for r in rewards_list]}")

    assert room.is_done(), "Episode should complete"
    assert total > 0, f"Total reward should be positive, got {total:.3f}"
    assert progress == 1.0
    assert executed == step_actions, f"Expected optimal sequence from LLM: {step_actions}, got {executed}"
    print(f"[PASS] LLM-driven episode follows plan: total_reward={total:.3f}")


def test_dynamic_reward_normalization():
    from inference import _calculate_dynamic_min_reward, _calculate_dynamic_max_reward

    room = WarRoom(seed=42, max_steps=30)
    room.reset()

    min_r = _calculate_dynamic_min_reward(room.env, 15)
    max_r = _calculate_dynamic_max_reward(room.env, 15)

    assert min_r < 0, f"min_reward should be negative: {min_r}"
    assert max_r > 0, f"max_reward should be positive: {max_r}"
    assert max_r > min_r, f"max_reward ({max_r}) should be > min_reward ({min_r})"
    print(f"[PASS] Dynamic normalization: min={min_r:.3f}, max={max_r:.3f}")

    room2 = WarRoom(seed=42, max_steps=30)
    room2.reset()
    min_r2 = _calculate_dynamic_min_reward(room2.env, 30)
    max_r2 = _calculate_dynamic_max_reward(room2.env, 30)
    assert min_r2 != min_r or max_r2 != max_r
    print(f"[PASS] Dynamic normalization adapts to max_steps: min30={min_r2:.3f}, max30={max_r2:.3f}")


if __name__ == "__main__":
    test_agent_memory()
    print()
    test_plan_tracker()
    print()
    test_diff_states()
    print()
    test_helpers()
    print()
    test_state_aware_select()
    print()
    test_state_aware_no_optimal_dependency()
    print()
    test_parse_planning_response()
    print()
    test_conflict_avoidance()
    print()
    test_full_episode_state_aware_fallback()
    print()
    test_full_episode_llm_decides()
    print()
    test_dynamic_reward_normalization()
    print()
    print("=== ALL TESTS PASSED ===")
