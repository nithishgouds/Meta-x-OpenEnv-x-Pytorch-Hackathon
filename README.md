---
title: OpsSim-AI
emoji: 🚨
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
---

# OpsSim-AI: Distributed War Room (9-Agent Cascading Failure System)

Real production failures rarely stay inside one component.

A checkout outage might begin as a Redis failure, spread into payment timeouts, leave stale database connections behind, and tempt an agent into restarting the wrong service too early. OpsSim-AI turns that kind of messy incident into a structured multi-agent environment: agents see only their own slice of the system, communicate through a shared war-room channel, and are rewarded for diagnosing, sequencing, delegating, and recovering correctly.

The goal is not just to pick the right action. The goal is to behave like a reliable incident team under uncertainty: investigate first, identify the root cause, coordinate across domains, avoid unsafe shortcuts, and restore SLA health before the cascade gets worse.

## Overview

OpsSim-AI is an OpenEnv-compatible 9-agent environment simulating cascading production failures resolved through a distributed war room with partial observability, strict responsibility enforcement, and a mathematically rigorous reward function.

The environment targets a specific capability gap in agent evaluation: most benchmarks test whether an agent can solve a task after seeing the whole state. Real incidents are different. Each specialist sees only part of the failure, the root cause may be upstream of the visible symptom, and premature actions can make the system worse.

OpsSim-AI makes those pressures explicit:

- **Partial observability:** each execution agent sees only its domain-specific state.
- **Coordination pressure:** the Incident Commander must synthesize reports and delegate.
- **Responsibility boundaries:** agents are penalized for acting outside their domain.
- **Long-horizon recovery:** many scenarios require investigation, preconditions, and ordered remediation.
- **Safety oversight:** the Supervisor can veto harmful directives before execution.

## Architecture: 9 Agents

### Execution Layer (7 agents)

| Agent | Domain | Responsibility |
|-------|--------|---------------|
| **AppOps** | app | Application metrics, checkout, payment, latency |
| **InfraOps** | infra | Infrastructure, caching, pod health, compute |
| **DatabaseOps** | database | DB pools, replication, queries, locks |
| **NetworkOps** | network | Connectivity, load balancers, DNS, BGP routes |
| **SecOps** | security | Auth, certificates, firewall, access controls |
| **MiddlewareOps** | middleware | API gateway, message queues, service mesh, circuit breakers |
| **ObservabilityOps** | observability | Monitoring, alerts, log pipelines, metrics collection |

### Coordination Layer

| Agent | Role |
|-------|------|
| **Incident Commander (IC)** | Reads all reports, issues directives, plans multi-step resolution |

### Oversight Layer

| Agent | Role |
|-------|------|
| **Supervisor (Fleet AI)** | Evaluates IC directives, vetoes catastrophic actions |

## Environment Mental Model

Think of OpsSim-AI as a simulated production war room.

At the start of each episode, the system is already degraded. The failure may appear in one place, but the cause can live somewhere else: a cache outage causing payment failures, a bad canary deployment creating latency spikes, a database deadlock collapsing checkout, or a regional failover problem creating downstream symptoms.

The agent team must recover the system by repeatedly answering three questions:

1. **What do we know?**
   Each execution agent receives a filtered observation. AppOps may see checkout and payment state; DatabaseOps may see pool or lock state; NetworkOps may see DNS and routing state. No single execution agent starts with the full picture.

2. **What should happen next?**
   Agents communicate observations into the incident channel. The Incident Commander reads the shared channel, tracks progress toward SLA goals, and chooses a target agent plus action.

3. **Is the action safe and correctly owned?**
   The Supervisor evaluates the directive. The environment then checks whether the target agent is responsible for that action, whether prerequisites are met, and whether the action improves or damages the system.

### What the Agent Sees

The observation includes:

- available actions
- current visible system state
- playbook text
- logs / incident description
- incident-channel messages
- SLA goal state
- progress toward recovery
- domain-filtered views for specialist agents

This creates a useful tension: the IC needs a global plan, but the evidence arrives through narrow domain windows.

### What the Agent Can Do

Agents can:

- investigate symptoms and dependencies
- report findings through the incident channel
- execute domain-specific remediation actions
- follow playbook guidance
- recover state through transition rules
- make mistakes such as acting too early, repeating actions, choosing unsafe actions, or assigning work to the wrong domain

Many scenarios include tempting but harmful shortcuts. For example, blindly restarting a service may carry a penalty if the actual root cause has not been fixed.

## Execution Flow (Strict 8-Phase)

```
1. ObservabilityOps analyzes metrics and surfaces root cause
2. Execution agents observe their domain and report to incident channel
3. IC reads all reports from incident channel
4. IC issues directive (target_agent + action)
5. Supervisor evaluates directive — approves or vetoes
6. Target agent executes action
7. Environment updates state
8. Reward is computed (13 components)
```

## Reward Function (13 Components)

The total reward at step t follows the formal equation:

$$R_t = \Delta H(s_t, s_{t-1}) - \left(B_{sys}(s_t) + \sum_{d \in D} B_{loc}(s_t, d)\right) - \lambda \cdot t$$
$$+ Q_{act}(a_t) + R_{seq}(a_t, h_t) - P_{resp}(a_t, e) - P_{conf}(a_t, a_{t-1})$$
$$+ R_{coord}(IC_t, a_t) + R_{obs}(m_{obs}, s_t) + R_{sup}(IC_t) - \gamma \cdot \Sigma(m_t)$$
$$+ \mathbb{1}_{SLA\_Met} \cdot R_{succ}$$

The reward is designed to score incident behavior, not just endpoint success. It rewards health improvements, valid actions, correct sequencing, good delegation, useful observability, and supervisor judgment. It penalizes ongoing degradation, wasted time, wrong-domain actions, conflicts, repeated or idle behavior, and excessive communication.

### Pillar I: System Health & Degradation

| Component | Symbol | Description |
|-----------|--------|-------------|
| Global System Health | ΔH | Positive delta when SLA metrics improve |
| Dynamic Global Bleed | B_sys | Sum of active incident severity weights (not hardcoded) |
| Local Domain Bleed | B_loc | Per-domain degradation penalties |
| Urgency Penalty | P_urg | Linear time-decay: λ·t where λ = 1/max_steps |

### Pillar II: Action & Sequencing Execution

| Component | Symbol | Description |
|-----------|--------|-------------|
| Action Quality | Q_act | Rewards valid actions, penalizes invalid/redundant/do_nothing |
| Sequencing Reward | R_seq | Rewards topologically correct execution order |
| Responsibility Penalty | P_resp | Massive penalty (-5.0) if agent acts outside its domain |
| Conflict Penalty | P_conf | Penalizes mutually exclusive or repeated consecutive actions |

### Pillar III: Coordination & Communication

| Component | Symbol | Description |
|-----------|--------|-------------|
| Coordination Reward | R_coord | IC correctly delegates to right domain agent |
| Observability Contribution | R_obs | ObservabilityOps surfaces correct root cause keywords |
| Supervisor Effect | R_sup | Correct veto of harmful actions / penalty for rubber-stamping |
| Communication Cost | P_comm | γ·Σ(messages) prevents LLM chatter loops |

### Pillar IV: Terminal

| Component | Symbol | Description |
|-----------|--------|-------------|
| Success Reward | R_succ | +2.0 when all SLA conditions met |

## do_nothing Prevention

1. **Action Filtering**: do_nothing removed from available actions when valid actions exist
2. **Escalating Penalty**: -0.3 × (consecutive_count)^1.5
3. **Stagnation Detection**: Extra penalty when no health improvement for 3+ steps

## Long-Horizon Planning

- **Goal Decomposition**: IC receives unmet SLA goals and must plan multi-step resolution
- **Progress Tracking**: Real-time % of SLA conditions met, injected into all prompts
- **Dependency Awareness**: Action domains + sequencing reward enforce causal ordering
- **Persistent Memory**: Full action history with outcomes tracked across episode
- **Recovery**: Supervisor can veto and suggest alternatives when stuck

## Why It Matters

OpsSim-AI is useful for researchers and builders who care about agent reliability in operational settings.

Real-world incident response is not a single-tool benchmark. It requires diagnosis under partial information, coordination between specialists, safe sequencing, and discipline around ownership. A strong model should know when to investigate, when to act, when to escalate, and when not to touch something.

This environment is interesting because it combines:

- **multi-agent coordination** with explicit roles
- **partial observability** across operational domains
- **stateful cascading failures** with transition rules
- **reward shaping** for investigation, delegation, sequencing, and safety
- **OpenEnv compatibility** for evaluation and training loops

That makes it a compact testbed for studying whether LLM agents can move from "answering correctly" toward operating responsibly in a simulated production system.

## Why This Project Fits the Hackathon Themes

### Theme #1 – Multi-Agent Interactions

- Seven specialist agents observe different parts of the system, so no single agent has the full answer.
- The Incident Commander coordinates through a shared war-room channel, then delegates actions to the right domain owner.
- The Supervisor adds an oversight layer by checking whether directives are safe before execution.

### Theme #2 – Long-Horizon Planning

- Incidents require ordered recovery: investigate symptoms, identify root cause, satisfy preconditions, then remediate.
- Rewards are delayed because early diagnostic steps may only pay off after later fixes restore SLA health.
- The environment tracks history, progress, failed actions, and stagnation so agents must recover from mistakes.

### Theme #3 – World Modeling (Professional Tasks)

- The environment simulates realistic production systems with application, infrastructure, database, network, security, middleware, and observability state.
- Agents operate under partial observability, matching how real incident teams only see the signals available to their role.
- Success depends on causal reasoning: agents must connect upstream failures to downstream symptoms instead of chasing surface alerts.

### Theme #4 – Self-Improvement

- The reward function breaks behavior into interpretable feedback signals such as sequencing, delegation, safety, and communication cost.
- `generate_sft_dataset.py` and `train.py` support improvement loops by turning scenarios and reward-guided behavior into data for supervised or RL-style training.
- Evaluation traces can show where an agent stalls, repeats bad actions, or delegates incorrectly, making failures easier to learn from.

## Project Structure

```
├── env.py              # DevOpsEnv — 13-component reward, 7 domains, OpenEnv-compatible
├── models.py           # Pydantic models (Action, Observation, Reward with 13 fields, State)
├── multi_agent.py      # WarRoom + 9 agents (7 execution + IC + Supervisor)
├── inference.py        # LLM loop: 8-phase execution with Supervisor veto
├── train.py            # GRPO training with TRL (tool-calling, LoRA)
├── generate_sft_dataset.py # SFT dataset generation from cascade scenarios
├── server/app.py       # FastAPI server for HF Space
├── tasks/cascade.json  # 10 scenarios × 7 domains
├── eval_results/       # Evaluation artifacts
├── openenv.yaml        # OpenEnv manifest
├── Dockerfile          # Container
└── requirements.txt    # Dependencies
```

## Usage

### Inference

```bash
export HF_TOKEN=your_token
python inference.py
```

### Training

```bash
python train.py --model Qwen/Qwen3-0.6B --output_dir ./opssim-grpo-output
```

### Server

```bash
python server/app.py
```

## Authors

- Sandeep
- Venkatesh
- Nithish

## What Was Improved

- Replaced the confusing opening line with a simpler sentence about failures spreading across components.
- Added a short hackathon-theme alignment section focused on multi-agent interaction, long-horizon planning, professional world modeling, and applicable self-improvement loops.
- Kept the edits targeted and left the existing technical, training, reward, and usage content intact.
