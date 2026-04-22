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

## Overview

OpsSim-AI is an OpenEnv-compatible 9-agent environment simulating cascading production failures resolved through a distributed war room with partial observability, strict responsibility enforcement, and a mathematically rigorous reward function.

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

## Project Structure

```
├── env.py              # DevOpsEnv — 13-component reward, 7 domains, OpenEnv-compatible
├── models.py           # Pydantic models (Action, Observation, Reward with 13 fields, State)
├── multi_agent.py      # WarRoom + 9 agents (7 execution + IC + Supervisor)
├── inference.py        # LLM loop: 8-phase execution with Supervisor veto
├── train.py            # GRPO training with TRL (tool-calling, LoRA)
├── server/app.py       # FastAPI server for HF Space
├── tasks/cascade.json  # 10 scenarios × 7 domains
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
