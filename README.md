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

# OpsSim-AI: Distributed War Room (Multi-Agent Cascading Failure System)

*When systems cascade-fail, no single operator sees the full picture.*

## Overview

OpsSim-AI is an OpenEnv-compatible multi-agent environment that simulates cascading production failures resolved through a distributed war room. A team of four specialized agents must coordinate under partial observability to diagnose and recover interconnected services before SLA violations escalate.

### Why Multi-Agent?

Real incidents are never handled by one person. Production war rooms involve:
- **Domain specialists** who each see only their own systems
- **An Incident Commander** who reads reports and issues directives
- **Shared communication** through an incident channel
- **Coordination overhead** — every message costs time, every wrong directive bleeds revenue

OpsSim-AI captures this structure exactly.

## Architecture

### Four Agents

| Agent | Role | Visibility |
|-------|------|-----------|
| **IncidentCommander** | Reads all reports, issues directives | Incident channel + playbook |
| **AppOps** | Application layer specialist | App metrics (checkout, payment, latency) |
| **InfraOps** | Infrastructure specialist | Infra metrics (redis, DNS, network, load balancers) |
| **DatabaseOps** | Database specialist | DB metrics (pools, replication, locks, queries) |

### Decision Flow

```
1. Each domain agent OBSERVES its partial view of the system
2. Each domain agent COMMUNICATES findings to the incident channel
3. The Incident Commander READS all reports
4. The IC issues a DIRECTIVE: which agent should execute which action
5. The targeted agent EXECUTES the action
6. The environment updates state and calculates reward
7. Repeat until SLA is met, violated, or max steps reached
```

### Partial Observability

Each domain agent only sees state fields relevant to its domain. For example, in a checkout meltdown scenario:
- **AppOps** sees: `checkout_status`, `payment_service`, `phase`
- **InfraOps** sees: `redis_cache`, `dns_resolution`
- **DatabaseOps** sees: `database_pool`

No single agent can solve the incident alone. The IC must synthesize partial reports to make correct decisions.

## Task: Cascade Scenarios

All scenarios are cascading failures (`tasks/cascade.json`). Each scenario includes:

- **initial_state**: The system state when the incident begins
- **playbook_text**: Operational runbook the IC should follow
- **available_actions**: All possible actions across all domains
- **penalties**: Actions that cause harm (guardrail violations)
- **optimal_solution_path**: The correct sequence of actions
- **transition_rules**: How actions change system state (with conditions)
- **bleed_rules**: Ongoing damage while the system stays broken
- **sla_rules**: Required conditions for resolution + forbidden states
- **domain_visibility**: Which state fields each domain can see
- **action_domains**: Which actions each domain agent can execute

### 10 Scenarios

1. **cascade_001** — Checkout Meltdown (payment → cache → DB cascade)
2. **cascade_002** — Deploy Gone Wrong (bad deploy → rollback → DNS)
3. **cascade_003** — Network Partition (cross-region connectivity failure)
4. **cascade_004** — Storage Corruption Chain (disk → replication → data loss)
5. **cascade_005** — Auth Breach Chain (credential leak → lockdown → recovery)
6. **cascade_006** — Network Outage Cascade (BGP → CDN → routing failure)
7. **cascade_007** — Database Overload Chain (query storm → pool exhaustion)
8. **cascade_008** — App Crash Cascade (OOM → restart loops → dependency failures)
9. **cascade_009** — Multi-Region Failover (primary region failure → failover)
10. **cascade_010** — DB Lock Revenue Cascade (lock contention → transaction failures)

## Reward System

Every action produces a composite reward:

| Component | Description |
|-----------|-------------|
| **Bleed** | Ongoing damage from unresolved failures (dynamic per state) |
| **Action Penalty** | Cost of harmful, invalid, or do-nothing actions |
| **Repeat Penalty** | -0.15 × repeat count for repeated actions |
| **Urgency Penalty** | -0.05 × step number (time pressure) |
| **Progress Reward** | +0.05 to +0.3 for state improvements |
| **Coordination Reward** | +0.1 if agent acts in own domain, -0.1 if not |
| **Conflict Penalty** | -0.05 if same action as previous step |
| **Communication Cost** | -0.01 × total messages (coordination overhead) |
| **SLA Success** | +1.0 for meeting all required SLA conditions |
| **SLA Violation** | Large penalty (scenario-specific) for forbidden states |

### Normalization

Scores are normalized to [0, 1] using dynamic min/max reward bounds calculated from each scenario's bleed rules, penalties, and SLA configuration.

## Project Structure

```
├── env.py              # DevOpsEnv — OpenEnv-compatible environment
├── models.py           # Pydantic models (Action, Observation, Reward, State)
├── multi_agent.py      # WarRoom orchestrator + DomainAgent + IncidentCommander
├── inference.py        # LLM-based multi-agent inference loop + grading
├── train.py            # GRPO training with TRL (tool-calling, LoRA)
├── server/
│   └── app.py          # FastAPI server for HF Space deployment
├── tasks/
│   └── cascade.json    # 10 cascading failure scenarios
├── eval_results/
│   └── results.json    # Evaluation output
├── openenv.yaml        # OpenEnv manifest
├── pyproject.toml      # Package configuration
├── Dockerfile          # Container for HF Space
├── requirements.txt    # Dependencies
└── README.md
```

## Usage

### Inference (LLM-based)

```bash
export HF_TOKEN=your_token_here
python inference.py
```

Runs all scenarios with the configured LLM. Each episode:
1. Resets the war room
2. Domain agents observe and communicate
3. IC reads reports and issues directives
4. Actions execute until resolution or timeout

### Training (GRPO)

```bash
python train.py --model Qwen/Qwen3-0.6B --output_dir ./opssim-grpo-output
```

Uses TRL's GRPOTrainer with tool-calling:
- The model plays as Incident Commander
- Tools: `observe_domain`, `communicate`, `execute_directive`
- LoRA (r=16, alpha=32) for memory efficiency
- Compatible with 2 vCPU / 8 GB RAM

### Server

```bash
python server/app.py
```

Endpoints:
- `POST /reset` — Start new incident
- `POST /communicate` — Post to incident channel
- `POST /step` — Execute IC directive
- `GET /state` — Current system state
- `GET /incident_channel` — All messages
- `GET /health` — Health check

### Docker

```bash
docker build -t opssim-ai .
docker run -p 7860:7860 opssim-ai
```

## Design Principles

- **Deterministic**: temperature=0, fixed seed=42, no uncontrolled randomness
- **OpenEnv-compliant**: Extends OpenEnv Action, Observation, State base types
- **Resource-efficient**: Runs within 2 vCPU / 8 GB RAM constraints
- **Cascade-only**: No easy/medium/hard tiers — every scenario is a multi-service cascading failure
- **Partial observability**: Domain agents see only their slice of state
- **Communication has cost**: Every message to the incident channel costs -0.02 reward

## Authors

- Sandeep
- Venkatesh
- Nithish
