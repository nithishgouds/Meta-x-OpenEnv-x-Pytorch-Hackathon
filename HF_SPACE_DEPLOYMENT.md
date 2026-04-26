# Hugging Face Space Deployment Guide — OpsSim-AI

> Step-by-step guide for pushing the OpsSim-AI environment demo to a Hugging Face Space so judges
> can run it directly from their browser.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [SDK Recommendation: Gradio](#2-sdk-recommendation-gradio)
3. [Architecture: What the Space Does](#3-architecture-what-the-space-does)
4. [Files to Include / Exclude](#4-files-to-include--exclude)
5. [Create the Root-Level `app.py` (Gradio)](#5-create-the-root-level-apppy-gradio)
6. [Create Space-Specific `requirements.txt`](#6-create-space-specific-requirementstxt)
7. [README Metadata Block](#7-readme-metadata-block)
8. [Create the Space on Hugging Face](#8-create-the-space-on-hugging-face)
9. [Push from Local Git](#9-push-from-local-git)
10. [Test the Space After Deployment](#10-test-the-space-after-deployment)
11. [Link Everything from README](#11-link-everything-from-readme)
12. [Troubleshooting Common Failures](#12-troubleshooting-common-failures)
13. [Final Hackathon Submission Checklist](#13-final-hackathon-submission-checklist)

---

## 1. Prerequisites

| Requirement | How to Get It |
|-------------|---------------|
| Hugging Face account | https://huggingface.co/join |
| `huggingface-cli` installed | `pip install huggingface_hub` |
| CLI authenticated | `huggingface-cli login` (paste a **write** token) |
| Git installed | https://git-scm.com |
| Git LFS installed | `git lfs install` (only needed if pushing binary files like plots) |
| Python 3.11+ locally | For testing `app.py` before pushing |

**Get a write token:**
1. Go to https://huggingface.co/settings/tokens
2. Create a new token with **Write** access
3. Run:
   ```bash
   huggingface-cli login
   # paste your token when prompted
   ```

---

## 2. SDK Recommendation: Gradio

**Use Gradio SDK**, not Docker. Here's why:

| Factor | Gradio | Docker |
|--------|--------|--------|
| Setup complexity | Low — just `app.py` + `requirements.txt` | Medium — need to maintain `Dockerfile` |
| Judge experience | Click "Run" in browser, see UI immediately | Need to hit API endpoints manually |
| HF integration | Native SDK support, auto-restart, logs in UI | Manual port config, less visibility |
| Dependencies | Only runtime deps (no torch/transformers) | Full image build, slower cold start |
| Free tier | Works on free CPU Spaces | Works but heavier |

Your existing `server/app.py` is a FastAPI REST API — great for programmatic use, but judges
want a **clickable UI**. Gradio gives you that with minimal code.

> **Key insight:** The Space runs the **environment** (simulation), NOT the trained model.
> The model weights live separately at `meancodi/opssim-qwen-3b-v1`. The Space doesn't need
> `torch`, `transformers`, `trl`, `peft`, or `accelerate`.

---

## 3. Architecture: What the Space Does

```
┌─────────────────────────────────────────────────────────┐
│  Hugging Face Space (Gradio SDK, free CPU tier)         │
│                                                         │
│  app.py (Gradio UI)                                     │
│    ├── Scenario selector dropdown                       │
│    ├── "Reset" button → WarRoom.reset()                 │
│    ├── Action input + Agent selector → WarRoom.step()   │
│    ├── Incident channel display (chat-like)             │
│    ├── System state display (JSON or table)             │
│    ├── Reward breakdown display (13 components)         │
│    └── Progress / SLA status                            │
│                                                         │
│  env.py + models.py + multi_agent.py                    │
│    └── DevOpsEnv + WarRoom (pure Python, no GPU)        │
│                                                         │
│  tasks/cascade.json                                     │
│    └── 10 cascading failure scenarios                   │
└─────────────────────────────────────────────────────────┘

         ↕ NOT included in Space

┌─────────────────────────────────────────────────────────┐
│  Hugging Face Model Hub (separate repo)                 │
│  meancodi/opssim-qwen-3b-v1                             │
│    └── LoRA adapter weights (linked from README)        │
└─────────────────────────────────────────────────────────┘
```

The Space is **CPU-only** — DevOpsEnv is pure Python simulation (no GPU needed).

---

## 4. Files to Include / Exclude

### ✅ Include in the Space

| File / Directory | Purpose |
|------------------|---------|
| `app.py` | **NEW** — root-level Gradio app (see Section 5) |
| `env.py` | DevOpsEnv environment |
| `models.py` | Pydantic models (Action, Observation, Reward, State) |
| `multi_agent.py` | WarRoom orchestrator + 9 agents |
| `tasks/cascade.json` | 10 scenario definitions |
| `openenv.yaml` | OpenEnv manifest (discoverability) |
| `requirements.txt` | **MODIFIED** — Space-specific deps only (see Section 6) |
| `README.md` | **MODIFIED** — with correct Gradio SDK metadata (see Section 7) |
| `__init__.py` | Package init |
| `plots/` | Training plots (small PNGs — OK to include for display) |
| `eval_results/` | Evaluation results JSON |

### ❌ Do NOT Include in the Space

| File / Directory | Reason |
|------------------|--------|
| `train_grpo.py` | Training script — not needed at runtime |
| `train_sft.py` | Training script |
| `train.py` | Training script |
| `generate_sft_dataset.py` | Dataset generation — not needed at runtime |
| `submit_hf_job.py` | HF Jobs submission script |
| `inference.py` | LLM inference loop (requires OpenAI client + model) |
| `run_trained_inference.py` | Model inference helper |
| `test_inference_fix.py` | Test file |
| `plot_training_logs.py` | Plotting utility |
| `Dockerfile` | Not needed for Gradio SDK Spaces |
| `pyproject.toml` | Not needed (Space uses `requirements.txt`) |
| `server/` | FastAPI app — replaced by root `app.py` |
| `*.ipynb` | Notebooks — link to Colab/GitHub instead |
| `HF_TRAINING.md` | Docs — link from README |
| `train_details.md` | Docs |
| `training_analysis.md` | Docs |
| `pre-v.bash` | Script |
| `themes_req.txt` | Not needed |

> **Strategy:** Don't delete these files from your GitHub repo. Use a `.gitignore` or
> selective `git add` when pushing to the HF Space remote (which is a **separate git repo**).

---

## 5. Create the Root-Level `app.py` (Gradio)

Create `app.py` at the repo root. This wraps `WarRoom` in a Gradio UI.

```python
"""OpsSim-AI — Gradio Demo for Hugging Face Spaces."""

import json
import gradio as gr
from env import DevOpsEnv, EXECUTION_AGENTS
from models import Action
from multi_agent import WarRoom

# ── Global state ──────────────────────────────────────────
war_room = WarRoom(seed=42, max_steps=30)
scenarios = DevOpsEnv._DATA_CACHE or []

# Load scenarios if cache is empty (first import triggers it)
_tmp = DevOpsEnv(seed=0)
_tmp.reset()
scenarios = DevOpsEnv._DATA_CACHE or []
del _tmp

SCENARIO_IDS = [s["scenario_id"] for s in scenarios]


def format_state(state_dict: dict) -> str:
    """Pretty-print system state."""
    if not state_dict:
        return "No state available."
    lines = []
    for key, val in sorted(state_dict.items()):
        emoji = "🔴" if val in ("failing", "offline", "dead", "severed", "down") else \
                "🟡" if val in ("degraded", "overloaded", "stressed", "stalled") else "🟢"
        lines.append(f"{emoji} **{key}**: `{val}`")
    return "\n".join(lines)


def format_reward(reward_obj) -> str:
    """Format 13-component reward breakdown."""
    if reward_obj is None:
        return "No reward yet."
    d = reward_obj.model_dump() if hasattr(reward_obj, "model_dump") else vars(reward_obj)
    lines = [f"**Total reward: {d.get('value', 0):.4f}**\n"]
    for k, v in d.items():
        if k == "value":
            continue
        emoji = "✅" if v > 0 else ("⚠️" if v < 0 else "➖")
        lines.append(f"{emoji} {k}: `{v:+.4f}`")
    return "\n".join(lines)


def format_channel(channel: list) -> str:
    """Format incident channel as chat-like text."""
    if not channel:
        return "*No messages yet.*"
    lines = []
    for msg in channel[-20:]:  # last 20 messages
        sender = msg.get("from", "Unknown")
        text = msg.get("message", "")
        lines.append(f"**[{sender}]** {text}")
    return "\n\n".join(lines)


def reset_env(scenario_name: str):
    """Reset the environment to a chosen scenario."""
    global war_room
    if not scenario_name:
        return "Select a scenario first.", "", "", "", ""

    try:
        idx = SCENARIO_IDS.index(scenario_name)
    except ValueError:
        return f"Unknown scenario: {scenario_name}", "", "", "", ""

    war_room = WarRoom(seed=42, max_steps=30)
    war_room.env.scenario_index = idx
    obs, domain_obs = war_room.reset()

    state = war_room.env.get_state()
    available = obs.available_actions or []
    scenario_data = scenarios[idx]
    optimal = scenario_data.get("optimal_solution_path", [])

    state_md = format_state(state.get("state", {}))
    actions_md = "\n".join(f"- `{a}`" for a in available) if available else "No actions available."
    info_md = (
        f"**Scenario:** {scenario_name}\n\n"
        f"**Description:** {scenario_data.get('description', 'N/A')}\n\n"
        f"**Optimal path length:** {len(optimal)} steps\n\n"
        f"**Available agents:** {', '.join(EXECUTION_AGENTS)}"
    )
    channel_md = format_channel(war_room.get_incident_channel())

    return info_md, state_md, actions_md, "Environment reset. Choose an action.", channel_md


def step_env(action: str, target_agent: str):
    """Execute one step in the environment."""
    if not action:
        return "", "", "", "Please enter an action.", ""

    if not target_agent:
        target_agent = "AppOps"

    try:
        result = war_room.execute_directive(target_agent, action, supervisor_approved=True)

        obs = result["observation"]
        reward_obj = result["reward"]
        done = result["done"]
        info = result["info"]

        state = war_room.env.get_state()
        state_md = format_state(state.get("state", {}))

        available = obs.available_actions or []
        actions_md = "\n".join(f"- `{a}`" for a in available) if available else "No actions available."

        reward_md = format_reward(reward_obj)
        if done:
            progress = war_room.get_progress()
            reward_md += f"\n\n---\n**Episode finished!** Progress: {progress:.1%}"
            error = war_room.env.last_action_error
            if error:
                reward_md += f"\nLast error: {error}"

        channel_md = format_channel(war_room.get_incident_channel())

        return state_md, actions_md, reward_md, \
            f"Executed `{action}` via **{target_agent}**. Done={done}", channel_md

    except Exception as e:
        return "", "", "", f"Error: {e}", ""


# ── Gradio UI ─────────────────────────────────────────────

with gr.Blocks(
    title="OpsSim-AI: War Room Simulator",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        "# 🚨 OpsSim-AI: War Room Simulator\n"
        "Simulate cascading production incidents with a 9-agent distributed war room.\n\n"
        "**Trained model:** [meancodi/opssim-qwen-3b-v1](https://huggingface.co/meancodi/opssim-qwen-3b-v1) · "
        "**GitHub:** [nithishgouds/Meta-X-OpenEnv-X-Pytorch-Hackathon](https://github.com/nithishgouds/Meta-X-OpenEnv-X-Pytorch-Hackathon)"
    )

    with gr.Row():
        scenario_dd = gr.Dropdown(
            choices=SCENARIO_IDS, label="Scenario", value=SCENARIO_IDS[0] if SCENARIO_IDS else None
        )
        reset_btn = gr.Button("🔄 Reset Environment", variant="primary")

    with gr.Row():
        with gr.Column(scale=2):
            info_box = gr.Markdown(label="Scenario Info", value="*Select a scenario and click Reset.*")
            state_box = gr.Markdown(label="System State")
            channel_box = gr.Markdown(label="Incident Channel")
        with gr.Column(scale=1):
            actions_box = gr.Markdown(label="Available Actions")
            reward_box = gr.Markdown(label="Reward Breakdown")

    with gr.Row():
        action_input = gr.Textbox(label="Action", placeholder="e.g. check_connection_pools")
        agent_dd = gr.Dropdown(choices=EXECUTION_AGENTS, label="Target Agent", value="AppOps")
        step_btn = gr.Button("▶️ Execute Step", variant="secondary")

    status_box = gr.Markdown(label="Status")

    # Wire up events
    reset_btn.click(
        fn=reset_env,
        inputs=[scenario_dd],
        outputs=[info_box, state_box, actions_box, status_box, channel_box],
    )
    step_btn.click(
        fn=step_env,
        inputs=[action_input, agent_dd],
        outputs=[state_box, actions_box, reward_box, status_box, channel_box],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
```

> **Save this as `app.py` at the repo root** (not inside `server/`).

**Test locally before pushing:**

```bash
pip install gradio pydantic openenv-core==0.2.3 jmespath
python app.py
# Open http://localhost:7860
```

---

## 6. Create Space-Specific `requirements.txt`

The Space only runs the environment simulation — it does **not** need training dependencies.

Create a file called `requirements_space.txt` (or overwrite `requirements.txt` **only in the Space repo**):

```
pydantic
openenv-core==0.2.3
jmespath
gradio>=4.44.0
```

That's it. Four dependencies. No torch, no transformers, no CUDA.

> **Important:** When you push to the HF Space remote, the `requirements.txt` at the root is
> what HF uses to install dependencies. You have two strategies:
>
> **Option A — Overwrite in Space repo:** Copy only the Space files to a clean directory,
> use the stripped `requirements.txt`, and push that.
>
> **Option B — Keep both:** Rename the training requirements to `requirements_training.txt`
> in GitHub, and keep `requirements.txt` as the slim Space version. This is cleaner but
> requires a one-time rename in your GitHub repo.
>
> **Recommended: Option A** — push from a clean staging directory so your GitHub repo stays
> unchanged.

---

## 7. README Metadata Block

HF Spaces reads YAML frontmatter from `README.md`. Your current README already has this, but
verify it matches exactly:

```yaml
---
title: OpsSim-AI
emoji: 🚨
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: "5.0.0"
app_file: app.py
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - multi-agent
---
```

**Critical fields:**
- `sdk: gradio` — tells HF to use the Gradio runtime
- `sdk_version: "5.0.0"` — must match a real Gradio release (check [pypi.org/project/gradio](https://pypi.org/project/gradio/#history) for latest)
- `app_file: app.py` — must match your actual root file name

> If the sdk_version is too new and not yet available in HF's build system, drop it to `4.44.1`
> or remove the field entirely (HF will use the latest compatible version).

---

## 8. Create the Space on Hugging Face

### Option A — Via CLI (Recommended)

```bash
# Create the Space (replace YOUR_USERNAME with your HF username or org)
huggingface-cli repo create opssim-ai-demo --type space --space-sdk gradio

# This creates: https://huggingface.co/spaces/YOUR_USERNAME/opssim-ai-demo
```

### Option B — Via Web UI

1. Go to https://huggingface.co/new-space
2. Name: `opssim-ai-demo`
3. SDK: **Gradio**
4. Visibility: **Public**
5. Hardware: **CPU basic** (free tier is fine — no GPU needed)
6. Click "Create Space"

---

## 9. Push from Local Git

### Strategy: Push a clean subset of files (recommended)

```powershell
# 1. Create a staging directory
$STAGE = "$env:TEMP\opssim-space-stage"
New-Item -ItemType Directory -Force -Path $STAGE

# 2. Copy only the files the Space needs
$FILES = @(
    "app.py",
    "env.py",
    "models.py",
    "multi_agent.py",
    "__init__.py",
    "openenv.yaml",
    "README.md"
)
foreach ($f in $FILES) {
    Copy-Item ".\$f" "$STAGE\$f" -ErrorAction SilentlyContinue
}

# Copy directories
Copy-Item ".\tasks" "$STAGE\tasks" -Recurse
Copy-Item ".\plots" "$STAGE\plots" -Recurse
Copy-Item ".\eval_results" "$STAGE\eval_results" -Recurse

# 3. Create the Space-specific requirements.txt
@"
pydantic
openenv-core==0.2.3
jmespath
gradio>=4.44.0
"@ | Set-Content "$STAGE\requirements.txt"

# 4. Clone the Space repo and copy files in
git clone https://huggingface.co/spaces/YOUR_USERNAME/opssim-ai-demo "$STAGE\hf-space"
Copy-Item "$STAGE\app.py", "$STAGE\env.py", "$STAGE\models.py", "$STAGE\multi_agent.py", `
          "$STAGE\__init__.py", "$STAGE\openenv.yaml", "$STAGE\README.md", `
          "$STAGE\requirements.txt" -Destination "$STAGE\hf-space\"
Copy-Item "$STAGE\tasks" "$STAGE\hf-space\tasks" -Recurse
Copy-Item "$STAGE\plots" "$STAGE\hf-space\plots" -Recurse
Copy-Item "$STAGE\eval_results" "$STAGE\hf-space\eval_results" -Recurse

# 5. Commit and push
Push-Location "$STAGE\hf-space"
git add -A
git commit -m "Initial Space deployment: Gradio demo for OpsSim-AI"
git push
Pop-Location
```

### Alternative: Push directly from your repo with a Space remote

```bash
# Add HF Space as a second remote
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/opssim-ai-demo

# Push only specific files using a sparse approach:
# Create a .gitignore specifically for the space push, or use:
git push space main
```

> **Warning:** Pushing the full repo to the Space will cause build failures because
> `requirements.txt` contains `torch`, `transformers`, etc. which will either fail to install
> on a CPU Space or exceed the disk/memory limits. **Always use the staging approach or
> ensure `requirements.txt` is the slim version.**

### Bash / Linux / macOS Alternative

```bash
STAGE=$(mktemp -d)

# Copy files
for f in app.py env.py models.py multi_agent.py __init__.py openenv.yaml README.md; do
    cp "$f" "$STAGE/" 2>/dev/null
done
cp -r tasks/ plots/ eval_results/ "$STAGE/"

# Create slim requirements
cat > "$STAGE/requirements.txt" << 'EOF'
pydantic
openenv-core==0.2.3
jmespath
gradio>=4.44.0
EOF

# Clone, copy, push
git clone https://huggingface.co/spaces/YOUR_USERNAME/opssim-ai-demo "$STAGE/hf-space"
cp -r "$STAGE"/!(hf-space) "$STAGE/hf-space/"
cd "$STAGE/hf-space"
git add -A
git commit -m "Initial Space deployment: Gradio demo for OpsSim-AI"
git push
```

---

## 10. Test the Space After Deployment

After pushing, HF will build your Space. Monitor progress:

1. **Check build logs:** Go to `https://huggingface.co/spaces/YOUR_USERNAME/opssim-ai-demo` →
   click the "Logs" tab → "Build" sub-tab
2. **Build takes 2–5 minutes** on free tier (very few deps)
3. **Once running:** The Gradio UI should appear in the browser

### Manual smoke test

1. Select a scenario from the dropdown (e.g., `cascade_001_checkout_meltdown`)
2. Click **Reset Environment**
3. Verify the system state shows (red/yellow/green indicators)
4. Enter an action (e.g., `analyze_metrics`) and select an agent (e.g., `ObservabilityOps`)
5. Click **Execute Step**
6. Verify:
   - System state updates
   - Reward breakdown appears with 13 components
   - Incident channel shows messages
   - Available actions list updates

### Automated test (optional)

```python
from gradio_client import Client

client = Client("YOUR_USERNAME/opssim-ai-demo")

# Reset
result = client.predict("cascade_001_checkout_meltdown", api_name="/reset_env")
print(result)

# Step
result = client.predict("analyze_metrics", "ObservabilityOps", api_name="/step_env")
print(result)
```

---

## 11. Link Everything from README

Your README should have a clear "Resources" section linking all artifacts. Add or update:

```markdown
## Resources

| Resource | Link |
|----------|------|
| 🚀 **Live Demo** | [HF Space: opssim-ai-demo](https://huggingface.co/spaces/YOUR_USERNAME/opssim-ai-demo) |
| 🤖 **Trained Model** | [meancodi/opssim-qwen-3b-v1](https://huggingface.co/meancodi/opssim-qwen-3b-v1) |
| 💻 **GitHub** | [nithishgouds/Meta-X-OpenEnv-X-Pytorch-Hackathon](https://github.com/nithishgouds/Meta-X-OpenEnv-X-Pytorch-Hackathon) |
| 📓 **Training Notebook** | [Open in Colab](https://colab.research.google.com/github/nithishgouds/Meta-X-OpenEnv-X-Pytorch-Hackathon/blob/main/OpsSim_Training_Pipeline.ipynb) |
| 📊 **Evaluation Notebook** | [Open in Colab](https://colab.research.google.com/github/nithishgouds/Meta-X-OpenEnv-X-Pytorch-Hackathon/blob/main/OpsSim_inference_Before_vs_After.ipynb) |
| 📝 **Blog / Video** | [Coming soon] |
```

Also update the **Demo** section in your README to replace `[Coming soon]` with the actual Space URL.

---

## 12. Troubleshooting Common Failures

### Build fails with "ModuleNotFoundError: No module named 'torch'"

**Cause:** Your `requirements.txt` doesn't include torch but some imported module tries to use it.

**Fix:** The only files that import torch are training scripts. Ensure `app.py` does NOT import
`inference.py`, `run_trained_inference.py`, `train_grpo.py`, `generate_sft_dataset.py`, or any
file that imports `torch`/`transformers`. The files `env.py`, `models.py`, and `multi_agent.py`
are pure Python — they should work without torch.

Check with:
```bash
grep -r "import torch" env.py models.py multi_agent.py
# Should return nothing
```

### Build fails with "No module named 'openenv'"

**Cause:** `openenv-core` not installed, or the `env.py` fallback import isn't triggering.

**Fix:** `env.py` already has a try/except around `from openenv.core import Environment`:
```python
try:
    from openenv.core import Environment as OpenEnvEnvironment
except Exception:
    # fallback: define a stub
```

This means if `openenv-core` install fails, the env still works. But ensure
`openenv-core==0.2.3` is in your Space `requirements.txt`.

### Space shows "Running" but UI is blank

**Cause:** `app.py` isn't launching on port 7860, or the Gradio `demo.launch()` is missing.

**Fix:** Ensure `app.py` ends with:
```python
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
```

For Gradio Spaces, `demo.launch()` (without explicit port) also works — HF sets the port
automatically.

### "Error: disk quota exceeded"

**Cause:** You pushed model weights, large notebooks, or the full git history.

**Fix:** The Space repo should be < 500 MB total. Don't include:
- Model checkpoints (`.safetensors`, `.bin`)
- Large datasets (the `cascade.json` is ~100 KB, which is fine)
- Jupyter notebook outputs (strip with `jupyter nbconvert --clear-output`)

### Space crashes on startup with "killed"

**Cause:** OOM — your `app.py` is trying to load something too large for free tier (16 GB RAM).

**Fix:** The DevOpsEnv is lightweight (~10 MB RAM). If you're accidentally importing training
code that loads a model, that's the problem. Verify your `app.py` imports only `env`, `models`,
and `multi_agent`.

### `git push` to Space rejected ("authentication failed")

**Fix:**
```bash
# Re-authenticate
huggingface-cli login

# Or use token in URL
git remote set-url space https://YOUR_USERNAME:YOUR_TOKEN@huggingface.co/spaces/YOUR_USERNAME/opssim-ai-demo
```

### Space builds but shows "No app file found"

**Cause:** `app_file` in README metadata doesn't match the actual filename.

**Fix:** Ensure `app_file: app.py` in README YAML frontmatter, and the file is at the repo root
(not in `server/`).

---

## 13. Final Hackathon Submission Checklist

Use this checklist before submitting:

### Space & Deployment
- [ ] Space is **public** and accessible at the Space URL
- [ ] Space loads without errors (check Logs tab)
- [ ] Gradio UI renders with scenario dropdown, action input, agent selector
- [ ] Can reset environment and run at least one full episode (5+ steps)
- [ ] Reward breakdown shows all 13 components
- [ ] Incident channel displays agent communications
- [ ] No model weights or large binaries in the Space repo

### README
- [ ] YAML frontmatter has `sdk: gradio`, `app_file: app.py`, correct `tags`
- [ ] Problem motivation section present
- [ ] Environment explanation with agent table, scenario table, reward formula
- [ ] Training pipeline explanation (SFT → GRPO flow diagram)
- [ ] Training plots visible (reward curve, loss, quality metrics, KL, reward-vs-KL)
- [ ] Link to HF Space (live demo)
- [ ] Link to trained model (`meancodi/opssim-qwen-3b-v1`)
- [ ] Link to GitHub repo
- [ ] Link to Colab notebooks (training + evaluation)
- [ ] Link to blog/video/slides (or "[Coming soon]" placeholder)

### Training Evidence
- [ ] Training plots in `plots/` directory show clear learning signal
- [ ] `train_grpo.py` is a working, runnable training script
- [ ] `train_sft.py` is a working SFT script
- [ ] `OpsSim_Training_Pipeline.ipynb` runs end-to-end in Colab
- [ ] Evaluation notebook (`OpsSim_inference_Before_vs_After.ipynb`) shows before/after comparison

### OpenEnv Compatibility
- [ ] `openenv.yaml` manifest present at repo root
- [ ] `env.py` inherits from `OpenEnvEnvironment`
- [ ] `models.py` uses OpenEnv Action/Observation/State base classes
- [ ] `tasks/cascade.json` defines all 10 scenarios

### General
- [ ] No API keys, tokens, or secrets committed
- [ ] No large video files in the repo (link to YouTube/Loom instead)
- [ ] No model checkpoints in the repo (link to HF model hub instead)
- [ ] Authors credited in README and `pyproject.toml`

---

## Quick Reference: File Layout in the HF Space Repo

```
opssim-ai-demo/                  ← HF Space repo root
├── README.md                    ← with Gradio SDK frontmatter
├── app.py                       ← Gradio UI (new file)
├── requirements.txt             ← slim: pydantic, openenv-core, jmespath, gradio
├── env.py                       ← DevOpsEnv
├── models.py                    ← Pydantic models
├── multi_agent.py               ← WarRoom + agents
├── __init__.py                  ← package init
├── openenv.yaml                 ← OpenEnv manifest
├── tasks/
│   └── cascade.json             ← 10 scenarios
├── plots/                       ← training plot PNGs
│   ├── qwen25-3b-full/
│   └── qwen25-3b-grpo/
└── eval_results/
    └── results.json
```

Total size: **< 5 MB** (well within free tier limits).
