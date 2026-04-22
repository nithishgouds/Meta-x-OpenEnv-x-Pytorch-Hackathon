from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
from env import DevOpsEnv, EXECUTION_AGENTS, IC_NAME, SUPERVISOR_NAME
from models import Action
from multi_agent import WarRoom

app = FastAPI(title="OpsSim-AI War Room API")

war_room = WarRoom(seed=42, max_steps=15)


class DirectiveRequest(BaseModel):
    target_agent: str = "AppOps"
    action: str = "analyze_metrics"
    supervisor_approved: bool = True


class CommunicateRequest(BaseModel):
    agent: str
    message: str


@app.get("/")
def root():
    return {"status": "OpsSim-AI War Room running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset_env():
    try:
        obs, domain_obs = war_room.reset()
        return {
            "observation": obs.model_dump(),
            "domain_observations": {k: v.model_dump() for k, v in domain_obs.items()},
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/communicate")
def communicate(req: CommunicateRequest):
    try:
        war_room.observe_and_communicate(req.agent, req.message)
        return {"incident_channel": war_room.get_incident_channel()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step_env(directive: DirectiveRequest):
    try:
        result = war_room.execute_directive(directive.target_agent, directive.action, directive.supervisor_approved)
        return {
            "observation": result["observation"].model_dump(),
            "reward": result["reward"].model_dump(),
            "done": result["done"],
            "info": result["info"],
            "incident_channel": result["incident_channel"],
            "progress": war_room.get_progress(),
            "last_action_error": war_room.env.last_action_error,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def get_state():
    return {"state": war_room.env.get_state()}


@app.get("/incident_channel")
def get_channel():
    return {"incident_channel": war_room.get_incident_channel()}


@app.get("/progress")
def get_progress():
    return {
        "progress": war_room.get_progress(),
        "goal_state": war_room.get_goal_state(),
        "agents": EXECUTION_AGENTS,
    }


def main():
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
