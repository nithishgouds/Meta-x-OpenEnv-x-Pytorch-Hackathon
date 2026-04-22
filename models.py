from typing import Optional, Dict, Any, List
from pydantic import BaseModel, ConfigDict, Field

try:
    import importlib.util as _ilu
    import os as _os
    _pkg = _os.path.dirname(
        _ilu.find_spec("openenv").submodule_search_locations[0]  # type: ignore
    )
    _types_path = _os.path.join(_pkg, "openenv", "core", "env_server", "types.py")
    _spec = _ilu.spec_from_file_location("_openenv_types", _types_path)
    _mod = _ilu.module_from_spec(_spec)  # type: ignore
    _spec.loader.exec_module(_mod)  # type: ignore
    OpenEnvAction = _mod.Action
    OpenEnvObservation = _mod.Observation
    OpenEnvState = _mod.State
except Exception:
    class OpenEnvAction(BaseModel):  # type: ignore[no-redef]
        model_config = ConfigDict(extra="forbid", validate_assignment=True, arbitrary_types_allowed=True)
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class OpenEnvObservation(BaseModel):  # type: ignore[no-redef]
        model_config = ConfigDict(extra="forbid", validate_assignment=True, arbitrary_types_allowed=True)
        done: bool = False
        reward: float | int | bool | None = None
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class OpenEnvState(BaseModel):  # type: ignore[no-redef]
        model_config = ConfigDict(extra="forbid", validate_assignment=True, arbitrary_types_allowed=True)
        episode_id: Optional[str] = None
        step_count: int = 0


class Reward(BaseModel):
    value: float
    delta_health: float = 0.0
    global_bleed: float = 0.0
    local_bleed: float = 0.0
    urgency_penalty: float = 0.0
    action_quality: float = 0.0
    sequencing_reward: float = 0.0
    responsibility_penalty: float = 0.0
    conflict_penalty: float = 0.0
    coordination_reward: float = 0.0
    observability_reward: float = 0.0
    supervisor_reward: float = 0.0
    communication_cost: float = 0.0
    success_reward: float = 0.0


class OpsSIMObservation(OpenEnvObservation):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, arbitrary_types_allowed=True)
    available_actions: Optional[List[str]] = None
    system_state: Optional[Dict[str, Any]] = None
    playbook_text: Optional[str] = None
    logs: Optional[str] = None
    step_count: int = 0
    agent: Optional[str] = None
    domain_state: Optional[Dict[str, Any]] = None
    incident_channel: Optional[List[Dict[str, Any]]] = None
    goal_state: Optional[Dict[str, Any]] = None
    progress: Optional[float] = None


class OpsSIMAction(OpenEnvAction):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, arbitrary_types_allowed=True)
    action_type: str = Field(..., description="The action to execute")
    agent: Optional[str] = Field(default=None, description="Agent taking the action")
    target_agent: Optional[str] = Field(default=None, description="Target agent for IC directive")
    message: Optional[str] = Field(default=None, description="Message for communicate action")
    supervisor_approved: Optional[bool] = Field(default=None, description="Supervisor approval status")
    ic_directive: Optional[bool] = Field(default=None, description="Whether this is an IC directive")


class OpsSIMState(OpenEnvState):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, arbitrary_types_allowed=True)
    state_data: Dict[str, Any] = Field(default_factory=dict)
    incident_channel: List[Dict[str, Any]] = Field(default_factory=list)


Observation = OpsSIMObservation
Action = OpsSIMAction
