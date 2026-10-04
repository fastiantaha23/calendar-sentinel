"""HTTP contracts (domain independent) + the typed decision/state used by the agent loop."""
from typing import Any, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra='forbid')


class ExternalContext(Contract):
    source: str = Field(min_length=1, max_length=100)
    content: str = Field(max_length=10000)
    trust: Literal['untrusted'] = 'untrusted'


class Fault(Contract):
    type: Literal['none', 'tool_timeout', 'malformed_tool_output', 'invalid_agent_decision'] = 'none'
    trigger: Literal['first_matching_operation'] = 'first_matching_operation'


class ArenaConfig(Contract):
    max_steps: int = Field(default=6, ge=1, le=6)
    fault: Fault = Field(default_factory=Fault)

    @field_validator('fault', mode='before')
    @classmethod
    def accept_assignment_shorthand(cls, value):
        return {'type': value} if isinstance(value, str) else value


class ArenaRequest(Contract):
    arena_version: Literal['0.1'] = '0.1'
    request_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1, max_length=80)
    task: str = Field(min_length=1, max_length=10000)
    external_context: list[ExternalContext] = Field(default_factory=list, max_length=20)
    arena_config: ArenaConfig = Field(default_factory=ArenaConfig)

    @field_validator('task')
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError('Task must not be blank')
        return value


class ToolTrace(Contract):
    step: int = Field(ge=1, le=6)
    tool: str
    attempt: int = Field(default=1, ge=1, le=3)
    outcome: Literal['success', 'timeout', 'malformed_output', 'rejected', 'exception']
    latency_ms: float = Field(default=0, ge=0)


class Metrics(Contract):
    latency_ms: float = Field(default=0, ge=0)
    model_calls: int = Field(default=0, ge=0, le=6)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)


class ArenaResponse(Contract):
    arena_version: Literal['0.1'] = '0.1'
    request_id: str
    status: Literal['completed', 'needs_clarification', 'blocked', 'approval_required',
                    'tool_error', 'contract_error', 'budget_exceeded', 'failed']
    final_response: str = Field(min_length=1, max_length=2000)
    steps: int = Field(default=0, ge=0, le=6)
    stop_reason: str
    tool_calls: list[ToolTrace] = Field(default_factory=list)
    errors: list[dict] = Field(default_factory=list)
    events: list[dict] = Field(default_factory=list)
    metrics: Metrics = Field(default_factory=Metrics)


class ChatRequest(ArenaRequest):
    session_id: str = Field(min_length=16, max_length=80)
    model: str = Field(default='unconfigured', max_length=120)


# ---------------------------------------------------------------------------
# Agent contract (Section 8): the ONLY thing the model is allowed to return.
# ---------------------------------------------------------------------------
DecisionStatus = Literal['continue', 'needs_clarification', 'completed', 'blocked',
                         'approval_required', 'budget_exceeded', 'failed']
TERMINAL_STATUSES = ('needs_clarification', 'completed', 'blocked', 'approval_required',
                     'budget_exceeded', 'failed')


class AgentDecision(BaseModel):
    """Schema-level contract. Semantic checks (known tool, valid arguments) live in agent.validate_decision."""
    model_config = ConfigDict(extra='ignore')

    status: DecisionStatus
    action: Optional[str] = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    user_message: Optional[str] = None
    rationale: str = ''

    @model_validator(mode='after')
    def cross_field_rules(self):
        if self.status == 'continue':
            if not self.action:
                raise ValueError("status 'continue' requires an 'action'")
        else:
            if self.action:
                raise ValueError(f"status '{self.status}' must not carry an 'action'")
            if not (self.user_message and self.user_message.strip()):
                raise ValueError(f"status '{self.status}' requires a non-empty 'user_message'")
        return self


class AgentState(BaseModel):
    """Per-run execution state. Never shared between runs."""
    request_id: str
    goal: str
    model_id: str
    max_steps: int
    step: int = 0                       # model decisions made so far (repairs included)
    status: str = 'running'
    stop_reason: str = ''
    repairs_in_row: int = 0
    pending_approvals: list[str] = Field(default_factory=list)
    log: list[str] = Field(default_factory=list)          # decision/observation lines for the prompt
    created_entries: int = 0
    created_reminders: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    model_calls: int = 0
