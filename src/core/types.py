from __future__ import annotations
import uuid
from pydantic import BaseModel, Field
from enum import Enum
from typing import Any

class AgentRole(str, Enum):
    ORCHESTRATOR = "orchestrator"
    RESEARCH = "research"
    CODE = "code"
    ANALYSIS = "analysis"
    CREATIVE = "creative"
    PLANNER = "planner"
    MEMORY = "memory"

class Intent(BaseModel):
    """Parsed user intent."""
    category: AgentRole              # primary intent category
    sub_categories: list[AgentRole] = Field(default_factory=list)  # secondary intents,为什么不使用default，因为default只适用于不可变类型
    confidence: float = 0.0
    reasoning: str = ""

class Task(BaseModel):
    """A unit of work dispatched to a sub-agent."""
    id: str = Field(default_factory=lambda:uuid.uuid4().hex[:8])
    agent_role: AgentRole
    instruction: str                 # what the sub-agent should do
    context: dict[str, Any] = Field(default_factory=dict)     # relevant context
    depends_on: list[int] = Field(default_factory=list)

class AgentResult(BaseModel):
    """Result returned by a sub-agent."""
    task_id: str
    agent_role: AgentRole
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    error: str | None = None

class Message(BaseModel):
    role: str  # "user" | "assistant" | "system"
    content: str
