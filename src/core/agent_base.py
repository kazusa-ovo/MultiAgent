from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.core.types import Task, AgentResult, AgentRole
    from src.llm.client import LLMClient
    from src.tools.registry import ToolRegistry

class BaseAgent(ABC):
     """Every agent (including Orchestrator) inherits from this."""
     def __init__(
             self,
             role: AgentRole,
             llm: LLMClient,
             tools: ToolRegistry | None = None,
     ):
         self.role = role
         self.llm = llm
         self.tools = tools
         self._system_prompt: str = self._build_system_prompt()

     @abstractmethod
     def _build_system_prompt(self) -> str: ...

     @abstractmethod
     async def execute(self, task: Task) -> AgentResult: ...

     @property
     def expertise(self) -> list[str]:
         """Declared domain expertise — used by router."""
         return []

     @property
     def description(self) -> str:
         return ""

     @property
     def tools_schemas(self) -> list[dict]:
         if self.tools is None:
             return []
         return self.tools.list_schemas()
