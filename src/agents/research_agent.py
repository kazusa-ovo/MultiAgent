from __future__ import annotations

from typing import Any

from src.core.agent_base import BaseAgent
from src.core.types import AgentRole, AgentResult, Task
from src.llm.client import LLMClient
from src.tools.registry import ToolRegistry


class ResearchAgent(BaseAgent):
    def __init__(self,
                 llm: LLMClient,
                 tools:ToolRegistry | None = None,
                 ):
        super().__init__(
            role=AgentRole.RESEARCH,
            llm=llm,
            tools=tools
        )

    def _build_system_prompt(self) -> str:
        return (
            "You are a Research Specialist. Find, verify, and present factual "
            "information. Excel at web searches, documentation lookup, "
            "fact-checking, and retrieving current information.\n"
            "Guidelines:\n"
            "- Always cite sources when available\n"
            "- Distinguish facts from opinions\n"
            "- When you cannot find information, acknowledge it honestly\n"
            "- Use the web_search tool when you need current or external data"
        )

    @property
    def expertise(self) -> list[str]:
        return ["fact-checking", "web-search", "documentation", "current-events", "news", "research"]

    @property
    def description(self) -> str:
        return "Web search and information retrieval specialist"

    async def execute(self, task: Task) -> AgentResult:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt},
        ]
        for key, value in task.context.items():
            if key.startswith("dep_") and key.endswith("_result"):
                messages.append({"role": "assistant", "content": f"[Previous result]: {value}"})
        messages.append({"role": "user", "content": task.instruction})

        response = await self.llm.chat(
            messages=messages,
            tools=self.tools_schemas,
            temperature=0.3,
            max_tokens=2048,
        )
        return AgentResult(
            task_id=task.id,
            agent_role=self.role,
            content=response.get("content", ""),
        )



