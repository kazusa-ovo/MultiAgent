from __future__ import annotations

from typing import Any

from src.core.agent_base import BaseAgent
from src.core.types import AgentResult, AgentRole, Task
from src.llm.client import LLMClient
from src.tools.registry import ToolRegistry

class MemoryAgent(BaseAgent):
    def __init__(
            self,
            llm:LLMClient,
            tools: ToolRegistry | None = None,
    ):
        super().__init__(
            llm=llm,
            role=AgentRole.MEMORY,
            tools=tools
        )

    def _build_system_prompt(self) -> str:
        return (
            "You are a Memory Specialist. Track conversation context, recall past "
            "interactions, and manage user preferences and knowledge.\n"
            "Guidelines:\n"
            "- Summarize key points from conversation history\n"
            "- Identify recurring themes and user preferences\n"
            "- Flag important information for long-term retention\n"
            "- Help maintain context across multi-turn conversations"
        )

    @property
    def expertise(self) -> list[str]:
        return ["memory-recall", "context-tracking", "user-preferences", "summarization"]

    @property
    def description(self) -> str:
        return "Conversation history and user preference management specialist"

    async def execute(self, task: Task) -> AgentResult:
        messages:list[dict[str,Any]] = [
            {
                "role":"system",
                "content":self._system_prompt
            }
        ]
        for key, value in task.context.items():
            if key.startswith("dep_") and key.endswith("_result"):
                messages.append({
                    "role":"assistant",
                    "content":f"[History]:{value}"
                })
        messages.append({
            "role":"user",
            "content":task.instruction
        })

        response = await self.llm.chat(
            messages=messages,
            temperature=0.3,
            tools=self.tools_schemas,
            max_tokens=2048
        )
        return AgentResult(
            task_id=task.id,
            agent_role=self.role,
            content=response.get("content","")
        )
