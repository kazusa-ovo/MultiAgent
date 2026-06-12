from __future__ import annotations

from typing import Any

from src.llm.client import LLMClient
from src.tools.registry import ToolRegistry
from src.core.agent_base import BaseAgent
from src.core.types import AgentRole,AgentResult,Task

class PlannerAgent(BaseAgent):
    def __init__(self,
                 llm:LLMClient,
                 tools: ToolRegistry | None = None,
                ):
        super().__init__(
            role=AgentRole.PLANNER,
            llm=llm,
            tools=tools
        )

    def _build_system_prompt(self) -> str:
        return (
            "You are a Planning Specialist. Break down complex goals into "
            "structured, actionable plans with clear steps, dependencies, "
            "and milestones.\n"
            "Guidelines:\n"
            "- Create step-by-step plans with clear milestones\n"
            "- Identify dependencies between steps\n"
            "- Estimate effort and resources needed\n"
            "- Consider risks and mitigation strategies\n"
            "- Adapt detail level to the request scope"
        )

    @property
    def expertise(self) -> list[str]:
        return ["planning", "project-management", "methodology", "strategy", "organization"]

    @property
    def description(self) -> str:
        return "Task planning and step-by-step methodology specialist"

    async def execute(self, task: Task) -> AgentResult:
        messages: list[dict[str,Any]] = [
            {
                "role":"system",
                "content":self._system_prompt
            },
        ]
        for key, value in task.context.items():
            if key.startswith("dep_") and key.endswith("_result"):
                messages.append({
                    "role":"assistant",
                    "content":f"[Context]:{value}"
                })
        messages.append({
            "role":"user",
            "content":task.instruction
        })

        response = await self.llm.chat(
            messages=messages,
            tools=self.tools_schemas,
            temperature=0.4,
            max_tokens=4096,
        )

        return AgentResult(
            task_id=task.id,
            agent_role=self.role,
            content=response.get("content",""),
        )