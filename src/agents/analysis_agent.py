from __future__ import annotations

from typing import Any

from src.core.agent_base import BaseAgent
from src.llm.client import LLMClient
from src.core.types import AgentRole,AgentResult,Task
from src.tools.registry import ToolRegistry

class AnalysisAgent(BaseAgent):
    def __init__(self,
                 llm: LLMClient,
                 tools: ToolRegistry | None = None,
                 ):
        super().__init__(
            role=AgentRole.ANALYSIS,
            tools=tools,
            llm=llm,
        )

    def _build_system_prompt(self) -> str:
        return (
            "You are an Analysis Specialist. Perform data analysis, mathematical "
            "reasoning, logical problem-solving, and statistical evaluation.\n"
            "Guidelines:\n"
            "- Show your reasoning step by step\n"
            "- Acknowledge assumptions and limitations\n"
            "- Use the calculator tool for complex math\n"
            "- Validate your conclusions before presenting them"
        )

    @property
    def expertise(self) -> list[str]:
        return ["math", "statistics", "logic", "data-analysis", "reasoning", "problem-solving"]

    @property
    def description(self) -> str:
        return "Data analysis, mathematics, and logical reasoning specialist"

    async def execute(self, task: Task) -> AgentResult:
        messages: list[dict[str,Any]] = [
            {
                "role":"system",
                "content":self._system_prompt
            },
        ]
        for key,value in task.context.items():
            if key.startswith("dep_") and key.endswith("_result"):
                messages.append({
                    "role":"assistant",
                    "content":f"[Data]: {value}"
                })
        messages.append({
            "role":"user",
            "content":task.instruction
        })

        response = await self.llm.chat(
            messages=messages,
            temperature=0.2,
            tools=self.tools_schemas,
            max_tokens=4096
        )
        return AgentResult(
            task_id=task.id,
            agent_role=self.role,
            content=response.get("content","")
        )