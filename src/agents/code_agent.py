from __future__ import annotations

from typing import Any

from src.core.agent_base import BaseAgent
from src.core.types import AgentRole,AgentResult,Task
from src.llm.client import LLMClient
from src.tools.registry import ToolRegistry

class CodeAgent(BaseAgent):
    def __init__(
            self,
            llm:LLMClient,
            tools:ToolRegistry | None = None,
    ):
        super().__init__(role=AgentRole.CODE,llm=llm,tools=tools)

    def _build_system_prompt(self) -> str:
        return (
            "You are a Code Specialist. Write, review, debug, and explain code. "
            "Produce clean, correct, and secure code.\n"
            "Guidelines:\n"
            "- Never generate code with security vulnerabilities (SQL injection, XSS, etc.)\n"
            "- Use type hints in Python\n"
            "- Explain complex logic clearly\n"
            "- Use the code_executor tool to test code before presenting it"
        )

    @property
    def expertise(self) -> list[str]:
        return ["python", "javascript", "code-review", "debugging", "algorithms", "software-design", "programming"]

    @property
    def description(self) -> str:
        return "Software engineering and programming specialist"

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
                    "content":f"[Context]:{value}"
                })
        messages.append({
            "role":"user",
            "content":task.instruction
        })

        response = await self.llm.chat(
            messages=messages,
            tools=self.tools_schemas,
            temperature=0.2,
            max_tokens=4096,
        )
        return AgentResult(
            task_id=task.id,
            agent_role=self.role,
            content=response.get("content",""),
        )

