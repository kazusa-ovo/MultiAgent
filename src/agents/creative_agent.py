from __future__ import annotations

from typing import Any

from src.core.agent_base import BaseAgent
from src.core.types import AgentResult, AgentRole, Task
from src.llm.client import LLMClient
from src.tools.registry import ToolRegistry

class CreativeAgent(BaseAgent):
    def __init__(self,llm:LLMClient,tools:ToolRegistry | None=None):
        super().__init__(llm=llm,tools=tools,role=AgentRole.CREATIVE)

    def _build_system_prompt(self) -> str:
        return (
            "You are a Creative Specialist. Generate engaging content, brainstorm "
            "ideas, write stories, translate text, and produce creative work.\n"
            "Guidelines:\n"
            "- Adapt tone and style to the user's request\n"
            "- Be original and imaginative\n"
            "- Maintain coherence and clarity\n"
            "- For translations, preserve meaning and tone"
        )

    @property
    def expertise(self) -> list[str]:
        return ["writing", "storytelling", "brainstorming", "translation", "content-creation", "copywriting"]

    @property
    def description(self) -> str:
        return "Creative writing and content generation specialist"

    async def execute(self, task: Task) -> AgentResult:
        messages: list[dict[str,Any]] = [
            {
                "role":"system",
                "content":self._system_prompt,
            }
        ]
        for key,value in task.context.items():
            if key.startswith("dep_") and key.endswith("_result"):
                messages.append({
                    "role":"assistant",
                    "content":f"[Reference]: {value}"
                })
        messages.append({
            "role":"user",
            "content":task.instruction
        })

        response = await self.llm.chat(
            messages=messages,
            tools=self.tools_schemas,
            temperature=0.8,
            max_tokens=4096,
        )

        return AgentResult(
            task_id=task.id,
            agent_role=self.role,
            content=response.get("content",""),
        )
