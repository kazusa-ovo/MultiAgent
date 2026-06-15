from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.types import Task, AgentResult, AgentRole
    from src.llm.client import LLMClient
    from src.tools.registry import ToolRegistry

class BaseAgent(ABC):
     """Every agent (including Orchestrator) inherits from this."""

     MAX_RETRIES = 3

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

     @property
     def description(self) -> str:
         return ""

     @property
     def expertise(self) -> list[str]:
         """Declared domain expertise — used by router."""
         return []

     @property
     def tools_schemas(self) -> list[dict]:
         if self.tools is None:
             return []
         return self.tools.list_schemas()

     @property
     def _temperature(self) -> float:
         return 0.3

     @property
     def _max_tokens(self) -> int:
         return 2048

     def _build_messages(self, task: Task) -> list[dict]:
         from src.core.types import Task
         messages: list[dict] = [
             {
                 "role": "system",
                 "content": self._system_prompt,
             }
         ]
         for key, value in task.context.items():
             if key.startswith("dep_") and key.endswith("_result"):
                 messages.append(
                     {
                         "role": "assistant",
                         "content": f"[Previous result]: {value}",
                     }
                 )
         messages.append(
             {
                 "role": "user",
                 "content": task.instruction,
             }
         )
         return messages

     async def execute(self, task: Task) -> AgentResult:
         from src.core.types import AgentResult

         messages = self._build_messages(task)

         for attempt in range(self.MAX_RETRIES):
             response = await self.llm.chat(
                 messages=messages,
                 tools=self.tools_schemas or None,
                 temperature=self._temperature,
                 max_tokens=self._max_tokens,
             )

             tool_calls = response.get("tool_calls", [])

             if not tool_calls:
                 content = response.get("content", "")
                 if not content.strip():
                     messages.append(
                         {
                             "role": "user",
                             "content": "Your response was empty. Please provide an answer."
                         }
                     )
                     continue
                 return AgentResult(
                     task_id=task.id,
                     agent_role=task.agent_role,
                     content=content,
                 )

             tool_results = await self._execute_tools(tool_calls)

             messages.append(
                 {
                     "role": "assistant",
                     "content": None,
                     "tool_calls": self._to_api_tool_calls(tool_calls),
                 }
             )
             for tr in tool_results:
                 messages.append(
                     {
                         "role": "tool",
                         "tool_call_id": tr["id"],
                         "content": tr["content"],
                     }
                 )

             if self._any_tool_failed(tool_results):
                 messages.append(
                     {
                         "role": "user",
                         "content": (
                             "One or more tools returned an error. "
                             "Try a different approach, rephrase the query, "
                             "or use an alternative tool."
                         ),
                     }
                 )

         return AgentResult(
             task_id=task.id,
             agent_role=self.role,
             content=(
                 "I was unable to complete this task after several attempts. "
                 "Please try rephrasing your request or check the configuration."
             ),
             success=False,
             error="Max retries exhausted",
         )


     async def _execute_tools(self, tool_calls: list[dict]) -> list[dict]:
         results: list[dict] = []
         for tc in tool_calls:
             tool_name = tc["name"]
             tool = self.tools.get(tool_name) if self.tools else None
             if tool is None:
                 results.append(
                     {
                         "id": tc["id"],
                         "name": tool_name,
                         "content": f"Error: tool '{tool_name}' not found",
                     }
                 )
                 continue
             try:
                 output = await tool.run(**tc["arguments"])
             except Exception as e:
                 output = f"Error: {e}"
             results.append(
                 {
                     "id": tc["id"],
                     "name": tool_name,
                     "content": output,
                 }
             )
         return results

     @staticmethod
     def _any_tool_failed(tool_results: list[dict]) -> bool:
         for tr in tool_results:
             c = tr["content"]
             if c.startswith("Error: ") or c.startswith("[Timeout "):
                 return True
         return False

     @staticmethod
     def _to_api_tool_calls(tool_calls: list[dict]) -> list[dict]:
         return [
             {
                 "id": tc["id"],
                 "type": "function",
                 "function": {
                     "name": tc["name"],
                     "arguments": json.dumps(tc["arguments"], ensure_ascii=False),
                 }
             }
             for tc in tool_calls
         ]
