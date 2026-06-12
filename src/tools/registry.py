from __future__ import annotations

from typing import Any
from src.tools.base import BaseTool

class ToolRegistry:
    def __init__(self):
        self._tools : dict[str,BaseTool] = {}

    def register(self,tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self,name:str) -> BaseTool | None:
        return self._tools.get(name)

    def list_schemas(self) -> list[dict[str,Any]]:      # 将这个列表传给 LLM，让 LLM 知道有哪些工具可用、如何调用
        return [t.to_schema() for t in self._tools.values()]

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())
