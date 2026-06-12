from abc import ABC, abstractmethod
from dataclasses import dataclass,field
from typing import Any

@dataclass
class BaseTool(ABC):
    name: str
    description: str
    parameters: dict[str,Any] = field(default_factory=dict)  # JSON Schema for function calling

    def __post_init__(self):
        if not self.parameters:
            self.parameters = self._default_parameters()

    @abstractmethod
    async def run(self, **kwargs:Any) -> str: ...

    def _default_parameters(self) -> dict[str,Any]:
        return {
            "type":"object",
            "properties":{},
            "required":[],
        }

    def to_schema(self) -> dict[str,Any]:
        """转为 OpenAI function-calling 兼容的 JSON Schema"""
        return {
            "type":"function",
            "function":{
                "name":self.name,
                "description":self.description,
                "parameters":self.parameters,
            },
        }