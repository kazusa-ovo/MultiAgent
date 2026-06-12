from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

from src.tools.base import BaseTool

@dataclass
class CalculatorTool(BaseTool):
    name:str = "calculator"
    description:str = "Evaluate mathematical expressions safely"

    def _default_parameters(self) -> dict[str,Any]:
        return {
            "type":"object",
            "properties":{
                "expression":{
                    "type": "string",
                    "description": "A mathematical expression to evaluate",
                },
            },
            "required":["expression"],
        }

    async def run(self,expression:str = "", **kwargs:Any) -> str:
        allowed = {
            name: getattr(math, name)
            for name in dir(math)
            if not name.startswith("_")
        }
        allowed.update({
            "abs":abs,"round":round,"min":min,"max":max,
            "sum":sum,"pow":pow,"int":int,"float":float,
        })

        try:
            result = eval(expression,{"__builtins__":{}},allowed)
            return str(result)
        except Exception as e:
            return f"Error: {e}"
