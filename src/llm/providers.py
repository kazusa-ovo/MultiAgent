from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

import httpx


class BaseProvider(ABC):
    def __init__(self,model:str,api_key:str,base_url:str = ""):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    @abstractmethod
    async def chat(self,
                   messages:list[dict[str, Any]],
                   tools:list[dict[str,Any]] | None = None,
                   temperature:float = 0.5,
                   max_tokens:int = 4096
                   ) -> dict[str,Any]:...

    @abstractmethod
    async def chat_stream(self,
                          messages:list[dict[str, Any]],
                          tools:list[dict[str,Any]] | None = None,
                          temperature:float = 0.5,
                          max_tokens:int = 4096
                          ) -> AsyncIterator[str]:...

class OpenAIProvider(BaseProvider):
    async def chat(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None = None,
            temperature: float = 0.7,
            max_tokens: int = 4096,
    ) -> dict[str, Any]:
        url = (
            self.base_url.rstrip("/")+"/chat/completions"
            if self.base_url
            else "https://api.openai.com/v1/chat/completions"
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        body: dict[str,Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            body["tools"] = tools

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(url,headers=headers,json=body)
            resp.raise_for_status()
            data = resp.json()

        choice = data["choices"][0]

        result:dict[str,Any] = {
            "role":choice["message"].get("role","assistant"),
            "content":choice["message"].get("content") or "",
        }

        tool_calls = choice["message"].get("tool_calls")
        if tool_calls:
            result["tool_calls"] = [
                {
                    "id":tc["id"],
                    "name":tc["function"]["name"],
                    "arguments":json.loads(tc["function"]["arguments"]),
                }
                for tc in tool_calls
            ]
        return result

    async def chat_stream(
            self,
            messages:list[dict[str, Any]],
            tools:list[dict[str,Any]] | None = None,
            temperature: float = 0.5,
            max_tokens: int = 4096,
    ) -> AsyncIterator[str]:

        url = (
            self.base_url.rstrip("/")+"/chat/completions"
            if self.base_url
            else "https://api.openai.com/v1/chat/completions"
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        body:dict[str,Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        if tools:
            body["tools"] = tools

        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST",url,headers=headers,json=body) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():  # 异步迭代器，逐行读取响应数据
                    if line.startswith("data: "):
                        chunk = line[6:]
                        if chunk == "[DONE]":
                            break
                        try:
                            data = json.loads(chunk)
                            delta = data["choices"][0].get("delta",{})
                            content = delta.get("content","")
                            if content:
                                yield content
                        except (json.JSONDecodeError, KeyError,IndexError):
                            continue