from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import os
import httpx
from dotenv import load_dotenv

from src.tools.base import BaseTool

load_dotenv()

@dataclass
class WebSearchTool(BaseTool):
    name:str = "web_search"
    description:str = "Search the web for current information"

    def _default_parameters(self) -> dict[str,Any]:
        return {
            "type":"object",
            "properties":{
                "query":{
                    "type":"string",
                    "description":"The search query",
                },
                "num_results":{
                    "type":"integer",
                    "description":"Number of results (default 5,max 10)",
                    "default":5,
                },
            },
            "required":["query"],
        }

    async def run(self,query:str = "",num_results:int = 5,**kwargs:Any) -> str:
        api_key = os.environ.get("TAVILY_API_KEY","")
        if not api_key:
            return "Error: TAVILY_API_KEY not set in .env"

        url = "https://api.tavily.com/search"
        headers = {"Content-Type":"application/json"}
        body = {
            "api_key":api_key,
            "query":query,
            "max_results":min(num_results,10),
            "search_depth":"basic",
        }

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url,json=body,headers=headers)
            resp.raise_for_status()
            data = resp.json()

        results = data.get("results",[])
        if not results:
            return f"No results found for {query}"

        lines = [f"Search results for: {query}\n"]
        for i,r in enumerate(results,1):
            title = r.get("title","No title")
            content = r.get("content","No content")
            url = r.get("url","")
            lines.append(f"{i}. {title}")
            lines.append(f"   {content}")
            lines.append(f"   {url}\n")
        return "\n".join(lines)

