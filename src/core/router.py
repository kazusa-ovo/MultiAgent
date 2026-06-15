from __future__ import annotations
import json
from typing import Any

from src.core.types import AgentRole, Intent, Task, Message, AnalysisResult
from src.llm.client import LLMClient

AGENT_PROFILES:dict[AgentRole,dict[str,Any]] = {
    AgentRole.RESEARCH:{
        "description":"Web search and information retrieval specialist",
        "expertise":["fact-checking","web-search","documentation","current-events","news"],
    },
    AgentRole.CODE:{
        "description":"Software engineering and programming specialist",
        "expertise":["python", "javascript", "code-review", "debugging", "algorithms", "software-design"]
    },
    AgentRole.ANALYSIS:{
        "description":"Data analysis, mathematics, and logical reasoning specialist",
        "expertise":["math", "statistics", "logic", "data-analysis", "reasoning", "problem-solving"]
    },
    AgentRole.CREATIVE: {
      "description": "Creative writing and content generation specialist",
      "expertise": ["writing", "storytelling", "brainstorming", "translation", "content-creation"],
    },
    AgentRole.PLANNER: {
      "description": "Task planning and step-by-step methodology specialist",
      "expertise": ["planning", "project-management", "methodology", "strategy", "organization"],
    },
    AgentRole.MEMORY:{
        "description":"Conversation history and user preference management specialist",
        "expertise":["memory-recall", "context-tracking", "user-preferences", "summarization"]
    }
}

INTENT_PROMPT = """You are an intent classifier for a multi-agent system. Analyze the user message and determine which agent(s) should handle it.

  Available agents:
  {agent_list}

  Return a JSON object with these fields:
  - "category": the PRIMARY agent role (use the role name exactly as listed)
  - "sub_categories": list of secondary agent roles (empty list if none)
  - "confidence": number between 0.0 and 1.0
  - "reasoning": brief explanation of your classification

  User message: {user_message}

  Return ONLY valid JSON, no other text."""

DECOMPOSE_PROMPT = """You are a task decomposer. Break down the user's request into discrete tasks for specialized agents.

  Available agents:
  {agent_list}

  User request: {user_message}
  Primary intent: {primary_agent}
  Secondary intents: {secondary_agents}

  Create a task list as a JSON array. Each task object:
  - "agent_role": one of the available role names
  - "instruction": a clear, self-contained instruction for that agent
  - "depends_on": list of 0-based task indices this task depends on (empty if independent)

  Rules:
  1. Simple single-domain requests → exactly 1 task for the primary agent.
  2. Complex cross-domain requests → multiple tasks.
  3. Independent tasks → empty depends_on (they can run in parallel).
  4. Tasks that need prior results → list the earlier task indices in depends_on.

  Return ONLY a valid JSON array, no other text."""

ANALYZE_PROMPT = """You are a request analyzer for a multi-agent system.

  Available agents:
  {agent_list}

  Analyze the user message and decide whether it needs specialized agents.

  Return a JSON object:
  - "direct_answer": a brief, friendly answer if this is simple conversation (greetings, "who are you", chitchat). Set to null if this needs
  specialized expertise.
  - "confidence": 0.0 to 1.0
  - "tasks": array of task objects (empty when direct_answer is sufficient)
    - "agent_role": agent role name
    - "instruction": clear, self-contained instruction
    - "depends_on": list of 0-based task indices

  Rules:
  - Simple chat / self-introduction / small talk → use direct_answer, empty tasks.
  - Requests needing factual lookup, code, math, planning, or analysis → direct_answer: null, populate tasks.

  User message: {user_message}

  Return ONLY valid JSON, no other text."""


class IntentRouter:
    """Maps user input → Intent, then Intent → Agent(s)."""

    def __init__(self, llm: LLMClient):
        self.llm = llm
        self._agent_list_str = self._build_agent_list()

    @staticmethod
    def _build_agent_list() -> str:
        lines: list[str] = []
        for role,profile in AGENT_PROFILES.items():
            lines.append(
                f"-{role.value}:{profile['description']}."
                f"Expertise:{', '.join(profile['expertise'])}"
            )
        return "\n".join(lines)

    async def parse_intent(
            self,
            user_message:str,
            history:list[Message] | None = None,
    ) -> Intent:
        prompt = INTENT_PROMPT.format(
            agent_list = self._agent_list_str,
            user_message = user_message,
        )

        messages:list[dict[str,Any]] = [{
            "role":"system",
            "content":prompt,
        }]

        if history:
            for msg in history[-6:]:
                messages.append({"role":msg.role,"content":msg.content})
        messages.append({"role":"user","content":user_message})

        response = await self.llm.chat(
            messages = messages,
            temperature=0.1,
            max_tokens=512,
        )
        content = response.get("content","{}")

        try:
            data = json.loads(self._extract_json(content))
            return Intent(
                category=AgentRole(data.get("category","planner")),
                sub_categories=[
                    AgentRole(s) for s in data.get("sub_categories",[])
                ],
                confidence=float(data.get("confidence",0.5)),
                reasoning=str(data.get("reasoning","")),
            )
        except (json.JSONDecodeError,ValueError,KeyError):
            return Intent(
                category=AgentRole.PLANNER,
                confidence=0.3,
                reasoning="Failed to parse intent, defaulting to planner"
            )

    async def route(
            self,
            intent: Intent,
            user_message: str
    ) -> list[Task]:
        primary = intent.category.value
        secondary = ", ".join(s.value for s in intent.sub_categories) or "none"

        prompt = DECOMPOSE_PROMPT.format(
            agent_list = self._agent_list_str,
            user_message = user_message,
            primary_agent = primary,
            secondary_agents = secondary,
        )

        messages: list[dict[str,Any]] = [{"role":"system","content":prompt}]
        response = await self.llm.chat(
            messages = messages,
            temperature=0.1,
            max_tokens=1024,
        )
        content = response.get("content","[]")

        try:
            data = json.loads(self._extract_json(content))
            tasks:list[Task] = []
            for item in data:
                tasks.append(Task(
                    agent_role=AgentRole(item["agent_role"]),
                    instruction=item["instruction"],
                    depends_on=item.get("depends_on",[]),
                ))
            return tasks if tasks else [self._fallback_task(user_message,intent)]
        except (json.JSONDecodeError,ValueError,KeyError):
            return [self._fallback_task(user_message,intent)]

    def _fallback_task(
            self,
            user_message:str,
            intent:Intent,
    ) -> Task:
        return Task(
            agent_role=intent.category,
            instruction=user_message,
        )

    @staticmethod
    def _extract_json(text:str) -> str:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            end = next(
                (i for i,l in enumerate(lines) if l.startswith("```") and i > 0),
                len(lines),
            )
            text = "\n".join(lines[1:end])
        return text.strip()

    async def replan(
        self,
        user_message: str,
        all_results: list,
    ) -> list:
        """Generate recovery tasks based on previous failures."""
        lines: list[str] = []
        for i, r in enumerate(all_results):
            status = "OK" if r.success else f"FAILED: {r.error or 'unknown'}"
            preview = r.content[:300] + ("..." if len(r.content) > 300 else "")
            lines.append(f"[{i}] {r.agent_role.value}({status}): {preview}")

        prompt = self.REPLAN_PROMPT.format(
            agent_list=self._agent_list_str,
            user_message=user_message,
            previous_results="\n".join(lines),
        )

        messages: list[dict[str, Any]] = [{"role": "system", "content": prompt}]
        response = await self.llm.chat(
            messages=messages,
            temperature=0.2,
            max_tokens=1024,
        )
        content = response.get("content", "[]")

        try:
            data = json.loads(self._extract_json(content))
            tasks: list = []
            for item in data:
                tasks.append(Task(
                    agent_role=AgentRole(item["agent_role"]),
                    instruction=item["instruction"],
                    depends_on=item.get("depends_on", []),
                ))
            return tasks
        except (json.JSONDecodeError, ValueError, KeyError):
            return []

    REPLAN_PROMPT = """You are a task replanner for a multi-agent system.

  Available agents:
  {agent_list}

  Original user request: {user_message}

  Previous execution results (some may have failed):
  {previous_results}

  Some tasks failed or produced inadequate results. Generate NEW tasks to recover.
  Guidelines:
  - Try a DIFFERENT agent if the original one failed (e.g. if code agent failed, try planner to break it down)
  - Split complex failed tasks into smaller, simpler ones
  - If one agent's output was insufficient, assign another agent to supplement or verify
  - Set depends_on using 0-based indices of the NEW task list (not the old results)

  Return a JSON array of task objects with "agent_role", "instruction", "depends_on" fields.
  Return ONLY valid JSON, no other text. Return [] if no recovery is possible."""

    async def analyze(self, user_message: str) -> AnalysisResult:
        prompt = ANALYZE_PROMPT.format(
            agent_list=self._agent_list_str,
            user_message=user_message,
        )
        messages: list[dict[str, Any]] = [{
            "role": "system",
            "content": prompt,
        }]
        response = await self.llm.chat(
            messages=messages,
            temperature=0.1,
            max_tokens=1024,
        )
        content = response.get("content", "{}")

        try:
            data = json.loads(self._extract_json(content))
            tasks: list[Task] = []
            for item in data.get("tasks", []):
                tasks.append(Task(
                    agent_role=AgentRole(item["agent_role"]),
                    instruction=item["instruction"],
                    depends_on=item.get("depends_on", []),
                ))
            return AnalysisResult(
                direct_answer=data.get("direct_answer"),
                confidence=float(data.get("confidence", 0.5)),
                tasks=tasks,
                reasoning=str(data.get("reasoning", ""))
            )
        except (json.JSONDecodeError, ValueError, KeyError):
            return AnalysisResult(
                direct_answer=None,
                confidence=0.3,
                tasks=[self._fallback_task(user_message, Intent(
                    category=AgentRole.PLANNER,
                    confidence=0.3,
                    reasoning="Analyze failed, fallback"
                ))],
            )

