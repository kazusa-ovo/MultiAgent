from __future__ import annotations

import asyncio
from typing import Any

from src.core.agent_base import BaseAgent
from src.core.router import IntentRouter
from src.core.types import AgentRole, AgentResult, Task,Message
from src.llm.client import LLMClient

SYNTHESIZE_PROMPT = """You are a response synthesizer. Combine the outputs from multiple specialized agents into a single coherent, helpful
  response for the user.

  Original user request: {user_message}

  Agent outputs:
  {agent_outputs}

  Synthesize these into one well-structured response. If only one agent contributed, refine and present its output clearly. If multiple agents
  contributed, integrate their findings smoothly.

  Do NOT mention the internal agent names or architecture. Respond naturally as a helpful assistant."""

class OrchestratorAgent(BaseAgent):
    """Main agent. Parses intent → routes tasks → aggregates results."""

    MAX_REPLAN_ROUNDS = 2

    def __init__(self, llm: LLMClient, sub_agents: dict[AgentRole, BaseAgent]):
        super().__init__(role=AgentRole.ORCHESTRATOR, llm=llm)
        self.router = IntentRouter(llm)
        self.sub_agents = sub_agents
        self._history: list[Message] = []

    def _build_system_prompt(self) -> str:
        return (
            "you are an intelligent assistant that coordinates "
            "specialized agents to help users."
        )

    @staticmethod
    def _evaluate(results: list[AgentResult]) -> bool:
        """Return True if all results are acceptable, False if replan needed."""
        if not results:
            return False
        for r in results:
            if not r.success:
                return False
            content = r.content.strip() if r.content else ""
            if not content:
                return False
            if content.startswith("Error:") or content.startswith("[Timeout"):
                return False
        return True

    async def execute(self, task: Task) -> AgentResult:
        """Top-level entry: task.instruction = user message."""

        user_message = task.instruction
        self._history.append(Message(role="user",content=user_message))

        analysis = await self.router.analyze(user_message)

        if analysis.direct_answer is not None:
            final = AgentResult(
                task_id="direct",
                agent_role=AgentRole.ORCHESTRATOR,
                content=analysis.direct_answer,
                metadata={"source_agent": "orchestrator", "fast_path": True},
            )
            self._history.append(Message(role="assistant",content=final.content))
            return final

        all_results: list[AgentResult] = []
        pending_tasks = analysis.tasks

        for round_idx in range(self.MAX_REPLAN_ROUNDS):
            round_results = await self._dispatch(pending_tasks)
            all_results.extend(round_results)

            if self._evaluate(round_results):
                break

            if round_idx < self.MAX_REPLAN_ROUNDS - 1:
                pending_tasks = await self.router.replan(
                    user_message=user_message,
                    all_results=all_results,
                )
                if not pending_tasks:
                    break

        final = await self._synthesize(user_message, all_results)
        self._history.append(Message(role="assistant",content=final.content))
        return final

    async def chat(self,user_message: str) -> str:
        task = Task(agent_role=AgentRole.ORCHESTRATOR,instruction=user_message)
        results = await self.execute(task)
        return results.content

    @staticmethod
    def _check_failure(content: str) -> tuple[bool, str]:
        stripped = content.strip() if content else ""
        if not stripped:
            return True, "Empty response"
        if stripped.startswith("Error:"):
            return True, "Agent returned error"
        if stripped.startswith("[Timeout"):
            return True, "Execution timeout"
        return False, ""

    async def _dispatch(self, tasks: list[Task]) -> list[AgentResult]:
        """Run tasks. Parallel for independent, sequential for dependent."""
        if not tasks:
            return []

        finished: dict[int,AgentResult] = {}
        pending = list(enumerate(tasks))

        while pending:
            ready = [
                (i, t) for i, t in pending
                if all(dep in finished and finished[dep].success for dep in t.depends_on)
            ]
            if not ready:
                for i, t in pending:
                    finished[i] = AgentResult(
                        task_id=t.id,
                        agent_role=t.agent_role,
                        content=f"Unresolved dependencies: {t.depends_on}",
                        success=False,
                        error="Unresolved dependencies",
                    )
                break

            batch = []
            for i, t in ready:
                agent = self.sub_agents.get(t.agent_role)
                if agent is None:
                    finished[i] = AgentResult(
                        task_id=t.id,
                        agent_role=t.agent_role,
                        content=f"No agent for role: {t.agent_role.value}",
                        success=False,
                        error=f"Unknown role: {t.agent_role.value}",
                    )
                else:
                    context = dict(t.context)
                    for dep_idx in t.depends_on:
                        if dep_idx in finished:
                            context[f"dep_{dep_idx}_result"] = finished[dep_idx].content
                    t.context = context
                    batch.append((i, agent.execute(t)))

            if batch:
                batch_results = await asyncio.gather(
                    *[b[1] for b in batch],
                    return_exceptions=True,
                )
                for (i, _), result in zip(batch, batch_results):
                    if isinstance(result, BaseException):
                        finished[i] = AgentResult(
                            task_id=tasks[i].id,
                            agent_role=tasks[i].agent_role,
                            content=str(result),
                            success=False,
                            error=str(result),
                        )
                    else:
                        is_fail, reason = self._check_failure(result.content)
                        if is_fail:
                            result.success = False
                            result.error = result.error or reason
                        finished[i] = result

            failed_ids = {i for i, r in finished.items() if not r.success}
            for i, t in pending:
                if i not in finished and any(d in failed_ids for d in t.depends_on):
                    failed_deps = [str(d) for d in t.depends_on if d in failed_ids]
                    finished[i] = AgentResult(
                        task_id=t.id,
                        agent_role=t.agent_role,
                        content=f"Skipped: task {', '.join(failed_deps)} failed",
                        success=False,
                        error="Dependency failed",
                    )

            pending = [(i, t) for i, t in pending if i not in finished]

        return [finished[i] for i in sorted(finished.keys())]

    async def _synthesize(self, user_message: str, results: list[AgentResult]) -> AgentResult:
        """Combine sub-agent outputs into a single coherent response."""
        if len(results) == 0:
            return AgentResult(
                task_id = "synth",
                agent_role = AgentRole.ORCHESTRATOR,
                content = "I wasn't able to process your request."
            )

        if len(results) == 1:
            r = results[0]
            return AgentResult(
                task_id = "synth",
                agent_role = AgentRole.ORCHESTRATOR,
                content = r.content,
                success = r.success,
                metadata = {"source_agent":r.agent_role.value},
            )

        agent_output = "\n\n".join(
            f"[{r.agent_role.value}]: {r.content}" for r in results
        )

        prompt = SYNTHESIZE_PROMPT.format(
            user_message = user_message,
            agent_outputs = agent_output,
        )

        messages:list[dict[str,Any]] = [{
            "role":"system",
            "content":prompt,
        }]
        response = await self.llm.chat(
            messages = messages,
            temperature=0.5,
            max_tokens=2048,
        )
        return AgentResult(
            task_id = "synth",
            agent_role = AgentRole.ORCHESTRATOR,
            content = response.get("content",""),
            metadata = {
                "source_agent":[r.agent_role.value for r in results],
                "task_count":len(results),
            }

        )