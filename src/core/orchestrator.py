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

    async def execute(self, task: Task) -> AgentResult:
        """Top-level entry: task.instruction = user message."""

        user_message = task.instruction
        self._history.append(Message(role="user",content=user_message))

        intent = await self.router.parse_intent(user_message,self._history)
        sub_tasks = await self.router.route(intent,user_message)
        results = await self._dispatch(sub_tasks)
        final = await self._synthesize(task.instruction, results)

        self._history.append(Message(role="assistant",content=final.content))
        return final

    async def chat(self,user_message: str) -> str:
        task = Task(agent_role=AgentRole.ORCHESTRATOR,instruction=user_message)
        results = await self.execute(task)
        return results.content

    async def _dispatch(self, tasks: list[Task]) -> list[AgentResult]:
        """Run tasks. Parallel for independent, sequential for dependent."""
        if not tasks:
            return []

        finished: dict[int,AgentResult] = {}
        pending = list(enumerate(tasks))

        while pending:
            ready = [
                (i, t) for i, t in pending
                if all(dep in finished for dep in t.depends_on)
            ]
            if not ready:
                # 依赖图中存在无法满足的依赖（如引用了不存在的任务索引）
                for i, t in pending:
                    finished[i] = AgentResult(
                        task_id = t.id,
                        agent_role = t.agent_role,
                        content = f"Unresolved dependencies: {t.depends_on}",
                        success = False,
                        error = "Unresolved dependencies",
                    )
                break

            batch = []
            for i, t in ready:
                agent = self.sub_agents.get(t.agent_role)
                if agent is None:
                    finished[i] = AgentResult(
                        task_id = t.id,
                        agent_role = t.agent_role,
                        content = f"No agent for role: {t.agent_role.value}",
                        success = False,
                        error = f"Unknown role: {t.agent_role.value}",
                    )
                else:
                    # 将依赖任务的执行结果注入 context
                    context = dict(t.context)
                    for dep_idx in t.depends_on:
                        if dep_idx in finished:
                            context[f"dep_{dep_idx}_result"] = finished[dep_idx].content
                    t.context = context
                    batch.append((i,agent.execute(t)))

            if batch:
                batch_results = await asyncio.gather(*[b[1] for b in batch])
                for (i,_),result in zip(batch,batch_results):
                    finished[i] = result

            pending = [(i,t) for i,t in pending if i not in finished]

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