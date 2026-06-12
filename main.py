"""Multi-Agent Intelligent Conversation Assistant — CLI Entry Point."""

import asyncio
import os,sys

import yaml
from dotenv import load_dotenv

from src.core.agent_base import BaseAgent


def load_config() -> dict:
    config_path =  os.path.join(os.path.dirname(__file__), "config","settings.yaml")
    with open(config_path,"r",encoding="utf-8") as f:
        return yaml.safe_load(f)

def _build_system(config:dict):
    from src.llm.client import LLMClient
    from src.tools.registry import ToolRegistry
    from src.tools.web_search import WebSearchTool
    from src.tools.calculator import CalculatorTool
    from src.tools.code_executor import CodeExecutorTool

    from src.agents.research_agent import ResearchAgent
    from src.agents.code_agent import CodeAgent
    from src.agents.analysis_agent import AnalysisAgent
    from src.agents.creative_agent import CreativeAgent
    from src.agents.planner_agent import PlannerAgent
    from src.agents.memory_agent import MemoryAgent

    from src.core.orchestrator import OrchestratorAgent
    from src.core.types import AgentRole

    llm_cfg = config["llm"]

    llm = LLMClient(
        provider=llm_cfg["provider"],
        model=llm_cfg["model"],
        temperature=llm_cfg.get("temperature",0.7),
        max_tokens=llm_cfg.get("max_tokens",4096),
    )

    research_tools = ToolRegistry()
    research_tools.register(WebSearchTool())

    code_tools = ToolRegistry()
    code_tools.register(CodeExecutorTool())

    analysis_tools = ToolRegistry()
    analysis_tools.register(CalculatorTool())

    sub_agents: dict[AgentRole,BaseAgent] = {
        AgentRole.RESEARCH: ResearchAgent(llm,research_tools),
        AgentRole.CODE: CodeAgent(llm,code_tools),
        AgentRole.ANALYSIS: AnalysisAgent(llm,analysis_tools),
        AgentRole.CREATIVE:CreativeAgent(llm),
        AgentRole.PLANNER: PlannerAgent(llm),
        AgentRole.MEMORY: MemoryAgent(llm),
    }

    return OrchestratorAgent(llm,sub_agents)

async def main():
    load_dotenv()
    print("=" * 60)
    print("Multi-Agent Intelligent Conversation Assistant")
    print("=" * 60)

    try:
        config = load_config()
    except Exception as e:
        print(f"[Error] Failed to load config: {e}")
        sys.exit(1)

    print(f"Provider: {config['llm']['provider']}")
    print(f"Model: {config['llm']['model']}")
    print("Initializing agent...")

    orchestrator = _build_system(config)
    print("Ready! Type /exit to quit,/help for commands.\n")

    while True:
        try:
            user_input = input("You:").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/exit","/quit"):
            print("Goodbye!")
            break

        if user_input.lower() == "/help":
            print("Commands:")
            print("  /exit, /quit  - Exit")
            print("  /help         - Show this help")
            print("  /history      - Show conversation history")
            print("  /clear        - Clear conversation history")
            continue

        if user_input.lower() == "/history":
            if not orchestrator._history:
                print("(No history)")
            else:
                for m in orchestrator._history:
                    role = "You" if m.role == "user" else "Assistant"
                    preview = m.content[:200] + ("..." if len(m.content) > 200 else "")
                    print(f"[{role}]: {preview}")
            continue

        if user_input.lower() == "/clear":
            orchestrator._history.clear()
            print("History cleared!")
            continue

        print("Assistant: ",end="",flush=True)
        try:
            response = await orchestrator.chat(user_input)
            print(response)
        except Exception as e:
            print(f"[Error] {e}")

if __name__ == '__main__':
    asyncio.run(main())

