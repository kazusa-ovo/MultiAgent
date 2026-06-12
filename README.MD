# Multi-Agent Intelligent Conversation Assistant

A hierarchical multi-agent system that dispatches user requests to specialized sub-agents. One Orchestrator parses intent, decomposes tasks, routes them to domain-specific agents, and synthesizes the final response.

## Architecture

```
User Input
    │
    ▼
┌──────────────────────────────────────────────┐
│              Orchestrator Agent               │
│  ┌──────────┐  ┌──────────┐  ┌────────────┐  │
│  │ Intent   │  │  Task    │  │  Response  │  │
│  │ Parser   │─▶│ Decomp.  │─▶│ Synthesis  │  │
│  └──────────┘  └──────────┘  └────────────┘  │
│                      │                        │
│              ┌───────┴───────┐               │
│              │ Agent Router  │               │
│              └───────┬───────┘               │
└──────────────────────┼──────────────────────┘
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
   ┌──────────┐ ┌──────────┐ ┌──────────┐
   │ Research │ │   Code   │ │ Analysis │  ...
   │  Agent   │ │  Agent   │ │  Agent   │
   └──────────┘ └──────────┘ └──────────┘
```

**Flow:** User message → Intent parsing → Task decomposition → Agent dispatch (parallel for independent, sequential for dependent) → Response synthesis.

## Sub-Agents

| Agent | Role | Tools | Expertise |
|-------|------|-------|-----------|
| ResearchAgent | `research` | web_search | Fact retrieval, documentation, current events |
| CodeAgent | `code` | code_executor | Code generation, review, debugging |
| AnalysisAgent | `analysis` | calculator | Math, statistics, logical reasoning |
| CreativeAgent | `creative` | — | Writing, brainstorming, translation |
| PlannerAgent | `planner` | — | Task planning, step-by-step methodology |
| MemoryAgent | `memory` | — | Context tracking, user preferences |

## Project Structure

```
multiAgent/
├── app/
│   └── streamlit_app.py      # Streamlit web UI
├── src/
│   ├── core/
│   │   ├── types.py          # Pydantic data models
│   │   ├── agent_base.py     # Abstract BaseAgent class
│   │   ├── router.py         # IntentRouter — intent classification & task decomposition
│   │   └── orchestrator.py   # OrchestratorAgent — dispatch & synthesis
│   ├── agents/
│   │   ├── research_agent.py
│   │   ├── code_agent.py
│   │   ├── analysis_agent.py
│   │   ├── creative_agent.py
│   │   ├── planner_agent.py
│   │   └── memory_agent.py
│   ├── llm/
│   │   ├── client.py         # LLMClient — unified async interface
│   │   └── providers.py      # OpenAI-compatible provider (httpx)
│   ├── tools/
│   │   ├── base.py           # BaseTool abstract class
│   │   ├── registry.py       # ToolRegistry
│   │   ├── web_search.py     # Tavily search API
│   │   ├── code_executor.py  # Sandboxed Python subprocess
│   │   └── calculator.py     # Safe math expression evaluator
│   └── memory/
│       ├── conversation.py   # Sliding-window conversation buffer
│       └── user_profile.py   # Persistent user preferences (JSON)
├── config/
│   ├── settings.yaml         # LLM provider, model, system settings
│   └── agents.yaml           # Agent role definitions
├── tests/
├── main.py                   # CLI entry point
└── requirements.txt
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Create `.env` in the project root:

```env
OPENAI_API_KEY=sk-xxxxxxxx
OPENAI_BASE_URL=https://api.openai.com/v1   # optional

TAVILY_API_KEY=tvly-xxxxxxxx   # for web search
```

For DeepSeek:

```env
DEEPSEEK_API_KEY=sk-xxxxxxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
```

Then set `provider: deepseek` in `config/settings.yaml`.

### 3. Configuration

**`config/settings.yaml`** — LLM and system settings:

```yaml
llm:
  provider: openai        # openai | deepseek
  model: gpt-4o
  temperature: 0.5
  max_tokens: 4096

system:
  max_conversation_history: 20
  user_profile_path: "data/user_profile.json"
  log_level: INFO
```

**`config/agents.yaml`** — Agent definitions (role, expertise, tools).

## Usage

### CLI

```bash
python main.py
```

```
Multi-Agent Intelligent Conversation Assistant
Provider: openai
Model: gpt-4o
Ready! Type /exit to quit, /help for commands.

You: 帮我搜索最新的GPT-4新闻，然后总结重点
Assistant: [Orchestrator → ResearchAgent → AnalysisAgent]
...
```

CLI commands:

| Command | Action |
|---------|--------|
| `/exit`, `/quit` | Exit |
| `/help` | Show help |
| `/history` | Show conversation history |
| `/clear` | Clear history |

### Streamlit Web UI

```bash
streamlit run app/streamlit_app.py
```

Features:
- Chat-style interface with message history
- Displays which agents handled each request
- Sidebar with agent capability overview
- Clear history button
- Response time tracking

## How It Works

1. User sends a message (CLI or Web UI)
2. **IntentRouter** classifies the intent using the LLM, returning an `Intent` (primary + secondary agent roles)
3. **IntentRouter** decomposes the request into a `list[Task]`, each assigned to a specific agent with optional `depends_on` indices
4. **Orchestrator** dispatches tasks via topological order: independent tasks run in parallel (`asyncio.gather`), dependent tasks run sequentially with context injection
5. **Orchestrator** synthesizes all agent outputs into a single coherent response

## Tool System

Tools use OpenAI function-calling JSON Schema format. Each tool:

- Declares `name`, `description`, and JSON Schema `parameters`
- Has an `async run(**kwargs) -> str` method
- Registered via `ToolRegistry` and passed to the LLM as function schemas

Current tools:

| Tool | Description |
|------|-------------|
| `web_search` | Search the web via Tavily API |
| `code_executor` | Execute Python in a sandboxed subprocess |
| `calculator` | Safely evaluate math expressions |

## Design Principles

- **Black-box agents**: Each agent exposes only `execute(task) -> AgentResult`
- **LLM-driven routing**: Intent classification and task decomposition use the LLM itself
- **Fallback safety**: Routing failures default to PlannerAgent; synthesis handles 0/1/N results gracefully
- **No framework lock-in**: Pydantic models, httpx, and standard asyncio — no agent framework dependencies
