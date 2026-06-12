"""Multi Agent Intelligent Conversation Assistant"""

import asyncio
import sys
import os
import time

# 确保项目根目录在sys.path上
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from dotenv import load_dotenv
from main import  load_config, _build_system
from src.core.types import AgentRole,Task,AgentResult


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()



@st.cache_resource(show_spinner="Initializing multi-agent system...")
def init_orchestrator():
    load_dotenv()
    config = load_config()
    return _build_system(config)

ROLE_EMOJI: dict[str, str] = {
    "research": "🔍",
    "code": "💻",
    "analysis": "📊",
    "creative": "🎨",
    "planner": "📋",
    "memory": "🧠",
    "orchestrator": "🎯",
}

ROLE_DESCRIPTIONS: dict[str, str] = {
    "research": "Web search & fact retrieval",
    "code": "Code generation & review",
    "analysis": "Math, logic & data analysis",
    "creative": "Writing & content creation",
    "planner": "Task planning & strategy",
    "memory": "Context & preference tracking",
}

st.set_page_config(
    page_title="Multi-Agent Conversation Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "orchestrator_ready" not in st.session_state:
    st.session_state.orchestrator_ready = False

# 持久化 orchestrator 引用（避免每次 rerun 都重新获取 cache）
orchestrator = init_orchestrator()

with st.sidebar:
    st.title("Multi-Agent Conversation Assistant")
    st.markdown("---")

    try:
        config = load_config()
        st.caption(f"**Provider**: {config['llm']['provider']}")
        st.caption(f"**Model**: {config['llm']['model']}")
    except:
        st.caption("Config unavailable")

    st.markdown("---")

    if st.button("Clear History", use_container_width = True):
        st.session_state.messages = []
        orchestrator._history.clear()
        st.rerun()

    st.markdown("---")

    st.markdown("#### Agent Roles")
    for role, desc in ROLE_DESCRIPTIONS.items():
        emoji = ROLE_EMOJI.get(role,"🤖")
        st.caption(f"{emoji} **{role}** — {desc}")

    st.markdown("---")
    st.caption(f"Messages: {len(st.session_state.messages)}")

st.title("Multi-Agent Conversation Assistant")
st.caption("Ask a question - the orchestrator routes your request to specialized agents ")

for i,msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("agents"):
            tags = " ".join(
                f"{ROLE_EMOJI.get(a, '🤖')} `{a}`" for a in msg["agents"]
            )
            st.caption(tags)
        if msg.get("error"):
            st.caption(f"⚠️ {msg['error']}")
        if msg.get("elapsed"):
            st.caption(f"{msg['elapsed']:.1f} seconds")

if prompt := st.chat_input("Type your message here"):
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("Thinking...")

        try:
            t0 = time.perf_counter()
            task = Task(agent_role=AgentRole.ORCHESTRATOR,instruction=prompt)
            result: AgentResult = run_async(orchestrator.execute(task))
            elapsed = time.perf_counter() - t0

            source = result.metadata.get("source_agent",[])
            if isinstance(source,str):
                agent_used = [source]
            else:
                agent_used = source if source else ["orchestrator"]

            placeholder.write(result.content)
            tags = " ".join(
                f"{ROLE_EMOJI.get(a,'🤖')}`{a}`" for a in agent_used
            )
            st.caption(tags)
            st.caption(f"{elapsed:.1f} second")

            st.session_state.messages.append(
                {
                    "role":"assistant",
                    "content":result.content,
                    "elapsed":elapsed,
                    "agents":agent_used,
                }
            )

        except Exception as e:
            error_msg = f"Error: {e}"
            placeholder.error(error_msg)
            st.session_state.messages.append(
                {
                    "role":"assistant",
                    "content":error_msg,
                    "error":str(e),
                }
            )




