from __future__ import annotations

from unittest.mock import AsyncMock

import pytest


@pytest.fixture
def mock_llm():
    """返回一个 AsyncMock 封装的 LLMClient，提供基本的 chat 返回。"""
    llm = AsyncMock()
    llm.model = "test-model"
    llm.chat.return_value = {"role": "assistant", "content": "mock response"}
    llm.chat_stream.return_value = None
    return llm


@pytest.fixture
def sample_results():
    """测试用的 AgentResult 列表。"""
    from src.core.types import AgentResult, AgentRole
    return [
        AgentResult(task_id="t0", agent_role=AgentRole.RESEARCH, content="Research data"),
        AgentResult(task_id="t1", agent_role=AgentRole.CODE, content="Code solution"),
    ]