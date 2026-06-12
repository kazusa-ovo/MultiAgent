from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from src.core.router import IntentRouter, AGENT_PROFILES
from src.core.types import AgentRole, Intent, Message, Task


class TestAGENT_PROFILES:
  def test_all_roles_covered(self):
      sub_roles = {r for r in AgentRole if r != AgentRole.ORCHESTRATOR}
      assert set(AGENT_PROFILES.keys()) == sub_roles

  def test_each_has_description_and_expertise(self):
      for role, profile in AGENT_PROFILES.items():
          assert "description" in profile
          assert "expertise" in profile
          assert len(profile["expertise"]) > 0


class TestExtractJSON:
  def test_plain_json(self):
      assert IntentRouter._extract_json('{"a": 1}') == '{"a": 1}'

  def test_code_block(self):
      text = '```json\n{"a": 1}\n```'
      assert IntentRouter._extract_json(text) == '{"a": 1}'

  def test_code_block_no_lang(self):
      text = '```\n{"a": 1}\n```'
      assert IntentRouter._extract_json(text) == '{"a": 1}'

  def test_whitespace(self):
      assert IntentRouter._extract_json('  {"a": 1}  ') == '{"a": 1}'


class TestParseIntent:
  def test_parses_valid_json(self, mock_llm):
      mock_llm.chat.return_value = {
          "content": '{"category": "code", "sub_categories": [], "confidence": 0.95, "reasoning": "coding"}',
      }
      router = IntentRouter(mock_llm)

      async def _run():
          return await router.parse_intent("Write a Python function")
      intent = asyncio.run(_run())

      assert intent.category == AgentRole.CODE
      assert intent.confidence == 0.95

  def test_fallback_on_bad_json(self, mock_llm):
      mock_llm.chat.return_value = {"content": "not json"}
      router = IntentRouter(mock_llm)

      async def _run():
          return await router.parse_intent("test")
      intent = asyncio.run(_run())

      assert intent.category == AgentRole.PLANNER
      assert intent.confidence == 0.3

  def test_with_history(self, mock_llm):
      mock_llm.chat.return_value = {
          "content": '{"category": "research", "sub_categories": [], "confidence": 0.8, "reasoning": ""}',
      }
      router = IntentRouter(mock_llm)
      history = [Message(role="user", content="previous")]

      async def _run():
          return await router.parse_intent("new question", history)
      intent = asyncio.run(_run())

      assert intent.category == AgentRole.RESEARCH

  def test_no_history(self, mock_llm):
      mock_llm.chat.return_value = {
          "content": '{"category": "analysis", "sub_categories": [], "confidence": 0.7, "reasoning": ""}',
      }
      router = IntentRouter(mock_llm)

      async def _run():
          return await router.parse_intent("analyze this")
      intent = asyncio.run(_run())

      assert intent.category == AgentRole.ANALYSIS


class TestRoute:
  def test_single_task(self, mock_llm):
      mock_llm.chat.return_value = {
          "content": '[{"agent_role": "code", "instruction": "Write sort", "depends_on": []}]',
      }
      router = IntentRouter(mock_llm)
      intent = Intent(category=AgentRole.CODE)

      async def _run():
          return await router.route(intent, "Write sort")
      tasks = asyncio.run(_run())

      assert len(tasks) == 1
      assert tasks[0].agent_role == AgentRole.CODE

  def test_multiple_dependent_tasks(self, mock_llm):
      mock_llm.chat.return_value = {
          "content": (
              '[{"agent_role": "research", "instruction": "search", "depends_on": []},'
              '{"agent_role": "code", "instruction": "code after", "depends_on": [0]}]'
          ),
      }
      router = IntentRouter(mock_llm)
      intent = Intent(category=AgentRole.CODE, sub_categories=[AgentRole.RESEARCH])

      async def _run():
          return await router.route(intent, "search then code")
      tasks = asyncio.run(_run())

      assert len(tasks) == 2
      assert tasks[0].depends_on == []
      assert tasks[1].depends_on == [0]  # depends_on is list[int] now

  def test_fallback_on_bad_json(self, mock_llm):
      mock_llm.chat.return_value = {"content": "garbage"}
      router = IntentRouter(mock_llm)
      intent = Intent(category=AgentRole.CODE)

      async def _run():
          return await router.route(intent, "test")
      tasks = asyncio.run(_run())

      assert len(tasks) == 1
      assert tasks[0].agent_role == AgentRole.CODE
      assert tasks[0].instruction == "test"