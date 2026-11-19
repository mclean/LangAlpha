"""The Claude Agent SDK chat model against the real CLI and a loopback Messages API.

What it leans on is the CLI's behavior, not ours: a session written from
langalpha's history resumes as an interrupted turn and sends that history as
written, the PreToolUse hook hands every call back unrun, and ``output_format``
ends in the CLI's checked answer. A CLI release that changes any of them
fails here.
"""

from __future__ import annotations

import pytest
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel

from ptc_agent.agent.claude_code.chat_model import ChatClaudeAgentSDK, Seat
from ptc_agent.agent.claude_code.launcher import LocalLauncher

from .fake_api import TOKEN, FakeMessagesAPI

pytestmark = [pytest.mark.enable_socket, pytest.mark.asyncio]
MODEL = "claude-sonnet-5-5"
ran: list[str] = []


@tool
def get_price(command: str) -> str:
    """Price of a ticker."""
    ran.append(command)
    return f"{command} is 123"


class Title(BaseModel):
    title: str


def _model(api: FakeMessagesAPI, **kwargs) -> ChatClaudeAgentSDK:
    token = TOKEN if api.first_party else None
    seat = Seat(identity="user-1", launcher=lambda: LocalLauncher(env={**api.cli_env(), "CLAUDE_CODE_MAX_RETRIES": "0"}, token=token))
    return ChatClaudeAgentSDK(model=MODEL, seat=seat, **kwargs)


def _conversation(request: dict) -> list[dict]:
    return [m for m in request["messages"] if m["role"] != "system"]


@pytest.fixture(autouse=True)
def _no_runs():
    ran.clear()


async def test_a_reply_is_one_request_on_langalphas_prompt():
    async with FakeMessagesAPI() as api:
        reply = await _model(api).ainvoke([SystemMessage("You are terse."), HumanMessage("hello there")])
    assert reply.text == "ECHO hello there"
    assert reply.usage_metadata["input_tokens"] > 0 and reply.response_metadata["stop_reason"] == "end_turn"
    [request] = api.requests
    assert any(b["text"] == "You are terse." for b in request["system"])
    assert request["model"] == MODEL


async def test_every_call_goes_back_to_the_caller_and_none_runs():
    async with FakeMessagesAPI() as api:
        reply = await _model(api).bind_tools([get_price]).ainvoke([HumanMessage("CALLS get_price:AAPL get_price:MSFT")])
    assert [(c["name"], c["args"]) for c in reply.tool_calls] == [("get_price", {"command": "AAPL"}), ("get_price", {"command": "MSFT"})]
    # Had the CLI run them, it would have asked Claude again with their results.
    assert len(api.requests) == 1 and ran == []


async def test_tool_results_reach_claude_after_the_calls_as_the_history_holds_them():
    async with FakeMessagesAPI() as api:
        model = _model(api).bind_tools([get_price])
        first = await model.ainvoke([HumanMessage("CALLS get_price:AAPL get_price:MSFT")])
        results = [ToolMessage(f"{c['args']['command']} is 9", tool_call_id=c["id"]) for c in first.tool_calls]
        second = await model.ainvoke([HumanMessage("CALLS get_price:AAPL get_price:MSFT"), first, *results])
    assert second.text == "The tool said: MSFT is 9"
    asked, answered = _conversation(api.requests[-1])[1:3]
    assert [b.get("signature") for b in asked["content"] if b["type"] == "thinking"] == [first.content[0]["signature"]]
    assert [b["id"] for b in asked["content"] if b["type"] == "tool_use"] == [c["id"] for c in first.tool_calls]
    assert [b["tool_use_id"] for b in answered["content"] if b["type"] == "tool_result"] == [c["id"] for c in first.tool_calls]


async def test_an_agent_loop_runs_its_own_tools_through_it():
    async with FakeMessagesAPI() as api:
        out = await create_agent(_model(api), tools=[get_price]).ainvoke({"messages": [HumanMessage("CALLS get_price:NVDA")]})
    assert ran == ["NVDA"]
    assert out["messages"][-1].text == "The tool said: NVDA is 123"
    assert len(api.requests) == 2


async def test_structured_output_is_the_clis_checked_answer():
    async with FakeMessagesAPI() as api:
        out = await _model(api).with_structured_output(Title, include_raw=True).ainvoke([HumanMessage("name this")])
    assert out["parsed"] == Title(title="ECHO name this") and out["parsing_error"] is None
    assert out["raw"].usage_metadata["output_tokens"] > 0


async def test_it_signs_in_with_the_setup_token_alone():
    async with FakeMessagesAPI(first_party=True) as api:
        reply = await _model(api).ainvoke([HumanMessage("hello")])
    assert reply.text == "ECHO hello" and set(api.credentials) == {TOKEN}


async def test_a_model_bound_to_no_user_starts_no_cli():
    with pytest.raises(RuntimeError, match="saved Claude Code setup token"):
        await ChatClaudeAgentSDK(model=MODEL).ainvoke([HumanMessage("hello")])
