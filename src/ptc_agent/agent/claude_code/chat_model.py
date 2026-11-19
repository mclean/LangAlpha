"""Claude as a LangChain chat model, each call one request the Claude Code CLI makes.

The runtime (``runtime``) hands Claude Code a whole turn and lets the CLI run
its own loop. This model is the other way round: langalpha's loop calls it as
it calls any chat model, and the CLI signs the request in with the user's
saved setup token, so a Claude plan serves a model slot with no API key.

A call writes the conversation as the session the CLI resumes, as an
interrupted turn (``CLAUDE_CODE_RESUME_INTERRUPTED_TURN``): the CLI sends
the history as written, signed thinking included, and makes the one request
that carries it on. Sent as a query instead, tool results would never reach
Claude: on resume the CLI answers every call its session holds no result for
as cut off, then drops the results the query brings. Where the history ends
on tool results, the CLI adds its "Continue from where you left off." line
after them, as it does on any resumed turn.

The tools bound to the model are offered through an in-process MCP server
whose calls never run: the PreToolUse hook defers the first call of the
reply and denies the rest, so the CLI ends with nothing run and every call
goes back to the caller as a ``tool_call``. The reply is built from the
request's raw stream with langchain-anthropic's own converter, as ``relay``
builds a turn's, so it has the shape a ``ChatAnthropic`` call gives.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from operator import itemgetter
from typing import Any

from anthropic._models import construct_type
from anthropic.types.beta import BetaRawMessageStreamEvent
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    McpSdkServerConfig,
    RateLimitEvent,
    ResultMessage,
    StreamEvent,
)
from langchain_anthropic.chat_models import _format_messages, convert_to_anthropic_tool
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel, agenerate_from_stream
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda, RunnableMap, RunnablePassthrough
from langchain_core.utils.function_calling import convert_to_json_schema
from mcp.server import Server
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool
from pydantic import BaseModel, ConfigDict

from ptc_agent.agent.claude_code.cli import _Gated, connected, received, turn_failure
from ptc_agent.agent.claude_code.launcher import Launcher, Prepared, thread_dir
from ptc_agent.agent.claude_code.plan import PlanWatch
from ptc_agent.agent.claude_code.relay import _converter
from ptc_agent.agent.claude_code.tools_bridge import SERVER, allow_rule
from ptc_agent.agent.middleware.file_operations.multimodal import MAX_CONTENT_BYTES
from src.llms.claude_code import CHAT_SDK

#: The working directory every call's CLI runs in. It holds nothing: each
#: call's session and prompt live in a config dir of its own.
THREAD_KEY = "chat-model"
#: What a call Claude made, past the first of its reply, is answered with:
#: the CLI never runs any of them, the caller does.
NOT_RUN = "Not run here: the caller runs this reply's calls."
#: Why a call ended other than as a reply the caller can use.
_STRUCTURED_FAILED = "error_max_structured_output_retries"


@dataclass(frozen=True)
class Seat:
    """Who a call runs for: the user whose device the CLI reports
    (``identity``) and the launcher each call starts its CLI with, signed in
    with the setup token that user saved."""

    identity: str
    launcher: Callable[[], Launcher]


def session_records(
    messages: Sequence[Mapping[str, Any]], *, session_id: str, cwd: str, model: str,
) -> list[dict[str, Any]]:
    """The CLI session that ``messages`` (Anthropic message dicts, as
    ``_format_messages`` makes them) are: one record each, chained in order.

    A user record holding tool results names the assistant record whose calls
    they answer, as the CLI writes one, so it reads them as answered and not
    as calls cut off by a CLI that died.
    """
    now = datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    records: list[dict[str, Any]] = []
    parent: str | None = None
    for m in messages:
        content = _uncached(m["content"])
        record: dict[str, Any] = {
            "type": m["role"], "uuid": str(uuid.uuid4()), "parentUuid": parent, "sessionId": session_id,
            "timestamp": now, "isSidechain": False, "userType": "external", "cwd": cwd,
        }
        if m["role"] == "assistant":
            blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
            calls = any(b.get("type") == "tool_use" for b in blocks)
            record["message"] = {
                "id": f"msg_{uuid.uuid4().hex[:24]}", "type": "message", "role": "assistant", "model": model,
                "content": blocks, "stop_reason": "tool_use" if calls else "end_turn", "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }
        else:
            record["message"] = {"role": "user", "content": content}
            results = [b for b in content if b.get("type") == "tool_result"] if isinstance(content, list) else []
            if results:
                record["toolUseResult"] = results
                record["sourceToolAssistantUUID"] = parent
        records.append(record)
        parent = record["uuid"]
    return records


def _uncached(content: Any) -> Any:
    """``content`` without the cache breakpoints langalpha marks: the CLI
    places its own, and a request carrying more than four is refused."""
    if not isinstance(content, list):
        return content
    out = []
    for block in content:
        if isinstance(block, dict):
            block = {k: v for k, v in block.items() if k != "cache_control"}
            if block.get("type") == "tool_result":
                block["content"] = _uncached(block.get("content"))
        out.append(block)
    return out


def _system_text(system: str | list[dict[str, Any]] | None) -> str:
    if isinstance(system, list):
        return "\n\n".join(b.get("text", "") for b in system if isinstance(b, dict) and b.get("type") == "text")
    return system or ""


def _decision(decision: str, reason: str | None = None) -> dict[str, Any]:
    out = {"hookEventName": "PreToolUse", "permissionDecision": decision}
    return {"hookSpecificOutput": {**out, "permissionDecisionReason": reason} if reason else out}


class _HandBack:
    """The PreToolUse hook: the reply's first call to one of the bound tools
    is deferred, so the CLI ends its query with the call open, and the rest
    are denied; none runs. A tool of the CLI's own (``StructuredOutput``) is
    left to it."""

    def __init__(self, names: frozenset[str]) -> None:
        self.names = names
        self.deferred = False

    async def __call__(self, hook_input: Mapping[str, Any], tool_use_id: str | None, context: Any) -> dict[str, Any]:
        if hook_input.get("tool_name") not in self.names:
            return {}
        if not self.deferred:
            self.deferred = True
            return _decision("defer")
        return _decision("deny", NOT_RUN)


def _tool_server(tools: Sequence[Mapping[str, Any]]) -> McpSdkServerConfig:
    """The server offering ``tools`` (Anthropic tool dicts). Its calls are
    never asked for, since the hook stops each one first; one that came
    anyway is answered as not run rather than run here."""
    specs = [Tool(name=t["name"], description=t.get("description") or "", inputSchema=t["input_schema"]) for t in tools]

    async def on_list_tools(ctx: Any, params: Any) -> ListToolsResult:
        return ListToolsResult(tools=specs)

    async def on_call_tool(ctx: Any, params: Any) -> CallToolResult:
        return CallToolResult(content=[TextContent(type="text", text=NOT_RUN)], isError=True)

    server = Server(SERVER, version="1", on_list_tools=on_list_tools, on_call_tool=on_call_tool)
    return McpSdkServerConfig(type="sdk", name=SERVER, instance=server)


def _anthropic_tools(tools: Sequence[Any]) -> list[dict[str, Any]]:
    """``tools`` as the custom tools a ``ChatAnthropic`` call sends. A server
    tool (web search, code execution) runs on Anthropic's side, which the
    CLI has no way to offer, so it is refused rather than dropped."""
    out = []
    for tool in tools:
        wire = convert_to_anthropic_tool(tool)
        if "input_schema" not in wire:
            raise ValueError(f"{wire.get('name') or wire.get('type')} is a server tool, which a Claude Agent SDK model cannot offer")
        out.append({k: v for k, v in wire.items() if k != "cache_control"})
    return out


def usage_of(result: ResultMessage) -> UsageMetadata | None:
    """The tokens a call spent, as ``ChatAnthropic`` reports them, read off the
    CLI's result where no stream was converted (a structured call's)."""
    usage = result.usage or {}
    if not usage:
        return None
    read, written = usage.get("cache_read_input_tokens") or 0, usage.get("cache_creation_input_tokens") or 0
    prompt = (usage.get("input_tokens") or 0) + read + written
    output = usage.get("output_tokens") or 0
    return UsageMetadata(
        input_tokens=prompt, output_tokens=output, total_tokens=prompt + output,
        input_token_details={"cache_read": read, "cache_creation": written},
    )


class ChatClaudeAgentSDK(BaseChatModel):
    """A ``claude-agent-sdk`` row's client. Built unbound by the factory; the
    server binds the user it runs for (``seat``), since only a user's saved
    setup token signs it in."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str
    effort: str | None = None
    max_tokens: int | None = None
    seat: Seat | None = None
    #: The longest one call may take, its CLI's start included.
    timeout_s: float | None = 600.0

    @property
    def _llm_type(self) -> str:
        return CHAT_SDK

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model, "effort": self.effort}

    def bind_tools(self, tools: Sequence[Any], *, tool_choice: Any = None, **kwargs: Any) -> Runnable[LanguageModelInput, AIMessage]:
        """Bind ``tools``. ``tool_choice`` is not sent: the CLI picks no tool
        for Claude, so Claude decides whether to call one."""
        return self.bind(tools=_anthropic_tools(tools), **kwargs)

    def with_structured_output(
        self, schema: Any, *, include_raw: bool = False, **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Any]:
        """Answer in ``schema`` through the CLI's own structured output
        (``output_format``), which it checks before it ends the call."""
        json_schema = schema.model_json_schema() if _is_model(schema) else convert_to_json_schema(schema)
        bound = self.bind(output_format={"type": "json_schema", "schema": json_schema})
        parser = RunnableLambda(lambda message: _parsed(schema, message))
        if not include_raw:
            return bound | parser
        parsed = RunnablePassthrough.assign(parsed=itemgetter("raw") | parser, parsing_error=lambda _: None)
        unparsed = RunnablePassthrough.assign(parsed=lambda _: None)
        return RunnableMap(raw=bound) | parsed.with_fallbacks([unparsed], exception_key="parsing_error")

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        # langalpha calls its models async; a sync caller gets the same call
        # on an event loop of its own, which a loop already running here forbids.
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, self._agenerate(messages, stop, None, **kwargs)).result()

    async def _agenerate(self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        return await agenerate_from_stream(self._astream(messages, stop, run_manager, **kwargs))

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        output_format: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        if stop:
            raise ValueError("a Claude Agent SDK model takes no stop sequences")
        if self.seat is None:
            raise RuntimeError(
                f"{self.model} runs on a user's saved Claude Code setup token; build it for a user (resolve_model_client)"
            )
        system, formatted = _format_messages(messages, model=self.model)
        if not formatted or formatted[-1]["role"] != "user":
            raise ValueError("a Claude Agent SDK model continues a conversation that ends on the user's side")
        tools = list(tools or ())
        launcher = self.seat.launcher()
        plan = await PlanWatch.open(launcher, self.model, enforce=True)
        session_id = str(uuid.uuid4())
        entries = session_records(formatted, session_id=session_id, cwd=thread_dir(THREAD_KEY), model=self.model)
        prepared = await launcher.prepare(
            THREAD_KEY, session_id, entries, identity=self.seat.identity, system_prompt=_system_text(system),
        )
        result: ResultMessage | None = None
        errors: list[tuple[str, str]] = []
        stream = _Stream()
        client: ClaudeSDKClient | None = None
        said: AsyncIterator[Any] | None = None
        try:
            options = self._options(prepared, session_id, tools, output_format)
            client = ClaudeSDKClient(options, transport=_Gated(launcher.transport(options), plan))
            # The deadline bounds each read, never a yield: a timeout held
            # across one would cancel the caller's code instead of this read.
            deadline = asyncio.get_running_loop().time() + (self.timeout_s or float("inf"))
            async with asyncio.timeout_at(deadline):
                async with connected(client):
                    pass
            said = aiter(received(client))
            while True:
                async with asyncio.timeout_at(deadline):
                    m = await anext(said, None)
                if m is None:
                    break
                if isinstance(m, StreamEvent):
                    if m.parent_tool_use_id is None and output_format is None and not stream.done:
                        if (chunk := stream.convert(m.event)) is not None:
                            yield chunk
                        if stream.done and stream.stop_reason == "max_tokens":
                            # The CLI carries a reply cut at the limit on in a
                            # request of its own; the caller asked for one.
                            return
                elif isinstance(m, AssistantMessage) and m.error:
                    errors.append((m.error, " ".join(getattr(b, "text", "") for b in m.content).strip()))
                elif isinstance(m, RateLimitEvent):
                    plan.seen(m)
                elif isinstance(m, ResultMessage):
                    result = m
                    break
        finally:
            await asyncio.shield(_end(client, launcher, prepared, said))
        failure = turn_failure(result, errors, plan)
        if failure is not None:
            if failure.kind == "auth_failed":
                await launcher.refused()
            if failure.kind == "plan_stopped":
                await plan.settled()
            raise failure
        if output_format is not None and result is not None:
            if result.subtype == _STRUCTURED_FAILED or result.structured_output is None:
                raise ValueError(f"Claude gave no answer in the schema asked for: {result.subtype}")
            yield ChatGenerationChunk(message=AIMessageChunk(
                content=json.dumps(result.structured_output),
                usage_metadata=usage_of(result),
                response_metadata={"model_name": self.model, "structured_output": result.structured_output},
            ))

    def _options(
        self, prepared: Prepared, session_id: str, tools: Sequence[Mapping[str, Any]],
        output_format: Mapping[str, Any] | None,
    ) -> ClaudeAgentOptions:
        """Claude Code with none of its own prompt, settings, tools or
        compaction, resuming the call's session as an interrupted turn."""
        hooks = {"PreToolUse": [HookMatcher(matcher=None, hooks=[_HandBack(frozenset(t["name"] for t in tools))])]} if tools else {}
        return ClaudeAgentOptions(
            tools=[],
            system_prompt={"type": "file", "path": prepared.system_prompt_file},
            setting_sources=[],
            strict_mcp_config=True,
            extra_args={"disable-slash-commands": None},
            mcp_servers={SERVER: _tool_server(tools)} if tools else {},
            allowed_tools=[allow_rule(t["name"]) for t in tools],
            permission_mode="dontAsk",
            model=self.model,
            effort=self.effort,
            thinking={"type": "adaptive", "display": "summarized"},
            include_partial_messages=True,
            resume=session_id,
            output_format=dict(output_format) if output_format is not None else None,
            cwd=prepared.cwd,
            env={
                "CLAUDE_CONFIG_DIR": prepared.config_dir,
                "CLAUDE_CODE_RESUME_INTERRUPTED_TURN": "1",
                # langalpha compacts its own conversations.
                "DISABLE_AUTO_COMPACT": "1",
                "ENABLE_TOOL_SEARCH": "false",
                "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
                "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
                "CLAUDE_CODE_DISABLE_ADVISOR_TOOL": "1",
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                "CLAUDE_CODE_PROMPT_CACHE_TTL": "1h",
                # Claude sees the tools under the names the prompt uses.
                "CLAUDE_AGENT_SDK_MCP_NO_PREFIX": "1",
                **({"CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(self.max_tokens)} if self.max_tokens else {}),
            },
            settings=json.dumps({"includeGitInstructions": False}),
            hooks=hooks,
            max_buffer_size=2 * MAX_CONTENT_BYTES,
        )


class _Stream:
    """The first message of a call's stream, converted chunk by chunk as
    ``ClaudeStreamRelay`` converts a turn's."""

    def __init__(self) -> None:
        self._block_start: Any = None
        self._written: Any = None
        self.done = False
        self.stop_reason: str | None = None

    def convert(self, raw: dict[str, Any]) -> ChatGenerationChunk | None:
        kind = raw.get("type")
        # The converter reads the usage the message ends on, which does not
        # split the cache it wrote by its life; the usage it starts on does.
        if kind == "message_start":
            self._written = ((raw.get("message") or {}).get("usage") or {}).get("cache_creation")
        elif kind == "message_delta":
            self.stop_reason = (raw.get("delta") or {}).get("stop_reason") or self.stop_reason
            if self._written and (usage := raw.get("usage")) and "cache_creation" not in usage:
                raw = {**raw, "usage": {**usage, "cache_creation": self._written}}
        elif kind == "message_stop":
            self.done = True
        event = construct_type(type_=BetaRawMessageStreamEvent, value=raw)
        message, self._block_start = _converter()._make_message_chunk_from_anthropic_event(
            event, stream_usage=True, coerce_content_to_string=False, block_start_event=self._block_start,
        )
        return ChatGenerationChunk(message=message) if message is not None else None


async def _end(client: ClaudeSDKClient | None, launcher: Launcher, prepared: Prepared, said: Any) -> None:
    """Shielded: a second cancellation would otherwise leave the CLI running
    and its config dir behind."""
    with contextlib.suppress(Exception):
        if said is not None:
            await said.aclose()
    with contextlib.suppress(Exception):
        if client is not None:
            await client.disconnect()
    await launcher.release(prepared)


def _is_model(schema: Any) -> bool:
    return isinstance(schema, type) and issubclass(schema, BaseModel)


def _parsed(schema: Any, message: AIMessage) -> Any:
    value = message.response_metadata.get("structured_output")
    if value is None:
        value = json.loads(message.text)
    return schema.model_validate(value) if _is_model(schema) else value
