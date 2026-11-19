"""A Claude Agent SDK row through the server's own path, against real PostgreSQL
and Redis: the user's saved setup token is the only credential the backend's
CLI signs in with, and without one the row is refused as a Claude Code row is.
The CLI is the real one; the loopback Messages API stands in for Anthropic.
"""

from __future__ import annotations

import importlib.util
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
    pytest.mark.skipif(importlib.util.find_spec("claude_agent_sdk") is None, reason="needs the claude-code extra"),
]
ROW = "claude-haiku-5-5-sdk"


class Title(BaseModel):
    title: str


@pytest.fixture
def cli_reaches_the_fake(monkeypatch):
    """The fake Messages API, which the backend's CLI reaches the way it reaches
    Anthropic: through the proxy and CA its environment names."""
    from tests.unit.ptc_agent.agent.claude_code.fake_api import FakeMessagesAPI

    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", "test-byok-encryption-key-32char!")
    for name in ("https_proxy", "http_proxy", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "localhost,127.0.0.1")

    async def start():
        api = FakeMessagesAPI(first_party=True)
        await api.__aenter__()
        for name, value in api.cli_env().items():
            monkeypatch.setenv(name, value)
        return api

    return start


async def test_the_saved_token_alone_signs_in_every_call(seed_user, cli_reaches_the_fake):
    from ptc_agent.config.file_utils import ConfigContext
    from ptc_agent.config.loaders import load_from_files
    from src.server.database.oauth_tokens import CLAUDE_CODE_PROVIDER, delete_oauth_tokens, upsert_oauth_tokens
    from src.server.services.llm.clients import resolve_model_client
    from src.server.services.llm.payers import model_payers
    from src.server.services.llm.service import LLMService
    from tests.unit.ptc_agent.agent.claude_code.fake_api import TOKEN

    user = seed_user["user_id"]
    api = await cli_reaches_the_fake()
    try:
        with pytest.raises(HTTPException) as refused:
            await resolve_model_client(user, ROW, is_byok=True)
        assert refused.value.status_code == 400 and refused.value.detail["reason"] == "signed_out"

        await upsert_oauth_tokens(user, CLAUDE_CODE_PROVIDER, access_token=TOKEN, refresh_token="", account_id="",
                                  expires_at=datetime.now(timezone.utc) + timedelta(days=300))
        listed = {p.name: p for p in await model_payers(user)}[ROW]
        assert (listed.payer, listed.plan, listed.signed_in) == ("subscription", "Claude", True)

        resolved = await resolve_model_client(user, ROW, is_byok=True)
        assert str(resolved.credential_source) == "oauth"
        reply = await resolved.client.ainvoke([HumanMessage("hello from the server")])
        assert reply.text == "ECHO hello from the server"

        service = LLMService(agent_config=await load_from_files(context=ConfigContext.SDK))
        title = await service.complete(user_id=user, user_prompt="title this", request_model=ROW, response_schema=Title)
        assert title == Title(title="ECHO title this")

        assert api.credentials and set(api.credentials) == {TOKEN}
    finally:
        await delete_oauth_tokens(user, CLAUDE_CODE_PROVIDER)
        await api.__aexit__(None, None, None)
