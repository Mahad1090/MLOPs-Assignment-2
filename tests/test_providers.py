"""OpenAI / Anthropic / Gemini clients over real HTTP (local fake server) and provider selection."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import MagicMock, patch

import pytest

import llm
import loop as loop_module
from llm import AnthropicClient, LLMAPIError, OpenAIClient, http_json
from loop import AgentLoop, LLMError

REPO = "https://github.com/foo/bar"


# --------------------------------------------------------------------- fake HTTP server
class FakeServer:
    """Scripted JSON server: each request pops the next (status, body, headers) response."""

    def __init__(self):
        self.responses = []
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _handle(self):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length).decode() if length else ""
                outer.requests.append({
                    "method": self.command,
                    "path": self.path,
                    "headers": {k.lower(): v for k, v in self.headers.items()},
                    "body": json.loads(raw) if raw else None,
                })
                status, body, headers = outer.responses.pop(0) if outer.responses else (500, {"error": "no script"}, {})
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                for key, value in headers.items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(data)

            do_GET = _handle
            do_POST = _handle

            def log_message(self, *args):  # silence
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def queue(self, status, body, headers=None):
        self.responses.append((status, body, headers or {}))

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    srv = FakeServer()
    yield srv
    srv.close()


TOOLS = [{
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    },
}]
CONVERSATION = [
    {"role": "system", "content": "You audit."},
    {"role": "user", "content": "Go."},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\": \"a.py\"}"}}]},
    {"role": "tool", "tool_call_id": "c1", "name": "read_file", "content": "0001: x = 1"},
]


# ------------------------------------------------------------------------- error mapping
def test_http_errors_are_mapped_with_retry_hints(server):
    server.queue(429, {"error": {"message": "Slow down", "type": "rate_limit_error"}}, {"Retry-After": "7"})
    with pytest.raises(LLMAPIError) as exc:
        http_json("POST", server.url + "/x", {}, {"a": 1}, 5, "Test")
    assert exc.value.status_code == 429
    assert "rate_limit_error" in str(exc.value) and "retry-after=7s" in str(exc.value)

    server.queue(200, b"<html>not json</html>")
    with pytest.raises(LLMAPIError, match="invalid JSON") as exc2:
        http_json("GET", server.url + "/x", {}, None, 5, "Test")
    assert exc2.value.status_code is None  # retryable

    with pytest.raises(LLMAPIError, match="connection error") as exc3:
        http_json("GET", "http://127.0.0.1:1/never", {}, None, 2, "Test")
    assert exc3.value.status_code is None


def test_error_text_never_contains_api_key(server):
    server.queue(401, {"error": {"message": "Incorrect API key provided", "type": "invalid_request_error"}})
    client = OpenAIClient("sk-secret-key-value-1234567890", base_url=server.url)
    with pytest.raises(LLMAPIError) as exc:
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}])
    assert "sk-secret-key-value" not in str(exc.value)
    assert exc.value.status_code == 401


# ---------------------------------------------------------------------------- OpenAI
def test_openai_roundtrip_over_http(server):
    server.queue(200, {"choices": [{"finish_reason": "tool_calls", "message": {
        "content": None,
        "tool_calls": [{"id": "call_9", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\": \"b.py\"}"}}],
    }}]})
    client = OpenAIClient("sk-test", base_url=server.url + "/v1")
    resp = client.chat.completions.create(
        model="gpt-4.1-mini", messages=CONVERSATION, tools=TOOLS,
        tool_choice={"type": "function", "function": {"name": "read_file"}}, temperature=0.1,
    )
    sent = server.requests[0]
    assert sent["path"] == "/v1/chat/completions"
    assert sent["headers"]["authorization"] == "Bearer sk-test"
    body = sent["body"]
    assert body["tools"] == TOOLS and body["temperature"] == 0.1
    assert body["tool_choice"] == {"type": "function", "function": {"name": "read_file"}}
    tool_msg = [m for m in body["messages"] if m["role"] == "tool"][0]
    assert "name" not in tool_msg and tool_msg["tool_call_id"] == "c1"
    msg = resp.choices[0].message
    assert msg.content == "" and msg.tool_calls[0].id == "call_9"
    assert json.loads(msg.tool_calls[0].function.arguments) == {"path": "b.py"}


def test_openai_drops_unsupported_temperature_and_remembers(server):
    server.queue(400, {"error": {"message": "Unsupported value: 'temperature' does not support 0.1 with this model.",
                                 "type": "invalid_request_error", "code": "unsupported_value"}})
    server.queue(200, {"choices": [{"message": {"content": "hi"}}]})
    server.queue(200, {"choices": [{"message": {"content": "again"}}]})
    client = OpenAIClient("sk-test", base_url=server.url)
    msgs = [{"role": "user", "content": "x"}]
    assert client.chat.completions.create(model="gpt-5", messages=msgs).choices[0].message.content == "hi"
    assert "temperature" in server.requests[0]["body"] and "temperature" not in server.requests[1]["body"]
    client.chat.completions.create(model="gpt-5", messages=msgs)
    assert "temperature" not in server.requests[2]["body"]  # remembered for this model


def test_openai_models_and_empty_choices(server):
    server.queue(200, {"data": [{"id": "gpt-4.1-mini"}, {"id": "text-embedding-3-small"}]})
    client = OpenAIClient("sk-test", base_url=server.url)
    assert [m.id for m in client.models.list().data] == ["gpt-4.1-mini", "text-embedding-3-small"]
    server.queue(200, {"choices": []})
    with pytest.raises(LLMAPIError, match="no choices"):
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}])


# ------------------------------------------------------------------------- Anthropic
def test_anthropic_message_translation():
    client = AnthropicClient("sk-ant-x")
    messages = CONVERSATION + [{"role": "user", "content": "Budget notice."}]
    out = client._to_anthropic(messages)
    assert out["system"] == "You audit."
    turns = out["messages"]
    assert [t["role"] for t in turns] == ["user", "assistant", "user"]
    assistant = turns[1]["content"]
    assert assistant == [{"type": "tool_use", "id": "c1", "name": "read_file", "input": {"path": "a.py"}}]
    last = turns[2]["content"]
    assert last[0]["type"] == "tool_result" and last[0]["tool_use_id"] == "c1"  # tool_result first
    assert last[1] == {"type": "text", "text": "Budget notice."}


def test_anthropic_repairs_missing_tool_results_and_leading_assistant():
    client = AnthropicClient("sk-ant-x")
    out = client._to_anthropic([
        {"role": "assistant", "content": "hello", "tool_calls": [
            {"id": "t1", "type": "function", "function": {"name": "git_log", "arguments": "{}"}},
            {"id": "t2", "type": "function", "function": {"name": "list_files", "arguments": "not json"}}]},
        {"role": "tool", "tool_call_id": "t1", "name": "git_log", "content": ""},
    ])
    turns = out["messages"]
    assert turns[0]["role"] == "user"  # conversation must start with a user turn
    results = {b["tool_use_id"]: b["content"] for b in turns[2]["content"] if b["type"] == "tool_result"}
    assert results == {"t1": "(empty output)", "t2": "(no result)"}
    assert turns[1]["content"][-1]["input"] == {}  # unparsable arguments become {}


def test_anthropic_roundtrip_over_http(server):
    server.queue(200, {"type": "message", "stop_reason": "tool_use", "content": [
        {"type": "text", "text": "Checking."},
        {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "c.py"}}]})
    client = AnthropicClient("sk-ant-test", base_url=server.url)
    resp = client.chat.completions.create(
        model="claude-sonnet-5-5", messages=CONVERSATION, tools=TOOLS,
        tool_choice={"type": "function", "function": {"name": "read_file"}},
        response_format={"type": "json_object"},
    )
    sent = server.requests[0]
    assert sent["path"] == "/v1/messages"
    assert sent["headers"]["x-api-key"] == "sk-ant-test"
    assert sent["headers"]["anthropic-version"] == "2023-06-01"
    body = sent["body"]
    assert body["max_tokens"] == 8192 and body["model"] == "claude-sonnet-5-5"
    assert body["tool_choice"] == {"type": "tool", "name": "read_file"}
    assert body["tools"][0]["input_schema"]["properties"]["path"] == {"type": "string"}
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "valid JSON object" in body["system"][0]["text"]
    msg = resp.choices[0].message
    assert msg.content == "Checking."
    assert msg.tool_calls[0].id == "toolu_1" and json.loads(msg.tool_calls[0].function.arguments) == {"path": "c.py"}


def test_anthropic_adjusts_to_model_limits(server):
    server.queue(400, {"type": "error", "error": {"type": "invalid_request_error", "message": "`temperature` is not supported for this model"}})
    server.queue(400, {"type": "error", "error": {"type": "invalid_request_error", "message": "max_tokens: 8192 > 4096, the maximum allowed"}})
    server.queue(200, {"content": [{"type": "text", "text": "ok"}]})
    client = AnthropicClient("sk-ant-test", base_url=server.url)
    resp = client.chat.completions.create(model="claude-haiku-x", messages=[{"role": "user", "content": "x"}])
    assert resp.choices[0].message.content == "ok"
    final = server.requests[-1]["body"]
    assert "temperature" not in final and final["max_tokens"] == 4096


def test_anthropic_models_pagination(server):
    server.queue(200, {"data": [{"id": "claude-a"}], "has_more": True, "last_id": "claude-a"})
    server.queue(200, {"data": [{"id": "claude-b"}], "has_more": False})
    client = AnthropicClient("sk-ant-test", base_url=server.url)
    assert [m.id for m in client.models.list().data] == ["claude-a", "claude-b"]
    assert "after_id=claude-a" in server.requests[1]["path"]


def test_anthropic_error_payload_raises():
    client = AnthropicClient("sk-ant-test")
    with pytest.raises(LLMAPIError, match="Anthropic error"):
        client._to_response({"type": "error", "error": {"type": "overloaded_error", "message": "busy"}})


# ----------------------------------------------------------------- provider selection
@pytest.fixture
def clean_env(monkeypatch):
    for name in ("LLM_PROVIDER", "GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                 "GROQ_API_KEY", "OPENAI_BASE_URL", "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


@pytest.mark.parametrize("key,expected", [
    ("sk-ant-api03-abc", "anthropic"),
    ("sk-proj-abc123", "openai"),
    ("sk-abc123", "openai"),
    ("AIzaSyABC", "gemini"),
    ("gsk_abc", "groq"),
])
def test_provider_detected_from_key_prefix(clean_env, key, expected):
    assert loop_module.detect_provider_from_key(key) == expected
    assert loop_module.resolve_provider(api_key=key) == expected
    assert loop_module.resolve_provider("gemini", api_key=key) == "gemini"  # explicit flag wins


def test_unknown_key_prefix_falls_back_to_configured_or_default(clean_env):
    assert loop_module.detect_provider_from_key("mystery") is None
    assert loop_module.resolve_provider(api_key="mystery") == "gemini"
    clean_env.setenv("LLM_PROVIDER", "openai")
    assert loop_module.resolve_provider(api_key="mystery") == "openai"


def test_provider_detected_from_environment_keys(clean_env):
    clean_env.setenv("OPENAI_API_KEY", "sk-x")
    assert loop_module.resolve_provider() == "openai"
    clean_env.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    assert loop_module.resolve_provider() == "anthropic"  # order: gemini, anthropic, openai, groq
    clean_env.setenv("GOOGLE_API_KEY", "g-x")  # alias of GEMINI_API_KEY
    assert loop_module.resolve_provider() == "gemini"
    assert loop_module.get_api_key("gemini") == "g-x"


def test_configured_provider_without_key_uses_provider_that_has_one(clean_env):
    clean_env.setenv("LLM_PROVIDER", "openai")
    clean_env.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    assert loop_module.resolve_provider() == "anthropic"
    clean_env.setenv("OPENAI_API_KEY", "sk-x")
    assert loop_module.resolve_provider() == "openai"


def test_blank_env_values_count_as_missing(clean_env):
    clean_env.setenv("GEMINI_API_KEY", "   ")
    clean_env.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    assert loop_module.resolve_provider() == "anthropic"
    assert loop_module.get_api_key("gemini") is None


def test_plan_models_matches_dated_aliases_and_trusts_requested():
    available = ["claude-sonnet-4-5-20250929", "claude-sonnet-4-5-20250101", "claude-3-haiku-20240307"]
    pref = loop_module.PROVIDERS["anthropic"]["preference"]
    model, fallbacks = loop_module.plan_models("claude-sonnet-4-5", available, pref, (), True)
    assert model == "claude-sonnet-4-5-20250929"  # alias resolved to the newest dated id
    # an unlisted requested model (gateway name / brand-new model) is still tried first when trusted
    model, fallbacks = loop_module.plan_models("my-gateway-model", available, pref, (), True)
    assert model == "my-gateway-model" and "claude-sonnet-4-5-20250929" in fallbacks
    model, _ = loop_module.plan_models("my-gateway-model", available, pref, (), False)
    assert model != "my-gateway-model"


def test_plan_models_excludes_non_chat_openai_models():
    available = ["text-embedding-3-small", "whisper-1", "dall-e-3", "gpt-5-codex", "o1-pro", "ft:gpt-4o:me",
                 "omni-moderation-latest", "gpt-4o-realtime-preview", "gpt-4.1-mini", "gpt-4o"]
    cfg = loop_module.PROVIDERS["openai"]
    model, fallbacks = loop_module.plan_models("gpt-4.1-mini", available, cfg["preference"], cfg["exclude_markers"], True)
    assert model == "gpt-4.1-mini" and fallbacks == ["gpt-4o"]


# --------------------------------------------------------------- AgentLoop integration
@pytest.mark.parametrize("provider,cls_name,env,url_var", [
    ("openai", "OpenAIClient", "OPENAI_API_KEY", "OPENAI_BASE_URL"),
    ("anthropic", "AnthropicClient", "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"),
    ("gemini", "GeminiClient", "GEMINI_API_KEY", None),
])
def test_agent_loop_builds_matching_client(clean_env, tmp_path, provider, cls_name, env, url_var):
    fake = MagicMock()
    fake.models.list.side_effect = Exception("offline")
    if url_var:
        clean_env.setenv(url_var, "https://gateway.example/v1")
    with patch.object(loop_module, cls_name, return_value=fake) as ctor:
        agent = AgentLoop(REPO, str(tmp_path), api_key="k-123", provider=provider, verbose=False)
    assert agent.client is fake and agent.provider == provider
    kwargs = ctor.call_args.kwargs
    assert kwargs["api_key"] == "k-123"
    if url_var:
        assert kwargs["base_url"] == "https://gateway.example/v1"
    clean_env.delenv(env, raising=False)


def test_missing_key_messages_name_the_right_variables(clean_env):
    with pytest.raises(ValueError, match="Missing ANTHROPIC_API_KEY"):
        AgentLoop(REPO, ".", api_key=None, provider="anthropic")
    with pytest.raises(ValueError, match="Missing OPENAI_API_KEY"):
        AgentLoop(REPO, ".", api_key=None, provider="openai")
    with pytest.raises(ValueError, match="GEMINI_API_KEY or GOOGLE_API_KEY"):
        AgentLoop(REPO, ".", api_key=None, provider="gemini")


def _agent_with_script(tmp_path, script):
    fake = MagicMock()
    fake.models.list.side_effect = Exception("offline")
    fake.chat.completions.create.side_effect = script
    with patch.object(loop_module, "OpenAIClient", return_value=fake):
        agent = AgentLoop(REPO, str(tmp_path), api_key="sk-x", provider="openai", verbose=False)
    agent.coverage_nudged = True
    return agent, fake


def test_billing_problem_is_fatal_without_retries(tmp_path, monkeypatch):
    monkeypatch.setattr(loop_module.time, "sleep", lambda *_: None)
    err = LLMAPIError("OpenAI HTTP 429: You exceeded your current quota [type=insufficient_quota]", 429)
    agent, fake = _agent_with_script(tmp_path, [err] * 8)
    with pytest.raises(LLMError, match="quota/billing"):
        agent._chat(messages=agent.messages)
    assert fake.chat.completions.create.call_count == 1


def test_context_overflow_shrinks_history_and_retries(tmp_path, monkeypatch):
    monkeypatch.setattr(loop_module.time, "sleep", lambda *_: None)
    ok = MagicMock()
    ok.choices = [MagicMock()]
    err = LLMAPIError("Anthropic HTTP 400: prompt is too long: 250000 tokens > 200000 maximum", 400)
    agent, fake = _agent_with_script(tmp_path, [err, ok])
    agent.messages.append({"role": "tool", "tool_call_id": "1", "name": "x", "content": "A" * 5000})
    assert agent._chat(messages=agent.messages) is ok
    assert len(agent.messages[-1]["content"]) < 600  # emergency-trimmed before the retry


def test_other_400_errors_are_fatal(tmp_path, monkeypatch):
    monkeypatch.setattr(loop_module.time, "sleep", lambda *_: None)
    err = LLMAPIError("OpenAI HTTP 400: Invalid parameter: messages", 400)
    agent, fake = _agent_with_script(tmp_path, [err] * 3)
    with pytest.raises(LLMError, match="rejected"):
        agent._chat(messages=agent.messages)
    assert fake.chat.completions.create.call_count == 1


def test_anthropic_overload_529_is_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(loop_module.time, "sleep", lambda *_: None)
    ok = MagicMock()
    ok.choices = [MagicMock()]
    err = LLMAPIError("Anthropic HTTP 529: Overloaded [type=overloaded_error]", 529)
    agent, fake = _agent_with_script(tmp_path, [err, err, ok])
    assert agent._chat(messages=agent.messages) is ok
    assert fake.chat.completions.create.call_count == 3


def test_retry_after_header_hint_is_used():
    assert loop_module._retry_delay(Exception("x [retry-after=12s]"), 1) >= 12


def test_secret_redaction_covers_provider_keys():
    from schemas import sanitize_evidence
    text = "k1=sk-ant-api03-" + "A" * 40 + " k2=sk-proj-" + "b" * 30 + " k3=AIza" + "C" * 35 + " k4=gsk_" + "d" * 30
    out = sanitize_evidence(text)
    for fragment in ("A" * 20, "b" * 20, "C" * 20, "d" * 20):
        assert fragment not in out
    assert llm.is_chat_capable("gpt-4.1-mini") and not llm.is_chat_capable("whisper-1")


def test_overloaded_model_rotates_to_another_and_remembers_the_first(tmp_path, monkeypatch):
    monkeypatch.setattr(loop_module.time, "sleep", lambda *_: None)
    ok = MagicMock()
    ok.choices = [MagicMock()]
    busy = LLMAPIError("Gemini HTTP 503: This model is currently experiencing high demand [status=UNAVAILABLE]", 503)
    agent, fake = _agent_with_script(tmp_path, [busy, busy, ok])
    agent.model, agent.model_fallbacks = "model-a", ["model-b", "model-c"]
    assert agent._chat(messages=agent.messages) is ok
    used = [call.kwargs["model"] for call in fake.chat.completions.create.call_args_list]
    assert used == ["model-a", "model-a", "model-b"]  # two overloads on model-a, then rotate
    assert agent.model == "model-b" and agent.model_fallbacks == ["model-c", "model-a"]


def test_single_overload_is_just_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(loop_module.time, "sleep", lambda *_: None)
    ok = MagicMock()
    ok.choices = [MagicMock()]
    busy = LLMAPIError("Anthropic HTTP 529: Overloaded", 529)
    agent, fake = _agent_with_script(tmp_path, [busy, ok])
    agent.model, agent.model_fallbacks = "model-a", ["model-b"]
    assert agent._chat(messages=agent.messages) is ok
    assert agent.model == "model-a"


def test_exhausted_models_do_not_consume_the_retry_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(loop_module.time, "sleep", lambda *_: None)
    ok = MagicMock()
    ok.choices = [MagicMock()]
    quota = LLMAPIError("Gemini HTTP 429: quota [quota=GenerateRequestsPerDayPerProjectPerModel-FreeTier]", 429)
    # 12 exhausted models in front of the one that works: more than the 8-attempt retry budget
    agent, fake = _agent_with_script(tmp_path, [quota] * 12 + [ok])
    agent.model = "m0"
    agent.model_fallbacks = [f"m{i}" for i in range(1, 13)]
    assert agent._chat(messages=agent.messages) is ok
    assert agent.model == "m12"
    assert fake.chat.completions.create.call_count == 13

