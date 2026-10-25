"""Tests for the Gemini REST client translation layer (no network)."""

import json

import pytest

import llm
from llm import GeminiAPIError, GeminiClient, convert_schema, convert_tools
from prompts import TOOL_DEFINITIONS


def test_convert_schema_handles_nullable_types_and_drops_unsupported_keys():
    out = convert_schema({
        "type": "object",
        "properties": {
            "file": {"type": ["string", "null"], "description": "x"},
            "n": {"type": "integer", "default": 3},
        },
        "additionalProperties": False,
    })
    assert out["properties"]["file"] == {"type": "string", "nullable": True, "description": "x"}
    assert "default" not in out["properties"]["n"]
    assert "additionalProperties" not in out


def test_all_agent_tools_convert():
    decls = convert_tools(TOOL_DEFINITIONS)[0]["functionDeclarations"]
    names = {d["name"] for d in decls}
    assert {"list_files", "read_file", "search_code", "git_log", "submit_final_report"} <= names
    no_arg = next(d for d in decls if d["name"] == "run_ruff")
    assert "parameters" not in no_arg  # Gemini rejects empty-object schemas


def test_message_translation_roundtrip():
    client = GeminiClient("k")
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "thinking", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "list_files", "arguments": "{}"}},
            {"id": "c2", "type": "function", "function": {"name": "git_log", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "c1", "name": "list_files", "content": "tree"},
        {"role": "tool", "tool_call_id": "c2", "name": "git_log", "content": "log"},
        {"role": "user", "content": "continue"},
    ]
    body = client._to_gemini_contents(messages)
    assert body["systemInstruction"]["parts"][0]["text"] == "sys"
    roles = [c["role"] for c in body["contents"]]
    assert roles == ["user", "model", "user"]  # strictly alternating, tool results merged
    model_parts = body["contents"][1]["parts"]
    assert [p.get("functionCall", {}).get("name") for p in model_parts[1:]] == ["list_files", "git_log"]
    user_parts = body["contents"][2]["parts"]
    assert sum("functionResponse" in p for p in user_parts) == 2


def test_create_sends_forced_tool_choice_and_parses_function_call(monkeypatch):
    client = GeminiClient("k")
    seen = {}

    def fake_request(method, url, body=None):
        seen["url"], seen["body"] = url, body
        return {"candidates": [{"finishReason": "STOP", "content": {"parts": [
            {"text": "reasoning"},
            {"thought": True, "text": "hidden"},
            {"functionCall": {"name": "submit_final_report", "args": {"violations": []}},
             "thoughtSignature": "sig123"},
        ]}}]}

    monkeypatch.setattr(client, "_request", fake_request)
    resp = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "x"}],
        tools=TOOL_DEFINITIONS,
        tool_choice={"type": "function", "function": {"name": "submit_final_report"}},
        temperature=0.1,
    )
    cfg = seen["body"]["toolConfig"]["functionCallingConfig"]
    assert cfg == {"mode": "ANY", "allowedFunctionNames": ["submit_final_report"]}
    assert seen["url"].endswith("/models/gemini-2.5-flash:generateContent")
    msg = resp.choices[0].message
    assert msg.content == "reasoning"  # thought parts are skipped
    call = msg.tool_calls[0]
    assert call.function.name == "submit_final_report"
    assert json.loads(call.function.arguments) == {"violations": []}
    # thought signature is echoed back on the next turn
    body = client._to_gemini_contents([
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": call.id, "type": "function", "function": {"name": "submit_final_report", "arguments": "{}"}}]},
    ])
    assert body["contents"][1]["parts"][0]["thoughtSignature"] == "sig123"


def test_json_mode_sets_mime_type(monkeypatch):
    client = GeminiClient("k")
    captured = {}
    monkeypatch.setattr(client, "_request", lambda m, u, b=None: captured.update(body=b) or
                        {"candidates": [{"content": {"parts": [{"text": "{}"}]}}]})
    client.chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}],
                                   response_format={"type": "json_object"})
    assert captured["body"]["generationConfig"]["responseMimeType"] == "application/json"


def test_blocked_or_malformed_responses_raise_retryable_errors():
    client = GeminiClient("k")
    with pytest.raises(GeminiAPIError, match="tool_use_failed") as e1:
        client._to_response({"candidates": [{"finishReason": "MALFORMED_FUNCTION_CALL"}]})
    assert e1.value.status_code is None
    with pytest.raises(GeminiAPIError):
        client._to_response({"promptFeedback": {"blockReason": "SAFETY"}})


def test_list_models_filters_generate_content(monkeypatch):
    client = GeminiClient("k")
    monkeypatch.setattr(client, "_request", lambda m, u, b=None: {"models": [
        {"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/embedding-001", "supportedGenerationMethods": ["embedContent"]},
    ]})
    assert [m.id for m in client.models.list().data] == ["gemini-2.5-flash"]


def test_list_models_excludes_non_chat_models_and_paginates(monkeypatch):
    client = GeminiClient("k")
    pages = iter([
        {"models": [
            {"name": "models/gemini-3.5-flash", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/gemini-3.1-flash-tts", "supportedGenerationMethods": ["generateContent"]},
        ], "nextPageToken": "p2"},
        {"models": [
            {"name": "models/gemini-3-pro-image", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/gemini-3-flash-preview", "supportedGenerationMethods": ["generateContent"]},
        ]},
    ])
    urls = []
    monkeypatch.setattr(client, "_request", lambda m, u, b=None: urls.append(u) or next(pages))
    assert [m.id for m in client.models.list().data] == ["gemini-3.5-flash", "gemini-3-flash-preview"]
    assert "pageToken=p2" in urls[1]


def test_is_chat_capable():
    assert llm.is_chat_capable("gemini-3.5-flash")
    assert llm.is_chat_capable("openai/gpt-oss-120b")
    for bad in ("gemini-3.1-flash-tts", "whisper-large-v3", "gemini-3-pro-image", "gemma-4-31b-it",
                "meta-llama/llama-prompt-guard-2-86m", "canopylabs/orpheus-v1-english"):
        assert not llm.is_chat_capable(bad)
