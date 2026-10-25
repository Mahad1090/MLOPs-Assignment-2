"""Provider clients exposing the same tiny surface the agent loop uses from the Groq SDK.

The loop only needs ``client.models.list().data[i].id`` and
``client.chat.completions.create(model=..., messages=..., tools=..., tool_choice=...,
response_format=..., temperature=...)`` returning ``choices[0].message.{content,tool_calls}``.

* ``GeminiClient``    - Google Gemini ``generateContent`` REST API
* ``OpenAIClient``    - OpenAI Chat Completions (also any OpenAI-compatible gateway via ``base_url``)
* ``AnthropicClient`` - Anthropic (Claude) Messages API

All three use plain HTTPS through urllib (no extra dependencies) and translate OpenAI-style
messages and tool schemas to the provider's native format and back. Errors are raised as
``LLMAPIError`` with a ``status_code`` so the loop can decide whether to retry; error text never
contains the API key.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
OPENAI_BASE_URL = "https://api.openai.com/v1"
ANTHROPIC_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"
USER_AGENT = "devops-auditing-agent/1.0 (+https://github.com; python-urllib)"
SCHEMA_DROP_KEYS = {"default", "additionalProperties", "$schema", "title"}

# Model ids that cannot drive a text/tool-calling agent (audio, image, embeddings, guards, ...)
NON_CHAT_MARKERS = (
    "tts", "image", "whisper", "orpheus", "guard", "embed", "transcribe", "robotics",
    "computer-use", "lyria", "veo", "imagen", "audio", "live", "nano-banana",
    "antigravity", "deep-research", "omni", "gemma",
)


def is_chat_capable(model_id: str, extra_markers: tuple = ()) -> bool:
    """True when the model id looks like a general text model usable for tool calling."""
    mid = model_id.lower()
    return not any(marker in mid for marker in NON_CHAT_MARKERS + tuple(extra_markers))


class LLMAPIError(Exception):
    """Provider request failure; ``status_code`` lets the loop decide whether to retry."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


# Backwards-compatible name used by earlier code and tests
GeminiAPIError = LLMAPIError


def _error_message(raw: str, headers: Any) -> str:
    """Flatten a provider error body into one safe line, keeping the facts retry logic needs."""
    extras: List[str] = []
    try:
        payload = json.loads(raw)
        err = payload.get("error", payload) if isinstance(payload, dict) else {}
        if isinstance(err, str):
            message = err
        else:
            message = str(err.get("message") or raw[:250])
            for key in ("type", "code", "status"):
                value = err.get(key)
                if isinstance(value, str) and value:
                    extras.append(f"{key}={value}")
            for det in err.get("details", []) or []:
                if not isinstance(det, dict):
                    continue
                for viol in det.get("violations", []) or []:
                    if isinstance(viol, dict) and viol.get("quotaId"):
                        extras.append(f"quota={viol['quotaId']}")
                if det.get("retryDelay"):
                    extras.append(f"retryDelay={det['retryDelay']}")
    except Exception:
        message = raw[:250]
    try:
        retry_after = headers.get("retry-after") if headers else None
    except Exception:
        retry_after = None
    if retry_after:
        extras.append(f"retry-after={retry_after}s")
    if extras:
        return f"{message[:220]} [{'; '.join(dict.fromkeys(extras))}]"
    return message[:300]


def http_json(
    method: str,
    url: str,
    headers: Dict[str, str],
    body: Optional[Dict[str, Any]],
    timeout: float,
    label: str,
) -> Dict[str, Any]:
    """Perform one JSON HTTP call, mapping failures to ``LLMAPIError``."""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    # CDNs in front of provider APIs (e.g. Cloudflare) reject urllib's default "Python-urllib" agent
    merged = {"User-Agent": USER_AGENT, "Accept": "application/json", **headers}
    req = urllib.request.Request(url, data=data, method=method, headers=merged)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise LLMAPIError(f"{label} HTTP {exc.code}: {_error_message(raw, exc.headers)}", exc.code) from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise LLMAPIError(f"{label} connection error: {exc}", None) from None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raise LLMAPIError(f"{label} returned invalid JSON (tool_use_failed-style retry)", None) from None


def _tool_call(call_id: str, name: str, arguments: str) -> Any:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _client_namespaces(client: Any) -> None:
    client.chat = SimpleNamespace(completions=SimpleNamespace(create=client._create))
    client.models = SimpleNamespace(list=client._list_models)


# ===================================================================== Gemini
def convert_schema(schema: Any) -> Any:
    """Translate a JSON-schema fragment to Gemini's OpenAPI subset."""
    if isinstance(schema, list):
        return [convert_schema(s) for s in schema]
    if not isinstance(schema, dict):
        return schema
    out: Dict[str, Any] = {}
    for key, value in schema.items():
        if key in SCHEMA_DROP_KEYS:
            continue
        if key == "type" and isinstance(value, list):
            non_null = [t for t in value if t != "null"]
            out["type"] = (non_null[0] if non_null else "string")
            if "null" in value:
                out["nullable"] = True
        elif key == "properties" and isinstance(value, dict):
            out["properties"] = {k: convert_schema(v) for k, v in value.items()}
        elif key == "items":
            out["items"] = convert_schema(value)
        else:
            out[key] = value
    return out


def convert_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    declarations = []
    for tool in tools:
        fn = tool.get("function", {})
        decl: Dict[str, Any] = {"name": fn["name"], "description": fn.get("description", "")}
        params = convert_schema(fn.get("parameters") or {})
        # Gemini rejects an object schema with an empty properties map
        if params and params.get("properties"):
            decl["parameters"] = params
        declarations.append(decl)
    return [{"functionDeclarations": declarations}]


class GeminiClient:
    """Duck-typed stand-in for ``groq.Groq`` backed by the Gemini REST API."""

    def __init__(self, api_key: str, timeout: float = 90.0, base_url: str = GEMINI_BASE_URL):
        self.api_key = api_key
        self.timeout = timeout
        self.base_url = base_url.rstrip("/")
        self._signatures: Dict[str, str] = {}  # tool_call_id -> thoughtSignature (Gemini 3 needs it echoed)
        self._counter = 0
        _client_namespaces(self)

    def _request(self, method: str, url: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}
        return http_json(method, url, headers, body, self.timeout, "Gemini")

    def _list_models(self) -> Any:
        models: List[Any] = []
        token = ""
        for _ in range(5):  # bounded pagination
            url = f"{self.base_url}/models?pageSize=100" + (f"&pageToken={token}" if token else "")
            payload = self._request("GET", url)
            for m in payload.get("models", []):
                model_id = m["name"].split("/", 1)[-1]
                if "generateContent" in m.get("supportedGenerationMethods", []) and is_chat_capable(model_id):
                    models.append(SimpleNamespace(id=model_id))
            token = payload.get("nextPageToken", "")
            if not token:
                break
        return SimpleNamespace(data=models)

    def _to_gemini_contents(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        system_parts: List[Dict[str, Any]] = []
        contents: List[Dict[str, Any]] = []

        def push(role: str, parts: List[Dict[str, Any]]) -> None:
            if contents and contents[-1]["role"] == role:
                contents[-1]["parts"].extend(parts)  # Gemini wants strictly alternating turns
            else:
                contents.append({"role": role, "parts": parts})

        for msg in messages:
            role = msg.get("role")
            text = msg.get("content") or ""
            if role == "system":
                system_parts.append({"text": text})
            elif role == "user":
                push("user", [{"text": text or "(no content)"}])
            elif role == "assistant":
                parts: List[Dict[str, Any]] = []
                if text.strip():
                    parts.append({"text": text})
                for tc in msg.get("tool_calls") or []:
                    try:
                        args = json.loads(tc["function"].get("arguments") or "{}")
                    except Exception:
                        args = {}
                    part: Dict[str, Any] = {"functionCall": {"name": tc["function"]["name"], "args": args}}
                    sig = self._signatures.get(tc.get("id", ""))
                    if sig:
                        part["thoughtSignature"] = sig
                    parts.append(part)
                if parts:
                    push("model", parts)
            elif role == "tool":
                push("user", [{
                    "functionResponse": {
                        "name": msg.get("name") or "tool",
                        "response": {"result": text},
                    }
                }])

        if not contents or contents[0]["role"] != "user":
            contents.insert(0, {"role": "user", "parts": [{"text": "Begin."}]})
        body: Dict[str, Any] = {"contents": contents}
        if system_parts:
            body["systemInstruction"] = {"parts": system_parts}
        return body

    def _create(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Any = None,
        response_format: Optional[Dict[str, Any]] = None,
        temperature: float = 0.1,
        **_: Any,
    ) -> Any:
        body = self._to_gemini_contents(messages)
        gen_config: Dict[str, Any] = {"temperature": temperature}
        if response_format and response_format.get("type") == "json_object":
            gen_config["responseMimeType"] = "application/json"
        body["generationConfig"] = gen_config

        if tools:
            body["tools"] = convert_tools(tools)
            mode, allowed = "AUTO", None
            if isinstance(tool_choice, dict):
                mode = "ANY"
                allowed = [tool_choice.get("function", {}).get("name")]
            elif tool_choice == "required":
                mode = "ANY"
            cfg: Dict[str, Any] = {"mode": mode}
            if allowed and allowed[0]:
                cfg["allowedFunctionNames"] = allowed
            body["toolConfig"] = {"functionCallingConfig": cfg}

        payload = self._request("POST", f"{self.base_url}/models/{model}:generateContent", body)
        return self._to_response(payload)

    def _to_response(self, payload: Dict[str, Any]) -> Any:
        candidates = payload.get("candidates") or []
        if not candidates:
            reason = (payload.get("promptFeedback") or {}).get("blockReason", "no candidates")
            raise LLMAPIError(f"Gemini returned no candidate ({reason}); tool_use_failed-style retry", None)
        cand = candidates[0]
        finish = cand.get("finishReason", "")
        if finish == "MALFORMED_FUNCTION_CALL":
            raise LLMAPIError("Gemini produced a malformed function call (tool_use_failed)", None)

        texts: List[str] = []
        tool_calls: List[Any] = []
        for part in (cand.get("content") or {}).get("parts", []):
            if part.get("thought"):
                continue
            if "text" in part:
                texts.append(part["text"])
            elif "functionCall" in part:
                self._counter += 1
                call_id = f"call_{self._counter}"
                fc = part["functionCall"]
                if part.get("thoughtSignature"):
                    self._signatures[call_id] = part["thoughtSignature"]
                tool_calls.append(_tool_call(call_id, fc.get("name", ""), json.dumps(fc.get("args") or {})))
        message = SimpleNamespace(content="".join(texts), tool_calls=tool_calls or None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish)])


# ===================================================================== OpenAI
class OpenAIClient:
    """OpenAI Chat Completions client (also works with OpenAI-compatible gateways via ``base_url``)."""

    def __init__(
        self,
        api_key: str,
        timeout: float = 90.0,
        base_url: str = OPENAI_BASE_URL,
        label: str = "OpenAI",
    ):
        self.api_key = api_key
        self.timeout = timeout
        self.base_url = base_url.rstrip("/")
        self.label = label
        self._no_temperature: set = set()  # models that reject a custom temperature (e.g. o-series, gpt-5)
        _client_namespaces(self)

    def _request(self, method: str, url: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        return http_json(method, url, headers, body, self.timeout, self.label)

    def _list_models(self) -> Any:
        payload = self._request("GET", f"{self.base_url}/models")
        data = [SimpleNamespace(id=m["id"]) for m in payload.get("data", []) if isinstance(m, dict) and m.get("id")]
        return SimpleNamespace(data=data)

    @staticmethod
    def _clean_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        cleaned: List[Dict[str, Any]] = []
        for msg in messages:
            m = dict(msg)
            if m.get("role") == "tool":
                m.pop("name", None)  # not part of the tool message schema
                m["content"] = m.get("content") or "(empty output)"
            elif m.get("role") == "assistant" and not m.get("tool_calls"):
                m["content"] = m.get("content") or ""
            cleaned.append(m)
        return cleaned

    def _create(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Any = None,
        response_format: Optional[Dict[str, Any]] = None,
        temperature: float = 0.1,
        **_: Any,
    ) -> Any:
        body: Dict[str, Any] = {"model": model, "messages": self._clean_messages(messages)}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice or "auto"
        if response_format:
            body["response_format"] = response_format
        if model not in self._no_temperature:
            body["temperature"] = temperature

        url = f"{self.base_url}/chat/completions"
        try:
            payload = self._request("POST", url, body)
        except LLMAPIError as exc:
            if exc.status_code == 400 and "temperature" in str(exc).lower() and "temperature" in body:
                self._no_temperature.add(model)
                body.pop("temperature")
                payload = self._request("POST", url, body)
            else:
                raise
        return self._to_response(payload)

    def _to_response(self, payload: Dict[str, Any]) -> Any:
        choices = payload.get("choices") or []
        if not choices:
            raise LLMAPIError(f"{self.label} returned no choices (tool_use_failed-style retry)", None)
        raw = choices[0].get("message") or {}
        tool_calls = []
        for tc in raw.get("tool_calls") or []:
            fn = tc.get("function") or {}
            tool_calls.append(_tool_call(tc.get("id") or f"call_{len(tool_calls) + 1}", fn.get("name", ""), fn.get("arguments") or "{}"))
        content = raw.get("content")
        if isinstance(content, list):  # some gateways return content parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        message = SimpleNamespace(content=content or "", tool_calls=tool_calls or None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=choices[0].get("finish_reason"))])


# =================================================================== Anthropic
class AnthropicClient:
    """Anthropic (Claude) Messages API client."""

    JSON_MODE_NOTE = "\n\nRespond with a single valid JSON object only, with no prose and no Markdown fences."

    def __init__(
        self,
        api_key: str,
        timeout: float = 120.0,
        base_url: str = ANTHROPIC_BASE_URL,
        max_tokens: int = 8192,
    ):
        self.api_key = api_key
        self.timeout = timeout
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self._no_temperature: set = set()
        _client_namespaces(self)

    def _request(self, method: str, url: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        return http_json(method, url, headers, body, self.timeout, "Anthropic")

    def _list_models(self) -> Any:
        models: List[Any] = []
        after = ""
        for _ in range(5):  # bounded pagination
            url = f"{self.base_url}/v1/models?limit=100" + (f"&after_id={after}" if after else "")
            payload = self._request("GET", url)
            models.extend(SimpleNamespace(id=m["id"]) for m in payload.get("data", []) if m.get("id"))
            if not payload.get("has_more") or not payload.get("last_id"):
                break
            after = payload["last_id"]
        return SimpleNamespace(data=models)

    # ----------------------------------------------------------- translation
    @staticmethod
    def _blocks(content: Any) -> List[Dict[str, Any]]:
        if isinstance(content, list):
            return list(content)
        return [{"type": "text", "text": content or "(no content)"}]

    def _to_anthropic(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        system_parts: List[str] = []
        turns: List[Dict[str, Any]] = []

        def push(role: str, blocks: List[Dict[str, Any]]) -> None:
            if turns and turns[-1]["role"] == role:
                turns[-1]["content"].extend(blocks)  # Anthropic wants strictly alternating turns
            else:
                turns.append({"role": role, "content": blocks})

        for msg in messages:
            role = msg.get("role")
            text = msg.get("content") or ""
            if role == "system":
                system_parts.append(text)
            elif role == "user":
                push("user", [{"type": "text", "text": text or "(no content)"}])
            elif role == "assistant":
                blocks: List[Dict[str, Any]] = []
                if text.strip():
                    blocks.append({"type": "text", "text": text})
                for tc in msg.get("tool_calls") or []:
                    try:
                        args = json.loads(tc["function"].get("arguments") or "{}")
                        if not isinstance(args, dict):
                            args = {}
                    except Exception:
                        args = {}
                    blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id") or f"toolu_{len(blocks)}",
                        "name": tc["function"]["name"],
                        "input": args,
                    })
                if blocks:
                    push("assistant", blocks)
            elif role == "tool":
                push("user", [{
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": text or "(empty output)",
                }])

        self._repair_tool_results(turns)
        # tool_result blocks must precede any text in the same user turn
        for turn in turns:
            if turn["role"] == "user":
                turn["content"].sort(key=lambda b: 0 if b.get("type") == "tool_result" else 1)
        if not turns or turns[0]["role"] != "user":
            turns.insert(0, {"role": "user", "content": [{"type": "text", "text": "Begin."}]})
        return {"system": "\n\n".join(p for p in system_parts if p), "messages": turns}

    @staticmethod
    def _repair_tool_results(turns: List[Dict[str, Any]]) -> None:
        """Every tool_use must be answered by a tool_result in the next user turn."""
        for i, turn in enumerate(turns):
            if turn["role"] != "assistant":
                continue
            wanted = [b["id"] for b in turn["content"] if b.get("type") == "tool_use"]
            if not wanted:
                continue
            if i + 1 >= len(turns) or turns[i + 1]["role"] != "user":
                turns.insert(i + 1, {"role": "user", "content": []})
            nxt = turns[i + 1]
            answered = {b.get("tool_use_id") for b in nxt["content"] if b.get("type") == "tool_result"}
            for tool_id in wanted:
                if tool_id not in answered:
                    nxt["content"].append({"type": "tool_result", "tool_use_id": tool_id, "content": "(no result)"})

    @staticmethod
    def _convert_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for tool in tools:
            fn = tool.get("function", {})
            schema = fn.get("parameters") or {"type": "object", "properties": {}}
            if schema.get("type") != "object":
                schema = {"type": "object", "properties": {}}
            out.append({
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": {k: v for k, v in schema.items() if k not in {"$schema", "title"}},
            })
        return out

    def _create(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Any = None,
        response_format: Optional[Dict[str, Any]] = None,
        temperature: float = 0.1,
        **_: Any,
    ) -> Any:
        converted = self._to_anthropic(messages)
        system_text = converted["system"]
        if response_format and response_format.get("type") == "json_object":
            system_text += self.JSON_MODE_NOTE
        body: Dict[str, Any] = {
            "model": model,
            "max_tokens": self.max_tokens,
            "messages": converted["messages"],
        }
        if system_text:
            # cache_control lets Anthropic reuse the (large, constant) system prompt across turns
            body["system"] = [{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}]
        if model not in self._no_temperature:
            body["temperature"] = temperature
        if tools:
            body["tools"] = self._convert_tools(tools)
            if isinstance(tool_choice, dict):
                name = tool_choice.get("function", {}).get("name")
                body["tool_choice"] = {"type": "tool", "name": name} if name else {"type": "any"}
            elif tool_choice == "required":
                body["tool_choice"] = {"type": "any"}
            else:
                body["tool_choice"] = {"type": "auto"}

        url = f"{self.base_url}/v1/messages"
        for _attempt in range(3):
            try:
                payload = self._request("POST", url, body)
                break
            except LLMAPIError as exc:
                text = str(exc).lower()
                if exc.status_code == 400 and "temperature" in text and "temperature" in body:
                    self._no_temperature.add(model)
                    body.pop("temperature")
                elif exc.status_code == 400 and "max_tokens" in text and body["max_tokens"] > 4096:
                    self.max_tokens = body["max_tokens"] = 4096
                else:
                    raise
        else:  # pragma: no cover - defensive; the loop above either breaks or raises
            raise LLMAPIError("Anthropic request could not be adjusted to the model's limits", 400)
        return self._to_response(payload)

    def _to_response(self, payload: Dict[str, Any]) -> Any:
        if payload.get("type") == "error":
            raise LLMAPIError(f"Anthropic error: {str(payload.get('error'))[:200]}", None)
        texts: List[str] = []
        tool_calls: List[Any] = []
        for block in payload.get("content") or []:
            kind = block.get("type")
            if kind == "text":
                texts.append(block.get("text", ""))
            elif kind == "tool_use":
                tool_calls.append(_tool_call(block.get("id") or f"toolu_{len(tool_calls) + 1}", block.get("name", ""), json.dumps(block.get("input") or {})))
        message = SimpleNamespace(content="".join(texts), tool_calls=tool_calls or None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=payload.get("stop_reason"))])
