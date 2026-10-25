"""Native LLM-Tool Loop for DevOps Repository Auditing Agent.

Implements the Observe -> Reason -> Select Tool -> Execute Tool -> Interpret Results -> Report
cycle without external agent frameworks (conforming to Section 3 restriction).

The LLM alone decides which tool to call next; this module only dispatches the calls,
keeps the conversation within a bounded context, retries transient API failures, and
validates the final report. If the LLM cannot be reached, it raises ``LLMError`` instead of
fabricating an (empty) "clean" report.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv  # type: ignore
except ImportError:
    def load_dotenv(dotenv_path: str = ".env") -> bool:
        """Lightweight native fallback to parse .env files if python-dotenv is not installed."""
        if not os.path.exists(dotenv_path):
            return False
        try:
            with open(dotenv_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k not in os.environ:
                        os.environ[k] = v
            return True
        except Exception:
            return False

try:
    import groq  # type: ignore
    from groq import Groq  # type: ignore
except ImportError:
    groq = None  # type: ignore
    Groq = None  # type: ignore

from llm import (
    ANTHROPIC_BASE_URL,
    OPENAI_BASE_URL,
    AnthropicClient,
    GeminiClient,
    OpenAIClient,
    is_chat_capable,
)
from prompts import SYSTEM_PROMPT, TOOL_DEFINITIONS
from schemas import AuditReport, Violation, sanitize_evidence
from tools import FIXTURE_DIR_NAMES, _resolve_safe_path, dispatch_tool_call, is_key_file

# Load local environment variables from .env if present
load_dotenv()

DEFAULT_PROVIDER = "gemini"
# Auto-detection order when several API keys are configured and nothing says which to use
PROVIDER_ORDER = ["gemini", "anthropic", "openai", "groq"]
OPENAI_EXCLUDE_MARKERS = (
    "embedding", "dall-e", "davinci", "babbage", "moderation", "realtime", "search", "transcribe",
    "instruct", "sora", "codex", "-pro", "deep-research", "ft:", "chatgpt-image", "gpt-image",
)
PROVIDERS: Dict[str, Dict[str, Any]] = {
    "gemini": {
        "key_envs": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "model_env": "GEMINI_MODEL",
        "default_model": "gemini-3.5-flash",
        "preference": [
            "gemini-3.5-flash",
            "gemini-3.8-flash",
            "gemini-3.7-flash",
            "gemini-3.6-flash",
            "gemini-3-flash-preview",
            "gemini-3.1-flash-lite",
            "gemini-flash-lite-latest",
        ],
        "key_prefix": "AIza",
    },
    "anthropic": {
        "key_envs": ["ANTHROPIC_API_KEY"],
        "model_env": "ANTHROPIC_MODEL",
        "base_url_env": "ANTHROPIC_BASE_URL",
        "default_model": "claude-sonnet-5-5",
        "preference": [
            "claude-sonnet-5-5",
            "claude-haiku-5-5",
            "claude-opus-5-5",
            "claude-fable-5-1",
            "claude-sonnet-4-5",
            "claude-haiku-4-5",
            "claude-opus-4-5",
        ],
        "key_prefix": "sk-ant-",
        "trust_requested": True,  # aliases / gateway model names need not appear in the model list
    },
    "openai": {
        "key_envs": ["OPENAI_API_KEY"],
        "model_env": "OPENAI_MODEL",
        "base_url_env": "OPENAI_BASE_URL",
        "default_model": "gpt-4.1-mini",
        "preference": ["gpt-4.1-mini", "gpt-4o-mini", "gpt-4.1", "gpt-4o", "gpt-5-mini", "gpt-5"],
        "exclude_markers": OPENAI_EXCLUDE_MARKERS,
        "key_prefix": "sk-",
        "trust_requested": True,
    },
    "groq": {
        "key_envs": ["GROQ_API_KEY"],
        "model_env": "GROQ_MODEL",
        "default_model": "openai/gpt-oss-120b",
        "preference": [
            "openai/gpt-oss-120b",
            "llama-3.3-70b-versatile",
            "qwen/qwen3.8-27b",
            "openai/gpt-oss-20b",
        ],
        "key_prefix": "gsk_",
    },
}
# Backwards-compatible aliases (Groq defaults)
DEFAULT_MODEL = PROVIDERS["groq"]["default_model"]
MODEL_PREFERENCE = PROVIDERS["groq"]["preference"]


def get_api_key(provider: str) -> Optional[str]:
    """First non-empty API key found in the provider's environment variables."""
    for name in PROVIDERS[provider]["key_envs"]:
        value = (os.getenv(name) or "").strip()
        if value:
            return value
    return None


def key_env_label(provider: str) -> str:
    return " or ".join(PROVIDERS[provider]["key_envs"])


def detect_provider_from_key(api_key: Optional[str]) -> Optional[str]:
    """Identify the provider from a key's well-known prefix (None when unrecognised)."""
    key = (api_key or "").strip()
    if key.startswith("sk-ant-"):
        return "anthropic"
    if key.startswith("gsk_"):
        return "groq"
    if key.startswith("AIza"):
        return "gemini"
    if key.startswith("sk-"):
        return "openai"
    return None


def _validate_provider(name: str) -> str:
    name = name.strip().lower()
    if name not in PROVIDERS:
        raise ValueError(f"Unknown LLM provider '{name}'. Choose one of: {', '.join(sorted(PROVIDERS))}.")
    return name


def resolve_provider(provider: Optional[str] = None, api_key: Optional[str] = None) -> str:
    """Pick the LLM backend.

    Order: explicit ``provider`` argument (strict) > well-known prefix of a supplied ``api_key``
    (sk-ant- Claude, gsk_ Groq, AIza Gemini, sk- OpenAI) > LLM_PROVIDER env var > whichever provider
    has an API key configured (several: gemini, anthropic, openai, groq in that order) > gemini.
    An LLM_PROVIDER whose key is missing falls back to a provider that does have a key.
    """
    if provider:
        return _validate_provider(provider)

    configured = _validate_provider(os.getenv("LLM_PROVIDER")) if (os.getenv("LLM_PROVIDER") or "").strip() else ""

    if api_key:
        return detect_provider_from_key(api_key) or configured or DEFAULT_PROVIDER

    if configured:
        if get_api_key(configured):
            return configured
        for other in PROVIDER_ORDER:
            if other != configured and get_api_key(other):
                return other
        return configured

    for name in PROVIDER_ORDER:
        if get_api_key(name):
            return name
    return DEFAULT_PROVIDER


def _match_model(candidate: str, usable: List[str]) -> Optional[str]:
    """Exact id, else the newest dated variant of an alias (claude-sonnet-4-5 -> claude-sonnet-4-5-2025...)."""
    if candidate in usable:
        return candidate
    dated = sorted((m for m in usable if m.startswith(candidate + "-")), reverse=True)
    return dated[0] if dated else None


def plan_models(
    requested: str,
    available: List[str],
    preference: List[str],
    exclude_markers: tuple = (),
    trust_requested: bool = False,
) -> "tuple[str, List[str]]":
    """Choose the starting model and an ordered fallback chain.

    Only models the key can actually list are used. Order: the requested model, the
    provider's preferred models, then any other chat-capable model (flash-class first).
    Non-chat models (TTS, image, audio, embeddings, guards, ...) are never selected.
    With ``trust_requested`` (Claude/OpenAI/gateways) an unlisted requested model is still tried
    first, because aliases, fine-tunes and gateway names often are not in the model list.
    """
    if not available:
        return requested, [m for m in preference if m != requested]
    usable = [m for m in available if is_chat_capable(m, exclude_markers)]
    ordered: List[str] = []
    matched_requested = _match_model(requested, usable)
    if matched_requested:
        ordered.append(matched_requested)
    elif trust_requested:
        ordered.append(requested)
    for cand in preference:
        hit = _match_model(cand, usable)
        if hit and hit not in ordered and hit != requested:
            ordered.append(hit)
    # Newest-looking names first (older generations are often closed to new keys), flash-class before others.
    extras = sorted((m for m in usable if m not in ordered), reverse=True)
    extras.sort(key=lambda m: 0 if "flash" in m else 1)
    ordered.extend(extras)
    if not ordered:
        return available[0], []
    return ordered[0], ordered[1:]


def default_model_for(provider: str) -> str:
    cfg = PROVIDERS[provider]
    return os.getenv(cfg["model_env"]) or cfg["default_model"]


# Credential-looking text in search_code output. Provider-prefixed tokens and private-key headers are
# reported as-is; generic NAME = "value" assignments are skipped when the value looks like a placeholder.
HARD_SECRET_LEAD = re.compile(
    r"AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|\bgh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
    r"|\bxox[baprs]-[A-Za-z0-9-]{10,}|\bsk-ant-[A-Za-z0-9_-]{20,}|\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}"
    r"|\bAIza[0-9A-Za-z_-]{30,}|\bgsk_[A-Za-z0-9]{20,}"
)
ASSIGNED_SECRET_LEAD = re.compile(
    r"""(?i)(?:api[_-]?key|secret[_-]?key|access[_-]?key|client[_-]?secret|secret|password|passwd|token)\w*\s*[:=]\s*["']([^"'\s]{8,})["']"""
)
PLACEHOLDER_VALUE = re.compile(
    r"(?i)change[_-]?me|example|your[_-]|placeholder|x{4,}|<[^>]+>|\$\{|%\(|dummy|sample|todo|\*{4,}|^\$|^env\.|getenv|os\.environ"
)

LLM_MAX_ATTEMPTS = 8
LLM_REQUEST_TIMEOUT = 90.0
LLM_MAX_BACKOFF = 65.0
KEEP_RECENT_TOOL_RESULTS = 3
OLD_TOOL_RESULT_CHARS = 500
RECENT_TOOL_RESULT_CHARS = 3500
MIN_TOOL_CALLS_BEFORE_REPORT = 4
TRACE_RESULT_CHARS = 1500
TRUNCATION_MARKER = "\n...[older tool output truncated to save context]"


class LLMError(RuntimeError):
    """Raised when the LLM provider cannot be used to complete the audit."""


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Extract and parse a JSON object from raw LLM output."""
    text = (text or "").strip()
    if not text:
        return None

    # Try direct parse
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # Try extracting from markdown ```json ... ``` code fence
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence_match:
        try:
            data = json.loads(fence_match.group(1))
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    # Try finding the first '{' and the last '}'
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    return None


def _violations_from_args(args: Dict[str, Any]) -> List[Any]:
    """Pull a violations list out of submit_final_report arguments (tolerating strings)."""
    violations = args.get("violations", [])
    if isinstance(violations, str):
        try:
            violations = json.loads(violations)
        except Exception:
            violations = []
    return violations if isinstance(violations, list) else []


def _retry_delay(exc: Exception, attempt: int) -> float:
    """Back-off delay, honouring 'try again in Xs' hints from rate-limit responses."""
    delay = min(2.0 ** attempt, LLM_MAX_BACKOFF)
    match = re.search(r"(?:(?:try again|retry) in\s+|retryDelay=|retry-after=)(?:(\d+)m)?\s*([\d.]+)s", str(exc), re.I)
    if match:
        hinted = float(match.group(2)) + 60.0 * float(match.group(1) or 0) + 0.5
        delay = max(delay, min(hinted, LLM_MAX_BACKOFF))
    return delay


class AgentLoop:
    """Manages the controlled LLM tool-calling loop."""

    def __init__(
        self,
        repo_url: str,
        repo_dir: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        max_iterations: int = 30,
        verbose: bool = True,
        timeout_seconds: float = 900.0,
        provider: Optional[str] = None,
    ):
        self.repo_url = repo_url
        self.repo_dir = repo_dir
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.deadline = time.monotonic() + timeout_seconds
        self.trace: List[Dict[str, Any]] = []
        self.successful_llm_calls = 0
        self.tool_call_count = 0
        self._tool_cache: Dict[str, str] = {}
        self.read_paths: set = set()
        self.workflows_inspected = False
        self.coverage_nudged = False
        self.high_risk_leads: List[tuple] = []
        self.secret_leads: List[tuple] = []  # (path, line) of credential-looking search hits

        self.provider = resolve_provider(provider, api_key)
        cfg = PROVIDERS[self.provider]
        self.key_env = key_env_label(self.provider)
        self.exclude_markers = tuple(cfg.get("exclude_markers", ()))

        self.api_key = api_key or get_api_key(self.provider)
        if not self.api_key:
            raise ValueError(
                f"Missing {self.key_env}. Please provide it via --api-key or set it in your .env file or environment."
            )

        requested_model = model or default_model_for(self.provider)
        self.client = self._build_client()

        # Verify or resolve available model
        self.model = requested_model
        self.model_fallbacks: List[str] = [m for m in cfg["preference"] if m != requested_model]
        try:
            available_models = [m.id for m in self.client.models.list().data]
            self.model, self.model_fallbacks = plan_models(
                requested_model,
                available_models,
                cfg["preference"],
                self.exclude_markers,
                bool(cfg.get("trust_requested")),
            )
        except Exception:
            pass

        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Target Repository URL: {self.repo_url}\n"
                    "The repository has been cloned locally (read-only inspection). Begin by "
                    "mapping the repository with list_files, then let what you observe decide "
                    "which tools to call next. Keep short running notes of confirmed findings "
                    "(file:line + evidence) in your reasoning so they survive context trimming."
                ),
            },
        ]

    # ------------------------------------------------------------------ helpers
    def _build_client(self) -> Any:
        """Construct the provider client. Retries are handled by _chat (max_retries=0 for SDKs)."""
        cfg = PROVIDERS[self.provider]
        base_url = (os.getenv(cfg.get("base_url_env", "")) or "").strip() if cfg.get("base_url_env") else ""
        if self.provider == "groq":
            if Groq is None:
                raise ImportError("The 'groq' package is required. Install it using 'pip install groq'.")
            return Groq(api_key=self.api_key, timeout=LLM_REQUEST_TIMEOUT, max_retries=0)
        if self.provider == "gemini":
            return GeminiClient(api_key=self.api_key, timeout=LLM_REQUEST_TIMEOUT)
        if self.provider == "openai":
            return OpenAIClient(api_key=self.api_key, timeout=LLM_REQUEST_TIMEOUT, base_url=base_url or OPENAI_BASE_URL)
        return AnthropicClient(api_key=self.api_key, timeout=max(LLM_REQUEST_TIMEOUT, 120.0), base_url=base_url or ANTHROPIC_BASE_URL)

    def _log(self, message: str) -> None:
        if self.verbose:
            print(sanitize_evidence(message), flush=True)

    def _time_left(self) -> float:
        return self.deadline - time.monotonic()

    def _record(self, **entry: Any) -> None:
        self.trace.append(entry)

    def _compact_history(self) -> None:
        """Bound context growth: keep the latest tool results, shorten older ones."""
        tool_idx = [i for i, m in enumerate(self.messages) if m.get("role") == "tool"]
        recent = set(tool_idx[-KEEP_RECENT_TOOL_RESULTS:])
        for i in tool_idx:
            content = self.messages[i].get("content") or ""
            if i in recent:
                limit = RECENT_TOOL_RESULT_CHARS
            else:
                limit = OLD_TOOL_RESULT_CHARS
            if len(content) > limit and not content.endswith(TRUNCATION_MARKER):
                self.messages[i]["content"] = content[:limit] + TRUNCATION_MARKER

    def _shrink_context(self) -> None:
        """Emergency trim after a 'request too large' error: cut every tool result hard."""
        for m in self.messages:
            if m.get("role") == "tool":
                content = m.get("content") or ""
                if len(content) > 400 and not content.endswith(TRUNCATION_MARKER):
                    m["content"] = content[:400] + TRUNCATION_MARKER

    def _chat(self, **kwargs: Any) -> Any:
        """One chat completion with retry/back-off. Raises LLMError when exhausted."""
        last_exc: Optional[Exception] = None
        overloads = 0  # consecutive "overloaded / high demand" errors on the current model
        for attempt in range(1, LLM_MAX_ATTEMPTS + 1):
            if self._time_left() <= 0:
                raise LLMError("Audit wall-clock timeout reached before the LLM responded.")
            try:
                response = self.client.chat.completions.create(
                    model=self.model, temperature=0.1, **kwargs
                )
                if not getattr(response, "choices", None):
                    raise LLMError("LLM returned an empty response (no choices).")
                self.successful_llm_calls += 1
                return response
            except LLMError as exc:
                last_exc = exc
            except Exception as exc:  # provider SDK errors
                last_exc = exc
                status = getattr(exc, "status_code", None)
                text = str(exc).lower()
                too_large = status == 413 or any(
                    marker in text
                    for marker in (
                        "request too large", "reduce your message size", "context_length_exceeded",
                        "maximum context length", "prompt is too long", "too many tokens",
                    )
                )
                fatal_auth = status in (401, 403)
                fatal_bad = status == 404 or (
                    status == 400
                    and not too_large
                    and "tool_use_failed" not in text
                    and "failed_generation" not in text
                )
                billing = "insufficient_quota" in text or "credit balance is too low" in text
                daily_quota = status == 429 and ("perday" in text or "per day" in text)
                if billing:
                    raise LLMError(
                        f"Provider quota/billing problem: {str(exc)[:200]}. Add credit or use another API key."
                    ) from exc
                if (status == 404 and "model" in text or daily_quota) and self.model_fallbacks:
                    old, self.model = self.model, self.model_fallbacks.pop(0)
                    self._log(f"[!] Model '{old}' unavailable or out of daily quota; switching to '{self.model}'.")
                    continue
                if daily_quota:
                    raise LLMError(f"Daily API quota exhausted for model '{self.model}' and no fallback models remain.") from exc
                overloaded = status in (503, 529) or any(
                    marker in text for marker in ("high demand", "overloaded", "currently unavailable")
                )
                overloads = overloads + 1 if overloaded else 0
                if overloaded and overloads >= 2 and self.model_fallbacks:
                    # Overload is usually per model: rotate to another one, and queue this one to be retried later.
                    old, self.model = self.model, self.model_fallbacks.pop(0)
                    self.model_fallbacks.append(old)
                    overloads = 0
                    self._log(f"[!] Model '{old}' is overloaded; trying '{self.model}' (will revisit '{old}' later).")
                    continue
                if too_large:
                    self._shrink_context()
                if fatal_auth:
                    raise LLMError(f"LLM authentication failed (HTTP {status}); check {self.key_env}.") from exc
                if fatal_bad:
                    raise LLMError(f"LLM request rejected (HTTP {status}): {str(exc)[:300]}") from exc
            if attempt < LLM_MAX_ATTEMPTS:
                delay = min(_retry_delay(last_exc, attempt), max(self._time_left(), 0.0))
                self._log(f"[!] LLM call failed ({str(last_exc)[:160]}). Retry {attempt}/{LLM_MAX_ATTEMPTS - 1} in {delay:.0f}s...")
                time.sleep(delay)
        raise LLMError(f"LLM request failed after {LLM_MAX_ATTEMPTS} attempts: {str(last_exc)[:300]}")

    def _execute_tool(self, name: str, args: Dict[str, Any]) -> str:
        key = f"{name}:{json.dumps(args, sort_keys=True)}"
        if key in self._tool_cache:
            return "[Duplicate call: identical to an earlier call, whose result is above. Try a different query or file.]"
        if name == "inspect_workflows":
            self.workflows_inspected = True
        elif name == "read_file" and isinstance(args.get("path"), str):
            self.read_paths.add(re.sub(r"^(\./|/)+", "", args["path"].replace("\\", "/")).lower())
        output = dispatch_tool_call(repo_dir=self.repo_dir, tool_name=name, arguments=args)
        if name == "inspect_workflows":
            self.high_risk_leads = [
                (m.group(1), int(m.group(2)), m.group(3))
                for m in re.finditer(
                    r"^\s+(\S+?):(\d+): \[(PR head code[^\]]*|SECRET printed[^\]]*|SCRIPT-INJECTION[^\]]*|WRITE-ALL[^\]]*)\]",
                    output,
                    re.M,
                )
            ]
        if name == "search_code":
            self._collect_secret_leads(output)
        self._tool_cache[key] = output
        return output

    # ---------------------------------------------------------------- coverage
    SKIP_DIRS = {".git", "node_modules", "venv", ".venv", "vendor", "third_party", "dist", "build", "__pycache__"}

    def _unreviewed_key_files(self, limit: int = 10) -> List[str]:
        """Important DevOps files (CI, Docker, manifests, ignore/env files) the agent has not opened."""
        base = Path(self.repo_dir)
        missing: List[str] = []
        try:
            for root, dirs, files in os.walk(base):
                rel_root = Path(root).relative_to(base)
                if len(rel_root.parts) >= 3:
                    dirs.clear()
                    continue
                dirs[:] = [d for d in dirs if d not in self.SKIP_DIRS and d.lower() not in {"tests", "test", "docs"}]
                for f in sorted(files):
                    rel = (rel_root / f).as_posix() if rel_root.parts else f
                    low = rel.lower()
                    is_workflow = low.startswith(".github/workflows/") and low.endswith((".yml", ".yaml"))
                    if is_workflow:
                        if not self.workflows_inspected and low not in self.read_paths:
                            missing.append(rel)
                    elif is_key_file(rel, include_lockfiles=False) and low not in self.read_paths:
                        missing.append(rel)
            # .github/workflows lives two levels deep and is not skipped above, but hidden dir walk covers it
        except Exception:
            return []
        missing.sort(key=lambda path: (path.count("/"), path))  # root-level files first
        return missing[:limit]

    def _absent_standard_items(self) -> List[str]:
        """Facts about standard repo items that are absent (the model decides whether each is a defect)."""
        base = Path(self.repo_dir)
        names = set()
        dirs_seen = set()
        try:
            for root, dirs, files in os.walk(base):
                rel_root = Path(root).relative_to(base)
                dirs[:] = [d for d in dirs if d not in self.SKIP_DIRS]
                if len(rel_root.parts) >= 3:
                    dirs.clear()
                    continue
                dirs_seen.update(d.lower() for d in dirs)
                for f in files:
                    names.add((rel_root / f).as_posix().lower() if rel_root.parts else f.lower())
        except Exception:
            return []
        absent: List[str] = []
        if ".gitignore" not in names:
            absent.append("no root .gitignore")
        if not any(n.startswith("readme") for n in names):
            absent.append("no README at the repository root")
        has_tests = (
            bool(dirs_seen & {"tests", "test", "__tests__", "spec", "specs", "e2e", "integration-tests"})
            or "cargo.toml" in names  # Rust keeps unit tests inline (#[test]); cannot tell from the file tree
            or any(
                re.search(
                    r"(^|/)(test_[^/]*\.py|[^/]*_test\.(py|go|rb|exs?|rs|cpp|cc)|[^/]*(Test|Tests|IT)\.(java|kt|cs|php|scala)"
                    r"|[^/]*_spec\.rb|[^/]*\.(test|spec)\.[jt]sx?)$",
                    n,
                    re.I,
                )
                for n in names
            )
        )
        if not has_tests:
            absent.append("no test files or test directory found")
        has_ci = (
            any(n.startswith(".github/workflows/") for n in names)
            or bool(names & {".gitlab-ci.yml", ".travis.yml", "jenkinsfile", "azure-pipelines.yml", "bitbucket-pipelines.yml"})
            or ".circleci" in dirs_seen
        )
        if not has_ci:
            absent.append("no CI configuration found")
        return absent

    def _collect_secret_leads(self, search_output: str) -> None:
        """Remember likely credentials surfaced by search_code (outside tests/docs/examples)."""
        for match in re.finditer(r"^(\S+?):(\d+): (.*)$", search_output, re.M):
            path, line_no, text = match.group(1), int(match.group(2)), match.group(3)
            parts = {part.lower() for part in path.replace("\\", "/").split("/")[:-1]}
            name = path.rsplit("/", 1)[-1].lower()
            if parts & FIXTURE_DIR_NAMES or name.endswith((".md", ".rst", ".txt")) or ".example" in name or ".sample" in name:
                continue
            hit = HARD_SECRET_LEAD.search(text)
            if not hit:
                assigned = ASSIGNED_SECRET_LEAD.search(text)
                hit = assigned if assigned and not PLACEHOLDER_VALUE.search(assigned.group(1)) else None
            if hit and (path, line_no) not in self.secret_leads:
                self.secret_leads.append((path, line_no))

    def _uncovered_secret_leads(self, violations: Optional[List[Any]]) -> List[str]:
        """Credential-looking search hits in files that no submitted finding mentions."""
        cited = {str(v.get("file") or "").lower() for v in violations or [] if isinstance(v, dict)}
        out, seen = [], set()
        for path, line_no in self.secret_leads:
            if path.lower() in cited or path in seen:
                continue
            seen.add(path)
            out.append(f"{path}:{line_no}")
        return out[:6]

    def _uncovered_leads(self, violations: Optional[List[Any]]) -> List[str]:
        """High-risk workflow leads whose file no submitted violation mentions."""
        cited = set()
        for v in violations or []:
            if isinstance(v, dict):
                cited.add(str(v.get("file") or "").lower())
                cited.update(re.findall(r"[\w./-]+\.ya?ml", str(v.get("evidence") or "").lower()))
        out, seen = [], set()
        for path, line, tag in self.high_risk_leads:
            if path.lower() in cited or path in seen:
                continue
            seen.add(path)
            out.append(f"{path}:{line} [{tag.split(' ')[0]}]")
        return out[:6]

    def _coverage_gap_message(self, violations: Optional[List[Any]] = None) -> Optional[str]:
        if self.coverage_nudged:
            return None
        missing = self._unreviewed_key_files()
        absent = self._absent_standard_items()
        leads = self._uncovered_leads(violations)
        secrets = self._uncovered_secret_leads(violations)
        if not missing and not absent and not leads and not secrets:
            return None
        self.coverage_nudged = True
        parts = []
        if secrets:
            parts.append(
                "search_code surfaced text that looks like credentials in files your findings do not mention: "
                + ", ".join(secrets)
                + ". Open each, and report it as a security finding if it is a real credential (not a placeholder, "
                  "test fixture or environment lookup); never copy the secret value")
        if leads:
            parts.append(
                "inspect_workflows flagged high-risk leads in files your findings do not mention: "
                + ", ".join(leads)
                + ". Read them and report each one that is a genuine defect (or drop it if it is not)"
            )
        if missing:
            parts.append(
                "these key DevOps files exist but you have not opened them: " + ", ".join(missing)
                + ". Read the ones that matter (CI/CD, Docker, dependency, config files) and check them for defects; "
                  "skip any that are clearly irrelevant"
            )
        if absent:
            parts.append(
                "facts from the file tree: " + "; ".join(absent)
                + ". Report each only if it is a genuine defect for this repository (check for equivalents first)"
            )
        return "Before reporting: " + ". Also, ".join(parts) + ". Then submit again."

    # --------------------------------------------------------------------- run
    def run(self) -> AuditReport:
        """Execute the autonomous tool-calling loop and return a validated AuditReport."""
        self._log(f"\n[+] Starting DevOps Audit Agent on: {self.repo_url}")
        self._log(f"[+] Provider: {self.provider} | Model: {self.model} | Max Iterations: {self.max_iterations}")
        self._log("[+] Framework: Native Python LLM Tool Loop\n")

        iteration = 0
        nudges = 0
        final_json_candidate: Optional[Dict[str, Any]] = None
        llm_error: Optional[LLMError] = None

        while iteration < self.max_iterations and self._time_left() > 0:
            iteration += 1
            self._log(f"--- Iteration {iteration}/{self.max_iterations} ---")

            if iteration == self.max_iterations - 1 and self.tool_call_count:
                self.messages.append({
                    "role": "user",
                    "content": "Budget notice: only one investigation step remains. Verify anything still uncertain, "
                               "then call submit_final_report.",
                })

            self._compact_history()
            try:
                response = self._chat(
                    messages=self.messages, tools=TOOL_DEFINITIONS, tool_choice="auto"
                )
            except LLMError as exc:
                llm_error = exc
                self._log(f"[!] LLM unavailable: {exc}")
                break

            message = response.choices[0].message
            thought = message.content or ""
            if thought.strip():
                self._log(f"\n[Agent Reasoning]\n{thought.strip()}\n")

            tool_calls = getattr(message, "tool_calls", None)

            assistant_msg: Dict[str, Any] = {"role": "assistant", "content": thought}
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ]
            self.messages.append(assistant_msg)

            # No tool calls: model believes it is done (or answered in plain text)
            if not tool_calls:
                parsed = extract_json_from_text(thought)
                if parsed and "violations" in parsed and self.tool_call_count >= MIN_TOOL_CALLS_BEFORE_REPORT:
                    gap = self._coverage_gap_message(parsed.get("violations"))
                    if gap and iteration < self.max_iterations - 1:
                        self._log("[Agent] Coverage gap: asking the model to review unopened key files.")
                        self.messages.append({"role": "user", "content": gap})
                        continue
                    final_json_candidate = parsed
                    break
                if self.tool_call_count < MIN_TOOL_CALLS_BEFORE_REPORT and nudges < 3:
                    nudges += 1
                    self._log("[Agent] Too little investigation so far; asking the model to continue with tools.")
                    self.messages.append({
                        "role": "user",
                        "content": f"You have only run {self.tool_call_count} tool call(s). Do not report from "
                                   "assumptions: keep investigating with tools (CI workflows, dependency manifests, "
                                   "security searches, tests, config) and submit via submit_final_report when done.",
                    })
                    continue
                self._log("[Agent] No further tool calls requested. Moving to final report synthesis...")
                if parsed and "violations" in parsed:
                    final_json_candidate = parsed
                break

            # Execute tool calls
            for tc in tool_calls:
                fn_name = tc.function.name
                fn_args_raw = tc.function.arguments or "{}"
                try:
                    fn_args = json.loads(fn_args_raw)
                    if not isinstance(fn_args, dict):
                        raise ValueError("arguments must be a JSON object")
                except Exception as exc:
                    self.messages.append({
                        "role": "tool", "tool_call_id": tc.id, "name": fn_name,
                        "content": f"Error: could not parse tool arguments as a JSON object ({exc}). Retry with valid JSON.",
                    })
                    continue

                if fn_name == "submit_final_report":
                    violations = _violations_from_args(fn_args)
                    if self.tool_call_count < MIN_TOOL_CALLS_BEFORE_REPORT and nudges < 3:
                        nudges += 1
                        self.messages.append({
                            "role": "tool", "tool_call_id": tc.id, "name": fn_name,
                            "content": "Rejected: investigate further with the inspection tools before submitting.",
                        })
                        continue
                    gap = self._coverage_gap_message(violations) if iteration < self.max_iterations - 1 else None
                    if gap:
                        self._log("[Agent] Coverage gap: asking the model to review unopened key files before submitting.")
                        self.messages.append({
                            "role": "tool", "tool_call_id": tc.id, "name": fn_name,
                            "content": "Rejected for now. " + gap,
                        })
                        continue
                    self._log(f"\n[Agent] Submitting final audit report via submit_final_report ({len(violations)} violations)...")
                    self._record(iteration=iteration, event="submit_final_report", violations=len(violations))
                    final_json_candidate = {"repository": self.repo_url, "violations": violations}
                    break

                self.tool_call_count += 1
                self._log(f"[Tool Selection] -> {fn_name}({json.dumps(fn_args)})")
                tool_output = self._execute_tool(fn_name, fn_args)

                lines = tool_output.strip().splitlines()
                preview = lines[0] if lines else "[empty output]"
                if len(lines) > 1:
                    preview += f" ... (+{len(lines) - 1} lines)"
                self._log(f"[Tool Result] <- {preview}\n")

                self._record(
                    iteration=iteration,
                    reasoning=sanitize_evidence(thought.strip())[:1200],
                    tool=fn_name,
                    arguments=fn_args,
                    result_preview=sanitize_evidence(tool_output)[:TRACE_RESULT_CHARS],
                    result_chars=len(tool_output),
                )

                self.messages.append({
                    "role": "tool", "tool_call_id": tc.id, "name": fn_name, "content": tool_output,
                })

            if final_json_candidate:
                break

        if not final_json_candidate:
            final_json_candidate = self._synthesize_report(llm_error)

        return self._finalize(final_json_candidate)

    # ---------------------------------------------------------------- synthesis
    def _synthesize_report(self, earlier_error: Optional[LLMError]) -> Dict[str, Any]:
        """Ask the model for its final report. Raises LLMError rather than inventing one."""
        if self.successful_llm_calls == 0:
            raise earlier_error or LLMError("The LLM never produced a response; no audit was performed.")

        self._log("\n[+] Finalizing Audit: Requesting structured JSON report from Agent...")
        self._compact_history()
        self.messages.append({
            "role": "user",
            "content": (
                "Your investigation is complete. Compile your final DevOps audit report by calling submit_final_report.\n"
                "Every violation needs rule_id, category, severity, file, line, description, evidence, recommendation.\n"
                "Report only defects you verified with tool output; never include secret values; no false positives; "
                "use an empty list if nothing genuine was found."
            ),
        })

        # The wall-clock deadline may have expired; allow a short grace window for the final answer.
        self.deadline = max(self.deadline, time.monotonic() + 120)
        errors: List[str] = []

        try:
            resp = self._chat(
                messages=self.messages,
                tools=TOOL_DEFINITIONS,
                tool_choice={"type": "function", "function": {"name": "submit_final_report"}},
            )
            calls = resp.choices[0].message.tool_calls
            if calls:
                args = json.loads(calls[0].function.arguments or "{}")
                return {"repository": self.repo_url, "violations": _violations_from_args(args)}
            parsed = extract_json_from_text(resp.choices[0].message.content or "")
            if parsed and "violations" in parsed:
                return parsed
        except Exception as exc:
            errors.append(str(exc))
            self._log(f"[!] Forced tool report failed ({str(exc)[:200]}), falling back to JSON mode...")

        try:
            resp = self._chat(messages=self.messages, response_format={"type": "json_object"})
            parsed = extract_json_from_text(resp.choices[0].message.content or "")
            if parsed and "violations" in parsed:
                return parsed
            errors.append("model output contained no 'violations' JSON")
        except Exception as exc:
            errors.append(str(exc))

        raise LLMError(
            "Investigation finished but the model did not return a usable report: " + "; ".join(errors)[:400]
        )

    # ----------------------------------------------------------------- finalize
    def _finalize(self, candidate: Dict[str, Any]) -> AuditReport:
        candidate["repository"] = self.repo_url
        if not isinstance(candidate.get("violations"), list):
            candidate["violations"] = []

        self._record(
            event="raw_submission",
            violations=sanitize_evidence(json.dumps(candidate["violations"], default=str))[:6000],
        )
        valid: List[Violation] = []
        dropped = 0
        used_ids: set = set()
        for item in candidate["violations"]:
            repaired = self._repair_violation(item, used_ids)
            if repaired is None:
                dropped += 1
                self._log(f"[!] Dropped a submitted finding: not an object with a description ({str(item)[:80]}).")
                continue
            try:
                valid.append(Violation(**repaired))
            except Exception as exc:
                dropped += 1
                self._log(f"[!] Dropped finding {repaired.get('rule_id')}: schema validation failed ({str(exc)[:160]}).")
        if dropped:
            self._log(f"[!] {dropped} submitted finding(s) could not be used.")

        report = AuditReport(repository=self.repo_url, violations=valid)
        self._verify_locations(report)
        report.deduplicate()
        self._record(event="final_report", violations=len(report.violations))
        self._log(f"[+] Audit completed successfully. {len(report.violations)} unique violation(s) identified.\n")
        return report

    RULE_PREFIX = {
        "repository_hygiene": "HYG", "dependency_management": "DEP", "code_quality": "QUAL", "testing": "TEST",
        "github_actions_ci": "CI", "security": "SEC", "configuration": "CFG", "collaboration": "COL",
        "build_and_release": "BLD",
    }
    PREFIX_CATEGORY = {
        "HYG": "repository_hygiene", "DEP": "dependency_management", "QUAL": "code_quality", "TEST": "testing",
        "CI": "github_actions_ci", "SEC": "security", "CFG": "configuration", "CONF": "configuration",
        "COL": "collaboration", "BLD": "build_and_release",
    }

    def _repair_violation(self, item: Any, used_ids: set) -> Optional[Dict[str, Any]]:
        """Fill fixable gaps in a model-submitted finding instead of losing a real defect.

        Models (especially small ones) sometimes omit rule_id/evidence/recommendation or give list-valued
        file/line. Aliases are mapped, missing text falls back to the description, and a finding with no
        description at all is rejected (nothing to report).
        """
        if not isinstance(item, dict):
            return None
        description = str(item.get("description") or item.get("summary") or item.get("title") or item.get("message") or "").strip()
        if not description:
            return None
        fixed: Dict[str, Any] = dict(item)
        fixed["description"] = description
        fixed["evidence"] = str(item.get("evidence") or item.get("details") or item.get("snippet") or description).strip() or description
        fixed["recommendation"] = str(
            item.get("recommendation") or item.get("remediation") or item.get("fix") or item.get("suggestion") or ""
        ).strip() or "Review this issue and apply the standard remediation for it."
        for key in ("file", "line"):
            value = fixed.get(key)
            if isinstance(value, (list, tuple)):
                fixed[key] = value[0] if value else None
        raw_id = str(fixed.get("rule_id") or fixed.get("id") or "").strip()
        category = str(fixed.get("category") or "").strip()
        if not category:
            prefix = raw_id.split("-")[0].upper() if raw_id else ""
            category = self.PREFIX_CATEGORY.get(prefix, "configuration")
            fixed["category"] = category
        if not raw_id:
            canonical = Violation.normalize_category(category)
            prefix = self.RULE_PREFIX.get(canonical, "GEN")
            number = 1
            while f"{prefix}-{number:03d}" in used_ids:
                number += 1
            raw_id = f"{prefix}-{number:03d}"
        fixed["rule_id"] = raw_id
        used_ids.add(raw_id)
        fixed["severity"] = fixed.get("severity") or "medium"
        return {
            k: fixed.get(k)
            for k in ("rule_id", "category", "severity", "file", "line", "description", "evidence", "recommendation")
        }

    INERT_COMMENT_FILES = re.compile(
        r"(^|/)(\.gitignore|\.dockerignore|dockerfile[\w.-]*|requirements[\w.-]*\.txt|[\w.-]+\.(ya?ml|toml|cfg|ini|conf))$",
        re.I,
    )

    # "This version does not exist" claims cannot be verified offline and the model's release knowledge
    # is older than many repositories, so they are treated as unsupported.
    VERSION_EXISTENCE_CLAIM = re.compile(
        r"(non-?existent|does(?:n'?t| not) exist|not (?:yet )?(?:been )?released|unreleased|future (?:version|release)"
        r"|invalid version|fictional|imaginary|not a real (?:version|release))",
        re.I,
    )
    VERSION_WORDS = re.compile(r"(version|release|tag|@v\d|\bv\d+(\.\d+)*\b)", re.I)

    def _cited_line(self, v: Violation) -> Optional[str]:
        """Return the text of the line a violation cites, or None when it cannot be read."""
        if not v.file or v.line is None:
            return None
        try:
            target = _resolve_safe_path(self.repo_dir, v.file)
            if not Path(target).is_file():
                return None
            lines = Path(target).read_text(encoding="utf-8", errors="replace").splitlines()
            return lines[v.line - 1] if 1 <= v.line <= len(lines) else None
        except Exception:
            return None

    def _verify_locations(self, report: AuditReport) -> None:
        """Evidence gate: fix or drop findings the cited file contradicts.

        * line numbers that do not exist in the cited file are set to null;
        * findings that cite a commented-out line of an inert config file (e.g. ``#Pipfile.lock``
          in .gitignore) are dropped (security findings are kept: a commented secret is still leaked);
        * claims about a ``latest`` tag are dropped when the cited line carries an explicit version tag;
        * claims that an action/tool/image version "does not exist" are dropped (not verifiable offline).
        """
        kept: List[Violation] = []
        for v in report.violations:
            claim_text = f"{v.description} {v.evidence}"
            if self.VERSION_EXISTENCE_CLAIM.search(claim_text) and self.VERSION_WORDS.search(claim_text):
                self._log(f"[!] Dropped finding {v.rule_id}: asserts a version does not exist, which cannot be verified offline.")
                continue
            if v.file and v.line is not None:
                text = self._cited_line(v)
                if text is None:
                    v.line = None
                else:
                    stripped = text.strip()
                    claim = f"{v.description} {v.evidence}"
                    if (
                        v.category != "security"
                        and stripped.startswith(("#", "//"))
                        and self.INERT_COMMENT_FILES.search(v.file)
                    ):
                        self._log(f"[!] Dropped finding {v.rule_id}: cited line {v.file}:{v.line} is a comment.")
                        continue
                    if (
                        re.search(r"\blatest\b", claim, re.I)
                        and "latest" not in stripped.lower()
                        and re.search(r"(^|\s)(FROM|image:|uses:)\s*\S*[:@]\S", stripped, re.I)
                    ):
                        self._log(f"[!] Dropped finding {v.rule_id}: claims 'latest' but {v.file}:{v.line} has an explicit tag.")
                        continue
            kept.append(v)
        report.violations = kept

