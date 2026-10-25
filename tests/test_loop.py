"""Unit tests for loop.py: JSON extraction and AgentLoop orchestration with a scripted fake LLM."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import loop as loop_module
from loop import AgentLoop, LLMError, extract_json_from_text
from schemas import AuditReport

REPO = "https://github.com/foo/bar"


def test_extract_json_from_text_direct():
    parsed = extract_json_from_text('{"repository": "https://github.com/foo/bar", "violations": []}')
    assert parsed is not None
    assert parsed["violations"] == []


def test_extract_json_from_markdown_fence():
    text = """Here is your report:
```json
{"repository": "x", "violations": [{"rule_id": "SEC-001"}]}
```
Thanks!"""
    parsed = extract_json_from_text(text)
    assert parsed["violations"][0]["rule_id"] == "SEC-001"


def test_extract_json_invalid():
    assert extract_json_from_text("This has no json") is None
    assert extract_json_from_text("") is None
    assert extract_json_from_text(None) is None


def test_agent_loop_missing_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Missing GROQ_API_KEY"):
        AgentLoop(repo_url=REPO, repo_dir=".", api_key=None, provider="groq")
    with pytest.raises(ValueError, match="Missing GEMINI_API_KEY"):
        AgentLoop(repo_url=REPO, repo_dir=".", api_key=None, provider="gemini")


def test_provider_resolution(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert loop_module.resolve_provider() == "gemini"
    monkeypatch.setenv("LLM_PROVIDER", "Groq")
    assert loop_module.resolve_provider() == "groq"
    assert loop_module.resolve_provider("gemini") == "gemini"
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        loop_module.resolve_provider("cohere")


# ----------------------------------------------------------------- fake LLM
def tool_call(name, args=None, call_id="c1"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(args or {})),
    )


def response(content="", tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


VIOLATION = {
    "rule_id": "HYG-001",
    "category": "repository_hygiene",
    "severity": "medium",
    "file": "README.md",
    "line": 999,
    "description": "Example",
    "evidence": "Example evidence",
    "recommendation": "Fix it",
}


def make_agent(mock_groq_class, script, repo_dir, coverage=False, **kwargs):
    client = MagicMock()
    client.models.list.side_effect = Exception("no network")
    client.chat.completions.create.side_effect = script
    mock_groq_class.return_value = client
    agent = AgentLoop(repo_url=REPO, repo_dir=str(repo_dir), api_key="test-key", verbose=False, provider="groq", **kwargs)
    agent.coverage_nudged = not coverage  # most tests exercise other behaviour; coverage tests opt in
    return agent, client


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "README.md").write_text("# hi\n", encoding="utf-8")
    return tmp_path


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(loop_module.time, "sleep", lambda *_: None)


@patch("loop.Groq")
def test_full_cycle_uses_llm_selected_tools_and_submits(mock_groq, repo):
    script = [
        response("Map the repo.", [tool_call("list_files", call_id="a")]),
        response("Read README.", [tool_call("read_file", {"path": "README.md"}, call_id="b")]),
        response("Search secrets.", [tool_call("search_code", {"pattern": "secret"}, call_id="c")]),
        response("History.", [tool_call("git_log", call_id="d")]),
        response("Done.", [tool_call("submit_final_report", {"violations": [VIOLATION]}, call_id="e")]),
    ]
    agent, client = make_agent(mock_groq, script, repo, max_iterations=10)
    report = agent.run()

    assert isinstance(report, AuditReport)
    assert report.repository == REPO
    assert [s.get("tool") for s in agent.trace if s.get("tool")] == [
        "list_files", "read_file", "search_code", "git_log",
    ]
    # Follow-up calls received earlier tool results in their context (observe -> reason -> act)
    last_messages = client.chat.completions.create.call_args_list[-1].kwargs["messages"]
    assert any(m.get("role") == "tool" for m in last_messages)
    # Hallucinated line number (README has 1 line) is nulled out by the verification gate
    assert report.violations[0].line is None


@patch("loop.Groq")
def test_early_stop_is_pushed_back_to_investigate(mock_groq, repo):
    script = [
        response("Looks fine.", None),  # lazy: no tools used at all
        response("ok", [tool_call("list_files", call_id="a")]),
        response("ok", [tool_call("read_file", {"path": "README.md"}, call_id="b")]),
        response("ok", [tool_call("search_code", {"pattern": "x"}, call_id="c")]),
        response("ok", [tool_call("git_log", call_id="d")]),
        response("done", [tool_call("submit_final_report", {"violations": []}, call_id="e")]),
    ]
    agent, _ = make_agent(mock_groq, script, repo, max_iterations=10)
    report = agent.run()
    assert report.violations == []
    assert agent.tool_call_count == 4


@patch("loop.Groq")
def test_transient_api_errors_are_retried(mock_groq, repo):
    script = [
        Exception("503 service unavailable"),
        Exception("Rate limit reached. Please try again in 2s."),
        response("ok", [tool_call("list_files", call_id="a")]),
        response("ok", [tool_call("read_file", {"path": "README.md"}, call_id="b")]),
        response("ok", [tool_call("search_code", {"pattern": "x"}, call_id="c")]),
        response("ok", [tool_call("git_log", call_id="d")]),
        response("done", [tool_call("submit_final_report", {"violations": []}, call_id="e")]),
    ]
    agent, client = make_agent(mock_groq, script, repo, max_iterations=10)
    agent.run()
    assert client.chat.completions.create.call_count == 7


@patch("loop.Groq")
def test_unreachable_llm_raises_instead_of_empty_report(mock_groq, repo):
    agent, _ = make_agent(mock_groq, [Exception("connection reset")] * 20, repo, max_iterations=3)
    with pytest.raises(LLMError):
        agent.run()


@patch("loop.Groq")
def test_auth_failure_is_fatal_immediately(mock_groq, repo):
    err = Exception("invalid api key")
    err.status_code = 401
    agent, client = make_agent(mock_groq, [err] * 5, repo, max_iterations=3)
    with pytest.raises(LLMError, match="authentication"):
        agent.run()
    assert client.chat.completions.create.call_count == 1


@patch("loop.Groq")
def test_malformed_violations_dropped_and_secrets_redacted(mock_groq, repo):
    good = dict(VIOLATION, line=1, evidence='API_KEY = "sk-live-abcdef123456"')
    bad = {"rule_id": "X"}  # missing required fields
    script = [
        response("ok", [tool_call("list_files", call_id="a")]),
        response("ok", [tool_call("read_file", {"path": "README.md"}, call_id="b")]),
        response("ok", [tool_call("search_code", {"pattern": "x"}, call_id="c")]),
        response("ok", [tool_call("git_log", call_id="d")]),
        response("done", [tool_call("submit_final_report", {"violations": [good, bad]}, call_id="e")]),
    ]
    agent, _ = make_agent(mock_groq, script, repo, max_iterations=10)
    report = agent.run()
    assert len(report.violations) == 1
    assert "sk-live" not in report.violations[0].evidence
    assert report.violations[0].line == 1


@patch("loop.Groq")
def test_history_is_compacted(mock_groq, repo):
    agent, _ = make_agent(mock_groq, [], repo)
    for i in range(8):
        agent.messages.append({"role": "tool", "tool_call_id": str(i), "name": "x", "content": "A" * 5000})
    agent._compact_history()
    tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
    assert len(tool_msgs[0]["content"]) < 1000
    assert len(tool_msgs[-1]["content"]) > 3000


@patch("loop.Groq")
def test_duplicate_tool_call_is_flagged(mock_groq, repo):
    agent, _ = make_agent(mock_groq, [], repo)
    first = agent._execute_tool("list_files", {})
    second = agent._execute_tool("list_files", {})
    assert "README.md" in first
    assert "Duplicate call" in second


@patch("loop.Groq")
def test_daily_quota_switches_to_fallback_model(mock_groq, repo):
    err = Exception("quota exceeded [quota=GenerateRequestsPerDayPerProjectPerModel-FreeTier; retryDelay=68062s]")
    err.status_code = 429
    agent, client = make_agent(mock_groq, [err, response("ok", None)], repo, max_iterations=1)
    agent.model_fallbacks = ["backup-model"]
    agent.model = "primary-model"
    agent._chat(messages=agent.messages)
    assert agent.model == "backup-model"
    assert client.chat.completions.create.call_args.kwargs["model"] == "backup-model"


@patch("loop.Groq")
def test_daily_quota_without_fallback_fails_fast(mock_groq, repo):
    err = Exception("quota exceeded [quota=GenerateRequestsPerDayPerProjectPerModel-FreeTier]")
    err.status_code = 429
    agent, client = make_agent(mock_groq, [err] * 10, repo, max_iterations=1)
    agent.model_fallbacks = []
    with pytest.raises(LLMError, match="Daily API quota"):
        agent._chat(messages=agent.messages)
    assert client.chat.completions.create.call_count == 1


def test_retry_delay_reads_retrydelay_hint():
    assert loop_module._retry_delay(Exception("x [retryDelay=12s]"), 1) >= 12


def test_provider_autodetect_from_available_key(monkeypatch):
    for name in ("LLM_PROVIDER", "GROQ_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    # Only a Groq key available -> groq
    monkeypatch.setenv("GROQ_API_KEY", "gsk_x")
    assert loop_module.resolve_provider() == "groq"
    # Only a Gemini key available -> gemini
    monkeypatch.delenv("GROQ_API_KEY")
    monkeypatch.setenv("GEMINI_API_KEY", "g-x")
    assert loop_module.resolve_provider() == "gemini"
    # Both -> gemini default; neither -> gemini default
    monkeypatch.setenv("GROQ_API_KEY", "gsk_x")
    assert loop_module.resolve_provider() == "gemini"
    monkeypatch.delenv("GROQ_API_KEY")
    monkeypatch.delenv("GEMINI_API_KEY")
    assert loop_module.resolve_provider() == "gemini"


def test_provider_autodetect_from_api_key_argument(monkeypatch):
    for name in ("LLM_PROVIDER", "GROQ_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert loop_module.resolve_provider(api_key="gsk_abc123") == "groq"
    assert loop_module.resolve_provider(api_key="AIzaSyABC") == "gemini"
    # explicit provider always wins
    assert loop_module.resolve_provider("gemini", api_key="gsk_abc123") == "gemini"


def test_configured_provider_without_key_falls_back_to_other(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_x")
    assert loop_module.resolve_provider() == "groq"
    # but a configured provider that HAS its key is respected
    monkeypatch.setenv("GEMINI_API_KEY", "g-x")
    assert loop_module.resolve_provider() == "gemini"


def test_plan_models_skips_non_chat_models_and_unlisted_preferences():
    available = ["gemini-2.5-flash-preview-tts", "gemini-3-pro-image", "gemini-4-argon", "gemini-3-flash-preview",
                 "gemma-4-31b-it", "lyria-3.5", "gemini-9-flash"]
    pref = ["gemini-3.5-flash", "gemini-3-flash-preview"]
    model, fallbacks = loop_module.plan_models("gemini-3.5-flash", available, pref)
    assert model == "gemini-3-flash-preview"  # preferred+listed beats everything else
    assert fallbacks == ["gemini-9-flash", "gemini-4-argon"]  # flash first, no tts/image/gemma/lyria


def test_plan_models_never_picks_non_chat_even_if_only_option():
    model, fallbacks = loop_module.plan_models("x", ["whisper-large-v3", "gemini-x-tts"], ["x"])
    assert fallbacks == []  # nothing chat-capable; falls back to listing order, loop will surface the API error
    assert model in ("whisper-large-v3", "gemini-x-tts")


def test_plan_models_without_listing_uses_preferences():
    assert loop_module.plan_models("a", [], ["a", "b", "c"]) == ("a", ["b", "c"])
    assert loop_module.plan_models("z", ["a", "b"], ["a", "b"]) == ("a", ["b"])


@patch("loop.Groq")
def test_coverage_gap_lists_unopened_key_files_once(mock_groq, tmp_path):
    (tmp_path / "Dockerfile").write_text("FROM python:3\n", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("flask\n", encoding="utf-8")
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents=True)
    (wf / "ci.yml").write_text("on: push\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "requirements.txt").write_text("pytest\n", encoding="utf-8")
    agent, _ = make_agent(mock_groq, [], tmp_path, coverage=True)
    missing = agent._unreviewed_key_files()
    assert "Dockerfile" in missing and "requirements.txt" in missing and ".github/workflows/ci.yml" in missing
    assert "tests/requirements.txt" not in missing

    agent._execute_tool("read_file", {"path": "Dockerfile"})
    agent._execute_tool("inspect_workflows", {})
    assert agent._unreviewed_key_files() == ["requirements.txt"]

    first = agent._coverage_gap_message()
    assert first and "requirements.txt" in first
    assert agent._coverage_gap_message() is None  # only nudges once


@patch("loop.Groq")
def test_submit_rejected_once_for_coverage_gap_then_accepted(mock_groq, tmp_path):
    (tmp_path / "Dockerfile").write_text("FROM python:3\n", encoding="utf-8")
    script = [
        response("a", [tool_call("list_files", call_id="a")]),
        response("b", [tool_call("git_log", call_id="b")]),
        response("c", [tool_call("search_code", {"pattern": "x"}, call_id="c")]),
        response("d", [tool_call("git_status", call_id="d")]),
        response("done", [tool_call("submit_final_report", {"violations": []}, call_id="e")]),
        response("reading", [tool_call("read_file", {"path": "Dockerfile"}, call_id="f")]),
        response("done", [tool_call("submit_final_report", {"violations": []}, call_id="g")]),
    ]
    agent, _ = make_agent(mock_groq, script, tmp_path, coverage=True, max_iterations=12)
    report = agent.run()
    assert report.violations == []
    assert "dockerfile" in agent.read_paths
    assert agent.coverage_nudged is True


def _violation(**over):
    base = dict(VIOLATION, line=1, file="x.cfg")
    base.update(over)
    return base


def _finalize(agent, *violations):
    return agent._finalize({"violations": list(violations)})


@patch("loop.Groq")
def test_gate_drops_findings_citing_commented_out_lines(mock_groq, tmp_path):
    (tmp_path / ".gitignore").write_text("*.pyc\n#Pipfile.lock\n", encoding="utf-8")
    (tmp_path / "ci.yml").write_text("a: 1\n# secret_token: abc\n", encoding="utf-8")
    agent, _ = make_agent(mock_groq, [], tmp_path)
    report = _finalize(
        agent,
        _violation(file=".gitignore", line=2, description="Pipfile.lock is ignored", evidence="#Pipfile.lock"),
        _violation(file=".gitignore", line=1, description="real", evidence="*.pyc", rule_id="HYG-002"),
        # a commented-out SECRET is still a leaked secret, so security findings are kept
        _violation(file="ci.yml", line=2, category="security", description="secret", evidence="token", rule_id="SEC-001"),
    )
    assert sorted(v.rule_id for v in report.violations) == ["HYG-002", "SEC-001"]


@patch("loop.Groq")
def test_gate_drops_latest_claim_when_line_has_explicit_tag(mock_groq, tmp_path):
    (tmp_path / "Dockerfile").write_text("FROM node:18\nFROM python:latest\nFROM debian\n", encoding="utf-8")
    agent, _ = make_agent(mock_groq, [], tmp_path)
    report = _finalize(
        agent,
        _violation(file="Dockerfile", line=1, description="Uses latest tag", evidence="FROM node:18", rule_id="BLD-001"),
        _violation(file="Dockerfile", line=2, description="Uses latest tag", evidence="FROM python:latest", rule_id="BLD-002"),
    )
    assert [v.rule_id for v in report.violations] == ["BLD-002"]


@patch("loop.Groq")
def test_absent_standard_items_reported_as_facts(mock_groq, tmp_path):
    agent, _ = make_agent(mock_groq, [], tmp_path, coverage=True)
    assert set(agent._absent_standard_items()) == {
        "no root .gitignore", "no README at the repository root",
        "no test files or test directory found", "no CI configuration found",
    }
    (tmp_path / ".gitignore").write_text("x\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("x\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents=True)
    (wf / "ci.yml").write_text("on: push\n", encoding="utf-8")
    assert agent._absent_standard_items() == []
    msg_agent, _ = make_agent(mock_groq, [], tmp_path, coverage=True)
    msg_agent.read_paths.update({".gitignore", ".github/workflows/ci.yml"})
    assert msg_agent._coverage_gap_message() is None  # nothing unopened and nothing absent


@patch("loop.Groq")
def test_nudge_mentions_uncovered_high_risk_workflow_leads(mock_groq, tmp_path):
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents=True)
    (wf / "toc.yml").write_text(
        "on:\n  pull_request_target:\njobs:\n  a:\n    steps:\n      - uses: actions/checkout@v4\n"
        "        with:\n          ref: ${{ github.event.pull_request.head.sha }}\n",
        encoding="utf-8",
    )
    (tmp_path / ".gitignore").write_text("x\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("x\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    agent, _ = make_agent(mock_groq, [], tmp_path, coverage=True)
    agent._execute_tool("inspect_workflows", {})
    agent.read_paths.add(".gitignore")
    assert agent.high_risk_leads and agent.high_risk_leads[0][0] == ".github/workflows/toc.yml"

    msg = agent._coverage_gap_message([{"file": "somewhere/else.py", "evidence": "x"}])
    assert msg and "toc.yml" in msg and "high-risk leads" in msg

    agent.coverage_nudged = False
    assert agent._coverage_gap_message([{"file": ".github/workflows/toc.yml", "evidence": "x"}]) is None


@patch("loop.Groq")
def test_gate_drops_unverifiable_version_does_not_exist_claims(mock_groq, tmp_path):
    (tmp_path / "ci.yml").write_text("uses: actions/checkout@v7\nruns-on: ubuntu-latest\n", encoding="utf-8")
    agent, _ = make_agent(mock_groq, [], tmp_path)
    report = _finalize(
        agent,
        _violation(file="ci.yml", line=1, category="github_actions_ci", rule_id="CI-001",
                   description="Use of non-existent or future versions for GitHub Actions", evidence="uses: actions/checkout@v7"),
        _violation(file="ci.yml", line=1, category="github_actions_ci", rule_id="CI-002",
                   description="Workflow file does not exist for deploys", evidence="no deploy.yml"),
        _violation(file="ci.yml", line=2, category="github_actions_ci", rule_id="CI-003",
                   description="Action pinned to mutable tag", evidence="uses: x@main"),
    )
    assert sorted(v.rule_id for v in report.violations) == ["CI-002", "CI-003"]


@patch("loop.Groq")
def test_incomplete_findings_are_repaired_not_lost(mock_groq, tmp_path):
    (tmp_path / "main.tf").write_text('resource "aws_s3_bucket" "b" {}\n', encoding="utf-8")
    agent, _ = make_agent(mock_groq, [], tmp_path)
    report = agent._finalize({"violations": [
        # no rule_id / evidence / recommendation, list-valued file and line, alias field names
        {"category": "security", "severity": "high", "file": ["main.tf", "x.tf"], "line": [1],
         "summary": "S3 bucket is public"},
        # no category: inferred from the rule id prefix
        {"rule_id": "CI-007", "description": "Tests never run", "evidence": "no pytest step",
         "recommendation": "Add a test job", "severity": "weird"},
        # nothing to report: dropped
        {"rule_id": "X-1", "evidence": "only evidence"},
        "not even an object",
    ]})
    by_id = {v.rule_id: v for v in report.violations}
    assert set(by_id) == {"SEC-001", "CI-007"}
    sec = by_id["SEC-001"]
    assert sec.file == "main.tf" and sec.line == 1 and sec.evidence == "S3 bucket is public"
    assert sec.recommendation and sec.category == "security"
    assert by_id["CI-007"].category == "github_actions_ci" and by_id["CI-007"].severity == "medium"
    raw = [s for s in agent.trace if s.get("event") == "raw_submission"]
    assert raw and "S3 bucket is public" in raw[0]["violations"]


@patch("loop.Groq")
def test_generated_rule_ids_do_not_collide(mock_groq, tmp_path):
    agent, _ = make_agent(mock_groq, [], tmp_path)
    report = agent._finalize({"violations": [
        {"category": "security", "description": "first", "file": "a.py"},
        {"category": "security", "description": "second", "file": "b.py"},
    ]})
    assert sorted(v.rule_id for v in report.violations) == ["SEC-001", "SEC-002"]


@patch("loop.Groq")
def test_nudge_lists_credential_looking_search_hits_not_covered(mock_groq, tmp_path):
    (tmp_path / "infra").mkdir()
    (tmp_path / "infra" / "ec2.tf").write_text('user_data = "AWS_KEY=AKIAABCDEFGHIJKLMNOP"\n', encoding="utf-8")
    (tmp_path / "infra" / "db.tf").write_text('password = "S3cretPassw0rd!"\n', encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "fixture.py").write_text('API_KEY = "AKIAZZZZZZZZZZZZZZZZ"\n', encoding="utf-8")
    (tmp_path / ".gitignore").write_text("x\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("password = 'not-a-secret-in-docs'\n", encoding="utf-8")
    (tmp_path / "ok.py").write_text('password = os.getenv("DB_PASSWORD", "changeme-placeholder")\n', encoding="utf-8")
    agent, _ = make_agent(mock_groq, [], tmp_path, coverage=True)
    agent._execute_tool("search_code", {"pattern": "password|AKIA|api_key"})
    paths = sorted(p for p, _ in agent.secret_leads)
    assert paths == ["infra/db.tf", "infra/ec2.tf"]  # fixtures, docs and env lookups are not leads

    agent.read_paths.update({".gitignore", "infra/ec2.tf", "infra/db.tf", "ok.py"})
    msg = agent._coverage_gap_message([{"file": "infra/ec2.tf", "evidence": "x"}])
    assert msg and "infra/db.tf:1" in msg and "infra/ec2.tf" not in msg.split("credentials in files")[1].split(".")[0]
    assert "never copy the secret value" in msg
    agent.coverage_nudged = False
    msg = agent._coverage_gap_message([{"file": "infra/ec2.tf"}, {"file": "infra/db.tf"}])
    assert "credentials" not in (msg or "")  # both credential hits are now covered by findings


def test_gemini_fallback_order_prefers_newer_full_models_over_lite():
    pref = loop_module.PROVIDERS["gemini"]["preference"]
    assert pref.index("gemini-3.8-flash") < pref.index("gemini-3.1-flash-lite") < pref.index("gemini-flash-lite-latest")
