"""Unusual inputs: huge repositories, non-Python stacks, odd files, bad CLI input, output handling."""

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import agent as agent_module
import tools
from agent import (
    check_output_writable,
    describe_clone_failure,
    normalize_repo_url,
    parse_args,
    validate_repo_url,
    write_json_atomic,
)
from loop import AgentLoop
from tools import is_key_file, list_files, read_file, search_code

ROOT = Path(__file__).resolve().parent.parent


def make_tree(base: Path, files: dict) -> Path:
    for rel, content in files.items():
        path = base / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return base


# ------------------------------------------------------------------ key-file detection
@pytest.mark.parametrize("path", [
    ".github/workflows/ci.yml", ".circleci/config.yml", "Dockerfile", "services/api/Dockerfile.prod",
    "docker-compose.override.yml", "Jenkinsfile", ".gitlab-ci.yml", "azure-pipelines.yml",
    "go.mod", "pom.xml", "build.gradle.kts", "Cargo.toml", "Gemfile", "composer.json",
    "src/App/App.csproj", "mix.exs", "pubspec.yaml", "Makefile", "requirements-dev.txt", "package.json",
    "infra/main.tf", "charts/app/Chart.yaml", ".gitignore", ".env", ".env.production", "CODEOWNERS",
    "go.sum", "Cargo.lock", "package-lock.json", "yarn.lock", "composer.lock", "Gemfile.lock",
])
def test_key_files_recognised_across_ecosystems(path):
    assert is_key_file(path)


@pytest.mark.parametrize("path", ["src/main.go", "README.md", "docs/guide.md", "tests/test_x.py", "main.rs"])
def test_ordinary_files_are_not_key_files(path):
    assert not is_key_file(path)


def test_lockfiles_excluded_when_requested():
    assert is_key_file("go.sum") and not is_key_file("go.sum", include_lockfiles=False)
    assert is_key_file("go.mod", include_lockfiles=False)


# ------------------------------------------------------------------------ huge repos
def test_list_files_large_repo_keeps_key_files_visible(tmp_path):
    files = {f"pkg{d:02d}/mod{m}/file{i}.go": "package x\n" for d in range(10) for m in range(3) for i in range(12)}
    files.update({
        ".github/workflows/ci.yml": "on: push\n",
        "services/api/Dockerfile": "FROM golang:1.22\n",
        "go.mod": "module x\n",
        "go.sum": "",
        "deploy/terraform/main.tf": "terraform {}\n",
    })
    make_tree(tmp_path, files)
    out = list_files(str(tmp_path))
    assert "Layout (files per top-level directory)" in out
    for key in (".github/workflows/ci.yml", "services/api/Dockerfile", "go.mod", "go.sum", "deploy/terraform/main.tf"):
        assert key in out, key
    shown = [line for line in out.splitlines() if line.startswith("  ") and "(" in line and "files)" not in line]
    assert len(shown) <= tools.MAX_LISTED_FILES + 5
    assert "more files" in out


def test_list_files_stops_walking_absurdly_large_trees(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "MAX_WALK_FILES", 60)
    make_tree(tmp_path, {f"d{i % 7}/f{i}.txt": "x" for i in range(300)})
    out = list_files(str(tmp_path))
    assert "(60+ total files" in out


def test_list_files_empty_repo_and_bad_arguments(tmp_path):
    assert "0 total files" in list_files(str(tmp_path))
    assert "does not exist" in list_files(str(tmp_path / "missing"))
    make_tree(tmp_path, {"a.txt": "x"})
    assert "a.txt" in list_files(str(tmp_path), max_depth="three")  # junk depth falls back to the default


def test_search_code_respects_time_budget_and_skips_node_modules(tmp_path, monkeypatch):
    make_tree(tmp_path, {"node_modules/lib/x.js": "SECRET_MARKER", "src/a.js": "SECRET_MARKER"})
    out = search_code(str(tmp_path), "SECRET_MARKER")
    assert "src/a.js:1" in out and "node_modules" not in out
    monkeypatch.setattr(tools, "SEARCH_TIME_BUDGET", -1)
    assert "search stopped" in search_code(str(tmp_path), "SECRET_MARKER")


# ---------------------------------------------------------------------------- odd files
def test_read_file_handles_binary_huge_and_minified(tmp_path, monkeypatch):
    (tmp_path / "blob.bin").write_bytes(b"\x00\x01\x02" * 100)
    assert "binary" in read_file(str(tmp_path), "blob.bin")

    (tmp_path / "min.js").write_text("a=1;" * 2000 + "\nsecond line\n", encoding="utf-8")
    out = read_file(str(tmp_path), "min.js")
    assert "line truncated" in out and "0002: second line" in out
    assert len(out) < tools.MAX_LINE_CHARS + 500

    monkeypatch.setattr(tools, "MAX_READ_BYTES", 10)
    (tmp_path / "big.txt").write_text("x" * 100, encoding="utf-8")
    assert "is large" in read_file(str(tmp_path), "big.txt")


def test_read_file_tolerates_bad_line_arguments_and_non_utf8(tmp_path):
    (tmp_path / "latin.txt").write_bytes("caf\xe9\nline2\n".encode("latin-1"))
    out = read_file(str(tmp_path), "latin.txt", start_line="x", end_line=None)
    assert "0002: line2" in out


# --------------------------------------------------------- non-Python stacks: absent items
def _agent(tmp_path):
    fake = MagicMock()
    fake.models.list.side_effect = Exception("offline")
    with patch("loop.GeminiClient", return_value=fake):
        a = AgentLoop("https://github.com/o/r", str(tmp_path), api_key="k", provider="gemini", verbose=False)
    return a


@pytest.mark.parametrize("files", [
    {"main.go": "package main", "main_test.go": "package main"},
    {"pom.xml": "<project/>", "src/test/java/AppTest.java": "class AppTest {}"},
    {"Cargo.toml": "[package]", "src/lib.rs": "#[test] fn t() {}"},
    {"Gemfile": "", "spec/app_spec.rb": ""},
    {"composer.json": "{}", "tests/FooTest.php": "<?php"},
    {"App.csproj": "<Project/>", "App.Tests/UnitTests.cs": ""},
    {"package.json": "{}", "src/app.test.ts": ""},
])
def test_tests_are_detected_in_other_ecosystems(tmp_path, files):
    make_tree(tmp_path, files)
    assert "no test files or test directory found" not in _agent(tmp_path)._absent_standard_items()


def test_missing_tests_still_reported_when_really_absent(tmp_path):
    make_tree(tmp_path, {"main.go": "package main", "go.mod": "module x"})
    assert "no test files or test directory found" in _agent(tmp_path)._absent_standard_items()


def test_key_files_of_other_stacks_are_flagged_for_review(tmp_path):
    make_tree(tmp_path, {"go.mod": "", "go.sum": "", "infra/main.tf": "", "charts/a/Chart.yaml": "", "src/main.go": ""})
    missing = _agent(tmp_path)._unreviewed_key_files()
    assert "go.mod" in missing and "infra/main.tf" in missing and "go.sum" not in missing  # lockfiles not required reading


# -------------------------------------------------------------------- URL / clone input
@pytest.mark.parametrize("raw,expected", [
    ("pallets/flask", "https://github.com/pallets/flask"),
    ("github.com/pallets/flask", "https://github.com/pallets/flask"),
    ("https://www.github.com/pallets/flask/tree/main/docs?x=1#y", "https://www.github.com/pallets/flask"),
    ("http://github.com/a/b.git", "http://github.com/a/b"),
    ("https://github.com/a/b/", "https://github.com/a/b"),
    ("git@github.com:a/b.git", "git@github.com:a/b"),
    ("  https://github.com/a/b  ", "https://github.com/a/b"),
])
def test_normalize_repo_url_variants(raw, expected):
    assert normalize_repo_url(raw) == expected
    assert validate_repo_url(normalize_repo_url(raw))


@pytest.mark.parametrize("raw", [
    "./local/path", "../x/y", "a/b/c", "https://gitlab.com/a/b", "ftp://github.com/a/b", "", "just-a-word",
    "https://user:pass@github.com/a/b", "-oProxyCommand=evil", "https://github.com/a",
])
def test_invalid_repo_inputs_are_rejected(raw):
    assert not validate_repo_url(normalize_repo_url(raw))


def test_describe_clone_failure_categories():
    msg, retry = describe_clone_failure("remote: Repository not found.\nfatal: repository 'x' not found")
    assert "private" in msg and not retry
    msg, retry = describe_clone_failure("fatal: could not read Username for 'https://github.com': terminal prompts disabled")
    assert "private" in msg and not retry
    msg, retry = describe_clone_failure("fatal: unable to access 'https://github.com/x/y/': Could not resolve host: github.com")
    assert "network problem" in msg and retry
    msg, retry = describe_clone_failure("fatal: unable to access 'x': The requested URL returned error: 429")
    assert retry
    msg, retry = describe_clone_failure("fatal: Could not read from remote repository. Please make sure you have the correct access rights")
    assert retry and "throttling" in msg
    msg, retry = describe_clone_failure("fatal: something exotic")
    assert "exotic" in msg and not retry


def test_clone_timeout_fails_cleanly(tmp_path, capsys):
    import subprocess as sp
    with patch.object(agent_module.subprocess, "run", side_effect=sp.TimeoutExpired(cmd="git", timeout=1)):
        assert agent_module.clone_repository("https://github.com/a/b", str(tmp_path), timeout=1) is False
    assert "--clone-timeout" in capsys.readouterr().err


def test_clone_retries_transient_network_failures(tmp_path, capsys):
    fail = MagicMock(returncode=128, stderr="fatal: unable to access: Could not resolve host: github.com")
    ok = MagicMock(returncode=0, stderr="")
    with patch.object(agent_module.subprocess, "run", side_effect=[fail, ok]) as run, patch.object(agent_module.time, "sleep"):
        assert agent_module.clone_repository("https://github.com/a/b", str(tmp_path)) is True
    assert run.call_count == 2


def test_clone_warns_on_partial_checkout(tmp_path, capsys):
    partial = MagicMock(returncode=0, stderr="warning: Clone succeeded, but checkout failed.")
    with patch.object(agent_module.subprocess, "run", return_value=partial):
        assert agent_module.clone_repository("https://github.com/a/b", str(tmp_path)) is True
    assert "could not be checked out" in capsys.readouterr().err


def test_clone_passes_non_interactive_environment(tmp_path):
    ok = MagicMock(returncode=0, stderr="")
    with patch.object(agent_module.subprocess, "run", return_value=ok) as run:
        agent_module.clone_repository("https://github.com/a/b", str(tmp_path))
    env = run.call_args.kwargs["env"]
    assert env["GIT_TERMINAL_PROMPT"] == "0" and run.call_args.args[0][-3] == "--"


# ----------------------------------------------------------------- output handling / CLI
def test_output_checks_and_atomic_write(tmp_path):
    assert check_output_writable(tmp_path / "sub" / "report.json") is None  # parent is created
    assert "directory" in check_output_writable(tmp_path)
    target = tmp_path / "out" / "report.json"
    write_json_atomic(target, '{"ok": true}')
    assert json.loads(target.read_text()) == {"ok": True}
    assert [p.name for p in target.parent.iterdir()] == ["report.json"]  # no temp files left behind
    write_json_atomic(target, '{"ok": false}')
    assert json.loads(target.read_text()) == {"ok": False}


def test_numeric_arguments_must_be_positive():
    for bad in (["--max-iterations", "0"], ["--timeout", "-5"], ["--clone-timeout", "abc"]):
        with pytest.raises(SystemExit):
            parse_args(["--repo", "a/b", "--output", "r.json"] + bad)
    args = parse_args(["--repo", "a/b", "--output", "r.json", "--clone-timeout", "900"])
    assert args.clone_timeout == 900


def run_cli(args, tmp_path, extra_env=None):
    env = dict(os.environ)
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY", "LLM_PROVIDER"):
        env[name] = ""  # blank values are treated as missing (and stop .env from filling them in)
    env.update(extra_env or {})
    return subprocess.run(
        [sys.executable, str(ROOT / "agent.py")] + args, capture_output=True, text=True, env=env, cwd=tmp_path, timeout=120,
    )


def test_cli_missing_key_exits_1_with_guidance(tmp_path):
    proc = run_cli(["--repo", "octocat/Hello-World", "--output", str(tmp_path / "r.json")], tmp_path)
    assert proc.returncode == 1
    assert "Missing API key" in proc.stderr and "ANTHROPIC_API_KEY" in proc.stderr and "OPENAI_API_KEY" in proc.stderr


def test_cli_unwritable_output_fails_before_any_work(tmp_path):
    proc = run_cli(
        ["--repo", "octocat/Hello-World", "--output", str(tmp_path), "--api-key", "sk-ant-dummy"], tmp_path,
    )
    assert proc.returncode == 4
    assert "directory" in proc.stderr and "Cloning" not in proc.stdout


def test_cli_unknown_repo_exits_2_quickly_with_hint(tmp_path):
    proc = run_cli(
        ["--repo", "no-such-owner-xyz-123/no-such-repo-456", "--output", str(tmp_path / "r.json"), "--api-key", "sk-ant-dummy"],
        tmp_path,
    )
    assert proc.returncode == 2
    assert "not found" in proc.stderr.lower() or "network" in proc.stderr.lower()
    assert not (tmp_path / "r.json").exists()


def test_cli_rejects_non_github_urls(tmp_path):
    proc = run_cli(["--repo", "https://gitlab.com/a/b", "--output", str(tmp_path / "r.json"), "--api-key", "gsk_x"], tmp_path)
    assert proc.returncode == 1 and "Invalid or unsupported" in proc.stderr
