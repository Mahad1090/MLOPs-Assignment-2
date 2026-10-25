"""Unit tests for tools.py: inspection tools, path traversal safety, and line numbers."""

import subprocess
import tempfile
from pathlib import Path

import pytest

from tools import (
    dispatch_tool_call,
    git_history_search,
    git_log,
    git_status,
    inspect_workflows,
    list_files,
    read_file,
    run_ruff,
    search_code,
)


@pytest.fixture
def dummy_repo():
    """Create a temporary initialized git repo with sample files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo_path = Path(tmp_dir)

        # Initialize git repo
        subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Tester"], cwd=repo_path, check=True)
        subprocess.run(["git", "config", "user.email", "tester@test.com"], cwd=repo_path, check=True)

        # Create standard files
        (repo_path / "README.md").write_text("# Test Repo\nDevOps audit test.\n", encoding="utf-8")
        (repo_path / ".gitignore").write_text("*.pyc\n__pycache__/\n", encoding="utf-8")

        # Create suspicious file (.env)
        (repo_path / ".env").write_text("DATABASE_URL=postgres://localhost:5432/db\n", encoding="utf-8")

        # Create code file
        code_dir = repo_path / "src"
        code_dir.mkdir()
        code_file = code_dir / "app.py"
        code_file.write_text(
            "import os\n"
            "SECRET_TOKEN = 'sk-1234567890'\n"
            "def run():\n"
            "    print('Running')\n",
            encoding="utf-8",
        )

        # Git add and commit
        subprocess.run(["git", "add", "."], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_path, check=True, capture_output=True)

        yield repo_path


def test_list_files_detects_layout_and_suspicious_files(dummy_repo):
    output = list_files(str(dummy_repo))
    assert "README.md" in output
    assert "src/app.py" in output
    assert "[!] WARNING: Potentially inappropriate / temporary files" in output
    assert ".env" in output


def test_read_file_line_numbering(dummy_repo):
    output = read_file(str(dummy_repo), "src/app.py", start_line=1, end_line=10)
    assert "0001: import os" in output
    assert "0002: SECRET_TOKEN" in output
    assert "0003: def run():" in output


def test_read_file_path_traversal_guard(dummy_repo):
    # Attempt to escape repo directory
    output = read_file(str(dummy_repo), "../../etc/passwd")
    assert "Path traversal detected" in output or "Error" in output


def test_read_file_nonexistent(dummy_repo):
    output = read_file(str(dummy_repo), "does_not_exist.py")
    assert "Error: File not found" in output


def test_search_code_finds_pattern(dummy_repo):
    output = search_code(str(dummy_repo), pattern="SECRET_TOKEN")
    assert "src/app.py:2" in output
    assert "SECRET_TOKEN" in output


def test_search_code_no_match(dummy_repo):
    output = search_code(str(dummy_repo), pattern="NON_EXISTENT_PATTERN_12345")
    assert "No matches found" in output


def test_git_log_and_git_status(dummy_repo):
    log_out = git_log(str(dummy_repo))
    assert "Initial commit" in log_out

    status_out = git_status(str(dummy_repo))
    assert "clean" in status_out.lower()


def test_dispatch_tool_call_safety(dummy_repo):
    # Valid dispatch
    result = dispatch_tool_call(str(dummy_repo), "read_file", {"path": "README.md"})
    assert "0001: # Test Repo" in result

    # Invalid tool name handled gracefully without crashing
    err_result = dispatch_tool_call(str(dummy_repo), "malicious_tool", {})
    assert "Unknown tool" in err_result

    # Bad arguments handled gracefully
    bad_arg = dispatch_tool_call(str(dummy_repo), "read_file", {"invalid_param": 123})
    assert "Error" in bad_arg


def test_list_files_does_not_flag_build_or_dist_dirs(dummy_repo):
    (dummy_repo / "build").mkdir()
    (dummy_repo / "build" / "x.py").write_text("x = 1\n", encoding="utf-8")
    assert "Directory: build" not in list_files(str(dummy_repo))


def test_search_code_skips_symlinks_and_binaries(dummy_repo, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside") / "secret.txt"
    outside.write_text("OUTSIDE_SECRET_MARKER\n", encoding="utf-8")
    link = dummy_repo / "link.txt"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    assert "No matches found" in search_code(str(dummy_repo), "OUTSIDE_SECRET_MARKER")


def test_inspect_workflows_reports_numbered_risky_lines(dummy_repo):
    wf = dummy_repo / ".github" / "workflows"
    wf.mkdir(parents=True)
    (wf / "ci.yml").write_text(
        "name: CI\n"
        "on:\n"
        "  pull_request_target:\n"
        "permissions: write-all\n"
        "jobs:\n"
        "  t:\n"
        "    steps:\n"
        "      - uses: actions/checkout@main\n"
        "      - run: pytest\n"
        "        continue-on-error: true\n"
        "      - run: echo ${{ github.event.issue.title }}\n",
        encoding="utf-8",
    )
    out = inspect_workflows(str(dummy_repo))
    assert ".github/workflows/ci.yml" in out
    for needle in ("0003:", "0004: permissions: write-all", "continue-on-error", "github.event.issue.title"):
        assert needle in out


def test_inspect_workflows_without_directory(dummy_repo):
    assert "No .github/workflows" in inspect_workflows(str(dummy_repo))


def test_git_history_search_finds_deleted_secret(dummy_repo):
    (dummy_repo / "leak.txt").write_text("password = hunter2hunter2\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=dummy_repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "add leak"], cwd=dummy_repo, check=True, capture_output=True)
    subprocess.run(["git", "rm", "-q", "leak.txt"], cwd=dummy_repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "remove leak"], cwd=dummy_repo, check=True, capture_output=True)
    out = git_history_search(str(dummy_repo), "hunter2")
    assert "add leak" in out and "leak.txt" in out
    assert "No commits" in git_history_search(str(dummy_repo), "zzz_not_present_zzz")
    assert "Invalid regular expression" in git_history_search(str(dummy_repo), "(")


def test_run_ruff_does_not_modify_repo(dummy_repo):
    before = sorted(p.name for p in dummy_repo.iterdir())
    out = run_ruff(str(dummy_repo))
    assert isinstance(out, str) and out
    assert sorted(p.name for p in dummy_repo.iterdir()) == before


def test_dispatch_truncates_huge_output(dummy_repo):
    (dummy_repo / "big.py").write_text("\n".join(f"x{i} = {i}" for i in range(100000)), encoding="utf-8")
    out = dispatch_tool_call(str(dummy_repo), "read_file", {"path": "big.py", "start_line": 1, "end_line": 300})
    assert len(out) <= 13000


def test_inspect_workflows_risk_leads(dummy_repo):
    wf = dummy_repo / ".github" / "workflows"
    wf.mkdir(parents=True)
    (wf / "danger.yml").write_text(
        "name: x\n"
        "on:\n"
        "  pull_request_target:\n"
        "jobs:\n"
        "  a:\n"
        "    runs-on: ubuntu-18.04\n"
        "    steps:\n"
        "      - uses: actions/checkout@v4\n"
        "        with:\n"
        "          ref: ${{ github.event.pull_request.head.sha }}\n"
        "      - uses: some/action@master\n"
        "      - run: |\n"
        "          echo ${{ github.event.pull_request.title }}\n"
        "          echo \"key: $PRIVATE_KEY\"\n"
        "          pytest || true\n",
        encoding="utf-8",
    )
    (wf / "empty.yml").write_text("", encoding="utf-8")
    out = inspect_workflows(str(dummy_repo))
    leads = out.split("WORKFLOW SUMMARIES")[0]
    for needle in (
        "PULL_REQUEST_TARGET", "PR head code checked out", "mutable branch", "retired runner",
        "SCRIPT-INJECTION", "SECRET printed", "FORCED-SUCCESS",
    ):
        assert needle in leads, needle
    assert "danger.yml:10:" in leads
    assert "EMPTY FILE" in out


def test_inspect_workflows_clean_workflow_has_no_leads(dummy_repo):
    wf = dummy_repo / ".github" / "workflows"
    wf.mkdir(parents=True)
    (wf / "ci.yml").write_text(
        "name: CI\non: [push, pull_request]\npermissions:\n  contents: read\njobs:\n  t:\n"
        "    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v4\n"
        "        with:\n          ref: ${{ github.head_ref }}\n      - run: pytest\n",
        encoding="utf-8",
    )
    assert "No heuristic risk leads detected" in inspect_workflows(str(dummy_repo))


def test_list_files_separates_fixture_env_files(dummy_repo):
    (dummy_repo / "tests").mkdir()
    (dummy_repo / "tests" / ".env").write_text("FOO=bar\n", encoding="utf-8")
    out = list_files(str(dummy_repo))
    warning_block = out.split("Tracked Files:")[0]
    before, _, after = warning_block.partition("intentional fixtures")
    assert "File: .env" in before  # root .env is still a warning
    assert "tests/.env" in after
    assert "tests/.env" not in before
