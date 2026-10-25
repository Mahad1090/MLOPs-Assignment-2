from agent import clone_repository, normalize_repo_url, parse_args, validate_repo_url
from loop import default_model_for, resolve_provider


def test_validate_repo_url_valid():
    assert validate_repo_url("https://github.com/octocat/Hello-World") is True
    assert validate_repo_url("https://github.com/torvalds/linux.git") is True
    assert validate_repo_url("git@github.com:owner/repo.git") is True


def test_validate_repo_url_invalid():
    assert validate_repo_url("ftp://example.com/repo") is False
    assert validate_repo_url("https://malicious-site.com/repo") is False
    assert validate_repo_url("not_a_url") is False
    assert validate_repo_url("") is False


def test_parse_args_required():
    args = parse_args(["--repo", "https://github.com/owner/repo", "--output", "report.json"])
    assert args.repo == "https://github.com/owner/repo"
    assert args.output == "report.json"
    assert args.max_iterations == 30
    assert args.model is None
    assert args.provider is None
    assert default_model_for(resolve_provider(args.provider))


def test_parse_args_custom_model_and_iterations():
    args = parse_args([
        "--repo", "https://github.com/owner/repo",
        "--output", "custom_report.json",
        "--model", "llama-3.1-8b-instant",
        "--max-iterations", "10",
        "--quiet",
    ])
    assert args.output == "custom_report.json"
    assert args.model == "llama-3.1-8b-instant"
    assert args.max_iterations == 10
    assert args.quiet is True


def test_normalize_repo_url_strips_browser_suffixes():
    assert normalize_repo_url("https://github.com/pallets/click/tree/main/docs") == "https://github.com/pallets/click"
    assert normalize_repo_url("https://github.com/a/b.git") == "https://github.com/a/b"
    assert normalize_repo_url(" https://github.com/a/b/ ") == "https://github.com/a/b"


def test_clone_nonexistent_repo_fails_fast_without_prompt(tmp_path):
    target = tmp_path / "t"
    target.mkdir()
    ok = clone_repository("https://github.com/this-owner-does-not-exist-xyz/none-abc", str(target), timeout=60)
    assert ok is False


def test_cli_exits_nonzero_on_bad_url(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    proc = subprocess.run(
        [sys.executable, str(root / "agent.py"), "--repo", "not-a-url", "--output", str(tmp_path / "r.json")],
        capture_output=True, text=True,
    )
    assert proc.returncode != 0
    assert "Invalid" in proc.stderr
    assert not (tmp_path / "r.json").exists()
