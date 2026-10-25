"""DevOps Repository Auditing Agent CLI Entry Point.

Conforms to Section 5:
python agent.py --repo <github_repository_url> --output report.json

Handles arguments, repository cloning into isolated temporary sandbox,
orchestrates the AgentLoop, performs post-processing verification,
and guarantees clean exit codes (0 on success, non-zero on fatal error).

Exit codes: 0 success | 1 invalid input/config | 2 clone failed | 3 agent error |
4 cannot write output | 5 LLM unavailable | 130 interrupted.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from loop import (
    PROVIDERS,
    AgentLoop,
    LLMError,
    default_model_for,
    get_api_key,
    resolve_provider,
)

GITHUB_HTTPS = r"https?://(?:www\.)?github\.com/[\w.-]+/[\w.-]+"
CLONE_RETRIES = 3
DEFAULT_CLONE_TIMEOUT = 300

NOT_FOUND_MARKERS = (
    "repository not found",
    "not found",
    "could not read username",
    "terminal prompts disabled",
    "authentication failed",
    "invalid username or password",
    "permission denied",
    "access denied",
)
NETWORK_MARKERS = (
    "could not resolve host",
    "failed to connect",
    "connection timed out",
    "connection reset",
    "unable to access",
    "could not read from remote repository",  # GitHub drops connections when throttling bursts of clones
    "early eof",
    "remote end hung up",
    "ssl",
    "returned error: 5",
    "returned error: 429",
    "rate limit",
)


def normalize_repo_url(url: str) -> str:
    """Normalise user input to https://github.com/<owner>/<repo>.

    Accepts browser URLs (/tree/main, /issues, ?query, #fragment, trailing slash or .git),
    scheme-less ``github.com/owner/repo`` and ``owner/repo`` shorthand. SSH URLs are preserved.
    """
    url = (url or "").strip()
    if not url:
        return url
    if re.match(r"^(www\.)?github\.com/", url, re.I):
        url = "https://" + url
    elif re.match(r"^(?!\.{1,2}/)[\w-][\w.-]*/[\w.-]+/?$", url):
        url = "https://github.com/" + url
    match = re.match(rf"^({GITHUB_HTTPS}?)(?:\.git)?(?:[/?#].*)?$", url)
    if match:
        return match.group(1)
    ssh = re.match(r"^(git@github\.com:[\w.-]+/[\w.-]+?)(?:\.git)?/?$", url)
    return ssh.group(1) if ssh else url


def validate_repo_url(url: str) -> bool:
    """Validate repository URL format (GitHub HTTPS or SSH)."""
    if not url or not isinstance(url, str):
        return False
    url = url.strip()
    https_pattern = rf"^{GITHUB_HTTPS}(\.git)?/?$"
    ssh_pattern = r"^git@github\.com:[\w.-]+/[\w.-]+(\.git)?$"
    return bool(re.match(https_pattern, url) or re.match(ssh_pattern, url))


def describe_clone_failure(stderr: str) -> Tuple[str, bool]:
    """Turn git's stderr into a short actionable message; second item says whether a retry may help."""
    low = (stderr or "").lower()
    if any(marker in low for marker in NOT_FOUND_MARKERS) and "returned error: 5" not in low:
        return (
            "Repository not found, or it is private. This agent only supports public GitHub "
            "repositories; check the URL spelling and that the repository is public.",
            False,
        )
    if any(marker in low for marker in NETWORK_MARKERS):
        return ("Could not reach the repository: network problem or GitHub throttling (wait a minute and retry; if it persists, check that the repository is public).", True)
    return ((stderr or "unknown git error").strip()[:400], False)


def clone_repository(
    repo_url: str,
    target_dir: str,
    depth: int = 50,
    timeout: int = DEFAULT_CLONE_TIMEOUT,
) -> bool:
    """Clone a public GitHub repository into a temporary directory."""
    cmd = [
        "git",
        "-c", "core.hooksPath=/dev/null",
        "-c", "protocol.file.allow=never",
        "-c", "core.longpaths=true",
        "clone",
        "--depth", str(depth),
        "--no-tags",
        "--single-branch",
        "--",
        repo_url,
        target_dir,
    ]
    # Never prompt for credentials: private/nonexistent repos must fail fast, unattended.
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_ASKPASS"] = "echo"
    env["GCM_INTERACTIVE"] = "never"

    for attempt in range(1, CLONE_RETRIES + 2):
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, check=False, env=env
            )
        except subprocess.TimeoutExpired:
            sys.stderr.write(
                f"Fatal error: git clone timed out after {timeout}s (very large repository or slow network). "
                "Retry with a larger --clone-timeout.\n"
            )
            return False
        except FileNotFoundError:
            sys.stderr.write("Fatal error: 'git' is not installed or not in PATH.\n")
            return False
        except Exception as exc:
            sys.stderr.write(f"Fatal error during git clone: {exc}\n")
            return False

        if proc.returncode == 0:
            if "checkout failed" in (proc.stderr or "").lower():
                sys.stderr.write(
                    "Warning: some files could not be checked out (for example names invalid on this "
                    "operating system); the audit covers only the files that were written.\n"
                )
            return True

        message, retryable = describe_clone_failure(proc.stderr)
        if retryable and attempt <= CLONE_RETRIES:
            time.sleep(3 * 2 ** (attempt - 1))  # 3s, 6s, 12s
            # git leaves a partial directory behind on failure; start the retry from an empty one
            for child in Path(target_dir).glob("*"):
                try:
                    if child.is_dir():
                        import shutil

                        shutil.rmtree(child, ignore_errors=True)
                    else:
                        child.unlink()
                except OSError:
                    pass
            continue
        sys.stderr.write(f"Fatal error cloning repository: {message}\n")
        return False
    return False


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not an integer") from None
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(args: Optional[list] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Autonomous AI Agent for DevOps Repository Auditing (Assignment 2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="Public GitHub repository URL or owner/repo to inspect (e.g. https://github.com/octocat/Hello-World)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output filepath to write the final JSON audit report (e.g. report.json)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=sorted(PROVIDERS),
        default=None,
        help="LLM backend (default: auto-detected from the API key / LLM_PROVIDER env var)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (default: <PROVIDER>_MODEL env var, else a built-in default)",
    )
    parser.add_argument(
        "--max-iterations",
        type=positive_int,
        default=30,
        help="Maximum tool calling iterations before triggering final report synthesis",
    )
    parser.add_argument(
        "--timeout",
        type=positive_int,
        default=1800,
        help="Overall wall-clock budget in seconds for the LLM investigation",
    )
    parser.add_argument(
        "--clone-timeout",
        type=positive_int,
        default=DEFAULT_CLONE_TIMEOUT,
        help="Seconds allowed for git clone (raise for very large repositories)",
    )
    parser.add_argument(
        "--trace",
        type=str,
        default=None,
        help="Path for the tool-call trace JSON (default: <output>.trace.json)",
    )
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="Do not write the tool-call trace file",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for the chosen provider (defaults to the provider's *_API_KEY environment variable)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress agent investigation traces on stdout",
    )
    return parser.parse_args(args)


def check_output_writable(output_path: Path) -> Optional[str]:
    """Fail early (before cloning and spending LLM calls) when the report cannot be written."""
    if output_path.is_dir():
        return f"Output path is a directory, not a file: {output_path}"
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        probe = tempfile.NamedTemporaryFile(dir=output_path.parent, prefix=".audit_probe_", delete=True)
        probe.close()
    except OSError as exc:
        return f"Cannot write to '{output_path.parent}': {exc}"
    return None


def write_json_atomic(path: Path, text: str) -> None:
    """Write via a temp file + rename so an interrupted run never leaves half-written JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def severity_summary(report: Any) -> str:
    counts: Dict[str, int] = {}
    for v in report.violations:
        counts[v.severity] = counts.get(v.severity, 0) + 1
    order = ["critical", "high", "medium", "low"]
    return ", ".join(f"{counts[s]} {s}" for s in order if s in counts) or "none"


def main() -> None:
    """CLI execution entry point."""
    args = parse_args()
    repo_url = normalize_repo_url(args.repo)
    output_path = Path(args.output).resolve()
    trace_path = None
    if not args.no_trace:
        trace_path = Path(args.trace).resolve() if args.trace else output_path.with_name(output_path.stem + ".trace.json")

    # Step 1: Validate input URL
    if not validate_repo_url(repo_url):
        sys.stderr.write(
            f"Fatal error: Invalid or unsupported repository URL: '{args.repo}'. "
            "Please provide a public GitHub repository URL (e.g., https://github.com/user/repo or user/repo).\n"
        )
        sys.exit(1)

    # Step 2: Resolve provider/model and validate API key prior to heavy operations
    try:
        provider = resolve_provider(args.provider, args.api_key)
    except ValueError as exc:
        sys.stderr.write(f"Fatal error: {exc}\n")
        sys.exit(1)
    model = args.model or default_model_for(provider)
    api_key = args.api_key or get_api_key(provider)
    if not api_key:
        names = ", ".join(sorted({k for cfg in PROVIDERS.values() for k in cfg["key_envs"]}))
        sys.stderr.write(
            f"Fatal error: Missing API key for provider '{provider}'. Set one of {names} "
            "in your .env file or environment (or pass --api-key).\n"
        )
        sys.exit(1)

    # Step 3: Make sure the report can be written before doing any expensive work
    problem = check_output_writable(output_path)
    if problem:
        sys.stderr.write(f"Fatal error: {problem}\n")
        sys.exit(4)

    # Step 4: Manage isolated temporary clone sandbox (leaving target repo unchanged)
    trace: list = []
    with tempfile.TemporaryDirectory() as temp_dir:
        clone_dir = Path(temp_dir) / "target_repo"
        clone_dir.mkdir(parents=True, exist_ok=True)

        print(f"[+] Cloning repository: {repo_url} into isolated sandbox...")
        if not clone_repository(repo_url, str(clone_dir), timeout=args.clone_timeout):
            sys.exit(2)

        # Step 5: Run autonomous agent loop
        try:
            agent = AgentLoop(
                repo_url=repo_url,
                repo_dir=str(clone_dir),
                model=model,
                provider=provider,
                api_key=api_key,
                max_iterations=args.max_iterations,
                verbose=not args.quiet,
                timeout_seconds=args.timeout,
            )
            try:
                report = agent.run()
            finally:
                trace = agent.trace
        except KeyboardInterrupt:
            sys.stderr.write("\nAudit interrupted by user.\n")
            sys.exit(130)
        except LLMError as exc:
            sys.stderr.write(f"Fatal error: the LLM could not complete the audit: {str(exc)}\n")
            sys.exit(5)
        except ValueError as exc:
            sys.stderr.write(f"Fatal error: {exc}\n")
            sys.exit(1)
        except Exception as exc:
            sys.stderr.write(f"Fatal error during agent loop execution: {str(exc)}\n")
            sys.exit(3)

    # Step 6: Write final machine-readable JSON report (atomically)
    try:
        write_json_atomic(output_path, report.model_dump_json(indent=2))
    except Exception as exc:
        sys.stderr.write(f"Fatal error writing output JSON report to '{output_path}': {str(exc)}\n")
        sys.exit(4)
    print(f"[+] Audit report successfully written to: {output_path}")
    print(f"[+] {len(report.violations)} violation(s): {severity_summary(report)}")

    # The trace is a convenience: failing to write it must never fail a completed audit
    if trace_path is not None:
        try:
            write_json_atomic(
                trace_path,
                json.dumps({"repository": repo_url, "provider": agent.provider, "model": agent.model, "steps": trace}, indent=2),
            )
            print(f"[+] Tool-call trace written to: {trace_path}")
        except Exception as exc:
            sys.stderr.write(f"Warning: could not write trace file '{trace_path}': {exc}\n")

    sys.exit(0)


if __name__ == "__main__":
    main()
