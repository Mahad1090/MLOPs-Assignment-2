"""Inspection tools for DevOps Repository Auditing Agent.

Enforces sandbox safety, path traversal protection, line-numbered outputs,
and compact context formatting conforming to Section 3 and Section 9.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from schemas import sanitize_evidence

# Suspicious file names or patterns to highlight in repository hygiene
SUSPICIOUS_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
    ".DS_Store",
    "Thumbs.db",
    "npm-debug.log",
    "yarn-error.log",
}

SUSPICIOUS_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".log",
    ".tmp",
    ".swp",
    ".swo",
    ".bak",
}

SUSPICIOUS_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    ".venv",
    "venv",
}


# Path components that mark test fixtures / examples, where dummy .env files etc. are intentional
FIXTURE_DIR_NAMES = {
    "tests", "test", "testing", "__tests__", "fixtures", "fixture", "testdata", "test_data",
    "examples", "example", "samples", "sample", "docs", "doc", "spec", "specs",
}

MAX_TOOL_OUTPUT_CHARS = 12000
MAX_SEARCH_FILE_BYTES = 1_000_000
BINARY_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".zip", ".tar", ".gz",
    ".pyc", ".woff", ".woff2", ".ttf", ".eot", ".mp4", ".mp3", ".jar", ".whl",
    ".exe", ".dll", ".so", ".bin", ".svg",
}


MAX_WALK_FILES = 20000      # stop walking absurdly large trees
MAX_LISTED_FILES = 150      # file lines shown by list_files
MAX_READ_BYTES = 5_000_000  # read_file refuses bigger files (use search_code)
MAX_LINE_CHARS = 1500       # read_file truncates minified/huge lines
SEARCH_TIME_BUDGET = 25.0   # seconds for one search_code call
SKIP_SEARCH_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv"}

# Key DevOps files (matched on the lower-cased basename): CI/CD, containers, build manifests,
# infrastructure-as-code, hygiene/config files, across Python, Node, Go, Java, Rust, Ruby, PHP, .NET...
KEY_FILE_REGEXES = [
    re.compile(r"^dockerfile[\w.-]*$|^[\w.-]+\.dockerfile$"),
    re.compile(r"^docker-compose[\w.-]*\.ya?ml$|^compose\.ya?ml$"),
    re.compile(
        r"^jenkinsfile[\w.-]*$|^\.gitlab-ci\.ya?ml$|^azure-pipelines[\w.-]*\.ya?ml$|^\.travis\.ya?ml$"
        r"|^bitbucket-pipelines\.ya?ml$|^cloudbuild\.ya?ml$|^\.drone\.ya?ml$|^buildkite[\w.-]*\.ya?ml$"
    ),
    re.compile(
        r"^requirements[\w.-]*\.txt$|^package\.json$|^pyproject\.toml$|^pipfile$|^setup\.(py|cfg)$|^tox\.ini$"
        r"|^go\.mod$|^pom\.xml$|^build\.gradle(\.kts)?$|^settings\.gradle(\.kts)?$|^gemfile$|^cargo\.toml$"
        r"|^composer\.json$|^mix\.exs$|^[\w.-]+\.csproj$|^directory\.build\.props$|^global\.json$"
        r"|^deno\.jsonc?$|^pubspec\.yaml$|^build\.sbt$|^cmakelists\.txt$|^makefile$"
    ),
    re.compile(
        r"^[\w.-]+\.tf$|^chart\.ya?ml$|^skaffold\.ya?ml$|^kustomization\.ya?ml$|^serverless\.ya?ml$"
        r"|^\.gitignore$|^\.dockerignore$|^\.env(\.[\w.-]+)?$|^codeowners$|^\.pre-commit-config\.ya?ml$"
        r"|^dependabot\.ya?ml$|^renovate\.json5?$"
    ),
]
LOCKFILE_REGEX = re.compile(
    r"^(package-lock\.json|yarn\.lock|pnpm-lock\.yaml|poetry\.lock|uv\.lock|pipfile\.lock|go\.sum|cargo\.lock"
    r"|gemfile\.lock|composer\.lock|packages\.lock\.json|gradle\.lockfile|mix\.lock)$"
)


def is_key_file(rel_path: str, include_lockfiles: bool = True) -> bool:
    """True for files an auditor should look at: CI/CD configs, Docker, manifests, IaC, ignore/env files."""
    low = rel_path.replace("\\", "/").lower()
    name = low.rsplit("/", 1)[-1]
    if low.startswith(".github/workflows/") and name.endswith((".yml", ".yaml")):
        return True
    if low.startswith(".circleci/"):
        return True
    if include_lockfiles and LOCKFILE_REGEX.match(name):
        return True
    return any(rx.match(name) for rx in KEY_FILE_REGEXES)


PER_FILE_MATCH_CAP = 5       # search_code: one noisy file must not fill the whole result list
MAX_COLLECTED_MATCHES = 400  # search_code: stop collecting after this many lines
MAX_SECRET_LEADS = 40
NOISY_FILE = re.compile(
    r"(^|/)(package-lock\.json|yarn\.lock|pnpm-lock\.yaml|poetry\.lock|uv\.lock|pipfile\.lock|go\.sum|cargo\.lock"
    r"|gemfile\.lock|composer\.lock|packages\.lock\.json|mix\.lock|npm-shrinkwrap\.json)$|\.min\.(js|css)$|\.map$|\.snap$",
    re.I,
)

# Credential-looking text. Provider-prefixed tokens and private-key headers are reported as-is; generic
# NAME = value assignments are skipped when the value looks like a placeholder, a boolean or a variable.
HARD_SECRET_LEAD = re.compile(
    r"AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|\bgh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
    r"|\bxox[baprs]-[A-Za-z0-9-]{10,}|\bsk-ant-[A-Za-z0-9_-]{20,}|\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}"
    r"|\bAIza[0-9A-Za-z_-]{30,}|\bgsk_[A-Za-z0-9]{20,}"
)
_SECRET_NAME = r"(?:api[_-]?key|secret[_-]?key|access[_-]?key|client[_-]?secret|secret|password|passwd|pwd|token)\w*"
QUOTED_SECRET_LEAD = re.compile(rf"""(?i)\b{_SECRET_NAME}["']?\s*[:=]\s*["']([^"']{{3,60}})["']""")
# unquoted values (YAML/env files): must mix letters and digits, which separates real secrets from identifiers
UNQUOTED_SECRET_LEAD = re.compile(
    rf"""(?i)\b{_SECRET_NAME}\s*[:=]\s*((?=[^\s"'(){{}}\[\]<>,;#]*\d)(?=[^\s"'(){{}}\[\]<>,;#]*[A-Za-z])[^\s"'(){{}}\[\]<>,;#$]{{6,}})"""
)
PLACEHOLDER_VALUE = re.compile(
    r"(?i)change[_-]?me|example|your[_-]|placeholder|x{4,}|<[^>]+>|\$\{|%\(|dummy|sample|todo|\*{4,}|^\$|^env\.|getenv|os\.environ"
    r"|^(true|false|null|none|nil|string|number|required|optional|secret|password|token|passwd|empty|redacted)$"
    r"|^\\"
)
DOC_CONTEXT = re.compile(r"(?i)\b(sample|example|e\.g\.|dummy|placeholder|lorem)\b")
# Generated / vendored / presentation files whose "secret"-named variables are not credentials
NON_SECRET_SUFFIXES = (".css", ".scss", ".sass", ".less", ".svg", ".ejs.map", ".lock")
VENDORED_DIRS = {"vendor", "third_party", "third-party", "bower_components", "dist", "build", "node_modules", "fontawesome"}


def secret_lead_kind(text: str) -> Optional[str]:
    """Classify a line as a likely hardcoded credential, or None."""
    if HARD_SECRET_LEAD.search(text):
        return "token/key pattern"
    if DOC_CONTEXT.search(text):
        return None  # prose/sample/example lines are documentation, not credentials
    match = QUOTED_SECRET_LEAD.search(text)
    if match:
        value = match.group(1)
        words = value.split()
        # short passphrases like 'keyboard cat' are plausible secrets; sentences are UI text
        if (" " not in value or (len(words) <= 2 and len(value) <= 30)) and not PLACEHOLDER_VALUE.search(value):
            return "credential assignment"
        return None
    match = UNQUOTED_SECRET_LEAD.search(text)
    if match and not PLACEHOLDER_VALUE.search(match.group(1)):
        return "credential assignment"
    return None


def triple_quoted_lines(lines: List[str]) -> set:
    """1-indexed line numbers that sit inside a Python triple-quoted string (docstring/example text)."""
    inside: set = set()
    state: Optional[str] = None
    for number, line in enumerate(lines, start=1):
        if state:
            inside.add(number)
        for match in re.finditer(r'"""|\'\'\'', line):
            token = match.group(0)
            if state is None:
                state = token
            elif state == token:
                state = None
        if state:
            inside.add(number)
    return inside


def python_docstring_lines(path: Path) -> set:
    """Docstring line numbers of a .py file (empty set for other files or on read errors)."""
    if path.suffix.lower() != ".py":
        return set()
    try:
        return triple_quoted_lines(path.read_text(encoding="utf-8", errors="ignore").splitlines())
    except OSError:
        return set()


def _search_priority(rel_path: str) -> int:
    """0 = source/config, 1 = prose docs, 2 = tests/fixtures/examples (listed last)."""
    parts = {part.lower() for part in rel_path.split("/")[:-1]}
    if parts & FIXTURE_DIR_NAMES:
        return 2
    if rel_path.lower().endswith((".md", ".rst", ".txt")):
        return 1
    return 0


def _git_env() -> Dict[str, str]:
    """Environment that makes git strictly non-interactive."""
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


def _resolve_safe_path(repo_dir: str, rel_path: str) -> Path:
    """Resolve a relative path securely within repo_dir, preventing path traversal attacks.

    Raises ValueError if the path resolves outside the repository sandbox.
    """
    base = Path(repo_dir).resolve()
    # Strip leading slash or backslash
    cleaned = rel_path.lstrip("/\\")
    target = (base / cleaned).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        raise ValueError(f"Path traversal detected: '{rel_path}' is outside repository root.") from None
    return target


def list_files(repo_dir: str, max_depth: int = 4) -> str:
    """Discover repository layout, files, sizes, and flag hygiene concerns.

    Walks repo_dir, omitting internal .git directory (while preserving .github). For large repositories
    the output switches to a layout summary (files per top-level directory) followed by the key DevOps
    files, so CI configs, Dockerfiles and manifests are never lost behind a truncated listing.
    """
    try:
        max_depth = max(1, min(int(max_depth), 12))
    except (TypeError, ValueError):
        max_depth = 4
    base = Path(repo_dir).resolve()
    if not base.exists() or not base.is_dir():
        return f"Error: Repository directory does not exist: {repo_dir}"

    records: List[tuple] = []  # (rel_path, size_str, depth)
    suspicious_findings: List[str] = []
    fixture_findings: List[str] = []
    top_counts: Counter = Counter()
    file_count = 0
    truncated = False

    for root, dirs, files in os.walk(base):
        # Exclude internal .git database, but keep .github
        dirs[:] = sorted(d for d in dirs if d != ".git")

        rel_root = Path(root).relative_to(base)
        depth = len(rel_root.parts)
        if depth >= max_depth:
            dirs.clear()  # Don't recurse deeper
            continue

        # Check for suspicious committed directories
        for d in dirs:
            if d in SUSPICIOUS_DIRS:
                suspicious_findings.append(f"Directory: {(rel_root / d).as_posix()}")

        for f in sorted(files):
            if file_count >= MAX_WALK_FILES:
                truncated = True
                break
            file_count += 1
            file_str = (rel_root / f).as_posix()
            full_path = Path(root) / f
            top_counts[rel_root.parts[0] + "/" if rel_root.parts else "(root files)"] += 1

            # Check suspicious files
            if f in SUSPICIOUS_NAMES or any(f.endswith(sfx) for sfx in SUSPICIOUS_SUFFIXES):
                parts = {part.lower() for part in rel_root.parts}
                if parts & FIXTURE_DIR_NAMES:
                    fixture_findings.append(f"File: {file_str}")
                else:
                    suspicious_findings.append(f"File: {file_str}")

            try:
                size_bytes = full_path.stat().st_size
                if size_bytes < 1024:
                    size_str = f"{size_bytes}B"
                elif size_bytes < 1024 * 1024:
                    size_str = f"{size_bytes / 1024:.1f}KB"
                else:
                    size_str = f"{size_bytes / (1024 * 1024):.1f}MB"
            except Exception:
                size_str = "unknown"

            records.append((file_str, size_str, depth))
        if truncated:
            break

    count_label = f"{file_count}+" if truncated else str(file_count)
    summary_lines = [f"Repository File Tree ({count_label} total files, max_depth={max_depth}):"]
    if suspicious_findings:
        cap = suspicious_findings[:40]
        summary_lines.append("\n[!] WARNING: Potentially inappropriate / temporary files tracked in repository:")
        summary_lines.extend(f"  - {finding}" for finding in cap)
        if len(suspicious_findings) > len(cap):
            summary_lines.append(f"  ... and {len(suspicious_findings) - len(cap)} more")
        summary_lines.append("")

    if fixture_findings:
        summary_lines.append(
            "\n[i] Artifact-like files inside test/example/docs directories (usually intentional fixtures "
            "with dummy values; read them before treating as a defect):"
        )
        summary_lines.extend(f"  - {finding}" for finding in fixture_findings[:20])
        summary_lines.append("")

    if len(records) <= MAX_LISTED_FILES:
        summary_lines.append("\nTracked Files:")
        summary_lines.extend(f"  {path} ({size})" for path, size, _ in records)
        return "\n".join(summary_lines)

    # Large repository: layout summary + key DevOps files + as many other files as fit
    summary_lines.append("\nLayout (files per top-level directory):")
    for name, count in top_counts.most_common(25):
        summary_lines.append(f"  {name} ({count} files)")
    if len(top_counts) > 25:
        summary_lines.append(f"  ... and {len(top_counts) - 25} more top-level entries")

    key = [r for r in records if r[2] == 0 or is_key_file(r[0])]
    key_shown = key[:100]
    summary_lines.append("\nKey files (root-level files, CI/CD, Docker, manifests, lockfiles, IaC):")
    summary_lines.extend(f"  {path} ({size})" for path, size, _ in key_shown)
    if len(key) > len(key_shown):
        summary_lines.append(f"  ... and {len(key) - len(key_shown)} more key files (use search_code with file_pattern)")

    key_paths = {r[0] for r in key_shown}
    others = [r for r in records if r[0] not in key_paths]
    room = max(0, MAX_LISTED_FILES - len(key_shown))
    summary_lines.append("\nOther files (sample):")
    summary_lines.extend(f"  {path} ({size})" for path, size, _ in others[:room])
    remaining = len(others) - room
    note = "+" if truncated else ""
    if remaining > 0:
        summary_lines.append(
            f"  ... and {remaining}{note} more files (use search_code with file_pattern, or list_files with a smaller max_depth)"
        )
    return "\n".join(summary_lines)


def read_file(repo_dir: str, path: str, start_line: int = 1, end_line: int = 200) -> str:
    """Read a specific file with 1-indexed line numbers and pagination.

    Prefixing line numbers prevents line-counting hallucinations.
    """
    try:
        target = _resolve_safe_path(repo_dir, path)
    except ValueError as e:
        return f"Error: {str(e)}"

    if not target.exists():
        return f"Error: File not found: {path}"
    if not target.is_file():
        return f"Error: Path is not a file: {path}"

    try:
        size = target.stat().st_size
        if size > MAX_READ_BYTES:
            return (
                f"Error: '{path}' is large ({size / (1024 * 1024):.1f}MB); read_file is for source/config files. "
                "Use search_code with a pattern and file_pattern instead."
            )
        with open(target, "rb") as fh:
            if b"\0" in fh.read(2048):
                return f"Error: '{path}' appears to be a binary file."
    except OSError as exc:
        return f"Error reading file '{path}': {str(exc)}"

    try:
        start_line = max(1, int(start_line))
        end_line = int(end_line)
    except (TypeError, ValueError):
        start_line, end_line = 1, 200
    if end_line < start_line:
        end_line = start_line + 100

    # Avoid context explosion: hard limit 300 lines per read
    if (end_line - start_line) > 300:
        end_line = start_line + 300

    try:
        content = target.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"Error reading file '{path}': {str(exc)}"

    lines = content.splitlines()
    total_lines = len(lines)

    if total_lines == 0:
        return f"[File '{path}' is empty (0 lines)]"

    actual_start = min(start_line, total_lines)
    actual_end = min(end_line, total_lines)

    output: List[str] = [
        f"--- File: {path} (Showing lines {actual_start}-{actual_end} of {total_lines}) ---"
    ]
    for idx in range(actual_start, actual_end + 1):
        line_content = lines[idx - 1]
        if len(line_content) > MAX_LINE_CHARS:
            line_content = line_content[:MAX_LINE_CHARS] + f"...[line truncated, {len(line_content)} chars]"
        output.append(f"{idx:04d}: {line_content}")

    if actual_end < total_lines:
        output.append(f"--- [Paging notice: {total_lines - actual_end} more lines remain in {path}] ---")

    return "\n".join(output)


def search_code(repo_dir: str, pattern: str, file_pattern: Optional[str] = None) -> str:
    """Search for regex or keyword patterns across text files with exact line numbers.

    Returns filepath:line_number: matched_line. At most 5 matches per file are shown so a single noisy
    file cannot fill the result list; lockfiles/minified files are skipped unless ``file_pattern`` is
    given; source and config files are listed before docs and tests.
    """
    base = Path(repo_dir).resolve()
    if not base.exists() or not base.is_dir():
        return f"Error: Repository directory does not exist: {repo_dir}"

    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return f"Error: Invalid regular expression '{pattern}': {str(e)}"

    max_matches = 50
    file_filter = None
    if file_pattern:
        try:
            file_filter = re.compile(file_pattern, re.IGNORECASE)
        except re.error:
            pass

    deadline = time.monotonic() + SEARCH_TIME_BUDGET
    timed_out = False
    scanned = 0
    per_file: Dict[str, List[str]] = {}
    extra: Dict[str, int] = {}
    collected = 0

    for root, dirs, files in os.walk(base):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_SEARCH_DIRS)
        if time.monotonic() > deadline:
            timed_out = True
            break
        for f in sorted(files):
            scanned += 1
            if scanned % 200 == 0 and time.monotonic() > deadline:
                timed_out = True
                break
            full_path = Path(root) / f
            rel_str = full_path.relative_to(base).as_posix()

            if file_filter and not file_filter.search(rel_str):
                continue
            if not file_filter and NOISY_FILE.search(rel_str):
                continue

            # Skip binaries, symlinks (could point outside the sandbox) and huge files
            if full_path.suffix.lower() in BINARY_SUFFIXES or full_path.is_symlink():
                continue

            try:
                if full_path.stat().st_size > MAX_SEARCH_FILE_BYTES:
                    continue
                text = full_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            hits = 0
            for idx, line in enumerate(text.splitlines(), start=1):
                if not regex.search(line):
                    continue
                hits += 1
                if hits <= PER_FILE_MATCH_CAP:
                    truncated = line.strip()
                    if len(truncated) > 160:
                        truncated = truncated[:157] + "..."
                    per_file.setdefault(rel_str, []).append(f"{rel_str}:{idx}: {truncated}")
                    collected += 1
                else:
                    extra[rel_str] = extra.get(rel_str, 0) + 1
            if collected >= MAX_COLLECTED_MATCHES:
                break
        if timed_out or collected >= MAX_COLLECTED_MATCHES:
            break

    note = (
        f"\n[search stopped after {SEARCH_TIME_BUDGET:.0f}s ({scanned} files scanned); "
        "narrow it with file_pattern]"
        if timed_out
        else ""
    )
    if not per_file:
        return f"No matches found for pattern: '{pattern}'" + note

    matches: List[str] = []
    capped = False
    for rel_str in sorted(per_file, key=lambda r: (_search_priority(r), r)):
        for line in per_file[rel_str]:
            if len(matches) >= max_matches:
                capped = True
                break
            matches.append(line)
        if rel_str in extra and len(matches) < max_matches:
            matches.append(f"  ... (+{extra[rel_str]} more matches in {rel_str})")
        if capped:
            break
    shown = len([m for m in matches if not m.startswith("  ...")])
    header = f"Found {shown} match(es) in {len({m.split(':', 1)[0] for m in matches if not m.startswith('  ...')})} file(s) for '{pattern}':"
    if capped or collected >= MAX_COLLECTED_MATCHES:
        header += f" (capped at {max_matches}; narrow with file_pattern)"
    return "\n".join([header] + matches) + note


def find_secrets(repo_dir: str) -> str:
    """List lines that look like hardcoded credentials (tokens, private keys, passwords) across the repo.

    Heuristic leads for the agent to verify with read_file. Values are redacted in the output. Test
    fixtures, docs, ``.example`` files, lockfiles and placeholder/variable values are excluded, and at most
    5 hits per file are shown.
    """
    base = Path(repo_dir).resolve()
    if not base.exists() or not base.is_dir():
        return f"Error: Repository directory does not exist: {repo_dir}"

    deadline = time.monotonic() + SEARCH_TIME_BUDGET
    leads: List[str] = []
    excluded = 0
    scanned = 0
    timed_out = False
    for root, dirs, files in os.walk(base):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_SEARCH_DIRS)
        if time.monotonic() > deadline:
            timed_out = True
            break
        for f in sorted(files):
            scanned += 1
            if scanned % 200 == 0 and time.monotonic() > deadline:
                timed_out = True
                break
            full_path = Path(root) / f
            rel = full_path.relative_to(base).as_posix()
            low = f.lower()
            if (
                NOISY_FILE.search(rel)
                or full_path.suffix.lower() in BINARY_SUFFIXES
                or full_path.is_symlink()
            ):
                continue
            parts_lower = {part.lower() for part in rel.split("/")[:-1]}
            skip_context = (
                _search_priority(rel) > 0
                or ".example" in low
                or ".sample" in low
                or low.endswith((".template",) + NON_SECRET_SUFFIXES)
                or bool(parts_lower & VENDORED_DIRS)
            )
            try:
                if full_path.stat().st_size > MAX_SEARCH_FILE_BYTES:
                    continue
                text = full_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            hits = 0
            doc_lines = triple_quoted_lines(text.splitlines()) if full_path.suffix.lower() == ".py" else set()
            for idx, line in enumerate(text.splitlines(), start=1):
                kind = secret_lead_kind(line)
                if not kind:
                    continue
                if kind != "token/key pattern" and idx in doc_lines:
                    excluded += 1  # a code example inside a docstring is documentation, not a credential
                    continue
                if skip_context:
                    excluded += 1
                    continue
                hits += 1
                if hits <= PER_FILE_MATCH_CAP and len(leads) < MAX_SECRET_LEADS:
                    preview = sanitize_evidence(line.strip())
                    if len(preview) > 140:
                        preview = preview[:137] + "..."
                    leads.append(f"{rel}:{idx}: [{kind}] {preview}")
        if timed_out or len(leads) >= MAX_SECRET_LEADS:
            break

    note = f"\n[scan stopped after {SEARCH_TIME_BUDGET:.0f}s ({scanned} files)]" if timed_out else ""
    footer = (
        f"\nExcluded {excluded} hit(s) in tests/docs/examples; placeholders, booleans and variable references "
        "are never listed. These are leads: confirm with read_file that the value is a real credential before "
        "reporting, and never copy the value."
    )
    if not leads:
        return "No credential-looking strings found outside tests/docs/examples." + footer + note
    return f"Found {len(leads)} credential-looking line(s) (values redacted):\n" + "\n".join(leads) + footer + note


def git_log(repo_dir: str, max_commits: int = 25) -> str:
    """Inspect recent commit history and statistics."""
    base = Path(repo_dir).resolve()
    if not (base / ".git").exists():
        return "Not a git repository (missing .git directory)."

    cmd = [
        "git",
        "-C",
        str(base),
        "log",
        f"-n{max_commits}",
        "--pretty=format:%h | %ad | %an | %s",
        "--date=short",
        "--stat",
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
            env=_git_env(),
        )
        if proc.returncode != 0:
            return f"git log failed: {proc.stderr.strip()}"
        out = proc.stdout.strip()
        return out if out else "No git commit history available."
    except Exception as exc:
        return f"Error executing git log: {str(exc)}"


def git_status(repo_dir: str) -> str:
    """Check working tree status (untracked files, modified files)."""
    base = Path(repo_dir).resolve()
    if not (base / ".git").exists():
        return "Not a git repository."

    cmd = ["git", "-C", str(base), "status", "--porcelain"]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            env=_git_env(),
        )
        if proc.returncode != 0:
            return f"git status failed: {proc.stderr.strip()}"
        out = proc.stdout.strip()
        if not out:
            return "Working tree clean. All files tracked and committed."
        return f"Uncommitted / Untracked files present:\n{out}"
    except Exception as exc:
        return f"Error executing git status: {str(exc)}"


def run_ruff(repo_dir: str) -> str:
    """Optional safe linter execution (Section 3).

    Runs ``ruff check`` as a pure static analyser (never imports or executes repository
    code). ``--no-cache`` keeps the inspected repository byte-for-byte unchanged.
    """
    base = Path(repo_dir).resolve()
    args = ["check", "--output-format=json", "--no-cache", "--exit-zero", str(base)]
    candidates = [["ruff"], [sys.executable, "-m", "ruff"]]
    proc = None
    for prefix in candidates:
        try:
            attempt = subprocess.run(
                prefix + args,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except FileNotFoundError:
            continue
        except subprocess.TimeoutExpired:
            return "Ruff timed out after 60s (run_ruff skipped)."
        except Exception as exc:
            return f"Ruff inspection error: {str(exc)}"
        if attempt.returncode == 0 or attempt.stdout.strip():
            proc = attempt
            break

    if proc is None:
        return "Ruff is not installed in the current environment (run_ruff skipped)."

    raw = proc.stdout.strip()
    if not raw:
        return "Ruff reported no output (no Python files, or no issues)."
    try:
        data = json.loads(raw)
    except Exception:
        return f"Ruff output could not be parsed: {raw[:300]}"
    if not isinstance(data, list) or not data:
        return "Ruff found no violations."

    formatted: List[str] = [f"Ruff detected {len(data)} issue(s):"]
    for item in data[:25]:
        file_path = item.get("filename", "")
        try:
            rel = Path(file_path).relative_to(base).as_posix()
        except Exception:
            rel = file_path
        code = item.get("code") or "UNK"
        message = item.get("message", "")
        row = (item.get("location") or {}).get("row", "?")
        formatted.append(f"  {rel}:{row} [{code}] {message}")
    if len(data) > 25:
        formatted.append(f"  ... and {len(data) - 25} more issues.")
    return "\n".join(formatted)


WORKFLOW_SIGNALS = [
    re.compile(r"^\s*(on|permissions|concurrency)\s*:", re.I),
    re.compile(r"^\s*(pull_request_target|pull_request|push|workflow_dispatch|workflow_run|schedule|release|issue_comment|issues)\s*:?\s*$", re.I),
    re.compile(r"write-all|contents:\s*write|id-token:\s*write|packages:\s*write|pull-requests:\s*write", re.I),
    re.compile(r"\buses\s*:", re.I),
    re.compile(r"continue-on-error", re.I),
    re.compile(r"\bif\s*:\s*(false|\$\{\{\s*false)", re.I),
    re.compile(r"\$\{\{\s*(github\.event\.|github\.head_ref|inputs\.)", re.I),
    re.compile(r"\b(pytest|unittest|npm (run )?test|yarn test|go test|cargo test|mvn test|gradle test|tox|nox|ruff|flake8|pylint|eslint|black|mypy|docker build|twine|npm publish|deploy|kubectl|terraform)\b", re.I),
    re.compile(r"\|\|\s*true\b|--exit-zero|set \+e", re.I),
    re.compile(r"\bref:\s*\$\{\{", re.I),
    re.compile(r"secrets\.", re.I),
]


MUTABLE_REF = re.compile(r"\buses\s*:\s*\S+@(main|master|latest|head|develop|dev|trunk)\s*(#.*)?$", re.I)
PR_HEAD_REF = re.compile(
    r"github\.event\.pull_request\.head\.(sha|ref)|github\.head_ref|gh pr checkout|refs/pull/", re.I
)
INJECTABLE_CONTEXT = re.compile(
    r"\$\{\{\s*(github\.event\.(issue|pull_request|comment|review|review_comment|discussion|head_commit|commits|pages)"
    r"[\w.\[\]*-]*\.(title|body|name|label|ref|message|email|head_ref|default_branch)"
    r"|github\.head_ref|github\.event\.workflow_run\.head_branch)",
    re.I,
)
DEPRECATED_RUNNER = re.compile(r"runs-on\s*:\s*\[?\s*[\"']?(ubuntu-(16|18|20)\.04|windows-(2016|2019)|macos-(10|11|12)\b[\w.-]*)", re.I)
SECRET_ECHO = re.compile(r"\becho\b.*\$\{?\w*(secret|token|password|passwd|private_?key|api_?key|credential)\w*", re.I)


def _workflow_risks(rel: str, lines: List[str]) -> List[str]:
    """Heuristic risk leads for one workflow (the agent must verify and interpret them)."""
    risks: List[str] = []
    text = "\n".join(lines)
    has_prt = bool(re.search(r"^\s*pull_request_target\s*:?", text, re.M | re.I))
    has_permissions = bool(re.search(r"^\s*permissions\s*:", text, re.M | re.I))
    run_indent = None  # indentation of the current run: block, if any

    def add(idx: int, tag: str, line: str) -> None:
        risks.append(f"{rel}:{idx}: [{tag}] {line.strip()[:140]}")

    if has_prt:
        for idx, line in enumerate(lines, start=1):
            if re.match(r"^\s*pull_request_target\s*:?", line, re.I):
                add(idx, "PULL_REQUEST_TARGET trigger (runs with secrets/write token; unsafe if PR code is checked out or executed)", line)
                break

    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if run_indent is not None and indent <= run_indent:
            run_indent = None
        if re.match(r"^\s*-?\s*run\s*:", line):
            run_indent = indent
            if re.search(r"run\s*:\s*[|>]", line):
                pass  # block scalar; following deeper-indented lines are script lines
        in_run = run_indent is not None and (indent > run_indent or re.match(r"^\s*-?\s*run\s*:\s*\S", line))

        if re.search(r"permissions\s*:\s*write-all", line, re.I):
            add(idx, "WRITE-ALL permissions", line)
        if MUTABLE_REF.search(line):
            add(idx, "ACTION pinned to mutable branch/latest", line)
        if has_prt and PR_HEAD_REF.search(line):
            add(idx, "PR head code checked out / referenced in pull_request_target workflow", line)
        if in_run and INJECTABLE_CONTEXT.search(line):
            add(idx, "SCRIPT-INJECTION untrusted context interpolated in run:", line)
        if re.search(r"continue-on-error\s*:\s*(true|\$\{\{)", line, re.I):
            add(idx, "CONTINUE-ON-ERROR hides failures", line)
        if in_run and re.search(r"\|\|\s*(true|exit 0)\b", line):
            add(idx, "FORCED-SUCCESS (|| true) hides failures", line)
        if in_run and SECRET_ECHO.search(line):
            add(idx, "SECRET printed to logs", line)
        if DEPRECATED_RUNNER.search(line):
            add(idx, "DEPRECATED/retired runner image", line)
        if re.match(r"^\s*if\s*:\s*(false|\$\{\{\s*false\s*\}\})\s*$", line, re.I):
            add(idx, "DISABLED job/step (if: false)", line)

    if not has_permissions:
        risks.append(f"{rel}:1: [NO permissions: block] token scope depends on repo defaults (low-severity lead)")
    return risks


def inspect_workflows(repo_dir: str) -> str:
    """GitHub Actions inspection: risk leads plus a filtered, line-numbered view of every workflow.

    Line numbers match ``read_file`` so findings can cite them. Risk leads are heuristics:
    the agent must read the workflow and decide whether each one is a genuine defect.
    """
    base = Path(repo_dir).resolve()
    wf_dir = base / ".github" / "workflows"
    if not wf_dir.is_dir():
        return "No .github/workflows directory found (no GitHub Actions workflows in this repository)."

    files = sorted(
        p for p in wf_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".yml", ".yaml"} and not p.is_symlink()
    )
    if not files:
        return ".github/workflows exists but contains no .yml/.yaml workflow files."

    loaded = []
    all_risks: List[str] = []
    for wf in files:
        rel = wf.relative_to(base).as_posix()
        try:
            lines = wf.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception as exc:
            loaded.append((rel, None, str(exc)))
            continue
        loaded.append((rel, lines, ""))
        all_risks.extend(_workflow_risks(rel, lines))

    out: List[str] = [f"Found {len(files)} workflow file(s)."]
    if all_risks:
        out.append("\nRISK LEADS (heuristic; verify each with read_file before reporting):")
        out.extend(f"  {r}" for r in all_risks[:70])
        if len(all_risks) > 70:
            out.append(f"  ... and {len(all_risks) - 70} more leads (read individual workflows)")
    else:
        out.append("\nNo heuristic risk leads detected.")

    out.append("\nWORKFLOW SUMMARIES (security/CI-relevant lines):")
    for rel, lines, err in loaded:
        if lines is None:
            out.append(f"\n=== {rel}: unreadable ({err}) ===")
            continue
        if not lines:
            out.append(f"\n=== {rel}: EMPTY FILE (0 lines) ===")
            continue
        out.append(f"\n=== {rel} ({len(lines)} lines) ===")
        shown = 0
        for idx, line in enumerate(lines, start=1):
            if line.strip().startswith("#"):
                continue
            if any(rx.search(line) for rx in WORKFLOW_SIGNALS):
                out.append(f"{idx:04d}: {line.rstrip()[:160]}")
                shown += 1
                if shown >= 40:
                    out.append("  ... (truncated; use read_file for the full workflow)")
                    break
    out.append("\nNote: filtered view; use read_file for full context before reporting.")
    return "\n".join(out)


def git_history_search(repo_dir: str, pattern: str, max_results: int = 15) -> str:
    """Search commit history diffs for a regex (e.g. secrets that were later deleted).

    Uses ``git log -G`` (pickaxe on added/removed lines). The clone is shallow, so only
    recent history is covered.
    """
    base = Path(repo_dir).resolve()
    if not (base / ".git").exists():
        return "Not a git repository."
    try:
        re.compile(pattern)
    except re.error as e:
        return f"Error: Invalid regular expression '{pattern}': {e}"
    max_results = max(1, min(int(max_results), 40))
    cmd = [
        "git", "-C", str(base), "log", "-i", f"-G{pattern}", f"-n{max_results}",
        "--pretty=format:%h | %ad | %an | %s", "--date=short", "--name-only",
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30, check=False, env=_git_env()
        )
    except subprocess.TimeoutExpired:
        return "git history search timed out."
    except Exception as exc:
        return f"Error executing git history search: {exc}"
    if proc.returncode != 0:
        return f"git history search failed: {proc.stderr.strip()}"
    out = proc.stdout.strip()
    if not out:
        return f"No commits in the (shallow) history touch lines matching: '{pattern}'"
    return f"Commits whose diffs add/remove lines matching '{pattern}':\n{out}"


# Tool Dispatch Table
AVAILABLE_TOOLS: Dict[str, Any] = {
    "list_files": list_files,
    "read_file": read_file,
    "search_code": search_code,
    "git_log": git_log,
    "git_status": git_status,
    "run_ruff": run_ruff,
    "inspect_workflows": inspect_workflows,
    "find_secrets": find_secrets,
    "git_history_search": git_history_search,
}


def dispatch_tool_call(repo_dir: str, tool_name: str, arguments: Dict[str, Any]) -> str:
    """Safely dispatch tool calls requested by the LLM agent.

    Enforces try/except guardrail: returns formatted error string instead of crashing.
    """
    tool_func = AVAILABLE_TOOLS.get(tool_name)
    if not tool_func:
        return f"Error: Unknown tool '{tool_name}'. Available tools are: {list(AVAILABLE_TOOLS.keys())}"

    # Prepend or inject repo_dir
    kwargs = dict(arguments)
    kwargs["repo_dir"] = repo_dir

    try:
        result = str(tool_func(**kwargs))
        if len(result) > MAX_TOOL_OUTPUT_CHARS:
            result = (
                result[:MAX_TOOL_OUTPUT_CHARS]
                + f"\n... [output truncated at {MAX_TOOL_OUTPUT_CHARS} chars; narrow the request]"
            )
        return result
    except TypeError as e:
        return f"Error in tool arguments for '{tool_name}': {str(e)}"
    except Exception as e:
        return f"Error executing tool '{tool_name}': {str(e)}"
