"""Data contracts, schema validation, and security sanitization for DevOps Auditing Agent.

Conforms to Section 5 of the assignment specifications.
"""

from __future__ import annotations

import re
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

# Allowed severity levels per specification
SeverityType = Literal["critical", "high", "medium", "low"]

# 9 Canonical DevOps categories from Section 4
VALID_CATEGORIES = {
    "repository_hygiene": "repository_hygiene",
    "hygiene": "repository_hygiene",
    "dependency_management": "dependency_management",
    "dependencies": "dependency_management",
    "code_quality": "code_quality",
    "testing": "testing",
    "github_actions_ci": "github_actions_ci",
    "github_actions": "github_actions_ci",
    "ci": "github_actions_ci",
    "ci_cd": "github_actions_ci",
    "security": "security",
    "configuration": "configuration",
    "collaboration": "collaboration",
    "build_and_release": "build_and_release",
    "build_release": "build_and_release",
}

# Fallback keyword -> category mapping for free-form category labels from the LLM
CATEGORY_KEYWORDS = [
    ("secur", "security"),
    ("secret", "security"),
    ("hygiene", "repository_hygiene"),
    ("repo", "repository_hygiene"),
    ("depend", "dependency_management"),
    ("lock", "dependency_management"),
    ("quality", "code_quality"),
    ("lint", "code_quality"),
    ("test", "testing"),
    ("action", "github_actions_ci"),
    ("workflow", "github_actions_ci"),
    ("ci", "github_actions_ci"),
    ("config", "configuration"),
    ("collab", "collaboration"),
    ("build", "build_and_release"),
    ("release", "build_and_release"),
    ("deploy", "build_and_release"),
    ("docker", "build_and_release"),
]

# Categories where one pattern repeated across files is a single defect to fix once
MERGEABLE_CATEGORIES = {
    "github_actions_ci",
    "dependency_management",
    "build_and_release",
    "code_quality",
    "repository_hygiene",
    "testing",
}

# Regex patterns to detect and redact real credentials/secrets in evidence
SECRET_PATTERNS = [
    # Generic assignments: api_key = "...", token = '...', password = "..."
    (
        re.compile(
            r"""(?i)(api[_-]?key|access[_-]?token|auth[_-]?token|secret[_-]?key|password|passwd|pwd|private[_-]?key|client[_-]?secret)\s*[:=]\s*["']([^"'\s]{6,})["']"""
        ),
        r'\1 = "[REDACTED_SECRET]"',
    ),
    # LLM provider API keys (Anthropic before OpenAI: sk-ant- also matches the generic sk- shape)
    (re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}"), "[REDACTED_ANTHROPIC_KEY]"),
    (re.compile(r"\bsk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_-]{20,}"), "[REDACTED_OPENAI_KEY]"),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{30,}"), "[REDACTED_GOOGLE_KEY]"),
    (re.compile(r"\bgsk_[A-Za-z0-9]{20,}"), "[REDACTED_GROQ_KEY]"),
    # Unquoted assignments: token=abc123..., SECRET: xyz..., export API_KEY=...
    (
        re.compile(
            r"""(?i)\b(api[_-]?key|access[_-]?token|auth[_-]?token|secret[_-]?key|client[_-]?secret|token|secret|password|passwd|pwd)\b(\s*[:=]\s*)(?!\[REDACTED)[^\s"',;]{8,}"""
        ),
        r"\1\2[REDACTED_SECRET]",
    ),
    # GitHub personal access tokens: ghp_, gho_, ghu_, etc.
    (re.compile(r"\b(gh[pousr]_[A-Za-z0-9_]{36,})\b"), "[REDACTED_GITHUB_TOKEN]"),
    # AWS access key ID and secret access key
    (re.compile(r"\b(AKIA[0-9A-Z]{16})\b"), "[REDACTED_AWS_KEY_ID]"),
    (
        re.compile(r"""(?i)(aws_secret_access_key\s*[:=]\s*["']?)[A-Za-z0-9/+=]{40}(["']?)"""),
        r"\1[REDACTED_AWS_SECRET]\2",
    ),
    # Slack bot tokens
    (re.compile(r"\b(xox[baprs]-[0-9A-Za-z-]{20,})\b"), "[REDACTED_SLACK_TOKEN]"),
    # JWT tokens
    (
        re.compile(r"\beyJ[A-Za-z0-9-_]+\.eyJ[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\b"),
        "[REDACTED_JWT_TOKEN]",
    ),
    # Private Key headers/bodies
    (
        re.compile(
            r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----[^-]+-----END (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
            re.DOTALL,
        ),
        "[REDACTED_PRIVATE_KEY_BLOCK]",
    ),
    # Credentials embedded in URLs: scheme://user:password@host
    (
        re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^\s/:@]+:[^\s/@]+@"),
        r"\1[REDACTED_CREDENTIALS]@",
    ),
    # Generic bearer tokens
    (
        re.compile(r"""(?i)(bearer\s+)[A-Za-z0-9\-._~+/]+=*"""),
        r"\1[REDACTED_TOKEN]",
    ),
]


def sanitize_evidence(evidence_text: str) -> str:
    """Scrub sensitive credentials, passwords, and tokens from evidence strings.

    Section 5 & 9 strict constraint: Never reproduce secret values.
    """
    if not evidence_text:
        return ""
    sanitized = evidence_text
    for pattern, replacement in SECRET_PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)
    return sanitized


class Violation(BaseModel):
    """Represents a single DevOps best-practice violation.

    Fields must strictly conform to Section 5:
    - rule_id: Identifier (e.g., 'SEC-001', 'HYG-001')
    - category: One of the 9 canonical DevOps categories
    - severity: critical, high, medium, low
    - file: Relative path or None (null in JSON)
    - line: 1-indexed line number or None (null in JSON)
    - description: Concise statement of the violation
    - evidence: Factual observation / code snippet (sanitized)
    - recommendation: Actionable remediation guidance
    """

    rule_id: str = Field(
        ...,
        description="Standardized rule identifier (e.g. SEC-001, HYG-001, CI-001)",
    )
    category: str = Field(
        ...,
        description="One of the 9 DevOps areas (e.g. security, repository_hygiene, testing)",
    )
    severity: SeverityType = Field(
        ...,
        description="Severity level: critical, high, medium, or low",
    )
    file: Optional[str] = Field(
        default=None,
        description="Relative file path within the repo, or null if repo-wide",
    )
    line: Optional[int] = Field(
        default=None,
        description="Exact 1-indexed line number where violation occurs, or null",
    )
    description: str = Field(
        ...,
        description="Clear, concise summary of the defect",
    )
    evidence: str = Field(
        ...,
        description="Direct factual evidence from code or configuration (secrets redacted)",
    )
    recommendation: str = Field(
        ...,
        description="Actionable, prescriptive recommendation to fix the violation",
    )

    @field_validator("category", mode="before")
    @classmethod
    def normalize_category(cls, v: str) -> str:
        if not isinstance(v, str):
            return "repository_hygiene"
        # Collapse multiple whitespace, slashes, hyphens into a single underscore
        key = re.sub(r"[\s\-_/]+", "_", v.strip().lower())
        if key in VALID_CATEGORIES:
            return VALID_CATEGORIES[key]
        for fragment, canonical in CATEGORY_KEYWORDS:
            if fragment in key:
                return canonical
        return key

    @field_validator("severity", mode="before")
    @classmethod
    def normalize_severity(cls, v: str) -> str:
        if not isinstance(v, str):
            return "medium"
        val = v.strip().lower()
        if val in {"critical", "high", "medium", "low"}:
            return val
        if val in {"warn", "warning"}:
            return "medium"
        if val in {"info", "informational"}:
            return "low"
        if val in {"fatal", "blocker", "error"}:
            return "high"
        return "medium"

    @field_validator("line", mode="before")
    @classmethod
    def validate_line(cls, v: Union[int, str, None]) -> Optional[int]:
        if v is None or v == "" or v == 0:
            return None
        try:
            val = int(v)
            return val if val > 0 else None
        except (ValueError, TypeError):
            return None

    @field_validator("file", mode="before")
    @classmethod
    def validate_file(cls, v: Optional[str]) -> Optional[str]:
        if not v or v.strip().lower() in {"none", "null", "n/a", ""}:
            return None
        # Normalize forward slashes for cross-platform consistency
        return v.strip().replace("\\", "/")

    @model_validator(mode="after")
    def scrub_and_verify(self) -> Violation:
        # Guarantee evidence never contains raw leaked secrets
        if self.evidence:
            self.evidence = sanitize_evidence(self.evidence)
        return self


class AuditReport(BaseModel):
    """Complete machine-readable JSON audit report matching Section 5."""

    repository: str = Field(
        ...,
        description="The inspected GitHub repository URL",
    )
    violations: List[Violation] = Field(
        default_factory=list,
        description="List of detected DevOps best-practice violations",
    )

    def deduplicate(self) -> AuditReport:
        """Remove duplicate findings.

        1. Exact duplicates (same file, line, rule_id and description) are dropped.
        2. The same *pattern* reported once per file (e.g. "actions pinned to mutable branches" in
           five workflows) is merged into one finding that names the other files in its evidence.
           Only for pattern-style categories; security and configuration findings stay per-file
           because each leaked credential is a distinct defect.
        """
        seen = set()
        unique_violations: List[Violation] = []
        for v in self.violations:
            key = (
                v.file or "",
                v.line or 0,
                v.rule_id,
                v.description.strip().lower(),
            )
            if key not in seen:
                seen.add(key)
                unique_violations.append(v)

        merged: List[Violation] = []
        by_pattern = {}
        also = {}  # id(first finding) -> other locations with the same pattern
        for v in unique_violations:
            if v.category not in MERGEABLE_CATEGORIES:
                merged.append(v)
                continue
            pattern_key = (v.category, re.sub(r"\W+", " ", v.description.lower()).strip())
            first = by_pattern.get(pattern_key)
            if first is None:
                by_pattern[pattern_key] = v
                also[id(v)] = []
                merged.append(v)
            elif v.file and v.file != first.file:
                also[id(first)].append(f"{v.file}:{v.line}" if v.line else v.file)
        for first in by_pattern.values():
            others = also[id(first)]
            if others:
                shown = ", ".join(others[:8]) + (f" (+{len(others) - 8} more)" if len(others) > 8 else "")
                first.evidence = f"{first.evidence} Also affects: {shown}."
        self.violations = merged
        return self
