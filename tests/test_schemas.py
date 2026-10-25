"""Unit tests for schemas.py, validation, deduplication, and secret sanitization."""

from schemas import AuditReport, Violation, sanitize_evidence


def test_sanitize_evidence_redacts_api_keys():
    raw = 'api_key = "sk-1234567890abcdef1234567890abcdef"'
    sanitized = sanitize_evidence(raw)
    assert "sk-1234567890abcdef1234567890abcdef" not in sanitized
    assert "[REDACTED_SECRET]" in sanitized


def test_sanitize_evidence_redacts_github_tokens():
    raw = "Bearer ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
    sanitized = sanitize_evidence(raw)
    assert "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890" not in sanitized
    assert "[REDACTED_GITHUB_TOKEN]" in sanitized or "[REDACTED_TOKEN]" in sanitized


def test_sanitize_evidence_redacts_aws_credentials():
    raw = 'AWS_KEY = "AKIA1234567890ABCDEF"'
    sanitized = sanitize_evidence(raw)
    assert "AKIA1234567890ABCDEF" not in sanitized
    assert "[REDACTED_AWS_KEY_ID]" in sanitized


def test_violation_schema_valid():
    v = Violation(
        rule_id="SEC-001",
        category="security",
        severity="critical",
        file="config.py",
        line=17,
        description="Hardcoded credential",
        evidence="API_KEY = 'sk-abcdef1234567890'",
        recommendation="Use environment variables",
    )
    assert v.rule_id == "SEC-001"
    assert v.category == "security"
    assert v.severity == "critical"
    assert v.file == "config.py"
    assert v.line == 17
    # Secret should be sanitized automatically
    assert "sk-abcdef1234567890" not in v.evidence
    assert "[REDACTED_SECRET]" in v.evidence


def test_violation_null_handling():
    v = Violation(
        rule_id="HYG-001",
        category="repository_hygiene",
        severity="medium",
        file=None,
        line=None,
        description="Missing README",
        evidence="README.md not found in repository root",
        recommendation="Add a descriptive README.md",
    )
    dumped = v.model_dump()
    assert dumped["file"] is None
    assert dumped["line"] is None


def test_violation_category_normalization():
    v = Violation(
        rule_id="CI-001",
        category="GitHub Actions / CI",
        severity="high",
        file=".github/workflows/ci.yml",
        line=10,
        description="Dangerous workflow trigger",
        evidence="on: pull_request_target",
        recommendation="Use pull_request instead",
    )
    assert v.category == "github_actions_ci"


def test_audit_report_deduplication():
    v1 = Violation(
        rule_id="SEC-001",
        category="security",
        severity="critical",
        file="config.py",
        line=10,
        description="Hardcoded API key",
        evidence="api_key = '[REDACTED]'",
        recommendation="Use env vars",
    )
    v2 = Violation(
        rule_id="SEC-001",
        category="security",
        severity="critical",
        file="config.py",
        line=10,
        description="Hardcoded API key",
        evidence="api_key = '[REDACTED]'",
        recommendation="Use env vars",
    )
    v3 = Violation(
        rule_id="DEP-001",
        category="dependency_management",
        severity="low",
        file="requirements.txt",
        line=1,
        description="Unpinned version",
        evidence="requests",
        recommendation="Pin version",
    )

    report = AuditReport(
        repository="https://github.com/example/repo",
        violations=[v1, v2, v3],
    )
    assert len(report.violations) == 3
    report.deduplicate()
    assert len(report.violations) == 2


def test_sanitize_redacts_credentials_in_urls():
    out = sanitize_evidence("curl https://admin:hunter2pass@example.com/basic_auth")
    assert "hunter2pass" not in out and "admin:" not in out
    assert "https://[REDACTED_CREDENTIALS]@example.com/basic_auth" in out
    assert sanitize_evidence("https://github.com/owner/repo") == "https://github.com/owner/repo"


def _v(**over):
    base = dict(rule_id="CI-001", category="github_actions_ci", severity="high", file="a.yml", line=1,
                description="Actions pinned to mutable branches", evidence="uses: x@master",
                recommendation="pin to SHA")
    base.update(over)
    return Violation(**base)


def test_dedup_merges_same_pattern_across_files_for_pattern_categories():
    report = AuditReport(repository="r", violations=[
        _v(file="a.yml", line=3), _v(file="b.yml", line=7), _v(file="c.yml", line=None),
    ]).deduplicate()
    assert len(report.violations) == 1
    kept = report.violations[0]
    assert kept.file == "a.yml"
    assert "b.yml:7" in kept.evidence and "c.yml" in kept.evidence and "Also affects" in kept.evidence


def test_dedup_keeps_security_findings_per_file():
    report = AuditReport(repository="r", violations=[
        _v(category="security", rule_id="SEC-001", file="a.py", description="Hardcoded credential"),
        _v(category="security", rule_id="SEC-002", file="b.py", description="Hardcoded credential"),
    ]).deduplicate()
    assert len(report.violations) == 2


def test_dedup_keeps_distinct_descriptions_in_same_category():
    report = AuditReport(repository="r", violations=[
        _v(description="Actions pinned to mutable branches"),
        _v(description="Retired runner image", file="b.yml"),
    ]).deduplicate()
    assert len(report.violations) == 2


def test_sanitize_redacts_unquoted_assignments_but_not_prose():
    out = sanitize_evidence("export API_KEY=abcdef123456789 and token: s3cr3tvalue99")
    assert "abcdef123456789" not in out and "s3cr3tvalue99" not in out
    # short values / ordinary prose are left alone
    assert sanitize_evidence("token=abc") == "token=abc"
    assert sanitize_evidence("The token is rotated weekly") == "The token is rotated weekly"
