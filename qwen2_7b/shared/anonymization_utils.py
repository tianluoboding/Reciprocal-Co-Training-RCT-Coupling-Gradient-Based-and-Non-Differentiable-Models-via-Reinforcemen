"""Utilities for checking anonymous-release redaction."""

from __future__ import annotations

import os
import re
from pathlib import Path

EMAIL_PATTERN = r"[A-Za-z0-9._%+-]+" + chr(64) + r"[A-Za-z0-9.-]+\.[A-Za-z]{2,}"


def default_forbidden_patterns() -> list[str]:
    """Return redaction patterns from the REVIEW_FORBIDDEN_PATTERNS env var.

    Keep reviewer-specific identity strings outside the public source tree.
    Separate multiple regex patterns with semicolons.
    """
    raw = os.environ.get("REVIEW_FORBIDDEN_PATTERNS", "")
    patterns = [part for part in raw.split(";") if part]
    return [EMAIL_PATTERN, *patterns]


def scan_text(path: Path, patterns: list[str] | None = None) -> list[dict[str, object]]:
    compiled = [re.compile(p) for p in (patterns or default_forbidden_patterns())]
    hits = []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return hits
    for line_no, line in enumerate(lines, start=1):
        for pattern in compiled:
            if pattern.search(line):
                hits.append({"path": str(path), "line": line_no, "pattern": pattern.pattern})
    return hits


def scan_repo(root: str | Path) -> list[dict[str, object]]:
    base = Path(root)
    hits = []
    skip_dirs = {".git", "__pycache__", ".pytest_cache", ".mypy_cache"}
    for path in base.rglob("*"):
        if any(part in skip_dirs for part in path.parts):
            continue
        if path.is_file():
            hits.extend(scan_text(path))
    return hits
