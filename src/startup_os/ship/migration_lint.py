"""Static backward-compatibility check for SQL migrations (run before staging migrate).

Migrations must be additive, so the code already in production keeps working while
and after they run:
  - no DROP (except `ALTER COLUMN ... DROP NOT NULL`, which only relaxes a constraint)
  - no RENAME
  - no NOT NULL without a DEFAULT on existing tables: `ADD COLUMN ... NOT NULL` needs a
    DEFAULT, and `ALTER COLUMN ... SET NOT NULL` is not allowed
  - also non-additive: `ALTER COLUMN ... TYPE` and `TRUNCATE`
New tables (`CREATE TABLE`) may use NOT NULL freely."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Violation:
    file: str
    statement: int
    rule: str
    sql: str

    def __str__(self) -> str:
        return f"{self.file} (statement {self.statement}): {self.rule} — `{self.sql[:120]}`"


def _strip(sql: str) -> str:
    """Remove comments, dollar-quoted bodies and string literals so keywords inside them don't count."""
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    sql = re.sub(r"--[^\n]*", " ", sql)
    sql = re.sub(r"\$([A-Za-z_]*)\$.*?\$\1\$", " $$ ", sql, flags=re.DOTALL)
    sql = re.sub(r"'(?:[^']|'')*'", "''", sql)
    return sql


def statements(sql: str) -> list[str]:
    return [" ".join(s.split()) for s in _strip(sql).split(";") if s.strip()]


_ADD_COL = re.compile(r"\bADD\s+(?:COLUMN\s+)?(?:IF\s+NOT\s+EXISTS\s+)?(.+?)(?=,\s*(?:ADD|ALTER|DROP)\b|$)", re.I)


def check_statement(stmt: str) -> list[str]:
    s = stmt.upper()
    problems = []
    if re.search(r"\bDROP\b", s):
        without_relax = re.sub(r"\bALTER\s+(?:COLUMN\s+)?\S+\s+DROP\s+NOT\s+NULL\b", "", s)
        if re.search(r"\bDROP\b", without_relax):
            problems.append("DROP is not backward-compatible (migrations must be additive)")
    if re.search(r"\bRENAME\b", s):
        problems.append("RENAME is not backward-compatible (add a new column/table instead)")
    if re.search(r"\bTRUNCATE\b", s):
        problems.append("TRUNCATE deletes data (migrations must be additive)")
    if s.startswith("ALTER TABLE"):
        if re.search(r"\bSET\s+NOT\s+NULL\b", s):
            problems.append("SET NOT NULL on an existing column is not backward-compatible")
        if re.search(r"\bALTER\s+(?:COLUMN\s+)?\S+\s+(?:SET\s+DATA\s+)?TYPE\b", s):
            problems.append("changing a column TYPE is not backward-compatible")
        for m in _ADD_COL.finditer(s):
            col = m.group(1)
            if col.startswith(("CONSTRAINT", "PRIMARY", "UNIQUE", "FOREIGN", "CHECK")):
                continue
            if re.search(r"\bNOT\s+NULL\b", col) and not re.search(r"\bDEFAULT\b", col):
                problems.append("ADD COLUMN ... NOT NULL needs a DEFAULT on an existing table")
    return problems


def lint_sql(sql: str, file: str = "<sql>") -> list[Violation]:
    out = []
    for i, stmt in enumerate(statements(sql), 1):
        for rule in check_statement(stmt):
            out.append(Violation(file, i, rule, stmt))
    return out


def lint_files(paths: list[Path], root: Path | None = None) -> list[Violation]:
    out = []
    for p in paths:
        name = str(p.relative_to(root)) if root else p.name
        out += lint_sql(p.read_text(encoding="utf-8", errors="replace"), name)
    return out
