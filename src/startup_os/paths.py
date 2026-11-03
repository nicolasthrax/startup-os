"""Validation of file paths that agents reference.

Agents (CEO, Research) may only reference files that exist inside company/.
`check_path` normalises a reference and returns (relative_path, problem); an
unknown path makes the item that carries it invalid, and the caller logs it."""
from __future__ import annotations

from pathlib import Path, PurePosixPath

from .memory import MemoryRepo
from .runlog import NullRunLog, RunLog


def check_path(memory: MemoryRepo, path: str, *, under: str | None = None,
               suffix: str | None = None) -> tuple[str | None, str | None]:
    """Return (path relative to company/, None) if valid, else (None, reason)."""
    raw = str(path or "").strip().strip("`'\"")
    if not raw:
        return None, "empty path"
    p = Path(raw)
    if p.is_absolute():
        try:
            p = p.resolve().relative_to(memory.root)
        except ValueError:
            return None, "outside company/"
    parts = PurePosixPath(p.as_posix()).parts
    if parts and parts[0] == memory.root.name:   # "company/..." from the project root
        parts = parts[1:]
    if not parts:
        return None, "empty path"
    if ".." in parts:
        return None, "contains '..'"
    if ".git" in parts:
        return None, "inside .git"
    rel = PurePosixPath(*parts).as_posix()
    if under and not (rel == under or rel.startswith(under.rstrip("/") + "/")):
        return None, f"not under {under}/"
    if suffix and not rel.endswith(suffix):
        return None, f"not a {suffix} file"
    full = (memory.root / rel).resolve()
    if memory.root not in full.parents:
        return None, "outside company/"
    if not full.is_file():
        return None, "no such file in company/"
    return rel, None


def filter_paths(memory: MemoryRepo, paths: list[str], *, agent: str, item: str,
                 log: RunLog | None = None, **kw) -> tuple[list[str], list[dict]]:
    """Split references into (valid relative paths, invalid entries). Invalid ones are logged."""
    log = log or NullRunLog()
    ok, bad = [], []
    for raw in paths or []:
        rel, problem = check_path(memory, raw, **kw)
        if problem:
            entry = {"path": raw, "reason": problem}
            bad.append(entry)
            log.event("invalid_path", agent=agent, item=item, **entry)
        else:
            ok.append(rel)
    return ok, bad
