"""Markdown files with a YAML front-matter block."""
from __future__ import annotations

import re
from typing import Any

import yaml


_FM = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*(?:\n|\Z)(.*)\Z", re.DOTALL)


def parse(text: str) -> tuple[dict[str, Any], str]:
    """Split `---\\n<yaml>\\n---\\n<body>`. Files without front matter return ({}, text)."""
    m = _FM.match(text)
    if not m:
        return {}, text
    meta = yaml.safe_load(m.group(1)) or {}
    if not isinstance(meta, dict):
        raise ValueError("front matter must be a YAML mapping")
    return meta, m.group(2).lstrip("\n")


def dump(meta: dict[str, Any], body: str) -> str:
    head = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True, width=1000).strip()
    return f"---\n{head}\n---\n\n{body.lstrip()}"
