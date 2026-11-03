"""SQLite FTS5 index over the memory repo's Markdown files."""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path


class SearchIndex:
    def __init__(self, db_path: Path, root: Path):
        self.db_path = Path(db_path)
        self.root = Path(root)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def rebuild(self) -> int:
        """Drop and rebuild the index from every *.md under root. Returns doc count."""
        docs = []
        for p in sorted(self.root.rglob("*.md")):
            if ".git" in p.relative_to(self.root).parts:
                continue
            text = p.read_text(encoding="utf-8", errors="replace")
            m = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
            title = m.group(1).strip() if m else p.stem
            docs.append((str(p.relative_to(self.root)), title, text))
        with self._connect() as conn:
            conn.execute("DROP TABLE IF EXISTS docs")
            conn.execute(
                "CREATE VIRTUAL TABLE docs USING fts5("
                "path UNINDEXED, title, body, tokenize='porter unicode61')"
            )
            conn.executemany("INSERT INTO docs(path, title, body) VALUES (?, ?, ?)", docs)
        return len(docs)

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """Full-text search. Free text is tokenised and OR-ed, ranked by BM25."""
        tokens = re.findall(r"\w+", query.lower())
        if not tokens:
            return []
        fts_query = " OR ".join(f'"{t}"' for t in tokens)
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT path, title, snippet(docs, 2, '[', ']', '…', 16), bm25(docs) "
                    "FROM docs WHERE docs MATCH ? ORDER BY bm25(docs) LIMIT ?",
                    (fts_query, limit),
                ).fetchall()
        except sqlite3.OperationalError:  # index not built yet
            return []
        return [
            {"path": r[0], "title": r[1], "snippet": r[2], "score": round(-r[3], 3)}
            for r in rows
        ]
