"""Two kinds of memory.

WorkingMemory - facts discovered during *this* run (invoice amount, due date ...),
each with its source. It is re-shown to the model on every step, so facts survive
even after old observations are trimmed from the context window. The verifier
also uses the sources to re-check the facts independently.

SiteNotes - small, durable lessons about a site that persist *across* runs
("ERP due date must be DD-MM-YYYY"). Next time the agent visits that host it
gets the note up front and avoids repeating the same mistake.
"""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse


class WorkingMemory:
    def __init__(self):
        self.facts: dict[str, dict] = {}

    def remember(self, key: str, value: str, source: str, step: int) -> None:
        self.facts[key] = {"value": value, "source": source, "step": step}

    def render(self) -> str:
        if not self.facts:
            return "(nothing saved yet)"
        return "\n".join(f"- {k}: {v['value']}   (source: {v['source']})"
                         for k, v in self.facts.items())


class SiteNotes:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.notes: dict[str, list[str]] = {}
        if self.path.exists():
            try:
                self.notes = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                self.notes = {}

    @staticmethod
    def key_for(url_or_site: str) -> str:
        parsed = urlparse(url_or_site)
        if parsed.hostname:
            first = parsed.path.strip("/").split("/")[0]
            return f"{parsed.hostname}/{first}" if first else parsed.hostname
        return url_or_site.strip().lower()

    def add(self, site: str, note: str) -> None:
        key = self.key_for(site)
        items = self.notes.setdefault(key, [])
        if note not in items:
            items.append(note)
            items[:] = items[-10:]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.notes, indent=2))

    def render(self) -> str:
        if not self.notes:
            return "(none yet)"
        return "\n".join(f"- {site}: {n}" for site, ns in self.notes.items() for n in ns)
