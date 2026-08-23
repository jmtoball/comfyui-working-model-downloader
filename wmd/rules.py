"""User-taught overrides.

Every time someone resolves an item by hand -- picking a folder, or pasting a URL
for a file we could not find -- we remember it here, so the same workflow (or the
next one referencing the same file) resolves itself next time.
"""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import asdict, dataclass
from typing import Any

from . import comfy_env

RULES_FILENAME = "rules.json"
_lock = threading.Lock()


@dataclass
class Rule:
    pattern: str
    field: str = "filename"  # filename | url
    folder: str | None = None
    url: str | None = None
    regex: bool = False

    def matches(self, *, filename: str = "", url: str = "") -> bool:
        subject = filename if self.field == "filename" else url
        if not subject or not self.pattern:
            return False
        if self.regex:
            try:
                return re.search(self.pattern, subject, re.IGNORECASE) is not None
            except re.error:
                return False
        return self.pattern.lower() in subject.lower()


def rules_path() -> str:
    return os.path.join(comfy_env.user_dir(), RULES_FILENAME)


def load() -> list[Rule]:
    try:
        with open(rules_path(), encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return []
    entries = data.get("rules") if isinstance(data, dict) else data
    if not isinstance(entries, list):
        return []
    out: list[Rule] = []
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("pattern"):
            continue
        out.append(
            Rule(
                pattern=str(entry["pattern"]),
                field=str(entry.get("field") or "filename"),
                folder=entry.get("folder") or None,
                url=entry.get("url") or None,
                regex=bool(entry.get("regex")),
            )
        )
    return out


def save(entries: list[Rule]) -> None:
    path = rules_path()
    with _lock:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump({"version": 1, "rules": [asdict(r) for r in entries]}, handle, indent=2)
        os.replace(tmp, path)


def remember(rule: Rule) -> list[Rule]:
    """Add a rule, replacing any earlier one with the same pattern and field."""
    entries = [r for r in load() if not (r.pattern == rule.pattern and r.field == rule.field)]
    entries.append(rule)
    save(entries)
    return entries


def lookup(*, filename: str = "", url: str = "") -> Rule | None:
    for rule in load():
        if rule.matches(filename=filename, url=url):
            return rule
    return None


def to_json() -> list[dict[str, Any]]:
    return [asdict(rule) for rule in load()]
