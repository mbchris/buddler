"""Versioned prompt files (AGENTS.md §6): `prompts/<workflow>/<name>.md`, version = content hash."""

import hashlib
from dataclasses import dataclass
from functools import cache
from importlib import resources

from buddler.models.enums import RunKind

VERSION_HASH_CHARS = 8


@dataclass(frozen=True)
class Prompt:
    text: str
    # Stored with every result, e.g. "resolve/decide@1a2b3c4d".
    version: str


@cache
def load_prompt(workflow: RunKind, name: str) -> Prompt:
    text = (resources.files("buddler.prompts") / workflow.value / f"{name}.md").read_text(
        encoding="utf-8"
    )
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:VERSION_HASH_CHARS]
    return Prompt(text=text.strip(), version=f"{workflow.value}/{name}@{digest}")
