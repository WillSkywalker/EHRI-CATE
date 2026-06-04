from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


# Paper-aligned ISAD(G) input fields, in concatenation order:
#   3.1.2 Title, 3.2.2 Admin/Biographical History, 3.2.3 Archival History, 3.3.1 Scope and Content.
INPUT_FIELDS: tuple[str, ...] = (
    "name",
    "biographicalHistory",
    "archivalHistory",
    "scopeAndContent",
)

FIELD_SEPARATOR = "\n\n"


@dataclass(frozen=True)
class LabelledDoc:
    doc_id: str
    repo_id: str
    repo_name: str
    language_code: str | None
    text: str
    gold_uris: tuple[str, ...]


def _value_to_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return FIELD_SEPARATOR.join(_value_to_text(v) for v in value if _value_to_text(v))
    return str(value).strip()


def build_input_text(data: dict) -> str:
    parts: list[str] = []
    for field in INPUT_FIELDS:
        text = _value_to_text(data.get(field))
        if text:
            parts.append(text)
    return FIELD_SEPARATOR.join(parts)


def load_corpus(path: str | Path) -> list[LabelledDoc]:
    path = Path(path)
    with path.open() as f:
        raw = json.load(f)

    cols: list[str] = raw["columns"]
    rows: list[list] = raw["data"]
    idx = {name: i for i, name in enumerate(cols)}

    docs: list[LabelledDoc] = []
    for row in rows:
        data = row[idx["data"]]
        docs.append(
            LabelledDoc(
                doc_id=row[idx["id"]],
                repo_id=row[idx["repoId"]],
                repo_name=row[idx["repo"]],
                language_code=data.get("languageCode"),
                text=build_input_text(data),
                gold_uris=tuple(row[idx["labels"]]),
            )
        )
    return docs


def iter_corpus(path: str | Path) -> Iterator[LabelledDoc]:
    yield from load_corpus(path)
