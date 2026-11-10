"""Sentence ids for patient notes.

Dataset notes are already numbered ("0. ... 1. ..."). A note typed by a user
without that prefix is split on sentence boundaries and numbered from zero.
The numbered text is what the model sees, so a cited id always points at one
visible sentence.
"""

from __future__ import annotations

import re

_NUMBERED = re.compile(r"(?:^|(?<=\s))(\d+)\.\s+")
_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")


def parse_numbered_note(note: str) -> dict[int, str] | None:
    """Return sentence id to text when the note starts with '0. '.

    Returns None when the note is not in the dataset numbering style.
    """
    text = note.strip()
    matches = list(_NUMBERED.finditer(text))
    if not matches or matches[0].start() != 0 or matches[0].group(1) != "0":
        return None
    sentences: dict[int, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sentence_id = int(match.group(1))
        if sentence_id in sentences:
            raise ValueError(f"Duplicate sentence id {sentence_id}.")
        body = text[start:end].strip()
        sentences[sentence_id] = body
    return sentences


def split_unnumbered(note: str) -> list[str]:
    text = note.strip()
    if not text:
        return []
    parts = [part.strip() for part in _BOUNDARY.split(text) if part.strip()]
    return parts or [text]


def format_numbered(sentences: dict[int, str]) -> str:
    return "\n".join(f"{sentence_id}. {sentences[sentence_id]}" for sentence_id in sorted(sentences))


def ensure_numbered(note: str) -> tuple[str, dict[int, str]]:
    """Return a canonical numbered note and the id map.

    An empty note returns an empty string and an empty map. The caller decides
    whether that is an error.
    """
    if not note or not note.strip():
        return "", {}
    parsed = parse_numbered_note(note)
    if parsed is not None:
        return format_numbered(parsed), parsed
    parts = split_unnumbered(note)
    sentences = {index: part for index, part in enumerate(parts)}
    return format_numbered(sentences), sentences
