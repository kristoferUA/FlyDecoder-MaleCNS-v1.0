"""Quality checks for decoded bytes when the original payload is unknown."""
from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class DecodeQuality:
    score: float
    plausible: bool
    utf8_valid: bool
    printable_fraction: float
    text: str | None
    reason: str


def assess_bytes(data: bytes, expected_pattern: str | None = None) -> DecodeQuality:
    if not data:
        return DecodeQuality(0.0, False, True, 0.0, "", "результат порожній")
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return DecodeQuality(0.0, False, False, 0.0, None, "результат не є валідним UTF-8")
    if not text:
        return DecodeQuality(0.0, False, True, 0.0, text, "результат порожній")
    printable = sum(ch.isprintable() or ch in "\r\n\t" for ch in text) / len(text)
    matched = bool(expected_pattern and re.search(expected_pattern, text))
    score = min(1.0, 0.35 + 0.6 * printable + (0.2 if matched else 0.0))
    plausible = matched or printable >= 0.82
    reason = "формат очікуваного прапора збігається" if matched else (
        f"UTF-8, друковані символи {printable:.0%}" if plausible else
        f"замало друкованих символів ({printable:.0%})"
    )
    return DecodeQuality(score, plausible, True, printable, text, reason)
