"""Strict byte decoders with useful errors for the UI."""
from __future__ import annotations

import base64
import binascii
import re
from urllib.parse import unquote_to_bytes


class DecodeError(ValueError):
    """The input does not match the requested textual encoding."""


def decode_base64(text: str) -> bytes:
    compact = "".join(text.split())
    if not compact:
        raise DecodeError("порожній рядок")
    core = compact.rstrip("=")
    padding = len(compact) - len(core)
    if padding > 2 or "=" in core or not re.fullmatch(r"[A-Za-z0-9+/_-]*", core):
        raise DecodeError("символи або padding не відповідають Base64")
    remainder = len(core) % 4
    if remainder == 1:
        raise DecodeError("довжина Base64 має зайвий символ")
    needed = (-len(core)) % 4
    if padding and (len(compact) % 4 != 0 or padding != needed):
        raise DecodeError("некоректний padding Base64")
    padded = core + ("=" * needed if not padding else "=" * padding)
    try:
        return base64.b64decode(padded.encode("ascii"), altchars=b"-_", validate=True)
    except (UnicodeEncodeError, binascii.Error, ValueError) as exc:
        raise DecodeError("рядок не є коректним Base64") from exc


def decode_hex(text: str) -> bytes:
    compact = re.sub(r"[\s,:;|_-]+", "", text)
    if not compact:
        raise DecodeError("порожній hex-рядок")
    if len(compact) % 2:
        raise DecodeError("у hex має бути парна кількість символів")
    if not re.fullmatch(r"[0-9A-Fa-f]+", compact):
        raise DecodeError("hex містить недопустимий символ")
    try:
        return bytes.fromhex(compact)
    except ValueError as exc:
        raise DecodeError("рядок не є коректним hex") from exc


def decode_binary(text: str) -> bytes:
    compact = re.sub(r"[\s,:;|_]+", "", text)
    if not compact:
        raise DecodeError("порожній binary-рядок")
    if not re.fullmatch(r"[01]+", compact):
        raise DecodeError("binary може містити лише 0 і 1 та роздільники")
    if len(compact) % 8:
        raise DecodeError("кількість бітів має ділитися на 8")
    return bytes(int(compact[i : i + 8], 2) for i in range(0, len(compact), 8))


def decode_url(text: str) -> bytes:
    if not text:
        raise DecodeError("порожній URL-рядок")
    valid = [match.group() for match in re.finditer(r"%[0-9A-Fa-f]{2}", text)]
    remainder = re.sub(r"%[0-9A-Fa-f]{2}", "", text)
    if "%" in remainder:
        raise DecodeError("кожен знак % має бути частиною послідовності %HH")
    if not valid and "+" not in text:
        raise DecodeError("не знайдено URL escape-послідовностей")
    return unquote_to_bytes(text.replace("+", " "))


DECODERS = {
    "base64": decode_base64,
    "hex": decode_hex,
    "binary": decode_binary,
    "url": decode_url,
}


def decode_bytes(text: str, action: str) -> bytes:
    try:
        decoder = DECODERS[action]
    except KeyError as exc:
        raise DecodeError(f"невідомий декодер: {action}") from exc
    return decoder(text)
