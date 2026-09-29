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
        raise DecodeError("empty string")
    core = compact.rstrip("=")
    padding = len(compact) - len(core)
    if padding > 2 or "=" in core or not re.fullmatch(r"[A-Za-z0-9+/_-]*", core):
        raise DecodeError("characters or padding do not match Base64")
    remainder = len(core) % 4
    if remainder == 1:
        raise DecodeError("Base64 has an invalid length remainder")
    needed = (-len(core)) % 4
    if padding and (len(compact) % 4 != 0 or padding != needed):
        raise DecodeError("invalid Base64 padding")
    padded = core + ("=" * needed if not padding else "=" * padding)
    try:
        return base64.b64decode(padded.encode("ascii"), altchars=b"-_", validate=True)
    except (UnicodeEncodeError, binascii.Error, ValueError) as exc:
        raise DecodeError("string is not valid Base64") from exc


def decode_hex(text: str) -> bytes:
    compact = re.sub(r"[\s,:;|_-]+", "", text)
    if not compact:
        raise DecodeError("empty hex string")
    if len(compact) % 2:
        raise DecodeError("hex must contain an even number of characters")
    if not re.fullmatch(r"[0-9A-Fa-f]+", compact):
        raise DecodeError("hex contains an invalid character")
    try:
        return bytes.fromhex(compact)
    except ValueError as exc:
        raise DecodeError("string is not valid hex") from exc


def decode_binary(text: str) -> bytes:
    compact = re.sub(r"[\s,:;|_]+", "", text)
    if not compact:
        raise DecodeError("empty binary string")
    if not re.fullmatch(r"[01]+", compact):
        raise DecodeError("binary can contain only 0, 1, and separators")
    if len(compact) % 8:
        raise DecodeError("bit count must be divisible by 8")
    return bytes(int(compact[i : i + 8], 2) for i in range(0, len(compact), 8))


def decode_url(text: str) -> bytes:
    if not text:
        raise DecodeError("empty URL string")
    valid = [match.group() for match in re.finditer(r"%[0-9A-Fa-f]{2}", text)]
    remainder = re.sub(r"%[0-9A-Fa-f]{2}", "", text)
    if "%" in remainder:
        raise DecodeError("each % must be part of a %HH sequence")
    if not valid and "+" not in text:
        raise DecodeError("no URL escape sequences found")
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
        raise DecodeError(f"unknown decoder: {action}") from exc
    return decoder(text)
