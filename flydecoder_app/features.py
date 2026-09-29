"""Numerical string features. No decoder identity or training label enters this module."""
from __future__ import annotations

from dataclasses import dataclass
import math
import re

import numpy as np


FEATURE_NAMES = (
    "log_length",
    "length_mod_4",
    "length_mod_8",
    "equals_fraction",
    "trailing_padding",
    "base64_alphabet_fraction",
    "base64url_fraction",
    "hex_fraction",
    "binary_fraction",
    "url_unreserved_fraction",
    "whitespace_fraction",
    "separator_fraction",
    "percent_escape_count",
    "valid_percent_fraction",
    "binary_only",
    "hex_only_even",
    "hex_mixed_case",
    "base64_length_aligned",
    "base64_padding_valid",
    "base64_quartet_score",
    "distinct_fraction",
    "printable_fraction",
    "contains_plus",
    "contains_url_reserved",
)

_B64 = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=")
_B64URL = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_= ")
_HEX = frozenset("0123456789abcdefABCDEF")
_UNRESERVED = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")
_SEPARATORS = frozenset(" \t\r\n,:;|_-")
_PERCENT = re.compile(r"%[0-9A-Fa-f]{2}")


@dataclass(frozen=True)
class FeatureVector:
    names: tuple[str, ...]
    values: np.ndarray
    summary: dict[str, float | int | bool]

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float32)
        if values.shape != (len(self.names),):
            raise ValueError("feature vector has the wrong shape")
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
            raise ValueError("feature values must be finite and normalized to [0, 1]")
        object.__setattr__(self, "values", values)


def extract_features(text: str) -> FeatureVector:
    """Describe syntax and character statistics without naming a likely decoder."""
    if not isinstance(text, str):
        raise TypeError("input must be text")
    n = len(text)
    denom = max(n, 1)
    compact = "".join(text.split())
    core = compact.rstrip("=")
    padding = len(compact) - len(core)
    chars = set(text)
    percent_matches = list(_PERCENT.finditer(text))
    valid_percent_chars = sum(len(m.group()) for m in percent_matches)
    b64_core = all(ch in _B64 - {"="} or ch in "-_" for ch in core)
    b64_pad_shape = padding <= 2 and (not padding or len(compact) % 4 == 0)
    b64_remainder = len(core) % 4
    b64_pad_matches = (padding == 0 and b64_remainder in (0, 2, 3)) or (
        padding == 1 and b64_remainder == 3
    ) or (padding == 2 and b64_remainder == 2)
    b64_padding_valid = b64_core and b64_pad_shape and b64_pad_matches
    hex_chars = [ch for ch in text if ch in _HEX]
    has_lower = any(ch in "abcdef" for ch in hex_chars)
    has_upper = any(ch in "ABCDEF" for ch in hex_chars)
    binary_only = bool(compact) and all(ch in "01" for ch in compact)
    cleaned_hex = "".join(ch for ch in text if ch not in _SEPARATORS)
    hex_only_even = bool(cleaned_hex) and len(cleaned_hex) % 2 == 0 and all(ch in _HEX for ch in cleaned_hex)

    def fraction(predicate) -> float:
        return sum(bool(predicate(ch)) for ch in text) / denom

    values = np.asarray(
        [
            min(math.log1p(n) / math.log1p(4096), 1.0),
            (n % 4) / 3.0,
            (n % 8) / 7.0,
            text.count("=") / denom,
            min(padding, 2) / 2.0,
            fraction(lambda ch: ch in _B64),
            fraction(lambda ch: ch in _B64 or ch in "-_"),
            fraction(lambda ch: ch in _HEX),
            fraction(lambda ch: ch in "01"),
            fraction(lambda ch: ch in _UNRESERVED),
            fraction(str.isspace),
            fraction(lambda ch: ch in _SEPARATORS),
            min(len(percent_matches), 8) / 8.0,
            valid_percent_chars / (3 * text.count("%")) if "%" in text else 0.0,
            float(binary_only),
            float(hex_only_even),
            float(has_lower and has_upper),
            float(bool(compact) and len(compact) % 4 == 0),
            float(b64_padding_valid),
            1.0 if b64_remainder == 0 else (0.65 if b64_remainder in (2, 3) else 0.0),
            len(chars) / denom,
            fraction(lambda ch: ch.isprintable() or ch in "\r\n\t"),
            float("+" in text),
            fraction(lambda ch: ch in "%/?&=#"),
        ],
        dtype=np.float32,
    )
    summary = {
        "length": n,
        "mod4": n % 4,
        "mod8": n % 8,
        "trailing_padding": padding,
        "percent_escapes": len(percent_matches),
        "binary_only": binary_only,
        "hex_only_even": hex_only_even,
        "b64_padding_valid": b64_padding_valid,
        "whitespace": any(ch.isspace() for ch in text),
    }
    return FeatureVector(FEATURE_NAMES, values, summary)
