"""Synthetic training examples. Encoding names are evaluator-only metadata."""
from __future__ import annotations

from dataclasses import dataclass
import base64
import random
import re
from urllib.parse import quote_from_bytes, quote_plus


TRAIN_MESSAGES = (
    "HELLO FLY",
    "flag{tiny_wings_42}",
    "bring coffee at 09:30",
    "The connectome stays fixed.",
    "two eyes, six legs, one decoder",
    "Привіт, мушко!",
    "café, tea, and code",
    "ZX-17 / room 4",
    "A1b2C3d4 mix Case",
    "https://example.test/fly?q=1",
    "01010101 is only a clue",
    "Mosaic-Delta_8",
    "read the bytes, then decide",
    "µm × 8nm",
)
BLIND_MESSAGES = (
    "blue teacup by the screen",
    "F1y{decoder-check_91}",
    "Нехай граф мовчить до стимулу.",
    "phase 7 / neuron 12",
    "input -> retina -> connectome",
    "6-bits meet 8-bit groups",
    "owl%20fruit%2Froom",
    "low-resolution pixel art",
    "some-padding==",
    "hex: bEa7 1c09",
    "50% of bytes?",
    "space between each byte",
    "after training, test again",
    "№24 · CPU only",
)


@dataclass(frozen=True)
class TrainingExample:
    input_text: str
    expected_bytes: bytes | None
    generated_codec: str


def _insert_spaces(text: str, width: int) -> str:
    return " ".join(text[i : i + width] for i in range(0, len(text), width))


def encode_example(message: str, codec: str, rng: random.Random) -> str:
    payload = message.encode("utf-8")
    if codec == "base64":
        if rng.random() < 0.4:
            encoded = base64.urlsafe_b64encode(payload).decode("ascii")
        else:
            encoded = base64.b64encode(payload).decode("ascii")
        if rng.random() < 0.45:
            encoded = encoded.rstrip("=")
        if rng.random() < 0.2:
            encoded = _insert_spaces(encoded, 4)
        return encoded
    if codec == "hex":
        encoded = payload.hex()
        if rng.random() < 0.5:
            encoded = "".join(ch.upper() if rng.random() < 0.5 else ch for ch in encoded)
        if rng.random() < 0.35:
            return ":".join(encoded[i : i + 2] for i in range(0, len(encoded), 2))
        if rng.random() < 0.25:
            return _insert_spaces(encoded, 2)
        return encoded
    if codec == "binary":
        encoded = "".join(f"{byte:08b}" for byte in payload)
        return _insert_spaces(encoded, 8) if rng.random() < 0.6 else encoded
    if codec == "url":
        encoded = quote_from_bytes(payload, safe="")
        if rng.random() < 0.25:
            encoded = quote_plus(message, safe="")
        if "%" not in encoded and "+" not in encoded:
            # Percent-encode one unreserved byte so the example represents an actual URL escape.
            encoded = f"%{payload[0]:02X}" + encoded[1:]
        if rng.random() < 0.5:
            encoded = re.sub(r"%[0-9A-F]{2}", lambda match: match.group().lower(), encoded)
        return encoded
    raise ValueError(f"unknown training codec: {codec}")


def make_example(rng: random.Random, partition: str = "train", forced_codec: str | None = None) -> TrainingExample:
    if partition not in ("train", "blind"):
        raise ValueError("partition must be 'train' or 'blind'")
    if forced_codec is not None and forced_codec not in ("base64", "hex", "binary", "url", "skip"):
        raise ValueError(f"unknown training codec: {forced_codec}")
    pool = TRAIN_MESSAGES if partition == "train" else BLIND_MESSAGES
    message = rng.choice(pool)
    if forced_codec == "skip" or (forced_codec is None and rng.random() < 0.18):
        # Plain text is a negative example. It has no escape marker or valid byte alphabet.
        return TrainingExample(message, None, "skip")
    codec = forced_codec or rng.choice(("base64", "hex", "binary", "url"))
    return TrainingExample(encode_example(message, codec, rng), message.encode("utf-8"), codec)


def examples(count: int, seed: int, partition: str = "blind") -> list[TrainingExample]:
    if count < 0:
        raise ValueError("count must be nonnegative")
    rng = random.Random(seed)
    schedule = [("base64", "hex", "binary", "url", "skip")[i % 5] for i in range(count)]
    rng.shuffle(schedule)
    return [make_example(rng, partition=partition, forced_codec=codec) for codec in schedule]
