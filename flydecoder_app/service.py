"""Coordinate feature extraction, the neural readout and safe decoder retries."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .codecs import DecodeError, decode_bytes
from .quality import DecodeQuality, assess_bytes
from .readout import ACTIONS, ActivityReadout
from .features import FeatureVector, extract_features


@dataclass(frozen=True)
class Attempt:
    action: str
    data: bytes | None
    quality: DecodeQuality | None
    error: str | None


@dataclass(frozen=True)
class InferenceResult:
    features: FeatureVector
    activity: np.ndarray
    scores: np.ndarray
    probabilities: np.ndarray
    first_action: str
    attempts: tuple[Attempt, ...]
    output: bytes | None
    skipped: bool
    message: str


def decode_with_retry(
    text: str,
    activity,
    readout: ActivityReadout,
    expected_pattern: str | None = None,
) -> InferenceResult:
    """Choose by neural activity, then try another tool when decoding looks implausible."""
    features = extract_features(text)
    scores = readout.scores(activity)
    probabilities = readout.probabilities(activity)
    first_idx = int(np.argmax(scores))
    first_action = ACTIONS[first_idx]
    codec_indices = sorted(range(4), key=lambda idx: float(scores[idx]), reverse=True)
    if first_action == "skip" and scores[4] - max(scores[:4]) >= 1.0:
        return InferenceResult(features, np.asarray(activity, dtype=np.float32), scores, probabilities,
                               first_action, (), None, True, "муха вирішила пропустити цей рядок")

    attempts: list[Attempt] = []
    for idx in codec_indices:
        action = ACTIONS[idx]
        try:
            data = decode_bytes(text, action)
        except DecodeError as exc:
            attempts.append(Attempt(action, None, None, str(exc)))
            continue
        quality = assess_bytes(data, expected_pattern=expected_pattern)
        attempts.append(Attempt(action, data, quality, None))
        if quality.plausible:
            message = "декодовано" if len(attempts) == 1 else f"успіх після повторного вибору: {action}"
            return InferenceResult(features, np.asarray(activity, dtype=np.float32), scores, probabilities,
                                   first_action, tuple(attempts), data, False, message)
    return InferenceResult(features, np.asarray(activity, dtype=np.float32), scores, probabilities,
                           first_action, tuple(attempts), None, False,
                           "усі варіанти відхилені; муха пропонує пропустити рядок")


def training_reward(text: str, expected: bytes | None, action: str) -> float:
    """Score exact byte recovery after a choice; the chosen encoding is not an argument."""
    if action == "skip":
        return 1.0 if expected is None else -1.0
    try:
        decoded = decode_bytes(text, action)
    except DecodeError:
        return -1.0
    if expected is None:
        return -1.0
    return 1.0 if decoded == expected else -1.0


def _exact_first_action(text: str, expected: bytes | None, action: str) -> bool:
    if action == "skip":
        return expected is None
    if expected is None:
        return False
    try:
        return decode_bytes(text, action) == expected
    except DecodeError:
        return False


def run_blind_evaluation(adapter, readout: ActivityReadout, count: int = 64, seed: int = 8041,
                         expected_pattern: str | None = None) -> dict:
    """Evaluate fresh generated samples; generator labels stay out of the simulation input."""
    from .generator import examples

    samples = examples(count, seed)
    matrix = {codec: {action: 0 for action in ACTIONS} for codec in ACTIONS}
    top_exact = retry_exact = 0
    failures: list[dict[str, str]] = []
    for sample in samples:
        vector = extract_features(sample.input_text)
        activity = adapter.activity(vector.values)
        scores = readout.scores(activity)
        first = ACTIONS[int(np.argmax(scores))]
        matrix[sample.generated_codec][first] += 1
        top_exact += int(_exact_first_action(sample.input_text, sample.expected_bytes, first))
        result = decode_with_retry(sample.input_text, activity, readout, expected_pattern)
        recovered = result.output == sample.expected_bytes if sample.expected_bytes is not None else result.skipped or result.output is None
        retry_exact += int(recovered)
        if not recovered and len(failures) < 8:
            failures.append({"input": sample.input_text[:100], "expected": sample.generated_codec,
                             "first": first, "result": result.message})
    denom = max(count, 1)
    return {
        "count": count,
        "top1_accuracy": top_exact / denom,
        "retry_accuracy": retry_exact / denom,
        "confusion": matrix,
        "failures": failures,
    }
