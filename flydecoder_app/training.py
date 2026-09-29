"""Outcome-driven readout training over features processed by the fixed connectome."""
from __future__ import annotations

from dataclasses import dataclass
import threading
from pathlib import Path

from .features import extract_features
from .generator import examples
from .readout import ACTIONS, ActivityReadout
from .service import training_reward


@dataclass(frozen=True)
class TrainingProgress:
    completed: int
    total: int
    correct: int
    last_action: str
    last_reward: float
    recent_accuracy: float


def train_readout(adapter, model: ActivityReadout, count: int, seed: int, save_path: str | Path,
                  progress=None, cancel: threading.Event | None = None) -> TrainingProgress:
    if count <= 0:
        raise ValueError("training episode count must be positive")
    samples = examples(count, seed, partition="train")
    recent: list[int] = []
    correct = 0
    latest = TrainingProgress(0, count, 0, "—", 0.0, 0.0)
    for episode, sample in enumerate(samples, start=1):
        if cancel is not None and cancel.is_set():
            break
        feature_vector = extract_features(sample.input_text)
        # The adapter receives only normalized numbers; expected bytes and the generator's
        # private codec field never cross into the simulation.
        activity = adapter.activity(feature_vector.values)
        chosen = model.choose(activity, explore=0.32 if model.updates < 100 else 0.16)
        action = ACTIONS[chosen]
        # The evaluator checks each possible tool against the original bytes after the choice.
        # These are outcome rewards, not the generator's private encoding name.
        outcomes = [training_reward(sample.input_text, sample.expected_bytes, candidate) for candidate in ACTIONS]
        reward = outcomes[chosen]
        model.learn_from_rewards(activity, outcomes)
        hit = reward > 0
        correct += int(hit)
        recent.append(int(hit))
        if len(recent) > 32:
            recent.pop(0)
        latest = TrainingProgress(episode, count, correct, action, reward, sum(recent) / len(recent))
        if progress is not None:
            progress(latest)
        if episode % 10 == 0 or episode == count:
            model.save(save_path)
    model.save(save_path)
    return latest
