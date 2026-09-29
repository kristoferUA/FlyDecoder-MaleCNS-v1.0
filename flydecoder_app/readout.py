"""A small trainable policy readout kept separate from the fixed connectome."""
from __future__ import annotations

from pathlib import Path

import numpy as np


ACTIONS = ("base64", "hex", "binary", "url", "skip")
MODEL_VERSION = 1


class ActivityReadout:
    """Linear action scores over measured FlyBrain population activity."""

    def __init__(self, input_count: int, seed: int = 0):
        if input_count <= 0:
            raise ValueError("input_count must be positive")
        self.weights = np.zeros((len(ACTIONS), input_count), dtype=np.float32)
        self.bias = np.zeros(len(ACTIONS), dtype=np.float32)
        self.updates = 0
        self.rng = np.random.default_rng(seed)

    @property
    def input_count(self) -> int:
        return int(self.weights.shape[1])

    def _activity(self, activity) -> np.ndarray:
        x = np.asarray(activity, dtype=np.float32)
        if x.shape != (self.input_count,) or not np.isfinite(x).all():
            raise ValueError(f"activity must be a finite vector of shape ({self.input_count},)")
        return np.clip(x, 0.0, 1.0)

    def scores(self, activity) -> np.ndarray:
        x = self._activity(activity)
        return self.weights @ x + self.bias

    def probabilities(self, activity) -> np.ndarray:
        scores = self.scores(activity)
        exp = np.exp(scores - np.max(scores))
        return exp / exp.sum()

    def choose(self, activity, explore: float = 0.0) -> int:
        if not 0.0 <= explore <= 1.0:
            raise ValueError("explore must be between 0 and 1")
        if explore and self.rng.random() < explore:
            return int(self.rng.integers(len(ACTIONS)))
        return int(np.argmax(self.scores(activity)))

    def reward(self, activity, chosen: int, reward: float, learning_rate: float = 0.12) -> None:
        """Reward an action from outcome only; no encoding label is accepted here."""
        x = self._activity(activity)
        if not 0 <= chosen < len(ACTIONS):
            raise ValueError("chosen action index is out of range")
        if reward not in (-1.0, 1.0):
            raise ValueError("reward must be -1 or +1")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        p = self.probabilities(x)
        gradient = -p
        gradient[chosen] += 1.0
        scale = learning_rate * reward
        self.weights += (scale * gradient[:, None] * x[None, :]).astype(np.float32)
        self.bias += (scale * gradient).astype(np.float32)
        self.updates += 1

    def learn_from_rewards(self, activity, rewards, learning_rate: float = 0.35) -> None:
        """Fit the policy to outcome rewards for candidate tools, never to encoding names."""
        x = self._activity(activity)
        outcome = np.asarray(rewards, dtype=np.float32)
        if outcome.shape != (len(ACTIONS),) or not np.isfinite(outcome).all():
            raise ValueError(f"rewards must be a finite vector of shape ({len(ACTIONS)},)")
        if np.any(~np.isin(outcome, (-1.0, 1.0))):
            raise ValueError("candidate rewards must be -1 or +1")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        successful = outcome > 0
        if not successful.any():
            raise ValueError("at least one candidate action must recover the training target")
        target = successful.astype(np.float32) / successful.sum()
        gradient = target - self.probabilities(x)
        self.weights += (learning_rate * gradient[:, None] * x[None, :]).astype(np.float32)
        self.bias += (learning_rate * gradient).astype(np.float32)
        self.updates += 1

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            version=np.int64(MODEL_VERSION),
            actions=np.asarray(ACTIONS, dtype="U16"),
            weights=self.weights,
            bias=self.bias,
            updates=np.int64(self.updates),
        )

    @classmethod
    def load(cls, path: str | Path, seed: int = 0) -> "ActivityReadout":
        with np.load(Path(path), allow_pickle=False) as saved:
            if int(saved["version"]) != MODEL_VERSION:
                raise ValueError("unsupported reader model version")
            actions = tuple(str(x) for x in saved["actions"].tolist())
            if actions != ACTIONS:
                raise ValueError("reader model action order does not match this app")
            weights = np.asarray(saved["weights"], dtype=np.float32)
            bias = np.asarray(saved["bias"], dtype=np.float32)
            if weights.ndim != 2 or weights.shape[0] != len(ACTIONS) or bias.shape != (len(ACTIONS),):
                raise ValueError("reader model has invalid dimensions")
            if not np.isfinite(weights).all() or not np.isfinite(bias).all():
                raise ValueError("reader model contains non-finite weights")
            model = cls(weights.shape[1], seed=seed)
            model.weights = weights.copy()
            model.bias = bias.copy()
            model.updates = int(saved["updates"])
            return model
