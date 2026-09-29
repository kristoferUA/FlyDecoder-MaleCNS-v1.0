"""Map numerical features into the real FlyBrain visual input and read neural activity."""
from __future__ import annotations

import os

import numpy as np


OPTIC_BANDS = 24
DOWNSTREAM_CHANNELS = ("visual_projection", "central", "mushroom_body", "descending", "vnc")
ACTIVITY_NAMES = tuple(
    name
    for band in range(OPTIC_BANDS)
    for name in (f"optic_band_{band:02}_mean", f"optic_band_{band:02}_active")
) + tuple(
    name
    for channel in DOWNSTREAM_CHANNELS
    for name in (f"{channel}_mean", f"{channel}_active")
)


def make_retina_frame(features, columns: int) -> np.ndarray:
    """Render the normalized feature vector as grayscale retinotopic bands."""
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("features must be a nonempty finite numeric vector")
    if np.any(values < 0) or np.any(values > 1):
        raise ValueError("features must be normalized to [0, 1]")
    if columns <= 0:
        raise ValueError("retina needs at least one column")
    frame = np.full((columns, 4), 0.04, dtype=np.float32)
    edges = np.linspace(0, columns, len(values) + 1, dtype=np.int64)
    for i, value in enumerate(values):
        start, stop = int(edges[i]), int(edges[i + 1])
        if stop > start:
            # FlyBrain's vision input order is UV, blue, green, red. Equal intensity keeps this a
            # brightness code; the feature's position selects its retinotopic band.
            frame[start:stop, :] = 0.04 + 1.5 * float(value)
    return frame


class FlyBrainAdapter:
    """A fresh visual sample is run through FlyBrain's retina, optic lobe and MaleCNS graph."""

    def __init__(self, device: str = "cpu", seed: int = 0, duration_ms: float = 30.0):
        if device.lower().startswith("cuda"):
            raise ValueError("FlyDecoder is configured for CPU; CUDA is not required")
        if duration_ms <= 0:
            raise ValueError("duration_ms must be positive")
        os.environ["FLYBRAIN_DEVICE"] = "cpu"
        from flybrain import connectome
        from flybrain.fly import FlyBrain
        from flybrain import regions

        self.c = connectome.load(verbose=True)
        self.simulation = FlyBrain(self.c, device="cpu", seed=seed)
        if self.simulation.retina is None or self.simulation.optic is None:
            raise RuntimeError("this MaleCNS graph did not provide the expected visual channels")
        self.duration_ms = float(duration_ms)
        labels = regions.labels(self.c)
        self.groups = {
            name: np.flatnonzero(labels == name)
            for name in DOWNSTREAM_CHANNELS
        }
        empty = [name for name, idx in self.groups.items() if len(idx) == 0]
        if empty:
            raise RuntimeError(f"connectome has no neurons for activity channels: {', '.join(empty)}")
        self.optic_bands = self._retinotopic_rate_groups()

    def _retinotopic_rate_groups(self) -> tuple[np.ndarray, ...]:
        """Map actual optic-lobe rate cells into 24 groups by their MaleCNS retinal columns."""
        retina = self.simulation.retina
        optic_idx = self.simulation.optic.rate_idx
        annotations = self.c.neurons.iloc[optic_idx]
        side = annotations.hex_side.fillna("").astype(str).to_numpy()
        h1 = np.asarray(annotations.hex1, dtype=np.float64)
        h2 = np.asarray(annotations.hex2, dtype=np.float64)
        column_of = {
            (str(retina.col_side[i]), int(retina.col_hex[i, 0]), int(retina.col_hex[i, 1])): i
            for i in range(retina.n_columns)
        }
        rate_column = np.full(len(optic_idx), -1, dtype=np.int32)
        for i, (cell_side, x, y) in enumerate(zip(side, h1, h2)):
            if cell_side and np.isfinite(x) and np.isfinite(y):
                rate_column[i] = column_of.get((cell_side, int(x), int(y)), -1)
        edges = np.linspace(0, retina.n_columns, OPTIC_BANDS + 1, dtype=np.int32)
        groups = tuple(np.flatnonzero((rate_column >= lo) & (rate_column < hi))
                       for lo, hi in zip(edges[:-1], edges[1:]))
        if any(len(group) == 0 for group in groups):
            raise RuntimeError("the MaleCNS annotations do not map optic-lobe activity across all retinal bands")
        return groups

    @staticmethod
    def _summary(values: np.ndarray) -> tuple[float, float]:
        if not values.size:
            return 0.0, 0.0
        mean_rate = float(np.mean(np.maximum(values, 0.0)))
        active_fraction = float(np.count_nonzero(values > 1.0) / values.size)
        return min(np.log1p(mean_rate) / np.log1p(100.0), 1.0), active_fraction

    def activity(self, features) -> np.ndarray:
        """Accept only numerical features; raw text and codec metadata stay in the wrapper."""
        self.simulation.reset()
        frame = make_retina_frame(features, self.simulation.retina.n_columns)
        background = np.full_like(frame, 0.04)
        self.simulation.vision(background)
        self.simulation.step(10.0)  # establish the optic lobe's operating point before the feature bars
        self.simulation.vision(frame)
        self.simulation.step(self.duration_ms)

        optic_delta = self.simulation.optic.delta_rate[0].detach().cpu().numpy()
        rates = self.simulation.brain.rate_np()
        values: list[float] = []
        for idx in self.optic_bands:
            signal = np.abs(optic_delta[idx])
            values.extend((float(np.clip(signal.mean() / 0.2, 0.0, 1.0)),
                           float(np.count_nonzero(signal > 0.03) / signal.size)))
        for name in DOWNSTREAM_CHANNELS:
            mean, active = self._summary(rates[self.groups[name]])
            values.extend((mean, active))
        return np.asarray(values, dtype=np.float32)
