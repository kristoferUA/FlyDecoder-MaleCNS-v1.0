"""The dino game: cacti come at the fly along its heading and it has to jump them.

The cactus is the looming ball (a black sphere the fly's eyes see) driven along the table at ground level. Without a
pilot the fly's own loom -> giant-fibre escape does the jumping and the game just keeps score. With a pilot (a
`Readout`) the reflex is switched off and a trained readout of brain activity decides when to jump; in "train" mode
the cacti come at random speeds, a hit does not end the round, and the readout learns from every one.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
import os
import re

import numpy as np


@dataclass
class DinoGame:
    radius: float = 0.03         # cactus radius (m): the loom detectors need a 3 cm ball; the giant fibre fires ~1.2 radii out
    speed0: float = 0.45         # m/s for the first cactus (l/v ~ 65 ms; the reflex hits 65-80 Hz on the giant fibre)
    speed_step: float = 0.03     # faster by this much per point
    speed_max: float = 1.5
    start_d: float = 0.35        # spawn this far ahead of the fly (m); keeps the cactus on the table
    gap: tuple = (1.6, 2.8)      # s between cacti (the escape has a 1 s landing refractory)
    lap_x: float = 0.2           # every hop carries the fly ~12 cm forward: past this x it goes back to the start
    lane_y: float = 0.05
    height: float = 0.006        # cactus centre above the table (m): near eye level, so the fly sees it expand head-on
    fly_r: float = 0.003         # the fly's body radius for the hit test
    seed: int = 0
    score: int = 0
    best: int = 0
    round: int = 1
    state: str = "wait"          # wait | cactus | over | lap
    timer: float = 3.0
    d: float = 0.0               # cactus distance ahead of its spawn origin along the track (m)
    origin: np.ndarray = field(default_factory=lambda: np.zeros(2))
    track: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    mode: str = "play"           # play | train (random speeds, a hit does not end the round, the pilot learns)
    train_speed: tuple = (0.45, 1.2)   # m/s range of the practice cacti
    pilot: object = None         # a Readout: decides the jumps instead of the giant-fibre reflex
    groups: list = None          # cell indices of each pilot input
    pre: deque = field(default_factory=lambda: deque(maxlen=30))   # activity just before a cactus appears
    trace: list = field(default_factory=list)
    jumped: bool = False
    cacti: int = 0
    falses: int = 0              # jumps at nothing

    @property
    def speed(self) -> float:
        return self._speed if self.mode == "train" else min(self.speed_max, self.speed0 + self.speed_step * self.score)

    # ------------------------------------------------------------------ lifecycle
    def start(self, sim) -> None:
        sim.reset_fly()
        sim.fly.y = self.lane_y; sim.fly.heading = 0.0
        self.score, self.round, self.state, self.timer = 0, 1, "wait", 3.0
        sim.loom_t = -1.0
        self._park(sim)
        self._speed = self.train_speed[0]
        if self.pilot is not None:
            import torch
            if sim.flight.gf_hz < 1e9:
                self._gf_hz, sim.flight.gf_hz = sim.flight.gf_hz, 1e9            # the pilot jumps, not the reflex (finite: it goes into JSON)
            self._idx = torch.as_tensor(np.concatenate(self.groups), device=sim.brain.device)
            self._cuts = np.cumsum([0] + [len(g) for g in self.groups])[:-1]
            self._n = np.array([len(g) for g in self.groups], np.float32)

    def stop(self, sim) -> None:
        self._park(sim)
        if self.pilot is not None:
            sim.flight.gf_hz = self._gf_hz

    def features(self, sim) -> np.ndarray:
        r = sim.brain.rate[0, self._idx].cpu().numpy()
        return (np.add.reduceat(r, self._cuts) / self._n / 20.0).astype(np.float32)

    def _pilot(self, sim, fly, out) -> None:
        """One frame of the pilot: remember the activity and jump when the readout says so."""
        x, p = self.features(sim), self.pilot
        ready = not fly.airborne and fly.ground_time >= sim.flight.landing_refractory_s
        if self.state == "cactus" and not self.jumped:
            self.trace.append(x)
        elif ready:
            self.pre.append(x)
        if fly.airborne:
            self._air = fly.air_time
        elif getattr(self, "_air", 0) > 0:                                       # just landed: how long was the hop
            p.airtime, self._air = 0.8 * p.airtime + 0.2 * self._air, 0.0
        if ready and p.fires(x):
            sim.flight.launch(fly, escape=True)
            if self.state == "cactus":
                self.jumped = True
            else:                                                                # a jump at nothing
                if self.mode == "train":
                    p.remember(list(self.pre), -1, True); p.learn()
                self.pre.clear(); self.falses += 1
                out.append(json.dumps({"dino": "false", "score": self.score}))

    def _result(self, cleared: bool) -> None:
        p = self.pilot
        self.cacti += 1
        p.history.append(int(cleared))
        if self.mode == "train":
            p.remember(self.trace, self._arrive, self.jumped); p.learn()

    def _park(self, sim) -> None:
        sim.world.move_sphere(sim.loom_idx, (9, 9, 9), (self.radius,) * 3)

    def _spawn(self, sim) -> None:
        f = sim.fly.forward[:2]
        self.track = f / max(float(np.linalg.norm(f)), 1e-6)
        self.origin = sim.fly.pos[:2].copy()
        self.d = self.start_d
        self.state = "cactus"
        if self.mode == "train":
            self._speed = float(self.rng.uniform(*self.train_speed))
        self.trace, self.jumped = list(self.pre), False
        self._arrive = len(self.trace) + int(round(self.start_d / self.speed / 0.01))

    def cactus_xy(self) -> np.ndarray:
        return self.origin + self.track * self.d

    # ------------------------------------------------------------------ one frame
    def step(self, sim, dt: float) -> list[str]:
        """Advance the game by `dt` seconds after the sim stepped. Returns viewer events (plain or JSON strings)."""
        fly, top, out = sim.fly, sim.info["table_top_z"], []
        if self.pilot is not None:
            self._pilot(sim, fly, out)
        self.fly_h, self.along = (max(0.0, float(fly.pos[2]) - top) if fly.airborne else 0.0), None   # for the 2D game screen
        if self.state == "cactus":
            self.d -= self.speed * dt
            c = self.cactus_xy()
            sim.world.move_sphere(sim.loom_idx, (c[0], c[1], top + self.height))
            along = self.along = float((c - fly.pos[:2]) @ self.track)
            if along <= 0:                       # the cactus centre reaches the fly: off the ground = cleared, like the real game
                self._park(sim)
                if self.pilot is not None:
                    self._result(fly.airborne)
                if fly.airborne:
                    self.score += 1
                    self.state, self.timer = "wait", float(self.rng.uniform(*self.gap))
                    out.append(json.dumps({"dino": "clear", "score": self.score}))
                elif self.mode == "train":                  # practice: no game over, the next cactus is coming
                    self.state, self.timer = "wait", float(self.rng.uniform(*self.gap))
                    out.append(json.dumps({"dino": "hit", "score": self.score}))
                else:
                    self.state, self.timer = "over", 2.5
                    self.best = max(self.best, self.score)
                    out += [json.dumps({"dino": "hit", "score": self.score}), f"Ouch. Game over at {self.score}"]
        elif self.state in ("wait", "lap"):
            self.timer -= dt
            if fly.x > self.lap_x and self.state == "wait" and not fly.airborne:
                sim.reset_fly(); fly = sim.fly
                fly.y = self.lane_y; fly.heading = 0.0
                self.state, self.timer = "lap", 3.0
                out.append("Back to the start of the table")
            elif self.timer <= 0 and not fly.airborne and fly.ground_time >= 1.0:
                self._spawn(sim)
        elif self.state == "over":
            self.timer -= dt
            if self.timer <= 0:
                self.round += 1
                self.score = 0
                sim.reset_fly(); sim.fly.y = self.lane_y; sim.fly.heading = 0.0
                self.state, self.timer = "wait", 3.0
                out.append(json.dumps({"dino": "round", "round": self.round}))
        return out

    def status(self) -> dict:
        out = {"on": True, "score": self.score, "best": self.best, "round": self.round, "state": self.state,
               "speed": self.speed, "r": self.radius, "h": self.height, "mode": self.mode, "pilot": None, "falses": self.falses,
               "fly_h": round(getattr(self, "fly_h", 0.0), 4), "along": getattr(self, "along", None)}
        if self.pilot is not None:
            p = self.pilot
            out["pilot"] = {"inputs": p.inputs, "n": len(p.names), "cacti": len(p.history), "history": p.history[-60:],
                            "top": p.top(), "reward": p.reward, "punish": p.punish, "airtime": round(p.airtime, 2)}
        return out


# ---------------------------------------------------------------------- the training ground
# Named sets of cell types a readout can listen to. Anything else is read as a comma-separated list of types.
INPUTS = {
    "gf": ["DNp01"],                                                          # the giant fibre alone: learn a threshold
    "loom": ["LC4", "LPLC2", "LPLC1", "LC6", "LC16", "LC12", "LC17", "LC11",   # loom-sensitive visual projection neurons
             "DNp01", "DNp02", "DNp03", "DNp04", "DNp06", "DNp11"],           # and the escape descending neurons
    "visual": ["~^(?:LC|LPLC|LPC)\\d"],                                         # every lobula columnar type
}
FLIES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out", "flies")


def feature_groups(c, inputs: str) -> tuple[list[np.ndarray], list[str]]:
    """(cell indices, name) per cell type and side for an INPUTS key or a comma-separated list of types ('~' = regex)."""
    types = INPUTS.get(inputs) or [t.strip() for t in inputs.split(",") if t.strip()]
    found = c.neurons.type.dropna()
    names = sorted({t for q in types for t in (found[found.str.contains(q[1:], regex=True)] if q.startswith("~") else [q])})
    groups = [(c.select(type=t, somaSide=s), f"{t} {s}") for t in names for s in "LR"]
    groups = [(i, n) for i, n in groups if len(i)]
    if not groups:
        raise ValueError(f"no neurons match inputs {inputs!r}")
    return [i for i, _ in groups], [n for _, n in groups]


class Readout:
    """A linear readout of brain activity that decides when to jump. The brain is never changed: only these weights
    learn. Jump when w . x + b > 0, x = mean rate of each input population / 20 Hz.

    Learning is reward and punishment with replay: every cactus is stored (the activity trace up to the jump or the
    hit), and after each one a cross-entropy search keeps the weights that would have earned the most reward over
    everything remembered: +reward for being in the air when a cactus arrives, -punish for a hit or a jump at nothing.
    """

    def __init__(self, names, inputs="loom", reward=1.0, punish=1.0, pop=64, iters=6, sigma=0.5, memory=150, seed=0):
        self.names, self.inputs = list(names), inputs
        self.reward, self.punish, self.pop, self.iters, self.memory = float(reward), float(punish), int(pop), int(iters), int(memory)
        d = len(self.names) + 1
        self.w = np.zeros(d, np.float32); self.w[-1] = -1.0          # bias last: an untrained fly never jumps
        self.std = np.full(d, float(sigma), np.float32)
        self.trials: list[tuple[np.ndarray, int, bool]] = []         # (X, arrive index or -1 for "no cactus", trace cut at a real jump)
        self.history: list[int] = []                                 # 1 clear, 0 hit, per cactus
        self.airtime = 0.3                                           # s a hop lasts; measured from the fly's own jumps
        self.rng = np.random.default_rng(seed)

    def fires(self, x: np.ndarray) -> bool:
        return float(self.w[:-1] @ x + self.w[-1]) > 0

    def remember(self, X, arrive: int, cut: bool) -> None:
        self.trials.append((np.c_[np.asarray(X, np.float32), np.ones(len(X), np.float32)], int(arrive), bool(cut)))
        del self.trials[:-self.memory]

    def fitness(self, cand: np.ndarray, dt: float = 0.01) -> np.ndarray:
        """Total reward each candidate weight vector (pop, d) would have earned over the remembered cacti."""
        air, fit = int(0.85 * self.airtime / dt), np.zeros(len(cand))
        for X, arrive, cut in self.trials:
            cross = (X @ cand.T) > 0
            any_, j = cross.any(0), cross.argmax(0)
            if arrive < 0:                                           # nothing was coming: any jump is a mistake
                fit -= self.punish * any_
                continue
            lo = max(arrive - air, 0)
            ok = any_ & (j >= lo)
            early = any_ & (j < lo)
            fit += self.reward * ok - early * self.punish * (1 + 0.2 * (lo - j) / max(lo, 1))
            fit -= ~any_ * self.punish * (0.5 if cut else 1.2)       # cut: the real fly jumped first, so later is unknown
        return fit

    def learn(self) -> None:
        if not self.trials:
            return
        k = max(2, self.pop // 5)
        for _ in range(self.iters):
            cand = self.w + self.std * self.rng.standard_normal((self.pop, len(self.w))).astype(np.float32)
            cand[0] = self.w                                         # the current weights compete too: never get worse on memory
            fit = self.fitness(cand)
            elite = cand[np.argsort(fit, kind="stable")[-k:]]
            self.w = elite[-1].copy() if fit.max() > fit[0] else self.w
            self.std = np.maximum(0.5 * self.std + 0.5 * elite.std(0), 0.03).astype(np.float32)

    def top(self, n=5) -> list:
        order = np.argsort(-np.abs(self.w[:-1]))[:n]
        return [[self.names[i], round(float(self.w[i]), 2)] for i in order]

    def save(self, name: str) -> str:
        name = re.sub(r"[^\w-]", "", name) or "fly"
        os.makedirs(FLIES, exist_ok=True)
        np.savez(os.path.join(FLIES, name + ".npz"), w=self.w, names=np.array(self.names), inputs=self.inputs,
                 airtime=self.airtime, history=np.array(self.history, np.int8))
        return name

    @classmethod
    def load(cls, name: str) -> "Readout":
        d = np.load(os.path.join(FLIES, re.sub(r"[^\w-]", "", name) + ".npz"))
        r = cls([str(n) for n in d["names"]], str(d["inputs"]))
        r.w, r.airtime, r.history = d["w"].astype(np.float32), float(d["airtime"]), d["history"].tolist()
        return r


def saved_flies() -> list[str]:
    return sorted(f[:-4] for f in os.listdir(FLIES) if f.endswith(".npz")) if os.path.isdir(FLIES) else []
