"""Two-node fusion: range from geometry instead of loudness.

Node A is the cueing node (co-located with the turret / camera). Node B is a
second passive array a known baseline away. Each frame, A's raw bearing and B's
bearing (its filtered track when confirmed, else its raw fix) are intersected.
The closest-approach point of the two rays gives range from A, and the bearing
sigmas give a range sigma via the intersection geometry. That fix is fed to A's
tracker, which fuses it with its own level-based estimate.
"""

import time
from dataclasses import dataclass

import numpy as np

from .geometry import SPEED_OF_SOUND, unit_vector
from .pipeline import FrameResult, Pipeline
from .tracker import TrackState


@dataclass
class RangeFix:
    range_m: float     # from node A along A's ray
    sigma_m: float
    miss_m: float      # closest-approach distance between the rays
    crossing_deg: float


def triangulate(pa, ua, sigma_a_deg, pb, ub, sigma_b_deg, max_speed=45.0, max_range=1500.0) -> RangeFix | None:
    """Intersect two bearing rays (origins p, unit directions u) in 3-D.

    Returns None when the rays are near-parallel, point away from each other, or
    miss by more than their combined angular uncertainty allows (wrong association
    or an outlier bearing)."""
    w0 = pa - pb
    b = ua @ ub
    d, e = ua @ w0, ub @ w0
    denom = 1.0 - b * b
    sin_g = np.sqrt(max(denom, 0.0))
    if sin_g < np.sin(np.radians(0.5)):
        return None
    s = (b * e - d) / denom   # distance along A's ray
    t = (e - b * d) / denom   # distance along B's ray
    if s <= 0 or t <= 0 or s > max_range:
        return None

    sa, sb = np.radians(sigma_a_deg), np.radians(sigma_b_deg)
    miss = float(np.linalg.norm(pa + s * ua - (pb + t * ub)))
    if miss > 3.0 * np.hypot(s * sa, t * sb) + 2.0:
        return None
    # Lateral error on B's ray slides the crossing along A's ray by 1/sin(gamma); A's own
    # error does so by cot(gamma). The two nodes also hear the drone at different emission
    # times (|s - t| / c apart), so a moving target shifts the crossing; half the worst
    # case is folded in as noise. The 8% floor covers far-field model error at short range
    # (fitted: normalised range error was ~1.6 sigma inside 60 m without it).
    cos_g = abs(b)
    geom = np.hypot(t * sb, s * sa * cos_g) / sin_g
    timing = 0.5 * max_speed * abs(s - t) / SPEED_OF_SOUND / sin_g
    sigma = np.sqrt(geom**2 + timing**2 + (0.08 * s) ** 2) + 0.5
    return RangeFix(float(s), float(sigma), miss, float(np.degrees(np.arcsin(sin_g))))


@dataclass
class FusedFrame:
    a: FrameResult
    b: FrameResult
    fix: RangeFix | None


class TwoNodeSystem:
    def __init__(self, mics_a: np.ndarray, mics_b: np.ndarray, fs: int, **kw):
        self.a = Pipeline(mics_a, fs, **kw)
        self.b = Pipeline(mics_b, fs, **kw)
        self.pa, self.pb = mics_a.mean(axis=0), mics_b.mean(axis=0)
        self.frame, self.hop, self.fs = self.a.frame, self.a.hop, fs

    def _b_bearing(self, rb: FrameResult):
        c = rb.cue
        if c.state == TrackState.CONFIRMED:
            return unit_vector(c.az_deg, c.el_deg), c.sigma_deg
        if rb.doa is not None:
            return unit_vector(rb.doa.az_deg, rb.doa.el_deg), rb.doa.sigma_deg
        return None

    def run(self, xa: np.ndarray, xb: np.ndarray) -> list[FusedFrame]:
        out = []
        for start in range(0, xa.shape[1] - self.frame + 1, self.hop):
            t = (start + self.frame / 2) / self.fs
            t0 = time.perf_counter()
            rb = self.b.step(xb[:, start:start + self.frame], t)
            det, doa, level = self.a.measure(xa[:, start:start + self.frame])
            fix = None
            bb = self._b_bearing(rb)
            if doa is not None and bb is not None:
                fix = triangulate(self.pa, unit_vector(doa.az_deg, doa.el_deg), doa.sigma_deg, self.pb, *bb)
            cue = self.a.tracker.update(t, doa, level, (fix.range_m, fix.sigma_m) if fix else None)
            ms = (time.perf_counter() - t0) * 1e3  # both nodes on one CPU: conservative
            out.append(FusedFrame(FrameResult(t, det, doa, cue, level, ms), rb, fix))
        return out
