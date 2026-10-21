"""Multi-target (swarm) tracking.

Separation is spectral first, spatial second. A 0.5 m array has a beam tens of
degrees wide, so two drones in formation are unresolvable by angle. Their
rotor rates differ, though. Each frame:

  1. `HarmonicDetector.detect_all` peels off one comb per airframe.
  2. Each comb's bins go to SRP-PHAT separately, so each drone gets its own
     bearing even when the drones are spatially unresolvable.
  2b. Track-guided re-detection: for every existing track the comb search missed,
     steer a delay-and-sum beam at its predicted direction (up to +9.5 dB coherent
     gain with 9 mics) and look for a comb near its known pitch. This keeps weak
     drones alive next to loud ones.
  3. Combs that agree in direction and folded pitch are merged (one quad's four
     rotors can drift apart enough to look like two combs).
  4. A global-nearest-neighbour multi-target tracker assigns fixes to tracks on
     bearing (Mahalanobis) plus pitch fingerprint, then runs each track's
     lifecycle (tentative -> confirmed -> coasting -> dropped).

With two nodes, A's fixes are paired with B's confirmed tracks on pitch and
ray-crossing quality, so each drone gets its own triangulated range.
"""

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .detect import Detection, HarmonicDetector, fold_f0, tonal_level_db
from .doa import DoaEstimate, SrpPhat
from .fusion import RangeFix, triangulate
from .geometry import angular_separation_deg, unit_vector
from .tracker import BearingTracker, Cue, TrackerConfig, TrackState

BIG = 1e9


@dataclass
class Measurement:
    doa: DoaEstimate
    level_db: float
    f0_hz: float          # octave-folded comb fundamental
    mask: np.ndarray


def _log_ratio(a, b):
    return abs(np.log(a / b)) if np.isfinite(a) and np.isfinite(b) else 0.0


class SwarmNode:
    """One array: frame -> list of per-airframe bearing measurements."""

    def __init__(self, mics: np.ndarray, fs: int, frame: int = 1024, hop: int = 512,
                 max_targets: int = 10, merge_deg: float = 4.0, merge_pitch: float = 0.08):
        self.fs, self.frame, self.hop = fs, frame, hop
        self.window = np.hanning(frame)
        self.freqs = np.fft.rfftfreq(frame, 1.0 / fs)
        self.detector = HarmonicDetector(self.freqs, n_channels=len(mics))
        self.doa = SrpPhat(mics, self.freqs)
        self.centre = mics.mean(axis=0)
        self.max_targets = max_targets
        self.merge_deg, self.merge_pitch = merge_deg, merge_pitch
        self.beam_accept_deg = 6.0

    def _measure_one(self, X, mask, snr, f0) -> Measurement | None:
        doa = self.doa(X, mask, snr)
        if doa is None:
            return None
        return Measurement(doa, tonal_level_db(X, mask, self.window), fold_f0(f0), mask)

    def measure(self, block: np.ndarray, looks: list[tuple[float, float, float]] = ()
                ) -> tuple[list[Measurement], list[Detection]]:
        """looks: predicted (az, el, folded pitch) of existing tracks, for beam re-detection."""
        X = np.fft.rfft(block * self.window, axis=-1)
        dets = self.detector.detect_all(X, self.max_targets)
        meas = [m for d in dets if (m := self._measure_one(X, d.bin_mask, d.snr_db, d.f0_hz))]

        snr = self.detector.whiten(X)
        for az, el, f0 in looks:
            if not np.isfinite(f0) or any(
                    angular_separation_deg(az, el, m.doa.az_deg, m.doa.el_deg) < 2 * self.merge_deg
                    and _log_ratio(f0, m.f0_hz) < np.log(1.2) for m in meas):
                continue
            det = self.detector.detect_near(self.doa.beam_power(X, az, el), f0)
            if not det.detected:
                continue
            # A beam also hears loud drones through its (wide) sidelobes. So localise the
            # beam-found comb over the WHOLE sky and accept it only if it really comes from
            # inside the beam; a leaked comb points back at the drone it belongs to.
            # (A cone-restricted search here lets a track walk off following the cone edge.)
            doa = self.doa(X, det.bin_mask, snr)  # sigma from array-averaged SNR, as calibrated
            if doa is not None and angular_separation_deg(az, el, doa.az_deg, doa.el_deg) < self.beam_accept_deg:
                meas.append(Measurement(doa, tonal_level_db(X, det.bin_mask, self.window),
                                        fold_f0(det.f0_hz), det.bin_mask))

        # Merge fixes from the same airframe: same direction AND same folded pitch.
        merged = True
        while merged and len(meas) > 1:
            merged = False
            for i in range(len(meas)):
                for j in range(i + 1, len(meas)):
                    a, b = meas[i], meas[j]
                    sep = angular_separation_deg(a.doa.az_deg, a.doa.el_deg, b.doa.az_deg, b.doa.el_deg)
                    if sep < self.merge_deg and _log_ratio(a.f0_hz, b.f0_hz) < np.log(1 + self.merge_pitch):
                        m = self._measure_one(X, a.mask | b.mask, snr, a.f0_hz)
                        if m is not None:
                            meas = [x for k, x in enumerate(meas) if k not in (i, j)] + [m]
                            merged = True
                            break
                if merged:
                    break
        return meas, dets


@dataclass
class MultiTargetTracker:
    cfg: TrackerConfig = field(default_factory=TrackerConfig)
    pitch_sigma: float = 0.06       # relative; throttle and Doppler move pitch between frames
    pitch_gate: float = 0.20        # never associate across a >20% pitch gap
    max_tracks: int = 10
    dup_deg: float = 2.0            # tracks closer than this with the same pitch are duplicates
    dup_pitch: float = 0.06         # "same pitch" for duplicates is much tighter than the gate
    pitch_smoothing: float = 0.2

    def __post_init__(self):
        self.tracks: list[BearingTracker] = []
        self._next_id = 0

    def _cost(self, trk: BearingTracker, m: Measurement) -> float:
        d2 = trk.distance2(m.doa)
        lr = _log_ratio(m.f0_hz, trk.f0)
        if d2 > self.cfg.gate_chi2 or lr > np.log(1 + self.pitch_gate):
            return BIG
        return d2 + (lr / self.pitch_sigma) ** 2

    def update(self, t: float, meas: list[Measurement], range_obs: dict[int, RangeFix] | None = None) -> list[Cue]:
        range_obs = range_obs or {}
        for trk in self.tracks:
            trk.predict(t)

        pairs: dict[int, int] = {}  # track row -> measurement column
        if self.tracks and meas:
            C = np.array([[self._cost(trk, m) for m in meas] for trk in self.tracks])
            # Confirmed tracks get first claim on ambiguous fixes over tentative ones.
            C += np.array([[0.0 if trk.state == TrackState.CONFIRMED else 2.0] for trk in self.tracks])
            rows, cols = linear_sum_assignment(C)
            for r, c in zip(rows, cols):
                if C[r, c] < BIG:
                    trk, m = self.tracks[r], meas[c]
                    trk.correct(t, m.doa, m.level_db)
                    a = self.pitch_smoothing
                    trk.f0 = m.f0_hz if not np.isfinite(trk.f0) else float(
                        np.exp((1 - a) * np.log(trk.f0) + a * np.log(m.f0_hz)))
                    pairs[r] = c

        survivors = []
        for r, trk in enumerate(self.tracks):
            fix = range_obs.get(pairs[r]) if r in pairs else None
            if trk.finish(t, r in pairs, (fix.range_m, fix.sigma_m) if fix else None):
                survivors.append(trk)
        self.tracks = survivors

        used = set(pairs.values())
        for c, m in enumerate(meas):
            if c in used or len(self.tracks) >= self.max_tracks:
                continue
            if any(self._same_target(trk.x[0], trk.x[1], trk.f0, m) for trk in self.tracks):
                continue
            trk = BearingTracker(self.cfg, track_id=self._next_id)
            self._next_id += 1
            fix = range_obs.get(c)
            trk.start(t, m.doa, m.level_db, (fix.range_m, fix.sigma_m) if fix else None)
            trk.f0 = m.f0_hz
            self.tracks.append(trk)

        self._prune_duplicates()
        return [trk.cue(t) for trk in self.tracks]

    def _same_target(self, az, el, f0, m: Measurement) -> bool:
        sep = angular_separation_deg(az, el, m.doa.az_deg, m.doa.el_deg)
        return sep < self.dup_deg * 2 and _log_ratio(f0, m.f0_hz) < np.log(1 + self.dup_pitch)

    def _prune_duplicates(self):
        """Two tracks on the same direction and pitch are one drone: keep the older."""
        keep = []
        for trk in sorted(self.tracks, key=lambda k: k.track_id):
            dup = any(angular_separation_deg(trk.x[0], trk.x[1], k.x[0], k.x[1]) < self.dup_deg
                      and _log_ratio(trk.f0, k.f0) < np.log(1 + self.dup_pitch) for k in keep)
            if not dup:
                keep.append(trk)
        self.tracks = keep


@dataclass
class SwarmFrame:
    t: float
    cues: list[Cue]                 # node A's tracks (the cueing node)
    meas: list[Measurement]
    fixes: dict[int, RangeFix]
    compute_ms: float


class SwarmSystem:
    """One or two nodes. With `mics_b`, A's fixes get per-drone triangulated range."""

    def __init__(self, mics_a: np.ndarray, fs: int, mics_b: np.ndarray | None = None,
                 tracker_cfg: TrackerConfig | None = None):
        self.a = SwarmNode(mics_a, fs)
        self.b = SwarmNode(mics_b, fs) if mics_b is not None else None
        self.trk_a = MultiTargetTracker(tracker_cfg or TrackerConfig())
        self.trk_b = MultiTargetTracker(tracker_cfg or TrackerConfig()) if self.b else None

    def _cross_associate(self, meas_a: list[Measurement], cues_b: list[Cue]) -> dict[int, RangeFix]:
        """Pair A fixes with B's confirmed tracks: pitch must match, rays must cross."""
        cues_b = [c for c in cues_b if c.state == TrackState.CONFIRMED]
        if not meas_a or not cues_b:
            return {}
        C = np.full((len(meas_a), len(cues_b)), BIG)
        fixes = {}
        for i, m in enumerate(meas_a):
            ua = unit_vector(m.doa.az_deg, m.doa.el_deg)
            for j, cb in enumerate(cues_b):
                lr = _log_ratio(m.f0_hz, cb.f0_hz)
                if lr > np.log(1.08):
                    continue
                fix = triangulate(self.a.centre, ua, m.doa.sigma_deg, self.b.centre,
                                  unit_vector(cb.az_deg, cb.el_deg), cb.sigma_deg)
                if fix is None:
                    continue
                fixes[i, j] = fix
                C[i, j] = (lr / 0.03) ** 2 + (fix.miss_m / max(fix.sigma_m, 1.0)) ** 2
        rows, cols = linear_sum_assignment(C)
        return {r: fixes[r, c] for r, c in zip(rows, cols) if C[r, c] < BIG}

    @staticmethod
    def _looks(trk: MultiTargetTracker, t: float):
        out = []
        for k in trk.tracks:
            if k.state != TrackState.CONFIRMED and k.state != TrackState.COASTING:
                continue
            dt = t - k.t_last
            out.append(((k.x[0] + k.x[2] * dt) % 360.0, k.x[1] + k.x[3] * dt, k.f0))
        return out

    def run(self, xa: np.ndarray, xb: np.ndarray | None = None) -> list[SwarmFrame]:
        node = self.a
        out = []
        for start in range(0, xa.shape[1] - node.frame + 1, node.hop):
            t = (start + node.frame / 2) / node.fs
            t0 = time.perf_counter()
            fixes = {}
            meas_a, _ = self.a.measure(xa[:, start:start + node.frame], self._looks(self.trk_a, t))
            if self.b is not None:
                meas_b, _ = self.b.measure(xb[:, start:start + node.frame], self._looks(self.trk_b, t))
                cues_b = self.trk_b.update(t, meas_b)
                fixes = self._cross_associate(meas_a, cues_b)
            cues = self.trk_a.update(t, meas_a, fixes)
            out.append(SwarmFrame(t, cues, meas_a, fixes, (time.perf_counter() - t0) * 1e3))
        return out
