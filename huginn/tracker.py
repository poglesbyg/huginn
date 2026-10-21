"""Bearing-only track filter and the coarse-to-fine handoff cue.

A single passive node measures direction, not range. The filter runs a
constant-velocity Kalman model in (azimuth, elevation) with M-of-N
confirmation, chi-square gating and bounded coasting, and emits a Cue: the
pointing command a fine (visual) tracker needs to acquire the target.

Acoustic lag: the sound arriving now left the drone r/c seconds ago, so the
bearing trails the drone by (tangential speed)/c, up to ~7 deg for a fast
crossing FPV. We lead the cue by r/c + system latency and size the cone by the
residual range uncertainty. Range comes from two sources, fused by inverse
variance in log-range:
  * received tonal level vs a source-level prior (always available, but biased
    by however loud this particular airframe is, so it never averages down);
  * triangulation against a second node (geometry, see fusion.py), filtered as a
    random walk in log-range so it does average down.
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .doa import DoaEstimate
from .geometry import SPEED_OF_SOUND, wrap_deg


class TrackState(str, Enum):
    NONE = "none"
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    COASTING = "coasting"


@dataclass
class Cue:
    t: float
    state: TrackState
    az_deg: float = np.nan           # filtered acoustic bearing (where the sound came from)
    el_deg: float = np.nan
    az_rate_dps: float = np.nan
    el_rate_dps: float = np.nan
    range_est_m: float = np.nan      # from received level vs source-level prior
    lead_az_deg: float = np.nan      # pointing command: led by r/c + system latency
    lead_el_deg: float = np.nan
    sigma_deg: float = np.nan        # 1-sigma uncertainty of the led pointing
    lag_resid_deg: float = np.nan    # lead error bound: range uncertainty + manoeuvre during lead
    cone_deg: float = np.nan         # half-angle the fine sensor should search
    handoff_ready: bool = False
    track_id: int = -1
    f0_hz: float = np.nan            # rotor-comb fingerprint (octave-folded), multi-target only


@dataclass
class TrackerConfig:
    accel_psd: float = 20.0**2        # (deg/s^2)^2 * s, angular manoeuvre noise
    gate_chi2: float = 13.8           # 2 dof, 99.9%
    confirm_m: int = 3
    confirm_n: int = 5
    max_coast_s: float = 0.6
    latency_s: float = 0.080          # frame tail + compute + gimbal slew (Edda-class: <15 ms)
    handoff_fov_deg: float = 20.0     # acquisition camera FOV the cue must land inside
    max_target_speed: float = 45.0    # m/s, caps the lag correction and its error
    max_target_accel: float = 30.0    # m/s^2 (~3 g) lateral manoeuvre during the lead interval
    source_level_db: float = 83.5     # effective detected tonal SPL at 1 m (scripts/calibrate.py)
    source_level_sigma_db: float = 4.0  # airframe-to-airframe spread
    range_smoothing: float = 0.25     # EMA weight on each new level-based log-range
    log_range_walk: float = 0.3       # log10(m)/sqrt(s), random walk of the triangulated range
    tri_timeout_s: float = 0.5        # drop the triangulated estimate if no fix this long


@dataclass
class BearingTracker:
    cfg: TrackerConfig = field(default_factory=TrackerConfig)

    track_id: int = -1

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.state = TrackState.NONE
        self.f0 = np.nan
        self.x = np.zeros(4)  # az, el, az_dot, el_dot
        self.P = np.eye(4)
        self.log_r = np.nan                       # level-based, log10 m
        self.log_r_tri, self.var_tri = np.nan, np.nan  # triangulated, log10 m and variance
        self.t_last_tri = None
        self.t_last = None
        self.t_last_hit = None
        self.history: list[bool] = []

    # -- Kalman primitives -------------------------------------------------
    def _predict(self, t):
        dt = t - self.t_last
        if dt <= 0:
            return
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        q = self.cfg.accel_psd
        Q1 = q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
        Q = np.zeros((4, 4))
        Q[np.ix_([0, 2], [0, 2])] = Q1
        Q[np.ix_([1, 3], [1, 3])] = Q1
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.t_last = t

    def _innovation(self, z: DoaEstimate):
        # Azimuth noise blows up near zenith; scale it by 1/cos(el).
        cos_el = max(np.cos(np.radians(self.x[1])), 0.2)
        R = np.diag([(z.sigma_deg / cos_el) ** 2, z.sigma_deg**2])
        y = np.array([wrap_deg(z.az_deg - self.x[0]), z.el_deg - self.x[1]])
        H = np.zeros((2, 4))
        H[0, 0] = H[1, 1] = 1.0
        S = H @ self.P @ H.T + R
        return y, H, S

    def _start(self, t, z: DoaEstimate, level_db):
        self.x = np.array([z.az_deg, z.el_deg, 0.0, 0.0])
        s2 = z.sigma_deg**2
        self.P = np.diag([s2 * 4, s2, 40.0**2, 40.0**2])
        self.t_last = self.t_last_hit = t
        self.history = [True]
        self.state = TrackState.TENTATIVE
        self.log_r = np.nan
        self.log_r_tri, self.var_tri, self.t_last_tri = np.nan, np.nan, None
        self._update_range(level_db)

    def _update_tri(self, t, range_obs):
        """1-D Kalman update of log10(range) from a triangulated (r, sigma_r) fix."""
        if np.isfinite(self.var_tri):
            self.var_tri += self.cfg.log_range_walk**2 * (t - self.t_last_tri)
            self.t_last_tri = t
            if t - self._t_tri_fix > self.cfg.tri_timeout_s:
                self.log_r_tri, self.var_tri = np.nan, np.nan
        if range_obs is None:
            return
        r, sigma_r = range_obs
        z, R = np.log10(r), (sigma_r / (r * np.log(10))) ** 2
        if not np.isfinite(self.var_tri):
            self.log_r_tri, self.var_tri = z, R
        else:
            k = self.var_tri / (self.var_tri + R)
            self.log_r_tri += k * (z - self.log_r_tri)
            self.var_tri *= 1 - k
        self.t_last_tri = self._t_tri_fix = t

    def range_estimate(self) -> tuple[float, float]:
        """Fused (log10 r, sigma of log10 r); nan if no source is available."""
        est = []
        if np.isfinite(self.log_r):
            est.append((self.log_r, (self.cfg.source_level_sigma_db / 20) ** 2))
        if np.isfinite(self.log_r_tri):
            est.append((self.log_r_tri, self.var_tri))
        if not est:
            return np.nan, np.nan
        w = np.array([1 / v for _, v in est])
        return float(np.sum(w * [m for m, _ in est]) / w.sum()), float(np.sqrt(1 / w.sum()))

    def _update_range(self, level_db):
        if level_db is None or not np.isfinite(level_db):
            return
        # Spherical spreading: L(r) = SL - 20 log10(r). Absorption is small inside 300 m.
        lr = np.clip((self.cfg.source_level_db - level_db) / 20.0, 0.0, 3.0)
        a = self.cfg.range_smoothing
        self.log_r = lr if np.isnan(self.log_r) else (1 - a) * self.log_r + a * lr

    # -- frame steps (used directly by the multi-target tracker) -----------
    def start(self, t, z: DoaEstimate, level_db=None, range_obs=None):
        self._start(t, z, level_db)
        self._update_tri(t, range_obs)

    def predict(self, t):
        self._predict(t)

    def distance2(self, z: DoaEstimate) -> float:
        """Squared Mahalanobis distance of a bearing fix from the predicted track."""
        y, _, S = self._innovation(z)
        return float(y @ np.linalg.solve(S, y))

    def correct(self, t, z: DoaEstimate, level_db=None):
        y, H, S = self._innovation(z)
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[0] %= 360.0
        self.P = (np.eye(4) - K @ H) @ self.P
        self.t_last_hit = t
        self._update_range(level_db)

    def finish(self, t, hit: bool, range_obs=None) -> bool:
        """Close out a frame: range filter, M-of-N confirmation, coasting. False = drop me."""
        self._update_tri(t, range_obs if hit else None)
        self.history = (self.history + [hit])[-self.cfg.confirm_n:]
        if self.state == TrackState.TENTATIVE:
            if sum(self.history) >= self.cfg.confirm_m:
                self.state = TrackState.CONFIRMED
            elif len(self.history) >= self.cfg.confirm_n:
                return False
        elif t - self.t_last_hit > self.cfg.max_coast_s:
            return False
        else:
            self.state = TrackState.CONFIRMED if hit else TrackState.COASTING
        return True

    # -- single-target convenience -------------------------------------------
    def update(self, t: float, z: DoaEstimate | None, level_db: float | None = None,
               range_obs: tuple[float, float] | None = None) -> Cue:
        """range_obs: optional triangulated (range m, 1-sigma m) from another node."""
        if self.state == TrackState.NONE:
            if z is not None:
                self.start(t, z, level_db, range_obs)
            return self._cue(t)

        self.predict(t)
        hit = z is not None and self.distance2(z) <= self.cfg.gate_chi2
        if hit:
            self.correct(t, z, level_db)
        if not self.finish(t, hit, range_obs):
            self.reset()
            if z is not None:
                self._start(t, z, level_db)
        return self._cue(t)

    def cue(self, t) -> Cue:
        return self._cue(t)

    def _cue(self, t) -> Cue:
        if self.state == TrackState.NONE:
            return Cue(t, self.state)
        c = self.cfg
        az, el, vaz, vel = self.x
        cos_el = np.cos(np.radians(el))
        omega = np.radians(np.hypot(vaz * cos_el, vel))  # rad/s on the sky

        # Lead by propagation delay + latency. Range uncertainty (source-level spread) turns
        # into a lag-correction error; both the correction and its error are capped by what an
        # FPV airframe can physically fly (v_t <= max_target_speed).
        log_r, sigma_log_r = self.range_estimate()
        r = 10**log_r if np.isfinite(log_r) else np.nan
        if np.isfinite(r):
            v_t = omega * r
            scale = min(1.0, c.max_target_speed / v_t) if v_t > 0 else 1.0
            lead_s = c.latency_s + scale * r / SPEED_OF_SOUND
            k = 10**sigma_log_r
            resid = min(omega * r * (k - 1 / k) / 2, c.max_target_speed) / SPEED_OF_SOUND
        else:
            lead_s = c.latency_s
            resid = c.max_target_speed / SPEED_OF_SOUND
        # The lead is a constant-rate extrapolation; a drone can pull ~3 g lateral during it.
        # Angular displacement from that over lead_s: 0.5 * (a / r) * lead_s^2.
        if np.isfinite(r):
            resid += 0.5 * c.max_target_accel / max(r, 5.0) * lead_s**2
        resid_deg = float(np.degrees(resid))

        # Extrapolating over lead_s amplifies rate noise. Shrink each rate toward zero by its
        # own SNR (a Wiener-style gain) and grow sigma by the propagated covariance, so a
        # long lead on a shaky track widens the cone instead of pointing confidently wrong.
        gain = np.array([vaz**2 / (vaz**2 + self.P[2, 2]), vel**2 / (vel**2 + self.P[3, 3])])
        F = np.eye(4)
        F[0, 2], F[1, 3] = gain * lead_s
        Pp = F @ self.P @ F.T
        sigma = float(np.sqrt(max(Pp[0, 0] * cos_el**2, Pp[1, 1])))
        cone = 2 * sigma + resid_deg
        ready = self.state == TrackState.CONFIRMED and cone < c.handoff_fov_deg / 2
        return Cue(t, self.state, float(az), float(el), float(vaz), float(vel), float(r),
                   float((az + gain[0] * vaz * lead_s) % 360.0), float(el + gain[1] * vel * lead_s),
                   sigma, resid_deg, float(cone), bool(ready), self.track_id, float(self.f0))
