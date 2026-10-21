"""Scoring a pipeline run against simulator truth."""

import numpy as np

from . import geometry as g
from .pipeline import FrameResult
from .scenario import Trajectory, acoustic_truth
from .tracker import TrackState

GROSS_ERROR_DEG = 5.0


def score_run(results: list[FrameResult], traj: Trajectory, latency_s: float) -> dict:
    t = np.array([r.t for r in results])
    ac, now = acoustic_truth(traj, t)
    rng = np.linalg.norm(now, axis=1)
    aaz, ael = g.az_el(ac)
    # The cue is consumed `latency_s` later, so score the lead-predicted pointing there.
    paz, pel = g.az_el(traj.pos(t + latency_s))

    det = np.array([r.detection.detected for r in results])
    conf = np.array([r.cue.state == TrackState.CONFIRMED for r in results])
    ready = np.array([r.cue.handoff_ready for r in results])

    raw_err = np.array([g.angular_separation_deg(r.doa.az_deg, r.doa.el_deg, aaz[i], ael[i]) if r.doa else np.nan
                        for i, r in enumerate(results)])
    trk_err = np.array([g.angular_separation_deg(r.cue.lead_az_deg, r.cue.lead_el_deg, paz[i], pel[i])
                        if r.cue.state in (TrackState.CONFIRMED, TrackState.COASTING) else np.nan
                        for i, r in enumerate(results)])
    cone = np.array([r.cue.cone_deg for r in results])
    r_est = np.array([r.cue.range_est_m for r in results])

    def first_range(mask, window_s=1.0, frac=0.5):
        """Range at which `mask` first holds for `frac` of a trailing window. Ignores
        isolated blips so the number means "sustained", not "lucky frame"."""
        n = max(1, int(round(window_s / (t[1] - t[0])))) if len(t) > 1 else 1
        rate = np.convolve(mask.astype(float), np.ones(n) / n)[: len(mask)]
        hit = rate >= frac
        return float(rng[np.argmax(hit)]) if hit.any() else None

    valid = ~np.isnan(raw_err)
    inl = valid & (raw_err < GROSS_ERROR_DEG)
    ms = np.array([r.compute_ms for r in results])
    hop_ms = 1e3 * (t[1] - t[0]) if len(t) > 1 else np.nan
    return {
        "sustained_detection_range_m": first_range(det),
        "first_confirmed_range_m": first_range(conf),
        "first_handoff_range_m": first_range(ready),
        "raw_bearing_err_median_deg": float(np.median(raw_err[inl])) if inl.any() else None,
        "raw_bearing_err_p90_deg": float(np.percentile(raw_err[inl], 90)) if inl.any() else None,
        "raw_gross_error_rate": float(np.mean(raw_err[valid] >= GROSS_ERROR_DEG)) if valid.any() else None,
        "track_vs_physical_err_median_deg": float(np.nanmedian(trk_err)) if np.isfinite(trk_err).any() else None,
        "track_vs_physical_err_p90_deg": float(np.nanpercentile(trk_err, 90)) if np.isfinite(trk_err).any() else None,
        "handoff_containment": float(np.mean(trk_err[ready] <= cone[ready])) if ready.any() else None,
        "range_est_err_median_pct": float(np.nanmedian(np.abs(r_est[conf] / rng[conf] - 1)) * 100) if conf.any() else None,
        "compute_ms_mean": float(ms.mean()),
        "compute_ms_p99": float(np.percentile(ms, 99)),
        "real_time_factor": float(hop_ms / np.percentile(ms, 99)),
        "_series": {"t": t, "range": rng, "acoustic_az": aaz, "acoustic_el": ael, "phys_az": paz, "phys_el": pel,
                    "raw_err": raw_err, "trk_err": trk_err, "cone": cone, "range_est": r_est, "det": det, "conf": conf, "ready": ready},
    }


def public(metrics: dict) -> dict:
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in metrics.items() if not k.startswith("_")}
