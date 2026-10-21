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


def score_swarm(frames, raiders, latency_s: float, eval_range_m: float = 200.0, match_deg: float = 5.0) -> dict:
    """CLEAR-MOT style scoring of a multi-target run against simulator truth.

    A confirmed track matches a drone if its acoustic bearing is within `match_deg`
    of that drone's acoustic truth (Hungarian assignment per frame). Ground truth
    counts only drones that are still airborne and inside `eval_range_m`; a track on
    a farther drone is not a false positive, just not credited.
    """
    from scipy.optimize import linear_sum_assignment

    t = np.array([f.t for f in frames])
    ac_dirs, ranges, phys = [], [], []
    for rd in raiders:
        ac, now = acoustic_truth(rd.trajectory, t)
        ac_dirs.append(g.az_el(ac))
        ranges.append(np.linalg.norm(now, axis=1))
        phys.append(g.az_el(rd.trajectory.pos(t + latency_s)))
    alive = np.array([t <= rd.t_end for rd in raiders])            # (D, T)
    in_env = alive & (np.array(ranges) <= eval_range_m)

    gt = tp = fn = fp = idsw = 0
    last_id = [None] * len(raiders)
    first_conf = [None] * len(raiders)
    cue_err, rng_err, contained = [], [], []
    for k, f in enumerate(frames):
        cues = [c for c in f.cues if c.state == TrackState.CONFIRMED]
        live = [d for d in range(len(raiders)) if alive[d, k]]
        C = np.full((len(cues), len(live)), 1e9)
        for i, c in enumerate(cues):
            for j, d in enumerate(live):
                sep = g.angular_separation_deg(c.az_deg, c.el_deg, ac_dirs[d][0][k], ac_dirs[d][1][k])
                if sep <= match_deg:
                    C[i, j] = sep
        # CLEAR-MOT: keep last frame's correspondence while it is still within the gate,
        # then assign the rest optimally. Re-solving from scratch every frame would count
        # flips between two nearby drones as identity switches the tracker never made.
        matched, used = {}, set()
        for j, d in enumerate(live):
            for i, c in enumerate(cues):
                if i not in used and c.track_id == last_id[d] and C[i, j] < 1e9:
                    matched[d] = c
                    used.add(i)
                    break
        rest_c = [i for i in range(len(cues)) if i not in used]
        rest_d = [j for j, d in enumerate(live) if d not in matched]
        if rest_c and rest_d:
            sub = C[np.ix_(rest_c, rest_d)]
            for a, b in zip(*linear_sum_assignment(sub)):
                if sub[a, b] < 1e9:
                    matched[live[rest_d[b]]] = cues[rest_c[a]]
        n_env = int(in_env[:, k].sum())
        gt += n_env
        for d in range(len(raiders)):
            if not in_env[d, k]:
                continue
            if d in matched:
                tp += 1
            else:
                fn += 1
        fp += len(cues) - len(matched)
        for d, c in matched.items():
            if first_conf[d] is None:
                first_conf[d] = float(ranges[d][k])
            if last_id[d] is not None and c.track_id != last_id[d]:
                idsw += 1
            last_id[d] = c.track_id
            err = g.angular_separation_deg(c.lead_az_deg, c.lead_el_deg, phys[d][0][k], phys[d][1][k])
            cue_err.append(err)
            if np.isfinite(c.range_est_m):
                rng_err.append(abs(c.range_est_m / ranges[d][k] - 1))
            if c.handoff_ready:
                contained.append(err <= c.cone_deg)

    ms = np.array([f.compute_ms for f in frames])
    hop_ms = 1e3 * (t[1] - t[0])
    return {
        "n_drones": len(raiders),
        "coverage": tp / gt if gt else None,           # fraction of in-envelope drone-frames tracked
        "false_track_frames_per_s": fp / (t[-1] - t[0]),
        "id_switches": idsw,
        "mota": 1 - (fn + fp + idsw) / gt if gt else None,
        "drones_ever_tracked": sum(r is not None for r in first_conf),
        "first_confirmed_range_m_median": float(np.median([r for r in first_conf if r is not None]))
        if any(r is not None for r in first_conf) else None,
        "cue_err_median_deg": float(np.median(cue_err)) if cue_err else None,
        "range_err_median_pct": 100 * float(np.median(rng_err)) if rng_err else None,
        "handoff_containment": float(np.mean(contained)) if contained else None,
        "real_time_factor": float(hop_ms / np.percentile(ms, 99)),
    }
