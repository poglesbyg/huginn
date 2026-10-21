"""Swarm evaluation.

1. Resolution map: two drones hovering 80 m out, swept over angular separation and
   rotor-pitch difference. Shows what one 0.5 m array can and cannot pull apart.
2. Raids: n FPVs (1, 2, 3, 5) from a 60-degree threat sector, two nodes, scored
   CLEAR-MOT style against truth.

    python scripts/evaluate_swarm.py [seeds_per_size]
Writes out/swarm_evaluation.json and out/swarm_resolution.png.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from huginn import geometry as g, scenario as sc  # noqa: E402
from huginn.metrics import score_swarm  # noqa: E402
from huginn.swarm import SwarmSystem  # noqa: E402
from huginn.tracker import TrackState  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "out"
NODE_B_OFFSET = np.array([40.0, -30.0, 0.0])
SEPS_DEG = [0, 3, 6, 10, 20, 40]
PITCH_PCT = [0, 3, 6, 10, 15]


def pair_resolved(sep_deg, pitch_pct, seed, rng_m=80.0, duration=2.5):
    """Fraction of frames (after 1 s settle) in which both drones hold distinct tracks."""
    mics = g.uca_with_apex()
    az0, el0 = 60.0, 10.0
    dirs = [(az0 - sep_deg / 2, el0), (az0 + sep_deg / 2, el0)]
    base = 250.0
    srcs, truth = [], []
    for (az, el), f in zip(dirs, (base, base * (1 + pitch_pct / 100))):
        pos = rng_m * g.unit_vector(az, el)
        srcs.append(sc.fpv_quad(sc.static(pos, duration), rotor_hz=f))
        truth.append((az, el))
    _, x = sc.render(sc.Scene(mics, duration=duration, sources=srcs, seed=seed))
    frames = SwarmSystem(mics, 16000).run(x)
    ok = n = 0
    for f in frames:
        if f.t < 1.0:
            continue
        cues = [c for c in f.cues if c.state == TrackState.CONFIRMED]
        n += 1
        if len(cues) < 2:
            continue
        C = np.array([[g.angular_separation_deg(c.az_deg, c.el_deg, *tr) for tr in truth] for c in cues])
        r, c = linear_sum_assignment(C)
        ok += int(np.sum(C[r, c] <= max(3.0, sep_deg / 2)) == 2)
    return ok / max(n, 1)


def resolution_map():
    grid = np.zeros((len(PITCH_PCT), len(SEPS_DEG)))
    for i, dp in enumerate(PITCH_PCT):
        for j, ds in enumerate(SEPS_DEG):
            grid[i, j] = np.mean([pair_resolved(ds, dp, seed) for seed in (1, 2)])
        print(f"pitch {dp:>2}%: " + " ".join(f"{v:4.2f}" for v in grid[i]))
    return grid


def raids(seeds):
    mics = g.uca_with_apex()
    mics_b = mics + NODE_B_OFFSET
    results = {}
    for n in (1, 2, 3, 5):
        runs = []
        for s in range(seeds):
            rng = np.random.default_rng(100 * n + s)
            raid = sc.swarm_raid(n, rng)
            dur = max(r.t_end for r in raid) + 0.5
            gen_az = rng.uniform(0, 2 * np.pi)
            srcs = [r.source() for r in raid] + [
                sc.generator([20 * np.cos(gen_az), 20 * np.sin(gen_az), 0.5], dur, mains_hz=rng.choice([50.0, 60.0]))]
            scene = sc.Scene(np.vstack([mics, mics_b]), duration=dur, sources=srcs, seed=7000 + 10 * n + s,
                             ambient_db=rng.uniform(40, 52), wind_db=rng.uniform(50, 65))
            _, x = sc.render(scene)
            frames = SwarmSystem(mics, scene.fs, mics_b).run(x[: len(mics)], x[len(mics):])
            m200 = score_swarm(frames, raid, 0.08, eval_range_m=200.0)
            m100 = score_swarm(frames, raid, 0.08, eval_range_m=100.0)
            m200["coverage_within_100m"] = m100["coverage"]
            m200["drone_minutes"] = sum(min(r.t_end, dur) for r in raid) / 60
            runs.append(m200)
            print(f"n={n} seed={s}: coverage {m200['coverage']:.2f} (<=100 m {m100['coverage']:.2f}), "
                  f"MOTA {m200['mota']:.2f}, IDSW {m200['id_switches']}, "
                  f"false {m200['false_track_frames_per_s']:.2f}/s, range err {m200['range_err_median_pct']}")
        keys = ["coverage", "coverage_within_100m", "mota", "false_track_frames_per_s", "cue_err_median_deg",
                "range_err_median_pct", "handoff_containment", "real_time_factor"]
        agg = {k: round(float(np.median([r[k] for r in runs if r[k] is not None])), 3) for k in keys}
        agg["id_switches_per_drone_minute"] = round(
            sum(r["id_switches"] for r in runs) / sum(r["drone_minutes"] for r in runs), 2)
        agg["drones_ever_tracked"] = f"{sum(r['drones_ever_tracked'] for r in runs)}/{n * seeds}"
        results[n] = agg
    return results


def plot(grid, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
    im = ax.imshow(grid, origin="lower", cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(SEPS_DEG)), [f"{s}°" for s in SEPS_DEG])
    ax.set_yticks(range(len(PITCH_PCT)), [f"{p}%" for p in PITCH_PCT])
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                    color="white" if grid[i, j] < 0.6 else "black", fontsize=9)
    ax.set(xlabel="angular separation", ylabel="rotor-pitch difference",
           title="Two drones at 80 m: fraction of time both are tracked separately")
    fig.colorbar(im, ax=ax, label="P(both resolved)")
    fig.savefig(path, dpi=130)


def main(seeds=4):
    OUT.mkdir(exist_ok=True)
    print("resolution map (rows: pitch difference, cols:", SEPS_DEG, "deg)")
    grid = resolution_map()
    plot(grid, OUT / "swarm_resolution.png")
    res = raids(seeds)
    report = {"resolution_map": {"separation_deg": SEPS_DEG, "pitch_diff_pct": PITCH_PCT,
                                 "p_both_resolved": np.round(grid, 3).tolist()},
              "raids_two_node": res}
    print(json.dumps(report["raids_two_node"], indent=2))
    (OUT / "swarm_evaluation.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
