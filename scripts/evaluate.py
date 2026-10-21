"""Monte Carlo evaluation: randomised engagements + clutter-only false-alarm runs.

Each engagement is rendered once onto both arrays and scored twice on identical
audio: node A alone (range from level) and A+B fused (range from triangulation).

    python scripts/evaluate.py [n_engagements] [clutter_minutes]
Writes out/evaluation.json and out/evaluation.png.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from huginn import geometry as g, scenario as sc  # noqa: E402
from huginn.fusion import TwoNodeSystem  # noqa: E402
from huginn.metrics import public, score_run  # noqa: E402
from huginn.pipeline import Pipeline  # noqa: E402
from huginn.tracker import TrackState  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "out"
NODE_B_OFFSET = np.array([40.0, -30.0, 0.0])  # 50 m baseline; approaches come from every azimuth


def engagements(n, rng):
    mics = g.uca_with_apex()
    mics_b = mics + NODE_B_OFFSET
    runs = {"single": [], "dual": []}
    curves = {"single": [], "dual": []}
    for i in range(n):
        traj = sc.inbound_fpv(start_range=350, az_deg=rng.uniform(0, 360), alt=rng.uniform(15, 60),
                              speed=rng.uniform(20, 40), weave_amp=rng.uniform(0, 30),
                              weave_period=rng.uniform(3, 7))
        drone = sc.fpv_quad(traj, rotor_hz=rng.uniform(220, 300), level_db_1m=rng.normal(86, 3))
        gen_az = rng.uniform(0, 2 * np.pi)
        clutter = [sc.generator([20 * np.cos(gen_az), 20 * np.sin(gen_az), 0.5], traj.duration,
                                mains_hz=rng.choice([50.0, 60.0]))]
        scene = sc.Scene(np.vstack([mics, mics_b]), duration=traj.duration, sources=[drone] + clutter,
                         seed=1000 + i, ambient_db=rng.uniform(40, 52), wind_db=rng.uniform(50, 65))
        _, x = sc.render(scene)
        xa, xb = x[: len(mics)], x[len(mics):]
        pipe = Pipeline(mics, scene.fs)
        lat = pipe.tracker.cfg.latency_s
        fused = TwoNodeSystem(mics, mics_b, scene.fs).run(xa, xb)
        for mode, res in (("single", pipe.run(xa)), ("dual", [f.a for f in fused])):
            m = score_run(res, traj, lat)
            runs[mode].append(public(m))
            curves[mode].append(m["_series"])
        n_fix = sum(f.fix is not None for f in fused)
        runs["dual"][-1]["triangulation_fix_rate"] = n_fix / max(1, sum(f.a.doa is not None for f in fused))
        print(f"engagement {i}: handoff at {runs['single'][-1]['first_handoff_range_m']} m (single) / "
              f"{runs['dual'][-1]['first_handoff_range_m']} m (dual), range err "
              f"{runs['single'][-1]['range_est_err_median_pct']}% / {runs['dual'][-1]['range_est_err_median_pct']}%")
    return runs, curves


def clutter_only(minutes, rng):
    """No drone: generators, wind gusts, ambient. Any confirmed track is a false track."""
    mics = g.uca_with_apex()
    false_tracks, frames, false_frames = 0, 0, 0
    chunks = int(np.ceil(minutes * 2))  # 30 s scenes
    for i in range(chunks):
        srcs = [sc.generator(rng.uniform(-40, 40, 3) * [1, 1, 0] + [0, 0, 0.5], 30.0,
                             mains_hz=rng.choice([50.0, 60.0]), level_db_1m=rng.uniform(75, 92))
                for _ in range(rng.integers(1, 4))]
        scene = sc.Scene(mics, duration=30.0, sources=srcs, seed=5000 + i,
                         ambient_db=rng.uniform(40, 55), wind_db=rng.uniform(55, 70))
        _, x = sc.render(scene)
        res = Pipeline(mics, scene.fs).run(x)
        conf = np.array([r.cue.state == TrackState.CONFIRMED for r in res])
        false_tracks += int(np.sum(np.diff(conf.astype(int)) == 1) + conf[0])
        false_frames += int(conf.sum())
        frames += len(res)
    hours = chunks * 30.0 / 3600
    return {"clutter_minutes": chunks * 0.5, "false_tracks": false_tracks,
            "false_tracks_per_hour": false_tracks / hours, "false_confirmed_frame_rate": false_frames / frames}


def summarise(runs):
    keys = [k for k in runs[0] if isinstance(runs[0][k], (int, float))]
    out = {}
    for k in keys:
        v = np.array([r[k] for r in runs if r[k] is not None], dtype=float)
        if v.size:
            out[k] = {"median": round(float(np.median(v)), 3), "p10": round(float(np.percentile(v, 10)), 3),
                      "p90": round(float(np.percentile(v, 90)), 3), "n": int(v.size)}
    return out


def binned(curves, fn, bins):
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        vals = [fn(c)[(c["range"] >= lo) & (c["range"] < hi)] for c in curves]
        v = np.concatenate(vals).astype(float)
        out.append(np.nanmean(v) if v.dtype == bool or np.isfinite(v).any() else np.nan)
    return np.array(out)


def plot(curves, path):
    bins = np.arange(0, 360, 20)
    x = bins[:-1] + 10
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), constrained_layout=True)
    a1, a2, a3 = axes
    sc_ = curves["single"]
    a1.plot(x, binned(sc_, lambda c: c["det"], bins), "o-", color="tab:blue", label="P(detect) per frame")
    a1.plot(x, binned(sc_, lambda c: c["conf"], bins), "s-", color="tab:orange", label="P(confirmed track)")
    for mode, ls in (("single", "--"), ("dual", "-")):
        a1.plot(x, binned(curves[mode], lambda c: c["ready"], bins), "^" + ls, color="tab:green",
                label=f"P(handoff ready), {'1 node' if mode == 'single' else '2 nodes'}")
        rel = lambda c: np.abs(c["range_est"] / c["range"] - 1) * 100  # noqa: E731
        a2.plot(x, [np.nanmedian(v) if np.isfinite(v).any() else np.nan for v in
                    [np.concatenate([rel(c)[(c["range"] >= lo) & (c["range"] < hi) & c["conf"]] for c in curves[mode]])
                     for lo, hi in zip(bins[:-1], bins[1:])]],
                "o" + ls, label="1 node: level prior" if mode == "single" else "2 nodes: triangulation")
        a3.plot(x, [np.nanmedian(v) if np.isfinite(v).any() else np.nan for v in
                    [np.concatenate([c["trk_err"][(c["range"] >= lo) & (c["range"] < hi)] for c in curves[mode]])
                     for lo, hi in zip(bins[:-1], bins[1:])]],
                "s" + ls, label=f"cue vs physical, {'1 node' if mode == 'single' else '2 nodes'}")
    a1.set(xlabel="range (m)", ylabel="probability", ylim=(0, 1.05),
           title=f"Performance envelope, {len(sc_)} randomised engagements")
    a2.set(xlabel="range (m)", ylabel="median |range error| (%)", title="Range estimate error (confirmed tracks)")
    a3.set(xlabel="range (m)", ylabel="median error (deg)", title="Cue pointing error vs range")
    for a in axes:
        a.invert_xaxis()
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.savefig(path, dpi=130)


def main(n=12, clutter_minutes=3.0):
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(2026)
    runs, curves = engagements(n, rng)
    fa = clutter_only(clutter_minutes, rng)
    report = {"single_node": summarise(runs["single"]), "two_node": summarise(runs["dual"]), "clutter": fa,
              "node_b_offset_m": NODE_B_OFFSET.tolist()}
    print(json.dumps(report, indent=2))
    (OUT / "evaluation.json").write_text(json.dumps(report, indent=2))
    plot(curves, OUT / "evaluation.png")
    print(f"wrote {OUT / 'evaluation.png'}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12, float(sys.argv[2]) if len(sys.argv) > 2 else 3.0)
