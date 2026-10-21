"""Swarm engagement: four FPVs from one sector, two nodes, past a generator.

    python scripts/swarm_demo.py [seed]   # writes out/swarm_demo.png and out/swarm_demo_metrics.json
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from huginn import geometry as g, scenario as sc  # noqa: E402
from huginn.metrics import score_swarm  # noqa: E402
from huginn.swarm import SwarmSystem  # noqa: E402
from huginn.tracker import TrackState  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "out"


def main(seed: int = 11, n: int = 4):
    OUT.mkdir(exist_ok=True)
    mics = g.uca_with_apex()
    mics_b = mics + np.array([40.0, -30.0, 0.0])
    rng = np.random.default_rng(seed)
    raid = sc.swarm_raid(n, rng)
    dur = max(r.t_end for r in raid) + 0.5
    srcs = [r.source() for r in raid] + [sc.generator([-15.0, -12.0, 0.5], dur)]
    _, x = sc.render(sc.Scene(np.vstack([mics, mics_b]), duration=dur, sources=srcs, seed=seed))
    xa, xb = x[: len(mics)], x[len(mics):]
    frames = SwarmSystem(mics, 16000, mics_b).run(xa, xb)

    m = score_swarm(frames, raid, 0.08)
    pub = {k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else v) for k, v in m.items()}
    print(json.dumps(pub, indent=2))
    (OUT / "swarm_demo_metrics.json").write_text(json.dumps(pub, indent=2))

    t = np.array([f.t for f in frames])
    colors = plt.cm.tab10(np.arange(10))
    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])
    fig.suptitle(f"Huginn swarm mode: {n} FPVs from one sector, two nodes, past a generator", fontsize=13)

    ax = fig.add_subplot(gs[0, :])
    f, ts, S = spectrogram(xa[0], fs=16000, nperseg=1024, noverlap=512)
    keep = f <= 2500
    ax.pcolormesh(ts, f[keep], 10 * np.log10(S[keep] + 1e-20), shading="auto", cmap="magma")
    for fr in frames:
        for c in fr.cues:
            if c.state == TrackState.CONFIRMED and np.isfinite(c.f0_hz):
                for k in (2, 4, 6):  # blade-pass family of the folded pitch
                    ax.plot(fr.t, k * c.f0_hz, ".", ms=2, color=colors[c.track_id % 10])
    ax.set(title="Node A spectrogram; dots = harmonics of each confirmed track's pitch fingerprint (colour = track)",
           xlabel="time (s)", ylabel="Hz")

    ac = [g.az_el(sc.acoustic_truth(r.trajectory, t)[0]) for r in raid]
    rng_true = [np.linalg.norm(r.trajectory.pos(t), axis=1) for r in raid]
    alive = [t <= r.t_end for r in raid]
    for col, (name, idx) in enumerate([("azimuth (deg)", 0), ("elevation (deg)", 1)]):
        ax = fig.add_subplot(gs[1, col])
        for d in range(n):
            ax.plot(t[alive[d]], np.unwrap(ac[d][idx][alive[d]], period=360) if idx == 0 else ac[d][idx][alive[d]],
                    "k-", lw=1, alpha=0.6, label="truth (acoustic)" if d == 0 else None)
        tracks = {}
        for fr in frames:
            for c in fr.cues:
                if c.state == TrackState.CONFIRMED:
                    tracks.setdefault(c.track_id, []).append((fr.t, c.az_deg if idx == 0 else c.el_deg))
        for tid, pts in tracks.items():
            p = np.array(pts)
            ax.plot(p[:, 0], p[:, 1], ".", ms=3, color=colors[tid % 10])
        ax.set(xlabel="time (s)", ylabel=name, title=f"Confirmed tracks vs truth: {name.split()[0]}")
        if col == 0:
            ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[2, 0])
    for d in range(n):
        ax.plot(t[alive[d]], rng_true[d][alive[d]], "k-", lw=1, alpha=0.6, label="true range" if d == 0 else None)
    for fr in frames:
        for c in fr.cues:
            if c.state == TrackState.CONFIRMED and np.isfinite(c.range_est_m):
                ax.plot(fr.t, c.range_est_m, ".", ms=3, color=colors[c.track_id % 10])
    ax.set(xlabel="time (s)", ylabel="range (m)", title="Per-drone range (level + two-node triangulation)",
           ylim=(0, 420))
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[2, 1])
    ax.axis("off")
    lines = [f"Drones: {n}   rotor rates: " + ", ".join(f"{r.rotor_hz:.0f}" for r in raid) + " Hz",
             f"Coverage (drone-frames tracked, <=200 m): {m['coverage']:.0%}",
             f"MOTA: {m['mota']:.2f}    ID switches: {m['id_switches']}",
             f"False-track frames: {m['false_track_frames_per_s']:.2f} /s",
             f"Drones ever confirmed: {m['drones_ever_tracked']}/{n}, median first range "
             f"{m['first_confirmed_range_m_median']:.0f} m",
             f"Cue error vs physical (median): {m['cue_err_median_deg']:.1f} deg",
             f"Range error (median): {m['range_err_median_pct']:.0f}%",
             f"Real-time factor (2 nodes, 1 CPU, p99): {m['real_time_factor']:.1f}x"]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=10)

    fig.savefig(OUT / "swarm_demo.png", dpi=120)
    print(f"wrote {OUT / 'swarm_demo.png'}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 11)
