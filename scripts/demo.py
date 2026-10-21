"""Single engagement: an FPV runs in at 30 m/s past a running generator.

Node A cues the camera; node B sits 50 m away and supplies triangulated range.
Both modes are scored on the same audio; the figure shows the two-node cue.

    python scripts/demo.py            # writes out/demo.png and out/demo_metrics.json
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
from huginn.fusion import TwoNodeSystem  # noqa: E402
from huginn.metrics import public, score_run  # noqa: E402
from huginn.pipeline import Pipeline  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "out"


def main(seed: int = 7):
    OUT.mkdir(exist_ok=True)
    mics = g.uca_with_apex()
    mics_b = mics + np.array([40.0, -30.0, 0.0])
    traj = sc.inbound_fpv(start_range=400, az_deg=40, alt=35, speed=30, weave_amp=20)
    gen_pos = np.array([-15.0, -12.0, 0.5])
    scene = sc.Scene(np.vstack([mics, mics_b]), duration=traj.duration, seed=seed,
                     sources=[sc.fpv_quad(traj), sc.generator(gen_pos, traj.duration)])
    _, x_all = sc.render(scene)
    x, xb = x_all[: len(mics)], x_all[len(mics):]

    pipe = Pipeline(mics, scene.fs)
    lat = pipe.tracker.cfg.latency_s
    single = public(score_run(pipe.run(x), traj, lat))
    results = [f.a for f in TwoNodeSystem(mics, mics_b, scene.fs).run(x, xb)]
    m = score_run(results, traj, lat)
    s = m["_series"]
    both = {"single_node": single, "two_node": public(m)}
    print(f"{'metric':38s} {'1 node':>10} {'2 nodes':>10}")
    for k in single:
        print(f"{k:38s} {single[k]!s:>10} {both['two_node'][k]!s:>10}")
    (OUT / "demo_metrics.json").write_text(json.dumps(both, indent=2))

    t = s["t"]
    doa_t = np.array([r.t for r in results if r.doa])
    doa_az = np.array([r.doa.az_deg for r in results if r.doa])
    doa_el = np.array([r.doa.el_deg for r in results if r.doa])
    trk = [r.cue for r in results]
    taz = np.array([c.az_deg for c in trk])
    tel = np.array([c.el_deg for c in trk])
    half_cone = s["cone"]

    fig = plt.figure(figsize=(14, 11), constrained_layout=True)
    gs = fig.add_gridspec(4, 2, height_ratios=[1.1, 1, 1, 0.35])
    fig.suptitle("Huginn — passive acoustic FPV cueing, two nodes (50 m baseline), inbound 30 m/s past a generator",
                 fontsize=13)

    ax = fig.add_subplot(gs[0, 0])
    f, ts, S = spectrogram(x[0], fs=scene.fs, nperseg=1024, noverlap=512)
    keep = f <= 3000
    ax.pcolormesh(ts, f[keep], 10 * np.log10(S[keep] + 1e-20), shading="auto", cmap="magma")
    det_t = np.array([r.t for r in results if r.detection.detected])
    det_f0 = np.array([r.detection.f0_hz for r in results if r.detection.detected])
    for k in (1, 2, 3, 4):
        ax.plot(det_t, k * det_f0, ".", ms=2, color="cyan", alpha=0.6, label="detected comb" if k == 1 else None)
    ax.set(title="Mic 0 spectrogram: rotor harmonics emerge from wind + 60 Hz hum",
           xlabel="time (s)", ylabel="Hz")
    ax.legend(loc="upper left", fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    # SRP-PHAT map at the last detected frame with good SNR.
    pick = max((r for r in results if r.doa), key=lambda r: r.detection.score)
    i0 = int(round(pick.t * scene.fs - pipe.frame / 2))
    X = np.fft.rfft(x[:, i0:i0 + pipe.frame] * pipe.window, axis=-1)
    Y = X[:, pick.detection.bin_mask]
    Y = Y / np.abs(Y)
    az, el = np.meshgrid(np.arange(0, 360, 1.0), np.arange(-10, 86, 1.0))
    p = pipe.doa._srp(Y, pipe.freqs[pick.detection.bin_mask], pipe.doa._delays(az.ravel(), el.ravel()))
    ax.pcolormesh(az, el, p.reshape(az.shape), shading="auto", cmap="viridis")
    gaz, gel = g.az_el(gen_pos)
    ax.plot(pick.doa.az_deg, pick.doa.el_deg, "r+", ms=14, mew=2, label="estimate")
    ax.plot(gaz, gel, "wx", ms=10, mew=2, label="generator (masked out)")
    ax.set(title=f"Harmonic-masked SRP-PHAT at t={pick.t:.2f}s, R={s['range'][results.index(pick)]:.0f} m",
           xlabel="azimuth (deg)", ylabel="elevation (deg)")
    ax.legend(loc="upper right", fontsize=8)

    for row, (name, raw, tr, ac, ph) in enumerate([
            ("azimuth", doa_az, taz, s["acoustic_az"], s["phys_az"]),
            ("elevation", doa_el, tel, s["acoustic_el"], s["phys_el"])], start=1):
        ax = fig.add_subplot(gs[row, 0])
        ax.plot(t, ac, "k-", lw=1, label="acoustic truth (where it was)")
        ax.plot(t, ph, "k--", lw=1, label="physical truth at t+latency")
        ax.plot(doa_t, raw, ".", ms=3, color="tab:blue", alpha=0.6, label="raw SRP-PHAT")
        ax.plot(t, tr, "-", color="tab:red", lw=1.5, label="track (lead-compensated cue)")
        ax.fill_between(t, tr - half_cone, tr + half_cone, color="tab:red", alpha=0.15, label="handoff cone")
        ax.set(ylabel=f"{name} (deg)", xlabel="time (s)")
        lo, hi = np.nanmin(ac[s["det"]]) if s["det"].any() else 0, np.nanmax(ac[s["det"]]) if s["det"].any() else 1
        pad = max(5.0, 0.25 * (hi - lo))
        ax.set_ylim(lo - pad, hi + pad)
        if row == 1:
            ax.legend(fontsize=7, loc="best")

    ax = fig.add_subplot(gs[1:3, 1])
    r = s["range"]
    ax.semilogy(r, s["raw_err"], ".", ms=3, color="tab:blue", label="raw bearing error vs acoustic truth")
    ax.semilogy(r, s["trk_err"], "-", color="tab:red", lw=1.2, label="cue error vs physical truth")
    ax.semilogy(r, half_cone, ":", color="tab:red", lw=1.2, label="cue half-cone (2 sigma + lag residual)")
    ax.axhline(pipe.tracker.cfg.handoff_fov_deg / 2, color="gray", lw=0.8, ls="--",
               label="handoff threshold (cone < FOV/2)")
    ax.invert_xaxis()
    ax.set(xlabel="range (m) — drone closing →", ylabel="degrees", title="Accuracy vs range")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    ax = fig.add_subplot(gs[3, :])
    for y, (mask, c, lbl) in enumerate([(s["det"], "tab:blue", "detected"), (s["conf"], "tab:orange", "confirmed"),
                                        (s["ready"], "tab:green", "handoff ready")]):
        ax.fill_between(t, y, y + 0.8, where=mask, color=c, step="mid", label=lbl)
    ax.set(yticks=[0.4, 1.4, 2.4], yticklabels=["detected", "confirmed", "handoff"], xlabel="time (s)")
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ticks = np.linspace(t[0], t[-1], 8)
    ax2.set_xticks(ticks, [f"{v:.0f} m" for v in np.interp(ticks, t, r)], fontsize=8)

    fig.savefig(OUT / "demo.png", dpi=130)
    print(f"wrote {OUT / 'demo.png'}")


if __name__ == "__main__":
    main()
