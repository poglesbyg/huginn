"""Replay a real (or simulated) recording through Huginn.

    # multichannel array recording: detect, localise, track
    python scripts/replay.py rec.wav --geometry rec.geometry.json [--swarm] [--truth rec.truth.csv]

    # phone / single mic: detection only (is there a drone? what pitch? how strong?)
    python scripts/replay.py phone.wav

Writes out/replay_<name>.png (spectrogram, detections, bearings) and
out/replay_<name>.csv (per-frame telemetry: detections or cues).

Truth CSV (optional): columns t_s, az_deg, el_deg in the array frame, e.g. from a
flight log's GPS converted to bearings from the array.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from huginn import geometry as g  # noqa: E402
from huginn.detect import HarmonicDetector, fold_f0  # noqa: E402
from huginn.pipeline import Pipeline  # noqa: E402
from huginn.recording import Geometry, load_wav  # noqa: E402
from huginn.swarm import SwarmSystem  # noqa: E402
from huginn.tracker import TrackerConfig, TrackState  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "out"
FRAME, HOP = 1024, 512


def array_report(mics: np.ndarray) -> dict:
    d = np.linalg.norm(mics[:, None] - mics[None], axis=-1)
    iu = np.triu_indices(len(mics), 1)
    aperture, spacing = d[iu].max(), d[iu].min()
    planar = np.ptp(mics[:, 2]) < 1e-3
    return {
        "mics": len(mics),
        "aperture_m": round(float(aperture), 3),
        "min_spacing_m": round(float(spacing), 3),
        "spatial_alias_above_hz": round(float(g.SPEED_OF_SOUND / (2 * spacing))),
        "beamwidth_at_1khz_deg": round(float(np.degrees(g.SPEED_OF_SOUND / (1000 * aperture))), 1),
        "planar": bool(planar),
    }


def detection_only(x, fs):
    freqs = np.fft.rfftfreq(FRAME, 1 / fs)
    det = HarmonicDetector(freqs, n_channels=x.shape[0])
    w = np.hanning(FRAME)
    rows = []
    for s in range(0, x.shape[1] - FRAME + 1, HOP):
        X = np.fft.rfft(x[:, s:s + FRAME] * w, axis=-1)
        found = det.detect_all(X)
        t = (s + FRAME / 2) / fs
        rows.append({"t_s": round(t, 3), "n_combs": len(found),
                     "pitches_hz": ";".join(f"{fold_f0(d.f0_hz):.0f}" for d in found),
                     "best_hits": max((d.hits for d in found), default=0),
                     "best_score_db": round(max((d.score for d in found), default=0.0), 2)})
    return rows


def run(args):
    name = Path(args.wav).stem
    geom = Geometry.load(args.geometry) if args.geometry else None
    x, fs = load_wav(args.wav, geom, args.start, args.end)
    multichannel = geom is not None and len(geom.mics) >= 3
    if not multichannel and x.shape[0] > 1:
        print("no geometry (or < 3 mics): detection only, channels averaged in power")
    summary = {"file": str(args.wav), "seconds": round(x.shape[1] / fs, 2), "channels_used": x.shape[0]}
    truth = None
    if args.truth:
        tr = np.genfromtxt(args.truth, delimiter=",", names=True)
        truth = (tr["t_s"], tr["az_deg"], tr["el_deg"])

    if not multichannel:
        rows = detection_only(x, fs)
        det = np.array([r["n_combs"] > 0 for r in rows])
        summary.update({"mode": "detection-only", "detection_rate": round(float(det.mean()), 3)})
        bearings = None
    else:
        summary["array"] = array_report(geom.mics)
        cfg = TrackerConfig(use_level_range=geom.calibrated)
        rows, bearings = [], []
        if args.swarm:
            frames = SwarmSystem(geom.mics, fs, tracker_cfg=cfg).run(x)
            for f in frames:
                for m in f.meas:
                    bearings.append((f.t, m.doa.az_deg, m.doa.el_deg, -1))
                for c in f.cues:
                    if c.state == TrackState.CONFIRMED:
                        rows.append(cue_row(f.t, c, geom))
            n_conf = len({r["track_id"] for r in rows})
            summary.update({"mode": "swarm", "confirmed_tracks": n_conf})
        else:
            pipe = Pipeline(geom.mics, fs, tracker_cfg=cfg)
            for r in pipe.run(x):
                if r.doa:
                    bearings.append((r.t, r.doa.az_deg, r.doa.el_deg, 0))
                row = cue_row(r.t, r.cue, geom)
                row["detected"] = int(r.detection.detected)
                row["pitch_hz"] = round(fold_f0(r.detection.f0_hz), 1) if r.detection.detected else ""
                rows.append(row)
            conf = np.array([row["state"] == "confirmed" for row in rows])
            summary.update({"mode": "single-target",
                            "detection_rate": round(float(np.mean([row["detected"] for row in rows])), 3),
                            "confirmed_rate": round(float(conf.mean()), 3)})
        if truth is not None and bearings:
            b = np.array([bb[:3] for bb in bearings])
            taz = np.interp(b[:, 0], truth[0], np.unwrap(truth[1], period=360)) % 360
            tel = np.interp(b[:, 0], truth[0], truth[2])
            err = g.angular_separation_deg(b[:, 1], b[:, 2], taz, tel)
            summary["bearing_err_vs_truth_deg"] = {"median": round(float(np.median(err)), 2),
                                                   "p90": round(float(np.percentile(err, 90)), 2),
                                                   "n": int(err.size)}
        if not geom.calibrated:
            summary["note"] = "no pa_per_full_scale: level-based range disabled; levels are relative"

    out = Path(args.out)
    out.mkdir(exist_ok=True, parents=True)
    with open(out / f"replay_{name}.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["t_s"])
        wr.writeheader()
        wr.writerows(rows)
    plot(x, fs, name, rows, bearings, truth, multichannel, out)
    print(json.dumps(summary, indent=2))
    (out / f"replay_{name}.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}/replay_{name}.png, .csv, .json")
    return summary


def cue_row(t, c, geom):
    tracked = c.state != TrackState.NONE
    return {"t_s": round(t, 3), "state": c.state.value, "track_id": c.track_id,
            "az_deg": round(c.az_deg, 2) if tracked else "", "el_deg": round(c.el_deg, 2) if tracked else "",
            "compass_deg": round(float(geom.compass(c.az_deg)), 1)
            if tracked and geom.heading_deg is not None else "",
            "lead_az_deg": round(c.lead_az_deg, 2) if tracked else "",
            "lead_el_deg": round(c.lead_el_deg, 2) if tracked else "",
            "cone_deg": round(c.cone_deg, 2) if tracked else "",
            "range_est_m": round(c.range_est_m, 1) if tracked and np.isfinite(c.range_est_m) else "",
            "handoff_ready": int(c.handoff_ready)}


def plot(x, fs, name, rows, bearings, truth, multichannel, out):
    n = 3 if multichannel else 2
    fig, axes = plt.subplots(n, 1, figsize=(13, 3.2 * n), constrained_layout=True, sharex=True)
    ax = axes[0]
    f, ts, S = spectrogram(x[0], fs=fs, nperseg=FRAME, noverlap=FRAME - HOP)
    keep = f <= 3000
    ax.pcolormesh(ts, f[keep], 10 * np.log10(S[keep] + 1e-20), shading="auto", cmap="magma")
    ax.set(title=f"{name}: channel 0 spectrogram", ylabel="Hz")
    if not multichannel:
        for r in rows:
            for p in filter(None, r["pitches_hz"].split(";")):
                for k in (1, 2, 4):
                    ax.plot(r["t_s"], k * float(p), ".", ms=2, color="cyan")
        ax = axes[1]
        ax.plot([r["t_s"] for r in rows], [r["best_hits"] for r in rows], "-", lw=1, label="harmonics hit (of 8)")
        ax.plot([r["t_s"] for r in rows], [r["n_combs"] for r in rows], "-", lw=1, label="combs found")
        ax.axhline(5, color="gray", ls="--", lw=0.8, label="detection threshold")
        ax.set(xlabel="time (s)", title="Detection")
        ax.legend(fontsize=8)
    else:
        b = np.array(bearings) if bearings else np.zeros((0, 4))
        for i, (col, lab) in enumerate([(1, "azimuth (deg, array frame)"), (2, "elevation (deg)")]):
            ax = axes[1 + i]
            ax.plot(b[:, 0], b[:, col], ".", ms=3, color="tab:blue", alpha=0.5, label="raw fixes")
            tk = {}
            for r in rows:
                if r["state"] == "confirmed":
                    tk.setdefault(r["track_id"], []).append((r["t_s"], r["az_deg"] if col == 1 else r["el_deg"]))
            for tid, pts in tk.items():
                p = np.array(pts, dtype=float)
                lab = "track" if tid < 0 else f"track {tid}"
                ax.plot(p[:, 0], p[:, 1], ".", ms=4, label=lab if len(tk) <= 6 else None)
            if truth is not None:
                ax.plot(truth[0], truth[col], "k-", lw=1, label="truth")
            ax.set(ylabel=lab)
            ax.legend(fontsize=7, loc="best")
        axes[-1].set(xlabel="time (s)")
    axes[0].set_xlim(0, x.shape[1] / fs)
    fig.savefig(out / f"replay_{name}.png", dpi=110)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav")
    ap.add_argument("--geometry", help="mic-geometry JSON (needed for bearings)")
    ap.add_argument("--swarm", action="store_true", help="multi-target mode")
    ap.add_argument("--truth", help="CSV with t_s, az_deg, el_deg")
    ap.add_argument("--start", type=float, default=0.0, help="seconds to skip")
    ap.add_argument("--end", type=float, default=None, help="stop at this time (s)")
    ap.add_argument("--out", default=str(OUT), help="output directory (default: out/)")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
