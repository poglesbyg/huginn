"""Write a simulated flyby as if it came off real hardware, to exercise the replay path.

    python scripts/make_sample_recording.py
    python scripts/replay.py samples/sim_flyby.wav --geometry samples/sim_flyby.geometry.json \
        --truth samples/sim_flyby.truth.csv
    python scripts/replay.py samples/sim_flyby_mono.wav

Produces a 48 kHz, 16-bit, 9-channel WAV (so resampling and int scaling get exercised),
its geometry JSON with calibration, a truth CSV, and a mono copy like a phone would give.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from huginn import geometry as g, scenario as sc  # noqa: E402
from huginn.recording import Geometry, write_wav  # noqa: E402

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


def main(fs: int = 48000, seed: int = 21):
    SAMPLES.mkdir(exist_ok=True)
    mics = g.uca_with_apex()
    traj = sc.inbound_fpv(start_range=250, az_deg=70, alt=30, speed=25, weave_amp=15)
    scene = sc.Scene(mics, fs=fs, duration=traj.duration, seed=seed,
                     sources=[sc.fpv_quad(traj, rotor_hz=255), sc.generator([-12, 8, 0.5], traj.duration)])
    _, x = sc.render(scene)

    pa_fs = write_wav(SAMPLES / "sim_flyby.wav", x, fs)
    write_wav(SAMPLES / "sim_flyby_mono.wav", x[:1], fs)
    Geometry(mics, channels=list(range(len(mics))), heading_deg=0.0, pa_per_full_scale=pa_fs,
             note="Huginn reference array: 8-mic ring r=0.25 m + apex mic 0.30 m up (simulated)"
             ).save(SAMPLES / "sim_flyby.geometry.json")

    t = np.arange(0, traj.duration, 0.05)
    ac, _ = sc.acoustic_truth(traj, t, origin=mics.mean(axis=0))
    az, el = g.az_el(ac)
    np.savetxt(SAMPLES / "sim_flyby.truth.csv", np.column_stack([t, az, el]), delimiter=",",
               header="t_s,az_deg,el_deg", comments="", fmt="%.4f")
    print(f"wrote {SAMPLES}/sim_flyby.wav (+ _mono.wav, .geometry.json, .truth.csv), {traj.duration:.1f} s")


if __name__ == "__main__":
    main()
