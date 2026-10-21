"""Fit the two empirical constants the pipeline relies on, from simulated truth:

  * SrpPhat.sigma_scale      - so predicted bearing sigma matches observed error
  * TrackerConfig.source_level_db / _sigma_db - the level->range prior

    python scripts/calibrate.py [n_runs]
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from huginn import geometry as g, scenario as sc  # noqa: E402
from huginn.pipeline import Pipeline  # noqa: E402

RAYLEIGH_MEDIAN = 1.1774  # median of a 2-D isotropic Gaussian's radial error, in sigmas


def main(n_runs: int = 6):
    mics = g.uca_with_apex()
    rng = np.random.default_rng(123)
    errs, sigmas, sl = [], [], []
    for i in range(n_runs):
        traj = sc.inbound_fpv(az_deg=rng.uniform(0, 360), alt=rng.uniform(15, 60),
                              weave_amp=rng.uniform(0, 30), speed=rng.uniform(20, 40))
        # Each airframe gets its own true source level: the prior has to absorb that spread.
        drone = sc.fpv_quad(traj, rotor_hz=rng.uniform(220, 300), level_db_1m=rng.normal(86, 3))
        scene = sc.Scene(mics, duration=traj.duration, seed=100 + i,
                         sources=[drone, sc.generator([-15, -12, 0.5], traj.duration)])
        _, x = sc.render(scene)
        res = Pipeline(mics, scene.fs).run(x)
        t = np.array([r.t for r in res])
        ac, _ = sc.acoustic_truth(traj, t)
        aaz, ael = g.az_el(ac)
        r_ac = np.linalg.norm(ac, axis=1)
        for j, r in enumerate(res):
            if r.doa:
                errs.append(g.angular_separation_deg(r.doa.az_deg, r.doa.el_deg, aaz[j], ael[j]))
                sigmas.append(r.doa.sigma_deg)
                sl.append(r.level_db + 20 * np.log10(r_ac[j]))
        print(f"run {i}: {sum(r.doa is not None for r in res)} fixes")

    e, s, sl = map(np.array, (errs, sigmas, sl))
    inl = e < 5.0
    ratio = np.median(e[inl] / s[inl]) / RAYLEIGH_MEDIAN
    print(f"\nfixes: {e.size}, gross (>5 deg): {100 * np.mean(~inl):.1f}%")
    print(f"sigma calibration ratio (want ~1.0): {ratio:.2f}  -> multiply sigma_scale by this")
    for lo, hi in [(0, 0.5), (0.5, 1), (1, 2), (2, 15)]:
        m = inl & (s >= lo) & (s < hi)
        if m.any():
            print(f"  predicted sigma {lo:>4}-{hi:<4}: n={m.sum():4d}  observed sigma ~ {np.median(e[m]) / RAYLEIGH_MEDIAN:.2f}")
    print(f"effective source level: median {np.median(sl):.1f} dB, robust sigma {1.4826 * np.median(np.abs(sl - np.median(sl))):.1f} dB")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
