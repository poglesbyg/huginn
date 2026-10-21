import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from huginn import geometry as g, scenario as sc  # noqa: E402
from huginn.fusion import TwoNodeSystem, triangulate  # noqa: E402
from huginn.tracker import BearingTracker, TrackState  # noqa: E402
from huginn.doa import DoaEstimate  # noqa: E402

PA, PB = np.zeros(3), np.array([40.0, -30.0, 0.0])


def bearing(p_from, target):
    v = target - p_from
    return v / np.linalg.norm(v)


def test_exact_bearings_recover_range():
    target = np.array([120.0, 90.0, 30.0])
    fix = triangulate(PA, bearing(PA, target), 0.5, PB, bearing(PB, target), 0.5)
    assert fix is not None
    assert abs(fix.range_m - np.linalg.norm(target)) < 0.01
    assert fix.miss_m < 1e-6


def test_rejects_parallel_and_diverging_rays():
    u = g.unit_vector(30.0, 10.0)
    assert triangulate(PA, u, 0.5, PB, u, 0.5) is None  # parallel: no crossing
    target = np.array([100.0, 50.0, 20.0])
    assert triangulate(PA, bearing(PA, target), 0.5, PB, -bearing(PB, target), 0.5) is None  # behind B


def test_rejects_rays_that_miss_badly():
    a, b = np.array([100.0, 50.0, 20.0]), np.array([100.0, 50.0, 80.0])  # two different targets
    assert triangulate(PA, bearing(PA, a), 0.3, PB, bearing(PB, b), 0.3) is None


def test_sigma_is_calibrated_under_bearing_noise():
    rng = np.random.default_rng(0)
    target = np.array([150.0, 120.0, 40.0])
    r_true = np.linalg.norm(target)
    z = []
    for _ in range(400):
        ua = g.unit_vector(*[a + rng.normal(0, 0.5) for a in g.az_el(bearing(PA, target))])
        ub = g.unit_vector(*[a + rng.normal(0, 0.5) for a in g.az_el(bearing(PB, target))])
        fix = triangulate(PA, ua, 0.5, PB, ub, 0.5)
        if fix:
            z.append((fix.range_m - r_true) / fix.sigma_m)
    # Sigma includes conservative timing/model terms, so it should not under-cover.
    assert len(z) > 300 and np.sqrt(np.mean(np.square(z))) < 1.3


def test_tracker_fuses_tighter_range_from_triangulation():
    trk = BearingTracker()
    for i in range(15):
        trk.update(i * 0.032, DoaEstimate(40.0, 10.0, 0.9, 0.5, 20), level_db=40.0,
                   range_obs=(100.0, 8.0) if i >= 5 else None)
    log_r, sig = trk.range_estimate()
    assert trk.state == TrackState.CONFIRMED
    assert abs(10**log_r - 100.0) < 20.0
    assert sig < trk.cfg.source_level_sigma_db / 20 / 2  # much tighter than level alone


def test_two_node_system_end_to_end():
    ma = g.uca_with_apex()
    mb = ma + PB
    target = np.array([70.0, 60.0, 25.0])
    scene = sc.Scene(np.vstack([ma, mb]), duration=1.5, seed=4,
                     sources=[sc.fpv_quad(sc.static(target, 1.5))])
    _, x = sc.render(scene)
    frames = TwoNodeSystem(ma, mb, scene.fs).run(x[:9], x[9:])
    fixes = [f.fix.range_m for f in frames if f.fix]
    assert len(fixes) > 20
    assert abs(np.median(fixes) / np.linalg.norm(target - ma.mean(axis=0)) - 1) < 0.05
