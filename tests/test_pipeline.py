import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from huginn import geometry as g, scenario as sc  # noqa: E402
from huginn.detect import HarmonicDetector, tonal_level_db  # noqa: E402
from huginn.doa import DoaEstimate, SrpPhat  # noqa: E402
from huginn.pipeline import Pipeline  # noqa: E402
from huginn.tracker import BearingTracker, TrackState  # noqa: E402

FS = 16000
MICS = g.uca_with_apex()


def hover_scene(az, el, rng_m, duration=1.0, seed=0, extra=()):
    pos = rng_m * g.unit_vector(az, el)
    drone = sc.fpv_quad(sc.static(pos, duration))
    return sc.Scene(MICS, duration=duration, sources=[drone, *extra], seed=seed)


def frames(x, n=1024):
    return [x[:, i:i + n] for i in range(0, x.shape[1] - n + 1, n // 2)]


@pytest.mark.parametrize("az,el", [(10, 5), (135, 30), (250, 60), (330, -5)])
def test_doa_recovers_static_source(az, el):
    _, x = sc.render(hover_scene(az, el, 40.0))
    p = Pipeline(MICS, FS)
    fixes = [r.doa for r in p.run(x) if r.doa]
    assert len(fixes) > 20
    errs = [g.angular_separation_deg(f.az_deg, f.el_deg, az, el) for f in fixes]
    assert np.median(errs) < 1.0


def test_generator_alone_is_rejected():
    duration = 10.0
    scene = sc.Scene(MICS, duration=duration, seed=3, wind_db=65,
                     sources=[sc.generator([-6, 4, 0.5], duration, level_db_1m=92)])
    _, x = sc.render(scene)
    res = Pipeline(MICS, FS).run(x)
    assert not any(r.cue.state == TrackState.CONFIRMED for r in res)


def test_drone_beats_louder_generator_in_beamformer():
    gen = sc.generator([-10, -10, 0.5], 1.0, level_db_1m=90)
    _, x = sc.render(hover_scene(60, 20, 60.0, extra=[gen]))
    fixes = [r.doa for r in Pipeline(MICS, FS).run(x) if r.doa]
    assert fixes and np.median([g.angular_separation_deg(f.az_deg, f.el_deg, 60, 20) for f in fixes]) < 1.5


def test_detector_finds_rotor_family():
    _, x = sc.render(hover_scene(0, 10, 30.0))
    freqs = np.fft.rfftfreq(1024, 1 / FS)
    det = HarmonicDetector(freqs)(np.fft.rfft(x[:, :1024] * np.hanning(1024), axis=-1))
    assert det.detected
    # Either the shaft rate or the blade-pass rate (2x) is an acceptable comb root.
    assert min(abs(det.f0_hz - 260), abs(det.f0_hz - 520)) < 30


def test_level_tracks_inverse_square_law():
    levels = []
    for r in (20.0, 80.0):
        _, x = sc.render(hover_scene(45, 10, r, seed=1))
        freqs = np.fft.rfftfreq(1024, 1 / FS)
        w = np.hanning(1024)
        X = np.fft.rfft(x[:, 4096:5120] * w, axis=-1)
        det = HarmonicDetector(freqs)(X)
        levels.append(tonal_level_db(X, det.bin_mask, w))
    # 4x range -> ~12 dB; allow slack for bins that drop below threshold at range.
    assert 9.0 < levels[0] - levels[1] < 16.0


def test_tracker_confirms_gates_outlier_and_drops():
    trk = BearingTracker()
    t = 0.0
    for i in range(10):
        t = i * 0.032
        cue = trk.update(t, DoaEstimate(100.0 + 5.0 * t, 10.0, 0.9, 0.5, 20), level_db=40.0)
    assert cue.state == TrackState.CONFIRMED
    az_before = cue.az_deg
    cue = trk.update(t + 0.032, DoaEstimate(280.0, 10.0, 0.5, 0.5, 20), level_db=40.0)  # mirror outlier
    assert abs(g.wrap_deg(cue.az_deg - az_before)) < 1.0
    for k in range(30):
        cue = trk.update(t + 0.064 + k * 0.032, None)
    assert cue.state == TrackState.NONE


def test_srp_handles_azimuth_wrap():
    freqs = np.fft.rfftfreq(1024, 1 / FS)
    srp = SrpPhat(MICS, freqs)
    u = g.unit_vector(359.5, 15.0)
    tau = MICS @ u / g.SPEED_OF_SOUND
    mask = (freqs > 400) & (freqs < 3000)
    X = np.exp(2j * np.pi * np.outer(tau, freqs))
    est = srp(X, mask, np.full(freqs.size, 20.0))
    assert g.angular_separation_deg(est.az_deg, est.el_deg, 359.5, 15.0) < 0.5
