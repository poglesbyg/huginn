import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from huginn import geometry as g, scenario as sc  # noqa: E402
from huginn.detect import HarmonicDetector, fold_f0  # noqa: E402
from huginn.doa import DoaEstimate  # noqa: E402
from huginn.swarm import Measurement, MultiTargetTracker, SwarmSystem  # noqa: E402
from huginn.tracker import TrackState  # noqa: E402

FS = 16000
MICS = g.uca_with_apex()


def hover(dirs_pitch, duration=2.5, rng_m=70.0, seed=0):
    srcs = [sc.fpv_quad(sc.static(rng_m * g.unit_vector(az, el), duration), rotor_hz=f) for az, el, f in dirs_pitch]
    _, x = sc.render(sc.Scene(MICS, duration=duration, sources=srcs, seed=seed))
    return x


def confirmed(frame):
    return [c for c in frame.cues if c.state == TrackState.CONFIRMED]


def test_fold_f0_maps_shaft_and_blade_pass_together():
    assert fold_f0(260.0) == fold_f0(520.0) == fold_f0(1040.0) == 260.0
    assert np.isnan(fold_f0(float("nan")))


def test_detect_all_finds_two_airframes():
    x = hover([(30, 10, 240), (200, 15, 300)])
    freqs = np.fft.rfftfreq(1024, 1 / FS)
    det = HarmonicDetector(freqs)
    X = np.fft.rfft(x[:, 8192:9216] * np.hanning(1024), axis=-1)
    pitches = sorted(fold_f0(d.f0_hz) for d in det.detect_all(X))
    assert any(abs(p / 240 - 1) < 0.06 for p in pitches)
    assert any(abs(p / 300 - 1) < 0.06 for p in pitches)


def test_separates_well_spaced_pair_into_two_tracks():
    truth = [(30, 10), (150, 20)]
    frames = SwarmSystem(MICS, FS).run(hover([(30, 10, 240), (150, 20, 300)]))
    late = [f for f in frames if f.t > 1.0]
    both = 0
    for f in late:
        cs = confirmed(f)
        hit = [any(g.angular_separation_deg(c.az_deg, c.el_deg, *tr) < 3 for c in cs) for tr in truth]
        both += all(hit)
    assert both / len(late) > 0.8


def test_spectral_separation_inside_one_beamwidth():
    """10 degrees apart is far inside the array's beam; pitch is what separates them."""
    frames = SwarmSystem(MICS, FS).run(hover([(60, 10, 230), (70, 10, 290)], seed=3))
    late = [f for f in frames if f.t > 1.0]
    two = np.mean([len(confirmed(f)) >= 2 for f in late])
    assert two > 0.6


def test_single_drone_gives_single_track():
    frames = SwarmSystem(MICS, FS).run(hover([(100, 15, 270)]))
    late = [f for f in frames if f.t > 1.0]
    assert np.mean([len(confirmed(f)) == 1 for f in late]) > 0.9
    ids = {c.track_id for f in late for c in confirmed(f)}
    assert len(ids) == 1


def test_mtt_keeps_identities_through_crossing_with_pitch():
    """Two bearings cross; pitch fingerprints keep identities from swapping."""
    mtt = MultiTargetTracker()
    ids_at_start = None
    for k in range(60):
        t = k * 0.032
        a = 40 + 20 * t      # sweeping up
        b = 80 - 20 * t      # sweeping down: they cross around t = 1 s
        meas = [Measurement(DoaEstimate(a, 10.0, 0.9, 0.4, 20), 50.0, 240.0, np.zeros(1, bool)),
                Measurement(DoaEstimate(b, 10.0, 0.9, 0.4, 20), 50.0, 300.0, np.zeros(1, bool))]
        cues = [c for c in mtt.update(t, meas) if c.state == TrackState.CONFIRMED]
        if k == 10:
            ids_at_start = {round(c.f0_hz): c.track_id for c in cues}
    end = {round(c.f0_hz / 10) * 10: c.track_id for c in cues}
    start = {round(f / 10) * 10: i for f, i in ids_at_start.items()}
    assert start == end


def test_raid_metrics_run_end_to_end():
    from huginn.metrics import score_swarm

    rng = np.random.default_rng(5)
    raid = sc.swarm_raid(2, rng)
    dur = min(max(r.t_end for r in raid), 6.0)
    _, x = sc.render(sc.Scene(MICS, duration=dur, sources=[r.source() for r in raid], seed=5))
    m = score_swarm(SwarmSystem(MICS, FS).run(x), raid, 0.08, eval_range_m=400.0)
    assert m["n_drones"] == 2 and 0.0 <= m["coverage"] <= 1.0
