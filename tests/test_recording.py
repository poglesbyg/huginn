import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from huginn import geometry as g, scenario as sc  # noqa: E402
from huginn.detect import HarmonicDetector, hit_threshold_db  # noqa: E402
from huginn.recording import Geometry, circular, load_wav, write_wav  # noqa: E402


def _replay():
    spec = importlib.util.spec_from_file_location("replay", ROOT / "scripts" / "replay.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_geometry_roundtrip_and_validation(tmp_path):
    geo = Geometry(circular(6, 0.05, centre_mic=True), channels=[1, 2, 3, 4, 5, 6, 0], heading_deg=90.0,
                   pa_per_full_scale=2.5, note="test")
    geo.save(tmp_path / "g.json")
    back = Geometry.load(tmp_path / "g.json")
    assert np.allclose(back.mics, geo.mics, atol=1e-5)
    assert back.channels == geo.channels and back.heading_deg == 90.0 and back.calibrated
    (tmp_path / "bad.json").write_text('{"mics_m": [[0,0,0],[1,0,0]], "channels": [0]}')
    with pytest.raises(ValueError):
        Geometry.load(tmp_path / "bad.json")


def test_compass_bearing_from_heading():
    geo = Geometry(np.zeros((3, 3)), heading_deg=90.0)  # array +x points east
    # +x (az 0) is east = 90; +y (az 90, CCW) is north = 0
    assert np.allclose(geo.compass([0.0, 90.0, 180.0]), [90.0, 0.0, 270.0])


def test_load_wav_maps_channels_scales_and_resamples(tmp_path):
    fs = 48000
    t = np.arange(fs) / fs
    sig = np.stack([np.sin(2 * np.pi * 300 * t) * (k + 1) for k in range(3)])  # channel k has amplitude k+1
    pa_fs = write_wav(tmp_path / "r.wav", sig, fs)
    geo = Geometry(np.eye(3), channels=[2, 0, 1], pa_per_full_scale=pa_fs)
    x, fs_out = load_wav(tmp_path / "r.wav", geo)
    assert fs_out == 16000 and x.shape == (3, 16000)
    amp = np.std(x, axis=1) * np.sqrt(2)
    assert np.allclose(amp, [3.0, 1.0, 2.0], rtol=0.01)  # reordered and back in pascals


@pytest.mark.parametrize("dtype", [np.float32, np.int32])
def test_load_wav_float_and_32bit(tmp_path, dtype):
    x = (0.25 * np.sin(np.linspace(0, 200, 16000)))[:, None].repeat(2, axis=1)
    data = x.astype(np.float32) if dtype == np.float32 else (x * 2**31).astype(np.int32)
    wavfile.write(tmp_path / "f.wav", 16000, data)
    y, _ = load_wav(tmp_path / "f.wav")
    assert y.shape == (2, 16000) and abs(np.max(np.abs(y)) - 0.25) < 0.01


def test_threshold_tracks_channel_count():
    assert hit_threshold_db(9) == pytest.approx(3.0)
    vals = [hit_threshold_db(m) for m in (1, 2, 4, 9, 16)]
    assert all(a > b for a, b in zip(vals, vals[1:])) and vals[0] > 7.5


def test_single_channel_noise_does_not_detect():
    _, x = sc.render(sc.Scene(g.uca_with_apex()[:1], duration=4.0, seed=8, wind_db=62,
                              sources=[sc.generator([-10, 5, 0.5], 4.0)]))
    freqs = np.fft.rfftfreq(1024, 1 / 16000)
    det = HarmonicDetector(freqs, n_channels=1)
    w = np.hanning(1024)
    hits = [bool(det.detect_all(np.fft.rfft(x[:, s:s + 1024] * w, axis=-1))) for s in range(0, x.shape[1] - 1024, 512)]
    assert np.mean(hits) < 0.02


def test_replay_end_to_end_matches_truth(tmp_path):
    mics = g.uca_with_apex()
    pos = 60.0 * g.unit_vector(120.0, 15.0)
    _, x = sc.render(sc.Scene(mics, fs=48000, duration=2.0, seed=4, sources=[sc.fpv_quad(sc.static(pos, 2.0))]))
    pa_fs = write_wav(tmp_path / "rec.wav", x, 48000)
    Geometry(mics, pa_per_full_scale=pa_fs).save(tmp_path / "rec.geometry.json")
    (tmp_path / "rec.truth.csv").write_text("t_s,az_deg,el_deg\n0,120,15\n2,120,15\n")
    summary = _replay().main([str(tmp_path / "rec.wav"), "--geometry", str(tmp_path / "rec.geometry.json"),
                              "--truth", str(tmp_path / "rec.truth.csv"), "--out", str(tmp_path)])
    assert summary["mode"] == "single-target" and summary["confirmed_rate"] > 0.8
    assert summary["bearing_err_vs_truth_deg"]["median"] < 1.0
    assert (tmp_path / "replay_rec.png").exists() and (tmp_path / "replay_rec.csv").exists()
