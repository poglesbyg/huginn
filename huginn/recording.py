"""Load real recordings for replay through the same pipeline the simulator feeds.

A recording is one multichannel WAV (all mics on ONE clock: a USB mic array or a
multichannel interface), or a single mono/stereo file for detection-only use.
Geometry lives in a small JSON file next to it:

    {
      "mics_m": [[x, y, z], ...],      # metres, one row per mic, array frame
      "channels": [0, 1, 2, ...],      # optional: WAV channel index for each mic row
      "heading_deg": 0.0,              # optional: compass bearing of the array's +x axis
      "pa_per_full_scale": 1.0,        # optional: pascals at digital full scale (enables
                                       # level-based range); omit if uncalibrated
      "note": "free text"              # optional
    }

Array frame: x forward, y left, z up. Bearings come out counter-clockwise from +x.
`heading_deg` converts them to compass bearings (clockwise from north) for the CSV.
"""

import json
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

PIPELINE_FS = 16000  # detector bins and calibration assume this rate


@dataclass
class Geometry:
    mics: np.ndarray                 # (M, 3) metres
    channels: list[int] | None = None
    heading_deg: float | None = None
    pa_per_full_scale: float | None = None
    note: str = ""

    @property
    def calibrated(self) -> bool:
        return self.pa_per_full_scale is not None

    def compass(self, az_deg):
        """Array-frame azimuth (CCW from +x) -> compass bearing (CW from north)."""
        if self.heading_deg is None:
            return np.full_like(np.asarray(az_deg, dtype=float), np.nan)
        return (self.heading_deg - np.asarray(az_deg)) % 360.0

    @classmethod
    def load(cls, path) -> "Geometry":
        d = json.loads(Path(path).read_text())
        mics = np.asarray(d["mics_m"], dtype=float)
        if mics.ndim != 2 or mics.shape[1] != 3:
            raise ValueError(f"{path}: mics_m must be a list of [x, y, z] triples")
        ch = d.get("channels")
        if ch is not None and len(ch) != len(mics):
            raise ValueError(f"{path}: channels has {len(ch)} entries for {len(mics)} mics")
        return cls(mics, ch, d.get("heading_deg"), d.get("pa_per_full_scale"), d.get("note", ""))

    def save(self, path):
        d = {"mics_m": np.round(self.mics, 5).tolist()}
        for k in ("channels", "heading_deg", "pa_per_full_scale"):
            if getattr(self, k) is not None:
                d[k] = getattr(self, k)
        if self.note:
            d["note"] = self.note
        Path(path).write_text(json.dumps(d, indent=2) + "\n")


def circular(n: int, radius_m: float, centre_mic: bool = False, first_mic_deg: float = 0.0) -> np.ndarray:
    """Mic positions for a planar circular array (most USB arrays). Measure your own
    board's radius and the angle of mic 0; vendor drawings differ between revisions."""
    a = np.radians(first_mic_deg) + 2 * np.pi * np.arange(n) / n
    ring = np.stack([radius_m * np.cos(a), radius_m * np.sin(a), np.zeros(n)], axis=1)
    return np.vstack([ring, [0.0, 0.0, 0.0]]) if centre_mic else ring


def _to_float(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.int16:
        return x.astype(np.float64) / 32768.0
    if x.dtype == np.int32:  # 24-bit WAVs are read left-aligned into int32
        return x.astype(np.float64) / 2147483648.0
    if x.dtype == np.uint8:
        return (x.astype(np.float64) - 128.0) / 128.0
    return x.astype(np.float64)


def load_wav(path, geometry: Geometry | None = None, start_s: float = 0.0, end_s: float | None = None,
             fs_out: int = PIPELINE_FS) -> tuple[np.ndarray, int]:
    """Returns (signals (M, N), fs_out). Channels are picked/ordered by the geometry,
    scaled to pascals if calibrated (else full-scale units), and resampled to fs_out."""
    fs, x = wavfile.read(str(path))
    x = _to_float(x)
    x = x[:, None] if x.ndim == 1 else x  # (N, C)
    a, b = int(start_s * fs), int(end_s * fs) if end_s is not None else x.shape[0]
    x = x[a:b].T  # (C, N)

    if geometry is not None:
        ch = geometry.channels if geometry.channels is not None else list(range(len(geometry.mics)))
        if max(ch) >= x.shape[0]:
            raise ValueError(f"geometry wants channel {max(ch)} but {path} has {x.shape[0]} channels")
        x = x[ch]
        if geometry.calibrated:
            x = x * geometry.pa_per_full_scale

    x = x - x.mean(axis=1, keepdims=True)  # strip DC offset some interfaces add
    if fs != fs_out:
        g = gcd(fs, fs_out)
        x = resample_poly(x, fs_out // g, fs // g, axis=1)
    return x, fs_out


def write_wav(path, x: np.ndarray, fs: int, peak: float = 0.5):
    """Write (M, N) as int16 multichannel WAV, normalised to `peak` of full scale.
    Returns the pascals-per-full-scale factor (for a geometry file's calibration)."""
    scale = peak / (np.max(np.abs(x)) + 1e-30)
    wavfile.write(str(path), fs, np.clip(x.T * scale * 32767, -32768, 32767).astype(np.int16))
    return 1.0 / scale
