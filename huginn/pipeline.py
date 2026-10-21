"""Frame-by-frame sensor pipeline: detect -> localise -> track -> cue."""

import time
from dataclasses import dataclass

import numpy as np

from .detect import Detection, HarmonicDetector, tonal_level_db
from .doa import DoaEstimate, SrpPhat
from .tracker import BearingTracker, Cue, TrackerConfig


@dataclass
class FrameResult:
    t: float                    # frame centre time (s)
    detection: Detection
    doa: DoaEstimate | None
    cue: Cue
    level_db: float | None
    compute_ms: float


class Pipeline:
    def __init__(self, mics: np.ndarray, fs: int, frame: int = 1024, hop: int = 512,
                 integrate: int = 1, tracker_cfg: TrackerConfig | None = None):
        self.fs, self.frame, self.hop = fs, frame, hop
        self.integrate = integrate
        self._recent: list[np.ndarray] = []
        self.window = np.hanning(frame).astype(np.float64)
        self.freqs = np.fft.rfftfreq(frame, 1.0 / fs)
        self.detector = HarmonicDetector(self.freqs)
        self.doa = SrpPhat(mics, self.freqs)
        self.tracker = BearingTracker(tracker_cfg or TrackerConfig())

    def measure(self, block: np.ndarray):
        """Detection + direction + level for one frame, without touching the tracker."""
        X = np.fft.rfft(block * self.window, axis=-1)
        # Integrate over the last few frames (~130 ms): incoherent power averaging for the
        # detector, and stacking PHAT-weighted snapshots for the beamformer.
        self._recent = (self._recent + [X])[-self.integrate:]
        stack = np.stack(self._recent)  # (N, M, F)
        det = self.detector(stack)
        doa = self.doa(stack, det.bin_mask, det.snr_db) if det.detected else None
        level = tonal_level_db(stack, det.bin_mask, self.window) if det.detected else None
        return det, doa, level

    def step(self, block: np.ndarray, t_centre: float, range_obs=None) -> FrameResult:
        t0 = time.perf_counter()
        det, doa, level = self.measure(block)
        cue = self.tracker.update(t_centre, doa, level, range_obs)
        return FrameResult(t_centre, det, doa, cue, level, (time.perf_counter() - t0) * 1e3)

    def run(self, signals: np.ndarray) -> list[FrameResult]:
        n = signals.shape[1]
        out = []
        for start in range(0, n - self.frame + 1, self.hop):
            t_c = (start + self.frame / 2) / self.fs
            out.append(self.step(signals[:, start:start + self.frame], t_c))
        return out
