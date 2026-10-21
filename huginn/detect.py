"""Harmonic-comb detector for multirotor propulsion noise.

Rotor noise is a stack of narrow lines at multiples of the shaft rate. We
whiten the array-averaged spectrum against a running noise floor, then score
every candidate fundamental by how many of its harmonics stand out. This
rejects broadband wind and low-order mains hum (generators, vehicles) that a
plain energy detector would trip on.
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter1d, median_filter

from .scenario import P_REF


@dataclass
class Detection:
    detected: bool
    f0_hz: float
    hits: int
    score: float
    bin_mask: np.ndarray   # bool (F,), bins to hand to the beamformer
    snr_db: np.ndarray     # whitened spectrum, for diagnostics


@dataclass
class HarmonicDetector:
    freqs: np.ndarray
    f0_range: tuple[float, float] = (180.0, 700.0)
    n_harmonics: int = 8
    hit_db: float = 3.0
    min_hits: int = 5
    band_hz: tuple[float, float] = (300.0, 4000.0)
    rpm_spread: float = 0.025  # rotors are a few % apart, so allow wider windows up the series
    floor_bins: int = 41

    def __post_init__(self):
        df = self.freqs[1] - self.freqs[0]
        self._df = df
        self._f0 = np.arange(*self.f0_range, 1.0)
        k = np.arange(1, self.n_harmonics + 1)
        self._idx = np.clip(np.rint(np.outer(self._f0, k) / df).astype(int), 0, len(self.freqs) - 1)
        self._tol = np.maximum(1, np.ceil(self.rpm_spread * np.outer(self._f0, k) / df)).astype(int)

    def whiten(self, X: np.ndarray) -> np.ndarray:
        """X is (M, F) or (N, M, F); power is averaged over every leading axis."""
        power = np.mean(np.abs(X) ** 2, axis=tuple(range(X.ndim - 1)))
        floor = median_filter(power, size=self.floor_bins, mode="nearest") + 1e-30
        return 10 * np.log10(power / floor + 1e-12)

    def __call__(self, X: np.ndarray) -> Detection:
        snr = self.whiten(X)
        snr_c = np.clip(snr, 0.0, 30.0)

        # Peak value within each harmonic's tolerance window.
        peaks = np.zeros(self._idx.shape)
        for tol in np.unique(self._tol):
            mx = maximum_filter1d(snr_c, size=2 * tol + 1, mode="nearest")
            sel = self._tol == tol
            peaks[sel] = mx[self._idx[sel]]

        hits = np.sum(peaks >= self.hit_db, axis=1)
        score = np.sum(peaks, axis=1) / self.n_harmonics
        best = int(np.argmax(hits * 100 + score))
        f0, h = float(self._f0[best]), int(hits[best])
        detected = h >= self.min_hits
        mask = self._harmonic_mask(f0, snr) if detected else np.zeros_like(snr, dtype=bool)
        return Detection(detected and mask.sum() >= 6, f0, h, float(score[best]), mask, snr)

    def _harmonic_mask(self, f0: float, snr: np.ndarray) -> np.ndarray:
        lo, hi = self.band_hz
        near = np.zeros_like(snr, dtype=bool)
        for k in range(1, int(hi // f0) + 1):
            fk = k * f0
            if fk < lo:
                continue
            tol = max(1.0, self.rpm_spread * fk / self._df)
            near |= np.abs(self.freqs - fk) <= tol * self._df
        return near & (snr >= self.hit_db)


def tonal_level_db(X: np.ndarray, mask: np.ndarray, window: np.ndarray) -> float:
    """SPL (dB re 20 uPa) of the energy in the masked bins, averaged over mics.

    By Parseval, a windowed sinusoid's mean-square pressure is
    2 * sum_k |X_k|^2 / (N * sum(w^2)) over its one-sided bins.
    """
    if not mask.any():
        return float("nan")
    X = X if X.ndim == 3 else X[None]
    e = np.mean(np.sum(np.abs(X[..., mask]) ** 2, axis=-1))
    ms = 2.0 * e / (window.size * np.sum(window**2))
    return float(10 * np.log10(ms / P_REF**2 + 1e-30))
