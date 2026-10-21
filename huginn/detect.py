"""Harmonic-comb detector for multirotor propulsion noise.

Rotor noise is a stack of narrow lines at multiples of the shaft rate. We
whiten the array-averaged spectrum against a running noise floor, then score
every candidate fundamental by how many of its harmonics stand out. This
rejects broadband wind and low-order mains hum (generators, vehicles) that a
plain energy detector would trip on.

For swarms, `detect_all` peels combs off one at a time: find the strongest
comb, blank every bin near any multiple of its fundamental, search again.
Airframes differ in rotor rate by a few percent or more, so this separates
drones the array cannot separate spatially (its beam is tens of degrees wide).
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter1d, median_filter
from scipy.stats import gamma

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
    n_channels: int = 9        # spectra averaged per bin; sets hit_db unless given explicitly
    hit_db: float | None = None
    min_hits: int = 5
    band_hz: tuple[float, float] = (300.0, 4000.0)
    rpm_spread: float = 0.025  # rotors are a few % apart, so allow wider windows up the series
    floor_bins: int = 41

    def __post_init__(self):
        if self.hit_db is None:
            self.hit_db = hit_threshold_db(self.n_channels)
        df = self.freqs[1] - self.freqs[0]
        self._df = df
        self._f0 = np.arange(*self.f0_range, 1.0)
        k = np.arange(1, self.n_harmonics + 1)
        self._idx = np.clip(np.rint(np.outer(self._f0, k) / df).astype(int), 0, len(self.freqs) - 1)
        self._tol = np.maximum(1, np.ceil(self.rpm_spread * np.outer(self._f0, k) / df)).astype(int)

    def whiten(self, X: np.ndarray) -> np.ndarray:
        """X is (M, F) or (N, M, F); power is averaged over every leading axis."""
        return self.whiten_power(np.mean(np.abs(X) ** 2, axis=tuple(range(X.ndim - 1))))

    def whiten_power(self, power: np.ndarray) -> np.ndarray:
        floor = median_filter(power, size=self.floor_bins, mode="nearest") + 1e-30
        return 10 * np.log10(power / floor + 1e-12)

    def __call__(self, X: np.ndarray) -> Detection:
        snr = self.whiten(X)
        return self._best(snr, snr)

    def detect_all(self, X: np.ndarray, max_targets: int = 6) -> list[Detection]:
        """Every comb present, strongest first, with disjoint bin masks."""
        snr = self.whiten(X)
        work = snr.copy()
        claimed = np.zeros_like(snr, dtype=bool)
        found = []
        for _ in range(2 * max_targets):
            det = self._best(work, snr)
            if det.hits < self.min_hits:
                break  # nothing comb-like left
            # A comb can pass on hits yet have too few usable bins: typically a loud drone's
            # leftover rotor lines at the edges of its blanked windows. Blank it and keep
            # searching, or it would hide every weaker airframe behind it.
            det.bin_mask &= ~claimed
            claimed |= det.bin_mask
            # Blank every multiple so the next search cannot re-find this comb. Blanking is
            # kept narrow on purpose: a wide window erases other airframes' upper harmonics.
            # The cost is that one quad's rotors can come back as a second comb; those are
            # merged downstream by direction + pitch (swarm.py). If the pick was the
            # blade-pass comb, the leftover shaft-rate comb has only odd harmonics
            # (<= 4 of 8 hits), which min_hits rejects.
            work[self._near_harmonics(det.f0_hz, lo=0.0)] = 0.0
            if det.detected and det.bin_mask.sum() >= 6:
                found.append(det)
                if len(found) >= max_targets:
                    break
        return found

    def detect_near(self, power: np.ndarray, f0_hz: float, tol: float = 0.12) -> Detection:
        """Comb search on a given (e.g. beamformed) power spectrum, restricted to
        fundamentals whose octave-folded pitch is within `tol` of `f0_hz`."""
        snr = self.whiten_power(power)
        folded = np.array([fold_f0(f) for f in self._f0])
        allowed = np.abs(np.log(folded / fold_f0(f0_hz))) < np.log(1 + tol)
        return self._best(snr, snr, allowed)

    def _best(self, work: np.ndarray, snr: np.ndarray, allowed: np.ndarray | None = None) -> Detection:
        snr_c = np.clip(work, 0.0, 30.0)

        # Peak value within each harmonic's tolerance window.
        peaks = np.zeros(self._idx.shape)
        for tol in np.unique(self._tol):
            mx = maximum_filter1d(snr_c, size=2 * tol + 1, mode="nearest")
            sel = self._tol == tol
            peaks[sel] = mx[self._idx[sel]]

        hits = np.sum(peaks >= self.hit_db, axis=1)
        score = np.sum(peaks, axis=1) / self.n_harmonics
        if allowed is not None:
            hits = np.where(allowed, hits, -1)
        best = int(np.argmax(hits * 100 + score))
        f0, h = float(self._f0[best]), int(hits[best])
        detected = h >= self.min_hits
        mask = self._harmonic_mask(f0, work) if detected else np.zeros_like(snr, dtype=bool)
        return Detection(detected and mask.sum() >= 6, f0, h, float(score[best]), mask, snr)

    def _near_harmonics(self, f0: float, lo: float | None = None, spread: float | None = None) -> np.ndarray:
        lo = self.band_hz[0] if lo is None else lo
        spread = self.rpm_spread if spread is None else spread
        hi = self.band_hz[1]
        near = np.zeros(self.freqs.size, dtype=bool)
        for k in range(1, int(hi // f0) + 1):
            fk = k * f0
            if fk < lo:
                continue
            tol = max(1.0, spread * fk / self._df)
            near |= np.abs(self.freqs - fk) <= tol * self._df
        return near

    def _harmonic_mask(self, f0: float, snr: np.ndarray) -> np.ndarray:
        return self._near_harmonics(f0) & (snr >= self.hit_db)


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


def hit_threshold_db(n_channels: int, ref_db: float = 3.0, ref_channels: int = 9, p: float = 0.01) -> float:
    """Per-harmonic SNR threshold for a spectrum averaged over `n_channels` mics.

    Noise power averaged over M independent channels is Gamma(M)-distributed, so its
    spread shrinks with M. The 3 dB threshold tuned on the 9-mic array is a ~1% per-bin
    exceedance over the median floor; other channel counts get the same exceedance.
    One mic needs ~8 dB: at 3 dB a phone recording "detects" pure noise in >99% of frames.
    """
    def t(m):
        return 10 * np.log10(gamma.ppf(1 - p, m, scale=1 / m) / gamma.median(m, scale=1 / m))
    return float(ref_db + t(max(1, n_channels)) - t(ref_channels))


def fold_f0(f_hz: float, lo: float = 200.0) -> float:
    """Fold a comb fundamental into one octave [lo, 2 lo) so shaft-rate and blade-pass
    picks of the same airframe compare equal."""
    if not np.isfinite(f_hz) or f_hz <= 0:
        return float("nan")
    while f_hz >= 2 * lo:
        f_hz /= 2
    while f_hz < lo:
        f_hz *= 2
    return float(f_hz)
