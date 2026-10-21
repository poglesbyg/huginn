"""Harmonic-masked SRP-PHAT direction finding.

Steered Response Power with PHAT weighting, evaluated only on the bins the
detector attributed to the drone. Masking is what makes this robust: the
generator, wind and ambient energy never enter the beamformer, so they cannot
pull the peak. Coarse-to-fine search keeps it cheap enough for an embedded CPU.
"""

from dataclasses import dataclass

import numpy as np

from .geometry import SPEED_OF_SOUND, unit_vector


@dataclass
class DoaEstimate:
    az_deg: float
    el_deg: float
    coherence: float  # normalised SRP peak in [0, 1]; 1 = perfectly coherent plane wave
    sigma_deg: float  # predicted 1-sigma angular error
    n_bins: int


def _grid(az_range, el_range, step):
    az = np.arange(az_range[0], az_range[1], step)
    el = np.arange(el_range[0], el_range[1] + 1e-9, step)
    A, E = np.meshgrid(az, el)
    return A.ravel(), E.ravel()


class SrpPhat:
    def __init__(self, mics: np.ndarray, freqs: np.ndarray, coarse_step=3.0, fine_step=0.25,
                 el_range=(-10.0, 85.0), sigma_scale=3.8, sigma_floor=0.15, cache_hz=4100.0):
        self.mics = mics - mics.mean(axis=0)
        self.freqs = freqs
        self.fine_step = fine_step
        self.coarse_step = coarse_step
        self.el_range = el_range
        self.sigma_scale = sigma_scale
        self.sigma_floor = sigma_floor
        self._caz, self._cel = _grid((0.0, 360.0), el_range, coarse_step)
        self._ctau = self._delays(self._caz, self._cel)
        # Coarse steering phases for every bin up to cache_hz, computed once. The complex
        # exponential dominated runtime; with a swarm there is one SRP per airframe per frame.
        nb = int(np.searchsorted(freqs, cache_hz, side="right"))
        self._cached_bins = nb
        self._csteer = np.exp(-2j * np.pi * self._ctau[:, :, None] * freqs[None, None, :nb]).astype(np.complex64)

    def _delays(self, az, el):
        # A plane wave from direction u reaches mic m earlier by (u . r_m) / c.
        return unit_vector(az, el) @ self.mics.T / SPEED_OF_SOUND  # (D, M)

    def _srp(self, Y, f, tau):
        steer = np.exp(-2j * np.pi * tau[:, :, None] * f[None, None, :])  # (D, M, F)
        beam = np.einsum("dmf,mf->df", steer, Y)
        return np.sum(np.abs(beam) ** 2, axis=1) / (Y.shape[0] ** 2 * Y.shape[1])

    def beam_power(self, X: np.ndarray, az: float, el: float) -> np.ndarray:
        """Delay-and-sum beam power spectrum steered at (az, el): coherent gain of up to
        10 log10(M) dB over the array-averaged spectrum for a source in the beam."""
        tau = self._delays(np.array([az]), np.array([el]))[0]
        steer = np.exp(-2j * np.pi * np.outer(tau, self.freqs))
        return np.abs(np.sum(X * steer, axis=0)) ** 2 / X.shape[0] ** 2

    def __call__(self, X: np.ndarray, mask: np.ndarray, snr_db: np.ndarray) -> DoaEstimate | None:
        """X is (M, F) or a stack of snapshots (N, M, F) sharing one bin mask."""
        if mask.sum() < 3:
            return None
        X = X if X.ndim == 3 else X[None]
        n_snap = X.shape[0]
        Y = np.concatenate([x[:, mask] for x in X], axis=1)  # snapshots act as extra bins
        Y = Y / (np.abs(Y) + 1e-20)
        f = np.tile(self.freqs[mask], n_snap)

        idx = np.flatnonzero(mask)
        if idx[-1] < self._cached_bins:
            steer = self._csteer[:, :, np.tile(idx, n_snap)]
            beam = np.einsum("dmf,mf->df", steer, Y.astype(np.complex64))
            p = np.sum(np.abs(beam) ** 2, axis=1) / (Y.shape[0] ** 2 * Y.shape[1])
        else:
            p = self._srp(Y, f, self._ctau)
        i = int(np.argmax(p))
        az0, el0 = self._caz[i], self._cel[i]

        s = self.coarse_step
        faz, fel = _grid((az0 - s, az0 + s), (max(el0 - s, self.el_range[0]), min(el0 + s, self.el_range[1])),
                         self.fine_step)
        pf = self._srp(Y, f, self._delays(faz, fel))
        j = int(np.argmax(pf))

        # Cramer-Rao-like scaling: error ~ 1 / sqrt(total SNR across the bins used), plus a
        # floor for model error (far-field assumption, grid). sigma_scale was fitted against
        # simulated truth (see scripts/calibrate.py).
        snr_lin = 10 ** (snr_db[mask] / 10)
        sigma = float(np.clip(np.hypot(self.sigma_floor, self.sigma_scale / np.sqrt(n_snap * np.sum(snr_lin))),
                              self.sigma_floor, 15.0))
        return DoaEstimate(float(faz[j] % 360.0), float(fel[j]), float(pf[j]), sigma, int(mask.sum()))
