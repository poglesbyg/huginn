"""Physics-based acoustic scene simulator.

Every tonal component is rendered analytically at the *retarded* (emission)
time for each microphone, so propagation delay, Doppler, spherical spreading
and atmospheric absorption all fall out of the model instead of being bolted on.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, sosfilt

from .geometry import SPEED_OF_SOUND

P_REF = 20e-6  # Pa, 0 dB SPL


def db_to_pa(db: float) -> float:
    return P_REF * 10 ** (db / 20)


def absorption_db_per_m(f_hz):
    """Rough ISO 9613-1 style air absorption at ~20 C / 50% RH."""
    f_k = np.asarray(f_hz) / 1000.0
    return 0.005 * f_k**1.7


@dataclass
class Trajectory:
    times: np.ndarray       # (N,)
    positions: np.ndarray   # (N, 3)

    def pos(self, t) -> np.ndarray:
        t = np.asarray(t)
        return np.stack([np.interp(t, self.times, self.positions[:, i]) for i in range(3)], axis=-1)

    @property
    def duration(self) -> float:
        return float(self.times[-1] - self.times[0])


def static(position, duration: float) -> Trajectory:
    p = np.asarray(position, dtype=float)
    return Trajectory(np.array([0.0, duration]), np.stack([p, p]))


def inbound_fpv(start_range=400.0, az_deg=40.0, alt=35.0, speed=30.0,
                weave_amp=20.0, weave_period=5.0, end_point=(6.0, 4.0, 3.0), dt=0.02) -> Trajectory:
    """One-way-attack FPV run: straight-line ingress with a lateral weave that
    tapers to zero on terminal approach."""
    az = np.radians(az_deg)
    start = np.array([start_range * np.cos(az), start_range * np.sin(az), alt])
    end = np.asarray(end_point, dtype=float)
    d = end - start
    length = np.linalg.norm(d)
    d /= length
    lateral = np.cross(d, [0.0, 0.0, 1.0])
    lateral /= np.linalg.norm(lateral)

    t = np.arange(0.0, length / speed + dt, dt)
    s = np.minimum(speed * t, length)
    taper = np.clip((length - s) / 80.0, 0.0, 1.0)
    pos = start + np.outer(s, d) + np.outer(weave_amp * taper * np.sin(2 * np.pi * t / weave_period), lateral)
    return Trajectory(t, pos)


@dataclass
class TonalSource:
    """A source made of harmonic "voices" (e.g. one per rotor)."""
    trajectory: Trajectory
    fundamentals_hz: list[float]
    harmonic_weights: np.ndarray          # (K,) relative amplitude of harmonic k=1..K
    level_db_1m: float                    # total tonal SPL at 1 m
    broadband_db_1m: float | None = None  # optional broadband component
    wander: float = 0.0                   # relative RPM wander (throttle changes)
    wander_timescale: float = 1.5
    t_off: float = np.inf                 # source stops emitting (impact) at this time


def fpv_quad(trajectory: Trajectory, rotor_hz=260.0, level_db_1m=86.0, t_off=np.inf) -> TonalSource:
    """5-inch class quad: 4 rotors a few % apart, 2-blade props, so even
    harmonics (blade-pass family) dominate the shaft-rate odd harmonics."""
    k = np.arange(1, 17)
    weights = np.where(k % 2 == 0, 1.0, 0.35) / k**0.6
    spread = [-0.018, -0.006, 0.007, 0.016]
    return TonalSource(trajectory, [rotor_hz * (1 + s) for s in spread], weights, level_db_1m,
                       broadband_db_1m=level_db_1m - 8, wander=0.04, t_off=t_off)


def generator(position, duration, level_db_1m=80.0, mains_hz=60.0) -> TonalSource:
    """Stationary diesel generator: strong 60 Hz family, a classic distractor."""
    k = np.arange(1, 11)
    return TonalSource(static(position, duration), [mains_hz], 1.0 / k, level_db_1m)


@dataclass
class Scene:
    mics: np.ndarray
    fs: int = 16000
    duration: float = 10.0
    sources: list[TonalSource] = field(default_factory=list)
    ambient_db: float = 45.0  # overall SPL of uncorrelated sensor + ambient noise
    wind_db: float = 55.0     # low-frequency wind rumble, < ~150 Hz
    seed: int = 0


def _emission_times(traj: Trajectory, mic: np.ndarray, t: np.ndarray, iters: int = 4):
    """Solve t_e = t - |p(t_e) - mic| / c by fixed-point iteration."""
    te = t.copy()
    for _ in range(iters):
        r = np.linalg.norm(traj.pos(te) - mic, axis=-1)
        te = t - r / SPEED_OF_SOUND
    return te, np.maximum(r, 0.5)


def _smooth_noise(rng, n, fs, timescale):
    sos = butter(1, 1.0 / timescale, fs=fs, output="sos")
    x = sosfilt(sos, rng.standard_normal(n))
    return x / (np.std(x) + 1e-12)


def _render_source(src: TonalSource, scene: Scene, t: np.ndarray, rng) -> np.ndarray:
    fs = scene.fs
    pad = 2.0  # seconds of history so early samples can look back to emission time
    tg = np.arange(-pad, scene.duration, 1.0 / fs)
    K = len(src.harmonic_weights)

    # Normalise so the summed tonal power across voices and harmonics hits the target level.
    n_voices = len(src.fundamentals_hz)
    p_total = db_to_pa(src.level_db_1m)
    norm = p_total / np.sqrt(n_voices * np.sum(src.harmonic_weights**2 / 2))
    amps = src.harmonic_weights * norm

    # Throttle is mostly common to all rotors; attitude control adds a small per-rotor part.
    voice_phase, voice_freq = [], []
    common = _smooth_noise(rng, tg.size, fs, src.wander_timescale) if src.wander else 0.0
    for f0 in src.fundamentals_hz:
        own = _smooth_noise(rng, tg.size, fs, src.wander_timescale / 3) if src.wander else 0.0
        drift = 1.0 + src.wander * (0.9 * common + 0.3 * own)
        f = f0 * drift * np.ones_like(tg)
        voice_freq.append(f)
        voice_phase.append(2 * np.pi * np.cumsum(f) / fs)
    offsets = rng.uniform(0, 2 * np.pi, size=(n_voices, K))

    broadband = None
    if src.broadband_db_1m is not None:
        sos = butter(4, [200, 6000], btype="band", fs=fs, output="sos")
        broadband = sosfilt(sos, rng.standard_normal(tg.size))
        broadband *= db_to_pa(src.broadband_db_1m) / np.std(broadband)

    out = np.zeros((len(scene.mics), t.size))
    for m, mic in enumerate(scene.mics):
        te, r = _emission_times(src.trajectory, mic, t)
        # Fade over 20 ms at t_off rather than hard-cutting, to avoid a broadband click.
        on = np.clip((src.t_off - te) / 0.02, 0.0, 1.0) if np.isfinite(src.t_off) else 1.0
        for v in range(n_voices):
            ph = np.interp(te, tg, voice_phase[v])
            f_inst = np.interp(te, tg, voice_freq[v])
            for k in range(1, K + 1):
                atten = on * 10 ** (-absorption_db_per_m(k * f_inst) * r / 20) / r
                out[m] += amps[k - 1] * atten * np.sin(k * ph + offsets[v, k - 1])
        if broadband is not None:
            out[m] += on * np.interp(te, tg, broadband) * 10 ** (-absorption_db_per_m(2000) * r / 20) / r
    return out


def render(scene: Scene) -> tuple[np.ndarray, np.ndarray]:
    """Returns (t, signals) with signals shaped (M, N) in pascals."""
    rng = np.random.default_rng(scene.seed)
    t = np.arange(0.0, scene.duration, 1.0 / scene.fs)
    x = np.zeros((len(scene.mics), t.size))
    for src in scene.sources:
        x += _render_source(src, scene, t, rng)

    # Uncorrelated per-channel noise: pinkish ambient + low-frequency wind buffeting.
    white = rng.standard_normal(x.shape)
    pink = sosfilt(butter(1, 500, fs=scene.fs, output="sos"), white, axis=-1) + 0.15 * white
    x += pink * db_to_pa(scene.ambient_db) / np.std(pink)
    wind = sosfilt(butter(2, 150, fs=scene.fs, output="sos"), rng.standard_normal(x.shape), axis=-1)
    x += wind * db_to_pa(scene.wind_db) / np.std(wind)
    return t, x


def acoustic_truth(traj: Trajectory, t: np.ndarray, origin=np.zeros(3)):
    """Where the drone *was* when it emitted the sound arriving at the array at t,
    plus where it *is* at t. The gap between the two is the acoustic lag."""
    te, _ = _emission_times(traj, origin, np.asarray(t, dtype=float))
    return traj.pos(te) - origin, traj.pos(t) - origin


@dataclass
class Raider:
    trajectory: Trajectory
    rotor_hz: float
    level_db_1m: float

    def source(self) -> "TonalSource":
        return fpv_quad(self.trajectory, self.rotor_hz, self.level_db_1m, t_off=self.t_end)

    @property
    def t_end(self) -> float:
        """Arrival time; after this the airframe would have impacted."""
        return float(self.trajectory.times[-1])


def swarm_raid(n: int, rng: np.random.Generator, sector_deg: float = 60.0) -> list[Raider]:
    """n FPVs from one threat sector, staggered in range, each with its own airframe
    (rotor rate, loudness) and its own terminal aim point near the defended site."""
    centre = rng.uniform(0, 360)
    out = []
    for _ in range(n):
        end = np.array([rng.uniform(-15, 15), rng.uniform(-15, 15), rng.uniform(1, 5)])
        traj = inbound_fpv(start_range=rng.uniform(250, 380),
                           az_deg=centre + rng.uniform(-sector_deg / 2, sector_deg / 2),
                           alt=rng.uniform(15, 60), speed=rng.uniform(20, 40),
                           weave_amp=rng.uniform(0, 25), weave_period=rng.uniform(3, 7), end_point=end)
        out.append(Raider(traj, rotor_hz=rng.uniform(200, 320), level_db_1m=rng.normal(86, 3)))
    return out
