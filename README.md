# Huginn: passive acoustic FPV cueing

*Huginn ("thought") was one of Odin's two ravens. It flew out, listened, and reported back.*

Huginn is a small, working model of the first stage of a counter-FPV kill chain. It hears a drone, finds its
direction, holds a track on it, and hands a pointing cue to a fine (visual) tracker. It uses no RF, emits
nothing, and has no line-of-sight requirement until handoff. A second node 50 m away triangulates range, so
the cue can correct for the fact that sound arrives late.

The design follows the shape of the problem 9 Mothers describes publicly: a passive acoustic sensor gives a
coarse track, then hands off to fine tracking, and nodes are networked into a grid. Here, everything upstream
of the camera is built, calibrated against simulated truth, and measured.

![demo](out/demo.png)

## Why this problem

Fiber-optic and autonomous FPVs make RF detection and jamming unreliable, but every multirotor still
*sounds* like a multirotor. It produces a stack of narrow tones at multiples of its rotor shaft rate. The hard
parts are not "run an FFT":

| Problem | What Huginn does about it |
|---|---|
| Wind, ambient noise and generators swamp energy detectors | **Harmonic-comb detector** that scores candidate fundamentals by how many harmonics stand out from a whitened noise floor. 60 Hz-family hum and broadband wind don't form the right comb. |
| Loud nearby clutter pulls a beamformer off target | **Harmonic-masked SRP-PHAT**: only bins attributed to the drone enter the beamformer, so a generator 20 m away doesn't move the peak. |
| You need an honest error bar, not just an angle | Per-fix σ from a Cramér-Rao-style model (`σ ∝ 1/√ΣSNR`), **calibrated against truth** (observed/predicted = 0.97). |
| Spurious fixes and mirror-image outliers | Kalman CV bearing filter with χ² gating, M-of-N confirmation, and bounded coasting. |
| **Sound is late.** The bearing points at where the drone *was* r/c seconds ago, up to ~7° off for a fast crossing target | The cue leads by `r/c + system latency`. Rate is shrunk by its own SNR so a long lead on a noisy track widens the cone instead of pointing confidently wrong. |
| …which needs range, and one array can't measure range | **Two-node triangulation** (`fusion.py`): closest approach of two bearing rays, with σ from crossing geometry plus inter-node emission-time skew. This is fused in log-range with a received-level estimate. The level estimate is always available but biased by how loud each airframe is, so it never averages down; the triangulated estimate does. |
| The camera needs to know *where to look and how wide* | Each `Cue` carries a pointing command, a cone half-angle (2σ + range-uncertainty residual + a 3 g manoeuvre during the lead), and `handoff_ready` when the cone fits inside a 20° acquisition FOV. |

## Results

### Monte Carlo, paired: one node vs two nodes (`scripts/evaluate.py`)

There are 12 randomised engagements. Each varies approach azimuth (from every direction, so some arrive along
the baseline, where triangulation geometry is poor), altitude (15–60 m), speed (20–40 m/s), weave, rotor rate
(220–300 Hz), airframe loudness (86 ± 3 dB at 1 m), ambient noise, wind, and the position of a 50 or 60 Hz
generator. Every engagement is rendered once onto both arrays, then scored twice on **identical audio**: node A
alone, and node A fused with node B (B is 50 m away).

| Metric (median [p10, p90] across runs) | 1 node (level prior) | 2 nodes (triangulation) |
|---|---|---|
| Range estimate error | 30% [13, 53] | **11% [6, 18]** |
| Handoff-ready range | 164 m [95, 282] | **189 m [127, 286]** |
| Cue error vs physical position, median per run | 1.9° [1.3, 2.3] | 1.7° [1.3, 2.5] |
| Cue error vs physical position, p90 per run | 5.7° | 5.6° |
| Handoff containment (drone actually inside the cone when "ready") | 98% [92, 100] | 97% [94, 98] |
| Confirmed track range | 199 m [130, 318] | same |
| Raw bearing error, median per run | 0.40° [0.34, 0.45] | same |
| Gross bearing errors (>5°) | 0.1% [0, 2.1] | same |
| Frames with a valid triangulation fix | — | 90% [79, 95] |
| Real-time factor at p99 compute | 1.8× | 1.0× (both nodes on one CPU) |

**What triangulation bought:** a range estimate that is roughly 3× better, with a far tighter worst case
(p90 of 18% vs 53%). That lets the cone shrink honestly, so handoff happens **25 m earlier in the median and
32 m earlier in the worst decile**, while staying at ≥95% containment. That is roughly 1 extra second of
camera time on a 30 m/s target.

**What it didn't buy:** pointing accuracy is essentially unchanged, about 1.8°. Once range is known to ~30%,
lag correction is no longer the limit; angular-rate noise and drone manoeuvres during the lead are. I only
found this by measuring. The first two-node build *lowered* containment (90% → 82%), because the loose level
prior had been hiding an overconfident tracker. The fix was a physical bound on manoeuvre during the lead
(0.5 · a · lead² / r, at 3 g), applied to both modes. That fix also raised single-node containment from 87%
to 98%.

Clutter-only scenes (3 minutes; 1–3 generators up to 92 dB, wind up to 70 dB): **0 false tracks**. That is a
smoke test, not a false-alarm-rate claim. Fusion cannot create tracks, because range is applied only to
gated hits on A's existing track, so this result holds in both modes.

![envelope](out/evaluation.png)

### Single engagement (`scripts/demo.py`)

An FPV runs in from 400 m at 30 m/s with a 20 m lateral weave, past a running 60 Hz generator. The figure above
shows the two-node cue.

| Metric | 1 node | 2 nodes |
|---|---|---|
| Confirmed track / handoff-ready from | 196 m / 173 m | 196 m / 183 m |
| Raw bearing error vs acoustic truth (median / p90) | 0.54° / 1.9° | same |
| Cue error vs physical position (median / p90) | 2.3° / 6.1° | 2.2° / 5.8° |
| Range estimate error | 29% | 13% |
| Handoff containment | 98% | 96% |

## Layout

```
huginn/
  scenario.py   physics sim: retarded-time rendering (Doppler, delay, 1/r, air absorption),
                4-rotor FPV model with throttle wander, generators, wind, ambient
  detect.py     harmonic-comb detector, tonal level (Parseval-calibrated SPL)
  doa.py        harmonic-masked SRP-PHAT, coarse-to-fine search, calibrated sigma
  tracker.py    bearing Kalman filter, M-of-N, gating, fused log-range (level + triangulation),
                lag-compensated Cue with manoeuvre-aware cone
  pipeline.py   per-node frame loop: measure (detect -> localise -> level) then track -> cue
  fusion.py     ray triangulation with calibrated sigma, two-node system
  metrics.py    scoring against acoustic truth and physical truth
scripts/
  demo.py       one engagement, both modes -> out/demo.png, out/demo_metrics.json
  evaluate.py   paired 1-node vs 2-node Monte Carlo + clutter-only runs -> out/evaluation.*
  calibrate.py  fits sigma_scale and the source-level prior from simulated truth
tests/          DOA accuracy, clutter rejection, inverse-square level, tracker gating, az wrap,
                triangulation geometry/rejection/sigma coverage, range fusion, two-node end-to-end
```

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q tests
.venv/bin/python scripts/demo.py
.venv/bin/python scripts/evaluate.py 12 3
```

Each node is an 8-mic, 0.25 m radius ring plus one apex mic 0.30 m up (so elevation is unambiguous), at 16 kHz,
using 1024-sample frames with a 512 hop. Node B is at (40, −30, 0) m relative to A. The simulation assumes
synchronised clocks; in the field, that means GPS-disciplined clocks.

## Honest limitations

- **It's a simulation.** The propagation physics is real, but there is no ground reflection, terrain, turbulence,
  rotor directivity, or mic mismatch. Real-world performance will be lower until it's calibrated on recorded data.
  The first thing I'd do with real hardware is replay range recordings through this same `Pipeline` and refit
  `calibrate.py`.
- **Single target.** Each node holds one track, and fusion assumes both nodes are tracking the same drone. A
  ray-miss gate rejects gross mismatches, but swarms need proper cross-node association (for example, matching
  combs by fundamental frequency, which is a strong fingerprint per airframe) and a peel-off beamformer.
- **Inter-node timing skew.** The two nodes hear the drone at emission times |r_A − r_B|/c apart. That skew is
  currently absorbed into σ, not corrected; correcting it is iterative and cheap. Range error rises inside
  ~30 m, where skew and near-field effects dominate.
- **Terminal phase (<20 m).** Angular acceleration in the final dive exceeds the constant-velocity model, and
  track continuity drops to ~73%. By then the visual tracker should own the target, but an IMM (CV + high-g
  model) would close this gap cheaply.
- **Baseline geometry.** Targets arriving along the A–B axis triangulate poorly; they fall back to the level
  prior automatically. A third node fixes this.
- **Spatial aliasing** at 0.19 m mic spacing gives a small rate of gross bearing errors at low SNR. The gate
  rejects them, but a larger or non-uniform aperture would reduce them at the source.

## What I'd build next

1. **Pointing accuracy is now the bottleneck, not range.** An IMM or manoeuvre-adaptive filter, plus using
   Doppler on the rotor comb as a radial-velocity observable, to tighten angular rate.
2. **N-node grid.** Generalise `fusion.py` to least-squares over any number of rays, with per-node clock-offset
   estimation and cross-node comb association. This is the networked-grid direction 9 Mothers has published.
3. **Learned detector head.** Swap the comb score for a small CNN on the whitened spectrogram, trained on this
   simulator plus range recordings. Keep the comb as the explainable fallback and as a label generator.
4. **C++/embedded port** of `detect` + `doa` with precomputed steering tables. One node already runs at ~1.8×
   real time at p99 in numpy, and the hot loop is a single einsum.
5. **Closed loop with the gimbal.** Feed `Cue.lead_*` into a slew-rate-limited mount model and measure time
   from first detection to camera-on-target.
