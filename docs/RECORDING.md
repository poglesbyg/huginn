# Recording a drone for Huginn

The goal is real audio that `scripts/replay.py` can run through the same pipeline the simulator feeds. Even
one good session answers the obvious question about this project: *does it work on a real quad?*

## 1. Pick your gear by what you want to learn

| Gear | What Huginn can do with it | Notes |
|---|---|---|
| **Phone, one mic** | Detection only: is a drone there, what is its rotor pitch, out to what range | The easiest first step. The detector raises its threshold automatically for one channel, so noise does not fire it. |
| **Stereo recorder** | Detection only | Two mics cannot give a 3-D bearing; the pipeline needs at least 3. |
| **USB mic array** (4–8 mics, ~5–10 cm across) | Detection + bearings | Must expose **raw** per-mic channels. Small aperture means a wide beam and coarse bearings, but it works. |
| **Multichannel interface + 4–9 mics on a frame** (0.3–0.6 m across) | Detection, bearings, tracking: closest to the reference array | All mics must go through **one interface (one sample clock)**. Separate USB mics drift apart in time and ruin direction-finding. |

## 2. Recorder settings

- **WAV, 48 kHz, 16- or 24-bit.** Huginn resamples to 16 kHz itself.
- **Turn off all processing:** automatic gain, noise suppression, echo cancellation, "voice" or "wind" modes.
  On a phone, use a recorder app that saves uncompressed WAV with processing off. Built-in voice-memo apps
  usually filter exactly the frequencies Huginn needs.
- **Set gain so a close pass peaks below about −6 dBFS.** Clipping smears the harmonics.
- Use a foam or furry windscreen if there is any wind.

## 3. Describe the array (multichannel only)

Write `myrec.geometry.json` (template: [`examples/geometry_circular6_template.json`](../examples/geometry_circular6_template.json)):

```json
{
  "mics_m": [[0.05, 0.0, 0.0], [0.025, 0.0433, 0.0], "..."],
  "channels": [1, 2, 3, 4, 5, 6],
  "heading_deg": 0.0,
  "note": "what this is"
}
```

- **Positions** are in metres, measured to about 5 mm. Axes: **x forward, y left, z up**, with the origin
  anywhere. A planar array can't tell above from below; Huginn assumes the drone is above.
- **Channel map:** tap each mic in order and see which WAV channel spikes. Many boards put a processed mix on
  channel 0 and raw mics after it. Don't trust the datasheet; check.
- **Heading:** the compass bearing the array's +x axis points to. With it, the output CSV includes compass
  bearings.
- **Calibration (optional):** `pa_per_full_scale` turns on level-based range. Play a 94 dB SPL calibrator
  (1 Pa rms) into one mic, measure that channel's rms in full-scale units, and set the value to `1 / rms`.
  Without it, Huginn disables level-based range and says so.

Check the geometry before going out by running `replay.py` once. Its `array` block prints the aperture, the
frequency above which spatial aliasing starts, and the beamwidth at 1 kHz.

## 4. Session plan (about 30 minutes of flying)

Fly only with a pilot who is legally allowed to fly there (in the US, FAA Part 107 or the recreational rules),
within visual line of sight, away from people and airports, and with the landowner's OK.

1. **Start with 60 s of background**, no drone. This is your false-alarm test.
2. **Sync mark:** clap or beep at the array while the flight log is running, so audio and log line up.
3. **Hovers at 10, 25, 50, 100 m**, 15 s each, at a few bearings. This gives detection-vs-range and bearing
   accuracy.
4. **Passes:** a slow crossing pass, a fast crossing pass, then an inbound run toward the array (stop well
   short). This exercises tracking and the acoustic-lag compensation.
5. **Clutter:** repeat one hover with a generator, idling car or leaf blower running nearby.
6. **Two drones** at once, if available. Ideally use different prop sizes so their pitches differ. This
   exercises swarm mode.
7. **End with 60 s of background.**

Keep a written log too: time, what was flown, distance, and bearing by eye or phone compass.

## 5. Ground truth

The flight controller's log (GPS, or a blackbox file) gives position over time. Convert it to a truth CSV
with columns `t_s, az_deg, el_deg` in the **array frame**:

- Express the drone position relative to the array as east E, north N, up U, in metres.
- Rotate into the array frame using `heading_deg` (h): `x = E·sin h + N·cos h`, `y = −E·cos h + N·sin h`.
- Then `az = atan2(y, x)` and `el = atan2(U, hypot(x, y))`.
- Shift time so that `t_s = 0` is the start of the WAV, using the sync mark.

Remember that sound is late: the true *acoustic* direction is where the drone was r/c seconds earlier, about
0.3 s at 100 m. For hovers it doesn't matter; for fast passes, compare against the delayed position.

## 6. Replay

```bash
.venv/bin/python scripts/replay.py phone.wav
```

```bash
.venv/bin/python scripts/replay.py myrec.wav --geometry myrec.geometry.json --truth myrec.truth.csv
```

```bash
.venv/bin/python scripts/replay.py myrec.wav --geometry myrec.geometry.json --swarm --start 120 --end 180
```

Each run writes `out/replay_<name>.png` (spectrogram, detections, bearings vs truth), `.csv` (per-frame
telemetry) and `.json` (summary).

What to look for:

- **Background segments:** no confirmed tracks. Any you see are false alarms; note what caused them.
- **Hovers:** the detection rate at each distance gives the real detection range; compare it with the
  simulator's ~200 m.
- **The pitch** in the CSV should match the prop rate you can hear and stay steady in a hover.
- **Bearing error vs truth:** compare with the simulator's ~0.4° median. Expect worse on a small array. A
  large *constant* offset means a heading or channel-map mistake, not a DSP problem.

## 7. Try it now, without a drone

```bash
.venv/bin/python scripts/make_sample_recording.py
```

```bash
.venv/bin/python scripts/replay.py samples/sim_flyby.wav --geometry samples/sim_flyby.geometry.json --truth samples/sim_flyby.truth.csv
```

```bash
.venv/bin/python scripts/replay.py samples/sim_flyby_mono.wav
```

This writes a simulated flyby to disk *as if it came off hardware*: 48 kHz, 16-bit, 9 channels, with a
geometry file and truth. That exercises file loading, channel mapping, scaling and resampling end to end.
