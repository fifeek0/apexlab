# iRacing Telemetry Monorepo

A **free, self-hosted post-session telemetry analysis app** for iRacing — an
alternative to paid tools (Garage 61 Pro, VRS, Track Titan, Trophi.ai,
Hotlap.ai) — plus the shared core it has in common with a real-time coaching
overlay.

The paid tools' only genuinely exclusive feature is their *global* reference-lap
database. Everything else — channel graphs, delta-time, sectors, track map, GG
diagram, theoretical best, AI text coaching — is local math, and this repo
replicates all of it. Reference laps come from **imported `.ibt` files**: your
own bests plus files shared by faster drivers and teammates (see
[Reference library](#reference-library) below).

## Repository layout

```
packages/iracing-core/      shared library `iracing_core`
    ibt.py                  .ibt parsing (channels via pyirsdk + fast numpy path)
    sessions.py             telemetry-folder scan, session grouping, lap splitting
    alignment.py            distance-based multi-lap alignment + delta-time
    live.py                 live telemetry ingest (overlay; sim shared memory)
    store/                  reference-lap library (SQLite index + parquet cache)
    models.py               SessionMeta / LapData
    testing/                synthetic session generator + .ibt writer (fixtures & demo)
apps/analysis/              this analysis app (`iracing_analysis`)
    analysis/               UI-agnostic analysis lib (workspace, delta, sectors,
                            corners, theoretical best, GG, consistency, track map)
    insights/               optional AI coaching module (pluggable providers)
    gui/                    PySide6 + pyqtgraph desktop UI
apps/overlay/               overlay integration skeleton (`iracing_overlay`)
```

Both apps depend on `iracing_core`; nothing in it is duplicated. The
**overlay package is a skeleton**: the original overlay codebase was not
present on this machine, so `apps/overlay` provides the monorepo integration
point (it already consumes `iracing_core.LiveTelemetry`). Drop the existing
overlay modules (cue logic, overlay UI) into that package and replace their
ingest/lap-store/alignment code with `iracing_core` imports.

### Why a native desktop app (and not a web dashboard)?

The primary front end is **PySide6 + pyqtgraph**: multi-lap trace overlays
with a live linked cursor redraw at full frame rate even with several laps ×
seven stacked channels × 1 m resolution, which pyqtgraph handles comfortably
and browser-based plotting (Plotly/Streamlit re-rendering on every hover)
does not. The analysis library is deliberately UI-agnostic
(`iracing_analysis.analysis.*` never imports Qt), so a Streamlit/Dash
dashboard can be added later as a thin alternative front end without touching
any math.

## Prerequisites

- Python **3.11+** (3.13 tested)
- Windows for recording telemetry (iRacing itself); the analysis app runs on
  Windows/macOS/Linux — `.ibt` parsing is pure Python and cross-platform
- ~1 GB disk for a comfortable reference library (parquet-compressed laps)

## Installation

**End users (Windows):** download `IRacingSuite-Setup.exe` from the Releases
page and run it — no Python required. It installs the analysis GUI plus the
`iracing-suite` command (`engineer`, `overlay`, `agent`, `harvest`,
`diagnose` subcommands) with Start Menu shortcuts. The bundle is built by CI
from `packaging/iracing_suite.spec` (PyInstaller) + `packaging/installer.iss`
(Inno Setup); voice extras (`piper-tts`/`pyttsx3`) are optional add-ons.

**Power users:** one PowerShell command (installs `uv`, no system Python
needed): see `packaging/install.ps1`.

**Developers:**

```bash
# with uv (recommended)
uv venv && uv pip install -e packages/iracing-core -e apps/analysis -e apps/overlay

# or plain pip
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e packages/iracing-core -e apps/analysis -e apps/overlay
```

Run the test suite (`pip install pytest pytest-qt` first):

```bash
python -m pytest
```

## Recording telemetry in iRacing

1. In the car, press **Alt+L** to toggle disk telemetry logging (a small disk
   icon appears in the top-right of the screen). Do this once per session —
   or set it permanently in `app.ini`: `[Misc] irsdkLog360Hz=0` and enable
   *record telemetry to disk* in Options ▸ Misc.
2. iRacing writes one `.ibt` file per car/session stint to
   **`Documents/iRacing/telemetry/<car>/`** — e.g.
   `mclarenmp4/spa_2026-07-12 14-32-11.ibt`. Files contain all channels at
   60 Hz plus the session YAML (track, car, driver, conditions).
3. The app auto-detects that folder; override it with the *Open folder…*
   toolbar button, `--telemetry-dir`, or Settings.

### Automatic recording + auto-import (`iracing-agent`)

Two pieces make telemetry collection hands-free:

1. **Always-on recording** — with iRacing *closed*, edit
   `Documents\iRacing\app.ini`, section `[Misc]`:

   ```ini
   irsdkEnableMem=1     ; live SDK (overlay needs it anyway)
   irsdkEnableDisk=1    ; write .ibt for every session, no Alt+L needed
   irsdkLog360Hz=0      ; 60 Hz is what the analysis uses
   ```

2. **Auto-import into the library** — run the watcher while you play:

   ```bash
   iracing-agent                          # defaults: best clean lap per session
   iracing-agent --mode clean --tags own,practice
   iracing-agent --once --import-existing  # one-off: harvest the whole backlog
   ```

   The agent watches the telemetry folder; when a `.ibt` stops growing for
   `--settle` seconds (session over), it imports the best/clean laps into the
   reference library tagged `auto-import` — so the library grows by itself
   with every drive. Imports are idempotent across restarts (a file already
   in the library is never re-imported). Unless `--no-recording` is given it
   also nudges the sim to start disk telemetry (the SDK equivalent of Alt+L,
   Windows only) every 30 s, as a safety net if you skip the `app.ini` step.

## Running the analysis app

```bash
iracing-analysis                     # scan the default telemetry folder
iracing-analysis --telemetry-dir D:\ibt_files
iracing-analysis --demo              # no iRacing? explore with a synthetic session
```

Workflow:

1. **Sessions** dock: sessions are grouped from the YAML metadata; expand a
   file to list laps with exact lap times and flags (`clean`, `out`, `in`,
   `pit`, `partial`).
2. Tick 2+ laps of the same track → **Analyze selected laps**.
3. The **Pit Wall** opens first — the whole analysis on one page: the track
   map coloured by where time is lost (corner labels, glowing scrub cursor
   synced with every plot), the distance-aligned trace stack with a corner
   ribbon (segments tinted by time lost), the at-cursor readout, ranked
   corners, and the engineer's report auto-generated by the local LLM the
   moment the page opens (tyre-temperature advice appears when the
   telemetry contains tyre channels — .ibt yes, Garage 61 CSV no).
4. Detail tabs:
   - **Traces** — stacked distance-aligned plots (Δt, speed, throttle, brake,
     steering, RPM, gear) with one crosshair linked across every plot, the
     track map and the readout table (per-lap values + spread at the cursor).
   - **Track Map** — from GPS channels; colour by reference speed or by
     where the compared lap gains/loses; click the map to move the cursor.
   - **Sectors** — official iRacing sectors (from `SplitTimeInfo`) or N
     mini-sectors; best per sector highlighted; theoretical best underneath.
   - **Corners** — auto-detected corners ranked by time lost, split into
     entry / exit / run-to-next-corner, with the likely *why* (braking point,
     apex speed, throttle application, trail-braking, line).
   - **GG Diagram** — combined grip usage per lap with 1 g / 2 g circles.
   - **Consistency** — lap-time distribution, channel histograms, per-metre
     cross-lap variability.
   - **Library** — your reference-lap collection (below).
   - **AI Report** — optional local-LLM coaching report (below).
4. The toolbar's *Reference lap* combo re-bases every delta on any selected
   lap or an imported reference.

## Reference library

**This replaces the paid tools' global lap database.** It is a local store
(`~/.iracing_analysis/library/`: SQLite index + one parquet file per lap)
of laps tagged by car, track, conditions, date, driver and a *reference*
flag. Grow it from three sources:

1. **Your own bests** — select a session file and import its best/clean laps.
2. **Teammates** — ask them for the `.ibt` from `Documents/iRacing/telemetry`
   (one file is typically 5–50 MB) and *Import .ibt…* in the Library tab.
3. **Faster/pro drivers** — many league/community drivers share `.ibt` files;
   any of them imports the same way. That single shared file gives you the
   full 60 Hz pro lap — the same thing the paid subscriptions sell.
4. **Garage 61 CSV exports** — garage61.net does not hand out other drivers'
   original `.ibt`, but for laps you can access (your own + teammates on the
   free plan) the lap menu offers *Export to CSV* with the full 60 Hz
   channels. Those files import directly (Library ▸ Import, or
   `iracing_core.import_garage61_csv`): missing axes (`SessionTime`,
   `LapDist`) are reconstructed, the official lap time is taken from the
   filename (some exports miss the last few % of telemetry samples), and
   the alignment engine rescales every lap in a set to a common track
   length so CSV laps compare cleanly against `.ibt` laps. There is also an
   official token-based API (`GET /api/v1/laps/{id}/csv`) for automating
   this — see garage61.net/developer.

### Bulk harvesting from Garage 61 (`garage61-harvest`)

With a Garage 61 account (Pro recommended — it unlocks telemetry-visibility
filters) the official API can fill the library across many tracks and car
categories in one command:

```bash
export G61_TOKEN=...   # garage61.net → My applications → personal access token
garage61-harvest --track spa --track okayama --car-group "Sports Car" --dry-run
garage61-harvest --track spa --car "GR86" --per-combo 12 --export real_summaries.jsonl
```

Per (track, car) combination it selects a few outright references plus a
percentile spread of the visible field (own laps, teammates, *followed*
drivers — follow fast drivers in the Garage 61 app to widen the net),
skips laps flagged with telemetry discontinuities before downloading,
imports through the physical-consistency validator, tags everything
(`discipline:road|open_wheel|oval|dirt_road|dirt_oval`, car, track) and
reconciles one reference per combination. Runs are resumable (a ULID
manifest in the cache dir), polite by default (one request per 2 s) and
`--dry-run` shows the selection before anything is downloaded.

Library laps load back as first-class laps: add one to an analysis and use it
as the reference for delta, corners and the AI report. Programmatic use:

```python
from iracing_core import LapLibrary
lib = LapLibrary("~/.iracing_analysis/library")
lib.import_ibt("spa_pro_lap.ibt", laps="best", tags=("pro", "dry"), is_reference=True)
records = lib.list_laps(track="spa", reference_only=True)
lap = lib.get_lap(records[0].lap_id)     # LapData, ready for align_laps(...)
```

## How the analysis works (and how to configure it)

- **Alignment** — every lap is resampled onto a common 1 m `LapDist` grid
  (`grid_spacing_m` in the config). Elapsed-time-at-distance is anchored to
  the exact S/F crossings (interpolated through the `LapDistPct` wrap), so
  the cumulative delta at the finish equals the lap-time difference to
  sub-millisecond precision.
- **Sectors** — official boundaries come from the session YAML
  (`SplitTimeInfo.Sectors`); mini-sectors are N equal-length splits
  (Settings ▸ *Mini-sector count*, default 20, or the spinner in the tab).
- **Corners** — local minima of gaussian-smoothed speed (≥1.5 m/s
  prominence, ≥80 m apart) validated by steering input; each corner window
  runs from the start of its braking zone to the first sustained full
  throttle; overlaps split at the fastest point between apexes. Thresholds
  are module constants in `iracing_analysis/analysis/corners.py`
  (`MIN_PROMINENCE_MS`, `MIN_SEPARATION_M`, `FULL_THROTTLE`, …).
- **Attribution** — per corner, the delta is split into entry (braking →
  apex), exit (apex → full throttle) and *carry* (full throttle → next
  corner): a slow exit keeps costing time all the way down the straight, and
  the ranking counts that.
- **Theoretical best** — sum of your best mini-sector times across the
  selected laps. **This is an optimistic upper bound**: it stitches together
  mini-sectors from different laps that may not be simultaneously achievable
  (a later-braking gain in one mini-sector steals speed from the next).
  Treat the gap as "spread of your own performance", not a directly
  reachable target.

## Optional AI coaching reports (local LLM)

Fully optional — the app is 100 % functional without it (the default
provider is a no-op). When enabled, the app sends a **structured JSON
summary** (per-corner deltas, braking points, apex speeds, throttle timing,
trail-brake overlap, sectors, theoretical best, consistency — never raw
telemetry) to any **OpenAI-compatible** endpoint and shows the returned
"where you lose time and why" report. Post-session analysis has no latency
pressure, which makes a big local model on a DGX Spark the ideal backend.

On the DGX Spark (or any box with a GPU), serve Gemma with vLLM:

```bash
pip install vllm
vllm serve google/gemma-3-27b-it --host 0.0.0.0 --port 8000
# Ollama works too:  ollama run gemma3:27b   (endpoint http://host:11434/v1)
```

Then in the app: **Settings… ▸ AI insights** → enable, set

- *Base URL*: `http://<spark-hostname>:8000/v1`
- *API key*: `none` (vLLM/Ollama accept any placeholder)
- *Model*: `google/gemma-3-27b-it`

*Test connection* checks the endpoint; **AI Report ▸ Generate** produces the
report (generation runs in a background thread). If the endpoint is
unreachable you get a clear message and everything else keeps working. The
same JSON can be exported for any other tooling. Custom backends implement
`iracing_analysis.insights.base.InsightProvider`.

Settings live in `~/.iracing_analysis/config.json`.

## Live comparison overlay (`iracing-overlay`)

A Bloops-style in-sim overlay comparing you live against a reference lap
from your library (frameless, always-on-top, translucent):

- **delta bar** — width scales to ±2 s, colour blends red→white→green with
  the speed gap (±5 km/h), like iRacing's own bar but against *your chosen
  reference*, not just your best;
- **input trace** — your last 250 m of throttle/brake drawn over the
  reference's inputs, including the next 150 m of the reference, so you see
  the braking point coming;
- **gear hint** — the gear bubble turns green when the reference runs a
  higher gear (shift up), red when lower;
- **audio cues** — three rising tones approaching each reference braking
  zone (learn braking points without looking down).

```bash
iracing-overlay                          # live (Windows, sim running)
iracing-overlay --replay stint.ibt --realtime   # try it anywhere, no sim needed
iracing-overlay --reference-id 3         # pin a specific library lap
iracing-overlay --check | --watch        # shared-core diagnostics
```

The reference resolves automatically from the library (fastest lap for the
current track). Combine with `iracing-engineer` for spoken lap-by-lap
analysis on top of the visual overlay.

## Synthetic sessions (no iRacing needed)

`iracing_core.testing` generates a physically plausible fantasy circuit
(closed 10-corner track, three-pass speed profile) and writes real `.ibt`
files that pyirsdk itself parses — that is what the test-suite fixtures and
`--demo` mode use, so every feature can be exercised on any machine.

## Live race engineer (`iracing-engineer`)

A Crew-Chief-style pit wall, but built on your own reference library: while
you drive, every completed lap is aligned against a chosen reference lap,
per-corner losses are attributed, and you get a short engineer update —
printed, optionally spoken, optionally phrased by your local LLM:

```bash
iracing-engineer                          # live (Windows, sim running), Polish
iracing-engineer --language en --tts      # spoken updates
iracing-engineer --llm                    # radio line phrased by the AI endpoint
iracing-engineer --replay stint.ibt       # dry-run the pit wall on a recorded stint
iracing-engineer --reference-id 3         # pin a specific library lap as reference
```

Voice: Crew Chief uses a bank of pre-recorded human clips; we use **Piper**,
a local neural TTS with real Polish voices — much better than the OS
default. One-time setup:

```bash
pip install piper-tts
python -m piper.download_voices pl_PL-darkman-medium --data-dir ~/.iracing_analysis/voices
```

`--tts` then picks Piper automatically (models are auto-discovered in that
folder; pin one with `--piper-model`, force a backend with
`--tts-engine piper|system|off`). Without Piper it falls back to pyttsx3
(`pip install pyttsx3`), and without any TTS updates are just printed.
Speech runs on a worker thread and never blocks telemetry.

Example update (template, instant, no LLM needed):

> Okrążenie 2:50.025, +7.23 do referencji. Najwięcej tracisz: T8 +1.50 (gaz
> 32 m za późno), T3 +1.05 (apeks 22 km/h za wolny). Skup się na T8.

The reference is auto-selected from the library (fastest lap matching the
current track, then car), or pinned with ``--reference-id``. Lap timing
stays accurate even at low sampling rates (crossing interpolation), and the
analysis runs in milliseconds — the update is ready before you reach T1.

## Validating a new telemetry file

First thing to run on a `.ibt` from a new source (your PC, a teammate, a pro):

```bash
python -m iracing_core.diagnose "path/to/stint.ibt"
```

It prints the file structure, YAML metadata, channel coverage, the lap table
with flags, and cross-checks the delta engine (delta at S/F vs lap-time
difference) on the two best clean laps. Exit code 0 = the whole toolchain
handles the file.

## Development

- `python -m pytest` — 76 tests: parsing (validated against pyirsdk as the
  oracle), alignment, delta, sectors/corners math, theoretical best, library
  round-trips, GUI behaviour (offscreen), AI provider against a fake server.
- Phase-gate acceptance criteria live in the test names, e.g.
  `test_delta_at_finish_matches_lap_time_difference`,
  `test_corner_detection_matches_track_layout`,
  `test_theoretical_best_not_slower_than_actual`.
