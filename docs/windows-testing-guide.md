# Windows PC Testing Guide

Step-by-step testing checklist for the Apex Lab suite on a Windows gaming PC with iRacing installed.

## Prerequisites

- Windows 10/11
- iRacing installed and working
- Python 3.11+ (or just uv — it installs Python for you)
- Git

## Installation

### Option A: Developer install (recommended for first test)

```powershell
git clone https://github.com/fifeek0/apexlab.git
cd apexlab

# Install uv if not present
irm https://astral.sh/uv/install.ps1 | iex

# Create venv and install everything
uv venv
uv pip install -e packages/iracing-core -e apps/analysis -e apps/overlay
uv pip install pytest pytest-qt
```

### Option B: PyInstaller bundle (test after Option A works)

```powershell
uv pip install pyinstaller
cd packaging
pyinstaller iracing_suite.spec --noconfirm
# Output: dist/iracing-suite/IRacingAnalysis.exe + iracing-suite.exe
```

## Test Checklist

### 1. Basic smoke test (no iRacing needed)

```powershell
# All commands available?
iracing-analysis --help
iracing-engineer --help
iracing-overlay --help
iracing-agent --help

# Demo mode (synthetic session, no iRacing)
iracing-analysis --demo
```

**Expected:** GUI opens with dark theme, session browser shows "Fantasia International", 5 clean laps listed. Tick 3 laps → "Analyze selected laps" → Pit Wall tab with traces, track map, corners, readout. Cursor moves across all plots.

**Check:**
- [ ] GUI launches without errors
- [ ] Dark theme applied (no white/light panels)
- [ ] Track map renders closed circuit with T1-T9 labels
- [ ] Linked cursor works across all plots + map
- [ ] Corners tab shows ranked time losses
- [ ] Sectors tab shows sector times with best highlighted

### 2. Test suite

```powershell
set QT_QPA_PLATFORM=offscreen
python -m pytest
```

**Expected:** 142+ tests pass. Some Qt tests may need `offscreen` platform.

**Check:**
- [ ] All tests pass (or note which fail on Windows)

### 3. .ibt file parsing (no live sim needed)

Find any `.ibt` file in `Documents\iRacing\telemetry\` and run:

```powershell
python -m iracing_core.diagnose "C:\Users\<you>\Documents\iRacing\telemetry\<car>\<file>.ibt"
```

**Expected:** File structure, YAML metadata (track, car, driver), channel coverage, lap table with times and flags, delta cross-check "OK".

**Check:**
- [ ] Track name, car, driver correctly parsed from YAML
- [ ] Lap times match what iRacing showed
- [ ] Clean/out/in lap flags correct
- [ ] Delta check passes (OK)
- [ ] No encoding errors in driver/track names (Polish, German, etc.)

### 4. Telemetry folder scan

```powershell
iracing-analysis --telemetry-dir "C:\Users\<you>\Documents\iRacing\telemetry"
```

**Expected:** Sessions grouped by track/car in the browser dock. Expand → laps listed.

**Check:**
- [ ] Sessions discovered and grouped
- [ ] Lap times displayed correctly
- [ ] Multiple .ibt files from same session grouped together

### 5. Real lap analysis

In the GUI: tick 2+ clean laps from the same track → "Analyze selected laps".

**Check:**
- [ ] Pit Wall dashboard loads with all instruments
- [ ] Delta graph shows realistic values (matches lap time difference at S/F)
- [ ] Speed/Throttle/Brake/Steering traces look correct
- [ ] Track map shape matches the real circuit
- [ ] Corner detection finds reasonable number of corners
- [ ] Cursor hover shows per-lap values in readout table
- [ ] Reference switch (toolbar combo) re-bases all deltas

### 6. Telemetry agent (auto-import)

```powershell
# Enable disk telemetry in iRacing: Alt+L or app.ini irsdkEnableDisk=1

# Run agent watching the telemetry folder
iracing-agent --telemetry-dir "C:\Users\<you>\Documents\iRacing\telemetry" --once --import-existing --settle 0 --no-recording
```

**Expected:** Imports the best clean lap from each session file into the library.

**Check:**
- [ ] Agent finds .ibt files
- [ ] Laps imported into `~/.iracing_analysis/library/`
- [ ] Re-running imports 0 new laps (idempotent)
- [ ] Imported laps visible in GUI Library tab

### 7. Live sim connection (iRacing must be running)

#### 7a. Live telemetry check

```powershell
iracing-overlay --watch
```

**Expected:** Prints speed, lap number, lap % once per second while in the car.

**Check:**
- [ ] Connects to sim (no "not running" error)
- [ ] Speed values match the in-sim speedometer
- [ ] Lap counter increments at S/F

#### 7b. Race engineer (replay first, then live)

```powershell
# Replay mode (safe, no sim needed)
iracing-engineer --replay "C:\Users\<you>\Documents\iRacing\telemetry\<car>\<file>.ibt" --language en

# Live mode (sim running, car on track)
iracing-engineer --language en
```

**Expected replay:** One engineer message per lap with delta, top corners, "Focus on TX".

**Expected live:** Same, but after each real lap. Reference auto-selected from library.

**Check:**
- [ ] Replay prints messages for each flying lap
- [ ] Live connects and waits for first lap completion
- [ ] Messages show correct corner names and reasonable deltas
- [ ] Language matches --language flag (pl/en)

#### 7c. Live overlay (sim running)

```powershell
# First load a reference lap into the library (if not already done)
iracing-agent --once --import-existing --settle 0 --no-recording

# Then run overlay
iracing-overlay
```

**Expected:** Translucent overlay window appears on top of the sim with delta bar, input trace, gear hint, braking-zone audio cues.

**Check:**
- [ ] Window renders on top of iRacing (frameless, translucent)
- [ ] Delta bar updates in real-time (positive = slower than reference)
- [ ] Input trace shows your throttle/brake over the reference's
- [ ] Gear hint turns green (shift up) / red (shift down)
- [ ] Audio tones play before braking zones (3 rising pitches)
- [ ] Click and drag to reposition the overlay

### 8. AI coaching (requires LLM endpoint)

#### 8a. With local Ollama (recommended for PC test)

```powershell
# Install Ollama: https://ollama.com
ollama pull gemma4:e4b
# OR use the fine-tuned model:
# Copy racing-coach-v5 GGUF → create with ollama

iracing-analysis
# Settings → AI insights → Enable, Base URL: http://localhost:11434/v1, Model: gemma4:e4b
# Open Pit Wall → AI Report tab → Generate
```

#### 8b. With remote DGX

```powershell
iracing-analysis
# Settings → AI insights → Enable, Base URL: http://192.168.1.190:8000/v1, Model: gemma-4-26b-a4b
```

**Check:**
- [ ] AI Report generates without errors
- [ ] Report is in the correct language (matches lap analysis language)
- [ ] Report references actual corners from the analysis
- [ ] Numbers in the report match the telemetry data

### 9. Voice output (optional)

```powershell
# System TTS
iracing-engineer --tts --tts-engine system --replay <file.ibt>

# Piper TTS (better quality)
pip install piper-tts
python -m piper.download_voices pl_PL-darkman-medium --data-dir %USERPROFILE%\.iracing_analysis\voices
iracing-engineer --tts --replay <file.ibt>
```

**Check:**
- [ ] System TTS speaks the engineer messages
- [ ] Piper TTS speaks with the Polish neural voice
- [ ] Speech doesn't block the telemetry loop (async)

### 10. PyInstaller bundle (if testing Option B)

```powershell
cd packaging\dist\iracing-suite

# Help
.\iracing-suite.exe help

# Diagnose
.\iracing-suite.exe diagnose "C:\path\to\file.ibt"

# Agent
.\iracing-suite.exe agent --once --import-existing --settle 0 --no-recording

# GUI
.\IRacingAnalysis.exe
```

**Check:**
- [ ] All subcommands work from the frozen exe
- [ ] GUI launches and renders correctly
- [ ] No missing DLL errors

## Known Issues / Expected Failures

1. **`pyirsdk` shared memory:** Only works on Windows (macOS/Linux: test_file mode only)
2. **Qt platform:** If GUI fails with platform error, set `QT_QPA_PLATFORM=windows`
3. **Audio cues:** May require a default audio device; headless servers will fail
4. **Firewall:** Windows Defender may prompt when first running network features (G61 API, LLM endpoint)
5. **Long paths:** Some .ibt filenames with Unicode driver names may hit Windows path limits — use short telemetry dir paths

## Reporting Results

After testing, report:
- Windows version
- Python version
- GPU (for overlay performance)
- Which checks passed/failed
- Any error messages (screenshot or copy)
- iRacing build number (for pyirsdk compatibility)
