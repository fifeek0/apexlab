# PyInstaller spec: one shared bundle, two executables.
#
#   IRacingAnalysis   — windowed GUI (analysis app)
#   iracing-suite     — console launcher (engineer / overlay / agent / harvest / diagnose)
#
# Build:  pyinstaller packaging/iracing_suite.spec --noconfirm
# Output: dist/iracing-suite/

import sys

HIDDEN = [
    # dynamically imported bits PyInstaller's static analysis can miss
    "iracing_core.diagnose",
    "iracing_core.watcher",
    "iracing_analysis.engineer",
    "iracing_analysis.harvest",
    "iracing_analysis.export_summaries",
    "iracing_overlay.window",
    "pyqtgraph",
]

# voice/TTS extras stay out of the base bundle (optional pip installs);
# the apps degrade gracefully without them
EXCLUDES = [
    "piper", "pyttsx3", "onnxruntime",
    "tkinter", "matplotlib", "IPython", "jupyter",
]

console = Analysis(
    ["launcher.py"],
    pathex=[],
    hiddenimports=HIDDEN,
    excludes=EXCLUDES,
    noarchive=False,
)
gui = Analysis(
    ["launcher_gui.py"],
    pathex=[],
    hiddenimports=HIDDEN,
    excludes=EXCLUDES,
    noarchive=False,
)

MERGE((console, "iracing-suite", "iracing-suite"), (gui, "IRacingAnalysis", "IRacingAnalysis"))

console_pyz = PYZ(console.pure)
console_exe = EXE(
    console_pyz,
    console.scripts,
    exclude_binaries=True,
    name="iracing-suite",
    console=True,
    icon=None,
)

gui_pyz = PYZ(gui.pure)
gui_exe = EXE(
    gui_pyz,
    gui.scripts,
    exclude_binaries=True,
    name="IRacingAnalysis",
    console=False,
    icon=None,
)

COLLECT(
    console_exe,
    console.binaries,
    console.datas,
    gui_exe,
    gui.binaries,
    gui.datas,
    name="iracing-suite",
)
