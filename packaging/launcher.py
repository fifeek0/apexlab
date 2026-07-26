"""Single console entry point for the frozen (PyInstaller) distribution.

Usage: ``iracing-suite <command> [args...]`` where command is one of
``analysis`` (GUI), ``engineer``, ``overlay``, ``agent``, ``harvest``,
``diagnose``. Keeping one binary keeps the installer small — every command
shares the same bundled runtime.
"""

from __future__ import annotations

import sys

COMMANDS = {
    "analysis": ("iracing_analysis.__main__", "post-session analysis GUI"),
    "engineer": ("iracing_analysis.engineer", "live race engineer (radio updates)"),
    "harvest": ("iracing_analysis.harvest", "bulk-import reference laps from Garage 61"),
    "overlay": ("iracing_overlay.__main__", "Bloops-style live comparison overlay"),
    "agent": ("iracing_core.watcher", "auto-import finished sessions to the library"),
    "diagnose": ("iracing_core.diagnose", "validate a .ibt telemetry file"),
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print("iracing-suite — free, self-hosted iRacing telemetry toolkit\n")
        print("usage: iracing-suite <command> [options]\n\ncommands:")
        for name, (_, desc) in COMMANDS.items():
            print(f"  {name:<10} {desc}")
        print("\nrun 'iracing-suite <command> --help' for command options")
        return 0

    command = sys.argv[1]
    if command not in COMMANDS:
        print(f"unknown command {command!r}; run 'iracing-suite help'")
        return 2

    module_name, _ = COMMANDS[command]
    import importlib

    module = importlib.import_module(module_name)
    sys.argv = [f"iracing-suite {command}", *sys.argv[2:]]
    return int(module.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
