"""
Create a dedicated Spyder profile for the Azalyst live monitor.

The profile is isolated from the user's default Spyder settings so the
launcher can auto-run the local monitor without changing global IDE config.
"""

from __future__ import annotations

from configparser import RawConfigParser
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROFILE_DIR = ROOT / ".spyder_azalyst"
CONFIG_FILE = PROFILE_DIR / "spyder.ini"
MONITOR_FILE = ROOT / "spyder_live_monitor.py"


def build_config() -> RawConfigParser:
    config = RawConfigParser()
    config.optionxform = str
    config["ipython_console"] = {
        "pylab": "True",
        "pylab/autoload": "True",
        "pylab/backend": "auto",
        "startup/use_run_file": "True",
        "startup/run_file": str(MONITOR_FILE),
    }
    return config


def main() -> None:
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    config = build_config()
    with CONFIG_FILE.open("w", encoding="utf-8") as fh:
        config.write(fh)
    print(CONFIG_FILE)


if __name__ == "__main__":
    main()
