"""
Kill stale Azalyst team/simulator processes and remove their lock files.
Leaves checkpoint/log files intact so the next run resumes where dropped.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


LOCK_DIR = Path(".azalyst_locks")
TARGET_LOCKS = [
    LOCK_DIR / "autonomous_team.lock",
    LOCK_DIR / "walkforward_simulator.lock",
]


def _read_lock(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _kill(pid: int) -> None:
    if pid <= 0:
        return
    cmd = ["taskkill", "/PID", str(pid), "/F"]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def main() -> None:
    if not LOCK_DIR.exists():
        return
    for lock in TARGET_LOCKS:
        if not lock.exists():
            continue
        data = _read_lock(lock)
        pid = data.get("pid")
        owner_pid = data.get("owner_pid")
        for candidate in (owner_pid, pid):
            if isinstance(candidate, int):
                _kill(candidate)
            elif isinstance(candidate, str) and candidate.isdigit():
                _kill(int(candidate))
        try:
            lock.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    main()
