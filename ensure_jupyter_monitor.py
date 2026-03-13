from __future__ import annotations

import os
import shutil
import subprocess
import time
import webbrowser
from pathlib import Path
from urllib.parse import quote, urljoin


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "Azalyst_Live_Monitor.ipynb"
LOG_FILE = ROOT / "jupyter_monitor.log"
WAIT_SECONDS = 30


def _iter_running_servers() -> list[dict]:
    try:
        from jupyter_server.serverapp import list_running_servers

        return list(list_running_servers())
    except Exception:
        pass

    try:
        from notebook.notebookapp import list_running_servers

        return list(list_running_servers())
    except Exception:
        return []


def _same_path(left: str | Path, right: str | Path) -> bool:
    try:
        return Path(left).resolve().samefile(Path(right).resolve())
    except Exception:
        return str(Path(left).resolve()).lower() == str(Path(right).resolve()).lower()


def _find_server_for_root() -> dict | None:
    for server in _iter_running_servers():
        root_dir = server.get("root_dir") or server.get("notebook_dir") or ""
        if root_dir and _same_path(root_dir, ROOT):
            return server
    return None


def _build_notebook_url(server: dict) -> str:
    relative_path = NOTEBOOK.relative_to(ROOT).as_posix()
    token = server.get("token") or ""
    query = f"?token={token}" if token else ""
    return urljoin(server["url"], f"notebooks/{quote(relative_path)}") + query


def _find_jupyter_exe() -> str | None:
    local_candidate = (
        Path(os.environ.get("LOCALAPPDATA", ""))
        / "Programs"
        / "Python"
        / "Python311"
        / "Scripts"
        / "jupyter-notebook.exe"
    )
    if local_candidate.exists():
        return str(local_candidate)

    return shutil.which("jupyter-notebook")


def _start_server(jupyter_exe: str) -> None:
    log_handle = LOG_FILE.open("a", encoding="utf-8")
    log_handle.write(f"\n=== Launch {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    log_handle.flush()

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS

    subprocess.Popen(
        [jupyter_exe, "--no-browser", str(NOTEBOOK.name)],
        cwd=str(ROOT),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        creationflags=creationflags,
        close_fds=False,
    )


def main() -> int:
    if not NOTEBOOK.exists():
        print(f"[jupyter] Notebook not found: {NOTEBOOK}")
        return 1

    server = _find_server_for_root()
    if server is None:
        jupyter_exe = _find_jupyter_exe()
        if not jupyter_exe:
            print("[jupyter] jupyter-notebook.exe not found.")
            return 1

        print("[jupyter] Starting notebook server...")
        _start_server(jupyter_exe)

        deadline = time.time() + WAIT_SECONDS
        while time.time() < deadline:
            time.sleep(1)
            server = _find_server_for_root()
            if server is not None:
                break

    if server is None:
        print(f"[jupyter] Notebook server did not come up. See {LOG_FILE.name}.")
        return 1

    url = _build_notebook_url(server)
    print(f"[jupyter] Opening {url}")
    try:
        os.startfile(url)  # type: ignore[attr-defined]
    except Exception:
        webbrowser.open(url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
