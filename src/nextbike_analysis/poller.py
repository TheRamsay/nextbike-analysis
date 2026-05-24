from __future__ import annotations

import os
from pathlib import Path


def pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_pid(pid_path: Path) -> int | None:
    if not pid_path.exists():
        return None
    try:
        return int(pid_path.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def poller_paths(data_dir: Path) -> tuple[Path, Path]:
    return data_dir / "poller.pid", data_dir / "poller.log"

