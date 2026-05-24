from __future__ import annotations

from pathlib import Path


def bike_risk(num_bikes_available: int) -> str:
    if num_bikes_available <= 0:
        return "empty"
    if num_bikes_available == 1:
        return "high"
    if num_bikes_available <= 3:
        return "medium"
    return "low"


def directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def tail_lines(path: Path, line_count: int = 5) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-line_count:]

