"""Atomic, reusable storage for camera/robot calibration results."""
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


class CalibrationStore:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
            raise ValueError("Calibration name must contain only letters, digits, '-' or '_'.")
        return self.directory / f"{name}.json"

    def save(self, name: str, data: dict[str, Any]) -> Path:
        target = self._path(name)
        # Never leave a partially written calibration if a process fails mid-write.
        tmp_name = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.directory,
                prefix=f".{name}.", suffix=".tmp", delete=False,
            ) as file:
                tmp_name = file.name
                json.dump(data, file, ensure_ascii=False, indent=2)
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(tmp_name, target)
        finally:
            if tmp_name and os.path.exists(tmp_name):
                os.unlink(tmp_name)
        return target

    def load(self, name: str) -> dict[str, Any] | None:
        path = self._path(name)
        if not path.exists():
            return None
        with path.open(encoding="utf-8") as file:
            result = json.load(file)
        if not isinstance(result, dict):
            raise ValueError(f"Invalid calibration: {path}")
        return result

    def names(self) -> list[str]:
        return sorted(p.stem for p in self.directory.glob("*.json"))
