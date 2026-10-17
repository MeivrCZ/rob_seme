"""Hot-load extensions from ``extensions/`` without restarting the hardware.

Only files exposing a module-level ``setup(ctx)`` are treated as extensions;
helper modules (such as prokop_bridge_runner.py) are not loaded by ``load all``.
"""
from __future__ import annotations

import ast
import importlib
from pathlib import Path
import sys
from typing import Any


class ExtensionManager:
    def __init__(self, context: Any):
        self.context = context
        self.instances: dict[str, Any] = {}

    def available(self) -> list[str]:
        """Discover loadable extensions *without importing/executing* their code."""
        directory = Path(self.context.root) / "extensions"
        if not directory.is_dir():
            return []
        names = []
        for path in sorted(directory.glob("*.py")):
            if not path.stem.isidentifier() or path.stem.startswith("_"):
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (SyntaxError, OSError, UnicodeError) as exc:
                print(f"[extension] Cannot inspect {path.name}: {exc}")
                continue
            if any(isinstance(node, ast.FunctionDef) and node.name == "setup"
                   for node in tree.body):
                names.append(path.stem)
        return names

    def load_all(self) -> dict:
        """Load each not-yet-loaded extension, reporting failures individually.

        Setup must not connect hardware or move the robot. Running a module's
        command (including a viewer start) remains an explicit separate step.
        """
        names = self.available()
        loaded, skipped, failed = [], [], {}
        for name in names:
            if name in self.instances:
                skipped.append(name)
                continue
            try:
                self.load(name)
                loaded.append(name)
            except Exception as exc:
                failed[name] = f"{type(exc).__name__}: {exc}"
                print(f"[extension] FAILED to load {name}: {failed[name]}")
        print(f"[extension] load all: loaded {len(loaded)}, already loaded {len(skipped)}, "
              f"failed {len(failed)}.")
        if not names:
            print("[extension] No extensions with setup(ctx) found in extensions/.")
        return {"loaded": loaded, "skipped": skipped, "failed": failed}

    def load(self, name: str) -> None:
        if not name.isidentifier() or name.startswith("_"):
            raise ValueError("Invalid extension name.")
        if name in self.instances:
            raise RuntimeError(f"Extension '{name}' is already loaded.")
        module_name = f"extensions.{name}"
        importlib.invalidate_caches()
        if module_name in sys.modules:
            module = importlib.reload(sys.modules[module_name])
        else:
            module = importlib.import_module(module_name)
        # setup must not reinitialize the robot or reopen the camera.
        instance = module.setup(self.context)
        self.instances[name] = instance
        print(f"[extension] loaded: {name}")

    def unload(self, name: str) -> None:
        if name not in self.instances:
            raise RuntimeError(f"Extension '{name}' is not loaded.")
        instance = self.instances[name]
        close = getattr(instance, "close", None)
        if callable(close):
            close()  # stop extension-owned workers before forgetting it
        del self.instances[name]
        print(f"[extension] unloaded: {name}")

    def reload(self, name: str) -> None:
        if name in self.instances:
            self.unload(name)
        self.load(name)

    def call(self, name: str, arguments: list[str]) -> None:
        if name not in self.instances:
            raise RuntimeError(f"Extension '{name}' is not loaded.")
        self.instances[name].run(arguments)

    def close(self) -> None:
        for name in list(reversed(self.instances)):
            try:
                self.unload(name)
            except Exception as exc:
                print(f"[warning] Couldn't unload {name}: {exc}")
