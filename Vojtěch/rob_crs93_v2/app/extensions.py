"""Hot-load plugins from the extensions package in the same Python process."""
from __future__ import annotations

import importlib
import sys
from typing import Any


class ExtensionManager:
    def __init__(self, context: Any):
        self.context = context
        self.instances: dict[str, Any] = {}

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
        # 'setup' should not reinitialize the robot or reopen the camera.
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
