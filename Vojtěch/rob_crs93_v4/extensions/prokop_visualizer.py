"""Run Prokop's independent Viser scene from the CRS93 extension manager.

Place this file in the existing host's extensions/ directory. Prokop/ must
remain in the shared Git repository. No edits to Prokop's sources are needed.

Commands:
    load prokop_visualizer
    call prokop_visualizer start [PORT]
    call prokop_visualizer status
    call prokop_visualizer stop
    unload prokop_visualizer

This first bridge STARTS/STOPS the viewer; it DOES NOT synchronize joints,
objects, camera extrinsics or robot motion with our host yet.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

DEFAULT_PORT = 8080


class ProkopVisualizer:
    def __init__(self, context):
        self.ctx = context
        self.process: subprocess.Popen | None = None
        self.port: int | None = None

    def _locate_viewer(self) -> tuple[Path, Path]:
        """Support host at repo root or in a team-member subdirectory."""
        host = Path(self.ctx.root).resolve()
        for repo_root in (host, *host.parents):
            viewer = repo_root / "Prokop" / "view_models.py"
            if viewer.is_file():
                return repo_root, viewer
        raise FileNotFoundError(
            "Can't find Prokop/view_models.py. Run the host inside the "
            "checked-out rob_seme repository (or put the host in a subfolder)."
        )

    def _running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, port: int = DEFAULT_PORT) -> None:
        if not 1 <= port <= 65535:
            raise ValueError("Port must be in the range 1..65535.")
        if self._running():
            raise RuntimeError(
                f"Viewer is already running at http://localhost:{self.port}."
            )
        repo_root, viewer = self._locate_viewer()
        # Use the SAME Python environment as the host (important with uv).
        # stdout/stderr stay attached to the console; no pipes to deadlock.
        process = subprocess.Popen(
            [sys.executable, str(viewer), "--port", str(port)],
            cwd=str(repo_root),
        )
        self.process = process
        self.port = port
        # Report an immediate interpreter/import error rather than saying it ran.
        try:
            process.wait(timeout=0.15)
        except subprocess.TimeoutExpired:
            pass
        else:
            code = process.returncode
            self.process = None
            self.port = None
            raise RuntimeError(
                f"Prokop viewer exited immediately (code {code}). "
                "Check the error above and install repository dependencies."
            )
        print(f"Prokop viewer started (pid {process.pid}).")
        print(f"Open http://localhost:{port} in your browser.")
        print("Note: viewer joints are currently independent of the host robot state.")

    def stop(self) -> None:
        process = self.process
        if process is None:
            print("Prokop viewer is not running.")
            return
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=4)
        finally:
            self.process = None
            self.port = None
        print("Prokop viewer stopped. Robot/camera connections are unchanged.")

    def status(self) -> None:
        if self._running():
            print(f"Prokop viewer running: http://localhost:{self.port}")
        elif self.process is not None:
            print(f"Prokop viewer exited (code {self.process.returncode}).")
        else:
            print("Prokop viewer stopped.")
        print("Joint synchronization: not implemented (standalone viewer).")

    def run(self, argv: list[str]) -> None:
        command = argv[0].lower() if argv else "status"
        if command == "start" and len(argv) <= 2:
            port = int(argv[1]) if len(argv) == 2 else DEFAULT_PORT
            self.start(port)
        elif command == "stop" and len(argv) == 1:
            self.stop()
        elif command == "status" and len(argv) == 1:
            self.status()
        elif command == "help" and len(argv) == 1:
            print("Usage: call prokop_visualizer [start [PORT]|stop|status|help]")
        else:
            raise ValueError(
                "Usage: call prokop_visualizer [start [PORT]|stop|status|help]"
            )

    def close(self) -> None:
        """Called by unload/reload/host shutdown; leave hardware untouched."""
        self.stop()


def setup(ctx):
    return ProkopVisualizer(ctx)
