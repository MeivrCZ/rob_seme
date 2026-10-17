"""Switch robot + camera backends without replacing Context or loaded extensions.

Switches do not perform any robot move unless the operator explicitly requests
REAL with `home=True`. A running real robot must be stationary to disconnect.
"""
from __future__ import annotations

from pathlib import Path
from app.context import Context


class ModeManager:
    def __init__(self, context: Context):
        self.ctx = context
        self._last_virtual_q = None
        self._last_replay_dir: Path | None = None

    @property
    def mode(self) -> str:
        robot = self.ctx.robot.mode
        camera = self.ctx.camera.mode
        if robot == "virtual" and camera in ("replay", None):
            return "VIRTUAL"
        if robot == "real" and camera == "real":
            return "REAL"
        if robot is None and camera is None:
            return "DISCONNECTED"
        return "PARTIAL"  # e.g. camera-only calibration, or manually closed camera

    def describe(self) -> str:
        return (f"mode: {self.mode}\n"
                f"robot: {self.ctx.robot.mode or 'disconnected'}\n"
                f"robot feedback: {'release attempted - state UNKNOWN' if self.ctx.robot.released and not self.ctx.robot.release_confirmed else 'release requested (API completed)' if self.ctx.robot.released else 'not released by this app'}\n"
                f"camera: {self.ctx.camera.mode or 'closed'}\n"
                f"replay directory: {self.ctx.camera.replay_directory or '(none)'}\n"
                f"calibrations stored: {len(self.ctx.calibration.names())}")

    def _remember_virtual(self) -> None:
        if self.ctx.robot.mode == "virtual":
            self._last_virtual_q = self.ctx.robot.get_q().copy()
        if self.ctx.camera.mode == "replay":
            self._last_replay_dir = self.ctx.camera.replay_directory

    def _restore_virtual(self, replay_dir: Path | None) -> None:
        self.ctx.robot.connect_virtual()
        if self._last_virtual_q is not None:
            self.ctx.robot.virtual_set_q(self._last_virtual_q)
        if replay_dir is not None:
            self.ctx.camera.open_replay(replay_dir)

    def switch_virtual(self, replay_dir: Path | None = None) -> None:
        """Switch to kinematics-only robot, optional recorded camera images.

        Does not open real hardware, perform homing, or discard calibrations.
        """
        if replay_dir is not None:
            replay_dir = Path(replay_dir).resolve()
            # Validate before stopping any current backend.
            self.ctx.camera.validate_replay_directory(replay_dir)
            self._last_replay_dir = replay_dir
        elif self.ctx.camera.mode == "replay":
            replay_dir = self.ctx.camera.replay_directory
        else:
            replay_dir = self._last_replay_dir
            if replay_dir is not None:
                try:
                    self.ctx.camera.validate_replay_directory(replay_dir)
                except ValueError:
                    replay_dir = None  # stored directory has disappeared

        if self.ctx.robot.mode == "real":
            self.ctx.robot.assert_stationary()

        if self.ctx.robot.mode == "virtual":
            self._remember_virtual()
            if self.ctx.camera.mode == "replay" and self.ctx.camera.replay_directory == replay_dir:
                return
            if self.ctx.camera.connected:
                self.ctx.camera.close()
            if replay_dir is not None:
                self.ctx.camera.open_replay(replay_dir)
            return

        if self.ctx.camera.connected:
            self.ctx.camera.close()
        if self.ctx.robot.connected:
            self.ctx.robot.close()
        self._restore_virtual(replay_dir)

    def switch_real(self, *, home: bool) -> None:
        """Open Basler, then initialize physical CRS93 (only if not already real).

        `home` must be confirmed by the CLI operator. Re-entering REAL never
        runs an additional homing. On failure, attempts to restore VIRTUAL.
        """
        if self.ctx.robot.mode == "real":
            if self.ctx.robot.released:
                raise RuntimeError("Robot was released: close/reinitialize after restoring the station; cannot re-arm via mode real while connected.")
            if self.ctx.camera.mode == "replay":
                self.ctx.camera.close()
            self.ctx.camera.open()
            return

        previous_virtual = self.ctx.robot.mode == "virtual"
        self._remember_virtual()
        replay = self._last_replay_dir if previous_virtual else None
        camera_was_real = self.ctx.camera.mode == "real"

        # Check Basler before closing virtual robot or moving the real robot.
        if self.ctx.camera.mode == "replay":
            self.ctx.camera.close()
        try:
            self.ctx.camera.open()
        except Exception:
            if previous_virtual and replay is not None:
                self.ctx.camera.open_replay(replay)
            raise

        if previous_virtual:
            self.ctx.robot.close()
        try:
            self.ctx.robot.connect(home=home)
        except Exception:
            if not camera_was_real:
                self.ctx.camera.close()
            if previous_virtual:
                try:
                    self._restore_virtual(replay)
                except Exception as restore_err:
                    print(f"[warning] Could not restore virtual backend: {restore_err}")
            raise

    def disconnect(self) -> None:
        """Release connections without commanding motion or erasing calibration."""
        if self.ctx.robot.mode == "real":
            self.ctx.robot.assert_stationary()
        self._remember_virtual()
        self.ctx.camera.close()
        self.ctx.robot.close()
