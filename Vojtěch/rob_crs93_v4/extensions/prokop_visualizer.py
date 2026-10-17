"""Read-only live CRS93 joint-state bridge to Prokop's existing Viser scene.

Runs Prokop/view_models.py without modifying its source. A child process inserts
one callback after Prokop's `set_q` definition in memory. Localhost UDP carries
only six joint positions in radians. This extension NEVER sends robot commands.
"""
from __future__ import annotations

import json
import math
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

DEFAULT_PORT = 8080
DEFAULT_SYNC_HZ = 5.0


class ProkopVisualizer:
    def __init__(self, context):
        self.ctx = context
        self.process: subprocess.Popen | None = None
        self.port: int | None = None
        self.sync_port: int | None = None
        self._log_file = None
        self._log_path: Path | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_sample: list[float] | None = None
        self._last_sent_at: float | None = None
        self._sync_error: str | None = None
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()

    def _locate_viewer(self) -> tuple[Path, Path]:
        host = Path(self.ctx.root).resolve()
        for repo_root in (host, *host.parents):
            viewer = repo_root / 'Prokop' / 'view_models.py'
            if viewer.is_file():
                return repo_root, viewer
        raise FileNotFoundError('Missing Prokop/view_models.py above project directory')

    def _running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, port: int = DEFAULT_PORT, *, auto_sync: bool = True) -> None:
        if not 1 <= port <= 65534:
            raise ValueError('Viewer port must be 1..65534 (sync uses port + 1)')
        if self._running():
            raise RuntimeError(f'Viewer already running at http://localhost:{self.port}')
        repo_root, viewer = self._locate_viewer()
        runner = Path(__file__).with_name('prokop_bridge_runner.py')
        if not runner.is_file():
            raise RuntimeError('Missing prokop_bridge_runner.py; copy both extension files')
        logs = Path(self.ctx.root) / 'data' / 'logs'
        logs.mkdir(parents=True, exist_ok=True)
        self._log_path = logs / 'prokop_visualizer.log'
        self._log_file = self._log_path.open('a', encoding='utf-8')
        try:
            self.process = subprocess.Popen(
                [sys.executable, '-u', str(runner), '--viewer', str(viewer),
                 '--port', str(port), '--sync-port', str(port + 1)],
                cwd=str(repo_root), stdout=self._log_file, stderr=subprocess.STDOUT,
            )
            self.port = port
            self.sync_port = port + 1
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                self.process.wait(timeout=0.25)
            except subprocess.TimeoutExpired:
                pass
            else:
                code = self.process.returncode
                self.stop()
                raise RuntimeError(f'Viewer exited with code {code}. Read log: {self._log_path}')
            print(f'Prokop Viser started (pid {self.process.pid}): http://localhost:{port}')
            print(f'Viewer log: {self._log_path}')
            if auto_sync and getattr(self.ctx.robot, 'connected', False):
                self.sync_on()
        except Exception:
            self.stop()
            raise

    def _send_one(self):
        robot = self.ctx.robot
        if not robot.connected:
            raise RuntimeError('Robot disconnected (choose mode virtual or connect real robot).')
        if getattr(robot, 'released', False):
            raise RuntimeError('Robot released: joint feedback disabled, cannot synchronize.')
        q = [float(v) for v in robot.get_q()]
        if len(q) != 6 or not all(math.isfinite(v) for v in q):
            raise ValueError('Robot returned invalid 6-joint vector')
        assert self._sock is not None and self.sync_port is not None
        packet = json.dumps({'q': q}, separators=(',', ':')).encode('utf-8')
        self._sock.sendto(packet, ('127.0.0.1', self.sync_port))
        with self._lock:
            self._last_sample = q
            self._last_sent_at = time.monotonic()
            self._sync_error = None

    def sync_once(self):
        if not self._running():
            raise RuntimeError('Start viewer first: call prokop_visualizer start')
        self._send_one()
        print('6 joint angles sent to viewer (radians).')

    def _sync_loop(self, period: float):
        last_error = None
        while not self._stop_event.is_set() and self._running():
            try:
                self._send_one()
                last_error = None
            except Exception as exc:
                message = f'{type(exc).__name__}: {exc}'
                with self._lock:
                    self._sync_error = message
                if message != last_error:
                    print(f'\n[viewer sync] {message}', flush=True)
                    last_error = message
            self._stop_event.wait(period)

    def sync_on(self, hz: float = DEFAULT_SYNC_HZ):
        if not self._running():
            raise RuntimeError('Start viewer first')
        if not self.ctx.robot.connected:
            raise RuntimeError('First connect robot: mode virtual OR mode real nohome')
        if getattr(self.ctx.robot, 'released', False):
            raise RuntimeError('Robot released: cannot sync joint feedback')
        if not math.isfinite(hz) or not 0.2 <= hz <= 10.0:
            raise ValueError('Sync rate must be between 0.2 and 10 Hz')
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError('Sync already running; use sync off first')
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sync_loop,
                                         args=(1.0 / hz,), daemon=True,
                                         name='crs93-viewer-sync')
        self._thread.start()
        print(f'Live joint sync enabled at {hz:g} Hz (read-only; no robot motion).')

    def sync_off(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                print('Warning: sync reader is still waiting on robot I/O; it will stop when I/O returns.')
            else:
                self._thread = None
        print('Live joint sync disabled.')

    def stop(self):
        self.sync_off()
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=3.0)
            self.process = None
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
        self.port = None
        self.sync_port = None

    def status(self):
        state = (f'http://localhost:{self.port}' if self._running() else 'stopped')
        sync = self._thread is not None and self._thread.is_alive() and not self._stop_event.is_set()
        with self._lock:
            sample, error = self._last_sample, self._sync_error
        print(f'Viewer: {state}; sync: {"ON" if sync else "OFF"}; robot mode: {self.ctx.robot.mode}')
        if sample is not None:
            print('Last sent q [rad]:', ' '.join(f'{v:.4f}' for v in sample))
        if error:
            print('Last sync error:', error)
        if self._log_path:
            print('Log:', self._log_path)

    def run(self, argv: list[str]):
        cmd = argv[0].lower() if argv else 'status'
        if cmd == 'start' and len(argv) <= 2:
            self.start(int(argv[1]) if len(argv) == 2 else DEFAULT_PORT)
        elif cmd == 'stop' and len(argv) == 1:
            self.stop()
        elif cmd == 'status' and len(argv) == 1:
            self.status()
        elif cmd == 'sync' and len(argv) in (2, 3):
            action = argv[1].lower()
            if action == 'on':
                self.sync_on(float(argv[2]) if len(argv) == 3 else DEFAULT_SYNC_HZ)
            elif action == 'off' and len(argv) == 2:
                self.sync_off()
            elif action == 'once' and len(argv) == 2:
                self.sync_once()
            else:
                raise ValueError('Use: sync on [HZ] | sync off | sync once')
        elif cmd == 'help' and len(argv) == 1:
            print('Usage: call prokop_visualizer start [PORT] | sync on [HZ] | sync off | sync once | status | stop')
        else:
            raise ValueError('Use: start [PORT] | sync on [HZ] | sync off | sync once | status | stop')

    def close(self):
        self.stop()


def setup(ctx):
    return ProkopVisualizer(ctx)
