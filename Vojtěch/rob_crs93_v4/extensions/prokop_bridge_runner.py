"""Run Prokop's unmodified viewer with an in-memory hook for set_q(q).

This is an isolated subprocess. The hook injects a localhost-only UDP receiver
just after `def set_q(...)` is defined inside the viewer source; it does not edit
any file owned by Prokop. The UDP protocol is JSON {"q": [6 radians] }.
"""
from __future__ import annotations

import argparse
import ast
import inspect
import json
import math
import socket
import sys
import threading
from pathlib import Path


def attach_joint_listener(set_q, port: int, *, stop_event: threading.Event | None = None):
    """Receive 6 joint angles in radians over 127.0.0.1 UDP and call set_q."""
    signature = inspect.signature(set_q)
    try:
        signature.bind([0.0] * 6)
    except TypeError as e:
        raise RuntimeError("Viewer's set_q must accept a single 6-joint vector") from e

    ready = threading.Event()
    result = {"error": None}

    def worker():
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.bind(("127.0.0.1", port))
            sock.settimeout(0.3)
            print(f"[CRS93 sync] Listening for joint angles at 127.0.0.1:{port}", flush=True)
        except OSError as exc:
            result["error"] = exc
            ready.set()
            sock.close()
            return
        ready.set()
        try:
            while stop_event is None or not stop_event.is_set():
                try:
                    payload, source = sock.recvfrom(4096)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if source[0] != "127.0.0.1":
                    continue
                try:
                    message = json.loads(payload.decode("utf-8"))
                    q = message["q"]
                    if not isinstance(q, list) or len(q) != 6:
                        continue
                    q = [float(v) for v in q]
                    if not all(math.isfinite(v) for v in q):
                        continue
                    # Prokop's set_q() is the only part touching Viser handles.
                    # Nothing in this bridge accesses physical robot hardware.
                    set_q(q)
                except Exception as exc:
                    print(f"[CRS93 sync] Viewer rejected update: {exc}", flush=True)
        finally:
            sock.close()

    thread = threading.Thread(target=worker, daemon=True, name="prokop-joint-receiver")
    thread.start()
    ready.wait(timeout=2)
    if result["error"] is not None:
        raise RuntimeError(f"Could not bind joint sync port {port}") from result["error"]
    if not ready.is_set():
        raise RuntimeError("Joint receiver did not initialize")
    return thread


def instrument_viewer(source: str, filename: str):
    """Compile viewer after hooking exactly one named `set_q` definition.

    The AST is modified in memory only; Prokop's on-disk source stays intact.
    This works even if set_q lives in a with/if block inside main().
    """
    tree = ast.parse(source, filename=filename)
    count = 0
    for node in ast.walk(tree):
        for field, value in ast.iter_fields(node):
            if not isinstance(value, list):
                continue
            new_children = []
            for item in value:
                new_children.append(item)
                if isinstance(item, ast.FunctionDef) and item.name == 'set_q':
                    count += 1
                    new_children.append(ast.Expr(value=ast.Call(
                        func=ast.Name(id='__crs93_attach_listener__', ctx=ast.Load()),
                        args=[ast.Name(id='set_q', ctx=ast.Load())], keywords=[])))
            setattr(node, field, new_children)
    if count != 1:
        raise RuntimeError(
            f"Expected one def set_q in {filename}, found {count}. "
            "Please send view_models.py for adaptation."
        )
    return compile(ast.fix_missing_locations(tree), filename, 'exec')


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--viewer', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--sync-port', type=int, default=8081)
    args = parser.parse_args(argv)
    viewer = args.viewer.resolve()
    if not viewer.is_file():
        raise FileNotFoundError(viewer)
    if not (1 <= args.port <= 65535 and 1 <= args.sync_port <= 65535):
        raise ValueError('Ports must be in range 1-65535')
    if args.port == args.sync_port:
        raise ValueError('Different ports required for Viser and joint sync')
    source = viewer.read_text(encoding='utf-8')
    code = instrument_viewer(source, str(viewer))
    # Reproduce execution of: python Prokop/view_models.py --port N
    sys.path.insert(0, str(viewer.parent))
    sys.argv = [str(viewer), '--port', str(args.port)]
    globals_ = {
        '__name__': '__main__',
        '__file__': str(viewer),
        '__package__': None,
        '__crs93_attach_listener__': lambda f: attach_joint_listener(f, args.sync_port),
    }
    exec(code, globals_, globals_)


if __name__ == '__main__':
    main()
