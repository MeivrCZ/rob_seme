"""Persistent command-line host for CRS93 projects. Run: python main.py"""
from __future__ import annotations

import json
import shlex
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4

from app.context import Context
from app.robot import RobotManager
from app.camera import CameraManager
from app.store import CalibrationStore
from app.extensions import ExtensionManager

ROOT = Path(__file__).resolve().parent
SETTINGS = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def write_snapshot(ctx: Context):
    import cv2
    path = ROOT / "data" / "captures" / "snapshots"
    path.mkdir(parents=True, exist_ok=True)
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ_") + uuid4().hex[:8] + ".png"
    target = path / name
    image = ctx.camera.capture()
    if not cv2.imwrite(str(target), image):
        raise RuntimeError(f"Couldn't save snapshot: {target}")
    print(f"Snapshot: {target} | resolution: {image.shape}")


def dispatch(ctx: Context, manager: ExtensionManager, args: list[str]) -> bool:
    if not args:
        return True
    command = args[0].lower()
    if command in ("quit", "exit"):
        return False
    if command == "help":
        print("""Commands:
  robot connect virtual    OFFLINE CRS93 kinematics - zero physical motion
  robot virtual-set deg Q0 Q1 Q2 Q3 Q4 Q5   set virtual state [degrees]
  robot virtual-set rad Q0 Q1 Q2 Q3 Q4 Q5   set virtual state [radians]
  robot connect home       initialize + physical hard/soft home (MOVES ROBOT)
  robot connect nohome     initialize without homing (only when reference valid)
  robot status             read joint positions and flange pose
  robot soft-home          reset virtual joints OR physical move (REAL ROBOT MOVES)
  robot disconnect         close robot connection, does not move robot
  camera open|close        acquire/release Basler camera
  camera replay DIR        use a directory of recorded images (no hardware)
  camera status            real/replay camera status
  camera snap              save raw snapshot
  load NAME                load extension without restarting hardware
  unload NAME              unload extension, hardware stays connected
  reload NAME              re-read edited extension source code
  loaded                   list loaded extensions
  call NAME [args...]      run command inside extension
  calib list               list persistent saved calibrations
  calib show NAME          show calibration data
  help                     show this help
  quit                     close connections and exit

Extensions included: diagnostics, camera_intrinsics, eye_to_hand
  call camera_intrinsics help
  call eye_to_hand help
""")
    elif command == "robot":
        if args[1:] == ["connect", "virtual"]:
            ctx.robot.connect_virtual()
            print("Virtual CRS93 connected - FK/IK model only. No physical hardware touched.")
        elif len(args) == 9 and args[1:3] == ["virtual-set", "deg"]:
            import numpy as np
            ctx.robot.virtual_set_q(np.deg2rad([float(n) for n in args[3:]]))
            print("Virtual joint configuration set (degrees).")
        elif len(args) == 9 and args[1:3] == ["virtual-set", "rad"]:
            ctx.robot.virtual_set_q([float(n) for n in args[3:]])
            print("Virtual joint configuration set (radians).")
        elif args[1:] == ["connect", "home"]:
            ans = input("Homing WILL MOVE the robot. Cage clear, permitted to operate? Type HOME: ")
            if ans == "HOME":
                ctx.robot.connect(home=True)
                print("Robot connected; hard/soft home complete.")
            else:
                print("Cancelled.")
        elif args[1:] == ["connect", "nohome"]:
            ans = input("Use only if controller stayed powered and joints remain referenced. Type REFERENCE: ")
            if ans == "REFERENCE":
                ctx.robot.connect(home=False)
                print("Robot connected without homing (reference validity is your responsibility).")
            else:
                print("Cancelled.")
        elif args[1:] == ["disconnect"]:
            ctx.robot.close()
            print("Robot disconnected.")
        elif args[1:] == ["status"]:
            q = ctx.robot.get_q()
            import numpy as np
            print("mode:", ctx.robot.mode)
            print("q [rad]:", np.array2string(q, precision=4))
            print("T_base_flange:\n", ctx.robot.flange_pose())
        elif args[1:] == ["soft-home"]:
            if ctx.robot.mode == "virtual":
                ctx.robot.soft_home()
                print("Virtual state reset to q_home. No physical motion.")
            else:
                ans = input("Robot WILL MOVE to home. Type MOVE: ")
                if ans == "MOVE":
                    ctx.robot.soft_home()
                    print("Reached soft home.")
                else:
                    print("Cancelled.")
        else:
            print("Try: robot connect virtual|home|nohome, robot virtual-set deg|rad Q0..Q5, robot status|soft-home|disconnect")
    elif command == "camera":
        if args[1:] == ["open"]:
            ctx.camera.open()
            print(f"Camera opened: {ctx.camera.name}")
        elif args[1:] == ["close"]:
            ctx.camera.close()
            print("Camera closed.")
        elif len(args) == 3 and args[1] == "replay":
            directory = Path(args[2])
            if not directory.is_absolute():
                directory = ctx.root / directory
            ctx.camera.open_replay(directory)
            print(f"Replay camera: {directory}")
        elif args[1:] == ["status"]:
            print(f"camera mode: {ctx.camera.mode or 'closed'}")
        elif args[1:] == ["snap"]:
            write_snapshot(ctx)
        else:
            print("Try: camera open|close|snap|status or camera replay DIRECTORY")
    elif command in ("load", "unload", "reload") and len(args) == 2:
        getattr(manager, command)(args[1])
    elif command == "loaded":
        print("Loaded:", ", ".join(manager.instances) or "(none)")
    elif command == "call" and len(args) >= 2:
        manager.call(args[1], args[2:])
    elif args == ["calib", "list"]:
        print("Saved calibrations:", ", ".join(ctx.calibration.names()) or "(none)")
    elif len(args) == 3 and args[:2] == ["calib", "show"]:
        data = ctx.calibration.load(args[2])
        print(json.dumps(data, indent=2, ensure_ascii=False) if data else "No saved calibration.")
    else:
        print("Unknown command. Enter: help")
    return True


def main() -> None:
    ctx = Context(
        root=ROOT,
        robot=RobotManager(SETTINGS["robot"]["tty_dev"]),
        camera=CameraManager(SETTINGS["camera"]["name"]),
        calibration=CalibrationStore(ROOT / "data" / "calibration"),
    )
    manager = ExtensionManager(ctx)
    print("CRS93 host ready. Hardware is NOT initialized. Enter 'help'.")
    try:
        while True:
            try:
                line = input("crs93> ")
            except EOFError:
                break
            try:
                if not dispatch(ctx, manager, shlex.split(line)):
                    break
            except (ValueError, RuntimeError, ImportError, OSError) as exc:
                print(f"[error] {type(exc).__name__}: {exc}")
            except KeyboardInterrupt:
                print("\nInterrupted command. Check hardware state; enter 'quit' when safe.")
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        manager.close()
        ctx.camera.close()
        ctx.robot.close()
        print("Connections closed. Calibrations and images remain on disk.")


if __name__ == "__main__":
    main()
