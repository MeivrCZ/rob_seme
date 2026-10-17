"""Persistent command-line host for CRS93 projects. Run: python main.py"""
from __future__ import annotations

import json
import shlex
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4
import numpy as np

from app.context import Context
from app.robot import RobotManager
from app.camera import CameraManager
from app.store import CalibrationStore
from app.extensions import ExtensionManager
from app.modes import ModeManager

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


def request_move(ctx: Context, target, *, label: str):
    """Operator-side preview/confirmation. Never auto-approve REAL movement."""
    current, goal = ctx.robot.preview_move(target)
    print(f"{label} | mode={ctx.robot.mode}")
    print("  q NOW  [deg]:", np.array2string(np.rad2deg(current), precision=2))
    print("  q GOAL [deg]:", np.array2string(np.rad2deg(goal), precision=2))
    print("  delta  [deg]:", np.array2string(np.rad2deg(goal - current), precision=2))
    if ctx.robot.mode == "virtual":
        ctx.robot.move_to_q(goal)
        print("VIRTUAL joints changed (no motors, no collision simulation).")
        return
    if ctx.robot.mode != "real":
        raise RuntimeError("Connect robot before motion")
    print("WARNING: REAL ROBOT WILL MOVE. No obstacle/collision checks in this app.")
    print("Make sure cage closed, work area clear, Arm Power on, E-STOP accessible.")
    confirmation = input("Physical move approved? Type MOVE: ")
    if confirmation != "MOVE":
        print("Cancelled. No motion command sent.")
        return
    print("Sending move_to_q; terminal waits until motion stops. Emergency: physical E-STOP.")
    ctx.robot.move_to_q(goal, expected_q=current)
    print("Driver reported motion stopped. Verify physical position.")


def dispatch(ctx: Context, manager: ExtensionManager, modes: ModeManager, args: list[str]) -> bool:
    if not args:
        return True
    command = args[0].lower()
    if command in ("quit", "exit"):
        return False
    if command == "help":
        print("""Commands:
  mode status              current program mode, robot/camera backends
  mode virtual [DIR]      simulated CRS93 + optional replay images, no hardware
  mode real nohome        connect REAL CRS93 + Basler, when reference is valid
  mode real home          connect REAL CRS93 + Basler, HOMING MOVES ROBOT
  mode off                disconnect both, loaded extensions/calibrations remain
  robot connect virtual    OFFLINE CRS93 kinematics - zero physical motion
  robot virtual-set deg Q0 Q1 Q2 Q3 Q4 Q5   set virtual state [degrees]
  robot virtual-set rad Q0 Q1 Q2 Q3 Q4 Q5   set virtual state [radians]
  robot jog q0..q5 DELTA_DEG  small relative move of a single joint
  robot goto deg Q0 Q1 Q2 Q3 Q4 Q5  absolute joint target in degrees
  robot goto rad Q0 Q1 Q2 Q3 Q4 Q5  absolute joint target in radians
  robot connect home       initialize + physical hard/soft home (MOVES ROBOT)
  robot connect nohome     initialize without homing (only when reference valid)
  robot status             read joint positions and flange pose
  robot limits             show original ctu_crs limits and effective ranges [deg]
  robot soft-home          reset virtual joints OR physical move (REAL ROBOT MOVES)
  robot release            REAL only: engage brakes + disconnect feedback (confirmation)
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

Extensions included: diagnostics, camera_intrinsics, eye_to_hand, prokop_visualizer
  call prokop_visualizer help
  call camera_intrinsics help
  call eye_to_hand help
""")
    elif command == "mode":
        if args[1:] == ["status"]:
            print(modes.describe())
        elif len(args) in (2, 3) and args[1] == "virtual":
            directory = None
            if len(args) == 3:
                directory = Path(args[2])
                if not directory.is_absolute():
                    directory = ctx.root / directory
            modes.switch_virtual(directory)
            print(modes.describe())
        elif args[1:] in (["real", "home"], ["real", "nohome"]):
            if ctx.robot.mode != "real":
                home = args[2] == "home"
                if home:
                    confirmation = input("HOMING WILL MOVE PHYSICAL ROBOT. Cage clear? Type HOME: ")
                    if confirmation != "HOME":
                        print("Cancelled.")
                        return True
                else:
                    confirmation = input("Controller stayed powered AND reference valid? Type REFERENCE: ")
                    if confirmation != "REFERENCE":
                        print("Cancelled.")
                        return True
            if ctx.robot.mode == "real" and not ctx.robot.released:
                print("Already connected to real CRS93. No new homing will run.")
            modes.switch_real(home=args[2] == "home")
            print(modes.describe())
        elif args[1:] in (["off"], ["disconnect"]):
            modes.disconnect()
            print(modes.describe())
        else:
            print("Try: mode status | mode virtual [DIR] | mode real home|nohome | mode off")
    elif command == "robot":
        if args[1:] == ["connect", "virtual"]:
            if ctx.camera.mode == "real":
                raise RuntimeError("Close the real camera first or use: mode virtual")
            ctx.robot.connect_virtual()
            print("Virtual CRS93 connected - FK/IK model only. No physical hardware touched.")
        elif len(args) == 9 and args[1:3] == ["virtual-set", "deg"]:
            ctx.robot.virtual_set_q(np.deg2rad([float(n) for n in args[3:]]))
            print("Virtual joint configuration set (degrees).")
        elif len(args) == 9 and args[1:3] == ["virtual-set", "rad"]:
            ctx.robot.virtual_set_q([float(n) for n in args[3:]])
            print("Virtual joint configuration set (radians).")
        elif len(args) == 4 and args[1] == "jog":
            joint = args[2].lower()
            if len(joint) != 2 or joint[0] != "q" or joint[1] not in "012345":
                raise ValueError("Joint must be q0, q1, q2, q3, q4 or q5")
            delta_deg = float(args[3])
            if not np.isfinite(delta_deg) or delta_deg == 0:
                raise ValueError("DELTA_DEG must be finite and nonzero")
            # Actual current configuration, not q_home. Preview + revalidation
            # prevents unintended incremental moves after mode changes.
            q = ctx.robot.get_q()
            target = q.copy()
            target[int(joint[1])] += np.deg2rad(delta_deg)
            request_move(ctx, target, label=f"jog {joint} {delta_deg:g} deg")
        elif len(args) == 9 and args[1] == "goto" and args[2] in ("deg", "rad"):
            q = np.asarray([float(x) for x in args[3:]], dtype=float)
            target = np.deg2rad(q) if args[2] == "deg" else q
            request_move(ctx, target, label="goto " + args[2])
        elif args[1:] == ["connect", "home"]:
            ans = input("Homing WILL MOVE the robot. Cage clear, permitted to operate? Type HOME: ")
            if ans == "HOME":
                if ctx.camera.mode == "replay":
                    raise RuntimeError("Replay camera active; use: mode real home")
                ctx.robot.connect(home=True)
                print("Robot connected; hard/soft home complete.")
            else:
                print("Cancelled.")
        elif args[1:] == ["connect", "nohome"]:
            ans = input("Use only if controller stayed powered and joints remain referenced. Type REFERENCE: ")
            if ans == "REFERENCE":
                if ctx.camera.mode == "replay":
                    raise RuntimeError("Replay camera active; use: mode real nohome")
                ctx.robot.connect(home=False)
                print("Robot connected without homing (reference validity is your responsibility).")
            else:
                print("Cancelled.")
        elif args[1:] == ["release"]:
            if ctx.robot.mode != "real":
                raise RuntimeError("robot release requires a connected REAL CRS93. VIRTUAL mode is unaffected.")
            if ctx.robot.released:
                raise RuntimeError("Robot has already been released.")
            # First check motion while feedback is still active; fail closed.
            ctx.robot.assert_stationary()
            print("WARNING: This engages brakes and disconnects feedback; the arm may DROP slightly.")
            print("Use only with stopped robot, enough clearance, and in accordance with station procedure.")
            print("After release, built-in motion commands are blocked until a new connection.")
            ans = input("Robot stopped, safe clearance, authorized? Type RELEASE: ")
            if ans == "RELEASE":
                # Check again immediately before the command.
                ctx.robot.release()
                print("robot.release() completed: brakes engaged, feedback disconnected.")
            else:
                print("Cancelled. No release command sent.")
        elif args[1:] == ["disconnect"]:
            ctx.robot.assert_stationary()
            ctx.robot.close()
            print("Robot disconnected.")
        elif args[1:] == ["limits"]:
            raw, effective = ctx.robot.get_joint_limits_deg()
            if raw is None or effective is None:
                print(f"No usable q_min/q_max from ctu_crs: {ctx.robot._limits_error}")
            else:
                print("Raw ctu_crs joint limits [deg]:")
                print(np.array2string(raw, precision=2, suppress_small=True))
                print("Effective software limits after margin/optional restriction [deg]:")
                print(np.array2string(effective, precision=2, suppress_small=True))
                print(f"Real motion enabled in config: {ctx.robot.real_motion_enabled}")
                print("Software limits are NOT collision avoidance or safety-rated protection.")
        elif args[1:] == ["status"]:
            if ctx.robot.released:
                print("mode: real; release requested (feedback may be disconnected).")
                print("Joint positions / flange pose are not reliable. Re-enable station and reconnect before reading.")
                return True
            q = ctx.robot.get_q()
            print("mode:", ctx.robot.mode)
            print("q [rad]:", np.array2string(q, precision=4))
            print("T_base_flange:\n", ctx.robot.flange_pose())
        elif args[1:] == ["soft-home"]:
            if ctx.robot.released:
                raise RuntimeError("Robot released: restore arm power and reinitialize the connection before motion.")
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
            print("Try: robot jog q0 2 | robot goto deg Q0..Q5 | robot status | robot release | robot soft-home")
    elif command == "camera":
        if args[1:] == ["open"]:
            if ctx.robot.mode == "virtual":
                raise RuntimeError("In VIRTUAL mode use replay, or switch via: mode real nohome|home")
            ctx.camera.open()
            print(f"Camera opened: {ctx.camera.name}")
        elif args[1:] == ["close"]:
            ctx.camera.close()
            print("Camera closed.")
        elif len(args) == 3 and args[1] == "replay":
            if ctx.robot.mode == "real":
                raise RuntimeError("Real robot active; use: mode virtual DIRECTORY")
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
        robot=RobotManager(SETTINGS["robot"]["tty_dev"], **SETTINGS.get("motion", {})),
        camera=CameraManager(SETTINGS["camera"]["name"]),
        calibration=CalibrationStore(ROOT / "data" / "calibration"),
    )
    manager = ExtensionManager(ctx)
    modes = ModeManager(ctx)
    print("CRS93 host ready in DISCONNECTED mode. Enter 'mode virtual' or 'help'.")
    try:
        while True:
            try:
                line = input("crs93> ")
            except EOFError:
                break
            try:
                if not dispatch(ctx, manager, modes, shlex.split(line)):
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
