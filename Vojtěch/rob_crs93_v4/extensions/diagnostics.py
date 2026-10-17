"""Example plugin: can be edited and reloaded while hardware stays connected."""
import numpy as np


class Diagnostics:
    def __init__(self, context):
        self.ctx = context

    def run(self, argv):
        mode = argv[0] if argv else "status"
        if mode == "status":
            print(f"robot: {self.ctx.robot.mode or 'disconnected'}")
            print(f"camera: {self.ctx.camera.mode or 'closed'}")
            print(f"calibrations: {', '.join(self.ctx.calibration.names()) or '(none)'}")
        elif mode == "joints":
            q = self.ctx.robot.get_q()
            print("q [rad] =", np.array2string(q, precision=5))
            print("q [deg] =", np.array2string(np.rad2deg(q), precision=2))
        elif mode == "pose":
            print("T_base_flange =")
            print(np.array2string(self.ctx.robot.flange_pose(), precision=5))
        else:
            print("Usage: call diagnostics [status|joints|pose]")

    def close(self):
        pass


def setup(ctx):
    return Diagnostics(ctx)
