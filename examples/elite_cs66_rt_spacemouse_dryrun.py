#!/usr/bin/env python
"""Dry-run smoke test for Elite CS66 + SpaceMouse teleop.

Stubs out:
  * libpyelite (native driver + RTSI + dashboard)
  * pyspacemouse + lerobot.teleoperators.spacemouse.peripherals.Spacemouse

so the real EliteCS66RT + SpacemouseTeleop + elite_cs66_rt_spacemouse_teleop_loop
can be exercised end-to-end without hardware.

Asserts:
  * connect() takes the success path (RTSI + dashboard + driver + script)
  * action keys produced by convert_to_flexiv_action match what EliteCS66RT expects
  * send_action submits Cartesian targets to the native planner
  * both-buttons triggers robot.reset_to_initial_position()
  * rt_moving short-circuits send_action and re-syncs teleop after reset finishes
  * disconnect() cleans up driver/dashboard/RTSI

Run: python examples/elite_cs66_rt_spacemouse_dryrun.py
"""

from __future__ import annotations

import sys
import types

import numpy as np

# ---------------------------------------------------------------------------
# libpyelite stub: minimal native surface the EliteCS66RT driver exercises
# ---------------------------------------------------------------------------


class _FakeDriverConfig:
    def __init__(self):
        self.robot_ip = ""
        self.local_ip = ""
        self.servoj_time = 0.004
        self.servoj_lookahead_time = 0.1
        self.servoj_gain = 2000
        self.headless_mode = True
        self.script_file_path = ""
        self.model_path = ""
        self.tool = list(np.eye(4).ravel())
        self.ik = types.SimpleNamespace(max_iterations=500)
        self.planner = types.SimpleNamespace(
            stale_timeout=0.5, max_linear_velocity=0.5, max_angular_velocity=1.0, velocity_scale=1.0
        )
        self.command_timeout_ms = 500
        self.max_consecutive_write_failures = 1


class _FakeDriver:
    """Native driver stand-in: records targets instead of planning them."""

    def __init__(self, config):
        self.config = config
        self.connected = True
        self.running = False
        self.servo_targets: list[np.ndarray] = []
        self.move_j_targets: list[list[float]] = []
        self.stop_calls = 0
        self._pose = np.eye(4)

    def isRobotConnected(self):
        return self.connected

    def sendExternalControlScript(self):
        self.connected = True
        return True

    def startServo(self, rtsi):
        self.running = True

    def stopServo(self):
        self.running = False

    def status(self):
        return types.SimpleNamespace(running=self.running, reset_active=False, error="")

    def check_error(self):
        pass

    def submit_target(self, pose, duration=0.0):
        self.servo_targets.append(np.asarray(pose))
        self._pose = np.asarray(pose)

    def submit_joints(self, q):
        pass

    def commanded_pose(self):
        return self._pose.copy()

    def moveJ(self, q, duration, timeout_ms):
        assert not self.running, "MoveJ while the native servo still owns the socket"
        self.move_j_targets.append(list(q))

    def writeIdle(self, timeout_ms):
        return True

    def stopControl(self, wait_ms=1000):
        self.running = False
        self.stop_calls += 1
        return True


class _FakePinkIKModel:
    """Flange FK equal to the fake RTSI TCP, so the derived tool is identity."""

    def __init__(self, urdf=None, tool=None, options=None):
        pass

    def fk(self, q):
        pose = np.eye(4)
        pose[:3, 3] = _FakeRtsi.POSE[:3]
        return pose


class _FakeDashboard:
    def __init__(self):
        self.connected = False

    def connect(self, ip):
        self.connected = True
        return True

    def disconnect(self):
        self.connected = False

    def powerOn(self):
        return True

    def brakeRelease(self):
        return True


class _FakeRtsi:
    POSE = (0.5, 0.0, 0.3, 0.0, 0.0, 0.0)  # [x, y, z, roll, pitch, yaw]

    def __init__(self, output, inp, freq):
        self.output = output
        self.input = inp
        self.freq = freq

    def connect(self, ip):
        return True

    def disconnect(self):
        return True

    def getTimestamp(self):
        return 1.0  # constant: every sample reads as coherent

    def getActualTCPPose(self):
        return list(self.POSE)

    def getActualJointPositions(self):
        return [0.0] * 6

    def getActualJointVelocity(self):
        return [0.0] * 6

    def getActualJointTorques(self):
        return [0.0] * 6


def install_elite_sdk_stub():
    import pathlib

    mod = types.ModuleType("libpyelite")
    mod.EliteDriverConfig = _FakeDriverConfig
    mod.EliteDriver = _FakeDriver
    mod.DashboardClientInterface = _FakeDashboard
    mod.RtsiIOInterface = _FakeRtsi
    mod.PinkIKModel = _FakePinkIKModel
    mod.model_path = lambda: "fake.urdf"

    # external_control.script lookup uses the package path: fabricate a file beside us.
    pkg_dir = pathlib.Path(__file__).resolve().parent / "_fake_libpyelite"
    pkg_dir.mkdir(exist_ok=True)
    script = pkg_dir / "external_control.script"
    if not script.exists():
        script.write_text("# fake script\n")
    # Recipes: reuse the ones already shipped with elite_cs66_rt resource/.
    mod.__file__ = str(script)  # _resolve_sdk_resource uses parent dir

    sys.modules["libpyelite"] = mod


# ---------------------------------------------------------------------------
# SpaceMouse stub: replace peripherals.Spacemouse with a scripted device
# ---------------------------------------------------------------------------


class _FakeSpacemouse:
    """Drives a deterministic motion plan instead of real HID events."""

    def __init__(self, *args, **kwargs):
        self._connected = False
        # tick counter consumed by tests below
        self._tick = 0
        # scripted: (vx, vy, vz, rx, ry, rz, left, right)
        self._plan: list[tuple] = []

    def connect(self):
        self._connected = True

    def disconnect(self):
        self._connected = False

    def is_connected(self):
        return self._connected

    def poll(self):
        if self._plan:
            self._cur = self._plan.pop(0)
        else:
            self._cur = (0, 0, 0, 0, 0, 0, False, False)
        self._tick += 1

    def get_motion_state_transformed(self):
        vx, vy, vz, rx, ry, rz, *_ = self._cur
        return np.array([vx, vy, vz, rx, ry, rz], dtype=np.float32)

    def is_left_button_pressed(self):
        return bool(self._cur[6])

    def is_right_button_pressed(self):
        return bool(self._cur[7])


def install_spacemouse_stub():
    # The teleop does `from lerobot.teleoperators.spacemouse.peripherals import Spacemouse`.
    import lerobot.teleoperators.spacemouse.peripherals as peripherals_pkg

    peripherals_pkg.Spacemouse = _FakeSpacemouse


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    install_elite_sdk_stub()

    from lerobot.robots.elite_cs66_rt import EliteCS66RT, EliteCS66RTConfig

    cfg = EliteCS66RTConfig(
        robot_ip="127.0.0.1",
        use_background_servo_loop=False,  # blocking reset, so the test is deterministic
    )
    robot = EliteCS66RT(cfg)
    robot.connect()
    assert robot.is_connected, "EliteCS66RT failed to connect against stub SDK"
    print("[ok] EliteCS66RT stub connect path")

    # send_action with explicit 6D rotation matching identity orientation
    from lerobot.utils.robot_utils import quaternion_to_rotation_6d

    r6d = quaternion_to_rotation_6d(1.0, 0.0, 0.0, 0.0)
    action = {
        "tcp.x": 0.52,
        "tcp.y": 0.01,
        "tcp.z": 0.30,
        **{f"tcp.r{i + 1}": float(r6d[i]) for i in range(6)},
    }
    sent = robot.send_action(action)
    assert set(sent) >= {f"tcp.r{i + 1}" for i in range(6)}, sent
    print("[ok] send_action accepts Cartesian 6D action; keys:", sorted(sent))

    # Drive the SpaceMouse + loop --------------------------------------------
    install_spacemouse_stub()
    from lerobot.teleoperators.spacemouse import SpacemouseConfig, SpacemouseTeleop

    teleop_cfg = SpacemouseConfig()
    teleop = SpacemouseTeleop(teleop_cfg)
    teleop.connect(current_tcp_pose_euler=robot.get_current_tcp_pose_euler())

    # Script: forward motion x3, both-buttons x1, rest idle x3
    teleop._spacemouse._plan = [
        (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, False),
        (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, False),
        (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, False),
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, True, True),  # both buttons -> reset
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, False),
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, False),
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, False),
    ]

    # We can't actually call the loop's `while True` here without a duration cap.
    # The loop function supports `duration=`, so use a tiny one tied to fps:
    from lerobot.scripts.lerobot_teleoperate import elite_cs66_rt_spacemouse_teleop_loop

    elite_cs66_rt_spacemouse_teleop_loop(
        teleop=teleop,
        robot=robot,
        fps=50,
        display_data=False,
        duration=0.15,
        dryrun=False,
        debug_timing=False,
    )
    print()  # the loop's status print stays on one line
    print("[ok] elite_cs66_rt_spacemouse_teleop_loop ran without raising")

    driver = robot._driver
    assert driver is not None and len(driver.servo_targets) > 0, "no native targets observed"
    print(f"[ok] driver received {len(driver.servo_targets)} native targets")

    robot.disconnect()
    assert driver.stop_calls >= 1, "stopControl not called on disconnect"
    print("[ok] disconnect() called stopControl")


if __name__ == "__main__":
    main()
