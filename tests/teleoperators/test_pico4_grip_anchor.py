"""Pico4 grip re-anchoring: a new grip press restarts from the robot's measured TCP."""

import unittest

import numpy as np

from lerobot.teleoperators.pico4.config_pico4 import Pico4Config
from lerobot.teleoperators.pico4.teleop_pico4 import Pico4


class _FakeXrt:
    def __init__(self):
        self.pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # [x, y, z, qx, qy, qz, qw], Pico4 frame
        self.grip = 0.0

    def get_right_controller_pose(self):
        return list(self.pose)

    def get_right_grip(self):
        return self.grip

    def get_right_trigger(self):
        return 0.0

    def get_A_button(self):  # noqa: N802 - Pico4 SDK method name
        return False

    def get_B_button(self):  # noqa: N802 - Pico4 SDK method name
        return False

    def close(self):
        pass


def _teleop(anchor=True):
    config = Pico4Config(
        use_right_controller=True,
        use_left_controller=False,
        grip_anchor_to_robot=anchor,
        max_pos_velocity=0.0,
        max_rot_velocity=0.0,
    )
    teleop = Pico4(config)
    teleop._xrt = _FakeXrt()
    teleop._is_connected = True
    teleop.reset_to_pose(np.array([0.5, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0]))
    return teleop


def _position(action):
    return np.array([action["tcp.x"], action["tcp.y"], action["tcp.z"]])


class GripAnchorTests(unittest.TestCase):
    def drag_then_regrip(self, teleop, robot_tcp):
        xrt = teleop._xrt
        teleop.set_grip_anchor_source(lambda: robot_tcp)
        xrt.grip = 1.0
        teleop.get_action()
        xrt.pose[0] += 0.2  # the hand keeps going (the arm is held at an obstacle)
        pushed = _position(teleop.get_action())
        xrt.grip = 0.0
        teleop.get_action()
        xrt.grip = 1.0
        return pushed, _position(teleop.get_action())

    def test_regrip_restarts_from_the_robot_tcp(self):
        teleop = _teleop()
        robot_tcp = np.array([0.55, 0.02, 0.31, 1.0, 0.0, 0.0, 0.0, 0.0])
        pushed, regripped = self.drag_then_regrip(teleop, robot_tcp)
        self.assertGreater(np.linalg.norm(pushed - [0.5, 0.0, 0.3]), 0.1)
        np.testing.assert_allclose(regripped, robot_tcp[:3], atol=1e-6)

    def test_without_anchoring_the_last_target_is_kept(self):
        teleop = _teleop(anchor=False)
        pushed, regripped = self.drag_then_regrip(teleop, np.array([0.55, 0.02, 0.31, 1.0, 0.0, 0.0, 0.0]))
        np.testing.assert_allclose(regripped, pushed, atol=1e-6)

    def test_unreadable_robot_keeps_the_last_target(self):
        teleop = _teleop()

        def broken():
            raise RuntimeError("RTSI down")

        xrt = teleop._xrt
        teleop.set_grip_anchor_source(broken)
        xrt.grip = 1.0
        first = _position(teleop.get_action())
        np.testing.assert_allclose(first, [0.5, 0.0, 0.3], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
