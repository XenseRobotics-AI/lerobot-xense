"""Offline adapter contract: model selection, TCP passthrough, world/base SE(3)."""

import unittest
from importlib.util import find_spec
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

# Skip only an absent SDK; broken installations must still fail collection.
if find_spec("libpyelite") is None:
    pytest.skip("libpyelite SDK not installed", allow_module_level=True)

import libpyelite
from libpyelite.assembly import Assembly

from lerobot.robots.bi_elite_cs66_rt.bi_elite_cs66_rt import BiEliteCS66RT
from lerobot.robots.bi_elite_cs66_rt.config_bi_elite_cs66_rt import BiEliteCS66RTConfig
from lerobot.robots.elite_cs66_rt.config_elite_cs66_rt import EliteCS66RTConfig
from lerobot.robots.elite_cs66_rt.elite_cs66_rt import (
    _configure_native_driver,
    _controller_tcp_to_pose6,
    _pose6_to_matrix,
)


class AssemblyAdapterTests(unittest.TestCase):
    def make_robot(self, name, assembly=True):
        cfg = BiEliteCS66RTConfig(
            native_assembly=name if assembly else None,
            left_use_gripper=False,
            right_use_gripper=False,
            left_world_rotation=np.eye(3).tolist(),
            right_world_rotation=np.eye(3).tolist(),
        )
        with patch("lerobot.robots.bi_elite_cs66_rt.bi_elite_cs66_rt.make_cameras_from_configs", return_value={}):
            robot = BiEliteCS66RT(cfg)
        robot._cs = libpyelite
        return robot

    def test_world_base_transform_and_driver_selection(self):
        for name in ("diagonal-07", "diagonal-08"):
            robot = self.make_robot(name)
            assembly = Assembly(name)
            for side in ("left", "right"):
                tool = np.eye(4)
                tool[:3, 3] = [0.0068, -0.0063, 0.1941]
                setattr(robot.config, side + "_tool_transform", tool.ravel().tolist())
                cfg = robot._make_driver_config(side)
                self.assertEqual(cfg.model_path, assembly.control_model_path(side))
                np.testing.assert_allclose(cfg.tool, tool.ravel())
                pose = np.array([0.31, -0.22, 0.48, 0.3, -0.5, 0.9])
                world = robot._base_pose6_to_world(side, pose)
                np.testing.assert_allclose(
                    _pose6_to_matrix(world), assembly.world_from_base(side) @ _pose6_to_matrix(pose), atol=1e-12
                )
                np.testing.assert_allclose(
                    _pose6_to_matrix(robot._world_pose6_to_base(side, world)), _pose6_to_matrix(pose), atol=1e-12
                )

    def test_legacy_mapping_stays_rotation_only(self):
        robot = self.make_robot("diagonal-08", assembly=False)
        self.assertIsNone(robot._assembly)
        for side in ("left", "right"):
            np.testing.assert_allclose(robot._p_wb[side], np.zeros(3))
            pose = np.array([0.3, -0.1, 0.2, 0.0, 0.0, 0.0])
            np.testing.assert_allclose(robot._base_pose6_to_world(side, pose)[:3], robot._R_wb[side] @ pose[:3])

    def test_conflicting_config_is_rejected(self):
        # Assembly names are owned by libpyelite.assembly, checked when the robot is built.
        with self.assertRaises(ValueError):
            BiEliteCS66RT(BiEliteCS66RTConfig(native_assembly="unknown"))
        with self.assertRaises(ValueError):
            BiEliteCS66RTConfig(native_assembly="diagonal-08", native_model_path="custom.urdf")

    def test_assembly_needs_no_legacy_mount_angles(self):
        config = BiEliteCS66RTConfig(native_assembly="diagonal-08", left_use_gripper=False, right_use_gripper=False)
        robot = BiEliteCS66RT(config)
        self.assertIsNotNone(robot._assembly)

    def test_recipe_hardware_is_not_overridden(self):
        config = BiEliteCS66RTConfig(
            native_assembly="diagonal-07",
            left_robot_ip="10.0.0.1",
            right_robot_ip="10.0.0.2",
            left_start_position_rad=[0.1] * 6,
        )
        self.assertEqual(config.left_robot_ip, "10.0.0.1")
        self.assertEqual(config.right_robot_ip, "10.0.0.2")
        self.assertEqual(config.left_start_position_rad, [0.1] * 6)

    def test_existing_speed_caps_reach_native_planner(self):
        config = EliteCS66RTConfig(max_lin_speed=0.1, max_ang_speed=0.2)
        native = libpyelite.EliteDriverConfig()
        _configure_native_driver(native, config, None)
        self.assertEqual(native.planner.max_linear_velocity, 0.1)
        self.assertEqual(native.planner.max_angular_velocity, 0.2)

    def test_bimanual_tcp_cap_defaults_to_the_native_default(self):
        robot = self.make_robot("diagonal-08")
        robot.config.left_tool_transform = np.eye(4).ravel().tolist()
        planner = robot._make_driver_config("left").planner
        self.assertEqual((planner.max_linear_velocity, planner.max_angular_velocity), (0.5, 1.0))

    def test_collision_profile_needs_an_assembly(self):
        with self.assertRaises(ValueError):
            BiEliteCS66RTConfig(native_collision_profile="auto")

    def test_collision_layer_uses_the_effective_controller_tools(self):
        robot = self.make_robot("diagonal-08")
        robot.config.native_collision_profile = "auto"
        expected = []
        for side, offset in (("left", 0.12), ("right", 0.21)):
            tool = np.eye(4)
            tool[2, 3] = offset
            expected.append(tool.ravel().tolist())
            robot._rtsi[side] = Mock()
            with patch(
                "lerobot.robots.bi_elite_cs66_rt.bi_elite_cs66_rt._derive_controller_tool",
                return_value=expected[-1],
            ):
                cfg = robot._make_driver_config(side)
            np.testing.assert_allclose(cfg.tool, expected[-1])
            robot._driver[side] = SimpleNamespace(_native=Mock())
        coordinator = Mock()
        robot._cs = SimpleNamespace(_native=SimpleNamespace(StationCoordinator=Mock(return_value=coordinator)))
        with patch("libpyelite.collision.StationLayer") as layer:
            robot._start_station_layer()
            np.testing.assert_allclose(layer.call_args.args[2], expected)
        robot._stop_station_layer()

    def test_collision_layer_rejects_unconfigured_tools(self):
        robot = self.make_robot("diagonal-08")
        robot.config.native_collision_profile = "auto"
        with self.assertRaisesRegex(RuntimeError, "TCP transforms must be configured"):
            robot._start_station_layer()

    def test_collision_profile_puts_both_arms_under_one_coordinator(self):
        robot = self.make_robot("diagonal-08")
        robot.config.native_collision_profile = "auto"
        native = {side: Mock() for side in ("left", "right")}
        for side in ("left", "right"):
            robot._driver[side] = SimpleNamespace(_native=native[side])
            robot._rtsi[side] = Mock()
            robot._native_tools[side] = np.eye(4).ravel().tolist()
        coordinator = Mock()
        robot._cs = SimpleNamespace(_native=SimpleNamespace(StationCoordinator=Mock(return_value=coordinator)))
        robot._start_station_layer()
        for side in ("left", "right"):
            native[side].setExternalPlanning.assert_called_once_with(True)
        args = robot._cs._native.StationCoordinator.call_args.args
        self.assertIs(args[0], robot._station.guard)
        self.assertEqual(args[1:], (native["left"], native["right"], robot._rtsi["left"], robot._rtsi["right"]))
        coordinator.start.assert_called_once()
        robot._stop_station_layer()
        coordinator.stop.assert_called_once()
        self.assertIsNone(robot._station)

    def test_movej_through_the_table_is_refused(self):
        from libpyelite.collision import StationLayer

        robot = self.make_robot("diagonal-08")
        assembly = Assembly("diagonal-08")
        robot._station = StationLayer(assembly, "auto", [np.eye(4)] * 2)
        start = {side: assembly.initial_pose(side) for side in ("left", "right")}
        for side in ("left", "right"):
            robot._rtsi[side] = Mock(getActualJointPositions=Mock(return_value=start[side].tolist()))
        robot._check_joint_move({"left": start["left"]}, "no-op move")  # staying put is fine
        down = start["left"].copy()
        down[1] += 1.5  # shoulder well down into the table
        with self.assertRaisesRegex(RuntimeError, "refused by the station collision layer"):
            robot._check_joint_move({"left": down}, "left MoveJ")

    def test_rpy_feedback_matches_native_conversion(self):
        for tcp in ([0.2, 0.3, 0.4, 0.7, -0.4, 1.2], [0, 0, 0, 2.1, 1.0, -2.0]):
            np.testing.assert_allclose(
                _pose6_to_matrix(_controller_tcp_to_pose6(tcp)),
                _pose6_to_matrix(libpyelite.controller_tcp_to_pose6(tcp)),
            )

    def test_cartesian_actions_use_native_targets(self):
        robot = self.make_robot("diagonal-08")
        for side in ("left", "right"):
            driver = Mock()
            driver.status.return_value = SimpleNamespace(running=True, reset_active=False, error="")
            pose = np.eye(4)
            pose[:3, 3] = [0.3, 0.0, 0.4]
            driver.commanded_pose.return_value = pose
            robot._driver[side] = driver
            sent = {}
            robot._send_arm_action(side, {f"{side}_tcp.x": 0.6}, sent)
            driver.submit_target.assert_called_once()
            driver.writeServoj.assert_not_called()
            self.assertIn(f"{side}_tcp.x", sent)

    def test_native_bridge_keeps_payload_and_dh_interfaces(self):
        self.assertTrue(hasattr(libpyelite._native.NativeDriver, "setPayload"))
        self.assertTrue(hasattr(libpyelite._native.NativeDriver, "getPrimaryPackage"))
        self.assertEqual(len(libpyelite.KinematicsInfo().dh_alpha_), 6)


if __name__ == "__main__":
    unittest.main()
