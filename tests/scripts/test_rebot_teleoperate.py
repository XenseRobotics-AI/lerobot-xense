"""B601 handoffs preserve commanded TCP poses; dry runs use measured poses."""

import contextlib
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

import draccus
import numpy as np

from lerobot.robots.rebot_b601_rt.exit_policy import B601ControlFaultError
from lerobot.scripts import lerobot_teleoperate as cli, teleop_device_loops as loops
from tests.scripts.conftest_teleop_doubles import FakeBiPico4Teleop, FakeRobot, FakeTeleop, run_loop
from tests.scripts.rebot_test_assets import isolated_rebot_assets

ROOT = Path(__file__).resolve().parents[2]


class TestRebotTeleoperate(unittest.TestCase):
    def setUp(self):
        self.enterContext(isolated_rebot_assets())
        self.enterContext(patch.object(cli, "logger"))

    def test_recipes_switch_to_position_velocity_with_only_the_mode_override(self):
        for robot_type in ("rebot_b601_rt", "bi_rebot_b601_rt"):
            for recipe in ("dm-taccap00-09.yaml", "dm-taccap00-10.yaml", "rs-taccap05.yaml"):
                with self.subTest(robot_type=robot_type, recipe=recipe):
                    prefixes = (
                        ("robot.left_arm_config", "robot.right_arm_config")
                        if robot_type == "bi_rebot_b601_rt"
                        else ("robot",)
                    )
                    config = draccus.parse(
                        cli.TeleoperateConfig,
                        config_path=ROOT / "recipes/teleop" / robot_type / recipe,
                        args=[f"--{prefix}.control_mode=pos_vel" for prefix in prefixes],
                    )
                    arms = (
                        (config.robot.left_arm_config, config.robot.right_arm_config)
                        if robot_type == "bi_rebot_b601_rt"
                        else (config.robot,)
                    )
                    for arm in arms:
                        self.assertEqual(arm.control_mode, "pos_vel")
                        self.assertEqual(arm.gravity_compensation_scale, 0.0)

    def test_single_arm_recipe_and_pico4_startup(self):
        config = draccus.parse(
            cli.TeleoperateConfig,
            config_path=ROOT / "recipes/teleop/rebot_b601_rt/rs-taccap05.yaml",
            args=["--display_data=false", "--dryrun=true"],
        )
        self.assertEqual(config.robot.motor_family, "rs")
        self.assertEqual(config.robot.control_mode, "mit")
        self.assertEqual(config.robot.gravity_compensation_scale, 1.0)
        self.assertTrue(config.robot.interpolate_cartesian_targets)
        self.assertFalse(config.robot.mit_velocity_feedforward)
        self.assertTrue(config.robot.reset_on_exit)
        self.assertEqual(config.robot.reset_position_rad, [0, 0.8, 0.8, -0.5, 0, 0])
        self.assertEqual(config.robot.max_joint_speeds, [1.5] * 6)
        self.assertEqual(config.robot.reset_joint_speed, 0.5)
        self.assertTrue(config.robot.uses_taccap_model)
        self.assertFalse(config.robot.has_gripper)
        pose = np.array([0.1, 0.2, 0.3, 1, 0, 0, 0, 0.7])
        robot = Mock(get_current_tcp_pose_quat=Mock(return_value=pose))
        teleop = Mock()
        with (
            patch.object(cli, "make_robot_from_config", return_value=robot),
            patch.object(cli, "make_teleoperator_from_config", return_value=teleop),
            patch.object(cli, "run_cartesian_teleop_loop") as loop,
            patch.object(cli, "_cleanup"),
        ):
            cli.teleoperate(config)
        robot.connect.assert_called_once_with(go_to_start=False)
        np.testing.assert_allclose(teleop.connect.call_args.kwargs["current_tcp_pose_quat"], pose)
        self.assertIsInstance(loop.call_args.kwargs["policy"], cli.Pico4Policy)
        self.assertTrue(loop.call_args.kwargs["dryrun"])
        robot.disconnect.assert_called_once_with(reset=False)

    def test_bimanual_recipe_waits_for_vr_then_seeds_each_arm(self):
        config = draccus.parse(
            cli.TeleoperateConfig,
            config_path=ROOT / "recipes/teleop/bi_rebot_b601_rt/rs-taccap05.yaml",
            args=["--display_data=false"],
        )
        self.assertEqual(config.robot.left_arm_config.channel, "can1")
        self.assertEqual(config.robot.right_arm_config.channel, "can0")
        self.assertTrue(config.robot.reset_on_exit)
        for arm in (config.robot.left_arm_config, config.robot.right_arm_config):
            self.assertEqual(arm.motor_family, "rs")
            self.assertEqual(arm.control_mode, "mit")
            self.assertEqual(arm.gravity_compensation_scale, 1.0)
            self.assertTrue(arm.interpolate_cartesian_targets)
            self.assertFalse(arm.mit_velocity_feedforward)
            self.assertEqual(arm.reset_position_rad, [0, 0.8, 0.8, -0.5, 0, 0])
            self.assertEqual(arm.max_joint_speeds, [1.5] * 6)
            self.assertEqual(arm.reset_joint_speed, 0.5)
            self.assertTrue(arm.uses_taccap_model)
            self.assertFalse(arm.has_gripper)
        left = np.array([0.1, 0.2, 0.3, 1, 0, 0, 0, 0.7])
        right = np.array([0.4, 0.5, 0.6, 1, 0, 0, 0, 0.3])
        events = []
        robot = Mock(
            get_commanded_tcp_pose_quat=Mock(return_value=(left, right)),
            get_current_tcp_pose_quat=Mock(side_effect=AssertionError("Measured pose used for startup target")),
        )
        robot.connect.side_effect = lambda **kwargs: events.append("arm")
        teleop = Mock()
        teleop.pre_init.side_effect = lambda: events.append("vr")
        with (
            patch.object(cli, "make_robot_from_config", return_value=robot),
            patch.object(cli, "make_teleoperator_from_config", return_value=teleop),
            patch.object(cli, "run_cartesian_teleop_loop") as loop,
            patch.object(cli, "_cleanup"),
        ):
            cli.teleoperate(config)
        self.assertEqual(events, ["vr", "arm"])
        robot.connect.assert_called_once_with(go_to_start=True)
        np.testing.assert_allclose(teleop.connect.call_args.kwargs["left_tcp_pose_quat"], left)
        np.testing.assert_allclose(teleop.connect.call_args.kwargs["right_tcp_pose_quat"], right)
        self.assertIsInstance(loop.call_args.kwargs["policy"], cli.BiPico4Policy)
        loop.call_args.kwargs["policy"].resync(robot, teleop)
        for actual, expected in zip(
            teleop.reset_to_pose.call_args.args, (left[:7], right[:7], left[7], right[7]), strict=True
        ):
            np.testing.assert_allclose(actual, expected)
        robot.disconnect.assert_called_once_with(reset=True)

    def test_missing_b601_sdk_fails_before_peripherals_start(self):
        config = draccus.parse(
            cli.TeleoperateConfig,
            config_path=ROOT / "recipes/teleop/bi_rebot_b601_rt/dm-taccap00-09.yaml",
            args=["--display_data=false"],
        )
        with (
            patch.object(cli, "make_robot_from_config", side_effect=ImportError("missing rebot_rt")),
            patch.object(cli, "make_teleoperator_from_config") as make_teleop,
            patch.object(cli, "_cleanup") as cleanup,
        ):
            cli.teleoperate(config)
        make_teleop.assert_not_called()
        cleanup.assert_called_once_with(None, None, False)
        self.assertIn("missing rebot_rt", cli.logger.error.call_args.args[0])

    def test_disconnect_skips_return_only_for_arm_faults_or_dry_runs(self):
        for mode in ("mit", "pos_vel"):
            for outcome, enabled, dryrun, expected in (
                (None, True, False, True),
                (KeyboardInterrupt(), True, False, True),
                (B601ControlFaultError("B601 C++ control fault: motor fault"), True, False, False),
                (RuntimeError("control fault: CAN timeout"), True, False, False),
                (TimeoutError("camera frame timeout"), True, False, True),
                (RuntimeError("Pico4 VR stream lost"), True, False, True),
                (ValueError("TCP rotation columns are parallel"), True, False, True),
                (KeyboardInterrupt(), False, False, True),
                (KeyboardInterrupt(), True, True, False),
            ):
                with self.subTest(mode=mode, outcome=outcome, enabled=enabled, dryrun=dryrun):
                    config = draccus.parse(
                        cli.TeleoperateConfig,
                        config_path=ROOT / "recipes/teleop/rebot_b601_rt/rs-taccap05.yaml",
                        args=[
                            "--display_data=false",
                            f"--robot.control_mode={mode}",
                            f"--robot.gravity_compensation_scale={1.0 if mode == 'mit' else 0.0}",
                        ],
                    )
                    config.robot.reset_on_exit = enabled
                    config.dryrun = dryrun
                    robot, teleop = Mock(), Mock()
                    events = Mock()
                    events.attach_mock(robot.disconnect, "disconnect")
                    with (
                        patch.object(cli, "make_robot_from_config", return_value=robot),
                        patch.object(cli, "make_teleoperator_from_config", return_value=teleop),
                        patch.object(cli, "run_cartesian_teleop_loop", side_effect=outcome),
                        patch.object(cli, "_cleanup", new=events.cleanup),
                        # teleoperate() logs and swallows ordinary exceptions; Ctrl+C is suppressed in the loop.
                        contextlib.suppress(KeyboardInterrupt),
                    ):
                        cli.teleoperate(config)
                    self.assertEqual(
                        events.mock_calls, [call.disconnect(reset=expected), call.cleanup(robot, teleop, False)]
                    )

    def test_teleop_startup_failure_after_arm_startup_still_returns_the_arm(self):
        # The arm is already enabled at its start pose when the VR/SpaceMouse
        # connect fails; disabling it in place would drop it.
        for error in (KeyboardInterrupt(), RuntimeError("VR service unavailable")):
            with self.subTest(error=error):
                config = draccus.parse(
                    cli.TeleoperateConfig,
                    config_path=ROOT / "recipes/teleop/rebot_b601_rt/rs-taccap05.yaml",
                    args=["--display_data=false"],
                )
                robot, teleop = Mock(), Mock()
                teleop.connect.side_effect = error
                with (
                    patch.object(cli, "make_robot_from_config", return_value=robot),
                    patch.object(cli, "make_teleoperator_from_config", return_value=teleop),
                    patch.object(cli, "_cleanup") as cleanup,
                    contextlib.suppress(KeyboardInterrupt),
                ):
                    cli.teleoperate(config)
                robot.disconnect.assert_called_once_with(reset=True)
                cleanup.assert_called_once_with(robot, teleop, False)

    def test_single_arm_spacemouse_uses_commanded_euler_pose(self):
        config = draccus.parse(
            cli.TeleoperateConfig,
            config_path=ROOT / "recipes/teleop/rebot_b601_rt/rs-taccap05.yaml",
            args=["--teleop.type=spacemouse", "--display_data=false"],
        )
        pose = np.array([0.1, 0.2, 0.3, 0, 0, 1.5, 0.7])
        robot = Mock(
            get_commanded_tcp_pose_euler=Mock(return_value=pose),
            get_current_tcp_pose_euler=Mock(side_effect=AssertionError("Measured pose used for startup target")),
        )
        teleop = Mock()
        with (
            patch.object(cli, "make_robot_from_config", return_value=robot),
            patch.object(cli, "make_teleoperator_from_config", return_value=teleop),
            patch.object(cli, "run_cartesian_teleop_loop") as loop,
            patch.object(cli, "_cleanup"),
        ):
            cli.teleoperate(config)
        np.testing.assert_allclose(teleop.connect.call_args.kwargs["current_tcp_pose_euler"], pose)
        self.assertIsInstance(loop.call_args.kwargs["policy"], cli.SpaceMousePolicy)

    def test_a_reset_stands_aside_then_resyncs_before_sending_again(self):
        robot = FakeRobot(iterations=5, name="rebot_b601_rt", rt_moving_frames={1, 2})
        pose = np.array([0.15, 0.25, 0.4, 1, 0, 0, 0, 0.7])
        robot.get_commanded_tcp_pose_quat = Mock(return_value=pose)
        teleop = FakeTeleop(reset_frames={0})
        with (
            patch.object(loops, "precise_sleep"),
            patch.object(loops, "_sleep_to_fps"),
            patch.object(cli, "_print_obs_state"),
            patch("builtins.print"),
        ):
            run_loop(
                loops.run_cartesian_teleop_loop,
                iterations=5,
                teleop=teleop,
                robot=robot,
                fps=30,
                policy=cli.Pico4Policy(use_commanded_pose=True),
            )
        self.assertEqual(robot.names().count("reset_to_initial_position"), 1)
        self.assertEqual(robot.names().count("send_action"), 1)
        self.assertEqual(teleop.names().count("reset_to_pose"), 1)
        robot.get_commanded_tcp_pose_quat.assert_called_once()
        self.assertNotIn("get_current_tcp_pose_quat", robot.names())
        reset = next(call for call in teleop.calls if call[0] == "reset_to_pose")
        np.testing.assert_allclose(reset[1], pose[:7])

    def test_commanded_pose_is_opt_in_and_dryrun_still_uses_measurement(self):
        for dual in (False, True):
            with self.subTest(dual=dual):
                robot = FakeRobot(iterations=1, bimanual=dual)
                robot.get_commanded_tcp_pose_quat = Mock(side_effect=AssertionError("Unexpected commanded pose"))
                teleop = FakeBiPico4Teleop() if dual else FakeTeleop()
                policy = cli.BiPico4Policy if dual else cli.Pico4Policy
                policy().resync(robot, teleop)
                policy(use_commanded_pose=True).dryrun_resync(robot, teleop)
                self.assertEqual(robot.names().count("get_current_tcp_pose_quat"), 2)
                robot.get_commanded_tcp_pose_quat.assert_not_called()


if __name__ == "__main__":
    unittest.main()
