"""Offline model selection and independent gripper routing for B601 RT arms."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import draccus
import numpy as np

try:
    import rebot_rt
except ImportError as exc:
    raise unittest.SkipTest("Build libpyrebot to run the B601 RT adapter tests") from exc

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.grippers import SerialGripperConfig, TaccapFollowerConfig
from lerobot.robots.bi_rebot_b601_rt import BiRebotB601RT, BiRebotB601RTConfig, bi_rebot_b601_rt as bi_driver
from lerobot.robots.rebot_b601_rt import RebotB601RT, RebotB601RTConfig, rebot_b601_rt as driver
from lerobot.robots.rebot_b601_rt.dm_serial_discovery import DMSerialDevice
from lerobot.robots.rebot_b601_rt.gravity_compensation import load_joint_torque_offsets

ASSETS = Path(rebot_rt.__file__).parent / "assets"
TACCAP_URDFS = {
    "dm": "DM_TacCap00/urdf/rebot_b601_dm_taccap_v1_gripper.urdf",
    "rs": "RS_TacCap05/urdf/rebot_b601_rs_taccap_v2_gripper.urdf",
}
INITIAL_POSE = [0.0, 0.8, 0.8, -0.5, 0.0, 0.0]


def fake_gripper():
    gripper = SimpleNamespace(is_connected=False)
    gripper.connect = Mock(side_effect=lambda: setattr(gripper, "is_connected", True))
    gripper.disconnect = Mock(side_effect=lambda: setattr(gripper, "is_connected", False))
    gripper.get_gripper_position = Mock(return_value=0.73)
    gripper.set_gripper_position = Mock()
    return gripper


def fake_native():
    native = SimpleNamespace(connected=False)
    native.connect = Mock(side_effect=lambda: setattr(native, "connected", True))
    native.start = Mock()
    native.stop = Mock()
    native.close = Mock(side_effect=lambda: setattr(native, "connected", False))
    native.get_state = Mock(
        return_value=SimpleNamespace(
            fault=False,
            q=[0.0] * 6 + [-1.2],
            command=[0.0] * 6 + [-1.2],
            target=[0.0] * 6 + [-1.2],
            command_velocity=[0.0] * 7,
            mit_command_velocity=[0.0] * 7,
            cartesian_ik_valid=False,
        )
    )
    native.get_target_joints = Mock(return_value=[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, -1.2])
    native.set_target_joints = Mock()
    native.queue_target_pose = Mock(return_value=1)
    native.hold_position = Mock()
    native.forward_kinematics = Mock(return_value=np.eye(4))
    native.set_target_pose = Mock(return_value=SimpleNamespace(success=True, target=[0.0] * 6 + [-1.2]))
    return native


class TestRebotB601RT(unittest.TestCase):
    def setUp(self):
        temp_dir = self.enterContext(TemporaryDirectory())
        self.calibration_dir = Path(temp_dir)
        self.native = fake_native()
        self.native_factory = self.enterContext(patch.object(driver.rebot_rt, "Robot", return_value=self.native))
        self.gripper_factory = self.enterContext(patch.object(driver, "make_gripper_from_config", return_value=None))

    def make_robot(self, **kwargs):
        kwargs.setdefault("interpolate_cartesian_targets", False)
        config = RebotB601RTConfig(calibration_dir=self.calibration_dir, **kwargs)
        robot = RebotB601RT(config)
        self.addCleanup(lambda: robot.disconnect() if robot.is_connected else None)
        return robot

    def test_calibration_delegates_without_connecting_rt_controller(self):
        robot = self.make_robot(motor_family="rs")
        with patch("lerobot.robots.rebot_b601_rt.calibration.calibrate_b601") as calibrate:
            robot.calibrate()
        calibrate.assert_called_once_with(robot.config)
        self.assertIsNone(robot._native)

    def test_calibration_rejects_active_rt_controller(self):
        robot = self.make_robot()
        robot.connect()
        with (
            patch("lerobot.robots.rebot_b601_rt.calibration.calibrate_b601") as calibrate,
            self.assertRaisesRegex(RuntimeError, "Disconnect the RT controller"),
        ):
            robot.calibrate()
        calibrate.assert_not_called()

    def test_gravity_configuration_is_validated_before_connect(self):
        for kwargs in (
            {"gravity_compensation_scale": -0.1},
            {"gravity_compensation_scale": float("nan")},
            {"gravity_compensation_scale": 1.1},
            {"control_mode": "mit", "gravity_compensation_scale": 0.5},
            {"gravity_torque_limits_nm": [1] * 5},
            {"gravity_torque_limits_nm": [-1] * 6},
            {"gravity_torque_limits_nm": [float("nan")] * 6},
            {"gravity_slew_rate_nm_s": 0},
            {"gravity_vector": [0, 0]},
            {"gravity_vector": [0, 0, float("nan")]},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.make_robot(**kwargs)
        self.native.connect.assert_not_called()

    def test_auto_serial_is_dm_only_and_rejects_unknown_selectors(self):
        for family, channel in (("rs", "auto:first"), ("dm", "auto"), ("dm", "auto:third")):
            with self.subTest(family=family, channel=channel), self.assertRaises(ValueError):
                self.make_robot(motor_family=family, channel=channel)

    def test_cartesian_speed_limits_validate_and_reach_native_for_both_families(self):
        for name in ("max_cartesian_linear_speed", "max_cartesian_angular_speed"):
            for value in (0, -1, float("nan"), float("inf")):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    self.make_robot(interpolate_cartesian_targets=True, **{name: value})
            with self.assertRaisesRegex(ValueError, "interpolate_cartesian_targets"):
                self.make_robot(**{name: 1.0})
        self.native.connect.assert_not_called()
        for family in ("dm", "rs"):
            with self.subTest(family=family):
                robot = self.make_robot(
                    motor_family=family,
                    interpolate_cartesian_targets=True,
                    max_cartesian_linear_speed=1.0,
                    max_cartesian_angular_speed=6.28,
                )
                robot.connect()
                self.assertEqual(self.native_factory.call_args.kwargs["max_cartesian_linear_speed"], 1.0)
                self.assertEqual(self.native_factory.call_args.kwargs["max_cartesian_angular_speed"], 6.28)
                robot.disconnect()

    def test_cartesian_limits_reject_old_native_before_connect(self):
        robot = self.make_robot(interpolate_cartesian_targets=True, max_cartesian_linear_speed=1.0)
        with (
            patch.object(
                driver.rebot_rt, "State", SimpleNamespace(mit_command_velocity=None, cartesian_step_scale=None)
            ),
            self.assertRaisesRegex(RuntimeError, "Cartesian speed limit support"),
        ):
            robot.connect()
        self.native.connect.assert_not_called()

    def test_single_auto_serial_is_resolved_again_on_reconnect(self):
        robot = self.make_robot(motor_family="dm", channel="auto:first")
        with (
            patch.object(driver, "discover_dm_serial_devices") as discover,
            patch.object(driver, "verify_dm_serial_device") as verify,
        ):
            for channel, timestamp in (("/dev/ttyACM7", 100), ("/dev/ttyACM3", 300)):
                devices = (DMSerialDevice(channel, timestamp),)
                discover.return_value = devices
                robot.connect()
                self.assertEqual(self.native_factory.call_args.args[1], channel)
                self.assertEqual(robot.config.channel, "auto:first")
                verify.assert_called_with(channel, devices)
                robot.disconnect()
            self.assertEqual(discover.call_count, 2)

    def test_auto_serial_reenumeration_aborts_before_native_connect(self):
        robot = self.make_robot(motor_family="dm", channel="auto:first")
        with (
            patch.object(driver, "discover_dm_serial_devices", return_value=(DMSerialDevice("/dev/ttyACM7", 100),)),
            patch.object(driver, "verify_dm_serial_device", side_effect=RuntimeError("changed during startup")),
            self.assertRaisesRegex(RuntimeError, "changed during startup"),
        ):
            robot.connect()
        self.native.connect.assert_not_called()

    def test_bimanual_auto_serial_resolves_both_before_connect_from_one_snapshot(self):
        devices = (DMSerialDevice("/dev/ttyACM7", 100), DMSerialDevice("/dev/ttyACM0", 200))
        natives = [fake_native(), fake_native()]
        self.native_factory.side_effect = natives
        robot = BiRebotB601RT(
            BiRebotB601RTConfig(
                calibration_dir=self.calibration_dir,
                left_arm_config=RebotB601RTConfig(channel="auto:second"),
                right_arm_config=RebotB601RTConfig(channel="auto:first"),
            )
        )
        with (
            patch.object(bi_driver, "discover_dm_serial_devices", return_value=devices) as discover,
            patch.object(driver, "discover_dm_serial_devices") as single_discover,
            patch.object(driver, "verify_dm_serial_device") as verify,
        ):
            try:
                robot.connect()
                self.assertEqual(
                    [c.args[1] for c in self.native_factory.call_args_list], ["/dev/ttyACM0", "/dev/ttyACM7"]
                )
                discover.assert_called_once()
                single_discover.assert_not_called()
                self.assertEqual(verify.call_args_list, [call("/dev/ttyACM0", devices), call("/dev/ttyACM7", devices)])
            finally:
                robot.disconnect()

    def test_bimanual_auto_serial_missing_or_duplicate_device_fails_before_either_arm_connects(self):
        devices = (DMSerialDevice("/dev/ttyACM7", 100),)
        for left_channel, error in (("auto:second", RuntimeError), ("/dev/ttyACM7", ValueError)):
            robot = BiRebotB601RT(
                BiRebotB601RTConfig(
                    calibration_dir=self.calibration_dir,
                    left_arm_config=RebotB601RTConfig(channel=left_channel),
                    right_arm_config=RebotB601RTConfig(channel="auto:first"),
                )
            )
            with (
                patch.object(bi_driver, "discover_dm_serial_devices", return_value=devices),
                self.assertRaises(error),
            ):
                robot.connect()
        self.native_factory.assert_not_called()

    def test_gravity_is_opt_in_and_configured_before_native_connect(self):
        self.native.configure_gravity_compensation = Mock()
        robot = self.make_robot()
        robot.connect()
        self.native.configure_gravity_compensation.assert_not_called()
        robot.disconnect()
        robot = self.make_robot(
            control_mode="mit", gravity_compensation_scale=0.5, gravity_torque_limits_nm=[1, 8, 8, 3, 1, 1]
        )
        parent = Mock()
        parent.attach_mock(self.native.configure_gravity_compensation, "configure")
        parent.attach_mock(self.native.connect, "connect")
        robot.connect()
        self.assertEqual(
            parent.mock_calls[:2],
            [
                call.configure(
                    scale=0.5, torque_limits_nm=[1, 8, 8, 3, 1, 1], slew_rate_nm_s=5.0, gravity=[0.0, 0.0, -9.81]
                ),
                call.connect(),
            ],
        )

    def test_enabled_gravity_requires_updated_native_before_hardware_access(self):
        robot = self.make_robot(control_mode="mit", gravity_compensation_scale=0.5, gravity_torque_limits_nm=[1] * 6)
        with self.assertRaisesRegex(RuntimeError, "gravity compensation"):
            robot.connect()
        self.native.connect.assert_not_called()

    def test_position_velocity_mode_skips_gravity_and_offset_setup(self):
        self.native.configure_gravity_compensation = Mock()
        robot = self.make_robot(
            control_mode="pos_vel",
            gravity_compensation_scale=1.0,
            gravity_use_joint_offsets=True,
        )
        with patch.object(driver, "load_joint_torque_offsets") as load_offsets:
            robot.connect()
        self.assertEqual(robot.config.gravity_compensation_scale, 0.0)
        self.assertEqual(self.native_factory.call_args.args[9], "pos_vel")
        self.native.configure_gravity_compensation.assert_not_called()
        load_offsets.assert_not_called()
        self.native.connect.assert_called_once()

    def test_identified_offsets_are_loaded_before_connect_and_missing_file_is_rejected(self):
        self.native.configure_gravity_compensation = Mock()
        urdf = ASSETS / TACCAP_URDFS["dm"]
        offsets = load_joint_torque_offsets(urdf)
        np.testing.assert_allclose(
            offsets, [0.08083891458, 0.02049686659, -0.05986744802, 0.05837195786, -0.008452037538, -0.01073598411]
        )
        robot = self.make_robot(
            control_mode="mit",
            urdf_path=str(urdf),
            has_gripper=False,
            gravity_compensation_scale=1,
            gravity_torque_limits_nm=[1, 22, 12, 4, 2, 1],
            gravity_use_joint_offsets=True,
        )
        parent = Mock()
        parent.attach_mock(self.native.configure_gravity_compensation, "configure")
        parent.attach_mock(self.native.connect, "connect")
        robot.connect()
        self.assertEqual(parent.mock_calls[0].kwargs["torque_offsets_nm"], offsets)
        self.assertEqual(parent.mock_calls[1], call.connect())
        robot.disconnect()
        self.native.connect.reset_mock()
        copied = self.calibration_dir / "custom.urdf"
        copied.write_text(urdf.read_text())
        robot = self.make_robot(
            control_mode="mit",
            urdf_path=str(copied),
            has_gripper=False,
            gravity_compensation_scale=1,
            gravity_torque_limits_nm=[1] * 6,
            gravity_use_joint_offsets=True,
        )
        with self.assertRaises(FileNotFoundError):
            robot.connect()
        self.native.connect.assert_not_called()

    def test_offset_loader_uses_names_and_rejects_incomplete_or_nonfinite_values(self):
        urdf = self.calibration_dir / "custom.urdf"
        urdf.write_text((ASSETS / TACCAP_URDFS["dm"]).read_text())
        offsets = urdf.with_name("custom_offset.yaml")
        good = "joints:\n" + "".join(f"  joint{i}: {{offset: {i / 100}}}\n" for i in range(6, 0, -1))
        offsets.write_text(good)
        self.assertEqual(load_joint_torque_offsets(urdf), [i / 100 for i in range(1, 7)])
        for bad in (
            good.replace("joint6", "joint7"),
            good.replace("0.06", ".nan"),
            good.replace("0.06", "true"),
            "joints: []",
            "{}",
        ):
            offsets.write_text(bad)
            with self.assertRaises(ValueError):
                load_joint_torque_offsets(urdf)

    def test_native_cartesian_queue_in_both_control_modes(self):
        self.assertTrue(RebotB601RTConfig().interpolate_cartesian_targets)
        for mode in ("mit", "pos_vel"):
            with self.subTest(mode=mode):
                robot = self.make_robot(control_mode=mode, has_gripper=False, interpolate_cartesian_targets=True)
                robot.connect()
                robot.logger = Mock()
                action = driver._action_from_pose(np.eye(4))
                robot.send_action(action)
                self.native.queue_target_pose.assert_called()
                self.native.set_target_pose.assert_not_called()
                state = self.native.get_state.return_value
                state.cartesian_ik_valid = True
                state.cartesian_ik_success = False
                state.cartesian_ik_error = 0.1
                robot.get_observation()
                robot.logger.warn.assert_called_once()
                state.cartesian_ik_success = True
                robot.get_observation()
                robot.logger.info.assert_called_once()
                robot.disconnect()

    def test_outdated_native_api_reports_loaded_package_before_connecting(self):
        robot = self.make_robot(has_gripper=False)
        old_state = SimpleNamespace(mit_command_velocity=None)
        with patch.object(driver.rebot_rt, "State", old_state), self.assertRaises(RuntimeError) as caught:
            robot.connect()
        self.assertIn(str(driver.rebot_rt.__file__), str(caught.exception))
        self.assertIn("PYTHONPATH", str(caught.exception))
        self.assertIn("per-joint limits", str(caught.exception))
        self.native_factory.assert_not_called()

    def test_partial_tracking_is_not_reported_as_an_ik_rejection(self):
        robot = self.make_robot(has_gripper=False, interpolate_cartesian_targets=True)
        robot.connect()
        robot.logger = Mock()
        state = self.native.get_state.return_value
        state.cartesian_ik_valid = True
        state.cartesian_ik_success = False
        state.cartesian_tracking_limited = True
        state.cartesian_tracking_progress = True
        with patch.object(driver.time, "monotonic", return_value=10.0):
            robot.get_observation()
        robot.logger.warn.assert_not_called()
        self.assertIn("approaching", robot.logger.info.call_args.args[0])
        # Limit transitions can occur at every producer update. Do not flood
        # the terminal when tracking alternates between partial and complete.
        with patch.object(driver.time, "monotonic", return_value=10.1):
            state.cartesian_tracking_limited = False
            robot.get_observation()
            state.cartesian_tracking_limited = True
            robot.get_observation()
        self.assertEqual(robot.logger.info.call_count, 1)
        with patch.object(driver.time, "monotonic", return_value=11.1):
            state.cartesian_tracking_limited = False
            robot.get_observation()
        self.assertIn("reached", robot.logger.info.call_args.args[0])

    def test_motor_family_and_gripper_select_matching_models(self):
        for family in ("dm", "RS"):
            for taccap in (False, True):
                with self.subTest(family=family, taccap=taccap):
                    gripper = fake_gripper() if taccap else None
                    self.gripper_factory.return_value = gripper
                    robot = self.make_robot(
                        motor_family=family,
                        gripper=TaccapFollowerConfig(auto_discover_cameras=False) if taccap else None,
                    )
                    robot.connect()
                    args = self.native_factory.call_args.args
                    model = (
                        TACCAP_URDFS[family.lower()]
                        if taccap
                        else f"{family.upper()}/urdf/ReBot_Arm_{family.upper()}.urdf"
                    )
                    self.assertEqual(args[0], rebot_rt.Family.DM if family == "dm" else rebot_rt.Family.RS)
                    self.assertEqual(Path(args[2]), ASSETS / model)
                    self.assertTrue(Path(args[2]).is_file())
                    self.assertEqual(args[3], "taccap_eef_link" if taccap else "")
                    self.assertEqual(args[4], not taccap)
                    self.assertEqual(args[9], "pos_vel")
                    self.assertIn("gripper.pos", robot.action_features)
                    self.assertIn("gripper.pos", robot.observation_features)
                    obs = robot.get_observation()
                    self.assertEqual(obs["gripper.pos"], 0.73 if taccap else -1.2)
                    action = {**driver._action_from_pose(np.eye(4)), "gripper.pos": 0.4 if taccap else -0.4}
                    sent = robot.send_action(action)
                    self.assertEqual(self.native.set_target_pose.call_args.args[1], 0.0 if taccap else -0.4)
                    if taccap:
                        gripper.set_gripper_position.assert_called_once_with(0.4)
                        self.assertEqual(sent["gripper.pos"], 0.4)
                    robot.disconnect()
                    if taccap:
                        gripper.connect.assert_called_once()
                        gripper.disconnect.assert_called_once()

    def test_recipe_decodes_a_taccap_block_for_both_families(self):
        for family in ("DM", "rs"):
            with self.subTest(family=family):
                config = draccus.decode(
                    RebotB601RTConfig,
                    {
                        "motor_family": family,
                        "gripper": {"type": "taccap_follower", "side": "right"},
                    },
                )
                self.assertEqual(config.default_urdf, TACCAP_URDFS[family.lower()])
                self.assertTrue(config.uses_taccap_model)
                self.assertEqual(config.gripper.side, "right")
                self.assertFalse(config.has_gripper)

    def test_control_mode_and_position_gains_can_be_overridden(self):
        robot = self.make_robot(
            control_mode="MIT", mit_velocity_feedforward=False, motor_configs=[{"vel_kp": 7.0}, {}, {}, {}, {}, {}, {}]
        )
        robot.connect()
        args = self.native_factory.call_args.args
        self.assertEqual(args[9], "mit")
        self.assertFalse(args[10])
        self.assertEqual(args[8][0].vel_kp, 7.0)
        with self.assertRaisesRegex(ValueError, "control_mode"):
            RebotB601RTConfig(control_mode="invalid")

    def test_six_motor_arm_and_serial_gripper_keep_original_model(self):
        for family in ("dm", "rs"):
            for gripper_config in (None, SerialGripperConfig(side="left", auto_discover_cameras=False)):
                with self.subTest(family=family, gripper=gripper_config):
                    self.gripper_factory.return_value = fake_gripper() if gripper_config else None
                    robot = self.make_robot(motor_family=family, has_gripper=False, gripper=gripper_config)
                    robot.connect()
                    self.assertEqual(
                        Path(self.native_factory.call_args.args[2]),
                        ASSETS / family.upper() / "urdf" / f"ReBot_Arm_{family.upper()}.urdf",
                    )
                    self.assertEqual("gripper.pos" in robot.action_features, gripper_config is not None)
                    robot.disconnect()

    def test_explicit_model_and_tool_frame_override_taccap_defaults(self):
        urdf = ASSETS / "DM/urdf/ReBot_Arm_DM.urdf"
        for family in ("dm", "rs"):
            for end_frame in (None, "link6"):
                with self.subTest(family=family, end_frame=end_frame):
                    self.gripper_factory.return_value = fake_gripper()
                    robot = self.make_robot(
                        motor_family=family,
                        urdf_path=str(urdf),
                        end_frame=end_frame,
                        gripper=TaccapFollowerConfig(auto_discover_cameras=False),
                    )
                    self.assertFalse(robot.config.uses_taccap_model)
                    robot.connect()
                    self.assertEqual(self.native_factory.call_args.args[2:5], (str(urdf), end_frame or "", False))
                    robot.disconnect()

    def test_explicit_taccap_model_disables_native_gripper_for_both_families(self):
        for family in ("dm", "rs"):
            for model in TACCAP_URDFS.values():
                with self.subTest(family=family, model=model):
                    urdf = str(ASSETS / model)
                    robot = self.make_robot(motor_family=family, urdf_path=urdf)
                    robot.connect()
                    self.assertEqual(self.native_factory.call_args.args[2:5], (urdf, "taccap_eef_link", False))
                    self.assertNotIn("gripper.pos", robot.action_features)
                    robot.disconnect()

    def test_bimanual_mixed_families_select_each_arms_taccap_model(self):
        self.native_factory.side_effect = [fake_native(), fake_native()]
        self.gripper_factory.side_effect = [fake_gripper(), fake_gripper()]
        robot = BiRebotB601RT(
            BiRebotB601RTConfig(
                calibration_dir=self.calibration_dir,
                left_arm_config=RebotB601RTConfig(
                    motor_family="dm",
                    channel="can1",
                    gripper=TaccapFollowerConfig(side="left", auto_discover_cameras=False),
                ),
                right_arm_config=RebotB601RTConfig(
                    motor_family="rs",
                    channel="can0",
                    gripper=TaccapFollowerConfig(side="right", auto_discover_cameras=False),
                ),
            )
        )
        try:
            robot.connect()
            for family, native_call in zip(("dm", "rs"), self.native_factory.call_args_list, strict=True):
                self.assertEqual(Path(native_call.args[2]), ASSETS / TACCAP_URDFS[family])
                self.assertEqual(native_call.args[3:5], ("taccap_eef_link", False))
        finally:
            robot.disconnect()

    def test_external_gripper_cannot_also_enable_seventh_arm_motor(self):
        with self.assertRaisesRegex(ValueError, "has_gripper=False"):
            RebotB601RTConfig(gripper=TaccapFollowerConfig(), has_gripper=True)

    def test_bad_external_target_does_not_command_the_arm(self):
        self.gripper_factory.return_value = fake_gripper()
        robot = self.make_robot(gripper=TaccapFollowerConfig(auto_discover_cameras=False))
        robot.connect()
        for target in (-0.1, 1.1, float("nan")):
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
                robot.send_action({**driver._action_from_pose(np.eye(4)), "gripper.pos": target})
        self.native.set_target_pose.assert_not_called()

    def test_ik_rejection_holds_last_target_and_recovers_without_disconnecting(self):
        gripper = fake_gripper()
        self.gripper_factory.return_value = gripper
        robot = self.make_robot(gripper=TaccapFollowerConfig(auto_discover_cameras=False))
        robot.connect()
        robot.logger = Mock()
        held_pose = np.eye(4)
        held_pose[:3, 3] = [0.1, 0.2, 0.3]
        self.native.forward_kinematics.return_value = held_pose
        self.native.set_target_pose.return_value = SimpleNamespace(
            success=False,
            error=0.00101552,
            target=[99.0] * 7,
        )
        action = {**driver._action_from_pose(np.eye(4)), "gripper.pos": 0.4}
        with patch.object(driver.time, "monotonic", return_value=10.0):
            for _ in range(2):
                sent = robot.send_action(action)
        self.assertTrue(robot.is_connected)
        self.assertTrue(robot._ik_failed)
        self.native.forward_kinematics.assert_called_with(self.native.get_target_joints.return_value)
        np.testing.assert_allclose([sent[k] for k in ("tcp.x", "tcp.y", "tcp.z")], [0.1, 0.2, 0.3])
        self.assertEqual(sent["gripper.pos"], 0.4)
        gripper.set_gripper_position.assert_called_with(0.4)
        robot.logger.warn.assert_called_once()
        self.native.set_target_pose.return_value = SimpleNamespace(success=True, target=[0.0] * 7)
        robot.send_action(action)
        self.assertFalse(robot._ik_failed)
        robot.logger.info.assert_called_once()

    def test_ik_rejection_does_not_hide_a_motor_fault(self):
        robot = self.make_robot(has_gripper=False)
        robot.connect()
        self.native.set_target_pose.return_value = SimpleNamespace(success=False, error=0.001)
        self.native.get_state.return_value = SimpleNamespace(fault=True, fault_reason="feedback lost")
        with self.assertRaisesRegex(RuntimeError, "feedback lost"):
            robot.send_action(driver._action_from_pose(np.eye(4)))
        self.native.get_target_joints.assert_not_called()

    def test_reset_returns_to_configured_joints_and_waits_for_measured_settling(self):
        robot = self.make_robot(reset_position_rad=INITIAL_POSE, max_joint_speed=1.5, reset_joint_speed=0.5)
        robot.connect(go_to_start=False)
        with patch.object(driver.time, "monotonic", return_value=0.0) as clock:
            robot.reset_to_initial_position()
            self.native.set_target_joints.assert_called_with(INITIAL_POSE + [-1.2], max_joint_speed=0.5)
            self.native.get_state.return_value = SimpleNamespace(
                fault=False,
                q=[0.2] * 6 + [-1.2],
                command=[0.1] * 6 + [-1.2],
            )
            clock.return_value = 0.2
            self.assertTrue(robot.rt_moving)
            with self.assertRaisesRegex(RuntimeError, "reset is in progress"):
                robot.send_action(driver._action_from_pose(np.eye(4)))
            self.native.get_state.return_value.command = INITIAL_POSE + [-1.2]
            self.assertTrue(robot.rt_moving)  # Command arrived; measured arm has not.
            self.native.get_state.return_value.q = [q + 0.001 for q in INITIAL_POSE] + [-1.2]
            clock.return_value = 0.3
            self.assertTrue(robot.rt_moving)
            clock.return_value = 0.41
            self.assertFalse(robot.rt_moving)
            self.assertIsNone(robot._reset_target)
        self.native.set_target_pose.assert_not_called()
        # Every watchdog refresh must retain the slower reset speed.
        for request in self.native.set_target_joints.call_args_list:
            self.assertEqual(request.kwargs, {"max_joint_speed": 0.5})

    def test_reset_speed_never_exceeds_the_arm_limit(self):
        robot = self.make_robot(reset_position_rad=[0.0] * 6, max_joint_speed=0.25)
        robot.connect(go_to_start=False)
        robot.reset_to_initial_position()
        self.native.set_target_joints.assert_called_with([0.0] * 6 + [-1.2], max_joint_speed=0.25)
        self.assertTrue(robot.rt_moving)
        self.native.set_target_joints.assert_called_with([0.0] * 6 + [-1.2], max_joint_speed=0.25)

    def test_joint_speed_configuration_requires_finite_positive_values(self):
        for field in ("max_joint_speed", "reset_joint_speed"):
            for speed in (0, -0.1, float("nan"), float("inf")):
                with self.subTest(field=field, speed=speed), self.assertRaisesRegex(ValueError, "positive"):
                    RebotB601RTConfig(**{field: speed})
        for limits in ([], [1.0] * 5, [1.0] * 7, [0.0] * 6, [-1.0] * 6, [float("nan")] * 6):
            with self.subTest(limits=limits), self.assertRaisesRegex(ValueError, "max_joint_speeds"):
                RebotB601RTConfig(max_joint_speeds=limits)

    def test_joint_limits_and_cartesian_coordination_reach_native_in_both_modes(self):
        limits = [0.2, 0.8, 1.1, 0.9, 0.7, 0.6]
        for mode in ("mit", "pos_vel"):
            with self.subTest(mode=mode):
                robot = self.make_robot(control_mode=mode, max_joint_speeds=limits)
                robot.connect()
                self.assertEqual(
                    self.native_factory.call_args.kwargs,
                    {
                        "max_joint_speeds": limits,
                        "coordinate_cartesian_speed": True,
                    },
                )
                robot.disconnect()

    def test_reset_caps_using_per_joint_limits_instead_of_unused_scalar(self):
        robot = self.make_robot(reset_position_rad=[0.0] * 6, max_joint_speed=1.5, max_joint_speeds=[0.25] * 6)
        robot.connect(go_to_start=False)
        robot.reset_to_initial_position()
        self.native.set_target_joints.assert_called_with([0.0] * 6 + [-1.2], max_joint_speed=0.25)
        self.assertTrue(robot.rt_moving)
        self.native.set_target_joints.assert_called_with([0.0] * 6 + [-1.2], max_joint_speed=0.25)

    def test_mixed_joint_limits_leave_per_joint_capping_to_native(self):
        # rebot_rt caps each joint at min(its limit, speed) and rejects a speed
        # above the fastest joint's limit; the adapter must stay within that.
        limits = [1.0, 0.2, 0.3, 1.0, 1.0, 1.0]
        for reset_speed, expected in ((0.5, 0.5), (2.0, 1.0)):
            with self.subTest(reset_speed=reset_speed):
                robot = self.make_robot(
                    reset_position_rad=[0.0] * 6, max_joint_speeds=limits, reset_joint_speed=reset_speed
                )
                robot.connect(go_to_start=False)
                robot.reset_to_initial_position()
                self.native.set_target_joints.assert_called_with([0.0] * 6 + [-1.2], max_joint_speed=expected)
                self.assertTrue(robot.rt_moving)
                self.native.set_target_joints.assert_called_with([0.0] * 6 + [-1.2], max_joint_speed=expected)
                robot.disconnect(reset=False)

    def test_dual_arm_decoding_keeps_independent_speed_limits(self):
        left, right = [0.2, 0.8, 1.1, 0.9, 0.7, 0.6], [0.4] * 6
        config = draccus.decode(
            BiRebotB601RTConfig,
            {
                "left_arm_config": {"max_joint_speeds": left, "coordinate_cartesian_speed": True},
                "right_arm_config": {"channel": "can1", "max_joint_speeds": right, "coordinate_cartesian_speed": False},
            },
        )
        self.assertEqual(config.left_arm_config.max_joint_speeds, left)
        self.assertEqual(config.right_arm_config.max_joint_speeds, right)
        self.assertTrue(config.left_arm_config.coordinate_cartesian_speed)
        self.assertFalse(config.right_arm_config.coordinate_cartesian_speed)

    def test_reset_timeout_holds_current_command(self):
        robot = self.make_robot(reset_position_rad=[0.0] * 6, reset_timeout_sec=1.0)
        robot.connect(go_to_start=False)
        with patch.object(driver.time, "monotonic", return_value=0.0) as clock:
            robot.reset_to_initial_position()
            clock.return_value = 1.1
            with self.assertRaisesRegex(TimeoutError, "reset_timeout_sec"):
                _ = robot.rt_moving
        self.native.hold_position.assert_called_once()
        self.assertIsNone(robot._reset_target)

    def test_reset_reports_measured_error_without_accepting_command_only(self):
        robot = self.make_robot(control_mode="mit", channel="can1", reset_position_rad=INITIAL_POSE)
        robot.connect(go_to_start=False)
        robot.logger = Mock()
        state = self.native.get_state.return_value
        state.command = INITIAL_POSE + [-1.2]
        state.q = [0.0, 0.8, 0.75, -0.55, 0.0, 0.0, -1.2]
        with patch.object(driver.time, "monotonic", return_value=0.0) as clock:
            robot.reset_to_initial_position()
            robot.logger.info.reset_mock()
            clock.return_value = 2.0
            self.assertTrue(robot.rt_moving)
            message = robot.logger.info.call_args.args[0]
            self.assertIn("can1", message)
            self.assertIn("command_arrived=True", message)
            self.assertIn("error_rad=[0.0, 0.0, 0.05, 0.05, 0.0, 0.0]", message)
            self.assertIn("waiting_for=['J3', 'J4']", message)
            clock.return_value = 2.1
            self.assertTrue(robot.rt_moving)
            robot.logger.info.assert_called_once()
            clock.return_value = 30.1
            with self.assertRaisesRegex(TimeoutError, "waiting_for=.*J3.*J4.*reset_timeout_sec"):
                _ = robot.rt_moving

    def test_reset_reports_torques_before_observations_using_custom_gains_and_sent_command(self):
        target = [0, -0.8, -0.8, 0.5, 0, 0]
        robot = self.make_robot(
            motor_family="dm",
            control_mode="mit",
            channel="/dev/ttyACM5",
            reset_position_rad=target,
            reset_tolerance_rad=0.06,
            trace_joint_tracking=False,
            motor_configs=[{}, {}, {"kp": 60, "kd": 2}, {}, {}, {}, {}],
        )
        robot.connect(go_to_start=False)
        robot.logger = Mock()
        state = self.native.get_state.return_value
        state.q = [0, -0.8, -0.7338, 0.5, 0, 0, -1.2]
        state.dq = [0, 0, 0.1, 0, 0, 0, 0]
        state.gravity_torque = [0, 0, -8.0977, 0, 0, 0, 0]
        state.mit_command_torque = [0, 0, -8.1576, 0, 0, 0, 0]
        state.tau = [0, 0, -10.5, 0, 0, 0, 0]
        state.gravity_torque_limited = [False] * 7
        with patch.object(driver.time, "monotonic", return_value=0.0) as clock:
            robot.reset_to_initial_position()
            for now, command, pd, total in ((2, -0.8, -4.172, -12.3296), (4, -0.7, 1.828, -6.3296)):
                state.command = [0, -0.8, command, 0.5, 0, 0, -1.2]
                clock.return_value = now
                self.assertTrue(robot.rt_moving)
                message = robot.logger.info.call_args.args[0]
                self.assertIn("/dev/ttyACM5", message)
                self.assertIn("waiting_for=['J3']", message)
                self.assertIn("gravity_nm=[0.0, 0.0, -8.0977", message)
                self.assertIn("mit_tau_ff_nm=[0.0, 0.0, -8.1576", message)
                self.assertIn("measured_tau_nm=[0.0, 0.0, -10.5", message)
                self.assertIn(f"mit_pd_est_nm=[0.0, 0.0, {pd}", message)
                self.assertIn(f"mit_total_est_nm=[0.0, 0.0, {total}", message)
                self.native.set_target_joints.assert_called_with(target + [-1.2], max_joint_speed=0.5)

    def test_startup_and_button_reset_accept_observed_mit_offsets_within_configured_tolerance(self):
        measured = (
            [0.0006, 0.8055, 0.7534, -0.5375, -0.0063, -0.0075],
            [0.0036, 0.8067, 0.7522, -0.5367, -0.0006, -0.0021],
        )
        for bimanual in (False, True):
            with self.subTest(bimanual=bimanual):
                natives = [fake_native() for _ in range(2 if bimanual else 1)]
                self.native_factory.side_effect = natives
                configs = [
                    RebotB601RTConfig(
                        motor_family="rs",
                        control_mode="mit",
                        channel=f"can{i}",
                        calibration_dir=self.calibration_dir,
                        reset_position_rad=INITIAL_POSE,
                        reset_tolerance_rad=0.06,
                    )
                    for i in range(len(natives))
                ]
                robot = (
                    BiRebotB601RT(
                        BiRebotB601RTConfig(
                            calibration_dir=self.calibration_dir,
                            left_arm_config=configs[0],
                            right_arm_config=configs[1],
                        )
                    )
                    if bimanual
                    else RebotB601RT(configs[0])
                )
                for native, feedback in zip(natives, measured, strict=False):
                    native.get_state.return_value.command = INITIAL_POSE + [-1.2]
                    native.get_state.return_value.q = feedback + [-1.2]
                with patch.object(driver.time, "monotonic", return_value=0.0) as clock:

                    def advance(_):
                        clock.return_value += 0.02
                        self.assertLess(clock.return_value, 1.0, "Startup did not accept the measured pose")

                    try:
                        with patch.object(driver.time, "sleep", side_effect=advance):
                            robot.connect()
                        self.assertTrue(robot.is_connected)
                        self.assertGreaterEqual(clock.return_value, 0.1)
                        # A reset still waits if any joint exceeds 0.06 rad.
                        robot.reset_to_initial_position()
                        natives[-1].get_state.return_value.q[2] = 0.73
                        self.assertTrue(robot.rt_moving)
                        clock.return_value += 0.2
                        self.assertTrue(robot.rt_moving)
                        natives[-1].get_state.return_value.q[2] = 0.7522
                        self.assertTrue(robot.rt_moving)
                        clock.return_value += 0.11
                        self.assertFalse(robot.rt_moving)
                    finally:
                        robot.disconnect(reset=False)

    def test_reset_tolerances_must_be_positive_and_finite(self):
        for field in ("reset_tolerance_rad", "exit_reset_tolerance_rad"):
            for value in (0, -0.01, float("nan"), float("inf")):
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, "reset tolerance"):
                    RebotB601RTConfig(**{field: value})

    def test_reset_needs_a_configured_six_joint_pose(self):
        robot = self.make_robot()
        robot.connect()
        with self.assertRaisesRegex(ValueError, "reset_position_rad"):
            robot.reset_to_initial_position()
        self.native.set_target_joints.assert_not_called()
        for angles in ([0.0] * 5, [float("nan")] * 6):
            with self.assertRaisesRegex(ValueError, "six finite"):
                RebotB601RTConfig(reset_position_rad=angles)

    def test_exit_reset_is_independent_of_the_initial_pose(self):
        self.assertFalse(RebotB601RTConfig().reset_on_exit)
        self.assertIsNone(RebotB601RTConfig(reset_on_exit=True).reset_position_rad)
        for missing_side in ("left", "right"):
            with self.subTest(side=missing_side):
                BiRebotB601RTConfig(
                    reset_on_exit=True,
                    left_arm_config=RebotB601RTConfig(
                        channel="can0", reset_position_rad=None if missing_side == "left" else [0] * 6
                    ),
                    right_arm_config=RebotB601RTConfig(
                        channel="can1", reset_position_rad=None if missing_side == "right" else [0] * 6
                    ),
                )

    def test_reset_checks_native_fault_before_sending_goal(self):
        robot = self.make_robot(reset_position_rad=[0.0] * 6)
        robot.connect(go_to_start=False)
        self.native.get_state.return_value.fault = True
        self.native.get_state.return_value.fault_reason = "motor fault"
        with self.assertRaisesRegex(RuntimeError, "motor fault"):
            robot.reset_to_initial_position()
        self.native.set_target_joints.assert_not_called()
        # Also surface faults after this arm has settled, while a second arm
        # might still be returning in the shared exit-reset polling loop.
        with self.assertRaisesRegex(RuntimeError, "motor fault"):
            _ = robot.rt_moving

    def test_disconnect_waits_for_reset_feedback_in_both_control_modes(self):
        for mode in ("mit", "pos_vel"):
            with self.subTest(mode=mode):
                native = fake_native()
                self.native_factory.return_value = native
                robot = self.make_robot(
                    control_mode=mode, reset_on_exit=True, reset_position_rad=INITIAL_POSE, reset_tolerance_rad=0.06
                )
                robot.connect(go_to_start=False)
                camera = SimpleNamespace(is_connected=True, disconnect=Mock(), read_latest=Mock())
                robot.cameras["wrist"] = camera
                state = native.get_state.return_value
                state.command = state.q = [0.4] * 6
                with patch.object(driver.time, "monotonic", return_value=0.0) as clock:

                    def advance(_, native=native, state=state):
                        native.close.assert_not_called()
                        clock.return_value += 0.05
                        if clock.return_value >= 0.15:
                            state.command = [0.0] * 6
                            state.q = [0.05] * 6  # Within startup tolerance, outside exit tolerance.
                        if clock.return_value >= 0.4:
                            state.q = [0.02] * 6

                    with patch.object(driver.time, "sleep", side_effect=advance):
                        robot.disconnect()
                    self.assertGreaterEqual(clock.return_value, 0.5)
                self.assertGreater(native.set_target_joints.call_count, 1)
                for request in native.set_target_joints.call_args_list:
                    self.assertEqual(request, call([0.0] * 6 + [-1.2], max_joint_speed=0.5))
                native.close.assert_called_once()
                camera.disconnect.assert_called_once()
                camera.read_latest.assert_not_called()
                self.assertIsNone(robot._reset_target)
                robot.disconnect()
                native.close.assert_called_once()

    def test_disconnect_skips_homing_when_disabled_or_faulted(self):
        for enabled, reset, fault in ((False, True, False), (True, False, False), (True, True, True)):
            with self.subTest(enabled=enabled, reset=reset, fault=fault):
                native = fake_native()
                self.native_factory.return_value = native
                robot = self.make_robot(reset_on_exit=enabled, reset_position_rad=[0.0] * 6)
                robot.connect(go_to_start=False)
                native.get_state.return_value.fault = fault
                native.get_state.return_value.fault_reason = "motor fault"
                robot.disconnect(reset=reset)
                native.set_target_joints.assert_not_called()
                native.close.assert_called_once()

    def test_disconnect_releases_devices_if_exit_return_is_interrupted_or_fails(self):
        for failure in ("interrupt", "timeout", "fault", "write"):
            with self.subTest(failure=failure):
                native = fake_native()
                self.native_factory.return_value = native
                robot = self.make_robot(reset_on_exit=True, reset_position_rad=[0.0] * 6, reset_timeout_sec=0.1)
                robot.connect(go_to_start=False)
                robot.logger = Mock()
                camera = SimpleNamespace(is_connected=True, disconnect=Mock())
                robot.cameras["wrist"] = camera
                native.get_state.return_value.command = [0.4] * 6
                if failure == "write":
                    native.set_target_joints.side_effect = RuntimeError("CAN write failed")
                with patch.object(driver.time, "monotonic", return_value=0.0) as clock:

                    def advance(_, failure=failure, native=native):
                        if failure == "interrupt":
                            raise KeyboardInterrupt
                        if failure == "fault":
                            native.get_state.return_value.fault = True
                            native.get_state.return_value.fault_reason = "motor fault"
                        clock.return_value += 0.2

                    with patch.object(driver.time, "sleep", side_effect=advance):
                        robot.disconnect()
                native.close.assert_called_once()
                camera.disconnect.assert_called_once()
                self.assertFalse(robot.is_connected)
                if failure == "interrupt":
                    robot.logger.warn.assert_called_once()
                else:
                    robot.logger.error.assert_called()
                if failure == "timeout":
                    native.hold_position.assert_called_once()

    def test_tracking_trace_separates_limiting_from_motor_tracking_error(self):
        robot = self.make_robot(trace_joint_tracking=True, control_mode="mit", mit_velocity_feedforward=False)
        robot.connect()
        robot.logger = Mock()
        self.native.get_state.return_value = SimpleNamespace(
            fault=False,
            q=[0.1] * 7,
            target=[0.4] * 7,
            command=[0.2] * 7,
            command_velocity=[0.5] * 7,
            mit_command_velocity=[0.0] * 7,
        )
        robot.get_observation()
        message = robot.logger.info.call_args.args[0]
        self.assertIn("goal=[0.4", message)
        self.assertIn("command=[0.2", message)
        self.assertIn("measured=[0.1", message)
        self.assertIn("velocity=[0.5", message)
        self.assertIn("mit_dq=[0.0", message)

    def test_teleop_pose_uses_wxyz_quaternion_and_external_gripper(self):
        self.gripper_factory.return_value = fake_gripper()
        pose = np.array([[0.0, -1.0, 0.0, 0.1], [1.0, 0.0, 0.0, 0.2], [0.0, 0.0, 1.0, 0.3], [0.0, 0.0, 0.0, 1.0]])
        self.native.forward_kinematics.return_value = pose
        robot = self.make_robot(
            use_joint_observation=True,
            gripper=TaccapFollowerConfig(auto_discover_cameras=False),
        )
        robot.connect()
        np.testing.assert_allclose(
            robot.get_current_tcp_pose_quat(),
            [0.1, 0.2, 0.3, 2**-0.5, 0.0, 0.0, 2**-0.5, 0.73],
        )
        np.testing.assert_allclose(robot.get_current_tcp_pose_euler(), [0.1, 0.2, 0.3, 0, 0, np.pi / 2, 0.73])

    def test_reset_handoff_does_not_send_measured_sag_as_a_new_target(self):
        from rebot_rt._native import Robot as NativeRobot

        from lerobot.scripts.teleop_device_loops import Pico4Policy
        from lerobot.teleoperators.pico4 import Pico4, Pico4Config

        # Load only the kinematic model: no connect/start or hardware IO.
        model = NativeRobot(
            rebot_rt.Family.RS, "offline", str(ASSETS / TACCAP_URDFS["rs"]), "taccap_eef_link", has_gripper=False
        )
        self.native.forward_kinematics.side_effect = model.forward_kinematics
        self.gripper_factory.return_value = fake_gripper()
        robot = self.make_robot(
            motor_family="rs",
            gripper=TaccapFollowerConfig(auto_discover_cameras=False),
            interpolate_cartesian_targets=True,
        )
        robot.connect()
        state = self.native.get_state.return_value
        state.command = [*INITIAL_POSE, 0.0]
        # Reproduce the steady MIT tracking error from the startup log.
        state.q = [0.0033, 0.8067, 0.753, -0.5398, -0.0006, -0.0017, 0.0]
        commanded = model.forward_kinematics(state.command)
        measured = model.forward_kinematics(state.q)
        self.assertGreater(np.linalg.norm(commanded[:3, 3] - measured[:3, 3]), 0.005)
        np.testing.assert_allclose(robot.get_current_tcp_pose_quat()[:3], measured[:3, 3])
        self.assertAlmostEqual(robot.get_commanded_tcp_pose_quat()[7], 0.73)
        np.testing.assert_allclose(robot.get_commanded_tcp_pose_euler()[:3], commanded[:3, 3])

        teleop = Pico4(Pico4Config(calibration_dir=self.calibration_dir))
        teleop._is_connected = True
        teleop._xrt = SimpleNamespace(
            get_right_controller_pose=lambda: [0, 0, 0, 0, 0, 0, 1],
            get_right_grip=Mock(return_value=0.0),
            get_right_trigger=lambda: 0.0,
            get_A_button=lambda: False,
            get_B_button=lambda: False,
        )
        policy = Pico4Policy(use_commanded_pose=True)
        for grip in (0.0, 1.0):
            # Both a released grip and a stationary grip held across reset
            # must keep the arm's command rather than adding another sag step.
            teleop._xrt.get_right_grip.return_value = grip
            policy.resync(robot, teleop)
            robot.send_action(teleop.get_action())
            np.testing.assert_allclose(self.native.queue_target_pose.call_args.args[0], commanded, atol=1e-6)
            self.assertAlmostEqual(robot.get_observation()["tcp.z"], measured[2, 3])

    def test_failed_camera_connect_releases_arm_and_external_gripper(self):
        gripper = fake_gripper()
        self.gripper_factory.return_value = gripper
        camera = SimpleNamespace(is_connected=False, connect=Mock(side_effect=RuntimeError("camera failed")))
        self.enterContext(patch.object(driver, "make_cameras_from_configs", return_value={"scene": camera}))
        robot = self.make_robot(gripper=TaccapFollowerConfig(auto_discover_cameras=False))
        with self.assertRaisesRegex(RuntimeError, "camera failed"):
            robot.connect()
        self.native.close.assert_called_once()
        gripper.disconnect.assert_called_once()
        self.assertFalse(robot.is_connected)

    def test_startup_interrupt_releases_native_and_gripper(self):
        gripper = fake_gripper()
        self.gripper_factory.return_value = gripper
        camera = SimpleNamespace(is_connected=False, connect=Mock(side_effect=KeyboardInterrupt))
        self.enterContext(patch.object(driver, "make_cameras_from_configs", return_value={"wrist": camera}))
        robot = self.make_robot(gripper=TaccapFollowerConfig(auto_discover_cameras=False))
        with self.assertRaises(KeyboardInterrupt):
            robot.connect()
        self.native.close.assert_called_once()
        gripper.disconnect.assert_called_once()

    def test_disconnect_closes_native_after_peripheral_loss_and_is_idempotent(self):
        gripper = fake_gripper()
        self.gripper_factory.return_value = gripper
        robot = self.make_robot(gripper=TaccapFollowerConfig(auto_discover_cameras=False))
        robot.connect()
        gripper.is_connected = False
        self.assertFalse(robot.is_connected)
        robot.disconnect()
        robot.disconnect()
        self.native.close.assert_called_once()
        self.assertIsNone(robot._native)

    def test_exit_reset_still_runs_after_peripheral_loss_but_not_after_controller_loss(self):
        for lost in ("gripper", "camera", "native"):
            with self.subTest(lost=lost):
                native = fake_native()
                self.native_factory.return_value = native
                gripper = fake_gripper()
                self.gripper_factory.return_value = gripper
                robot = self.make_robot(reset_on_exit=True, gripper=TaccapFollowerConfig(auto_discover_cameras=False))
                robot.connect(go_to_start=False)
                robot.logger = Mock()
                camera = SimpleNamespace(is_connected=True, disconnect=Mock())
                robot.cameras["wrist"] = camera
                if lost == "gripper":
                    gripper.is_connected = False
                elif lost == "camera":
                    camera.is_connected = False
                else:
                    native.connected = False
                self.assertFalse(robot.is_connected)
                state = native.get_state.return_value
                state.command = state.q = [0.4] * 6

                def settle(_, state=state):
                    state.command = state.q = [0.0] * 6

                with patch.object(driver.time, "sleep", side_effect=settle):
                    robot.disconnect()
                if lost == "native":
                    native.set_target_joints.assert_not_called()
                    robot.logger.warn.assert_called_once()
                    self.assertIn("skipped", robot.logger.warn.call_args.args[0])
                else:
                    native.set_target_joints.assert_called_with([0.0] * 6 + [-1.2], max_joint_speed=0.5)
                    robot.logger.warn.assert_not_called()
                native.close.assert_called_once()
                self.assertIsNone(robot._native)

    def test_bimanual_exit_reset_survives_peripheral_loss_and_skips_on_controller_loss(self):
        for lost in ("camera", "native"):
            with self.subTest(lost=lost):
                left, right = fake_native(), fake_native()
                self.native_factory.side_effect = [left, right]
                robot = BiRebotB601RT(
                    BiRebotB601RTConfig(
                        calibration_dir=self.calibration_dir,
                        reset_on_exit=True,
                        left_arm_config=RebotB601RTConfig(channel="can1"),
                        right_arm_config=RebotB601RTConfig(channel="can0"),
                    )
                )
                robot.connect(go_to_start=False)
                robot.logger = Mock()
                if lost == "camera":
                    robot.right_arm.cameras["lost"] = SimpleNamespace(is_connected=False)
                else:
                    right.connected = False
                self.assertFalse(robot.is_connected)
                for native in (left, right):
                    native.get_state.return_value.command = native.get_state.return_value.q = [0.0] * 6
                with patch.object(driver.time, "sleep"):
                    robot.disconnect()
                for native in (left, right):
                    if lost == "camera":
                        native.set_target_joints.assert_called_with([0.0] * 6 + [-1.2], max_joint_speed=0.5)
                    else:
                        native.set_target_joints.assert_not_called()
                    native.stop.assert_called_once()
                    native.close.assert_called_once()
                if lost == "native":
                    self.assertIn("['right']", robot.logger.warn.call_args.args[0])
                else:
                    robot.logger.warn.assert_not_called()

    def test_bimanual_startup_interrupt_closes_both_native_controllers(self):
        left, right = fake_native(), fake_native()
        right.start.side_effect = KeyboardInterrupt
        self.native_factory.side_effect = [left, right]
        robot = BiRebotB601RT(
            BiRebotB601RTConfig(
                calibration_dir=self.calibration_dir,
                reset_on_exit=True,
                left_arm_config=RebotB601RTConfig(channel="can1", reset_on_exit=True, reset_position_rad=[0] * 6),
                right_arm_config=RebotB601RTConfig(channel="can0", reset_on_exit=True, reset_position_rad=[0] * 6),
            )
        )
        with self.assertRaises(KeyboardInterrupt):
            robot.connect()
        left.close.assert_called_once()
        right.close.assert_called_once()
        left.set_target_joints.assert_not_called()
        right.set_target_joints.assert_not_called()
        robot.disconnect()

    def test_bimanual_exit_reset_is_coordinated_and_closes_both_on_failure(self):
        for outcome in ("complete", "fault", "interrupt", "skip"):
            with self.subTest(outcome=outcome):
                left, right = fake_native(), fake_native()
                self.native_factory.side_effect = [left, right]
                robot = BiRebotB601RT(
                    BiRebotB601RTConfig(
                        calibration_dir=self.calibration_dir,
                        reset_on_exit=True,
                        left_arm_config=RebotB601RTConfig(
                            channel="can1",
                            reset_on_exit=True,
                            reset_position_rad=INITIAL_POSE,
                            reset_tolerance_rad=0.06,
                            exit_reset_tolerance_rad=0.02,
                        ),
                        right_arm_config=RebotB601RTConfig(
                            channel="can0",
                            reset_on_exit=True,
                            reset_position_rad=INITIAL_POSE,
                            reset_tolerance_rad=0.06,
                        ),
                    )
                )
                robot.connect(go_to_start=False)
                if outcome == "fault":
                    right.get_state.return_value.fault = True
                    right.get_state.return_value.fault_reason = "right motor fault"
                left.get_state.return_value.q = [0.025] * 6
                right.get_state.return_value.q = [0.05] * 6
                order = Mock()
                for side, native in (("left", left), ("right", right)):
                    order.attach_mock(native.stop, f"{side}_stop")
                    order.attach_mock(native.close, f"{side}_close")
                with (
                    patch.object(driver.time, "monotonic", return_value=0.0) as clock,
                    patch.object(
                        robot.left_arm, "_reset_to_position", wraps=robot.left_arm._reset_to_position
                    ) as left_reset,
                    patch.object(
                        robot.right_arm, "_reset_to_position", wraps=robot.right_arm._reset_to_position
                    ) as right_reset,
                ):

                    def advance(_, left=left, right=right, outcome=outcome):
                        # Both goals are submitted before waiting for either arm.
                        left_reset.assert_called_once_with([0.0] * 6, tolerance_rad=0.02)
                        right_reset.assert_called_once_with([0.0] * 6, tolerance_rad=0.03)
                        left.close.assert_not_called()
                        right.close.assert_not_called()
                        if outcome == "interrupt":
                            raise KeyboardInterrupt
                        clock.return_value += 0.05
                        if clock.return_value >= 0.2:
                            left.get_state.return_value.q = [0.0] * 6
                            right.get_state.return_value.q = [0.0] * 6

                    with patch.object(driver.time, "sleep", side_effect=advance):
                        robot.disconnect(reset=outcome != "skip")
                    if outcome in ("fault", "skip"):
                        left.set_target_joints.assert_not_called()
                        right.set_target_joints.assert_not_called()
                    else:
                        left_reset.assert_called_once()
                        right_reset.assert_called_once()
                    if outcome == "complete":
                        self.assertGreaterEqual(clock.return_value, 0.3)
                self.assertEqual(
                    order.mock_calls, [call.left_stop(), call.right_stop(), call.right_close(), call.left_close()]
                )
                self.assertFalse(robot.is_connected)
                robot.disconnect()
                left.close.assert_called_once()
                right.close.assert_called_once()

    def test_bimanual_disconnect_stops_both_even_when_one_peripheral_disappears(self):
        left, right = fake_native(), fake_native()
        self.native_factory.side_effect = [left, right]
        robot = BiRebotB601RT(
            BiRebotB601RTConfig(
                calibration_dir=self.calibration_dir,
                left_arm_config=RebotB601RTConfig(channel="can1"),
                right_arm_config=RebotB601RTConfig(channel="can0"),
            )
        )
        robot.connect()
        robot.right_arm.cameras["lost"] = SimpleNamespace(is_connected=False)
        self.assertFalse(robot.is_connected)
        robot.disconnect()
        robot.disconnect()
        for native in (left, right):
            native.stop.assert_called_once()
            native.close.assert_called_once()

    def test_bimanual_cameras_with_the_same_name_on_each_arm_are_both_kept(self):
        def cam_config():
            return OpenCVCameraConfig(index_or_path=0, width=4, height=2, fps=30)

        self.enterContext(
            patch.object(
                driver,
                "make_cameras_from_configs",
                side_effect=lambda cfgs: {name: Mock(use_depth=False) for name in cfgs},
            )
        )
        robot = BiRebotB601RT(
            BiRebotB601RTConfig(
                calibration_dir=self.calibration_dir,
                cameras={"scene": cam_config()},
                left_arm_config=RebotB601RTConfig(channel="can0", cameras={"wrist": cam_config()}),
                right_arm_config=RebotB601RTConfig(channel="can1", cameras={"wrist": cam_config()}),
            )
        )
        self.assertEqual(set(robot.cameras), {"scene", "left_wrist", "right_wrist"})
        self.assertIs(robot.cameras["left_wrist"], robot.left_arm.cameras["wrist"])
        self.assertIs(robot.cameras["right_wrist"], robot.right_arm.cameras["wrist"])
        self.assertIs(robot.cameras["scene"], robot.left_arm.cameras["scene"])
        image_keys = {k for k, v in robot.observation_features.items() if isinstance(v, tuple)}
        self.assertEqual(set(robot.cameras), image_keys)

    def test_taccap_camera_discovery_pins_the_separate_gripper(self):
        gripper = fake_gripper()
        self.gripper_factory.return_value = gripper
        inject = self.enterContext(patch.object(driver, "inject_taccap_cameras", return_value={"right": "/dev/fake"}))
        self.make_robot(motor_family="DM", gripper=TaccapFollowerConfig(side="right"))
        self.assertEqual(inject.call_args.kwargs["sides"], ("right",))
        self.assertEqual(gripper._mcu_device, "/dev/fake")

    def test_bimanual_arms_choose_their_own_model_and_gripper(self):
        self.gripper_factory.side_effect = [fake_gripper(), None]
        self.native_factory.side_effect = [fake_native(), fake_native()]
        config = BiRebotB601RTConfig(
            calibration_dir=self.calibration_dir,
            left_arm_config=RebotB601RTConfig(
                motor_family="dm",
                channel="can0",
                gripper=TaccapFollowerConfig(side="left", auto_discover_cameras=False),
            ),
            right_arm_config=RebotB601RTConfig(motor_family="RS", channel="can1"),
        )
        robot = BiRebotB601RT(config)
        try:
            robot.connect()
            left, right = self.native_factory.call_args_list
            self.assertEqual(Path(left.args[2]), ASSETS / TACCAP_URDFS["dm"])
            self.assertEqual(Path(right.args[2]), ASSETS / "RS/urdf/ReBot_Arm_RS.urdf")
            obs = robot.get_observation()
            self.assertEqual(obs["left_gripper.pos"], 0.73)
            self.assertEqual(obs["right_gripper.pos"], -1.2)
            left_pose, right_pose = robot.get_current_tcp_pose_quat()
            np.testing.assert_allclose(left_pose, [0, 0, 0, 1, 0, 0, 0, 0.73])
            np.testing.assert_allclose(right_pose, [0, 0, 0, 1, 0, 0, 0, -1.2])
            for arm, z in ((robot.left_arm, 0.4), (robot.right_arm, 0.5)):
                arm.get_commanded_tcp_pose_quat = Mock(return_value=np.array([0, 0, z, 1, 0, 0, 0, 0.6]))
            left_pose, right_pose = robot.get_commanded_tcp_pose_quat()
            self.assertEqual(left_pose[2], 0.4)
            self.assertEqual(right_pose[2], 0.5)
        finally:
            if robot.is_connected:
                robot.disconnect()

    def test_bimanual_reset_refreshes_both_watchdogs_and_finishes_together(self):
        left, right = fake_native(), fake_native()
        self.native_factory.side_effect = [left, right]
        config = BiRebotB601RTConfig(
            calibration_dir=self.calibration_dir,
            left_arm_config=RebotB601RTConfig(channel="can0", reset_position_rad=[0.0] * 6),
            right_arm_config=RebotB601RTConfig(channel="can1", reset_position_rad=[0.0] * 6),
        )
        robot = BiRebotB601RT(config)
        try:
            robot.connect(go_to_start=False)
            with patch.object(driver.time, "monotonic", return_value=0.0) as clock:
                robot.reset_to_initial_position()
                self.assertTrue(robot.rt_moving)
                self.assertGreaterEqual(left.set_target_joints.call_count, 2)
                self.assertGreaterEqual(right.set_target_joints.call_count, 2)
                clock.return_value = 0.11
                self.assertFalse(robot.rt_moving)
        finally:
            if robot.is_connected:
                robot.disconnect()

    def test_bimanual_reset_checks_both_arms_before_starting_either(self):
        left, right = fake_native(), fake_native()
        shutdown = Mock()
        shutdown.attach_mock(left.stop, "left_stop")
        shutdown.attach_mock(right.stop, "right_stop")
        shutdown.attach_mock(left.close, "left_close")
        shutdown.attach_mock(right.close, "right_close")
        self.native_factory.side_effect = [left, right]
        robot = BiRebotB601RT(
            BiRebotB601RTConfig(
                calibration_dir=self.calibration_dir,
                left_arm_config=RebotB601RTConfig(channel="can0", reset_position_rad=[0] * 6),
                right_arm_config=RebotB601RTConfig(channel="can1", reset_position_rad=[0] * 6),
            )
        )
        robot.connect(go_to_start=False)
        try:
            right.get_state.return_value.fault = True
            right.get_state.return_value.fault_reason = "right arm fault"
            with self.assertRaisesRegex(RuntimeError, "right arm fault"):
                robot.reset_to_initial_position()
            left.set_target_joints.assert_not_called()
            right.set_target_joints.assert_not_called()
        finally:
            robot.disconnect()
        left.stop.assert_called_once()
        right.stop.assert_called_once()
        self.assertEqual(
            shutdown.mock_calls, [call.left_stop(), call.right_stop(), call.right_close(), call.left_close()]
        )

    def test_startup_waits_for_the_initial_pose_in_both_control_modes(self):
        for mode in ("mit", "pos_vel"):
            for go_to_start in (True, False):
                with self.subTest(mode=mode, go_to_start=go_to_start):
                    native = fake_native()
                    self.native_factory.return_value = native
                    robot = self.make_robot(control_mode=mode, reset_position_rad=INITIAL_POSE)
                    state = native.get_state.return_value
                    with patch.object(driver.time, "monotonic", return_value=0.0) as clock:

                        def advance(_, state=state):
                            clock.return_value += 0.05
                            state.command = INITIAL_POSE + [-1.2]
                            if clock.return_value >= 0.15:
                                state.q = INITIAL_POSE + [-1.2]

                        with patch.object(driver.time, "sleep", side_effect=advance):
                            if go_to_start:
                                robot.connect()  # Startup movement is the default.
                            else:
                                robot.connect(go_to_start=False)
                        self.assertFalse(robot.rt_moving)
                        if go_to_start:
                            self.assertGreaterEqual(clock.return_value, 0.25)
                            native.set_target_joints.assert_called_with(INITIAL_POSE + [-1.2], max_joint_speed=0.5)
                        else:
                            native.set_target_joints.assert_not_called()
                    robot.disconnect()

    def test_bimanual_startup_moves_together_and_cleans_up_on_failure(self):
        for outcome in ("complete", "interrupt", "timeout", "fault", "skip"):
            with self.subTest(outcome=outcome):
                left, right = fake_native(), fake_native()
                self.native_factory.side_effect = [left, right]
                robot = BiRebotB601RT(
                    BiRebotB601RTConfig(
                        calibration_dir=self.calibration_dir,
                        left_arm_config=RebotB601RTConfig(
                            channel="can1", reset_position_rad=INITIAL_POSE, reset_timeout_sec=1.0
                        ),
                        right_arm_config=RebotB601RTConfig(
                            channel="can0", reset_position_rad=INITIAL_POSE, reset_timeout_sec=1.0
                        ),
                    )
                )

                def check_connected(*args, left=left, right=right, **kwargs):
                    self.assertTrue(left.connected)
                    self.assertTrue(right.connected)

                left.set_target_joints.side_effect = check_connected
                with patch.object(driver.time, "monotonic", return_value=0.0) as clock:

                    def advance(_, left=left, right=right, outcome=outcome):
                        for native in (left, right):
                            native.set_target_joints.assert_called_with(INITIAL_POSE + [-1.2], max_joint_speed=0.5)
                            native.close.assert_not_called()
                        if outcome == "interrupt":
                            raise KeyboardInterrupt
                        clock.return_value += 0.05
                        if outcome == "fault":
                            right.get_state.return_value.fault = True
                            right.get_state.return_value.fault_reason = "right motor fault"
                        elif outcome != "timeout":
                            left.get_state.return_value.q = left.get_state.return_value.command = INITIAL_POSE
                            if clock.return_value >= 0.25:
                                right.get_state.return_value.q = right.get_state.return_value.command = INITIAL_POSE

                    with patch.object(driver.time, "sleep", side_effect=advance):
                        if outcome in ("complete", "skip"):
                            robot.connect(go_to_start=outcome != "skip")
                            self.assertFalse(robot.rt_moving)
                            if outcome == "complete":
                                self.assertGreaterEqual(clock.return_value, 0.35)
                            else:
                                left.set_target_joints.assert_not_called()
                                right.set_target_joints.assert_not_called()
                        else:
                            error = {"interrupt": KeyboardInterrupt, "timeout": TimeoutError, "fault": RuntimeError}
                            with self.assertRaises(error[outcome]):
                                robot.connect()
                            self.assertFalse(robot.is_connected)
                            self.assertIsNone(robot.left_arm._reset_target)
                            self.assertIsNone(robot.right_arm._reset_target)
                robot.disconnect(reset=False)
                left.close.assert_called_once()
                right.close.assert_called_once()

    def test_startup_interrupt_and_timeout_return_to_zero_before_disabling(self):
        for bimanual in (False, True):
            for outcome in ("interrupt", "timeout", "second_interrupt", "fault"):
                with self.subTest(bimanual=bimanual, outcome=outcome):
                    natives = [fake_native() for _ in range(2 if bimanual else 1)]
                    self.native_factory.side_effect = natives
                    if bimanual:
                        robot = BiRebotB601RT(
                            BiRebotB601RTConfig(
                                calibration_dir=self.calibration_dir,
                                reset_on_exit=True,
                                left_arm_config=RebotB601RTConfig(
                                    channel="can1",
                                    control_mode="mit",
                                    reset_position_rad=INITIAL_POSE,
                                    reset_timeout_sec=1.0,
                                ),
                                right_arm_config=RebotB601RTConfig(
                                    channel="can0",
                                    control_mode="mit",
                                    reset_position_rad=INITIAL_POSE,
                                    reset_timeout_sec=1.0,
                                ),
                            )
                        )
                    else:
                        robot = self.make_robot(
                            control_mode="mit",
                            reset_on_exit=True,
                            reset_position_rad=INITIAL_POSE,
                            reset_timeout_sec=1.0,
                        )
                    # A stationary MIT offset keeps startup waiting, even after
                    # the command has reached the requested initial pose.
                    for native in natives:
                        state = native.get_state.return_value
                        state.command = INITIAL_POSE + [-1.2]
                        state.q = [0.0, 0.8, 0.75, -0.55, 0.0, 0.0, -1.2]
                    interrupted = False
                    with patch.object(driver.time, "monotonic", return_value=0.0) as clock:

                        def advance(_, natives=natives, outcome=outcome):
                            nonlocal interrupted
                            clock.return_value += 0.05
                            for native in natives:
                                native.stop.assert_not_called()
                                native.close.assert_not_called()
                            returning_to_zero = natives[0].set_target_joints.call_args.args[0][:6] == [0.0] * 6
                            if returning_to_zero:
                                for native in natives:
                                    native.set_target_joints.assert_called_with([0.0] * 6 + [-1.2], max_joint_speed=0.5)
                                if outcome == "second_interrupt":
                                    raise KeyboardInterrupt
                                # Delay actual feedback after the zero command:
                                # motors must stay enabled until both settle.
                                for native in natives:
                                    state = native.get_state.return_value
                                    if state.command[:6] == [0.0] * 6:
                                        state.q = [0.0] * 6 + [-1.2]
                                    state.command = [0.0] * 6 + [-1.2]
                            elif outcome == "fault":
                                natives[-1].get_state.return_value.fault = True
                                natives[-1].get_state.return_value.fault_reason = "motor fault"
                            elif outcome != "timeout" and not interrupted:
                                interrupted = True
                                raise KeyboardInterrupt

                        error = (
                            TimeoutError
                            if outcome == "timeout"
                            else (RuntimeError if outcome == "fault" else KeyboardInterrupt)
                        )
                        with patch.object(driver.time, "sleep", side_effect=advance), self.assertRaises(error):
                            robot.connect()
                    self.assertFalse(robot.is_connected)
                    for native in natives:
                        native.close.assert_called_once()
                        targets = [request.args[0][:6] for request in native.set_target_joints.call_args_list]
                        self.assertIn(INITIAL_POSE, targets)
                        if outcome == "fault":
                            self.assertNotIn([0.0] * 6, targets)
                        else:
                            self.assertIn([0.0] * 6, targets)
                            if outcome != "second_interrupt":
                                self.assertEqual(native.get_state.return_value.q[:6], [0.0] * 6)


if __name__ == "__main__":
    unittest.main()
