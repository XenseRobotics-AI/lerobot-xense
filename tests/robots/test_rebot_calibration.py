"""Exercise B601 zero calibration with a fake bus; never access hardware."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import draccus

from lerobot.robots import RobotConfig
from lerobot.robots.rebot_b601_rt import RebotB601RTConfig, calibration as driver


class TestRebotCalibration(unittest.TestCase):
    family = "dm"

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config = RebotB601RTConfig(
            motor_family=self.family,
            channel="/dev/fake" if self.family == "dm" else "can0",
            id="follower1",
            calibration_dir=Path(self.directory.name),
        )
        self.path = self.config.calibration_dir / self.family / "follower1.json"
        self.path.parent.mkdir()

        def defaults(family):
            return [
                SimpleNamespace(
                    send_id=i,
                    feedback_id=i + 0x10 if family == "dm" else 0xFD,
                    model=("4340P" if i <= 3 else "4310") if family == "dm" else ("rs-06" if i <= 3 else "rs-00"),
                )
                for i in range(1, 8)
            ]

        self.enterContext(patch.object(driver, "_native_motor_defaults", side_effect=defaults))
        self.controller = MagicMock()
        self.motors = {}

        def add_motor(send_id, recv_id, model):
            motor = MagicMock()
            motor.get_register_u32.return_value = send_id
            motor.robstride_ping_host_id.return_value = (send_id, send_id)
            motor.get_register_f32.return_value = 0.0002
            motor.robstride_get_param_f32_host_id.return_value = 0.0002
            self.motors[send_id] = motor
            return motor

        self.controller.add_damiao_motor.side_effect = add_motor
        self.controller.add_robstride_motor.side_effect = add_motor
        self.open_controller = self.enterContext(patch.object(driver, "_open_controller", return_value=self.controller))
        self.enterContext(patch.object(driver.time, "sleep"))

    def run_calibration(self, answers=("",)):
        with patch("builtins.input", side_effect=answers):
            driver.calibrate_b601(self.config)

    def assert_closed(self):
        self.controller.close.assert_called_once()
        for motor in self.motors.values():
            motor.close.assert_called_once()

    def assert_no_zeros(self):
        for motor in self.motors.values():
            motor.set_zero_position.assert_not_called()

    def position_reader(self, motor):
        if self.family == "dm":
            return motor.get_register_f32
        return motor.robstride_get_param_f32_host_id

    def test_success_waits_for_pose_and_never_enables_or_changes_modes(self):
        def read_position(*args, **kwargs):
            for selected in self.motors.values():
                selected.set_zero_position.assert_called_once()
            self.assertFalse(self.path.exists())
            return 0.0002

        def confirm_pose(_):
            self.assert_no_zeros()
            self.controller.disable_all.assert_called_once()
            for motor in self.motors.values():
                self.position_reader(motor).side_effect = read_position
            return ""

        with patch("builtins.input", side_effect=confirm_pose), patch.object(driver, "logger") as logger:
            driver.calibrate_b601(self.config)

        self.assertEqual(set(self.motors), set(range(1, 8)))
        saved = json.loads(self.path.read_text())
        self.assertEqual(set(saved), {*(f"joint_{i}" for i in range(1, 7)), "gripper"})
        self.assertEqual(saved["gripper"]["id"], 7)
        self.assertEqual(saved["joint_1"]["homing_offset"], 0)
        for motor in self.motors.values():
            motor.set_zero_position.assert_called_once()
            motor.ensure_mode.assert_not_called()
            motor.enable.assert_not_called()
            motor.get_state.assert_not_called()
            if self.family == "dm":
                self.assertEqual(motor.get_register_u32.call_count, 2)
                motor.robstride_ping_host_id.assert_not_called()
                motor.get_register_f32.assert_called_once_with(80, timeout_ms=500)
                motor.robstride_get_param_f32_host_id.assert_not_called()
            else:
                self.assertEqual(motor.robstride_ping_host_id.call_count, 2)
                motor.get_register_u32.assert_not_called()
                motor.robstride_get_param_f32_host_id.assert_called_once_with(0x7019, 0xFD, timeout_ms=500)
                motor.get_register_f32.assert_not_called()
        output = "\n".join(call.args[0] for call in logger.info.call_args_list)
        self.assertEqual(output.count("Zero check PASS:"), 7)
        self.assertIn("+0.000200 rad (+0.011 deg)", output)
        self.controller.enable_all.assert_not_called()
        self.assert_closed()

    def test_nonzero_readback_reports_all_failing_motors_and_preserves_file(self):
        self.path.write_text('{"previous": true}')
        # A generous motion tolerance must not weaken the zero check.
        self.config.reset_tolerance_rad = 0.3

        def confirm(_):
            if not self.motors:
                return "c"
            self.position_reader(self.motors[3]).return_value = -0.025
            self.position_reader(self.motors[4]).return_value = 0.02
            return ""

        with patch("builtins.input", side_effect=confirm), self.assertRaises(RuntimeError) as caught:
            driver.calibrate_b601(self.config)
        message = str(caught.exception)
        self.assertIn("joint_3", message)
        self.assertIn("-0.025000 rad", message)
        self.assertIn("joint_4", message)
        self.assertIn("+0.020000 rad", message)
        self.assertIn("Zero commands have already been sent", message)
        for motor in self.motors.values():
            self.position_reader(motor).assert_called_once()
        self.assertEqual(self.path.read_text(), '{"previous": true}')
        self.assert_closed()

    def test_fresh_read_timeout_does_not_accept_cached_zero(self):
        self.config.command_timeout_sec = 0.2

        def confirm(_):
            self.motors[3].get_state.return_value = SimpleNamespace(pos=0.0)
            self.position_reader(self.motors[3]).side_effect = TimeoutError("no fresh reply")
            return ""

        with (
            patch("builtins.input", side_effect=confirm),
            self.assertRaisesRegex(RuntimeError, "joint_3.*no fresh reply"),
        ):
            driver.calibrate_b601(self.config)
        self.motors[3].get_state.assert_not_called()
        self.assertEqual(self.position_reader(self.motors[3]).call_args.kwargs["timeout_ms"], 200)
        self.assertFalse(self.path.exists())
        self.assert_closed()

    def test_invalid_readbacks_fail_without_saving(self):
        def confirm(_):
            for motor_id, position in enumerate((float("nan"), float("inf"), -float("inf"), None), 1):
                self.position_reader(self.motors[motor_id]).return_value = position
            return ""

        with patch("builtins.input", side_effect=confirm), self.assertRaises(RuntimeError) as caught:
            driver.calibrate_b601(self.config)
        for motor_id in range(1, 5):
            self.assertIn(f"joint_{motor_id}", str(caught.exception))
        self.assertFalse(self.path.exists())
        self.assert_closed()

    def test_interrupted_readback_cleans_up_and_does_not_save(self):
        def confirm(_):
            self.position_reader(self.motors[3]).side_effect = KeyboardInterrupt
            return ""

        with patch("builtins.input", side_effect=confirm), self.assertRaises(KeyboardInterrupt):
            driver.calibrate_b601(self.config)
        self.assertFalse(self.path.exists())
        self.assert_closed()

    def test_separate_usb_gripper_calibrates_only_six_arm_motors(self):
        self.config.has_gripper = False
        self.run_calibration()
        self.assertEqual(set(self.motors), set(range(1, 7)))
        self.assertNotIn("gripper", json.loads(self.path.read_text()))

    def test_missing_motor_aborts_before_any_zero_and_identifies_can_ids(self):
        original_add = self.controller.add_damiao_motor.side_effect

        def add_motor(send_id, recv_id, model):
            motor = original_add(send_id, recv_id, model)
            if send_id == 7:
                motor.get_register_u32.side_effect = TimeoutError("no reply")
                motor.robstride_ping_host_id.side_effect = TimeoutError("no reply")
            return motor

        self.controller.add_damiao_motor.side_effect = add_motor
        self.controller.add_robstride_motor.side_effect = add_motor
        with patch("builtins.input") as prompt:
            feedback = "17" if self.family == "dm" else "fd"
            with self.assertRaisesRegex(ConnectionError, rf"gripper \(send=0x07, feedback=0x{feedback}\)"):
                driver.calibrate_b601(self.config)
            prompt.assert_not_called()
        self.assert_no_zeros()
        self.assertFalse(self.path.exists())
        self.assert_closed()

    def test_disconnect_while_positioning_aborts_before_any_zero(self):
        def lose_communication(_):
            self.motors[3].get_register_u32.side_effect = TimeoutError("unplugged")
            self.motors[3].robstride_ping_host_id.side_effect = TimeoutError("unplugged")
            return ""

        with (
            patch("builtins.input", side_effect=lose_communication),
            self.assertRaisesRegex(ConnectionError, "joint_3"),
        ):
            driver.calibrate_b601(self.config)
        self.assert_no_zeros()
        self.assertFalse(self.path.exists())
        self.assert_closed()

    def test_keyboard_interrupt_cleans_up_without_setting_zero(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_calibration((KeyboardInterrupt(),))
        self.assert_no_zeros()
        self.assertFalse(self.path.exists())
        self.assert_closed()

    def test_keep_existing_file_does_not_open_bus(self):
        self.path.write_text('{"previous": true}')
        self.run_calibration()
        self.open_controller.assert_not_called()
        self.assertEqual(self.path.read_text(), '{"previous": true}')

    def test_partial_zero_failure_preserves_existing_file(self):
        self.path.write_text('{"previous": true}')

        def confirm(_):
            if not self.motors:
                return "c"
            self.motors[3].set_zero_position.side_effect = RuntimeError("write failed")
            return ""

        with (
            patch("builtins.input", side_effect=confirm),
            self.assertRaisesRegex(RuntimeError, "Failed to set zero for joint_3"),
        ):
            driver.calibrate_b601(self.config)
        self.motors[1].set_zero_position.assert_called_once()
        self.motors[4].set_zero_position.assert_not_called()
        self.assertEqual(self.path.read_text(), '{"previous": true}')
        self.assert_closed()

    def test_cleanup_closes_other_handles_after_one_close_fails(self):
        def interrupt(_):
            self.motors[1].close.side_effect = RuntimeError("close failed")
            raise KeyboardInterrupt

        with patch("builtins.input", side_effect=interrupt), self.assertRaises(KeyboardInterrupt):
            driver.calibrate_b601(self.config)
        self.assert_closed()

    def test_cli_dispatches_rt_config_without_starting_rt_control(self):
        from lerobot.scripts import lerobot_calibrate as cli

        arguments = [
            "lerobot-calibrate",
            "--robot.type=rebot_b601_rt",
            f"--robot.motor_family={self.family}",
            f"--robot.channel={'/dev/ttyACM0' if self.family == 'dm' else 'can0'}",
            "--robot.id=follower1",
            "--robot.has_gripper=false",
        ]
        with (
            patch.object(sys, "argv", arguments),
            patch.object(cli, "register_third_party_plugins"),
            patch.object(cli, "make_robot_from_config") as robot_factory,
            patch.object(driver, "calibrate_b601") as calibrate,
        ):
            cli.main()
        config = calibrate.call_args.args[0]
        self.assertEqual(config.motor_family, self.family)
        self.assertEqual(config.channel, "/dev/ttyACM0" if self.family == "dm" else "can0")
        self.assertFalse(config.has_gripper)
        robot_factory.assert_not_called()
        self.open_controller.assert_not_called()

    def test_cli_registers_type_without_importing_hardware_sdks(self):
        code = """
import sys
from importlib.abc import MetaPathFinder
class BlockHardware(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ('motorbridge', 'rebot_rt'):
            raise AssertionError('Hardware SDK imported while registering calibration CLI')
sys.meta_path.insert(0, BlockHardware())
from lerobot.scripts import lerobot_calibrate
from lerobot.robots import RobotConfig
assert 'rebot_b601_rt' in RobotConfig.get_known_choices()
assert 'seeed_b601_dm_follower' not in RobotConfig.get_known_choices()
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_motor_ids_and_models_follow_rt_overrides(self):
        config = draccus.decode(
            RobotConfig,
            {
                "type": "rebot_b601_rt",
                "motor_family": self.family,
                "motor_configs": [{"send_id": 8, "feedback_id": 24, "model": "4310"}, {}, {}, {}, {}, {}, {}],
                "has_gripper": False,
            },
        )
        targets = driver._motor_targets(config)
        self.assertEqual(len(targets), 6)
        self.assertEqual(targets[0], driver.MotorTarget("joint_1", 8, 24, "4310"))

    def test_duplicate_ids_are_rejected_before_opening_bus(self):
        self.config.motor_configs = [{"send_id": 2}, {}, {}, {}, {}, {}, {}]
        with self.assertRaisesRegex(ValueError, "unique"):
            self.run_calibration()
        self.open_controller.assert_not_called()

    def test_records_from_other_family_do_not_skip_calibration(self):
        other = self.config.calibration_dir / ("rs" if self.family == "dm" else "dm") / "follower1.json"
        other.parent.mkdir()
        other.write_text('{"other_family": true}')
        self.run_calibration()
        self.open_controller.assert_called_once()
        self.assertTrue(self.path.exists())
        self.assertEqual(other.read_text(), '{"other_family": true}')


class TestRebotRSCalibration(TestRebotCalibration):
    family = "rs"
