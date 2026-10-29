"""Hardware-free Elite gripper publisher and collision startup regressions."""

import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from lerobot.robots.bi_elite_cs66_rt.bi_elite_cs66_rt import BiEliteCS66RT
from lerobot.robots.bi_elite_cs66_rt.config_bi_elite_cs66_rt import BiEliteCS66RTControlMode


class PublisherTests(unittest.TestCase):
    def robot(self):
        robot = object.__new__(BiEliteCS66RT)
        robot._is_connected = False
        robot._driver = {}
        robot._dashboard = {}
        robot._rtsi = {}
        robot._station_gripper_stop = threading.Event()
        robot._station_gripper_thread = None
        robot._station_gripper_error = None
        robot._station_gripper_samples = None
        robot._station_gripper_readers = {}
        robot._coordinator = None
        robot.config = SimpleNamespace(native_collision_gripper_publish_hz=100.0, native_collision_profile=None)
        robot._gripper = {side: Mock() for side in ("left", "right")}
        for gripper in robot._gripper.values():
            gripper.config.type = "taccap_follower"
            gripper.is_connected = True
            gripper._gripper.get_gripper_config.return_value = SimpleNamespace(
                flags=1, min_open_rad=0.0, max_open_rad=1.0
            )
            gripper._loop.snapshot.return_value = SimpleNamespace(
                running=True, observation=SimpleNamespace(raw_pos=0.2, age_ms=1.0, seq=2, valid=True)
            )
        robot._station = Mock()
        robot.logger = Mock()
        self.addCleanup(setattr, robot, "_is_connected", False)
        return robot

    def station_robot(self):
        robot = self.robot()
        robot._driver = {side: Mock() for side in ("left", "right")}
        robot._rtsi = {side: Mock() for side in ("left", "right")}
        robot._coordinator = Mock()
        robot._coordinator.status.return_value = SimpleNamespace(running=True, error="")
        return robot

    def test_station_fault_prevents_servo_restart_and_prechecked_movej(self):
        robot = self.station_robot()
        robot._coordinator.status.return_value = SimpleNamespace(running=False, error="Right gripper is stale")
        for operation in (
            lambda: robot._start_servo_loop("right"),
            lambda: robot._move_j_blocking("right", [0.0] * 6, 3.0, path_checked=True),
            lambda: robot._check_joint_move({"right": [0.0] * 6}, "return-to-home"),
        ):
            with self.subTest(operation=operation), self.assertRaisesRegex(RuntimeError, "Right gripper is stale"):
                operation()
        robot._driver["right"].startServo.assert_not_called()
        robot._driver["right"].moveJ.assert_not_called()
        robot._station.check_joint_path.assert_not_called()

    def test_station_fault_during_servo_start_stops_the_resumed_sender(self):
        robot = self.station_robot()

        def fail_station(_rtsi):
            robot._coordinator.status.return_value = SimpleNamespace(running=False, error="gripper invalid")

        robot._driver["right"].startServo.side_effect = fail_station
        with self.assertRaisesRegex(RuntimeError, "gripper invalid"):
            robot._start_servo_loop("right")
        robot._driver["right"].stopServo.assert_called_once()

    def test_station_fault_takes_precedence_over_sender_deadline(self):
        robot = self.station_robot()
        robot._coordinator.status.return_value = SimpleNamespace(running=False, error="Right gripper is stale")
        robot._driver["right"].check_error.side_effect = RuntimeError("Native planner deadline exceeded")
        robot._log_native_fault_diagnostics = Mock()
        with self.assertRaisesRegex(RuntimeError, "Right gripper is stale"):
            robot._raise_servo_error_if_any("right")
        for driver in robot._driver.values():
            driver.stopServo.assert_called_once()
        robot._log_native_fault_diagnostics.assert_called_once()

    def test_stopped_coordinator_without_error_also_blocks_servo(self):
        robot = self.station_robot()
        robot._coordinator.status.return_value = SimpleNamespace(running=False, error="")
        with self.assertRaisesRegex(RuntimeError, "coordinator is not running"):
            robot._start_servo_loop("left")
        robot._driver["left"].startServo.assert_not_called()

    def connect_robot(self, events):
        robot = self.station_robot()
        robot._station = None
        robot.config.control_mode = BiEliteCS66RTControlMode.JOINT_SERVO
        robot.config.start_move_duration_s = 3.0
        robot._arm_ip = Mock(return_value="test")
        robot._connect_arm = Mock()
        robot.cameras = {"wrist": Mock()}
        robot.cameras["wrist"].connect.side_effect = lambda: events.append("camera")
        robot._start_station_layer = Mock(side_effect=lambda: events.append("station"))
        robot._arm_start_pose = Mock(return_value=[0.0] * 6)
        robot._move_j_blocking = Mock(side_effect=lambda *args, **kwargs: events.append("move"))
        for driver in robot._driver.values():
            driver.startServo.side_effect = lambda _rtsi: events.append("servo")
        robot._cleanup_after_failed_connect = Mock()
        return robot

    def test_connect_initializes_cameras_before_station_and_motion(self):
        events = []
        robot = self.connect_robot(events)
        with (
            patch("lerobot.robots.bi_elite_cs66_rt.bi_elite_cs66_rt._import_elite_sdk"),
            patch("lerobot.robots.bi_elite_cs66_rt.bi_elite_cs66_rt.attach_wrist_fisheye_calibration"),
        ):
            robot.connect()
        self.assertEqual(events[:2], ["camera", "station"])
        self.assertCountEqual(events[2:], ["move", "move", "servo", "servo"])
        self.assertTrue(robot.is_connected)

    def test_connect_does_not_restart_servo_after_station_fault_during_start_move(self):
        robot = self.connect_robot([])

        def fail_station(*args, **kwargs):
            robot._coordinator.status.return_value = SimpleNamespace(running=False, error="Right gripper is stale")

        robot._move_j_blocking.side_effect = fail_station
        robot._log_native_fault_diagnostics = Mock()
        with (
            patch("lerobot.robots.bi_elite_cs66_rt.bi_elite_cs66_rt._import_elite_sdk"),
            patch("lerobot.robots.bi_elite_cs66_rt.bi_elite_cs66_rt.attach_wrist_fisheye_calibration"),
            self.assertRaisesRegex(RuntimeError, "Right gripper is stale"),
        ):
            robot.connect()
        self.assertFalse(robot.is_connected)
        robot._cleanup_after_failed_connect.assert_called_once()
        for driver in robot._driver.values():
            driver.startServo.assert_not_called()
        self.assertNotIn("BiEliteCS66RT connected and ready.", [c.args[0] for c in robot.logger.info.call_args_list])

    def test_connect_checks_station_again_before_reporting_ready_without_start_move(self):
        robot = self.connect_robot([])
        started = set()

        def start_servo(side):
            started.add(side)
            if len(started) == 2:
                robot._coordinator.status.return_value = SimpleNamespace(running=False, error="gripper invalid")

        robot._start_servo_loop = Mock(side_effect=start_servo)
        robot._log_native_fault_diagnostics = Mock()
        with (
            patch("lerobot.robots.bi_elite_cs66_rt.bi_elite_cs66_rt._import_elite_sdk"),
            patch("lerobot.robots.bi_elite_cs66_rt.bi_elite_cs66_rt.attach_wrist_fisheye_calibration"),
            self.assertRaisesRegex(RuntimeError, "gripper invalid"),
        ):
            robot.connect(go_to_start=False)
        robot._move_j_blocking.assert_not_called()
        robot._cleanup_after_failed_connect.assert_called_once()
        self.assertFalse(robot.is_connected)
        self.assertNotIn("BiEliteCS66RT connected and ready.", [c.args[0] for c in robot.logger.info.call_args_list])

    def test_disconnect_skips_home_after_station_fault(self):
        robot = self.station_robot()
        robot._is_connected = True
        robot._coordinator.status.return_value = SimpleNamespace(running=False, error="gripper invalid")
        robot.cameras = {}
        robot.config.command_timeout_ms = 100
        robot._dashboard = dict.fromkeys(("left", "right"))
        robot._arm_home_pose = Mock(return_value=[0.0] * 6)
        robot._return_home_arm = Mock()
        robot.disconnect()
        robot._return_home_arm.assert_not_called()
        self.assertFalse(robot.is_connected)

    def test_joint_reset_checks_the_combined_sweep_once_before_moving(self):
        robot = self.station_robot()
        robot._is_connected = True
        robot.cameras = {}
        robot.config.control_mode = BiEliteCS66RTControlMode.JOINT_SERVO
        robot.config.reset_duration_s = 2.0
        robot.config.move_j_timeout_ms = 100
        robot._servo_lock = {side: threading.Lock() for side in ("left", "right")}
        robot._last_tcp_command = {}
        robot._arm_start_pose = Mock(side_effect=lambda side: [0.1 if side == "left" else 0.2] * 6)
        for side, rtsi in robot._rtsi.items():
            rtsi.getActualJointPositions.return_value = [0.0] * 6
            robot._driver[side].status.return_value = SimpleNamespace(running=False, error="")
        robot._station.check_joint_path.return_value = SimpleNamespace(safe=True)
        robot.reset_to_initial_position()
        robot._station.check_joint_path.assert_called_once()
        _, end = robot._station.check_joint_path.call_args.args
        self.assertEqual(end["left"].tolist(), [0.1] * 6)
        self.assertEqual(end["right"].tolist(), [0.2] * 6)
        for side in ("left", "right"):
            robot._driver[side].moveJ.assert_called_once()

    def test_paired_failure_keeps_the_other_arm_from_starting_its_move(self):
        robot = self.station_robot()
        robot._servo_lock = {side: threading.Lock() for side in ("left", "right")}
        robot._last_tcp_command = {}
        robot.config.move_j_timeout_ms = 100
        for driver in robot._driver.values():
            driver.status.return_value = SimpleNamespace(running=False, error="")

        def work(side, paired_failed):
            if side == "left":
                raise RuntimeError("Controller rejected/aborted MoveJ")
            self.assertTrue(paired_failed.wait(1.0))
            robot._move_j_blocking(side, [0.0] * 6, 1.0, path_checked=True, paired_failed=paired_failed)

        with self.assertRaisesRegex(RuntimeError, "rejected/aborted"):
            robot._run_arms_paired(work, "reset move")
        robot._driver["right"].moveJ.assert_not_called()
        logged = " ".join(c.args[0] for c in robot.logger.error.call_args_list)
        self.assertIn("left arm: reset move failed", logged)
        self.assertIn("right arm: reset move failed", logged)

    def test_disconnect_surfaces_a_failed_return_home_after_teardown(self):
        robot = self.station_robot()
        robot._is_connected = True
        robot._station = None
        robot.cameras = {}
        robot.config.command_timeout_ms = 100
        dashboards = {side: Mock() for side in ("left", "right")}
        robot._dashboard = dict(dashboards)
        rtsis = dict(robot._rtsi)
        robot._return_home_arm = Mock(side_effect=lambda side, _failed: (_ for _ in ()).throw(RuntimeError(side)))
        with self.assertRaisesRegex(RuntimeError, "Return-to-home failed"):
            robot.disconnect()
        self.assertFalse(robot.is_connected)
        for side in ("left", "right"):
            dashboards[side].disconnect.assert_called_once()
            rtsis[side].disconnect.assert_called_once()
            self.assertIsNone(robot._driver[side])

    def test_publisher_runs_with_held_arms_without_observations_then_joins(self):
        robot = self.robot()
        station = robot._station
        published = threading.Event()
        seen = []

        def publish(samples):
            seen.append(samples)
            if len(seen) >= 3:
                published.set()

        station.publish_gripper_snapshots.side_effect = publish
        robot._start_station_gripper_publisher()
        thread = robot._station_gripper_thread
        self.assertTrue(published.wait(0.5))
        robot._stop_station_layer()
        self.assertFalse(thread.is_alive())
        self.assertIsNone(robot._station)
        self.assertIsNone(robot._station_gripper_thread)
        self.assertEqual(robot._station_gripper_readers, {})
        self.assertGreaterEqual(len(seen), 3)
        for gripper in robot._gripper.values():
            gripper.get_gripper_position.assert_not_called()
            gripper.set_gripper_position.assert_not_called()

    def test_snapshot_exception_invalidates_the_whole_pair(self):
        robot = self.robot()
        station = robot._station
        readers = {"left": Mock()}
        readers["left"].read.side_effect = RuntimeError("sensor unavailable")
        stop = threading.Event()
        station.publish_gripper_snapshots.side_effect = lambda samples: stop.set()
        robot._station_gripper_loop(station, stop, readers)
        station.publish_gripper_snapshots.assert_called_once_with({})
        self.assertIn("sensor unavailable", robot._station_gripper_error)

    def test_publisher_cleanup_even_if_coordinator_stop_raises(self):
        robot = self.robot()
        robot._start_station_gripper_publisher()
        thread = robot._station_gripper_thread
        robot._coordinator = Mock()
        robot._coordinator.stop.side_effect = RuntimeError("stop failed")
        with self.assertRaisesRegex(RuntimeError, "stop failed"):
            robot._stop_station_layer()
        self.assertFalse(thread.is_alive())
        self.assertIsNone(robot._station)

    def test_stuck_reader_still_completes_cleanup_and_is_not_revived(self):
        robot = self.robot()
        release = threading.Event()
        entered = threading.Event()

        def stuck_read():
            entered.set()
            release.wait(5.0)
            return Mock()

        robot._start_station_gripper_publisher()
        old_thread = robot._station_gripper_thread
        old_stop = robot._station_gripper_stop
        robot._station_gripper_readers["left"].read = Mock(side_effect=stuck_read)
        self.assertTrue(entered.wait(1.0))
        robot._coordinator = Mock()
        with patch.object(old_thread, "join"), self.assertRaisesRegex(RuntimeError, "did not stop"):
            robot._stop_station_layer()
        self.assertIsNone(robot._coordinator)
        self.assertIsNone(robot._station)
        self.assertIsNone(robot._station_gripper_thread)
        self.assertEqual(robot._station_gripper_readers, {})

        # Reconnect: the new publisher gets its own Event; the stuck one stays stopped.
        robot._station = Mock()
        robot._start_station_gripper_publisher()
        self.assertTrue(old_stop.is_set())
        self.assertIsNot(robot._station_gripper_stop, old_stop)
        release.set()
        old_thread.join(1.0)
        self.assertFalse(old_thread.is_alive())
        robot._stop_station_layer()

    def test_coordinator_stop_error_is_not_masked_by_stuck_reader(self):
        robot = self.robot()
        thread = Mock()
        thread.is_alive.return_value = True
        robot._station_gripper_thread = thread
        robot._coordinator = Mock()
        robot._coordinator.stop.side_effect = RuntimeError("stop failed")
        with self.assertRaisesRegex(RuntimeError, "stop failed"):
            robot._stop_station_layer()
        self.assertIsNone(robot._coordinator)
        self.assertIsNone(robot._station)
        self.assertIsNone(robot._station_gripper_thread)

    def test_guard_disabled_does_not_start_snapshot_thread(self):
        robot = self.robot()
        robot._start_station_layer()
        self.assertIsNone(robot._station_gripper_thread)
        for gripper in robot._gripper.values():
            gripper._gripper.get_gripper_config.assert_not_called()
            gripper._loop.snapshot.assert_not_called()

    def test_retired_backend_is_rejected_even_with_a_custom_station(self):
        robot = self.robot()
        robot._gripper["left"].config.type = "serial"
        robot.config.native_collision_profile = "/custom/collision.json"
        robot._assembly = Mock()
        robot._connect_arm = Mock()
        with (
            patch.dict(sys.modules, {"libpyelite.collision": SimpleNamespace(StationLayer=Mock())}),
            patch("lerobot.robots.bi_elite_cs66_rt.bi_elite_cs66_rt._import_elite_sdk") as sdk,
            self.assertRaisesRegex(ValueError, "XGripper collision checking is unsupported"),
        ):
            robot.connect()
        sdk.assert_not_called()
        robot._connect_arm.assert_not_called()
        for gripper in robot._gripper.values():
            gripper.connect.assert_not_called()

    def test_calibration_read_failure_does_not_start_publisher(self):
        robot = self.robot()
        robot._gripper["right"]._gripper.get_gripper_config.side_effect = RuntimeError("no config reply")
        with self.assertRaisesRegex(RuntimeError, "no config reply"):
            robot._start_station_gripper_publisher()
        self.assertIsNone(robot._station_gripper_thread)
        self.assertEqual(robot._station_gripper_readers, {})
        robot._station.publish_gripper_snapshots.assert_not_called()

    def test_missing_calibration_is_rejected_before_any_hardware_connect(self):
        robot = self.robot()
        robot.config.native_collision_profile = "auto"
        robot._assembly = Mock()
        robot._configure_station_grippers = Mock(side_effect=ValueError("missing calibration"))
        robot._connect_arm = Mock()
        with (
            patch.dict(
                sys.modules,
                {
                    "libpyelite": SimpleNamespace(),
                    "libpyelite.collision": SimpleNamespace(StationLayer=Mock(return_value=Mock())),
                },
            ),
            patch("lerobot.robots.bi_elite_cs66_rt.bi_elite_cs66_rt._import_elite_sdk") as sdk,
            self.assertRaisesRegex(ValueError, "missing calibration"),
        ):
            robot.connect()
        sdk.assert_not_called()
        robot._connect_arm.assert_not_called()
        for gripper in robot._gripper.values():
            gripper.connect.assert_not_called()

    def test_failed_connection_cleanup_joins_publisher_before_disconnecting_grippers(self):
        robot = self.robot()
        robot.logger = Mock()
        robot.cameras = {}
        robot._stop_servo_loop = Mock()
        robot._driver = dict.fromkeys(("left", "right"))
        robot._rtsi = dict.fromkeys(("left", "right"))
        robot._dashboard = dict.fromkeys(("left", "right"))
        robot._start_station_gripper_publisher()
        thread = robot._station_gripper_thread
        for gripper in robot._gripper.values():
            gripper._is_connected = True
            gripper.disconnect.side_effect = lambda: self.assertFalse(thread.is_alive())
        robot._cleanup_after_failed_connect()
        self.assertFalse(thread.is_alive())
        self.assertIsNone(robot._station)
        for gripper in robot._gripper.values():
            gripper.disconnect.assert_called_once()


if __name__ == "__main__":
    unittest.main()
