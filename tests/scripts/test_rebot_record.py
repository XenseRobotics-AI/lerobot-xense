"""Offline B601 recording: data schema, reset ownership, and shutdown order."""

import contextlib
import json
import unittest
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import draccus
import numpy as np
import pyarrow.parquet as pq

from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.robots.rebot_b601_rt.exit_policy import B601ControlFaultError
from lerobot.scripts import lerobot_record as cli
from tests.scripts.conftest_teleop_doubles import (
    FakeBiPico4Teleop,
    FakeDataset,
    FakeRobot,
    FakeSpaceMouseTeleop,
    FakeTeleop,
    fresh_events,
)
from tests.scripts.rebot_test_assets import isolated_rebot_assets

ROOT = Path(__file__).resolve().parents[2]


class RecordingRobot(FakeRobot):
    def __init__(self, bimanual=False, joints=False):
        super().__init__(iterations=100, bimanual=bimanual)
        self.name = "bi_rebot_b601_rt" if bimanual else "rebot_b601_rt"
        self.robot_type = self.name
        if bimanual:
            self.action_features = {
                f"{side}_{k}": v for side in ("left", "right") for k, v in self.action_features.items()
            }
        self.observation_features = (
            {f"joint_{i}.pos": float for i in range(12 if bimanual else 6)} if joints else self.action_features.copy()
        )
        self.observation_features["wrist"] = (2, 3, 3)
        self.is_connected = True
        self.config = SimpleNamespace(reset_on_exit=True)
        self.connect = Mock()
        self.disconnect = Mock(side_effect=lambda **kwargs: setattr(self, "is_connected", False))
        self.cameras = {"wrist": Mock()}
        self.reset_polls = 0

    def get_observation(self):
        super().get_observation()
        obs = dict.fromkeys(self.observation_features, 0.25)
        obs["wrist"] = np.full((2, 3, 3), self._obs_count, dtype=np.uint8)
        return obs

    def get_commanded_tcp_pose_quat(self):
        self.calls.append(("get_commanded_tcp_pose_quat",))
        pose = np.array([0.15, 0.25, 0.4, 1, 0, 0, 0, 0.5])
        return (pose, pose.copy()) if self._bimanual else pose

    def get_commanded_tcp_pose_euler(self):
        self.calls.append(("get_commanded_tcp_pose_euler",))
        return np.array([0.15, 0.25, 0.4, 0, 0, 0, 0.5])

    def send_action(self, action):
        self.assert_not_resetting()
        super().send_action(action)
        return {key: value * 0.5 for key, value in action.items()}

    def assert_not_resetting(self):
        if self.reset_polls:
            raise AssertionError("VR command sent during native reset")

    def reset_to_initial_position(self):
        super().reset_to_initial_position()
        self.reset_polls = 5

    @property
    def rt_moving(self):
        if self.reset_polls:
            self.reset_polls -= 1
            return True
        return False


def teleoperator(bimanual=False, reset_frames=()):
    teleop = FakeBiPico4Teleop(reset_frames=reset_frames) if bimanual else FakeTeleop(reset_frames=reset_frames)
    teleop.name = "bi_pico4" if bimanual else "pico4"
    teleop.is_connected = True
    teleop.connect = Mock()
    teleop.disconnect = Mock(side_effect=lambda: setattr(teleop, "is_connected", False))
    teleop._button_pressed = False
    teleop._previous_button = False

    def poll_buttons():
        # Pico can refresh button state independently, before computing this frame's action.
        teleop._button_pressed = teleop._frame + 1 in teleop._reset_frames

    def get_reset_button():
        pressed = teleop._button_pressed and not teleop._previous_button
        teleop._previous_button = teleop._button_pressed
        return pressed

    teleop.poll_buttons = Mock(side_effect=poll_buttons)
    teleop.get_reset_button = get_reset_button
    return teleop


def dataset_for(robot):
    return FakeDataset(
        fps=1,
        features={
            **hw_to_dataset_features(robot.action_features, "action"),
            **hw_to_dataset_features(robot.observation_features, "observation"),
        },
    )


class TestB601RecordLoop(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.enterContext(patch.object(cli, "logger"))
        self.enterContext(patch.object(cli.time, "perf_counter", side_effect=lambda: self.now))
        self.enterContext(patch.object(cli.time, "sleep", side_effect=self.advance))
        self.enterContext(patch.object(cli, "_record_loop_sleep", side_effect=lambda *args, **kwargs: self.advance(1)))

    def advance(self, amount):
        self.now += amount

    def test_single_and_dual_actions_use_accepted_targets_and_current_images(self):
        for dual in (False, True):
            with self.subTest(dual=dual):
                robot, teleop = RecordingRobot(dual), teleoperator(dual)
                dataset = dataset_for(robot)
                loop = partial(cli.rebot_b601_rt_record_loop, robot, teleop=teleop)
                loop(fresh_events(), 1, dataset=dataset, control_time_s=3, single_task="test task")
                self.assertEqual(len(dataset.frames), 3)
                self.assertEqual(dataset.frames[0]["action"].shape, (20 if dual else 10,))
                self.assertAlmostEqual(float(dataset.frames[0]["action"][0]), 0.055)
                for index, frame in enumerate(dataset.frames, 1):
                    self.assertEqual(frame["task"], "test task")
                    self.assertTrue((frame["observation.images.wrist"] == index).all())

    def test_each_episode_waits_for_fresh_cameras_before_entering_rt_loop(self):
        for dual in (False, True):
            with self.subTest(dual=dual):
                robot, teleop = RecordingRobot(dual), teleoperator(dual)
                dataset = dataset_for(robot)
                order = Mock()
                with (
                    patch(
                        "lerobot.robots.rebot_b601_rt.tactile_camera.wait_for_fresh_camera_frames",
                        new=order.wait,
                    ),
                    patch.object(cli, "run_rt_record_loop", new=order.record),
                ):
                    for _ in range(2):
                        cli.rebot_b601_rt_record_loop(robot, fresh_events(), 30, dataset, teleop)
                self.assertEqual([call[0] for call in order.mock_calls], ["wait", "record", "wait", "record"])
                order.wait.assert_called_with(robot.cameras)

    def test_camera_start_timeout_prevents_recording_or_sending_actions(self):
        robot, teleop = RecordingRobot(True), teleoperator(True)
        with (
            patch(
                "lerobot.robots.rebot_b601_rt.tactile_camera.wait_for_fresh_camera_frames",
                side_effect=TimeoutError("camera did not recover"),
            ),
            patch.object(cli, "run_rt_record_loop") as loop,
            self.assertRaisesRegex(TimeoutError, "camera did not recover"),
        ):
            cli.rebot_b601_rt_record_loop(robot, fresh_events(), 30, dataset_for(robot), teleop)
        loop.assert_not_called()
        self.assertNotIn("send_action", robot.names())

    def test_a_reset_records_measured_pose_and_resyncs_before_reading_next_action(self):
        for dual in (False, True):
            with self.subTest(dual=dual):
                robot, teleop = RecordingRobot(dual), teleoperator(dual, reset_frames={1})
                dataset = dataset_for(robot)
                read_action = teleop.get_action
                state = SimpleNamespace(resynced=False)

                def resync(*args, state=state):
                    state.resynced = True

                def action(read_action=read_action, state=state):
                    return dict.fromkeys(read_action(), 0.8 if state.resynced else 0.1)

                teleop.reset_to_pose = Mock(side_effect=resync)
                teleop.get_action = action
                loop = partial(cli.rebot_b601_rt_record_loop, robot, teleop=teleop)
                loop(fresh_events(), 1, dataset=dataset, control_time_s=7)
                self.assertEqual(robot.names().count("reset_to_initial_position"), 1)
                teleop.reset_to_pose.assert_called_once()
                pose = robot.get_commanded_tcp_pose_quat()
                expected = (pose[0][:7], pose[1][:7], pose[0][7], pose[1][7]) if dual else (pose[:7], pose[7])
                for actual, reference in zip(teleop.reset_to_pose.call_args.args, expected, strict=True):
                    np.testing.assert_allclose(actual, reference)
                self.assertNotIn("get_current_tcp_pose_quat", robot.names())
                # Shared RT semantics: skip the trigger frame, then pair the
                # previous observation with measured TCP while reset is moving.
                self.assertEqual(len(dataset.frames), 6)
                np.testing.assert_allclose(dataset.frames[0]["action"], 0.05)
                np.testing.assert_allclose(dataset.frames[1]["action"], 0.25)
                # After reset, get_action uses the new origin before any send_action.
                for frame in dataset.frames[2:]:
                    np.testing.assert_allclose(frame["action"], 0.4)
                for index, frame in zip((1, 2, 4, 5, 6, 7), dataset.frames, strict=True):
                    self.assertTrue((frame["observation.images.wrist"] == index).all())

    def test_reset_finishes_before_generic_preparation_sends_actions(self):
        for dual in (False, True):
            with self.subTest(dual=dual):
                robot, teleop = RecordingRobot(dual), teleoperator(dual)
                dataset = dataset_for(robot)
                loop = partial(cli.rebot_b601_rt_record_loop, robot, teleop=teleop)
                events = fresh_events()
                events["go_start"] = True
                loop(events, 1, dataset=dataset, control_time_s=0.5)
                self.assertFalse(robot.rt_moving)
                self.assertEqual(teleop.names().count("reset_to_pose"), 1)
                self.assertEqual(len(robot.sent_actions()), 0)
                self.assertEqual(len(dataset.frames), 0)
                with patch.object(cli, "precise_sleep", side_effect=self.advance):
                    cli.record_loop(robot, events, 1, teleop=teleop, control_time_s=0.5)
                self.assertEqual(len(robot.sent_actions()), 1)

    def test_joint_only_observations_are_rejected_before_recording(self):
        for dual in (False, True):
            with self.subTest(dual=dual):
                robot, teleop = RecordingRobot(dual, joints=True), teleoperator(dual)
                with self.assertRaisesRegex(ValueError, "use_joint_observation=false"):
                    cli.B601RecordPolicy(robot, teleop)
                self.assertEqual(robot.calls, [])
                robot.connect.assert_not_called()

    def test_record_loop_polls_controller_events_and_ignores_retake_intent(self):
        robot, teleop = RecordingRobot(True), teleoperator(True)
        loop = partial(cli.rebot_b601_rt_record_loop, robot, teleop=teleop)
        events = fresh_events()
        events["rerecord_episode"] = True
        refresh = Mock(side_effect=lambda: events.update(go_start=refresh.call_count == 1))
        events["_refresh_events"] = refresh
        loop(events, 1, control_time_s=2)
        self.assertEqual(refresh.call_count, 2)
        self.assertEqual(robot.names().count("reset_to_initial_position"), 1)
        self.assertEqual(len(robot.sent_actions()), 0)

    def test_spacemouse_reset(self):
        robot, teleop = RecordingRobot(), FakeSpaceMouseTeleop(both_pressed_frames={0})
        cli.rebot_b601_rt_record_loop(robot, fresh_events(), 1, teleop=teleop, control_time_s=1)
        self.assertEqual(robot.names().count("reset_to_initial_position"), 1)

    def test_ctrl_c_propagates_without_changing_session_events(self):
        robot, teleop = RecordingRobot(), teleoperator()
        dataset = dataset_for(robot)
        original = robot.get_observation
        robot.get_observation = Mock(side_effect=[original(), KeyboardInterrupt()])
        events = fresh_events()
        with self.assertRaises(KeyboardInterrupt):
            cli.rebot_b601_rt_record_loop(robot, events, 1, teleop=teleop, dataset=dataset, control_time_s=3)
        self.assertFalse(events["stop_recording"])
        self.assertFalse(events["exit_early"])
        self.assertEqual(len(dataset.frames), 1)
        # The entrypoint, not the loop, owns device shutdown.
        robot.disconnect.assert_not_called()

    def test_image_writer_uses_common_exception_cleanup(self):
        for error in (RuntimeError("camera disconnected"), KeyboardInterrupt()):
            with self.subTest(error=error):
                robot, teleop = RecordingRobot(), teleoperator()
                dataset = dataset_for(robot)
                dataset.image_writer = Mock()
                robot.get_observation = Mock(side_effect=error)
                with self.assertRaises(type(error)):
                    cli.rebot_b601_rt_record_loop(
                        robot, fresh_events(), 1, teleop=teleop, dataset=dataset, control_time_s=1
                    )
                if isinstance(error, Exception):
                    dataset.image_writer.stop.assert_called_once()
                else:
                    dataset.image_writer.stop.assert_not_called()

    def test_resync_failure_uses_shared_loop_error_handling(self):
        robot, teleop = RecordingRobot(), teleoperator(reset_frames={0})
        loop = partial(cli.rebot_b601_rt_record_loop, robot, teleop=teleop)
        teleop.reset_to_pose = Mock(side_effect=RuntimeError("pose unavailable"))
        loop(fresh_events(), 1, control_time_s=6)
        cli.logger.error.assert_any_call("Failed to sync teleop after RT reset: pose unavailable")
        teleop.reset_to_pose.assert_called_once()
        robot.disconnect.assert_not_called()


class TestB601RecordEntrypoint(unittest.TestCase):
    def setUp(self):
        self.enterContext(isolated_rebot_assets())

    def setup_record(self, dual=True, episodes=1):
        robot, teleop = RecordingRobot(dual), teleoperator(dual)
        recipe = "bi_rebot_b601_rt" if dual else "rebot_b601_rt"
        cfg = draccus.parse(
            cli.RecordConfig,
            config_path=ROOT / f"recipes/record/{recipe}/rs-taccap05.yaml",
            args=[
                "--display_data=false",
                "--play_sounds=false",
                "--resume=false",
                f"--dataset.num_episodes={episodes}",
            ],
        )
        robot.config = cfg.robot
        self.assertEqual(cfg.robot.reset_on_exit, True)
        arms = [cfg.robot.left_arm_config, cfg.robot.right_arm_config] if dual else [cfg.robot]
        for arm in arms:
            self.assertEqual(arm.control_mode, "mit")
            self.assertFalse(arm.mit_velocity_feedforward)
            self.assertEqual(arm.reset_position_rad, [0, 0.8, 0.8, -0.5, 0, 0])
        if dual:
            self.assertEqual([arm.channel for arm in arms], ["can1", "can0"])
        dataset = Mock(fps=30, num_episodes=0, episode_buffer={"size": 0})
        order = []
        dataset.save_episode.side_effect = lambda: order.append("save")
        dataset.finalize.side_effect = lambda: order.append("finalize")
        robot.disconnect.side_effect = lambda **kwargs: (
            order.append("disconnect"),
            setattr(robot, "is_connected", False),
        )

        @contextlib.contextmanager
        def manager(_dataset):
            try:
                yield
            finally:
                order.append("encoder_cleanup")

        for name, replacement in (
            ("make_robot_from_config", Mock(return_value=robot)),
            ("make_teleoperator_from_config", Mock(return_value=teleop)),
            ("VideoEncodingManager", manager),
            ("init_keyboard_listener", Mock(return_value=(None, fresh_events()))),
            ("is_headless", Mock(return_value=True)),
            ("init_logging", Mock()),
            ("log_say", Mock()),
            ("logger", Mock()),
        ):
            self.enterContext(patch.object(cli, name, replacement))
        self.enterContext(patch.object(cli.LeRobotDataset, "create", return_value=dataset))
        self.enterContext(patch.object(cli.time, "sleep"))
        return cfg, robot, teleop, dataset, order

    def test_recipes_seed_commanded_poses_and_close_after_encoding(self):
        for dual in (False, True):
            with self.subTest(dual=dual):
                cfg, robot, teleop, dataset, order = self.setup_record(dual)
                dataset.episode_buffer["size"] = 2
                with patch.object(cli, "rebot_b601_rt_record_loop") as loop:
                    cli.record(cfg)
                pose = robot.get_commanded_tcp_pose_quat()
                expected = (
                    {"left_tcp_pose_quat": pose[0], "right_tcp_pose_quat": pose[1]}
                    if dual
                    else {"current_tcp_pose_quat": pose}
                )
                for key, value in expected.items():
                    np.testing.assert_allclose(teleop.connect.call_args.kwargs[key], value)
                self.assertNotIn("get_current_tcp_pose_quat", robot.names())
                self.assertEqual(loop.call_count, 1)
                robot.disconnect.assert_called_once_with(reset=True)
                self.assertEqual(order, ["save", "encoder_cleanup", "finalize", "disconnect"])
                dataset.stop_image_writer.assert_not_called()
                dataset.push_to_hub.assert_not_called()

    def test_preparation_uses_generic_loop_and_retake_discards_episode(self):
        cfg, robot, _, dataset, _ = self.setup_record(episodes=2)
        phases = []

        def run(**kwargs):
            phases.append(kwargs.get("dataset"))
            dataset.episode_buffer["size"] = 2
            if len(phases) == 1:
                kwargs["events"]["rerecord_episode"] = True

        def prepare(**kwargs):
            self.assertIs(kwargs["robot"], robot)
            self.assertNotIn("dataset", kwargs)
            self.assertEqual(kwargs["control_time_s"], cfg.dataset.reset_time_s)
            phases.append(None)

        with (
            patch.object(cli, "rebot_b601_rt_record_loop", side_effect=run) as capture,
            patch.object(cli, "record_loop", side_effect=prepare) as generic,
        ):
            cli.record(cfg)
        self.assertEqual(phases, [dataset, None, dataset, None, dataset])
        self.assertEqual(dataset.save_episode.call_count, 2)
        dataset.clear_episode_buffer.assert_called_once()
        self.assertEqual(capture.call_count, 3)
        self.assertEqual(generic.call_count, 2)

    def test_ctrl_c_before_first_frame_skips_episode_save(self):
        cfg, robot, _, dataset, _ = self.setup_record()
        robot.get_observation = Mock(side_effect=KeyboardInterrupt)
        with self.assertRaises(KeyboardInterrupt):
            cli.record(cfg)
        dataset.save_episode.assert_not_called()
        dataset.clear_episode_buffer.assert_not_called()
        dataset.stop_image_writer.assert_not_called()
        robot.disconnect.assert_called_once_with(reset=True)

    def test_record_fault_uses_common_disconnect_after_finalize(self):
        cfg, robot, _, dataset, order = self.setup_record()
        with (
            patch.object(cli, "rebot_b601_rt_record_loop", side_effect=RuntimeError("camera disconnected")),
            self.assertRaisesRegex(RuntimeError, "camera disconnected"),
        ):
            cli.record(cfg)
        self.assertNotIn("reset_to_initial_position", robot.names())
        robot.disconnect.assert_called_once_with(reset=True)
        self.assertEqual(order, ["encoder_cleanup", "finalize", "disconnect"])
        dataset.save_episode.assert_not_called()

    def test_arm_control_fault_disconnects_without_exit_return(self):
        for error in (
            B601ControlFaultError("B601 C++ control fault: motor 3 overcurrent"),
            RuntimeError("control fault: x"),
        ):
            with self.subTest(error=error):
                cfg, robot, _, _, order = self.setup_record()
                with patch.object(cli, "rebot_b601_rt_record_loop", side_effect=error), self.assertRaises(RuntimeError):
                    cli.record(cfg)
                robot.disconnect.assert_called_once_with(reset=False)
                self.assertEqual(order, ["encoder_cleanup", "finalize", "disconnect"])

    def test_lost_peripheral_still_closes_and_returns_the_arm(self):
        # B601 is_connected is False once any camera/gripper drops; the arm
        # controller must still be closed, with the configured return.
        cfg, robot, _, _, order = self.setup_record()

        def drop_camera(**kwargs):
            robot.is_connected = False
            raise TimeoutError("wrist camera frame timeout")

        with (
            patch.object(cli, "rebot_b601_rt_record_loop", side_effect=drop_camera),
            self.assertRaises(TimeoutError),
        ):
            cli.record(cfg)
        robot.disconnect.assert_called_once_with(reset=True)
        self.assertEqual(order, ["encoder_cleanup", "finalize", "disconnect"])

    def test_encoder_cleanup_failure_uses_common_disconnect(self):
        cfg, robot, teleop, _, order = self.setup_record()

        @contextlib.contextmanager
        def failed_encoder_cleanup(_dataset):
            yield
            order.append("encoder_cleanup")
            raise RuntimeError("encoder cleanup failed")

        with (
            patch.object(cli, "rebot_b601_rt_record_loop"),
            patch.object(cli, "VideoEncodingManager", failed_encoder_cleanup),
            self.assertRaisesRegex(RuntimeError, "encoder cleanup failed"),
        ):
            cli.record(cfg)
        self.assertNotIn("reset_to_initial_position", robot.names())
        self.assertEqual(order, ["save", "encoder_cleanup", "finalize", "disconnect"])
        robot.disconnect.assert_called_once_with(reset=True)
        teleop.disconnect.assert_called_once()

    def test_ctrl_c_skips_current_episode_and_disconnects_through_driver(self):
        cfg, robot, teleop, dataset, order = self.setup_record()
        dataset.features = dataset_for(robot).features
        original = robot.get_observation
        robot.get_observation = Mock(side_effect=[original(), KeyboardInterrupt()])
        dataset.episode_buffer["size"] = 1
        with patch.object(cli, "_record_loop_sleep"), self.assertRaises(KeyboardInterrupt):
            cli.record(cfg)
        self.assertEqual(dataset.add_frame.call_count, 1)
        robot.disconnect.assert_called_once_with(reset=True)
        self.assertEqual(order, ["encoder_cleanup", "finalize", "disconnect"])
        dataset.save_episode.assert_not_called()
        dataset.stop_image_writer.assert_not_called()
        robot.disconnect.assert_called_once()
        teleop.disconnect.assert_called_once()

    def test_exit_reset_configuration_is_left_to_driver(self):
        cfg, robot, _, dataset, _ = self.setup_record()
        cfg.robot.reset_on_exit = False
        dataset.episode_buffer["size"] = 2
        with patch.object(cli, "rebot_b601_rt_record_loop"):
            cli.record(cfg)
        self.assertNotIn("reset_to_initial_position", robot.names())
        robot.disconnect.assert_called_once_with(reset=True)

    def test_spacemouse_connection_uses_commanded_euler_pose(self):
        cfg, robot, teleop, dataset, _ = self.setup_record(dual=False)
        cfg.teleop = cli.spacemouse.SpacemouseConfig()
        teleop.name = "spacemouse"
        dataset.episode_buffer["size"] = 2
        with patch.object(cli, "rebot_b601_rt_record_loop"):
            cli.record(cfg)
        np.testing.assert_allclose(
            teleop.connect.call_args.kwargs["current_tcp_pose_euler"], robot.get_commanded_tcp_pose_euler()
        )
        self.assertNotIn("get_current_tcp_pose_euler", robot.names())

    def test_teleop_startup_interrupt_uses_common_disconnect(self):
        cfg, robot, teleop, _, order = self.setup_record()
        teleop.connect.side_effect = KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            cli.record(cfg)
        self.assertNotIn("reset_to_initial_position", robot.names())
        robot.disconnect.assert_called_once_with(reset=True)
        self.assertEqual(order, ["finalize", "disconnect"])


class TestExistingRobotRecordEntrypoint(unittest.TestCase):
    """Adding B601 must preserve the other robots' connection and exit order."""

    def check_existing_robots(self, error=None):
        cases = (
            ("so100_follower", "so100_leader", "record_loop"),
            ("flexiv_rizon4_rt", "pico4", "flexiv_rizon4_rt_record_loop"),
            ("elite_cs66_rt", "pico4", "elite_cs66_rt_record_loop"),
            ("elite_cs66_rt", "spacemouse", "elite_cs66_rt_record_loop"),
            ("bi_flexiv_rizon4_rt", "bi_pico4", "flexiv_rizon4_rt_record_loop"),
            ("bi_elite_cs66_rt", "bi_pico4", "flexiv_rizon4_rt_record_loop"),
        )
        for robot_type, teleop_type, record_loop in cases:
            with self.subTest(robot=robot_type, teleop=teleop_type), contextlib.ExitStack() as patches:
                dual = robot_type.startswith("bi_")
                robot, teleop = RecordingRobot(dual), teleoperator(dual)
                robot.name, teleop.name = robot_type, teleop_type
                teleop.pre_init = Mock()
                cfg = cli.RecordConfig(
                    robot=SimpleNamespace(type=robot_type),
                    teleop=SimpleNamespace(type=teleop_type, ori_sensitivity=1.0),
                    dataset=cli.DatasetRecordConfig(
                        repo_id="test/record", single_task="test", num_episodes=2, push_to_hub=False
                    ),
                    play_sounds=False,
                )
                dataset = Mock(num_episodes=0)
                order, phases = [], []
                dataset.save_episode.side_effect = lambda order=order: order.append("save")
                dataset.finalize.side_effect = lambda order=order: order.append("finalize")
                robot.disconnect.side_effect = lambda order=order: order.append("robot_disconnect")
                teleop.disconnect.side_effect = lambda order=order: order.append("teleop_disconnect")

                @contextlib.contextmanager
                def video_manager(_dataset, order=order):
                    try:
                        yield
                    finally:
                        order.append("encoder_cleanup")

                def capture(order=order, phases=phases, **kwargs):
                    phases.append(kwargs.get("dataset"))
                    order.append("capture" if kwargs.get("dataset") is not None else "prepare")
                    if error is not None:
                        raise error

                for name, replacement in (
                    ("make_robot_from_config", Mock(return_value=robot)),
                    ("make_teleoperator_from_config", Mock(return_value=teleop)),
                    ("VideoEncodingManager", video_manager),
                    ("init_keyboard_listener", Mock(return_value=(None, fresh_events()))),
                    ("is_headless", Mock(return_value=True)),
                    ("init_logging", Mock()),
                    ("log_say", Mock()),
                    ("logger", Mock()),
                ):
                    patches.enter_context(patch.object(cli, name, replacement))
                # Elite + Pico4 bypasses make_teleoperator_from_config for the Elite grip-anchor adapter.
                patches.enter_context(
                    patch(
                        "lerobot.robots.bi_elite_cs66_rt.pico4.make_elite_pico4_teleoperator",
                        Mock(return_value=teleop),
                    )
                )
                patches.enter_context(patch.object(cli.LeRobotDataset, "create", return_value=dataset))
                loops = {
                    name: patches.enter_context(patch.object(cli, name, side_effect=capture))
                    for name in ("record_loop", "flexiv_rizon4_rt_record_loop", "elite_cs66_rt_record_loop")
                }
                b601_loop = patches.enter_context(patch.object(cli, "rebot_b601_rt_record_loop"))
                b601_policy = patches.enter_context(patch.object(cli, "B601RecordPolicy"))

                with self.assertRaises(type(error)) if error is not None else contextlib.nullcontext():
                    cli.record(cfg)

                expected = ["capture"] if error is not None else ["capture", "prepare", "save", "capture", "save"]
                self.assertEqual(
                    order, [*expected, "encoder_cleanup", "finalize", "robot_disconnect", "teleop_disconnect"]
                )
                self.assertEqual(phases, [dataset] if error is not None else [dataset, None, dataset])
                self.assertGreater(loops[record_loop].call_count, 0)
                for name, loop in loops.items():
                    if name not in {record_loop, "record_loop"}:
                        loop.assert_not_called()
                b601_loop.assert_not_called()
                b601_policy.assert_not_called()
                dataset.stop_image_writer.assert_not_called()
                robot.connect.assert_called_once_with(
                    **({} if robot_type == "so100_follower" else {"go_to_start": True})
                )
                if dual:
                    teleop.pre_init.assert_called_once()
                    left, right = robot.get_current_tcp_pose_quat()
                    expected_pose = {"left_tcp_pose_quat": left, "right_tcp_pose_quat": right}
                elif teleop_type == "pico4":
                    expected_pose = {"current_tcp_pose_quat": robot.get_current_tcp_pose_quat()}
                elif teleop_type == "spacemouse":
                    expected_pose = {"current_tcp_pose_euler": robot.get_current_tcp_pose_euler()}
                else:
                    expected_pose = {}
                self.assertEqual(set(teleop.connect.call_args.kwargs), set(expected_pose))
                for key, value in expected_pose.items():
                    np.testing.assert_allclose(teleop.connect.call_args.kwargs[key], value)
                if robot_type == "bi_elite_cs66_rt":
                    self.assertEqual(cfg.teleop.ori_sensitivity, 0.5)

    def test_normal_recording_keeps_existing_connection_phases_and_cleanup(self):
        self.check_existing_robots()

    def test_record_fault_keeps_existing_cleanup_order(self):
        self.check_existing_robots(RuntimeError("capture failed"))

    def test_ctrl_c_keeps_existing_cleanup_order(self):
        self.check_existing_robots(KeyboardInterrupt())


class TestB601DatasetRoundTrip(unittest.TestCase):
    def setUp(self):
        self.enterContext(isolated_rebot_assets())

    def test_dual_arm_streaming_video_save_and_resume(self):
        # Exercise the real dataset/encoder and CLI, replacing only hardware,
        # keyboard and speech. H.264 uses CPU so this needs no GPU or camera.
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "dataset"
            cfg = draccus.parse(
                cli.RecordConfig,
                config_path=ROOT / "recipes/record/bi_rebot_b601_rt/rs-taccap05.yaml",
                args=[
                    "--display_data=false",
                    "--play_sounds=false",
                    "--dataset.num_episodes=1",
                    "--dataset.vcodec=h264",
                    "--dataset.encoder_threads=1",
                    f"--dataset.root={root}",
                ],
            )
            for resume in (False, True):
                cfg.resume = resume
                robot, teleop = RecordingRobot(True), teleoperator(True)
                robot.config = cfg.robot
                robot.observation_features["wrist"] = (16, 16, 3)
                original_read = robot.get_observation

                def read_frame(original_read=original_read):
                    obs = original_read()
                    obs["wrist"] = np.full((16, 16, 3), 64, dtype=np.uint8)
                    return obs

                robot.get_observation = read_frame
                events = fresh_events()

                def end_after_two_frames(*_args, events=events, robot=robot, **_kwargs):
                    events["exit_early"] = robot._obs_count == 2

                with (
                    patch.object(cli, "make_robot_from_config", return_value=robot),
                    patch.object(cli, "make_teleoperator_from_config", return_value=teleop),
                    patch.object(cli, "init_keyboard_listener", return_value=(None, events)),
                    patch.object(cli, "is_headless", return_value=True),
                    patch.object(cli, "init_logging"),
                    patch.object(cli, "log_say"),
                    patch.object(cli, "_record_loop_sleep", side_effect=end_after_two_frames),
                    patch.object(cli, "logger"),
                ):
                    dataset = cli.record(cfg)
                self.assertEqual(dataset.num_episodes, 2 if resume else 1)
                robot.disconnect.assert_called_once()
            info = json.loads((root / "meta/info.json").read_text())
            self.assertEqual(info["total_frames"], 4)
            self.assertEqual(info["total_episodes"], 2)
            self.assertEqual(info["features"]["action"]["shape"], [20])
            self.assertEqual(info["features"]["observation.images.wrist"]["dtype"], "video")
            tables = [pq.read_table(path).to_pydict() for path in root.glob("data/**/*.parquet")]
            self.assertEqual(sum(len(table["action"]) for table in tables), 4)
            np.testing.assert_allclose(tables[0]["action"][0][0], 0.055)
            self.assertTrue(list(root.glob("videos/**/*.mp4")))


if __name__ == "__main__":
    unittest.main()
