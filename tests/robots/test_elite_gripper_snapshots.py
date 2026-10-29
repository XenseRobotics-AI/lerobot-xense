"""Elite collision samples preserve actual position, age and connection identity."""

import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import yaml

from lerobot.grippers.taccap import taccap_follower as taccap_module
from lerobot.grippers.taccap.configuration_taccap import TaccapFollowerConfig
from lerobot.grippers.taccap.taccap_follower import TaccapFollower
from lerobot.robots.bi_elite_cs66_rt.gripper_snapshots import TaccapGripperSnapshotReader


class SnapshotTests(unittest.TestCase):
    def follower(self):
        gripper = TaccapFollower(TaccapFollowerConfig(side="left", init_open=False))
        gripper._is_connected = True
        gripper._gripper = Mock()
        gripper._gripper.get_gripper_config.return_value = SimpleNamespace(
            flags=0x0003, min_open_rad=0.1, max_open_rad=1.1
        )
        gripper._loop = Mock()
        gripper._loop.snapshot.return_value = SimpleNamespace(
            running=True, observation=SimpleNamespace(raw_pos=-0.6, age_ms=20.0, seq=3, valid=True)
        )
        self.addCleanup(setattr, gripper, "_is_connected", False)
        return gripper

    def test_actual_unclamped_reverse_calibration_and_age(self):
        gripper = self.follower()
        reader = TaccapGripperSnapshotReader(gripper)
        with patch("lerobot.robots.bi_elite_cs66_rt.gripper_snapshots.time.monotonic", return_value=100):
            sample = reader.read()
        self.assertTrue(sample.valid)
        self.assertAlmostEqual(sample.position, 0.5)
        self.assertAlmostEqual(sample.sampled_at, 99.98)
        self.assertEqual(sample.sequence, 3)
        self.assertEqual(sample.raw_position, -0.6)
        gripper._loop.snapshot.return_value.observation.raw_pos = -1.3
        self.assertAlmostEqual(reader.read().position, 1.2)
        gripper._loop.snapshot.return_value.observation.raw_pos = 0.1
        self.assertAlmostEqual(reader.read().position, -0.2)

    def test_forward_calibration(self):
        gripper = self.follower()
        gripper._gripper.get_gripper_config.return_value.flags = 0x0001
        gripper._loop.snapshot.return_value.observation.raw_pos = 0.6
        self.assertAlmostEqual(TaccapGripperSnapshotReader(gripper).read().position, 0.5)

    def test_reading_same_sensor_sample_does_not_refresh_timestamp(self):
        gripper = self.follower()
        reader = TaccapGripperSnapshotReader(gripper)
        with patch("lerobot.robots.bi_elite_cs66_rt.gripper_snapshots.time.monotonic", side_effect=[100, 101]):
            first = reader.read()
            gripper._loop.snapshot.return_value.observation.age_ms = 1020
            second = reader.read()
        self.assertAlmostEqual(first.sampled_at, second.sampled_at)
        self.assertEqual(first.sequence, second.sequence)

    def test_reader_uses_live_sdk_cache_without_position_getter_or_device_io(self):
        gripper = self.follower()
        reader = TaccapGripperSnapshotReader(gripper)
        gripper._gripper.reset_mock()
        gripper.get_gripper_position = Mock(side_effect=AssertionError("legacy getter cache"))
        first = reader.read()
        gripper._loop.snapshot.return_value.observation.raw_pos = -0.9
        gripper._loop.snapshot.return_value.observation.seq = 4
        second = reader.read()
        self.assertNotEqual(first.position, second.position)
        self.assertEqual(second.sequence, 4)
        self.assertEqual(gripper._gripper.mock_calls, [])
        gripper.get_gripper_position.assert_not_called()

    def test_invalid_samples_and_errors_never_use_cached_opening(self):
        for attributes in ({"valid": False}, {"seq": 0}, {"age_ms": -1}, {"age_ms": math.nan}, {"raw_pos": math.inf}):
            with self.subTest(attributes=attributes):
                gripper = self.follower()
                reader = TaccapGripperSnapshotReader(gripper)
                for key, value in attributes.items():
                    setattr(gripper._loop.snapshot.return_value.observation, key, value)
                self.assertFalse(reader.read().valid)
        gripper = self.follower()
        reader = TaccapGripperSnapshotReader(gripper)
        gripper._cached_position = 0.75
        gripper._loop.snapshot.return_value.running = False
        self.assertFalse(reader.read().valid)
        gripper._loop.snapshot.side_effect = RuntimeError("read failed")
        self.assertFalse(reader.read().valid)

    def test_calibration_errors_prevent_collision_reader_startup(self):
        for cfg in (
            SimpleNamespace(flags=0, min_open_rad=0.0, max_open_rad=1.0),
            SimpleNamespace(flags=1),
            SimpleNamespace(flags=1, min_open_rad=None, max_open_rad=1.0),
            SimpleNamespace(flags=1, min_open_rad=0.0, max_open_rad=math.nan),
            SimpleNamespace(flags=1, min_open_rad=1.0, max_open_rad=1.0),
            SimpleNamespace(flags=1, min_open_rad=1.1, max_open_rad=1.0),
        ):
            with self.subTest(config=cfg):
                gripper = self.follower()
                gripper._gripper.get_gripper_config.return_value = cfg
                with self.assertRaisesRegex(RuntimeError, "calibrated TacCap opening"):
                    TaccapGripperSnapshotReader(gripper)

    def test_disconnected_or_replaced_session_cannot_reuse_calibration(self):
        for method in ("_release_after_failed_connect", "disconnect"):
            gripper = self.follower()
            reader = TaccapGripperSnapshotReader(gripper)
            getattr(gripper, method)()
            self.assertFalse(reader.read().valid)
        for handle in ("_gripper", "_loop"):
            gripper = self.follower()
            reader = TaccapGripperSnapshotReader(gripper)
            setattr(gripper, handle, Mock())
            self.assertFalse(reader.read().valid)
        gripper = self.follower()
        reader = TaccapGripperSnapshotReader(gripper)
        gripper._is_connected = False
        self.assertFalse(reader.read().valid)
        with self.assertRaisesRegex(RuntimeError, "Connect the TacCap"):
            TaccapGripperSnapshotReader(gripper)
        gripper._is_connected = True
        gripper._gripper.get_gripper_config.side_effect = RuntimeError("no config reply")
        with self.assertRaisesRegex(RuntimeError, "no config reply"):
            TaccapGripperSnapshotReader(gripper)

    def test_shared_taccap_connect_and_commands_need_no_collision_metadata(self):
        gripper = TaccapFollower(TaccapFollowerConfig(side="left", init_open=False, mcu_device="/unused"))
        hardware = Mock()
        hardware.get_gripper_config.return_value = SimpleNamespace(flags=taccap_module._CALIBRATED_FLAG)
        loop = Mock()
        loop.snapshot.return_value = SimpleNamespace(
            grasp_torque_nm=1.0, hold_torque_limit_nm=2.0, observation=SimpleNamespace(position=0.4)
        )
        with (
            patch.object(taccap_module, "taccap", SimpleNamespace(FollowerGripper=Mock(return_value=hardware))),
            patch.object(gripper, "_make_sdk_controller", return_value=loop),
        ):
            gripper.connect()
        self.addCleanup(setattr, gripper, "_is_connected", False)
        self.assertTrue(gripper.is_connected)
        hardware.get_gripper_config.assert_called_once()
        hardware.motor.enable.assert_called_once()
        self.assertEqual(gripper.get_gripper_position(), 0.4)
        gripper.set_gripper_position(0.6)
        loop.set_target.assert_called_once_with(0.6)

    def test_retired_xgripper_recipes_keep_collision_disabled(self):
        recipes = Path(__file__).resolve().parents[2] / "recipes"
        paths = sorted(recipes.glob("*/bi_elite_cs66_rt/*xgripper.yaml"))
        self.assertEqual(len(paths), 3)
        for path in paths:
            with self.subTest(recipe=path):
                config = yaml.safe_load(path.read_text())["robot"]
                self.assertEqual(config["gripper"]["type"], "serial")
                self.assertIsNone(config["native_collision_profile"])


if __name__ == "__main__":
    unittest.main()
