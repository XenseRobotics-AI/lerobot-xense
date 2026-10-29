"""B601 camera extensions must not change the shared Xense camera driver."""

import itertools
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from lerobot.cameras.xense import XenseOutputType, XenseTactileCamera, XenseTactileCameraConfig, camera_xense as driver
from lerobot.robots.rebot_b601_rt.tactile_camera import (
    RebotTactileCamera,
    make_rebot_cameras,
    wait_for_fresh_camera_frames,
)
from lerobot.utils.errors import DeviceNotConnectedError

SERIAL = "GSPS01A34Z0021"


class TestXenseCameraConnect(unittest.TestCase):
    def setUp(self):
        root = Path(self.enterContext(TemporaryDirectory()))
        self.cache = root / "config"
        self.enterContext(patch.object(driver, "_XENSE_CONFIG_CACHE_DIR", self.cache))
        self.enterContext(patch.object(driver, "_patch_ctypes_find_library_for_udev"))
        self.sleep = self.enterContext(patch.object(driver.time, "sleep"))
        clock = itertools.count(step=0.2)
        self.enterContext(patch.object(driver.time, "monotonic", side_effect=lambda: next(clock)))
        self.sensor = Mock()
        self.sdk = SimpleNamespace(
            create=Mock(return_value=self.sensor),
            scanSerialNumber=Mock(return_value={SERIAL: 8}),
        )
        self.enterContext(patch.dict(sys.modules, xensesdk=SimpleNamespace(Sensor=self.sdk)))
        self.camera = RebotTactileCamera(
            XenseTactileCameraConfig(
                serial_number=SERIAL,
                output_types=[XenseOutputType.RECTIFY],
            )
        )
        self.addCleanup(lambda: self.camera.disconnect() if self.camera.is_connected else None)

    def save_cache(self):
        self.cache.mkdir(exist_ok=True)
        (self.cache / SERIAL).write_bytes(b"fake calibration")

    def test_normal_connect_does_not_scan_or_wait(self):
        self.camera.connect(warmup=False)
        self.assertIs(self.camera.sensor, self.sensor)
        self.sdk.create.assert_called_once()
        self.sdk.scanSerialNumber.assert_not_called()
        self.sleep.assert_not_called()

    def test_first_flash_read_reopens_from_new_cache_after_serial_reappears(self):
        def create(serial, **kwargs):
            if "config_path" not in kwargs:
                self.save_cache()
                raise RuntimeError("Cannot open camera 8")
            return self.sensor

        self.sdk.create.side_effect = create
        self.sdk.scanSerialNumber.side_effect = [{}, {SERIAL: 12}]
        self.camera.connect(warmup=False)
        first, retry = self.sdk.create.call_args_list
        self.assertNotIn("config_path", first.kwargs)
        self.assertEqual(retry.args, (SERIAL,))
        self.assertEqual(retry.kwargs["config_path"], self.cache)
        self.assertTrue(retry.kwargs["disable_infer"])
        self.assertIs(self.camera.sensor, self.sensor)
        self.assertEqual(self.sdk.scanSerialNumber.call_count, 2)

    def test_unrelated_sdk_error_is_not_retried(self):
        self.save_cache()
        error = RuntimeError("Invalid calibration")
        self.sdk.create.side_effect = error
        with self.assertRaises(ConnectionError) as caught:
            self.camera.connect(warmup=False)
        self.assertIs(caught.exception.__cause__, error)
        self.sdk.create.assert_called_once()
        self.sdk.scanSerialNumber.assert_not_called()

    def test_missing_cache_does_not_repeat_flash_reads(self):
        self.sdk.create.side_effect = RuntimeError("Cannot open camera 8")
        with self.assertRaisesRegex(ConnectionError, "Cannot open camera 8"):
            self.camera.connect(warmup=False)
        self.sdk.create.assert_called_once()
        self.sdk.scanSerialNumber.assert_not_called()

    def test_missing_device_times_out_without_trying_another_camera_index(self):
        self.save_cache()
        self.sdk.create.side_effect = RuntimeError("Cannot open camera 8")
        self.sdk.scanSerialNumber.return_value = {}
        with self.assertRaisesRegex(ConnectionError, "Cannot open camera 8"):
            self.camera.connect(warmup=False)
        self.sdk.create.assert_called_once()
        self.assertGreater(self.sdk.scanSerialNumber.call_count, 1)
        self.assertLess(self.sdk.scanSerialNumber.call_count, 20)
        self.assertFalse(self.camera.is_connected)

    def test_recovery_does_not_hide_a_new_sdk_error(self):
        self.save_cache()
        error = RuntimeError("Invalid calibration")
        self.sdk.create.side_effect = [RuntimeError("Cannot open camera 8"), error]
        with self.assertRaises(ConnectionError) as caught:
            self.camera.connect(warmup=False)
        self.assertIs(caught.exception.__cause__, error)
        self.assertEqual(self.sdk.create.call_count, 2)

    def test_read_latest_returns_the_formatted_sample_without_another_read(self):
        self.camera.connect(warmup=False)
        frame = np.zeros((4, 7, 3), dtype=np.uint8)
        self.camera._latest_sample = (frame, 9.9)
        with patch.object(driver.time, "monotonic", return_value=10.0):
            self.assertIs(self.camera.read_latest(), frame)
        self.sensor.selectSensorInfo.assert_not_called()

    def test_read_latest_waits_for_the_first_sample(self):
        self.camera.connect(warmup=False)
        frame = np.zeros((4, 7, 3), dtype=np.uint8)

        def first_sample(timeout_ms):
            self.camera._latest_sample = (frame, 10.0)
            return frame

        with (
            patch.object(self.camera, "async_read", side_effect=first_sample) as read,
            patch.object(driver.time, "monotonic", return_value=10.0),
        ):
            self.assertIs(self.camera.read_latest(), frame)
        read.assert_called_once_with(timeout_ms=500)

    def test_read_latest_rejects_stale_samples(self):
        self.camera.connect(warmup=False)
        self.camera._latest_sample = (np.zeros((4, 7, 3), dtype=np.uint8), 9.0)
        with (
            patch.object(driver.time, "monotonic", return_value=10.0),
            self.assertRaisesRegex(TimeoutError, "too old"),
        ):
            self.camera.read_latest(max_age_ms=500)

    def test_recording_start_waits_for_new_frames_after_encoder_warmup(self):
        self.camera.connect(warmup=False)
        old_frame = np.zeros((4, 7, 3), dtype=np.uint8)
        fresh_frame = np.ones_like(old_frame)
        self.camera._latest_sample = (old_frame, 9.0)

        def background_reader(_seconds):
            self.camera._latest_sample = (fresh_frame, 10.0)

        with (
            patch.object(driver.time, "monotonic", return_value=10.0),
            patch.object(driver.time, "sleep", side_effect=background_reader) as sleep,
        ):
            wait_for_fresh_camera_frames({"tactile": self.camera})
            self.assertIs(self.camera.read_latest(), fresh_frame)
        sleep.assert_called_once_with(0.01)
        self.sensor.selectSensorInfo.assert_not_called()

    def test_recording_start_rechecks_all_cameras_and_never_changes_age_limit(self):
        left, right = Mock(), Mock()
        left.read_latest.side_effect = [TimeoutError("left stale"), object(), object()]
        right.read_latest.side_effect = [TimeoutError("right stale"), object()]
        wait_for_fresh_camera_frames({"left": left, "right": right})
        self.assertEqual(left.read_latest.call_count, 3)
        self.assertEqual(right.read_latest.call_count, 2)
        for camera in (left, right):
            self.assertTrue(all(not call.args and not call.kwargs for call in camera.read_latest.call_args_list))
        self.assertEqual(self.sleep.call_count, 2)

    def test_recording_start_times_out_when_camera_does_not_recover(self):
        self.camera.connect(warmup=False)
        self.camera._latest_sample = (np.zeros((4, 7, 3), dtype=np.uint8), -10.0)
        with self.assertRaisesRegex(TimeoutError, "before recording") as caught:
            wait_for_fresh_camera_frames({"tactile": self.camera}, timeout_s=0.5)
        self.assertIn("too old", str(caught.exception.__cause__))
        self.assertLessEqual(self.sleep.call_count, 3)
        self.sensor.selectSensorInfo.assert_not_called()

    def test_recording_start_does_not_wait_for_fresh_caches_or_hide_device_errors(self):
        camera = Mock()
        wait_for_fresh_camera_frames({})
        wait_for_fresh_camera_frames({"camera": camera})
        camera.read_latest.side_effect = DeviceNotConnectedError("disconnected")
        with self.assertRaises(DeviceNotConnectedError):
            wait_for_fresh_camera_frames({"camera": camera})
        self.sleep.assert_not_called()

    def test_disconnect_clears_the_previous_stream_sample(self):
        self.camera.connect(warmup=False)
        self.camera._latest_sample = (np.zeros((4, 7, 3), dtype=np.uint8), 10.0)
        self.camera.new_frame_event.set()
        self.camera.disconnect()
        self.assertIsNone(self.camera.latest_data)
        self.assertIsNone(self.camera._latest_sample)
        self.assertFalse(self.camera.new_frame_event.is_set())
        with self.assertRaises(DeviceNotConnectedError):
            self.camera.read_latest()

    def test_background_and_warmup_reads_cache_formatted_frames(self):
        self.camera.connect(warmup=False)
        frame = np.zeros((4, 7, 3), dtype=np.uint8)
        with patch.object(XenseTactileCamera, "read", return_value=frame) as read:
            self.assertIs(self.camera.read(), frame)
            self.assertIs(self.camera.read_latest(), frame)
        read.assert_called_once_with(color_mode=None)

    def test_shared_camera_does_not_retry_or_use_b601_cached_reads(self):
        self.save_cache()
        self.sdk.create.side_effect = RuntimeError("Cannot open camera 8")
        shared = XenseTactileCamera(self.camera.config)
        with self.assertRaises(ConnectionError):
            shared.connect(warmup=False)
        self.sdk.create.assert_called_once()
        self.sdk.scanSerialNumber.assert_not_called()
        frame = np.zeros((4, 7, 3), dtype=np.uint8)
        with patch.object(shared, "async_read", return_value=frame) as read, self.assertWarns(FutureWarning):
            self.assertIs(shared.read_latest(), frame)
        read.assert_called_once_with()

    def test_only_b601_factory_uses_specialized_tactile_camera(self):
        from lerobot.cameras import make_cameras_from_configs

        config = {"pad": self.camera.config}
        self.assertIs(type(make_cameras_from_configs(config)["pad"]), XenseTactileCamera)
        self.assertIs(type(make_rebot_cameras(config)["pad"]), RebotTactileCamera)

    def test_b601_factory_preserves_camera_order_and_other_camera_types(self):
        from lerobot.robots.rebot_b601_rt import tactile_camera

        scene_config = SimpleNamespace(type="opencv")
        scene = object()
        config = {"pad": self.camera.config, "scene": scene_config}
        with patch.object(tactile_camera, "make_cameras_from_configs", return_value={"scene": scene}) as factory:
            cameras = make_rebot_cameras(config)
        self.assertEqual(list(cameras), ["pad", "scene"])
        self.assertIs(cameras["scene"], scene)
        factory.assert_called_once_with({"scene": scene_config})


if __name__ == "__main__":
    unittest.main()
