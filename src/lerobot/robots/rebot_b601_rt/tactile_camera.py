"""B601-only camera recovery and cached reads; shared camera behavior is unchanged."""

import time

from lerobot.cameras import make_cameras_from_configs
from lerobot.cameras.xense import XenseTactileCamera, camera_xense
from lerobot.utils.errors import DeviceNotConnectedError

_REOPEN_TIMEOUT_S = 3.0
_REOPEN_POLL_S = 0.2


class RebotTactileCamera(XenseTactileCamera):
    """Keep B601 observation reads independent of the tactile camera frame rate."""

    def __init__(self, config):
        super().__init__(config)
        self._latest_sample = None

    def connect(self, warmup: bool = True):
        try:
            super().connect(warmup=warmup)
            return
        except ConnectionError as error:
            self._check_retryable(error)
            last_error = error

        camera_xense.logger.warn(f"{self}: camera open failed; waiting for USB enumeration before reopening from cache")
        deadline = time.monotonic() + _REOPEN_TIMEOUT_S
        while time.monotonic() < deadline:
            time.sleep(_REOPEN_POLL_S)
            if self.serial_number not in self._Sensor.scanSerialNumber():
                continue
            try:
                # The shared driver now finds the saved cache. It still owns
                # Sensor.create's lock, camera settings and stream warmup.
                super().connect(warmup=warmup)
                return
            except ConnectionError as error:
                self._check_retryable(error)
                last_error = error
        raise last_error

    def _check_retryable(self, error: ConnectionError) -> None:
        cause = error.__cause__
        cache = camera_xense._XENSE_CONFIG_CACHE_DIR / self.serial_number
        if (
            not isinstance(cause, RuntimeError)
            or not str(cause).startswith("Cannot open camera")
            or not cache.is_file()
        ):
            raise error

    def read(self, color_mode=None):
        data = super().read(color_mode=color_mode)
        # Both warmup and the inherited background loop call this method.
        # Store data and timestamp together, after the existing RGB conversion.
        with self.frame_lock:
            self._latest_sample = (data, time.monotonic())
        return data

    def read_latest(self, max_age_ms: int = 500):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        with self.frame_lock:
            sample = self._latest_sample
        if sample is None:
            self.async_read(timeout_ms=max_age_ms)
            with self.frame_lock:
                sample = self._latest_sample
        if sample is None:
            raise RuntimeError(f"{self} has not captured any frames yet.")
        data, timestamp = sample
        age_ms = (time.monotonic() - timestamp) * 1000
        if age_ms > max_age_ms:
            raise TimeoutError(f"{self} latest frame is too old: {age_ms:.1f} ms (max allowed: {max_age_ms} ms).")
        return data

    def disconnect(self):
        super().disconnect()
        with self.frame_lock:
            self._latest_sample = None
            self.latest_data = None
        self.new_frame_event.clear()


def wait_for_fresh_camera_frames(cameras, timeout_s: float = 2.0) -> None:
    """Let camera caches recover after encoder warmup, before recording starts."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            for camera in cameras.values():
                camera.read_latest()
            return
        except TimeoutError as error:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"B601 cameras did not provide fresh frames within {timeout_s:.1f} s before recording: {error}"
                ) from error
            # Yield to the existing camera readers. Never accept stale images
            # or introduce this wait into the per-frame observation path.
            time.sleep(min(0.01, remaining))


def make_rebot_cameras(camera_configs):
    """Select the local tactile adapter only for cameras owned by a B601 arm."""
    cameras = make_cameras_from_configs({key: cfg for key, cfg in camera_configs.items() if cfg.type != "xense"})
    cameras.update({key: RebotTactileCamera(cfg) for key, cfg in camera_configs.items() if cfg.type == "xense"})
    return {key: cameras[key] for key in camera_configs}
