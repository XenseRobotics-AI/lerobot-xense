# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Provides the XenseTactileCamera class for capturing tactile data from Xense sensors.
"""

import contextlib
import time
from pathlib import Path
from threading import Event, Lock, RLock, Thread
from typing import Any

import numpy as np

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.utils.robot_utils import get_logger

from ..camera import Camera
from .configuration_xense import XenseOutputType, XenseTactileCameraConfig

logger = get_logger("XenseCam")

_CTYPES_FIND_LIBRARY_UDEV_PATCHED = False
_XENSE_SENSOR_CREATE_LOCK = RLock()
_XENSE_CONFIG_CACHE_DIR = Path("~/.xensesdk/config").expanduser()


def _patch_ctypes_find_library_for_udev() -> None:
    """
    Work around a conda-specific issue where `ctypes.util.find_library("udev")`
    can incorrectly resolve to `$CONDA_PREFIX/lib/udev/` (a directory), causing
    `pyudev` (and therefore `xensesdk`) to fail with:
        OSError: .../lib/udev: ... Is a directory

    We patch `ctypes.util.find_library` to return the real `libudev.so.1` path
    when resolving "udev".
    """

    global _CTYPES_FIND_LIBRARY_UDEV_PATCHED
    if _CTYPES_FIND_LIBRARY_UDEV_PATCHED:
        return

    import ctypes.util
    import os
    import subprocess

    orig_find_library = ctypes.util.find_library

    def _resolve_libudev_path() -> str | None:
        # Prefer ldconfig (most robust on Linux). Suppressed rather than logged:
        # the hard-coded fallbacks below are the answer on any box without it.
        with contextlib.suppress(Exception):
            out = subprocess.check_output(["/sbin/ldconfig", "-p"], text=True, stderr=subprocess.DEVNULL)
            for line in out.splitlines():
                if "libudev.so.1" in line and "=>" in line:
                    candidate = line.split("=>", 1)[1].strip()
                    if os.path.exists(candidate):
                        return candidate

        # Common fallbacks (Ubuntu/Debian)
        for candidate in (
            "/lib/x86_64-linux-gnu/libudev.so.1",
            "/usr/lib/x86_64-linux-gnu/libudev.so.1",
        ):
            if os.path.exists(candidate):
                return candidate
        return None

    def patched_find_library(name: str):  # noqa: ANN001 - match ctypes util signature
        if name != "udev":
            return orig_find_library(name)

        resolved = orig_find_library(name)
        if resolved is None or os.path.isdir(resolved):
            fixed = _resolve_libudev_path()
            if fixed is not None:
                return fixed
        return resolved

    ctypes.util.find_library = patched_find_library
    _CTYPES_FIND_LIBRARY_UDEV_PATCHED = True


def to_landscape(image: np.ndarray) -> np.ndarray:
    """Turn the SDK's portrait image (700, 400[, C]) into the stored landscape (400, 700[, C]).

    Rotate 90° clockwise — the same as the backpack collector (taccap-core
    ``RECTIFY_ROTATE_DEG = 270``), so both capture stacks store the same picture.

    This used to be ``np.transpose(image, (1, 0, ...))``. A transpose is not a
    rotation: it equals a 90° clockwise rotation followed by a left-right flip,
    so every tactile image recorded through this driver was mirrored — the
    contact tip that points right in the collector pointed left here, and shear
    to the left read as shear to the right.
    """
    return np.ascontiguousarray(np.rot90(image, k=-1, axes=(0, 1)))


class XenseTactileCamera(Camera):
    """
    Manages tactile sensor interactions using Xense SDK for efficient data recording.

    This class provides a high-level interface to connect to, configure, and read
    tactile data from Xense sensors. It supports both synchronous and asynchronous
    data reading with a background thread.

    A XenseTactileCamera instance requires a sensor serial number (e.g., "OG000344").
    The sensor provides various output types including force distribution, force
    resultant, depth maps, and 2D marker tracking.

    Example:
        ```python
        from lerobot.cameras.xense import XenseTactileCamera, XenseTactileCameraConfig, XenseOutputType

        # Basic usage with force sensing
        config = XenseTactileCameraConfig(
            serial_number="OG000344", fps=60, output_types=[XenseOutputType.FORCE, XenseOutputType.FORCE_RESULTANT]
        )
        sensor = XenseTactileCamera(config)
        sensor.connect()

        # Read data synchronously
        data = sensor.read()
        print(f"Force shape: {data['force'].shape}")
        print(f"Force resultant: {data['force_resultant']}")

        # Read data asynchronously
        async_data = sensor.async_read()

        # When done, properly disconnect
        sensor.disconnect()
        ```
    """

    def __init__(self, config: XenseTactileCameraConfig):
        """
        Initializes the XenseTactileCamera instance.

        Args:
            config: The configuration settings for the Xense sensor.
        """
        super().__init__(config)

        self.config = config
        self.serial_number = config.serial_number
        # The config declares `| list[str] | None` because YAML and the CLI hand it
        # strings; __post_init__ resolves those to enum members and fills the default.
        # Narrow once here rather than at each of the dozen uses below — and check it,
        # so a config that skipped __post_init__ fails at construction instead of
        # somewhere inside a read().
        if not config.output_types or not all(isinstance(ot, XenseOutputType) for ot in config.output_types):
            raise TypeError(
                f"{type(self).__name__} needs output_types normalized to XenseOutputType; "
                f"got {config.output_types!r}. XenseTactileCameraConfig.__post_init__ does this."
            )
        self.output_types: list[XenseOutputType] = list(config.output_types)  # type: ignore[arg-type]
        self.warmup_s = config.warmup_s
        self.rectify_size = config.rectify_size
        self.raw_size = config.raw_size
        self.disable_infer = config.disable_infer
        self.infer_mode = config.infer_mode
        self.camera_properties = config.camera_properties
        self.sensor: Any = None

        # Threading for async read
        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_data: dict[str, np.ndarray] | None = None
        self.new_frame_event: Event = Event()

        # Import xensesdk here to avoid import errors if not installed
        try:
            from xensesdk import Sensor

            self._Sensor = Sensor
        except ImportError as e:
            raise ImportError(
                "xensesdk is required for XenseTactileCamera. "
                "Please install it according to the manufacturer's instructions."
            ) from e

        # Pre-build sensor output types list for better performance
        # This avoids reconstructing the mapping on every read() call
        self._sensor_output_types_cache: list[Any] | None = None
        # Keep a key so changing output_types at runtime can invalidate cache
        self._sensor_output_types_cache_key: tuple[XenseOutputType, ...] | None = None

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.serial_number})"

    @property
    def is_connected(self) -> bool:
        """Checks if the sensor is currently connected."""
        return self.sensor is not None

    def connect(self, warmup: bool = True):
        """
        Connects to the Xense sensor specified in the configuration.

        Initializes the Xense Sensor object and performs initial checks.

        Raises:
            DeviceAlreadyConnectedError: If the sensor is already connected.
            ConnectionError: If the sensor fails to connect.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        try:
            _patch_ctypes_find_library_for_udev()
            # xensesdk>=2.0: `api`/`use_gpu` were removed; the SDK auto-selects the
            # camera backend and GPU. `disable_infer` skips the inference-engine load
            # for image-only outputs; `infer_mode` (None=SDK default) picks the variant.
            create_kwargs: dict[str, Any] = {
                "rectify_size": self.rectify_size,
                "raw_size": self.raw_size,
                "disable_infer": self.disable_infer,
            }
            cached_config_path = None
            if _XENSE_CONFIG_CACHE_DIR.is_dir():
                cached_config_path = _XENSE_CONFIG_CACHE_DIR
                create_kwargs["config_path"] = _XENSE_CONFIG_CACHE_DIR
            if self.infer_mode is not None:
                create_kwargs["infer_mode"] = self.infer_mode
            # Override the per-OS camera property template (exposure/brightness/
            # white-balance...) via Sensor.create(overrides=...). The override key
            # must be 2-level dotted ("camera.linux"/"camera.win"); the value is
            # the full property dict (it replaces the template for that OS).
            if self.camera_properties:
                import platform

                os_key = "win" if platform.system().lower().startswith("win") else "linux"
                create_kwargs["overrides"] = {f"camera.{os_key}": dict(self.camera_properties)}
            with _XENSE_SENSOR_CREATE_LOCK:
                self.sensor = self._Sensor.create(self.serial_number, **create_kwargs)
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to {self}. Error: {e}. "
                "Make sure the sensor is plugged in and the serial number is correct."
            ) from e

        if warmup:
            # time.sleep(2)
            # Start background thread first for async_read
            # Do warmup reads to stabilize the sensor
            start_time = time.time()
            while time.time() - start_time < self.warmup_s:
                # Failures here are the point of warming up: the sensor is not
                # streaming yet, and read() is what makes it start.
                with contextlib.suppress(Exception):
                    self.read()
                time.sleep(0.1)
            self._start_read_thread()

        logger.info(
            f"{self} connected (disable_infer={self.disable_infer}, infer_mode={self.infer_mode}, config_cache={cached_config_path})"
        )

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """
        Detects available Xense sensors connected to the system.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries,
            where each dictionary contains 'type', 'serial_number', and other info.
        """
        try:
            _patch_ctypes_find_library_for_udev()
            from xensesdk import Sensor

            # Get available devices - returns dict like {'OG000352': 10, 'OG000344': 8}
            # Key is serial number, value is cam_id (video device index)
            devices = Sensor.scanSerialNumber()

            found_sensors = []
            for serial_number, cam_id in devices.items():
                sensor_info = {
                    "name": f"Xense Tactile Sensor {serial_number}",
                    "type": "Xense",
                    "serial_number": serial_number,
                    "cam_id": cam_id,
                }
                found_sensors.append(sensor_info)

            return found_sensors

        except ImportError:
            logger.warn(
                "xensesdk not installed. Cannot detect Xense sensors. "
                "Please install xensesdk to use Xense tactile sensors."
            )
            return []
        except Exception as e:
            logger.error(f"Error detecting Xense sensors: {e}")
            return []

    def _read_sensor_data(self) -> np.ndarray | tuple[np.ndarray, ...]:
        """
        Internal method to read data from the sensor based on configured output types.

        Returns:
            np.ndarray if single output type, or tuple of np.ndarray if multiple output types.

        Raises:
            DeviceNotConnectedError: If the sensor is not connected.
            RuntimeError: If reading from the sensor fails.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        try:
            # Build sensor output types list (cached for performance)
            cache_key = tuple(self.output_types)
            if self._sensor_output_types_cache is None or self._sensor_output_types_cache_key != cache_key:
                # Map XenseOutputType to Sensor.OutputType
                # Note: SDK uses CamelCase for OutputType attributes (e.g., Force, ForceResultant)
                output_type_mapping = {
                    XenseOutputType.RECTIFY: self._Sensor.OutputType.Rectify,
                    XenseOutputType.DIFFERENCE: self._Sensor.OutputType.Difference,
                    XenseOutputType.DEPTH: self._Sensor.OutputType.Depth,
                    XenseOutputType.MARKER_2D: self._Sensor.OutputType.Marker2D,
                    XenseOutputType.FORCE: self._Sensor.OutputType.Force,
                    XenseOutputType.FORCE_NORM: self._Sensor.OutputType.ForceNorm,
                    XenseOutputType.FORCE_RESULTANT: self._Sensor.OutputType.ForceResultant,
                    XenseOutputType.MESH_3D: self._Sensor.OutputType.Mesh3D,
                    XenseOutputType.MESH_3D_INIT: self._Sensor.OutputType.Mesh3DInit,
                    XenseOutputType.MESH_3D_FLOW: self._Sensor.OutputType.Mesh3DFlow,
                }

                # Build list of sensor output types to request (cache it)
                self._sensor_output_types_cache = [
                    output_type_mapping[output_type] for output_type in self.output_types
                ]
                self._sensor_output_types_cache_key = cache_key

            # Call selectSensorInfo with cached sensor output types
            # Returns: single np.ndarray if one arg, or tuple of np.ndarray if multiple args
            results = self.sensor.selectSensorInfo(*self._sensor_output_types_cache)
            # for names, result in zip(self._sensor_output_types_cache, results):

            # image_outputs definition
            image_outputs = {
                XenseOutputType.RECTIFY,
                XenseOutputType.DIFFERENCE,
                XenseOutputType.DEPTH,
            }

            # DEBUG: Check if output type is in image_outputs

            if isinstance(results, tuple):
                # IMPORTANT: keep 1:1 correspondence with self.output_types
                # (previously, non-image outputs were accidentally dropped)
                processed_results: list[np.ndarray] = []
                for i, output_type in enumerate(self.output_types):
                    data = results[i]
                    if output_type in image_outputs and getattr(data, "ndim", 0) >= 2:
                        # portrait (700, 400) -> landscape (400, 700); see to_landscape
                        data = to_landscape(data)
                    if output_type == XenseOutputType.DEPTH:
                        data = np.asarray(data) + np.float32(0.01)  # add 10mm to avoid log zero
                    processed_results.append(data)
                return tuple(processed_results)
            else:
                # single output type
                if self.output_types[0] in image_outputs and getattr(results, "ndim", 0) >= 2:
                    results = to_landscape(results)
                if self.output_types[0] == XenseOutputType.DEPTH and results is not None:
                    results = np.asarray(results) + np.float32(0.01)  # add 10mm to avoid log zero
                return results

        except Exception as e:
            raise RuntimeError(f"{self} failed to read sensor data: {e}") from e

    def _format_read_result(self, data: np.ndarray | tuple[np.ndarray, ...]) -> np.ndarray | dict[str, np.ndarray]:
        """
        Convert SDK output into a stable, developer-friendly structure.

        - If exactly 1 output type is configured: return the single np.ndarray (backward-compatible).
        - If multiple output types are configured: return a dict keyed by XenseOutputType.value.
        """
        # NOTE: Xense SDK (via OpenCV backend) returns image-like outputs as BGR.
        # We convert to RGB here so downstream code can treat rectify/difference as RGB consistently.
        # This is applied for both sync read() and async_read().
        image_bgr_outputs = {
            XenseOutputType.RECTIFY,
            XenseOutputType.DIFFERENCE,
        }

        def _bgr_to_rgb_if_needed(output_type: XenseOutputType, arr: np.ndarray) -> np.ndarray:
            if output_type not in image_bgr_outputs:
                return arr
            if not isinstance(arr, np.ndarray):
                return arr
            # Expect HWC/WHC with 3 channels.
            if arr.ndim < 3 or arr.shape[-1] != 3:
                return arr
            # Swap last axis (BGR -> RGB). Make contiguous to avoid negative strides downstream.
            return np.ascontiguousarray(arr[..., ::-1])

        if len(self.output_types) == 1:
            if isinstance(data, tuple):
                # Defensive: SDK should not return tuple for a single requested output
                data0 = data[0]
                return _bgr_to_rgb_if_needed(self.output_types[0], data0)
            return _bgr_to_rgb_if_needed(self.output_types[0], data)

        # Multiple outputs -> dict in the same order as requested
        if not isinstance(data, tuple):
            # Defensive: SDK should return tuple when multiple outputs requested
            return {self.output_types[0].value: _bgr_to_rgb_if_needed(self.output_types[0], data)}

        if len(data) != len(self.output_types):
            raise RuntimeError(
                f"{self}: Internal error: SDK returned {len(data)} outputs but "
                f"{len(self.output_types)} were requested ({self.output_types})."
            )

        return {ot.value: _bgr_to_rgb_if_needed(ot, arr) for ot, arr in zip(self.output_types, data, strict=True)}

    def read(self, color_mode=None) -> np.ndarray | dict[str, np.ndarray]:
        """
        Reads tactile data synchronously from the sensor.

        This is a blocking call. It waits for the next available data from the sensor.

        Args:
            color_mode: Not used for Xense sensors, kept for API compatibility.

        Returns:
            np.ndarray if single output type configured, or tuple of np.ndarray if multiple.
            For example:
            - Single: array(35,20,3) for FORCE only
            - Multiple: (array(35,20,3), array(6,)) for FORCE and FORCE_RESULTANT

        Raises:
            DeviceNotConnectedError: If the sensor is not connected.
            RuntimeError: If reading from the sensor fails.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start_time = time.perf_counter()

        data = self._read_sensor_data()
        formatted = self._format_read_result(data)

        read_duration_ms = (time.perf_counter() - start_time) * 1e3
        logger.debug(f"{self} read took: {read_duration_ms:.1f}ms")

        return formatted

    def _read_loop(self):
        """
        Internal loop run by the background thread for asynchronous reading.

        On each iteration:
        1. Reads sensor data
        2. Stores result in latest_data (thread-safe)
        3. Sets new_frame_event to notify listeners
        4. Sleeps to maintain target FPS

        Stops on DeviceNotConnectedError, logs other errors and continues.
        """
        target_loop_time = 1.0 / self.fps if self.fps else 0

        while not self.stop_event.is_set():
            loop_start = time.perf_counter()

            try:
                data = self.read()

                with self.frame_lock:
                    self.latest_data = data
                self.new_frame_event.set()

            except DeviceNotConnectedError:
                break
            except Exception as e:
                logger.warn(f"Error reading data in background thread for {self}: {e}")

            # Sleep to maintain target FPS
            if target_loop_time > 0:
                elapsed = time.perf_counter() - loop_start
                sleep_time = target_loop_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def _start_read_thread(self) -> None:
        """Starts or restarts the background read thread if it's not running."""
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=0.1)
        if self.stop_event is not None:
            self.stop_event.set()

        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, args=(), name=f"{self}_read_loop")
        self.thread.daemon = True
        self.thread.start()

    def _stop_read_thread(self) -> None:
        """Signals the background read thread to stop and waits for it to join."""
        if self.stop_event is not None:
            self.stop_event.set()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        self.thread = None
        self.stop_event = None

    def async_read(self, timeout_ms: float = 200) -> np.ndarray | dict[str, np.ndarray]:
        """
        Reads the latest available data asynchronously.

        This method retrieves the most recent data captured by the background
        read thread. It returns immediately with the latest cached data, or waits
        up to timeout_ms if no data is available yet (e.g., first call after connect).

        Args:
            timeout_ms (float): Maximum time in milliseconds to wait for data
                to become available on first call. Defaults to 200ms (0.2 seconds).

        Returns:
            np.ndarray if single output type configured, or tuple of np.ndarray if multiple.
            For example:
            - Single: array(35,20,3) for FORCE only
            - Multiple: (array(35,20,3), array(6,)) for FORCE and FORCE_RESULTANT

        Raises:
            DeviceNotConnectedError: If the sensor is not connected.
            TimeoutError: If no data becomes available within the specified timeout.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.thread is None or not self.thread.is_alive():
            self._start_read_thread()

        # Try to get data immediately (non-blocking)
        with self.frame_lock:
            data = self.latest_data

        # If no data yet (first call), wait for it
        if data is None:
            if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
                thread_alive = self.thread is not None and self.thread.is_alive()
                raise TimeoutError(
                    f"Timed out waiting for data from sensor {self} after {timeout_ms} ms. "
                    f"Read thread alive: {thread_alive}."
                )

            with self.frame_lock:
                data = self.latest_data

            if data is None:
                raise RuntimeError(f"Internal error: Event set but no data available for {self}.")

        if data is None:
            raise RuntimeError(f"Internal error: Event set but no data available for {self}.")

        # NOTE: data is already formatted by read() in _read_loop, so we return it directly.
        # Do NOT call _format_read_result() again, as it would double-convert BGR<->RGB.
        return data

    def disconnect(self):
        """
        Disconnects from the sensor and cleans up resources.

        Stops the background read thread (if running) and releases the sensor.

        Raises:
            DeviceNotConnectedError: If the sensor is already disconnected.
        """
        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(f"{self} not connected.")

        if self.thread is not None:
            self._stop_read_thread()

        if self.sensor is not None:
            try:
                self.sensor.release()
            except Exception as e:
                logger.warn(f"Error releasing {self}: {e}")
            self.sensor = None

        logger.info(f"{self} disconnected.")
