#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
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

"""TacCap follower (actuated) gripper driver — arm-agnostic, shared across robots.

Wraps ``xense.taccap.FollowerGripper``. The recipe selects either the SDK's
position-impedance ``ControlLoop`` or contact-aware
``ForcePositionController``. Both own their background motor-status/control
loop and keep a thread-safe observation fresh, so ``get_gripper_position`` /
``set_gripper_position`` remain non-blocking.

Left/right units are told apart by the firmware-burned serial number via the
SDK's side-aware discovery (``find_left`` / ``find_right``). When the arm has
already run ``discover_taccap_sides()`` — it does, to wire the wrist + tactile
cameras — it passes the resolved device path through ``config.mcu_device`` so
this driver does not re-enumerate the bus a second time per side.

The ``xense.taccap`` SDK is imported lazily (guarded) so importing the grippers
package does not hard-fail on hosts where the native extension is not built —
the error is raised with rebuild guidance at ``connect()`` time instead.
"""

import threading
import time

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.utils.robot_utils import get_logger

from ..gripper import Gripper
from .configuration_taccap import TaccapFollowerConfig

try:
    import xense.taccap as taccap  # type: ignore

    _TACCAP_IMPORT_ERROR: Exception | None = None
except Exception as e:  # pragma: no cover - depends on native build being present
    taccap = None  # type: ignore
    _TACCAP_IMPORT_ERROR = e

# GripperConfig.flags bit 0 = calibrated (see SDK gripper_control_test.py).
_CALIBRATED_FLAG = 0x0001

# ForcePositionController::set_target resets its contact guard and contact
# counter. Teleop repeats the latest action every frame, so identical targets
# must be coalesced or the controller can never latch contact.
_FORCE_POSITION_TARGET_EPS = 1e-4

# The teleop UI owns terminal cursor movement. Gripper reads only publish their
# latest line here; printing directly from each side would scroll two new lines
# through the TCP panel on every status update.
_STATUS_LOCK = threading.Lock()
_STATUS_LINES: dict[str, str] = {}


def get_taccap_status_lines() -> tuple[str, ...]:
    """Return the latest enabled TacCap status lines in stable L/R order."""

    with _STATUS_LOCK:
        return tuple(_STATUS_LINES[side] for side in ("left", "right") if side in _STATUS_LINES)


def _set_taccap_status_line(side: str, line: str) -> None:
    with _STATUS_LOCK:
        _STATUS_LINES[side] = line


def _clear_taccap_status_line(side: str) -> None:
    with _STATUS_LOCK:
        _STATUS_LINES.pop(side, None)


class TaccapFollower(Gripper):
    """Wrapper around ``xense.taccap.FollowerGripper`` for arm robots.

    Normalized position convention (matches ``SerialGripper``):
        0.0  →  fully closed
        1.0  →  fully open

    Example::

        cfg = TaccapFollowerConfig(side="left")
        g = TaccapFollower(cfg)
        g.connect()
        g.set_gripper_position(0.5)
        print(g.get_gripper_position())
        g.disconnect()
    """

    config_class = TaccapFollowerConfig

    def __init__(self, config: TaccapFollowerConfig):
        super().__init__(config)
        self._config = config
        self._side = config.side
        self._mcu_device = config.mcu_device
        self._controller_name = config.controller
        self._init_open = config.init_open
        self._require_calibrated = config.require_calibrated
        self._print_status = config.print_status
        self._status_print_period_s = 1.0 / config.status_print_hz
        self._last_status_print_time = float("-inf")
        self._last_target_position: float | None = None

        self.logger = get_logger(f"TaccapFollower-{config.side}")
        self._is_connected: bool = False
        self._gripper = None  # xense.taccap.FollowerGripper
        self._loop = None  # active ControlLoop or ForcePositionController

        # Seed so get_gripper_position() returns something sane before the loop
        # produces its first observation.
        self._cached_position: float = 1.0 if config.init_open else 0.0

    def __str__(self) -> str:
        return f"TaccapFollower({self._side})"

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    # ── Connection lifecycle ───────────────────────────────────────────────────

    def _resolve_device(self) -> str:
        """Return the MCU device path for this side.

        Prefers ``config.mcu_device`` — the arm fills it in from the
        ``discover_taccap_sides()`` sweep it already ran for the cameras, so the
        common bimanual path enumerates the bus once instead of once per side
        again here. Falls back to a side-aware scan when nothing was injected.
        """
        if self._mcu_device:
            return self._mcu_device
        finder = taccap.find_left if self._side == "left" else taccap.find_right
        eps = finder()  # throws if the requested side isn't visible on the bus
        return eps.mcu_device

    def _make_sdk_controller(self):
        """Build the SDK controller selected by the recipe."""

        cfg = self._config
        if self._controller_name == "control_loop":
            # Recipes express feedforward in normalized gripper coordinates:
            # negative closes and positive opens. The mirrored right gripper
            # has a reversed raw-radian map, so flip its motor torque sign.
            direction_open = -1.0 if self._gripper.position_map().reverse else 1.0
            raw_feedforward_torque = cfg.feedforward_torque * direction_open
            phase = {
                "stream_locked": taccap.SubmitPhase.STREAM_LOCKED,
                "free_running": taccap.SubmitPhase.FREE_RUNNING,
            }[cfg.submit_phase]
            stall_action = {
                "hold_position": taccap.StallAction.HOLD_POSITION,
                "none": taccap.StallAction.NONE,
            }[cfg.stall_action]
            return taccap.ControlLoop(
                self._gripper,
                hz=cfg.control_hz,
                kp=cfg.kp,
                kd=cfg.kd,
                feedforward_torque=raw_feedforward_torque,
                motor_stream_hz=cfg.motor_stream_hz,
                phase=phase,
                max_position_torque_nm=cfg.max_position_torque_nm,
                rated_torque_nm=cfg.rated_torque_nm,
                rated_hold_ms=cfg.rated_hold_ms,
                rated_release_rad=cfg.rated_release_rad,
                stall_torque_nm=cfg.stall_torque_nm,
                stall_vel_radps=cfg.stall_vel_radps,
                stall_hold_ms=cfg.stall_hold_ms,
                stall_action=stall_action,
            )

        if not hasattr(taccap, "ForcePositionController") or not hasattr(taccap, "ForcePositionConfig"):
            raise RuntimeError(
                "The installed xense.taccap SDK does not expose ForcePositionController. "
                "Reinstall third_party/taccap-gripper in the active LeRobot environment."
            )
        force_cfg = taccap.ForcePositionConfig()
        missing_fields = []
        for name in (
            "close_position",
            "close_speed_radps",
            "grasp_torque_nm",
            "hold_torque_limit_nm",
            "motion_torque_limit_nm",
            "contact_torque_nm",
            "contact_vel_radps",
            "contact_vel_ratio",
            "contact_moved_rad",
            "position_kp",
            "position_kd",
            "brake_distance_rad",
            "close_endpoint_tolerance_rad",
            "contact_samples",
            "startup_guard_ms",
            "status_timeout_ms",
            "motor_stream_hz",
        ):
            # Keep an older installed native extension usable while it is being
            # rebuilt.  Newer SDKs expose every field; an old .so may not yet
            # expose fields added after its wheel was installed.  Passing such
            # a field raises AttributeError during connect and aborts both
            # grippers before the robot can start.  Skip only the unavailable
            # field and make the required SDK upgrade explicit in the log.
            if hasattr(force_cfg, name):
                setattr(force_cfg, name, getattr(cfg, name))
            else:
                missing_fields.append(name)
        if missing_fields:
            self.logger.warning(
                "Installed xense.taccap native extension lacks ForcePositionConfig "
                f"fields {missing_fields}; using SDK defaults for them. Reinstall "
                "third_party/taccap-gripper to enable the configured endpoint "
                "tolerance and other new safety parameters."
            )
        return taccap.ForcePositionController(self._gripper, force_cfg)

    def _latest_observation(self):
        """Return a GripperObservation from either SDK controller."""

        if self._controller_name == "control_loop":
            return self._loop.observation()
        return self._loop.snapshot().observation

    def _maybe_print_status(
        self,
        observation,
        *,
        controller_state: str | None = None,
        commanded_torque_nm: float | None = None,
    ) -> None:
        """Publish rate-limited diagnostics for the teleop live panel.

        This method deliberately performs no ``Motor.read_status()`` or other
        request/ACK transaction. ``observation`` already came from the motor
        status stream owned by the active SDK controller, so printing cannot
        contend with the control loop on the serial bus.
        """

        if not self._print_status:
            return
        now = time.monotonic()
        if now - self._last_status_print_time < self._status_print_period_s:
            return
        self._last_status_print_time = now

        status = (
            f"{self._side[0].upper()} "
            f"pos={float(observation.position):.3f} "
            f"raw={float(observation.raw_pos):+.4f}rad "
            f"vel={float(observation.velocity):+.2f}rad/s "
            f"tq={float(observation.torque):+.2f}Nm "
            f"temp={float(observation.motor_temp_c):.0f}C "
            f"age={float(observation.age_ms):.1f}ms"
        )
        if self._controller_name == "control_loop":
            status += f" hz={float(self._loop.submit_hz):.1f}"
        else:
            if controller_state is not None:
                status += f" state={controller_state}"
            if commanded_torque_nm is not None:
                status += f" cmd={float(commanded_torque_nm):+.2f}Nm"
        _set_taccap_status_line(self._side, status)

    def connect(self) -> None:
        """Discover the follower, verify calibration, enable the motor, and start
        the background control loop."""
        if self._is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        if taccap is None:
            raise RuntimeError(
                "xense.taccap is not importable — the TacCap SDK native extension is "
                "missing or stale. Rebuild it with `bash setup_env.sh --install` (or "
                "`pip install third_party/taccap-gripper`). "
                f"Original import error: {_TACCAP_IMPORT_ERROR!r}"
            )

        device = self._resolve_device()
        self.logger.info(f"Opening follower gripper (side={self._side}) on {device}...")
        try:
            self._gripper = taccap.FollowerGripper(device)
        except Exception as e:
            self._gripper = None
            raise RuntimeError(f"Failed to open TacCap follower (side={self._side}) on {device}: {e}") from e

        # From here the device handle is open. Anything that throws must release it
        # before propagating: _is_connected is still False, so disconnect() would
        # refuse to run and the handle would be stranded until GC.
        try:
            # Calibration is required for normalized [0, 1] control.
            try:
                cfg = self._gripper.get_gripper_config()
                calibrated = bool(cfg.flags & _CALIBRATED_FLAG)
            except Exception as e:
                calibrated = False
                self.logger.warn(f"Could not read gripper config (assuming uncalibrated): {e}")
            if not calibrated and self._require_calibrated:
                raise RuntimeError(
                    f"TacCap follower (side={self._side}) is not calibrated; normalized [0, 1] "
                    "control requires calibration. Calibrate first (zero at full close, then "
                    "write max_open), or enable firmware power-on auto-cal. See "
                    "third_party/taccap-gripper/python/examples/calibrate.py."
                )

            self._gripper.motor.clear_fault()
            self._loop = self._make_sdk_controller()

            # ForcePositionController.start() validates the persisted motor
            # torque limit before motion, so the SDK requires start-before-enable.
            # ControlLoop retains the existing enable-before-start sequence.
            if self._controller_name == "force_position":
                self._loop.start()
                self._gripper.motor.enable()
                self.logger.info(
                    "TacCap controller=force_position "
                    f"(grasp={self._config.grasp_torque_nm:.3f} Nm, "
                    f"hold_limit={self._config.hold_torque_limit_nm:.3f} Nm)."
                )
            else:
                self._gripper.motor.enable()
                self._loop.start()
                self.logger.info(
                    "TacCap controller=control_loop "
                    f"(kp={self._config.kp:.3f}, kd={self._config.kd:.3f}, "
                    f"ff={self._config.feedforward_torque:+.3f} Nm, "
                    f"phase={self._config.submit_phase})."
                )
        except Exception:
            self._release_after_failed_connect()
            raise

        self._last_target_position = None
        self._is_connected = True
        try:
            self._cached_position = float(self._latest_observation().position)
        except Exception:
            self._cached_position = 1.0 if self._init_open else 0.0
        self.logger.info(f"TacCap follower connected (side={self._side}) on {device}.")

        _clear_taccap_status_line(self._side)
        if self._init_open:
            try:
                self.initialize_gripper_position(1.0)
            except Exception as e:
                self.logger.warn(f"Gripper init-open failed (non-fatal): {e}")

    def _release_after_failed_connect(self) -> None:
        """Best-effort teardown of a partially-opened device.

        Mirrors disconnect(), but tolerates every step being absent — we may be
        anywhere between "handle open" and "loop running" when this fires.
        """
        if self._loop is not None:
            try:
                self._loop.stop()
            except Exception as e:  # pragma: no cover — best-effort
                self.logger.debug(f"Error stopping control loop during rollback: {e}")
            self._loop = None
        if self._gripper is not None:
            try:
                self._gripper.motor.disable()
            except Exception as e:  # pragma: no cover — best-effort
                self.logger.debug(f"Error disabling motor during rollback: {e}")
            self._gripper = None

        _clear_taccap_status_line(self._side)

    def read_wrist_fisheye_calibration(self):
        """The wrist lens' fisheye intrinsics for this gripper.

        Straight passthrough to ``Calibration.resolve_fisheye()``: deciding what
        to rectify with — this unit's stored calibration, or the SDK's reference
        values when it has none — is the SDK's policy, and it applies the same
        one internally when it owns the wrist UVC device. This layer used to
        re-derive it, and the two copies had already drifted on why a read fails.

        Returns:
            ``(calibration, is_reference)`` — the second value is True when the
            reference values stood in, so callers can label or refuse them.
        """
        if self._gripper is None:
            raise DeviceNotConnectedError(f"{self} is not connected; cannot read the wrist fisheye calibration.")

        calibration, is_reference, reason = self._gripper.calibration.resolve_fisheye()
        if is_reference:
            self.logger.warn(
                f"{self}: using the SDK's REFERENCE wrist fisheye intrinsics because "
                f"{reason}. Rectification will be approximate — lens placement varies "
                f"per assembly, so the principal point drifts. Calibrate this unit's "
                f"wrist lens with the PC tool for measurements taken off these frames."
            )
        return calibration, is_reference

    def disconnect(self) -> None:
        """Stop the control loop, disable the motor, and release the device."""
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self._loop is not None:
            try:
                self._loop.stop()
            except Exception as e:
                self.logger.debug(f"Error stopping control loop: {e}")
            self._loop = None

        if self._gripper is not None:
            try:
                self._gripper.motor.disable()
            except Exception as e:
                self.logger.debug(f"Error disabling follower motor: {e}")
            self._gripper = None

        self._is_connected = False
        _clear_taccap_status_line(self._side)
        self.logger.info(f"TacCap follower disconnected (side={self._side}).")

    # ── Position interface ─────────────────────────────────────────────────────

    def get_gripper_position(self) -> float:
        """Return normalized position in [0, 1] from the control loop's latest
        observation (non-blocking).

        Raises:
            DeviceNotConnectedError: If not connected. (This used to return 0.0,
                which is indistinguishable from a genuinely closed jaw — a
                disconnected gripper read as "fully closed" to every caller.)
        """
        if not self._is_connected or self._loop is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        try:
            if self._controller_name == "control_loop":
                observation = self._loop.observation()
                controller_state = None
                commanded_torque_nm = None
            else:
                snapshot = self._loop.snapshot()
                observation = snapshot.observation
                controller_state = str(snapshot.state)
                commanded_torque_nm = snapshot.commanded_torque_nm
            self._cached_position = _clamp01(float(observation.position))
            self._maybe_print_status(
                observation,
                controller_state=controller_state,
                commanded_torque_nm=commanded_torque_nm,
            )
        except Exception as e:
            self.logger.debug(f"observation() read failed, returning cached: {e}")
        return self._cached_position

    def set_gripper_position(self, normalized_pos: float) -> None:
        """Command a normalized target in [0, 1] (0 = closed, 1 = open)."""
        if not self._is_connected or self._loop is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if not 0.0 <= normalized_pos <= 1.0:
            raise ValueError(f"normalized_pos must be in [0, 1], got {normalized_pos}.")
        if (
            self._controller_name == "force_position"
            and self._last_target_position is not None
            and abs(normalized_pos - self._last_target_position) <= _FORCE_POSITION_TARGET_EPS
        ):
            return
        self._loop.set_target(normalized_pos)
        if self._controller_name == "force_position":
            self._last_target_position = normalized_pos


def _clamp01(x: float) -> float:
    return max(0.0, min(x, 1.0))
