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

"""Configuration for the TacCap follower (actuated) gripper.

Wraps ``xense.taccap.FollowerGripper``. The recipe selects either the
position-impedance ``ControlLoop`` or contact-aware
``ForcePositionController``; both run in the SDK background so reads/writes
remain non-blocking. Left/right units are told apart automatically by the
firmware-burned serial number (``side``), so no per-unit SN/port needs
configuring in the common case.
"""

import math
from dataclasses import dataclass

from ..configs import GripperConfig

# Hard bound on the constant feed-forward torque, to catch sign/scale typos before
# they reach the motor. The MIT impedance path applies feed-forward with NO firmware
# max_torque clamp (only position/velocity modes clamp), and ~3.5 Nm is the top of the
# motor's usable envelope (cf. the max_torque values in the codec tests). This is a
# safety rail, not a recommendation — the gentle-grasp example aborts at 0.30 Nm.
MAX_FEEDFORWARD_TORQUE_NM = 3.5
TACCAP_CONTROLLERS = ("control_loop", "force_position")
TACCAP_SUBMIT_PHASES = ("stream_locked", "free_running")
TACCAP_STALL_ACTIONS = ("hold_position", "none")
FORCE_POSITION_MAX_HOLD_TORQUE_NM = 1.8
FORCE_POSITION_MAX_MOTION_TORQUE_NM = 6.0


@GripperConfig.register_subclass("taccap_follower")
@dataclass(kw_only=True)
class TaccapFollowerConfig(GripperConfig):
    """Configuration for a single TacCap follower gripper.

    Identification:
        side:        Which physical unit to drive, ``"left"`` or ``"right"``.
                     Resolved at connect() time via the SDK's side-aware device
                     discovery (``find_left()`` / ``find_right()``), which reads
                     the firmware-burned serial number. Ignored when
                     ``mcu_device`` is set.
        mcu_device:  Optional explicit MCU device path (e.g. ``"/dev/ttyACM0"``)
                     to bypass side-based discovery. Use only when auto-discovery
                     is not viable.

    Controller:
        controller: Selects the SDK background controller at connect time.
                     ``control_loop`` is normalized position impedance;
                     ``force_position`` closes with bounded velocity/damping,
                     detects contact, then holds ``grasp_torque_nm`` using pure
                     feed-forward torque. Switching requires restarting the
                     LeRobot command; YAML is not hot-reloaded.

    ControlLoop:
        kp/kd:       Position stiffness and velocity damping.
        feedforward_torque: Constant torque bias added to every impedance frame.
                     Negative closes and positive opens. It is not a target
                     torque and remains active with an empty jaw.
        control_hz:  Used only by ``free_running``. The default
                     ``stream_locked`` phase submits once per motor-status frame,
                     at ``motor_stream_hz``.

    ForcePositionController:
        grasp_torque_nm: Positive target torque magnitude used after contact.
        contact_torque_nm: Contact detector floor, not the grasp target.
        hold_torque_limit_nm: Long-term torque ceiling (SDK maximum 1.8 Nm).
        motion_torque_limit_nm: Transient motion ceiling (SDK maximum 6.0 Nm).

    Behavior:
        init_open:       If True, drive fully open on ``connect()``.
        require_calibrated: If True, refuse to connect when the gripper reports
                     an uncalibrated ``GripperConfig`` (normalized [0, 1] control
                     needs calibration). Set False only for bring-up/debug.
    """

    # ── Identification ─────────────────────────────────────────────────────────
    side: str = "left"  # "left" | "right" (firmware-SN auto side)
    mcu_device: str | None = None  # optional explicit device path override

    # ── SDK controller selection ───────────────────────────────────────────────
    controller: str = "control_loop"  # "control_loop" | "force_position"

    # ── ControlLoop (position impedance) ───────────────────────────────────────
    kp: float = 8.0  # Nm/rad
    kd: float = 1.0  # Nm·s/rad
    feedforward_torque: float = 0.0  # Nm; NEGATIVE = closing/clamp, POSITIVE = opening
    control_hz: int = 100  # ControlLoop resubmit rate (ignored while phase-locked)
    submit_phase: str = "stream_locked"  # "stream_locked" | "free_running"
    max_position_torque_nm: float = 1.5
    rated_torque_nm: float = 2.0
    rated_hold_ms: int = 20
    rated_release_rad: float = 0.05
    stall_torque_nm: float = 1.2
    stall_vel_radps: float = 0.15
    stall_hold_ms: int = 60
    stall_action: str = "hold_position"  # "hold_position" | "none"

    # Both SDK controllers own the same motor-status stream. The current
    # transport is hardware-validated at no more than 100 Hz.
    motor_stream_hz: int = 100

    # ── ForcePositionController (contact-aware force/position) ─────────────────
    close_position: float = 0.0
    close_speed_radps: float = 0.5
    grasp_torque_nm: float = 0.35
    hold_torque_limit_nm: float = FORCE_POSITION_MAX_HOLD_TORQUE_NM
    motion_torque_limit_nm: float = FORCE_POSITION_MAX_MOTION_TORQUE_NM
    contact_torque_nm: float = 0.080
    contact_vel_radps: float = 0.035
    contact_vel_ratio: float = 0.25
    contact_moved_rad: float = 0.010
    position_kp: float = 20.0
    position_kd: float = 1.0
    brake_distance_rad: float = 0.10
    close_endpoint_tolerance_rad: float = 0.03
    contact_samples: int = 3
    startup_guard_ms: int = 250
    status_timeout_ms: int = 350

    # ── Behavior ───────────────────────────────────────────────────────────────
    init_open: bool = True
    require_calibrated: bool = True
    # Read the already-streamed SDK snapshot and publish one compact row per
    # gripper into the teleop live panel at this lower update rate. This never
    # polls Motor.read_status() and therefore adds no traffic to the control bus.
    print_status: bool = False
    status_print_hz: float = 5.0
    # On by default: a TacCap gripper is a self-contained USB hub carrying its own
    # wrist camera and two GSPS sensors, so they travel with the gripper and are
    # cheaper to sniff than to pin per bench.
    auto_discover_cameras: bool = True

    # ── Wrist fisheye rectification ────────────────────────────────────────────
    # The wrist lens is a fisheye and its intrinsics are burned into this
    # gripper's own MCU flash, which is why the switch belongs here rather than
    # on the arm: swap the gripper and both the lens and its calibration go with
    # it. The serial (XGripper) family holds no such record, so it has neither
    # field — a recipe that writes them on an XGripper block is refused at parse
    # rather than quietly ignored.
    #
    # Off by default: nothing changes for a rig that has not opted in. When a
    # unit's firmware holds no calibration, the SDK's shared reference intrinsics
    # are used with a warning — close, but not this unit's, so calibrate before
    # measuring in pixels off a rectified frame.
    undistort_wrist: bool = False
    # 0.0 keeps the calibrated focal length (natural view, the PC tool's default);
    # 1.0 shortens it to 0.70x for the widest field of view, with more black
    # border. Only fx/fy move — the principal point stays put, so the view does
    # not drift as this turns.
    fisheye_balance: float = 0.0

    def __post_init__(self):
        if self.side not in ("left", "right"):
            raise ValueError(f"TaccapFollowerConfig: side must be 'left' or 'right', got {self.side!r}.")
        if self.controller not in TACCAP_CONTROLLERS:
            raise ValueError(
                f"TaccapFollowerConfig: controller must be one of {TACCAP_CONTROLLERS}, "
                f"got {self.controller!r}."
            )
        if not self.kp > 0.0:
            raise ValueError(f"TaccapFollowerConfig: kp must be positive, got {self.kp}.")
        if not self.kd >= 0.0:
            raise ValueError(f"TaccapFollowerConfig: kd must be non-negative, got {self.kd}.")
        if abs(self.feedforward_torque) > MAX_FEEDFORWARD_TORQUE_NM:
            raise ValueError(
                f"TaccapFollowerConfig: |feedforward_torque| must be <= "
                f"{MAX_FEEDFORWARD_TORQUE_NM} Nm, got {self.feedforward_torque}. "
                "Sign: negative = closing/clamp, positive = opening. Values past "
                "~1 Nm are a hard crush (the SDK's gentle-grasp example aborts at 0.30 Nm)."
            )
        # The old ceiling was 500, taken from the firmware's slave control rate.
        # Measured against hw v1.1.2.0, free-running submits at 250 Hz cost
        # status frames on every run and 500 Hz collapsed the stream to 24
        # frames/s, so that ceiling was never safe to actually use. 200 is the
        # highest rate we tested without observing loss -- which is not the same
        # as proving it safe, since the collision is phase-dependent rather than
        # rate-dependent.
        if not 0 < self.control_hz <= 200:
            raise ValueError(
                f"TaccapFollowerConfig: control_hz must be in (0, 200], got {self.control_hz}. "
                "Rates at or above 250 Hz measurably cost motor-status frames when the SDK's "
                "control loop runs free (see tc-gu-01 issue #1); the default phase ignores this "
                "value entirely and submits at the status-stream rate."
            )
        if self.submit_phase not in TACCAP_SUBMIT_PHASES:
            raise ValueError(
                f"TaccapFollowerConfig: submit_phase must be one of {TACCAP_SUBMIT_PHASES}, "
                f"got {self.submit_phase!r}."
            )
        if self.stall_action not in TACCAP_STALL_ACTIONS:
            raise ValueError(
                f"TaccapFollowerConfig: stall_action must be one of {TACCAP_STALL_ACTIONS}, "
                f"got {self.stall_action!r}."
            )
        if not 0 < self.motor_stream_hz <= 100:
            raise ValueError(
                f"TaccapFollowerConfig: motor_stream_hz must be in [1, 100], got {self.motor_stream_hz}."
            )

        non_negative = {
            "max_position_torque_nm": self.max_position_torque_nm,
            "rated_torque_nm": self.rated_torque_nm,
            "rated_release_rad": self.rated_release_rad,
            "stall_torque_nm": self.stall_torque_nm,
            "stall_vel_radps": self.stall_vel_radps,
            "contact_moved_rad": self.contact_moved_rad,
            "position_kd": self.position_kd,
            "brake_distance_rad": self.brake_distance_rad,
            "close_endpoint_tolerance_rad": self.close_endpoint_tolerance_rad,
        }
        for name, value in non_negative.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"TaccapFollowerConfig: {name} must be finite and >= 0, got {value}."
                )
        for name, value in {
            "rated_hold_ms": self.rated_hold_ms,
            "stall_hold_ms": self.stall_hold_ms,
            "startup_guard_ms": self.startup_guard_ms,
        }.items():
            if value < 0:
                raise ValueError(f"TaccapFollowerConfig: {name} must be >= 0, got {value}.")

        if not math.isfinite(self.close_position) or not 0.0 <= self.close_position <= 1.0:
            raise ValueError(
                f"TaccapFollowerConfig: close_position must be in [0, 1], got {self.close_position}."
            )
        positive = {
            "close_speed_radps": self.close_speed_radps,
            "grasp_torque_nm": self.grasp_torque_nm,
            "contact_torque_nm": self.contact_torque_nm,
            "contact_vel_radps": self.contact_vel_radps,
            "position_kp": self.position_kp,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"TaccapFollowerConfig: {name} must be finite and > 0, got {value}."
                )
        if not 0.0 < self.hold_torque_limit_nm <= FORCE_POSITION_MAX_HOLD_TORQUE_NM:
            raise ValueError(
                "TaccapFollowerConfig: hold_torque_limit_nm must be in (0, 1.8], "
                f"got {self.hold_torque_limit_nm}."
            )
        if not 0.0 < self.motion_torque_limit_nm <= FORCE_POSITION_MAX_MOTION_TORQUE_NM:
            raise ValueError(
                "TaccapFollowerConfig: motion_torque_limit_nm must be in (0, 6.0], "
                f"got {self.motion_torque_limit_nm}."
            )
        if self.hold_torque_limit_nm > self.motion_torque_limit_nm:
            raise ValueError(
                "TaccapFollowerConfig: hold_torque_limit_nm must not exceed motion_torque_limit_nm."
            )
        if self.grasp_torque_nm > self.hold_torque_limit_nm:
            raise ValueError(
                "TaccapFollowerConfig: grasp_torque_nm must not exceed hold_torque_limit_nm."
            )
        if self.contact_torque_nm > self.grasp_torque_nm:
            raise ValueError(
                "TaccapFollowerConfig: contact_torque_nm must not exceed grasp_torque_nm; "
                "otherwise contact can never latch."
            )
        if not math.isfinite(self.contact_vel_ratio) or not 0.0 < self.contact_vel_ratio <= 1.0:
            raise ValueError(
                f"TaccapFollowerConfig: contact_vel_ratio must be in (0, 1], got {self.contact_vel_ratio}."
            )
        if self.contact_samples <= 0:
            raise ValueError(
                f"TaccapFollowerConfig: contact_samples must be > 0, got {self.contact_samples}."
            )
        if self.status_timeout_ms <= 0:
            raise ValueError(
                f"TaccapFollowerConfig: status_timeout_ms must be > 0, got {self.status_timeout_ms}."
            )
        if not math.isfinite(self.status_print_hz) or self.status_print_hz <= 0.0:
            raise ValueError(
                "TaccapFollowerConfig: status_print_hz must be finite and > 0, "
                f"got {self.status_print_hz}."
            )
        if not 0.0 <= self.fisheye_balance <= 1.0:
            raise ValueError(f"TaccapFollowerConfig: fisheye_balance must be in [0, 1], got {self.fisheye_balance}.")
        if self.undistort_wrist and not self.auto_discover_cameras:
            # This combination used to be accepted and do nothing at all: the
            # switch is applied to the wrist camera as it is discovered, so with
            # discovery off there is no camera for it to reach and the rig
            # recorded raw fisheye frames with the knob reading as on. A recipe
            # that pins its cameras by hand sets `undistort` on the wrist camera
            # block instead, where it is next to the resolution it constrains.
            raise ValueError(
                "TaccapFollowerConfig: undistort_wrist=True needs "
                "auto_discover_cameras=True — it is applied to the wrist camera "
                "as it is discovered. With cameras pinned by hand, set "
                "`undistort: true` on the wrist camera block itself."
            )
