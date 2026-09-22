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
# 力矩上限的两个常数(1.8 / 6.0)删了:它们只服务于 hold_torque_limit_nm 和
# motion_torque_limit_nm 的校验,而那两个字段已不在本配置里 —— SDK 自己有同样
# 的边界校验,在这边再留一份只会和 SDK 漂开。


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
                     Defaults to ``force_position``: one bounded-torque control
                     law for the whole move, with the PD request error-clamped
                     against the SDK's grasp budget. It does NOT detect contact
                     — saturation is contact; SDK 0.2.0 deleted the host-side
                     contact state machine because the MCU already runs the
                     same stall test at 500 Hz and is the authority.
                     ``control_loop`` is the lower-level normalized position
                     impedance loop; note it has no fault semantics — a failed
                     submit just stops the loop with nothing saying why.
                     Switching requires restarting the LeRobot command; YAML is
                     not hot-reloaded.

    ControlLoop:
        kp/kd:       Position stiffness and velocity damping.
        feedforward_torque: Constant torque bias added to every impedance frame.
                     Negative closes and positive opens. It is not a target
                     torque and remains active with an empty jaw.
        control_hz:  Used only by ``free_running``. The default
                     ``stream_locked`` phase submits once per motor-status frame,
                     at ``motor_stream_hz``.

    ForcePositionController (the default controller):
        close_speed_radps: Rate of the time-based SETPOINT RAMP during travel —
                     not a velocity command to the motor. Only this controller
                     reads it; ControlLoop's approach speed comes from
                     peak/kd instead.

        Nothing else is exposed. The torque budget, its two ceilings and the
        closed-end preload are all SDK defaults; see the block next to the
        fields below for why each one is better left there.

        The closed-endpoint preload (``close_preload_nm``, 0.25 Nm) is left at
        the SDK default and not exposed here. Add it to this dataclass and to
        the field list in ``taccap_follower.py`` if it ever needs to differ.

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
    # 默认 force_position:唯一在用的 recipe 就是它,而且它是受监督的那个 ——
    # ControlLoop 把 stalled / torque_capped 当成两个独立锁下的标志位,调用方
    # 轮询两者可能读到一个从没同时存在过的组合;更要命的是**它没有故障语义**,
    # 提交失败只会让循环断掉、running 变 false,不告诉你为什么。
    controller: str = "force_position"  # "control_loop" | "force_position"

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

    # ── ForcePositionController (single bounded-torque control law) ────────────
    #
    # SDK 0.2.0 cut ForcePositionConfig from sixteen fields to six (0.2.1 added
    # a seventh). The contact-detection constants and the position gains became
    # detail::ForcePositionTuning on the C++ side, which a caller cannot reach:
    # the MCU already runs the same stall test at 500 Hz and is the authority,
    # so the host was keeping a second copy of one physical event.
    #
    # The eleven knobs that used to live here (close_position,
    # contact_torque_nm, contact_vel_radps, contact_vel_ratio,
    # contact_moved_rad, position_kp, position_kd, brake_distance_rad,
    # close_endpoint_tolerance_rad, contact_samples, startup_guard_ms) were
    # kept past that change and silently stopped doing anything — the setter
    # loop skips fields the SDK no longer declares. Removed rather than left
    # looking tunable.
    #
    # 力矩三兄弟也不在这里了,一律走 SDK 默认:
    #
    #   grasp_torque_nm (SDK 1.1) —— 暴露它只会让人配出得不到的值。固件的
    #     clamp_torque 上限是 min(cont, effective_peak),本机包络 cont=1.1,
    #     所以配 1.8 实际拿到的还是 1.1,只多换来 I2t 累积和掉电风险。SDK 那段
    #     注释记着一次现场事故:1.5Nm 夹持对 1.6Nm 包络就把板子拉到欠压、连
    #     USB 一起断。(我们的 recipe 之前配的正是 1.8 对 1.1。)
    #     注意 1.1 不等于"永远安全":实测持续 1.1Nm 十分钟,电机 33->58°C 且
    #     未收敛。它是厂商的连续额定,不是无限期保证。
    #
    #   hold_torque_limit_nm —— 在当前控制律里**已经不钳任何输出**,只用来
    #     校验 grasp 的上界并在超过电机额定时告警。grasp 不可配之后它无事可做。
    #
    #   motion_torque_limit_nm —— 仍有实功能(预算外钳位 + 反馈超限跳故障),
    #     但 SDK 默认 6.0 就是器件上限,且启动时会和电机的 0x700B 交叉核对、
    #     以设备值为准。上层调低它属于刻意收紧安全边界,没人在做。
    close_speed_radps: float = 3.0
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
                f"TaccapFollowerConfig: controller must be one of {TACCAP_CONTROLLERS}, got {self.controller!r}."
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
                f"TaccapFollowerConfig: submit_phase must be one of {TACCAP_SUBMIT_PHASES}, got {self.submit_phase!r}."
            )
        if self.stall_action not in TACCAP_STALL_ACTIONS:
            raise ValueError(
                f"TaccapFollowerConfig: stall_action must be one of {TACCAP_STALL_ACTIONS}, got {self.stall_action!r}."
            )
        if not 0 < self.motor_stream_hz <= 100:
            raise ValueError(f"TaccapFollowerConfig: motor_stream_hz must be in [1, 100], got {self.motor_stream_hz}.")

        non_negative = {
            "max_position_torque_nm": self.max_position_torque_nm,
            "rated_torque_nm": self.rated_torque_nm,
            "rated_release_rad": self.rated_release_rad,
            "stall_torque_nm": self.stall_torque_nm,
            "stall_vel_radps": self.stall_vel_radps,
        }
        for name, value in non_negative.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"TaccapFollowerConfig: {name} must be finite and >= 0, got {value}.")
        for name, value in {
            "rated_hold_ms": self.rated_hold_ms,
            "stall_hold_ms": self.stall_hold_ms,
        }.items():
            if value < 0:
                raise ValueError(f"TaccapFollowerConfig: {name} must be >= 0, got {value}.")

        positive = {
            "close_speed_radps": self.close_speed_radps,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"TaccapFollowerConfig: {name} must be finite and > 0, got {value}.")
        if self.status_timeout_ms <= 0:
            raise ValueError(f"TaccapFollowerConfig: status_timeout_ms must be > 0, got {self.status_timeout_ms}.")
        if not math.isfinite(self.status_print_hz) or self.status_print_hz <= 0.0:
            raise ValueError(
                f"TaccapFollowerConfig: status_print_hz must be finite and > 0, got {self.status_print_hz}."
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
