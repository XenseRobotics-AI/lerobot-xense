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

Wraps ``xense.taccap.FollowerGripper``, driven by the SDK's
``ForcePositionController``, which runs its own background loop so reads and
writes here remain non-blocking. Left/right units are told apart automatically by the
firmware-burned serial number (``side``), so no per-unit SN/port needs
configuring in the common case.
"""

import math
from dataclasses import dataclass

from ..configs import GripperConfig

# 只剩一个控制器。`control_loop` 连同它那一整组字段(kp/kd/前馈/提交相位/
# 失速守卫)随 SDK 删除 `ControlLoop` 一起去掉了 —— 那是阻抗控制律的第二份
# 拷贝,已经和 ImpedanceController 漂开:预算不同,还多一个会把夹持力打塌的
# 失速守卫(实测接触后 60ms 在 0.35Nm 松手)。
#
# 元组只有一个元素但保留着:recipe 里的 `controller: force_position` 仍然要
# 合法,而下一个候选 —— 走 SDK 的 ImpedanceController —— 一旦接上来就是在这里
# 选。前馈力矩的那条上限常数(3.5 Nm)一并删了:本配置再没有前馈字段,SDK 侧
# 的 ImpedanceConfig 自己会校验 `预算 + |前馈| < 额定`。
TACCAP_CONTROLLERS = ("force_position",)
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
                     ``force_position`` is the only value: one bounded-torque
                     control law for the whole move, with the PD request
                     error-clamped against the SDK's grasp budget. It does NOT
                     detect contact — saturation is contact; SDK 0.2.0 deleted
                     the host-side contact state machine because the MCU already
                     runs the same stall test at 500 Hz and is the authority.
                     ``control_loop`` is gone along with the SDK's ``ControlLoop``
                     (see the note next to TACCAP_CONTROLLERS). A recipe still
                     naming it is refused at parse, not silently downgraded.

    ForcePositionController:
        close_speed_radps: Rate of the time-based SETPOINT RAMP during travel —
                     not a velocity command to the motor.

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
    # 只有一个合法值,见 TACCAP_CONTROLLERS 旁边那段。字段留着是因为 recipe 已经
    # 在写它,而且加 impedance 的话就是在这里选。
    controller: str = "force_position"  # "force_position"

    # The SDK controller owns the motor-status stream. The current transport is
    # hardware-validated at no more than 100 Hz.
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
        if not 0 < self.motor_stream_hz <= 100:
            raise ValueError(f"TaccapFollowerConfig: motor_stream_hz must be in [1, 100], got {self.motor_stream_hz}.")

        if not math.isfinite(self.close_speed_radps) or self.close_speed_radps <= 0.0:
            raise ValueError(
                f"TaccapFollowerConfig: close_speed_radps must be finite and > 0, got {self.close_speed_radps}."
            )
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
