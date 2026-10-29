#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Configuration for Elite Robots CS66 arms via libpyelite."""

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from lerobot.cameras.configs import CameraConfig
from lerobot.grippers import GripperConfig
from lerobot.robots.config import RobotConfig


# Not StrEnum, which UP042 suggests: StrEnum changes what str() returns —
# "cartesian_servo" instead of "EliteCS66RTControlMode.CARTESIAN_SERVO" — and
# this value round-trips through draccus into recipe YAML. A serialisation
# change is not a lint fix.
class EliteCS66RTControlMode(str, Enum):  # noqa: UP042
    CARTESIAN_SERVO = "cartesian_servo"
    JOINT_SERVO = "joint_servo"


@RobotConfig.register_subclass("elite_cs66_rt")
@dataclass
class EliteCS66RTConfig(RobotConfig):
    """Configuration for a single Elite CS66 arm.

    The default mode follows the LeRobot Cartesian convention:
    actions/observations use tcp.x/y/z plus 6D rotation tcp.r1..tcp.r6, and the
    driver converts that pose to Elite's native [x, y, z, rx, ry, rz] rotvec
    format at the SDK boundary; native IK takes SE(3) and sends joint servoJ.
    """

    robot_ip: str = "192.168.1.200"
    local_ip: str = ""
    control_mode: EliteCS66RTControlMode = EliteCS66RTControlMode.CARTESIAN_SERVO

    # Observation schema is composable: enable TCP and joint state
    # independently. Use both True for multi-modal datasets (e.g. VLA
    # policies that condition on joint proprio + TCP pose).
    #   observe_tcp=True    -> tcp.x/y/z + tcp.r1..r6 (9 floats)
    #   observe_joints=True -> joint_*.pos/vel/effort (18 floats)
    # Cameras / gripper are independent of both.
    observe_tcp: bool = True
    observe_joints: bool = False

    # Elite Pico4 adapter: each grip press starts from the measured TCP, so
    # re-gripping after a collision hold discards the target behind the obstacle.
    grip_anchor_to_robot: bool = True

    # Elite external control script. When unset, connect() resolves
    # libpyelite/external_control.script from the installed SDK package.
    script_file_path: str | Path | None = None

    # Servo streaming parameters. servoj_time is the controller's inner
    # interpolation period (matches the SDK example at 250 Hz).  servoj_lookahead_time
    # must lie in the SDK-documented range [0.03, 0.2]; outside that range the
    # external_control script aborts and tears down all reverse sockets.
    #
    # servoj_gain is the urscript servoj() position-following P gain. Elite CS
    # does NOT have native Cartesian impedance (unlike flexiv_rizon4_rt's
    # stiffness_ratio), so lowering gain is the only knob we have to make the
    # arm yield under external force. Reference points:
    #   2000  - SDK example default, industrial stiff
    #    500  - moderate, noticeable yield but still tracks target
    #    300  - compliant (default here); rough equivalent of flexiv's
    #           stiffness_ratio=0.2 in feel, NOT in dynamics
    #   <100  - too soft; gravity drift visible on heavy payloads
    # SDK-documented range is [100, 2000]; values outside are checked in
    # __post_init__.
    servoj_time: float = 0.004
    servoj_lookahead_time: float = 0.1
    servoj_gain: int = 300
    command_timeout_ms: int = 200
    # Native streaming always runs; this flag selects async vs blocking reset.
    use_background_servo_loop: bool = True
    command_stale_timeout_s: float = 0.5
    reset_duration_s: float = 3.0

    # Native libpyelite planner; coordinates are still converted by this adapter.
    native_model_path: str | Path | None = None
    native_ik_max_iterations: int = 500
    native_max_tcp_linear_velocity: float = 0.5  # m/s, planning cap
    native_max_tcp_angular_velocity: float = 1.0  # rad/s, planning cap
    native_joint_velocity_scale: float = 1.0  # fraction of official URDF limit
    # Explicit opt-in trajectory limits; values are rad/s, rad/s², rad/s³.
    native_joint_target_horizon: float = 0.0  # deprecated, ignored: limits now track the IK reference
    native_joint_tracking_gain: float | None = None  # 1/s, default 20
    # Cartesian smoothing before IK (opt-in). Limits are magnitudes of the linear /
    # angular TCP twist; velocity comes from native_max_tcp_*_velocity.
    native_cartesian_motion_limits: bool = False
    native_max_linear_acceleration: float | None = None  # m/s², default 2.0
    native_max_angular_acceleration: float | None = None  # rad/s², default 5.0
    native_max_linear_jerk: float | None = None  # m/s³, default 1000
    native_max_angular_jerk: float | None = None  # rad/s³, default 2500
    native_cartesian_tracking_gain: float | None = None  # 1/s, default 20
    native_joint_motion_limits: bool = False
    native_max_joint_velocity: list[float] | None = None
    native_max_joint_acceleration: list[float] | None = None
    native_max_joint_jerk: list[float] | None = None
    native_max_target_translation_jump: float = 0.0
    native_max_target_rotation_jump: float = 0.0
    native_max_target_joint_jump: float = 0.0
    native_check_robot_safety: bool = True
    native_max_contact_wrench: list[float] | None = None
    native_max_start_joint_speed: float = 0.02
    # Joint-current residual collision stop (any link), opt-in. Keys are
    # libpyelite ComplianceConfig fields plus "enabled"; see
    # third_party/libpyelite/examples/contact_monitor.json. gravity is required
    # (base frame), payload to payload_mass/payload_cog.
    native_contact_monitor: dict | None = None

    # Row-major tool0 <- TCP SE(3). None reads the active teach-pendant TCP back at
    # connect (FK_flange(q)^-1 @ measured TCP) and logs it; set it to pin the value.
    tool_transform: list[float] | None = None

    # RTSI state stream.
    rtsi_frequency: float = 250.0
    rtsi_output_recipe: str | Path | None = None
    rtsi_input_recipe: str | Path | None = None

    # Startup and shutdown behavior. The fleet hardcodes the canonical
    # sequence (power on → brake release → script start → clean stopControl
    # on disconnect → RT scheduling best-effort). The only knob left is the
    # overall timeout waiting for the controller-side script to handshake.
    connect_timeout_s: float = 10.0

    # Home / Start poses (J1..J6 in radians). MoveJ-style trajectory used to
    # reach these — see ``_move_j_blocking`` in EliteCS66RT.
    #   home  = safe park position; arm is moved here in disconnect() before
    #           reverse sockets are torn down. Service / shutdown pose.
    #   start = task-ready position; arm is moved here in connect() before
    #           streaming begins. Every episode starts from here.
    # The fields are kept separate by design even though our current fleet
    # uses identical values: most stations want the same candle pose for
    # both, but some workflows (overhead service position vs. workspace-
    # ready) need to differentiate. Override per station as needed.
    # The runtime ``connect(go_to_start=False)`` flag skips the start MoveJ
    # for crash-recovery / re-attach scenarios.
    home_position_rad: list[float] = field(default_factory=lambda: [0.0, -1.5708, -1.5708, -1.5708, 1.5708, 0.0])
    start_position_rad: list[float] = field(default_factory=lambda: [0.0, -1.5708, -1.5708, -1.5708, 1.5708, 0.0])
    start_move_duration_s: float = 3.0
    home_move_duration_s: float = 3.0
    move_j_timeout_ms: int = 200

    # Minimum total wall time between EliteDriver() construction and the
    # first writeServoj. SDK example sleeps ~1 s after isRobotConnected()
    # returns True; otherwise the controller-side script can RST the reverse
    # socket if hit too early. We collapse this to an elapsed-time check —
    # fast handshakes don't pay the full sleep.
    external_control_settle_s: float = 1.0

    # Consecutive reverse-socket write failures the native sender tolerates before
    # faulting. Writes fail transiently right after the external control script comes
    # up, which is when connect() starts the servo. 250 ticks * 0.004 s = 1 s.
    servo_failure_tolerance_ticks: int = 250

    # Trace every send_action and large per-step deltas to the spdlog file
    # sink (~/.xenselogs). Doesn't touch console; safe to leave enabled.
    trace_servoj: bool = True
    # Per-step delta thresholds above which the trace promotes to WARNING (also
    # captured in the file log). 5 cm or 0.5 rad in a single send_action call
    # is suspicious for steady-state teleop.
    trace_translation_threshold: float = 0.05
    trace_rotation_threshold: float = 0.5
    # Joint-mode trace: max per-joint delta (rad) above which the trace
    # promotes to WARNING. 0.3 rad ≈ 17° per send_action tick — at 50 Hz outer
    # loop that's ~855°/s, well above normal leader-follower joint speeds.
    trace_joint_threshold: float = 0.3

    # Recipe-level TCP speed caps (m/s, rad/s). The native planner limits its
    # Cartesian path to min(this, native_max_tcp_*), so these only ever tighten the
    # planner default; None leaves native_max_tcp_* as the effective cap. Set them to
    # smooth a jumpy leader (fast wrist rotation, VR tracking spike, clutch re-engage).
    max_lin_speed: float | None = None  # m/s; None keeps native_max_tcp_linear_velocity
    max_ang_speed: float | None = None  # rad/s; None keeps native_max_tcp_angular_velocity

    # Workspace reachability guard. Distance (m) from the base-frame origin beyond
    # which a commanded TCP target is treated as unreachable: the driver then HOLDS
    # the last in-reach pose instead of sending it, so the operator can't drive the
    # arm into the boundary singularity where the controller's IK fails and drops
    # external control. Conservative spherical guard from the base ORIGIN (not an
    # exact reachability test — the true workspace is offset to the shoulder, so it
    # can read too tight in some directions). Real reach ~0.91 m for CS66.
    #
    # DISABLED by default (None) per operator request 2026-07-03: 0.85 m was clipping
    # legitimate reach and holding the arm mid-teleop. Trade-off to be aware of: with
    # the guard off, over-reaching no longer HOLDS gracefully — the command is sent,
    # the controller hits the boundary singularity, and external control DROPS. Set a
    # value (e.g. 0.88-0.90, below the ~0.91 physical reach) to re-enable if that
    # dropout becomes a problem. Applies to both the background and direct servo paths.
    max_reach_radius: float | None = None  # m; None disables the guard

    # ── Gripper ── A typed block in the recipe, e.g.
    #     gripper:
    #       type: serial            # or taccap_follower
    #       side: left
    #       gripper_f_max: 30.0
    # Decoded through lerobot.grippers.GripperConfig, so a knob belonging to the
    # other backend (or a typo) is rejected at parse time rather than ignored.
    # None = no gripper; gripper.pos is then absent from the feature schema.
    #
    # A single arm has no side to infer, so `side` (or `port`/`sn`) has to be set
    # in the block for the serial backend to find its board.
    gripper: GripperConfig | None = None

    # External cameras.
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Set in __post_init__ from the gripper block's own auto_discover_cameras flag:
    # when on, the wrist/tactile cameras are sniffed by the robot at connect rather
    # than pinned in the recipe. Derived, not settable from a recipe.
    _taccap_autodiscover: bool = field(default=False, init=False)
    _serial_autodiscover: bool = field(default=False, init=False)

    def __post_init__(self):
        if self.native_ik_max_iterations < 1:
            raise ValueError("native_ik_max_iterations must be positive")
        for name in (
            "native_max_tcp_linear_velocity",
            "native_max_tcp_angular_velocity",
            "native_joint_velocity_scale",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.native_joint_velocity_scale > 1:
            raise ValueError("native_joint_velocity_scale must not exceed 1")
        for name in ("tool_transform",):
            tool = getattr(self, name)
            if tool is not None and (len(tool) != 16 or not all(math.isfinite(v) for v in tool)):
                raise ValueError(f"{name} must be a finite row-major 4x4 transform")
        super().__post_init__()

        if not 0.004 <= self.servoj_time <= 0.01:
            raise ValueError(
                f"servoj_time must be in [0.004, 0.01] s, got {self.servoj_time}: the native planner "
                "solves one IK step per servoj_time, and below 4 ms it overruns its period"
            )
        if not 0.03 <= self.servoj_lookahead_time <= 0.2:
            # Elite SDK EliteDriver.hpp says lookahead time must lie in [0.03, 0.2];
            # values outside this range cause the external_control script to abort
            # and tear down all reverse sockets (50001/50003/50004), leaving the
            # Python side writing into a dead socket forever.
            raise ValueError(
                "servoj_lookahead_time must be in [0.03, 0.2] (Elite SDK requirement), "
                f"got {self.servoj_lookahead_time}"
            )
        if not 100 <= self.servoj_gain <= 2000:
            # SDK-documented range for servoj() position-following gain.
            # Outside the range the controller's IK behavior is unspecified
            # (very low: unable to track even slow targets; very high: IK
            # oscillation / velocity-limit trips).
            raise ValueError(f"servoj_gain must be in [100, 2000] (Elite SDK requirement), got {self.servoj_gain}")
        for _name, _val in (
            ("max_lin_speed", self.max_lin_speed),
            ("max_ang_speed", self.max_ang_speed),
            ("max_reach_radius", self.max_reach_radius),
        ):
            if _val is not None and _val <= 0:
                raise ValueError(f"{_name} must be > 0 when set (None disables it), got {_val}")
        if self.command_timeout_ms < 5:
            raise ValueError(f"command_timeout_ms must be >= 5 (Elite SDK lower bound), got {self.command_timeout_ms}")
        if self.command_stale_timeout_s <= 0:
            raise ValueError(f"command_stale_timeout_s must be > 0, got {self.command_stale_timeout_s}")
        if self.command_stale_timeout_s * 1000 < self.command_timeout_ms:
            # The host-side stale gate must trip later than the controller's
            # own command timeout. Otherwise, during the window between
            # command_timeout_ms (controller stops) and command_stale_timeout_s
            # (host writes idle), nobody is sending anything and the controller
            # has already entered its self-protect stop, causing surprise
            # halts during transient outer-loop hiccups. Keeping
            # command_stale_timeout_s >> command_timeout_ms ensures the host
            # always remains the active party in the control channel.
            raise ValueError(
                "command_stale_timeout_s * 1000 must be >= command_timeout_ms "
                f"(host stale must trigger later than controller timeout); "
                f"got command_stale_timeout_s={self.command_stale_timeout_s}s, "
                f"command_timeout_ms={self.command_timeout_ms}ms"
            )
        if self.reset_duration_s <= 0:
            raise ValueError(f"reset_duration_s must be > 0, got {self.reset_duration_s}")
        if self.rtsi_frequency <= 0:
            raise ValueError(f"rtsi_frequency must be > 0, got {self.rtsi_frequency}")
        if self.connect_timeout_s <= 0:
            raise ValueError(f"connect_timeout_s must be > 0, got {self.connect_timeout_s}")
        if len(self.home_position_rad) != 6:
            raise ValueError(f"home_position_rad must have 6 elements (J1..J6), got {len(self.home_position_rad)}")
        if len(self.start_position_rad) != 6:
            raise ValueError(f"start_position_rad must have 6 elements (J1..J6), got {len(self.start_position_rad)}")
        if self.start_move_duration_s <= 0:
            raise ValueError(f"start_move_duration_s must be > 0, got {self.start_move_duration_s}")
        if self.home_move_duration_s <= 0:
            raise ValueError(f"home_move_duration_s must be > 0, got {self.home_move_duration_s}")
        if self.move_j_timeout_ms < 5:
            raise ValueError(
                f"move_j_timeout_ms must be >= 5 (Elite SDK lower bound; mirrors "
                f"command_timeout_ms), got {self.move_j_timeout_ms}"
            )
        if self.external_control_settle_s < 0:
            raise ValueError(f"external_control_settle_s must be >= 0, got {self.external_control_settle_s}")
        if self.servo_failure_tolerance_ticks < 1:
            raise ValueError(f"servo_failure_tolerance_ticks must be >= 1, got {self.servo_failure_tolerance_ticks}")

        # ── Cameras ── With auto-discovery on, the wrist + tactile cameras travel
        # with the gripper, so the robot sniffs them at connect (see
        # _inject_taccap_cameras / _inject_serial_gripper_cameras) and the recipe
        # pins only the scene cameras. Otherwise the recipe pins every camera.
        gt = self.gripper.type if self.gripper is not None else None
        discover = self.gripper is not None and self.gripper.auto_discover_cameras
        self._taccap_autodiscover = discover and gt == "taccap_follower"
        self._serial_autodiscover = discover and gt == "serial"
