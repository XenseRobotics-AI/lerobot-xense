#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Configuration for BiEliteCS66RT dual-arm robot (two Elite CS66 controllers).

Relationship to ``EliteCS66RTConfig`` mirrors ``BiFlexivRizon4RTConfig`` vs.
``FlexivRizon4RTConfig``: shared control/servo parameters keep the single-arm
names and defaults. Bench hardware (controller IPs, start/home poses, mount
geometry, camera SNs) is supplied by the recipe — each recipe under ``recipes/``
is self-contained. Serial grippers self-sort left/right by board-SN parity at
connect, so no gripper SN is configured. Action / observation keys are
``left_``/``right_`` prefixed.
"""

import math
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path

from lerobot.cameras.configs import CameraConfig
from lerobot.grippers import GripperConfig
from lerobot.robots.config import RobotConfig

ROBOT_TYPE = "bi_elite_cs66_rt"


class BiEliteCS66RTControlMode(str, Enum):
    CARTESIAN_SERVO = "cartesian_servo"
    JOINT_SERVO = "joint_servo"


@RobotConfig.register_subclass("bi_elite_cs66_rt")
@dataclass
class BiEliteCS66RTConfig(RobotConfig):
    """Configuration for two Elite CS66 arms via libpyelite.

    Each arm runs its own EliteDriver + RTSI stream + (optional) background
    Cartesian servo loop. ``send_action``/``get_observation`` use the same TCP /
    joint schema as the single-arm driver, ``left_``/``right_`` prefixed:
        left_tcp.x/y/z + left_tcp.r1..r6 (+ optional left_joint_*), left_gripper.pos
        right_tcp.x/y/z + right_tcp.r1..r6 (+ optional right_joint_*), right_gripper.pos
    Grippers are per-arm serial (USB) devices that self-sort left/right by board-SN
    parity at connect. Cameras (head + per-arm wrist + optional tactiles) live at the
    bimanual level; tactile images come from separate XenseTactileCamera devices, not
    the gripper.

    Bench hardware (controller IPs, start/home poses, mount geometry, camera SNs)
    is supplied by the recipe; see ``recipes/README.md``.
    """

    # ── Per-arm identity / connection ──
    left_robot_ip: str = ""
    right_robot_ip: str = ""
    left_local_ip: str = ""  # "" lets the OS route; set to pin the host NIC for RTSI
    right_local_ip: str = ""

    # ── Per-arm mounting → base↔world rotation ──
    # R_world←base = Rz(γ)·Rz(β)·Rx(α): α=tilt about base-X, β=rotate about Z (both
    # from the teach pendant, fixing only the gravity vector), γ=extra world-Z yaw
    # that aligns each arm's heading into ONE shared world frame (x=facing, y=left,
    # z=up). The driver lifts base→world in get_observation and maps world→base in
    # send_action. On the existing diagonal benches both pendants read α=45/β=90 and
    # the arms are point-symmetric, so left needs γ=180° (right γ=0° defines the
    # world) — set per bench in the recipe, not defaulted here.
    left_mount_tilt_deg: float | None = None
    left_mount_zrot_deg: float | None = None
    left_mount_world_yaw_deg: float | None = None
    right_mount_tilt_deg: float | None = None
    right_mount_zrot_deg: float | None = None
    right_mount_world_yaw_deg: float | None = None

    # Explicit per-arm world<-base rotation (3x3, rows = world X/Y/Z in base). When
    # set it OVERRIDES the angle build above for that arm — needed when the mounting
    # isn't a clean Rz·Rx (e.g. the left arm tilts about base-Y).
    #   None -> no matrix; build the rotation from the angles above
    #   []   -> same as None (accepted so a recipe can explicitly opt out)
    # Normalized to None in __post_init__, so the driver only ever sees None or 3x3.
    left_world_rotation: list[list[float]] | None = None
    right_world_rotation: list[list[float]] | None = None

    # ── Shared control mode + observation schema ──
    control_mode: BiEliteCS66RTControlMode = BiEliteCS66RTControlMode.CARTESIAN_SERVO
    observe_tcp: bool = True
    observe_joints: bool = False

    # Elite Pico4 adapter: each grip press starts from the measured TCP, so
    # re-gripping after a collision hold discards the target behind the obstacle.
    grip_anchor_to_robot: bool = True

    # Elite external control script (shared; resolved from the SDK when unset).
    script_file_path: str | Path | None = None

    # ── Shared servo streaming parameters (see config_elite_cs66_rt.py) ──
    servoj_time: float = 0.004
    servoj_lookahead_time: float = 0.1
    servoj_gain: int = 300
    # Preserve the station's reverse-socket receive timeout.
    command_timeout_ms: int = 500
    # Native streaming always runs; this flag selects async vs blocking reset.
    use_background_servo_loop: bool = True
    # Host-side stale threshold; validated as command_stale_timeout_s*1000 >=
    # command_timeout_ms so the host keeps feeding (idle) before the controller times out.
    command_stale_timeout_s: float = 0.5
    # Deprecated compatibility field; no Python FIFO threads are created.
    servo_fifo_scheduling: bool = False
    reset_duration_s: float = 3.0
    # Recipe-level TCP speed caps, per arm (m/s, rad/s). The native planner limits its
    # Cartesian path to min(this, native_max_tcp_*), so these only ever tighten the
    # planner default; None leaves native_max_tcp_* as the effective cap. Set them to
    # smooth a jumpy leader (fast wrist rotation, VR tracking spike, clutch re-engage).
    max_lin_speed: float | None = None  # m/s; None keeps native_max_tcp_linear_velocity
    max_ang_speed: float | None = None  # rad/s; None keeps native_max_tcp_angular_velocity

    # Workspace reachability guard (per arm). Distance (m) from each arm's base
    # origin beyond which a commanded TCP target is treated as unreachable: the
    # driver HOLDS the last in-reach pose instead of sending it, preventing the
    # operator from driving an arm into the boundary singularity where the
    # controller's IK fails and external control drops (the observed
    # "writeServoj failed 251 ticks in a row" crash). Conservative spherical guard
    # from the base ORIGIN (real reach ~0.91 m for CS66); can read too tight in
    # some directions since the true workspace is offset to the shoulder.
    #
    # DISABLED by default (None) per operator request 2026-07-03: 0.85 m clipped
    # legitimate reach and held the (right) arm mid-teleop. Trade-off: with it off,
    # over-reaching DROPS external control at the boundary instead of holding. Set
    # 0.88-0.90 to re-enable. See config_elite_cs66_rt.py for rationale.
    max_reach_radius: float | None = None  # m; None disables the guard

    # Native libpyelite planner; coordinates are still converted by this adapter.
    native_model_path: str | Path | None = None
    # Packaged assembly supplies arm chains and full world<-base transforms.
    # None retains the legacy single-arm model and rotation-only mapping. The name is
    # checked by libpyelite.assembly.Assembly when the robot is built, not duplicated here.
    native_assembly: str | None = None
    native_ik_max_iterations: int = 500
    native_max_tcp_linear_velocity: float = 0.5  # m/s, planning cap (native/single-arm default)
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
    # third_party/libpyelite/examples/contact_monitor.json. gravity defaults to
    # the mount rotation, payload to payload_mass/payload_cog.
    native_contact_monitor: dict | None = None
    # Collision avoidance (table, stand, self, other arm): null, "auto" (the
    # assembly's collision.json) or a path. Needs native_assembly; the robot then
    # runs both arms through the same StationGuard as libpyelite.sim.
    # Hardware collision checking supports TacCap followers only, not retired XGrippers.
    native_collision_profile: str | None = None
    # Actual sensor opening -> URDF driver coordinate calibration, per side.
    # {left: {left_tool_taccap_right_finger_joint: [[opening, rad], ...]}, right: {...}}
    # Must be measured; URDF limits do not establish motor/finger gearing.
    native_collision_gripper_calibration: dict[str, dict[str, list[list[float]]]] | None = None
    native_collision_gripper_max_age_s: float = 0.1
    native_collision_gripper_publish_hz: float = 100.0
    left_native_contact_monitor: dict | None = None  # per-arm overrides (fitted gravity, friction)
    right_native_contact_monitor: dict | None = None

    # Row-major tool0 <- TCP SE(3). None reads the active teach-pendant TCP back at
    # connect (FK_flange(q)^-1 @ measured TCP) and logs it; set it to pin the value.
    left_tool_transform: list[float] | None = None
    right_tool_transform: list[float] | None = None

    # ── Shared RTSI state stream ──
    rtsi_frequency: float = 250.0
    rtsi_output_recipe: str | Path | None = None
    rtsi_input_recipe: str | Path | None = None

    # ── Shared startup / shutdown behavior ──
    connect_timeout_s: float = 10.0
    external_control_settle_s: float = 1.0
    # See config_elite_cs66_rt.py: write failures tolerated before the native sender faults.
    servo_failure_tolerance_ticks: int = 250

    # ── Per-arm EliteDriver local TCP port offsets ──
    # Each EliteDriver opens host-side reverse/trajectory/script-command TCP
    # servers (SDK defaults 50001/50003/50004, +script_sender 50002). Two drivers
    # on one host must NOT share these, so the right arm's block is offset. The
    # SDK templates the offset ports into the pushed external_control.script, so
    # the controller connects back to the matching ports. Must differ between arms.
    left_driver_port_offset: int = 0
    right_driver_port_offset: int = 10

    # ── Per-arm Home / Start poses (J1..J6 radians) ──
    # Defaults are the single-arm candle pose; every real bench overrides all four
    # in its recipe (the arms are mounted at an angle, so the candle is not a
    # usable start there).
    left_start_position_rad: list[float] = field(default_factory=lambda: [0.0, -1.5708, -1.5708, -1.5708, 1.5708, 0.0])
    right_start_position_rad: list[float] = field(default_factory=lambda: [0.0, -1.5708, -1.5708, -1.5708, 1.5708, 0.0])
    left_home_position_rad: list[float] = field(default_factory=lambda: [0.0, -1.5708, -1.5708, -1.5708, 1.5708, 0.0])
    right_home_position_rad: list[float] = field(default_factory=lambda: [0.0, -1.5708, -1.5708, -1.5708, 1.5708, 0.0])
    start_move_duration_s: float = 3.0
    home_move_duration_s: float = 3.0
    # Controller-side reverse-socket recv budget for trajectory (MoveJ) commands.
    # This also arms the recv timeout that must survive the MoveJ -> servo-loop
    # HANDOFF: after the final trajectory writeIdle, the controller waits this
    # long for the servo loop's first command (thread start + FIFO-set + first
    # writeServoj/writeIdle). Must be >= the worst-case handoff latency, so keep
    # it >= command_timeout_ms (was 200ms, which under FIFO contention is shorter
    # than the handoff -> controller times out -> "socket timed out waiting for
    # command on reverse_socket" -> RST -> writeServoj fails N ticks).
    move_j_timeout_ms: int = 800

    # ── Shared servoj trace ──
    trace_servoj: bool = True
    trace_translation_threshold: float = 0.05
    trace_rotation_threshold: float = 0.5
    trace_joint_threshold: float = 0.3

    # ── Grippers ── One `gripper:` block in the recipe describes BOTH arms; the two
    # are always a matched pair, and each backend resolves its own side at connect
    # (serial: board-SN parity, odd → left; taccap: the firmware-burned SN). The
    # block is cloned per side in __post_init__ with `side` stamped, so nothing in
    # the recipe has to name a side.
    #
    # The block is typed (see lerobot.grippers.GripperConfig), so `type: serial`
    # with a taccap knob in it — or a typo — is rejected at parse time instead of
    # being silently ignored, which is what the old flat gripper_type + taccap_* /
    # gripper_* fields did.
    #
    # None = no gripper on either arm. Set {side}_use_gripper=False to drop just one.
    # Tactile sensors are separate XenseTactileCamera devices (see the block's own
    # `enable_tactile`), never the gripper object itself.
    gripper: GripperConfig | None = None
    left_use_gripper: bool = True
    right_use_gripper: bool = True

    # ── Robot payload (tool + gripper) ── Declared to each controller via setPayload at
    # connect. The CS66 has NO F/T sensor: collision detection is model/current-based, so an
    # UNDECLARED payload makes the controller read the tool's own weight & inertia as external
    # force and protective-stop on light contact (e.g. wiping a board). None = leave the
    # controller's existing payload (previous behavior). mass in kg; cog in m relative to the
    # flange frame. Verified fleet value: 0.7 kg @ [0, 0, 0.19].
    payload_mass: float | None = None
    payload_cog: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.19])

    # Per-side clones of `gripper`, built in __post_init__. Do not set directly.
    left_gripper: GripperConfig | None = field(default=None, init=False)
    right_gripper: GripperConfig | None = field(default=None, init=False)
    # Set in __post_init__ from the gripper block's own auto_discover_cameras flag:
    # when on, the wrist/tactile cameras are sniffed by the robot at connect rather
    # than pinned in the recipe.
    _taccap_autodiscover: bool = field(default=False, init=False)
    _serial_autodiscover: bool = field(default=False, init=False)

    # Bimanual cameras (head + per-arm wrist + tactiles). Supplied by the recipe.
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()

        self._validate_shared_servo_params()

        for side in ("left", "right"):
            # [] means "ignore the matrix, use the angles" — collapse it to None so
            # the driver's `is not None` check stays a two-way test.
            if getattr(self, f"{side}_world_rotation") == []:
                setattr(self, f"{side}_world_rotation", None)

        # ── Per-arm pose validation ──
        for name, pose in (
            ("left_start_position_rad", self.left_start_position_rad),
            ("right_start_position_rad", self.right_start_position_rad),
            ("left_home_position_rad", self.left_home_position_rad),
            ("right_home_position_rad", self.right_home_position_rad),
        ):
            if len(pose) != 6:
                raise ValueError(f"{name} must have 6 elements (J1..J6), got {len(pose)}")

        # ── Per-side gripper configs ── clone the shared block, stamping the side.
        self.left_gripper = self._side_gripper("left", self.left_use_gripper)
        self.right_gripper = self._side_gripper("right", self.right_use_gripper)

        # ── Bimanual cameras ── With auto-discovery on, the wrist + tactile cameras
        # travel with the gripper, so the robot sniffs them at connect (see
        # _inject_taccap_cameras / _inject_serial_gripper_cameras) and the recipe
        # pins only the head. Otherwise the recipe pins every camera.
        gt = self.gripper.type if self.gripper is not None else None
        discover = self.gripper is not None and self.gripper.auto_discover_cameras
        self._taccap_autodiscover = discover and gt == "taccap_follower"
        self._serial_autodiscover = discover and gt == "serial"

    def _validate_shared_servo_params(self) -> None:
        if self.payload_mass is not None:
            if self.payload_mass < 0 or self.payload_mass > 10.0:
                raise ValueError(f"payload_mass must be in [0, 10] kg when set, got {self.payload_mass}")
            if len(self.payload_cog) != 3:
                raise ValueError(f"payload_cog must have 3 elements [x, y, z] (m), got {self.payload_cog}")
        if self.native_assembly is not None and self.native_model_path is not None:
            raise ValueError("native_model_path and native_assembly are mutually exclusive")
        if self.native_collision_profile is not None and self.native_assembly is None:
            raise ValueError("native_collision_profile needs native_assembly (the station the profile models)")
        for name in ("native_collision_gripper_max_age_s", "native_collision_gripper_publish_hz"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
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
        for name in ("left_tool_transform", "right_tool_transform"):
            tool = getattr(self, name)
            if tool is not None and (len(tool) != 16 or not all(math.isfinite(v) for v in tool)):
                raise ValueError(f"{name} must be a finite row-major 4x4 transform")
        if not 0.004 <= self.servoj_time <= 0.01:
            raise ValueError(
                f"servoj_time must be in [0.004, 0.01] s, got {self.servoj_time}: the native planner "
                "solves one IK step per servoj_time, and below 4 ms it overruns its period"
            )
        if not 0.03 <= self.servoj_lookahead_time <= 0.2:
            raise ValueError(
                "servoj_lookahead_time must be in [0.03, 0.2] (Elite SDK requirement), "
                f"got {self.servoj_lookahead_time}"
            )
        if not 100 <= self.servoj_gain <= 2000:
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
        # Each arm uses a 4-port block based at SDK defaults 50001..50004 plus its
        # offset. The two blocks must not overlap (|Δoffset| >= 4) and must stay in
        # the unprivileged TCP range.
        for name, off in (
            ("left_driver_port_offset", self.left_driver_port_offset),
            ("right_driver_port_offset", self.right_driver_port_offset),
        ):
            if not (50001 + off >= 0 and 50004 + off <= 65535):
                raise ValueError(
                    f"{name}={off} pushes EliteDriver ports outside the valid TCP range "
                    "(50001..50004 + offset must stay within 1024..65535)"
                )
        if abs(self.left_driver_port_offset - self.right_driver_port_offset) < 4:
            raise ValueError(
                "left_driver_port_offset and right_driver_port_offset must differ by >= 4 so the "
                "two arms' EliteDriver port blocks (reverse/sender/trajectory/script_command) do "
                f"not overlap; got left={self.left_driver_port_offset}, "
                f"right={self.right_driver_port_offset}"
            )

    def _side_gripper(self, side: str, use_gripper: bool) -> GripperConfig | None:
        """The shared ``gripper`` block with ``side`` stamped, or None for no gripper.

        Both arms always run the same backend with the same tuning — they are a
        matched pair — so the recipe writes one block and each side gets a copy.
        ``side`` is what lets each backend resolve which physical unit is which
        (serial: board-SN parity; taccap: the firmware-burned SN).
        """
        if not use_gripper or self.gripper is None:
            return None
        return replace(self.gripper, side=side)

    @property
    def _autodiscover_cameras(self) -> bool:
        """True when the driver, not the recipe, supplies wrist + tactile cameras.

        Either gripper backend can do it; they differ only in what anchors a side
        (taccap: firmware SN; serial: board-SN parity). Everything downstream is
        the same, so callers branch on this rather than on which backend is in use.
        """
        return self._taccap_autodiscover or self._serial_autodiscover
