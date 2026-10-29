#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Bimanual Elite CS66 robot integration for LeRobot.

Two Elite CS66 controllers, each driven exactly like the single-arm
``EliteCS66RT`` (RTSI state stream + EliteDriver reverse-socket servoj, an
optional background Cartesian servo loop, rotvec-continuity handling and
min-jerk reset). Per-arm state is kept in ``{"left": ..., "right": ...}`` dicts
so the single-arm logic is reused per side rather than duplicated line-by-line.

Action / observation keys are ``left_``/``right_`` prefixed:
    left_tcp.x/y/z + left_tcp.r1..r6   (+ optional left_joint_*),  left_gripper.pos
    right_tcp.x/y/z + right_tcp.r1..r6  (+ optional right_joint_*), right_gripper.pos
Grippers are per-arm serial (USB) devices addressed by board SN (no IP/MAC).
Cameras (head + per-arm wrist + optional tactiles) live at the bimanual level;
tactile images come from separate XenseTactileCamera devices (already namespaced
left_tactile_* / right_tactile_*), not the gripper.
"""

import contextlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.grippers import Gripper, make_gripper_from_config
from lerobot.grippers.camera_injection import (
    adopt_taccap_mcu_device,
    attach_wrist_fisheye_calibration,
    inject_serial_gripper_cameras,
    inject_taccap_cameras,
)
from lerobot.robots.bi_elite_cs66_rt.config_bi_elite_cs66_rt import (
    BiEliteCS66RTConfig,
    BiEliteCS66RTControlMode,
)
from lerobot.robots.bi_elite_cs66_rt.gripper_snapshots import GripperSnapshot, TaccapGripperSnapshotReader
from lerobot.robots.elite_cs66_rt import elite_cs66_rt as _elite_mod
from lerobot.robots.elite_cs66_rt.elite_cs66_rt import (
    _configure_native_driver,
    _controller_tcp_to_pose6,
    _derive_controller_tool,
    _import_elite_sdk,
    _matrix_to_pose6,
    _pose6_to_matrix,
    _quaternion_to_rotvec,
    _reach_exceeded,
    _rotvec_to_quaternion,
    _validate_configuration_model,
)
from lerobot.robots.elite_cs66_rt.joint_reset import JointReset, joint_sample
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.utils.robot_utils import (
    best_effort,
    get_logger,
    quaternion_to_euler,
    quaternion_to_rotation_6d,
    rotation_6d_to_quaternion,
)
from lerobot.utils.rotation import Rotation

# Bare (unprefixed) per-arm schema; shared with the single-arm driver.
TCP_POSITION_KEYS = ("tcp.x", "tcp.y", "tcp.z")
TCP_ROTATION_6D_KEYS = ("tcp.r1", "tcp.r2", "tcp.r3", "tcp.r4", "tcp.r5", "tcp.r6")
JOINT_POSITION_KEYS = tuple(f"joint_{i}.pos" for i in range(1, 7))
JOINT_VELOCITY_KEYS = tuple(f"joint_{i}.vel" for i in range(1, 7))
JOINT_EFFORT_KEYS = tuple(f"joint_{i}.effort" for i in range(1, 7))

_SIDES = ("left", "right")

# Single-arm RTSI/recipe resource directory, reused as the on-disk fallback when
# the SDK package doesn't ship the recipes. No need to duplicate recipe files.
_ELITE_RESOURCE_DIR = Path(_elite_mod.__file__).resolve().parent / "resource"


class BiEliteCS66RT(Robot):
    """Two Elite CS66 arms using libpyelite external control.

    Cartesian mode: action/observation features are ``{side}_tcp.x/y/z`` plus
    ``{side}_tcp.r1..r6``. Joint mode: ``{side}_joint_1.pos .. {side}_joint_6.pos``
    streamed with ``writeServoj(..., cartesian=False)``.
    """

    config_class = BiEliteCS66RTConfig
    name = "bi_elite_cs66_rt"

    # Same required RTSI output fields as the single-arm driver (SDK helpers
    # silently return zero vectors when these are absent).
    _REQUIRED_RTSI_OUTPUT_FIELDS = _elite_mod.EliteCS66RT._REQUIRED_RTSI_OUTPUT_FIELDS

    def __init__(self, config: BiEliteCS66RTConfig):
        super().__init__(config)
        self.config = config
        logger_suffix = config.id if config.id is not None else hex(id(self))
        self.logger = get_logger(f"BiEliteCS66RT.{logger_suffix}")

        self._cs = None  # shared libpyelite module (imported in connect())
        self._is_connected = False

        # Per-arm SDK handles + servo state.
        self._driver: dict[str, Any] = dict.fromkeys(_SIDES)
        # Station collision layer (native_collision_profile): the libpyelite
        # StationLayer and the StationCoordinator planning both arms through it.
        self._station: Any = None
        self._coordinator: Any = None
        self._station_gripper_stop = threading.Event()
        self._station_gripper_thread: threading.Thread | None = None
        self._station_gripper_error: str | None = None
        self._station_gripper_samples: tuple[float, dict[str, GripperSnapshot]] | None = None
        self._station_gripper_readers: dict[str, TaccapGripperSnapshotReader] = {}
        self._native_tools: dict[str, list[float] | None] = dict.fromkeys(_SIDES)
        self._dashboard: dict[str, Any] = dict.fromkeys(_SIDES)
        self._rtsi: dict[str, Any] = dict.fromkeys(_SIDES)

        self._gripper: dict[str, Gripper | None] = {
            "left": make_gripper_from_config(config.left_gripper),
            "right": make_gripper_from_config(config.right_gripper),
        }

        self._last_tcp_command: dict[str, np.ndarray | None] = dict.fromkeys(_SIDES)
        self._servo_lock: dict[str, threading.Lock] = {s: threading.Lock() for s in _SIDES}
        # Monotonic time each arm's _last_tcp_command was submitted.
        self._last_submit_time: dict[str, float] = dict.fromkeys(_SIDES, 0.0)
        self._q_start: dict[str, np.ndarray | None] = dict.fromkeys(_SIDES)
        self._joint_reset = JointReset(config)
        self._reach_warn_time: dict[str, float] = dict.fromkeys(_SIDES, 0.0)  # workspace-guard warn throttle

        # Prefixed key tuples (built once).
        self._tcp_pos_keys = {s: tuple(f"{s}_{k}" for k in TCP_POSITION_KEYS) for s in _SIDES}
        self._tcp_rot_keys = {s: tuple(f"{s}_{k}" for k in TCP_ROTATION_6D_KEYS) for s in _SIDES}
        self._joint_pos_keys = {s: tuple(f"{s}_{k}" for k in JOINT_POSITION_KEYS) for s in _SIDES}
        self._joint_vel_keys = {s: tuple(f"{s}_{k}" for k in JOINT_VELOCITY_KEYS) for s in _SIDES}
        self._joint_effort_keys = {s: tuple(f"{s}_{k}" for k in JOINT_EFFORT_KEYS) for s in _SIDES}
        self._gripper_key = {s: f"{s}_gripper.pos" for s in _SIDES}

        # Per-arm world←base rotation R = Rz(γ)·Rz(β)·Rx(α): tilt α about base-X
        # and zrot β about Z come from the teach pendant (fix the gravity vector
        # only); world_yaw γ aligns each arm's heading into ONE shared gravity-
        # aligned world frame (x=facing, y=left, z=up). Used at the get_observation
        # / send_action boundaries; all internal servo state stays in base frame.
        self._R_wb: dict[str, np.ndarray] = {
            side: np.eye(3) if config.native_assembly is not None else self._resolve_world_rotation(config, side)
            for side in _SIDES
        }
        self._p_wb = {side: np.zeros(3, dtype=np.float64) for side in _SIDES}
        self._assembly = None
        if config.native_assembly is not None:
            from libpyelite.assembly import Assembly

            self._assembly = Assembly(config.native_assembly)
            for side in _SIDES:
                transform = self._assembly.world_from_base(side)
                self._R_wb[side] = transform[:3, :3]
                self._p_wb[side] = transform[:3, 3]
                self.logger.info(
                    f"{side} assembly={config.native_assembly}: "
                    f"world<-base xyz(m)={self._p_wb[side].tolist()}, "
                    f"control_model={self._assembly.control_model_path(side)}"
                )

        # In taccap_follower + auto-discover mode the wrist + GSPS tactile cameras belong
        # to the gripper hardware, so sniff them now and add to config.cameras before the
        # camera drivers are built (config._build_cameras wired only the head).
        if getattr(config, "_taccap_autodiscover", False):
            self._inject_taccap_cameras()
        elif getattr(config, "_serial_autodiscover", False):
            self._inject_serial_gripper_cameras()

        self.cameras = make_cameras_from_configs(config.cameras)

    def _inject_taccap_cameras(self) -> None:
        """Auto-discover per-side TacCap wrist + GSPS tactile devices into
        ``config.cameras``. Called only in taccap_follower auto-discover mode."""
        mcu_devices = inject_taccap_cameras(
            self.config.cameras,
            sides=_SIDES,
            enable_tactile=self.config.gripper.enable_tactile,
            logger=self.logger,
            undistort_wrist=self.config.gripper.undistort_wrist,
            fisheye_balance=self.config.gripper.fisheye_balance,
        )
        # The sweep already resolved each gripper's MCU path; pin it so the
        # driver's connect() skips a second scan of the same bus.
        for side, mcu_device in mcu_devices.items():
            adopt_taccap_mcu_device(self._gripper.get(side), side, mcu_device, self.logger)

    def _inject_serial_gripper_cameras(self) -> None:
        """Same as _inject_taccap_cameras for serial (parallel-jaw) grippers.

        Only asks about sides that actually have a gripper, so a bench running
        one arm without one does not fail discovery for the other.
        """
        inject_serial_gripper_cameras(
            self.config.cameras,
            sides=tuple(
                side
                for side, enabled in (
                    ("left", self.config.left_use_gripper),
                    ("right", self.config.right_use_gripper),
                )
                if enabled
            ),
            enable_tactile=self.config.gripper.enable_tactile,
            logger=self.logger,
        )

    @staticmethod
    def _resolve_world_rotation(config: BiEliteCS66RTConfig, side: str) -> np.ndarray:
        """world<-base rotation for one arm.

        Uses the explicit ``{side}_world_rotation`` matrix from config when set
        (re-orthonormalized defensively), else builds it from the tilt/zrot/yaw
        angles. The explicit path is needed when the mounting isn't a clean
        Rz·Rx (e.g. the left arm tilts about base-Y, not base-X).
        """
        override = getattr(config, f"{side}_world_rotation")
        if override is not None:
            R = np.asarray(override, dtype=np.float64)
            if R.shape != (3, 3):
                raise ValueError(f"{side}_world_rotation must be 3x3, got {R.shape}")
            U, _, Vt = np.linalg.svd(R)
            R = U @ Vt
            if np.linalg.det(R) < 0:
                R = U @ np.diag([1.0, 1.0, -1.0]) @ Vt
            return R
        return BiEliteCS66RT._mount_rotation(
            getattr(config, f"{side}_mount_tilt_deg"),
            getattr(config, f"{side}_mount_zrot_deg"),
            getattr(config, f"{side}_mount_world_yaw_deg"),
        )

    @staticmethod
    def _mount_rotation(tilt_deg: float, zrot_deg: float, world_yaw_deg: float) -> np.ndarray:
        """world←base rotation matrix for one arm: R = Rz(world_yaw)·Rz(zrot)·Rx(tilt).

        Built from axis-angle rotvecs (this repo's ``Rotation`` has no
        ``from_euler``): Rx(tilt) about base-X, Rz(zrot) about Z (teach-pendant
        mounting), then Rz(world_yaw) about world-Z to align headings across arms.
        """
        rx = Rotation.from_rotvec([np.deg2rad(tilt_deg), 0.0, 0.0]).as_matrix()
        rz = Rotation.from_rotvec([0.0, 0.0, np.deg2rad(zrot_deg)]).as_matrix()
        ryaw = Rotation.from_rotvec([0.0, 0.0, np.deg2rad(world_yaw_deg)]).as_matrix()
        return ryaw @ rz @ rx

    def _base_pose6_to_world(self, side: str, pose6: np.ndarray) -> np.ndarray:
        """Lift a base-frame ``[x,y,z,rx,ry,rz]`` (rotvec) pose into world frame."""
        R_wb = self._R_wb[side]
        pose6 = np.asarray(pose6, dtype=np.float64)
        pos = R_wb @ pose6[:3] + self._p_wb[side]
        rot = R_wb @ Rotation.from_rotvec(pose6[3:6]).as_matrix()
        rotvec = Rotation.from_matrix(rot).as_rotvec()
        return np.concatenate([pos, rotvec])

    def _world_pose6_to_base(self, side: str, pose6: np.ndarray) -> np.ndarray:
        """Map a world-frame ``[x,y,z,rx,ry,rz]`` (rotvec) pose back to base frame."""
        R_bw = self._R_wb[side].T
        pose6 = np.asarray(pose6, dtype=np.float64)
        pos = R_bw @ (pose6[:3] - self._p_wb[side])
        rot = R_bw @ Rotation.from_rotvec(pose6[3:6]).as_matrix()
        rotvec = Rotation.from_matrix(rot).as_rotvec()
        return np.concatenate([pos, rotvec])

    # =========================================================================
    # Per-arm config accessors
    # =========================================================================

    def _arm_ip(self, side: str) -> str:
        return getattr(self.config, f"{side}_robot_ip")

    def _arm_local_ip(self, side: str) -> str:
        return getattr(self.config, f"{side}_local_ip")

    def _arm_start_pose(self, side: str) -> list[float]:
        return list(getattr(self.config, f"{side}_start_position_rad"))

    def _arm_home_pose(self, side: str) -> list[float]:
        return list(getattr(self.config, f"{side}_home_position_rad"))

    # =========================================================================
    # Feature descriptors
    # =========================================================================

    @cached_property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        features: dict[str, type | tuple[int, int, int]] = {}

        for side in _SIDES:
            if self.config.observe_tcp:
                features.update(dict.fromkeys(self._tcp_pos_keys[side] + self._tcp_rot_keys[side], float))
            if self.config.observe_joints:
                features.update(dict.fromkeys(self._joint_pos_keys[side], float))
                features.update(dict.fromkeys(self._joint_vel_keys[side], float))
                features.update(dict.fromkeys(self._joint_effort_keys[side], float))

            if self._gripper[side] is not None:
                features[self._gripper_key[side]] = float

        # Tactile sensors are XenseTactileCamera entries in self.cameras, so they
        # are covered by the camera loop below (same as head / wrist cams).
        for cam_name in self.cameras:
            features[cam_name] = (
                self.config.cameras[cam_name].height,
                self.config.cameras[cam_name].width,
                3,
            )
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        features: dict[str, type] = {}
        for side in _SIDES:
            if self.config.control_mode == BiEliteCS66RTControlMode.JOINT_SERVO:
                features.update(dict.fromkeys(self._joint_pos_keys[side], float))
            else:
                features.update(dict.fromkeys(self._tcp_pos_keys[side] + self._tcp_rot_keys[side], float))
            if self._gripper[side] is not None:
                features[self._gripper_key[side]] = float
        return features

    # =========================================================================
    # Connection state
    # =========================================================================

    @property
    def is_connected(self) -> bool:
        return (
            self._is_connected
            and all(self._driver[s] is not None for s in _SIDES)
            and all(self._rtsi[s] is not None for s in _SIDES)
            and all(cam.is_connected for cam in self.cameras.values())
        )

    @property
    def is_calibrated(self) -> bool:
        # Elite CS66 is factory calibrated; no runtime calibration step.
        return True

    def calibrate(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

    def configure(self) -> None:
        pass

    # =========================================================================
    # Recipe / driver config helpers (shared across both arms)
    # =========================================================================

    def _resolve_sdk_resource(self, filename: str) -> str:
        assert self._cs is not None
        module_file = getattr(self._cs, "__file__", None)
        if not module_file:
            raise RuntimeError("Cannot resolve libpyelite package path.")
        path = Path(module_file).resolve().parent / filename
        if not path.exists():
            raise FileNotFoundError(f"Elite SDK resource not found: {path}")
        return str(path)

    @staticmethod
    def _read_recipe_fields(path: str) -> list[str]:
        lines = Path(path).read_text().splitlines()
        return [s for s in (line.strip() for line in lines) if s and not s.startswith("#")]

    _validate_output_recipe = _elite_mod.EliteCS66RT._validate_output_recipe

    def _resolve_recipe(self, configured: str | Path | None, filename: str) -> str:
        if configured is not None:
            path = Path(configured).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"Configured RTSI recipe not found: {path}")
            return str(path)

        from contextlib import suppress

        sdk_path = None
        with suppress(FileNotFoundError):
            sdk_path = self._resolve_sdk_resource(filename)
        if sdk_path:
            return sdk_path

        module_recipe = _ELITE_RESOURCE_DIR / filename
        if module_recipe.exists():
            return str(module_recipe)
        raise FileNotFoundError(
            f"Could not find {filename}. Set rtsi_output_recipe/rtsi_input_recipe in BiEliteCS66RTConfig."
        )

    def _make_driver_config(self, side: str):
        assert self._cs is not None
        cfg = self._cs.EliteDriverConfig()
        cfg.robot_ip = self._arm_ip(side)
        cfg.local_ip = self._arm_local_ip(side)
        cfg.servoj_time = self.config.servoj_time
        cfg.servoj_lookahead_time = self.config.servoj_lookahead_time
        cfg.servoj_gain = self.config.servoj_gain
        cfg.headless_mode = True
        # Two EliteDriver instances on one host can't share the local reverse /
        # trajectory / script-command TCP server ports — the 2nd arm would hit
        # "Address already in use". Offset one arm's ports; the SDK substitutes
        # these into the pushed external_control.script (REVERSE/TRAJECTORY/
        # SCRIPT_COMMAND port placeholders) so the controller connects back to
        # the matching ports.
        offset = getattr(self.config, f"{side}_driver_port_offset")
        cfg.reverse_port += offset
        cfg.script_sender_port += offset
        cfg.trajectory_port += offset
        cfg.script_command_port += offset
        if self.config.script_file_path is not None:
            cfg.script_file_path = str(Path(self.config.script_file_path).expanduser())
        else:
            cfg.script_file_path = self._resolve_sdk_resource("external_control.script")
        tool = getattr(self.config, f"{side}_tool_transform")
        _configure_native_driver(
            cfg,
            self.config,
            tool,
            side,
            None if self._assembly is None else self._assembly.world_from_base(side),
        )
        if self._assembly is not None:
            cfg.model_path = self._assembly.control_model_path(side)
        _validate_configuration_model(self._cs, cfg)
        if tool is None:
            assert self._rtsi[side] is not None
            cfg.tool = _derive_controller_tool(self._cs, self._rtsi[side], cfg.model_path or self._cs.model_path())
            offset_mm = np.asarray(cfg.tool, dtype=np.float64).reshape(4, 4)[:3, 3] * 1000
            self.logger.info(
                f"{side}_tool_transform unset; using the controller TCP, offset (mm) "
                f"{np.array2string(offset_mm, precision=2)}. Pin it in the recipe to make it explicit."
            )
        self.logger.info(
            f"{side} native planner TCP caps: {cfg.planner.max_linear_velocity:g} m/s, "
            f"{cfg.planner.max_angular_velocity:g} rad/s (min of max_*_speed and native_max_tcp_*)"
        )
        self._native_tools[side] = list(cfg.tool)
        return cfg

    # =========================================================================
    # Connect / disconnect
    # =========================================================================

    def connect(self, calibrate: bool = False, go_to_start: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected, do not run connect() twice.")

        if self.config.native_collision_profile is not None:
            from libpyelite.collision import StationLayer

            # Validate geometry/backend/calibration before powerOn/brakeRelease
            # or gripper enable. Actual TCP transforms are obtained later.
            preflight = StationLayer(self._assembly, self.config.native_collision_profile, [np.eye(4), np.eye(4)])
            self._configure_station_grippers(preflight)
        self._joint_reset.prepare_connection()
        self._q_start = dict.fromkeys(_SIDES)
        self._cs = _import_elite_sdk()
        # Native workers own the high-rate path; preserve the Python station logic.

        try:
            # --- Bring up both controllers (+ their grippers) in parallel ---
            self.logger.info(
                f"Connecting both arms in parallel: left={self._arm_ip('left')}, right={self._arm_ip('right')}"
            )
            with ThreadPoolExecutor(max_workers=2) as ex:
                futs = {side: ex.submit(self._connect_arm, side) for side in _SIDES}
                for side in _SIDES:
                    futs[side].result()
            # --- Connect bimanual cameras in parallel ---
            # Read the wrist fisheye intrinsics off each gripper's MCU while it is
            # open, before the cameras build their remap tables.
            attach_wrist_fisheye_calibration(self.cameras, self._gripper, self.logger)

            if self.cameras:
                self.logger.info(f"Connecting {len(self.cameras)} camera(s): {', '.join(self.cameras.keys())}...")
                with ThreadPoolExecutor(max_workers=len(self.cameras)) as ex:
                    cam_futs = [ex.submit(cam.connect) for cam in self.cameras.values()]
                    for f in cam_futs:
                        f.result()
            # Start freshness-sensitive publication after device initialization,
            # still before any MoveJ or servo command. Camera/MCU setup may block
            # Python threads longer than the gripper freshness budget.
            self._start_station_layer()
            self._check_station_health()
        except BaseException:
            with best_effort(self.logger, "logging connection fault diagnostics"):
                self._log_native_fault_diagnostics()
            self._cleanup_after_failed_connect()
            raise

        self._is_connected = True

        # --- Bring each arm to its start_position AND immediately hand off to
        #     its servo loop, per-arm, in parallel ---
        # The reverse socket the controller opened in _connect_arm has an
        # effectively-infinite recv timeout UNTIL the first command; the first
        # MoveJ command arms the move_j_timeout_ms recv budget. From then on any
        # feeding gap > that budget drops the connection. The dangerous gap is
        # the MoveJ -> servo-loop handoff. Doing MoveJ + seed + servo-loop start
        # inside ONE per-arm worker keeps the faster-finishing arm from sitting
        # idle while the slower arm finishes (the previous "wait for both, then
        # seed/start sequentially" structure starved whichever arm finished
        # first -> intermittent "socket timed out ... reverse_socket" RST).
        def _bring_arm_online(side: str, paired_failed: threading.Event) -> None:
            if go_to_start:
                self._move_j_blocking(
                    side,
                    self._arm_start_pose(side),
                    self.config.start_move_duration_s,
                    path_checked=True,
                    paired_failed=paired_failed,
                )
            if paired_failed.is_set():
                raise RuntimeError(f"{side} arm: servo start skipped, the paired arm failed to come online")
            if self.config.control_mode == BiEliteCS66RTControlMode.CARTESIAN_SERVO:
                current_tcp = _controller_tcp_to_pose6(self._rtsi[side].getActualTCPPose())
                self._last_tcp_command[side] = current_tcp.copy()
            self._start_servo_loop(side)

        try:
            if go_to_start:
                self._check_joint_move({side: self._arm_start_pose(side) for side in _SIDES}, "start move")
                self.logger.info(
                    f"Bi Elite CS66 moving both arms to start_position over {self.config.start_move_duration_s:.1f}s..."
                )
            self._run_arms_paired(_bring_arm_online, "start move")
            self._check_station_health()
            for side in _SIDES:
                self._driver[side].check_error()
            # Publish the pair only after both startups succeed. Each side
            # keeps its actual unwrapped configuration, including reattachment.
            self._q_start = {side: joint_sample(self._rtsi[side])[1] for side in _SIDES}
        except BaseException:
            with best_effort(self.logger, "logging connection fault diagnostics"):
                self._log_native_fault_diagnostics()
            self._is_connected = False
            self._cleanup_after_failed_connect()
            raise

        self.logger.info("BiEliteCS66RT connected and ready.")

    def _connect_arm(self, side: str) -> None:
        """Bring up one Elite controller: RTSI + dashboard + EliteDriver handshake (+ gripper)."""
        ip = self._arm_ip(side)

        output_recipe = self._resolve_recipe(self.config.rtsi_output_recipe, "output_recipe.txt")
        self._validate_output_recipe(output_recipe)
        input_recipe = self._resolve_recipe(self.config.rtsi_input_recipe, "input_recipe.txt")

        # Store each handle on self BEFORE its connect() check so a failed
        # connect (e.g. RTSI "IN_USE" when another client still holds the input
        # registers) leaves it for _cleanup_after_failed_connect() to
        # disconnect(). Assigning only on success would orphan the C++ object,
        # whose destructor then fires at interpreter shutdown ("terminate called
        # without an active exception" -> Aborted (core dumped)).
        rtsi = self._cs.RtsiIOInterface(output_recipe, input_recipe, self.config.rtsi_frequency)
        self._rtsi[side] = rtsi
        if not rtsi.connect(ip):
            raise ConnectionError(f"Failed to connect Elite RTSI server ({side}) at {ip}:30004")

        dashboard = self._cs.DashboardClientInterface()
        self._dashboard[side] = dashboard
        if not dashboard.connect(ip):
            raise ConnectionError(f"Failed to connect Elite dashboard ({side}) at {ip}")

        if not dashboard.powerOn():
            raise RuntimeError(f"Elite CS66 ({side}) powerOn() failed.")
        if not dashboard.brakeRelease():
            raise RuntimeError(f"Elite CS66 ({side}) brakeRelease() failed.")

        driver_config = self._make_driver_config(side)
        driver_construct_time = time.monotonic()
        driver = self._cs.EliteDriver(driver_config)
        self._driver[side] = driver

        if self.config.external_control_settle_s > 0:
            time.sleep(self.config.external_control_settle_s)

        if not driver.isRobotConnected() and not driver.sendExternalControlScript():
            raise RuntimeError(f"Failed to send Elite external control script ({side}).")

        deadline = time.monotonic() + self.config.connect_timeout_s
        while not driver.isRobotConnected():
            if time.monotonic() > deadline:
                raise TimeoutError(f"Timed out waiting for Elite external control script connection ({side}).")
            time.sleep(0.01)

        remaining = self.config.external_control_settle_s - (time.monotonic() - driver_construct_time)
        if remaining > 0:
            time.sleep(remaining)

        # Declare the tool+gripper payload so the controller's (F/T-less, model-based)
        # collision detection doesn't read the tool's own weight/inertia as external force
        # and protective-stop on light contact. Done before go_to_start / any servoj.
        if self.config.payload_mass is not None:
            ok = driver.setPayload(self.config.payload_mass, list(self.config.payload_cog))
            self.logger.info(
                f"{side} arm: setPayload(mass={self.config.payload_mass} kg, "
                f"cog={self.config.payload_cog}) -> {'ok' if ok else 'FAILED'}"
            )

        gripper = self._gripper[side]
        if gripper is not None:
            self.logger.info(f"{side} arm: connecting gripper ({type(gripper).__name__})...")
            gripper.connect()

    def _cleanup_after_failed_connect(self) -> None:
        self._joint_reset.close()
        self._q_start = dict.fromkeys(_SIDES)
        self._clear_reset_targets()
        for side in _SIDES:
            # A servo loop may already be running if the other arm's bring-up
            # failed after this one handed off; stop it before tearing down.
            with contextlib.suppress(Exception):
                self._stop_servo_loop(side)
        with best_effort(self.logger, "stopping the station collision layer"):
            self._stop_station_layer()
        for side in _SIDES:
            with best_effort(self.logger, f"stopping control ({side})"):
                if self._driver[side] is not None:
                    self._driver[side].stopControl(1000)
            with best_effort(self.logger, f"closing the dashboard connection ({side})"):
                if self._dashboard[side] is not None:
                    self._dashboard[side].disconnect()
            with best_effort(self.logger, f"closing the RTSI connection ({side})"):
                if self._rtsi[side] is not None:
                    self._rtsi[side].disconnect()
            gripper = self._gripper[side]
            if gripper is not None:
                with best_effort(self.logger, f"releasing the gripper ({side})"):
                    if getattr(gripper, "_is_connected", False):
                        gripper.disconnect()
            self._driver[side] = None
            self._dashboard[side] = None
            self._rtsi[side] = None

        for name, cam in self.cameras.items():
            with best_effort(self.logger, f"releasing camera {name}"):
                if cam.is_connected:
                    cam.disconnect()
        self._is_connected = False

    def disconnect(self) -> None:
        # Idempotent: quiet no-op if nothing was ever brought up.
        any_handle = any(
            self._driver[s] is not None or self._rtsi[s] is not None or self._dashboard[s] is not None for s in _SIDES
        )
        if not self._is_connected and not any_handle:
            self.logger.warn(f"{self} is not connected, skipping disconnect.")
            return

        self._joint_reset.close()
        self._q_start = dict.fromkeys(_SIDES)
        self._clear_reset_targets()
        for cam in self.cameras.values():
            if cam.is_connected:
                cam.disconnect()

        for side in _SIDES:
            self._stop_servo_loop(side)

        # Smooth return to home for both arms in parallel before teardown,
        # unless the collision layer finds the joint path to home blocked.
        go_home = True
        home_error: Exception | None = None
        if self._station is not None:
            try:
                self._check_joint_move({side: self._arm_home_pose(side) for side in _SIDES}, "return-to-home")
            except Exception as exc:
                go_home = False
                self.logger.error(f"Skipping return-to-home, arms left where they are: {exc}")
        if go_home:
            try:
                self._run_arms_paired(self._return_home_arm, "return-to-home")
            except Exception as exc:
                # Teardown still runs; the failure is raised once it is done.
                home_error = exc
        with best_effort(self.logger, "stopping the station collision layer"):
            self._stop_station_layer()

        for side in _SIDES:
            driver = self._driver[side]
            if driver is not None:
                try:
                    driver.writeIdle(self.config.command_timeout_ms)
                    driver.stopControl(1000)
                finally:
                    self._driver[side] = None

            dashboard = self._dashboard[side]
            if dashboard is not None:
                try:
                    dashboard.disconnect()
                finally:
                    self._dashboard[side] = None

            rtsi = self._rtsi[side]
            if rtsi is not None:
                try:
                    rtsi.disconnect()
                finally:
                    self._rtsi[side] = None

            gripper = self._gripper[side]
            if gripper is not None:
                try:
                    if getattr(gripper, "_is_connected", False):
                        gripper.disconnect()
                except Exception as exc:
                    self.logger.warn(f"{side} gripper disconnect failed: {exc}")

        self._is_connected = False
        if home_error is not None:
            raise RuntimeError(
                f"Return-to-home failed; arms were left where they stopped: {home_error}"
            ) from home_error

    def _return_home_arm(self, side: str, paired_failed: threading.Event | None = None) -> None:
        """Blocking MoveJ to home for one arm, then re-kill its servo loop."""
        if self._driver[side] is None or self._rtsi[side] is None:
            return
        try:
            self.logger.info(f"{side} arm: returning to home_position over {self.config.home_move_duration_s:.1f}s...")
            self._move_j_blocking(
                side,
                self._arm_home_pose(side),
                self.config.home_move_duration_s,
                path_checked=True,
                paired_failed=paired_failed,
            )
        finally:
            # _move_j_blocking may have restarted the servo loop in its finally block.
            self._stop_servo_loop(side)

    def _run_arms_paired(self, work, what: str) -> None:
        """Run ``work(side, paired_failed)`` for both arms in parallel.

        A both-arm station path check assumes the arms progress together, so once
        one arm fails the other must not go on into a combination nobody checked:
        ``paired_failed`` is set and the other arm skips any MoveJ or servo start
        it has not begun yet. A native MoveJ already in flight cannot be
        interrupted from here (moveJ holds the driver's lifecycle lock and the
        dashboard binding exposes no stop), so it finishes or is aborted by the
        controller. Every per-arm error is logged; the first one is raised.
        """
        paired_failed = threading.Event()

        def _run(side: str) -> None:
            try:
                work(side, paired_failed)
            except BaseException:
                paired_failed.set()
                raise

        with ThreadPoolExecutor(max_workers=2) as ex:
            futs = {side: ex.submit(_run, side) for side in _SIDES}
        errors = {side: futs[side].exception() for side in _SIDES if futs[side].exception() is not None}
        if not errors:
            return
        for side, exc in errors.items():
            self.logger.error(f"{side} arm: {what} failed: {exc}")
        raise next(iter(errors.values()))

    # =========================================================================
    # Servo loop (per arm)
    # =========================================================================

    def _start_servo_loop(self, side: str) -> None:
        assert self._driver[side] is not None and self._rtsi[side] is not None
        # startServo clears the driver's latched error. It must never revive a
        # sender whose external planner has already stopped with a station fault.
        self._check_station_health()
        self._driver[side].startServo(self._rtsi[side])
        try:
            self._check_station_health()
        except Exception:
            with best_effort(self.logger, f"stopping {side} after station fault during servo startup"):
                self._stop_servo_loop(side)
            raise

    def _stop_servo_loop(self, side: str) -> None:
        if self._driver[side] is not None:
            self._driver[side].stopServo()

    def _check_station_health(self) -> None:
        if self._coordinator is None:
            return
        status = self._coordinator.status()
        if status.error:
            raise RuntimeError(f"Elite station collision layer failed: {status.error}")
        if not status.running:
            raise RuntimeError("Elite station collision coordinator is not running")

    def _raise_servo_error_if_any(self, side: str) -> None:
        try:
            # Preserve the primary station fault instead of a subsequent sender
            # deadline error caused by the stopped external planner.
            self._check_station_health()
            # A fault on either arm stops command production for the pair.
            for selected in _SIDES:
                if self._driver[selected] is not None:
                    self._driver[selected].check_error()
        except Exception:
            with best_effort(self.logger, "logging native fault diagnostics"):
                self._log_native_fault_diagnostics()
            for selected in _SIDES:
                with best_effort(self.logger, f"stopping paired arm after native fault ({selected})"):
                    if self._driver[selected] is not None:
                        self._driver[selected].stopServo()
            raise

    def _log_native_fault_diagnostics(self) -> None:
        """Per-arm sender timing and the collision coordinator's state at a fault."""
        for side in _SIDES:
            if self._driver[side] is None:
                continue
            st = self._driver[side].status()
            self.logger.error(
                f"[{side}] native fault: error={st.error!r} max_sender_gap={st.max_sender_gap_s * 1000:.1f} ms "
                f"max_socket_write={st.max_write_s * 1000:.1f} ms planner_overruns={st.planner_overruns}"
            )
        if self._coordinator is not None:
            st = self._coordinator.status()
            self.logger.error(
                f"[station] ticks={st.ticks} overruns={st.overruns} max_tick={st.max_tick_s * 1000:.2f} ms "
                f"error={st.error!r} last: held={list(st.last.held)} limited={list(st.last.limited)} "
                f"clearance={st.last.clearance.minimum_distance * 1000:.1f} mm"
            )
        if self._station_gripper_error is not None:
            self.logger.error(f"[station] gripper publisher error: {self._station_gripper_error}")
        if self._station_gripper_samples is not None:
            published_at, samples = self._station_gripper_samples
            now = time.monotonic()
            for side, sample in samples.items():
                # This is the latest publication, not necessarily the one that
                # faulted. The native station error describes the failing sample.
                self.logger.error(
                    f"[{side}] latest gripper snapshot: opening={sample.position!r} raw={sample.raw_position!r} "
                    f"valid={sample.valid} sequence={sample.sequence} "
                    f"sample_age={(now - sample.sampled_at) * 1000:.1f} ms "
                    f"publication_age={(now - published_at) * 1000:.1f} ms"
                )

    def _start_station_layer(self) -> None:
        """Bring up the station collision layer when the recipe asks for it.

        Both drivers switch to external planning and one StationCoordinator plans
        them together through the same StationGuard libpyelite.sim runs. Must run
        before any servo loop starts (nothing else feeds the senders then).
        """
        profile = self.config.native_collision_profile
        if profile is None:
            return
        from libpyelite.collision import StationLayer

        # The same tool0 <- TCP the drivers plan with (_make_driver_config).
        if any(self._native_tools[side] is None for side in _SIDES):
            raise RuntimeError("Both native driver TCP transforms must be configured before the station layer")
        tools = [self._native_tools[side] for side in _SIDES]
        self._station = StationLayer(self._assembly, profile, tools)
        self._configure_station_grippers(self._station)
        self._start_station_gripper_publisher()
        native = {side: self._driver[side]._native for side in _SIDES}
        for side in _SIDES:
            native[side].setExternalPlanning(True)
        self._coordinator = self._cs._native.StationCoordinator(
            self._station.guard, native["left"], native["right"], self._rtsi["left"], self._rtsi["right"]
        )
        self._coordinator.start()
        self.logger.info(f"Station collision layer active: {self._station.path}")

    def _configure_station_grippers(self, station) -> None:
        for side in _SIDES:
            gripper = self._gripper[side]
            if gripper is None:
                raise ValueError(
                    f"Collision station models movable {side} gripper fingers but no sensor backend is enabled"
                )
            if gripper.config.type != "taccap_follower":
                raise ValueError(
                    f"Elite collision checking supports TacCap followers only; {side} backend is "
                    f"{gripper.config.type!r}. XGripper collision checking is unsupported; "
                    "set native_collision_profile to null."
                )
        station.configure_grippers(
            self.config.native_collision_gripper_calibration,
            self.config.native_collision_gripper_max_age_s,
        )

    def _start_station_gripper_publisher(self) -> None:
        # Query calibration only for this connected Elite collision session,
        # before starting the nonblocking publisher or any arm motion.
        self._station_gripper_readers = {side: TaccapGripperSnapshotReader(self._gripper[side]) for side in _SIDES}
        # An independent cache reader owns publication, including while arms
        # hold and when no get_observation() calls are made. No USB/serial
        # transaction is permitted in the reader's read().
        # A fresh Event per start: a reader that failed to join on an earlier
        # stop keeps its own (set) Event and can never be revived by this one.
        self._station_gripper_stop = threading.Event()
        self._station_gripper_error = None
        self._station_gripper_samples = None
        self._station_gripper_thread = threading.Thread(
            target=self._station_gripper_loop,
            args=(self._station, self._station_gripper_stop, self._station_gripper_readers),
            name="station-gripper-state",
            daemon=True,
        )
        self._station_gripper_thread.start()
        deadline = time.monotonic() + 1.0
        while True:
            try:
                self._station.station_q([0.0] * 6, [0.0] * 6)
                break
            except RuntimeError as exc:
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"No fresh calibrated gripper state for collision startup: {exc}") from exc
                time.sleep(0.01)

    def _station_gripper_loop(self, station, stop: threading.Event, readers: dict) -> None:
        period = 1.0 / self.config.native_collision_gripper_publish_hz
        while not stop.is_set():
            try:
                snapshots = {side: readers[side].read() for side in _SIDES}
                self._station_gripper_samples = (time.monotonic(), snapshots)
                station.publish_gripper_snapshots(snapshots)
                self._station_gripper_error = None
            except Exception as exc:
                self._station_gripper_error = str(exc)
                # Explicit invalidation, never refresh the old valid sample.
                try:
                    station.publish_gripper_snapshots({})
                except Exception:
                    return  # native timestamp validation still expires the last publication
            stop.wait(period)

    def _stop_station_layer(self) -> None:
        thread = self._station_gripper_thread
        try:
            if self._coordinator is not None:
                self._coordinator.stop()
        finally:
            # Cleanup always completes; a coordinator.stop() error stays the one raised.
            self._station_gripper_stop.set()
            self._station_gripper_thread = None
            if thread is not None:
                thread.join(timeout=1.0)
            self._station_gripper_readers.clear()
            self._coordinator = None
            self._station = None
        if thread is not None and thread.is_alive():
            raise RuntimeError("Gripper snapshot reader did not stop; backend must provide a nonblocking snapshot")

    def _measured_joints(self) -> dict[str, np.ndarray]:
        return {side: np.asarray(self._rtsi[side].getActualJointPositions(), dtype=np.float64) for side in _SIDES}

    def _check_joint_move(self, targets: dict[str, list[float]], what: str) -> None:
        """Refuse a MoveJ whose straight joint path the collision layer finds blocked.

        ``targets`` maps the moving arm(s) to their goal; any other arm stays at
        its measured joints. Arms moved together share one duration, so their
        joint-interpolated paths progress together along this segment.
        """
        self._check_station_health()
        if self._station is None:
            return
        start = self._measured_joints()
        end = {side: np.asarray(targets.get(side, start[side]), dtype=np.float64) for side in _SIDES}
        result = self._station.check_joint_path(start, end)
        if not result.safe:
            raise RuntimeError(
                f"{what} refused by the station collision layer: the joint path reaches "
                f"{result.minimum_distance * 1000:.1f} mm between {result.pair} (clearance {result.clearance * 1000:.1f} mm)"
            )

    def _move_j_blocking(
        self,
        side: str,
        target_joints: list[float],
        duration_s: float,
        path_checked: bool = False,
        paired_failed: threading.Event | None = None,
    ) -> None:
        driver = self._driver[side]
        assert driver is not None
        self._check_station_health()  # also required for a previously checked joint path
        if not path_checked:
            self._check_joint_move({side: list(target_joints)}, f"{side} MoveJ")
        if paired_failed is not None and paired_failed.is_set():
            raise RuntimeError(f"{side} MoveJ skipped: the paired arm's move failed")
        was_running = driver.status().running
        self._stop_servo_loop(side)
        # The arm moves under MoveJ, so the last streamed target no longer applies.
        with self._servo_lock[side]:
            self._last_tcp_command[side] = None
        try:
            driver.moveJ(list(target_joints), float(duration_s), self.config.move_j_timeout_ms)
        except BaseException:
            # Restart even after a failed MoveJ, or every later send_action hits a stopped
            # servo; the MoveJ error stays the one raised. startServo re-seeds from measured q.
            if was_running:
                with best_effort(self.logger, f"restarting the {side} native servo after a failed MoveJ", level="warn"):
                    self._start_servo_loop(side)
            raise
        if was_running:
            self._start_servo_loop(side)

    def _tcp_rotvec_to_feature_values(self, side: str, tcp_pose: np.ndarray) -> dict[str, float]:
        pos_keys = self._tcp_pos_keys[side]
        rot_keys = self._tcp_rot_keys[side]
        values = {
            pos_keys[0]: float(tcp_pose[0]),
            pos_keys[1]: float(tcp_pose[1]),
            pos_keys[2]: float(tcp_pose[2]),
        }
        quat = _rotvec_to_quaternion(tcp_pose[3:6])
        r6d = quaternion_to_rotation_6d(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
        values.update({key: float(value) for key, value in zip(rot_keys, r6d, strict=True)})
        return values

    def get_observation(self) -> dict[str, Any]:
        """Read both arms, both grippers and every camera into one flat dict.

        Also stamps ``self._last_obs_timing`` with a per-source millisecond
        breakdown. The record and teleop loops surface that breakdown when a
        frame runs slow, so an operator can tell which sensor stalled instead of
        only learning that *something* did; the keys match the ones
        ``bi_flexiv_rizon4_rt`` publishes so both consumers read either rig.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs: dict[str, Any] = {}
        _t = time.perf_counter
        arm_ms: dict[str, float] = {}
        grip_ms: dict[str, float] = {}

        t_start = _t()

        for side in _SIDES:
            rtsi = self._rtsi[side]
            assert rtsi is not None

            t_arm0 = _t()
            if self.config.observe_tcp:
                # RTSI reports the TCP pose in the (tilted) base frame; lift it
                # into the gravity-aligned world frame before publishing.
                tcp_pose = _controller_tcp_to_pose6(rtsi.getActualTCPPose())
                tcp_world = self._base_pose6_to_world(side, tcp_pose)
                obs.update(self._tcp_rotvec_to_feature_values(side, tcp_world))
            if self.config.observe_joints:
                joints = rtsi.getActualJointPositions()
                obs.update({k: float(v) for k, v in zip(self._joint_pos_keys[side], joints, strict=True)})
                joint_vel = rtsi.getActualJointVelocity()
                obs.update({k: float(v) for k, v in zip(self._joint_vel_keys[side], joint_vel, strict=True)})
                joint_effort = rtsi.getActualJointTorques()
                obs.update({k: float(v) for k, v in zip(self._joint_effort_keys[side], joint_effort, strict=True)})
            t_arm1 = _t()
            arm_ms[side] = (t_arm1 - t_arm0) * 1e3

            gripper = self._gripper[side]
            if gripper is not None:
                obs[self._gripper_key[side]] = gripper.get_gripper_position()
            grip_ms[side] = (_t() - t_arm1) * 1e3

        # Tactile images come from XenseTactileCamera entries in self.cameras.
        t_cams = _t()
        cam_timings: dict[str, float] = {}
        for cam_name, cam in self.cameras.items():
            tc0 = _t()
            obs[cam_name] = cam.async_read()
            cam_timings[cam_name] = (_t() - tc0) * 1e3
        t_end = _t()

        self._last_obs_timing = {
            **{f"{side}_arm_ms": arm_ms[side] for side in _SIDES},
            **{f"{side}_grip_ms": grip_ms[side] for side in _SIDES},
            "cameras_ms": (t_end - t_cams) * 1e3,
            "total_ms": (t_end - t_start) * 1e3,
            **{f"cam[{k}]_ms": v for k, v in cam_timings.items()},
        }

        return obs

    # =========================================================================
    # Action
    # =========================================================================

    def _warn_reach_exceeded(self, side: str, target: np.ndarray) -> None:
        now = time.monotonic()
        if now - self._reach_warn_time[side] < 1.0:
            return
        self._reach_warn_time[side] = now
        dist = float(np.linalg.norm(target[:3]))
        self.logger.warn(
            f"[{side}] commanded TCP {dist * 1000:.0f}mm from base exceeds max_reach_radius "
            f"{self.config.max_reach_radius * 1000:.0f}mm; holding last in-reach pose "
            f"(bring the target back inside the workspace to resume)."
        )

    def _native_tracking(self, side: str) -> bool:
        """Whether this arm's native servo is up and still working toward a submitted target."""
        driver = self._driver[side]
        if driver is None:
            return False
        status = driver.status()
        return bool(status.running and not status.error)

    def _merge_base_pose(self, side: str) -> np.ndarray:
        """Base-frame pose a partial action is merged against; see the single-arm driver."""
        with self._servo_lock[side]:
            last = None if self._last_tcp_command[side] is None else self._last_tcp_command[side].copy()
            fresh = time.monotonic() - self._last_submit_time[side] <= self.config.command_stale_timeout_s
        tracking = self._native_tracking(side)
        if last is not None and fresh and tracking:
            return last
        if tracking:
            return _matrix_to_pose6(self._driver[side].commanded_pose())
        assert self._rtsi[side] is not None
        return _controller_tcp_to_pose6(self._rtsi[side].getActualTCPPose())

    def _cartesian_action_to_tcp_pose(self, side: str, action: dict[str, Any]) -> np.ndarray:
        last_base = self._merge_base_pose(side)

        # The incoming action is in the world frame; merge it against the last
        # commanded pose expressed in world so partial (position-only) actions
        # keep the same per-axis semantics as the single-arm driver, then map the
        # merged target back into base for native interpolation / Pink-style IK.
        target_world = self._base_pose6_to_world(side, last_base)

        pos_keys = self._tcp_pos_keys[side]
        rot_keys = self._tcp_rot_keys[side]
        for i, key in enumerate(pos_keys):
            if key in action:
                target_world[i] = float(action[key])

        if any(key in action for key in rot_keys):
            if not all(key in action for key in rot_keys):
                raise ValueError(
                    f"Incomplete rotation-6D action ({side}). Expected {rot_keys[0]} through {rot_keys[-1]} together."
                )
            r6d = np.array([float(action[key]) for key in rot_keys], dtype=np.float64)
            target_world[3:6] = _quaternion_to_rotvec(rotation_6d_to_quaternion(r6d))

        target = self._world_pose6_to_base(side, target_world)

        # Workspace guard: last_base is in-reach by construction (last commanded
        # in-reach pose, or the physical current pose), so holding it keeps the arm
        # at the boundary instead of chasing an unreachable target into a drop.
        if _reach_exceeded(target, self.config.max_reach_radius) is not None:
            self._warn_reach_exceeded(side, target)
            return last_base.copy()
        return target

    def _trace_send_action(self, side: str, action: dict[str, Any], target_tcp: np.ndarray) -> None:
        if not self.config.trace_servoj:
            return
        try:
            current = _controller_tcp_to_pose6(self._rtsi[side].getActualTCPPose())
        except Exception:
            return
        last = self._last_tcp_command[side].copy() if self._last_tcp_command[side] is not None else current.copy()

        d_lin_vs_last = float(np.linalg.norm(target_tcp[:3] - last[:3]))
        tgt_rot = Rotation.from_rotvec(target_tcp[3:6])
        last_rot = Rotation.from_rotvec(last[3:6])
        d_ang_vs_last = float(np.linalg.norm((tgt_rot * last_rot.inv()).as_rotvec()))

        msg = (
            f"[{side}] send_action tgt=({target_tcp[0]:+.4f},{target_tcp[1]:+.4f},{target_tcp[2]:+.4f},"
            f"rv=[{target_tcp[3]:+.3f},{target_tcp[4]:+.3f},{target_tcp[5]:+.3f}]) "
            f"d_lin(vs_last={d_lin_vs_last * 1000:.2f}mm) "
            f"d_ang(vs_last={np.rad2deg(d_ang_vs_last):.2f}deg)"
        )
        self.logger.debug(msg)

        if (
            self.config.trace_translation_threshold > 0 and d_lin_vs_last > self.config.trace_translation_threshold
        ) or (self.config.trace_rotation_threshold > 0 and d_ang_vs_last > self.config.trace_rotation_threshold):
            self.logger.warn(f"LARGE STEP {msg}")

    def _trace_send_action_joint(self, side: str, target_joints: list[float], current_joints: list[float]) -> None:
        if not self.config.trace_servoj:
            return
        deltas = [t - c for t, c in zip(target_joints, current_joints, strict=True)]
        max_abs_delta = max((abs(d) for d in deltas), default=0.0)
        msg = (
            f"[{side}] joint send_action "
            f"tgt=[{','.join(f'{j:+.3f}' for j in target_joints)}] "
            f"delta=[{','.join(f'{d:+.3f}' for d in deltas)}] max_abs={max_abs_delta:.3f}rad"
        )
        self.logger.debug(msg)
        if self.config.trace_joint_threshold > 0 and max_abs_delta > self.config.trace_joint_threshold:
            self.logger.warn(f"LARGE JOINT STEP {msg}")

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        with self._joint_reset.lock:
            self._joint_reset.check_error()
            if self._joint_reset.closing:
                raise DeviceNotConnectedError(f"{self} is disconnecting.")
            if self._joint_reset.active:
                return {}
            return self._send_action(action)

    def _send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        sent: dict[str, Any] = {}
        for side in _SIDES:
            self._send_arm_action(side, action, sent)
        return sent or action

    def _send_arm_action(self, side: str, action: dict[str, Any], sent: dict[str, Any]) -> None:
        driver = self._driver[side]
        assert driver is not None
        self._raise_servo_error_if_any(side)

        gripper = self._gripper[side]
        gripper_key = self._gripper_key[side]

        if self.config.control_mode == BiEliteCS66RTControlMode.CARTESIAN_SERVO:
            target_tcp = self._cartesian_action_to_tcp_pose(side, action)
            self._trace_send_action(side, action, target_tcp)
            driver.submit_target(_pose6_to_matrix(target_tcp))
            with self._servo_lock[side]:
                self._last_tcp_command[side] = target_tcp.copy()
                self._last_submit_time[side] = time.monotonic()
            # Report the sent pose back in the world frame so callers (display /
            # replay) stay consistent with get_observation. The dataset action is
            # recorded from the teleop/policy action, not this return value.
            sent.update(self._tcp_rotvec_to_feature_values(side, self._base_pose6_to_world(side, target_tcp)))
        else:
            joint_keys = self._joint_pos_keys[side]
            if not all(key in action for key in joint_keys):
                missing = [key for key in joint_keys if key not in action]
                raise ValueError(f"Missing joint servo action keys ({side}): {missing}")
            target_joints = [float(action[key]) for key in joint_keys]
            assert self._rtsi[side] is not None
            current_joints = list(self._rtsi[side].getActualJointPositions())
            self._trace_send_action_joint(side, target_joints, current_joints)
            driver.submit_joints(target_joints)
            sent.update(dict(zip(joint_keys, target_joints, strict=True)))

        if gripper is not None and gripper_key in action:
            gripper.set_gripper_position(float(action[gripper_key]))
            sent[gripper_key] = float(action[gripper_key])

    # =========================================================================
    # Reset
    # =========================================================================

    def reset_to_initial_position(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self._joint_reset.start(
            self._driver,
            self._rtsi,
            self._q_start,
            lambda targets: self._check_joint_move(targets, "reset"),
            self._clear_reset_targets,
        )

    def _clear_reset_targets(self) -> None:
        for side in _SIDES:
            with self._servo_lock[side]:
                self._last_tcp_command[side] = None
                self._last_submit_time[side] = 0.0

    # =========================================================================
    # RT status + pose getters
    # =========================================================================

    def get_native_status(self) -> dict[str, Any]:
        """Per-arm native IK diagnostics, watchdog errors and planner overruns (+ the collision layer's)."""
        status = {side: self._driver[side].status() if self._driver[side] else None for side in _SIDES}
        if self._coordinator is not None:
            status["station"] = self._coordinator.status()
        return status

    @property
    def rt_running(self) -> bool:
        return all(self._driver[s] is not None and self._driver[s].status().running for s in _SIDES)

    @property
    def rt_moving(self) -> bool:
        return self._joint_reset.moving()

    def get_reset_status(self) -> dict:
        """One measured reset outcome for both independently saved arm targets."""
        return self._joint_reset.status()

    def _arm_tcp_pose_quat(self, side: str) -> np.ndarray:
        rtsi = self._rtsi[side]
        assert rtsi is not None
        # Return the pose in the gravity-aligned world frame, consistent with
        # get_observation (RTSI reports it in the tilted base frame).
        tcp_pose = self._base_pose6_to_world(side, _controller_tcp_to_pose6(rtsi.getActualTCPPose()))
        quat = _rotvec_to_quaternion(tcp_pose[3:6])
        gripper = self._gripper[side]
        gripper_pos = gripper.get_gripper_position() if gripper is not None else 0.0
        return np.array(
            [tcp_pose[0], tcp_pose[1], tcp_pose[2], quat[0], quat[1], quat[2], quat[3], gripper_pos],
            dtype=np.float64,
        )

    def _arm_tcp_pose_euler(self, side: str, tcp_pose: np.ndarray | None = None) -> np.ndarray:
        if tcp_pose is None:
            rtsi = self._rtsi[side]
            assert rtsi is not None
            tcp_pose = _controller_tcp_to_pose6(rtsi.getActualTCPPose())
        # ``tcp_pose`` (whether from RTSI or the native commanded pose) is in
        # the tilted base frame; lift it into world for a consistent report.
        tcp_pose = self._base_pose6_to_world(side, np.asarray(tcp_pose, dtype=np.float64))
        quat = _rotvec_to_quaternion(tcp_pose[3:6])
        euler = quaternion_to_euler(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
        gripper = self._gripper[side]
        gripper_pos = gripper.get_gripper_position() if gripper is not None else 0.0
        return np.array(
            [tcp_pose[0], tcp_pose[1], tcp_pose[2], euler[0], euler[1], euler[2], gripper_pos],
            dtype=np.float64,
        )

    def get_current_tcp_pose_quat(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        return self._arm_tcp_pose_quat("left"), self._arm_tcp_pose_quat("right")

    def get_current_tcp_pose_euler(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        return self._arm_tcp_pose_euler("left"), self._arm_tcp_pose_euler("right")

    def get_commanded_tcp_pose_euler(self) -> tuple[np.ndarray, np.ndarray]:
        """Native planned TCP pose (Euler + gripper) per arm.

        Falls back to the measured TCP for an arm whose native servo is down or
        faulted, when the planner's pose is either its initial value or frozen at the
        fault.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        poses = []
        for side in _SIDES:
            if self._native_tracking(side):
                poses.append(self._arm_tcp_pose_euler(side, _matrix_to_pose6(self._driver[side].commanded_pose())))
            else:
                poses.append(self._arm_tcp_pose_euler(side))
        return poses[0], poses[1]
