#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Elite CS66 robot integration for LeRobot."""

import importlib
import threading
import time
from contextlib import suppress
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
from lerobot.robots.elite_cs66_rt.config_elite_cs66_rt import (
    EliteCS66RTConfig,
    EliteCS66RTControlMode,
)
from lerobot.robots.elite_cs66_rt.manipulability import (
    SELFCHECK_POS_TOL_M,
    SELFCHECK_ROT_WARN_DEG,
    damping_scale,
    directional_scale,
    joint_velocity_scale,
    manipulability,
    pose_delta,
    tool_consistency,
)
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

TCP_POSITION_KEYS = ("tcp.x", "tcp.y", "tcp.z")
TCP_ROTATION_6D_KEYS = ("tcp.r1", "tcp.r2", "tcp.r3", "tcp.r4", "tcp.r5", "tcp.r6")
JOINT_POSITION_KEYS = tuple(f"joint_{i}.pos" for i in range(1, 7))
JOINT_VELOCITY_KEYS = tuple(f"joint_{i}.vel" for i in range(1, 7))
JOINT_EFFORT_KEYS = tuple(f"joint_{i}.effort" for i in range(1, 7))


def _import_elite_sdk():
    try:
        return importlib.import_module("libpyelite")
    except ImportError as exc:
        raise ImportError(
            "libpyelite is not installed in this environment. Build/install its wheel "
            "inside the active LeRobot environment before connecting an Elite CS66 robot."
        ) from exc


def _pose6_to_matrix(pose: np.ndarray) -> np.ndarray:
    """Convert the existing controller pose boundary to native SE(3)."""
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError("Expected finite six-element controller pose")
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_rotvec(pose[3:]).as_matrix()
    matrix[:3, 3] = pose[:3]
    return matrix


def _matrix_to_pose6(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    return np.concatenate((matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_rotvec()))


def _configure_native_driver(cfg, config, tool) -> None:
    cfg.ik.max_iterations = config.native_ik_max_iterations
    cfg.planner.stale_timeout = config.command_stale_timeout_s
    cfg.planner.max_linear_velocity = config.native_max_tcp_linear_velocity
    cfg.planner.max_angular_velocity = config.native_max_tcp_angular_velocity
    # Keep existing recipe caps effective after moving the servo loop to C++.
    if config.max_lin_speed is not None:
        cfg.planner.max_linear_velocity = min(cfg.planner.max_linear_velocity, config.max_lin_speed)
    if config.max_ang_speed is not None:
        cfg.planner.max_angular_velocity = min(cfg.planner.max_angular_velocity, config.max_ang_speed)
    cfg.planner.velocity_scale = config.native_joint_velocity_scale
    cfg.command_timeout_ms = config.command_timeout_ms
    if config.native_model_path is not None:
        cfg.model_path = str(Path(config.native_model_path).expanduser())
    if tool is not None:
        cfg.tool = list(tool)


def _rotvec_to_quaternion(rotvec: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = Rotation.from_rotvec(rotvec).as_quat()
    return np.array([qw, qx, qy, qz], dtype=np.float64)


def _quaternion_to_rotvec(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    if quat.shape != (4,):
        raise ValueError(f"Expected quaternion [qw, qx, qy, qz], got shape {quat.shape}")
    return Rotation.from_quat(np.array([quat[1], quat[2], quat[3], quat[0]])).as_rotvec()


def _controller_tcp_to_pose6(tcp_pose: np.ndarray) -> np.ndarray:
    """Convert Elite RTSI [x,y,z,roll,pitch,yaw] feedback to rotvec."""
    tcp_pose = np.asarray(tcp_pose, dtype=np.float64)
    if tcp_pose.shape != (6,) or not np.isfinite(tcp_pose).all():
        raise ValueError("Expected finite six-element Elite TCP feedback")
    roll, pitch, yaw = tcp_pose[3:]
    matrix = np.eye(4)
    matrix[:3, :3] = (
        Rotation.from_rotvec([0.0, 0.0, yaw]).as_matrix()
        @ Rotation.from_rotvec([0.0, pitch, 0.0]).as_matrix()
        @ Rotation.from_rotvec([roll, 0.0, 0.0]).as_matrix()
    )
    matrix[:3, 3] = tcp_pose[:3]
    return _matrix_to_pose6(matrix)


def _rotvec_continuity_shift(target_rotvec: np.ndarray, reference_rotvec: np.ndarray) -> np.ndarray:
    """Re-express ``target_rotvec`` so it lies in the same ±2π branch as ``reference_rotvec``.

    ``Rotation.as_rotvec()`` always returns a principal-branch rotvec (norm ≤ π),
    but the Elite controller stores arbitrary branches (the actual joint state's
    rotvec). When the principal branch and the reference disagree by ~2π along
    the axis, ``get_inverse_kin`` near the controller may pick a wrist-flipped
    joint solution and trip the joint velocity limit. Pick the branch closest to
    the reference so the IK seed stays continuous.
    """
    target = np.asarray(target_rotvec, dtype=np.float64)
    reference = np.asarray(reference_rotvec, dtype=np.float64)
    target_angle = float(np.linalg.norm(target))
    if target_angle < 1e-9:
        return target
    axis = target / target_angle
    ref_along = float(np.dot(reference, axis))
    # Choose k so |target_angle + k*2π - ref_along| is minimised.
    k = round((ref_along - target_angle) / (2.0 * np.pi))
    if k == 0:
        return target
    return axis * (target_angle + k * 2.0 * np.pi)


def _clamp_tcp_velocity(
    last: np.ndarray,
    target: np.ndarray,
    dt: float,
    max_lin_speed: float | None,
    max_ang_speed: float | None,
) -> np.ndarray:
    """Bound the per-tick motion of ``target`` away from ``last`` to velocity caps.

    Both poses are ``[x, y, z, rx, ry, rz]`` (rotvec) in the same frame. Called
    once per servo tick (``dt`` = ``servoj_time``) against the *last commanded*
    pose, so a far or jumpy target (fast wrist rotation, VR tracking spike, clutch
    re-engage) becomes a smooth bounded ramp instead of a single step that would
    imply a joint speed above the controller's ``JOINT_IGNORE_SPEED`` bound and
    trip a protective stop.

    - Translation is scaled so its step never exceeds ``max_lin_speed * dt``.
    - Rotation takes a geodesic (constant-axis) step toward ``target`` no larger
      than ``max_ang_speed * dt``, then is re-expressed on ``last``'s ±2π branch
      so the servoj stream stays continuous.

    A cap of ``None`` (or <= 0) disables that axis; both ``None`` returns ``target``
    untouched, so the default config preserves the historical no-clamp behaviour.
    """
    out = target.copy()

    if max_lin_speed is not None and max_lin_speed > 0.0:
        max_lin_step = max_lin_speed * dt
        delta = target[:3] - last[:3]
        dist = float(np.linalg.norm(delta))
        if dist > max_lin_step:
            out[:3] = last[:3] + delta * (max_lin_step / dist)

    if max_ang_speed is not None and max_ang_speed > 0.0:
        max_ang_step = max_ang_speed * dt
        last_rot = Rotation.from_rotvec(last[3:6])
        rel = (Rotation.from_rotvec(target[3:6]) * last_rot.inv()).as_rotvec()
        ang = float(np.linalg.norm(rel))
        if ang > max_ang_step:
            stepped = (Rotation.from_rotvec(rel * (max_ang_step / ang)) * last_rot).as_rotvec()
            out[3:6] = _rotvec_continuity_shift(stepped, last[3:6])

    return out


def _reach_exceeded(pose_base: np.ndarray, max_reach_radius: float | None) -> float | None:
    """Return the base-origin distance (m) of ``pose_base`` if it exceeds the
    reach radius, else ``None``.

    Distance is measured from the robot base-frame origin — the frame RTSI reports
    TCP poses in. A target beyond the arm's reachable radius drives the controller
    IK into a boundary singularity where ``servoj`` is rejected and external
    control drops (writeServoj fails for ~1 s, then the servo loop raises). Callers
    hold the last in-reach pose instead. ``None`` radius disables the guard.

    This is a conservative *spherical* guard from the base origin, not an exact
    reachability test — the true workspace is offset to the shoulder and excludes
    a column near the base — so keep a margin below the measured failure radius.
    """
    if max_reach_radius is None:
        return None
    dist = float(np.linalg.norm(np.asarray(pose_base, dtype=np.float64)[:3]))
    return dist if dist > max_reach_radius else None


def _slerp_quaternion_wxyz(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)

    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        quat = q0 + alpha * (q1 - q0)
        return quat / np.linalg.norm(quat)

    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * alpha
    sin_theta = np.sin(theta)
    scale_0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    scale_1 = sin_theta / sin_theta_0
    return scale_0 * q0 + scale_1 * q1


class EliteCS66RT(Robot):
    """Single Elite CS66 arm using libpyelite external control.

    Cartesian mode:
        action/observation features are tcp.x/y/z plus tcp.r1..tcp.r6.
        Elite's native [rx, ry, rz] rotation vector is kept as an internal SDK
        detail and converted inside send_action()/get_observation().

    Joint mode:
        action features are joint_1.pos ... joint_6.pos and are streamed with
        writeServoj(..., cartesian=False).
    """

    config_class = EliteCS66RTConfig
    name = "elite_cs66_rt"

    # RTSI fields we actively read via SDK helpers. Validated against the
    # output recipe in connect() so a recipe missing one of these raises
    # before we seed control state from zero-filled placeholder reads —
    # SDK getActualTCPPose() etc silently return [0]*6 on missing fields
    # (see RtsiRecipe.hpp::getValue), which would otherwise let the robot
    # MoveJ toward the world origin on the first reset.
    _REQUIRED_RTSI_OUTPUT_FIELDS = (
        "timestamp",
        "actual_TCP_pose",
        "actual_joint_positions",
        "actual_joint_speeds",
        "actual_joint_torques",
    )

    def __init__(self, config: EliteCS66RTConfig):
        super().__init__(config)
        self.config = config
        logger_suffix = config.id if config.id is not None else hex(id(self))
        self.logger = get_logger(f"EliteCS66RT.{logger_suffix}")

        self._cs = None
        self._dashboard = None
        self._driver = None
        self._rtsi = None
        self._is_connected = False
        self._gripper: Gripper | None = make_gripper_from_config(config.gripper)
        self._last_tcp_command: np.ndarray | None = None
        self._reach_warn_time: float = 0.0  # throttle for the workspace-guard warning
        # Singularity damping (set up at connect; stays disabled unless DH + self-check pass).
        self._dh: tuple[list[float], list[float], list[float]] | None = None
        self._damping_enabled: bool = False
        self._w_log_time: float = 0.0
        self._servo_lock = threading.Lock()
        self._start_tcp_pose: np.ndarray | None = None

        # External cameras — with the gripper's cameras auto-discovered, resolve
        # the wrist camera + GSPS tactile SNs from hardware and inject them into
        # config.cameras before building, so the feature schema sees them.
        if getattr(config, "_taccap_autodiscover", False):
            self._inject_taccap_cameras()
        elif getattr(config, "_serial_autodiscover", False):
            self._inject_serial_gripper_cameras()
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _gripper_side(self) -> str:
        """The side this arm's gripper reports itself as.

        A single arm has no side of its own, but the gripper does — it is stamped
        on the block in the recipe and burned into the unit's firmware SN. Camera
        keys follow it (``<side>_wrist``), which is what keeps a single-arm
        dataset's schema interchangeable with a bimanual one's.
        """
        return getattr(self.config.gripper, "side", "left")

    def _inject_taccap_cameras(self) -> None:
        """Auto-discover this arm's TacCap wrist + GSPS tactile devices into
        ``config.cameras``. Called only in taccap_follower auto-discover mode."""
        mcu_devices = inject_taccap_cameras(
            self.config.cameras,
            sides=(self._gripper_side,),
            enable_tactile=self.config.gripper.enable_tactile,
            logger=self.logger,
            undistort_wrist=self.config.gripper.undistort_wrist,
            fisheye_balance=self.config.gripper.fisheye_balance,
        )
        # The sweep already resolved the gripper's MCU path; pin it so the
        # driver's connect() skips a second scan of the same bus.
        for found_side, mcu_device in mcu_devices.items():
            adopt_taccap_mcu_device(self._gripper, found_side, mcu_device, self.logger)

    def _inject_serial_gripper_cameras(self) -> None:
        """Same as _inject_taccap_cameras for serial (parallel-jaw) grippers."""
        inject_serial_gripper_cameras(
            self.config.cameras,
            sides=(self._gripper_side,),
            enable_tactile=self.config.gripper.enable_tactile,
            logger=self.logger,
        )

    @cached_property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        features: dict[str, type | tuple[int, int, int]] = {}

        if self.config.observe_tcp:
            features.update(dict.fromkeys(TCP_POSITION_KEYS + TCP_ROTATION_6D_KEYS, float))
        if self.config.observe_joints:
            features.update(dict.fromkeys(JOINT_POSITION_KEYS, float))
            features.update(dict.fromkeys(JOINT_VELOCITY_KEYS, float))
            features.update(dict.fromkeys(JOINT_EFFORT_KEYS, float))

        if self._gripper is not None:
            features["gripper.pos"] = float

        for cam_name in self.cameras:
            features[cam_name] = (self.config.cameras[cam_name].height, self.config.cameras[cam_name].width, 3)
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        if self.config.control_mode == EliteCS66RTControlMode.JOINT_SERVO:
            features = dict.fromkeys(JOINT_POSITION_KEYS, float)
        else:
            features = dict.fromkeys(TCP_POSITION_KEYS + TCP_ROTATION_6D_KEYS, float)

        if self._gripper is not None:
            features["gripper.pos"] = float
        return features

    @property
    def is_connected(self) -> bool:
        return (
            self._is_connected
            and self._driver is not None
            and self._rtsi is not None
            and all(cam.is_connected for cam in self.cameras.values())
        )

    @property
    def is_calibrated(self) -> bool:
        # Elite CS66 is factory calibrated; there is no runtime calibration step,
        # so always return True (matches flexiv_rizon4_rt). Connection state is
        # a separate concern, checked via ``is_connected``.
        return True

    def calibrate(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

    def configure(self) -> None:
        pass

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
        """Parse a recipe file (one variable per line, blanks and # comments stripped)."""
        lines = Path(path).read_text().splitlines()
        return [s for s in (line.strip() for line in lines) if s and not s.startswith("#")]

    def _validate_output_recipe(self, path: str) -> None:
        fields = set(self._read_recipe_fields(path))
        missing = [f for f in self._REQUIRED_RTSI_OUTPUT_FIELDS if f not in fields]
        if missing:
            raise ValueError(
                f"RTSI output recipe at {path} is missing required field(s): {missing}. "
                "These are read by SDK helpers (getActualTCPPose, getActualJointPositions, "
                "getActualJointVelocity, getActualJointTorques) which silently return zero "
                "vectors when the field is absent — leaving the robot believing it's at the "
                "world origin and risking a MoveJ into the floor on the next reset."
            )

    def _resolve_recipe(self, configured: str | Path | None, filename: str) -> str:
        if configured is not None:
            path = Path(configured).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"Configured RTSI recipe not found: {path}")
            return str(path)

        sdk_path = None
        with suppress(FileNotFoundError):
            sdk_path = self._resolve_sdk_resource(filename)
        if sdk_path:
            return sdk_path

        module_recipe = Path(__file__).resolve().parent / "resource" / filename
        if module_recipe.exists():
            return str(module_recipe)
        raise FileNotFoundError(
            f"Could not find {filename}. Set rtsi_output_recipe/rtsi_input_recipe in EliteCS66RTConfig."
        )

    def _make_driver_config(self):
        assert self._cs is not None
        cfg = self._cs.EliteDriverConfig()
        cfg.robot_ip = self.config.robot_ip
        cfg.local_ip = self.config.local_ip
        cfg.servoj_time = self.config.servoj_time
        cfg.servoj_lookahead_time = self.config.servoj_lookahead_time
        cfg.servoj_gain = self.config.servoj_gain
        # Headless is the only supported deployment path for this fleet:
        # SDK injects external_control.script via primary 30001; no teach
        # pendant / External Control plug-in involved. See
        # config_elite_cs66_rt.py docstring.
        cfg.headless_mode = True
        if self.config.script_file_path is not None:
            cfg.script_file_path = str(Path(self.config.script_file_path).expanduser())
        else:
            cfg.script_file_path = self._resolve_sdk_resource("external_control.script")
        _configure_native_driver(cfg, self.config, self.config.tool_transform)
        return cfg

    def connect(self, calibrate: bool = False, go_to_start: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected, do not run connect() twice.")

        self._cs = _import_elite_sdk()
        # No Python real-time scheduling; native workers own the servo path.

        try:
            output_recipe = self._resolve_recipe(self.config.rtsi_output_recipe, "output_recipe.txt")
            self._validate_output_recipe(output_recipe)
            input_recipe = self._resolve_recipe(self.config.rtsi_input_recipe, "input_recipe.txt")

            self._rtsi = self._cs.RtsiIOInterface(output_recipe, input_recipe, self.config.rtsi_frequency)
            # Keep the handle on self even if connect() fails (e.g. RTSI "IN_USE"
            # when another client still holds the input registers) so
            # _cleanup_after_failed_connect() can disconnect() it. Dropping the
            # reference here would orphan the C++ object, whose destructor then
            # fires at interpreter shutdown ("terminate called ..." -> Aborted).
            if not self._rtsi.connect(self.config.robot_ip):
                raise ConnectionError(f"Failed to connect Elite RTSI server at {self.config.robot_ip}:30004")

            self._dashboard = self._cs.DashboardClientInterface()
            if not self._dashboard.connect(self.config.robot_ip):
                raise ConnectionError(f"Failed to connect Elite dashboard at {self.config.robot_ip}")

            if not self._dashboard.powerOn():
                raise RuntimeError("Elite CS66 powerOn() failed.")

            if not self._dashboard.brakeRelease():
                raise RuntimeError("Elite CS66 brakeRelease() failed.")

            driver_config = self._make_driver_config()
            driver_construct_time = time.monotonic()
            self._driver = self._cs.EliteDriver(driver_config)
            # Match SDK example timing: let EliteDriver finish wiring up its
            # reverse / trajectory / script-command sockets, AND give the
            # constructor's primary-port script push a chance to land before
            # we decide it failed. Without this pre-window we'd unconditionally
            # fire the safety-net sendExternalControlScript() on every connect.
            if self.config.external_control_settle_s > 0:
                time.sleep(self.config.external_control_settle_s)

            # EliteDriver's constructor already pushed external_control.script
            # to primary 30001. Re-send only if the controller hasn't connected
            # back to our reverse socket yet (transient write loss).
            if not self._driver.isRobotConnected() and not self._driver.sendExternalControlScript():
                raise RuntimeError("Failed to send Elite external control script.")

            deadline = time.monotonic() + self.config.connect_timeout_s
            while not self._driver.isRobotConnected():
                if time.monotonic() > deadline:
                    raise TimeoutError("Timed out waiting for Elite external control script connection.")
                time.sleep(0.01)

            # SDK example sleeps another second here before the first
            # writeServoj; without it the robot-side script can RST the
            # reverse socket. We collapse that into a "minimum total elapsed
            # time since EliteDriver construction" check — fast handshakes
            # don't pay the full second twice.
            remaining = self.config.external_control_settle_s - (time.monotonic() - driver_construct_time)
            if remaining > 0:
                time.sleep(remaining)

            # Gripper before cameras, which is the order the other three arms
            # use. It is not cosmetic here: the wrist fisheye intrinsics live in
            # the gripper's MCU flash, so the gripper has to be open to be read,
            # and a wrist camera has to hold the calibration before its own
            # connect() builds the remap tables. Connecting cameras first left
            # no point at which the calibration could be handed over.
            if self._gripper is not None:
                self.logger.info(f"Connecting gripper ({type(self._gripper).__name__})...")
                self._gripper.connect()

            attach_wrist_fisheye_calibration(self.cameras, {self._gripper_side: self._gripper}, self.logger)

            for cam in self.cameras.values():
                cam.connect()
        except BaseException:
            self._cleanup_after_failed_connect()
            raise

        self._is_connected = True

        # Capture a pre-start-move (joints, TCP) sample for the singularity-damping FK
        # self-check: the start-move below gives a well-separated second config, so the
        # inferred flange->TCP tool transform can be cross-checked for consistency.
        premove_sample = None
        if (
            self.config.singularity_w_high is not None or self.config.joint_vel_limits_rad_s is not None
        ) and self.config.control_mode == EliteCS66RTControlMode.CARTESIAN_SERVO:
            try:
                premove_sample = (
                    np.asarray(self._rtsi.getActualJointPositions(), dtype=np.float64),
                    _controller_tcp_to_pose6(self._rtsi.getActualTCPPose()),
                )
            except Exception:
                premove_sample = None

        # MoveJ to start_position before any servoj streaming. Pass
        # go_to_start=False to skip (crash-recovery / re-attach scenarios
        # where the arm is already mid-pose). MoveJ runs **before** the
        # servo loop starts so it can own the reverse socket exclusively.
        if go_to_start:
            try:
                self.logger.info(
                    f"Elite CS66 moving to start_position over {self.config.start_move_duration_s:.1f}s..."
                )
                self._move_j_blocking(
                    list(self.config.start_position_rad),
                    self.config.start_move_duration_s,
                )
            except BaseException:
                self._is_connected = False
                self._cleanup_after_failed_connect()
                raise

        if self.config.control_mode == EliteCS66RTControlMode.CARTESIAN_SERVO:
            current_tcp = _controller_tcp_to_pose6(self._rtsi.getActualTCPPose())
            self._last_tcp_command = current_tcp.copy()
            self._start_tcp_pose = current_tcp.copy()
        try:
            self._start_servo_loop()
            if self.config.control_mode == EliteCS66RTControlMode.CARTESIAN_SERVO:
                if self.config.singularity_w_high is not None or self.config.joint_vel_limits_rad_s is not None:
                    self._setup_singularity_damping(premove_sample)
        except BaseException:
            self._is_connected = False
            self._cleanup_after_failed_connect()
            raise

    def _cleanup_after_failed_connect(self) -> None:
        # Drop the driver / dashboard / RTSI handles first.
        with best_effort(self.logger, "stopping control"):
            if self._driver is not None:
                self._driver.stopControl(1000)
        with best_effort(self.logger, "closing the dashboard connection"):
            if self._dashboard is not None:
                self._dashboard.disconnect()
        with best_effort(self.logger, "closing the RTSI connection"):
            if self._rtsi is not None:
                self._rtsi.disconnect()
        # Also release any cameras that may have been opened in connect()'s
        # try-block before the failure point.
        for name, cam in self.cameras.items():
            with best_effort(self.logger, f"releasing camera {name}"):
                if cam.is_connected:
                    cam.disconnect()
        # And the gripper, if connect() got that far.
        if self._gripper is not None:
            with best_effort(self.logger, "releasing the gripper"):
                if self._gripper.is_connected:
                    self._gripper.disconnect()
        self._driver = None
        self._dashboard = None
        self._rtsi = None
        self._is_connected = False

    def _start_servo_loop(self) -> None:
        assert self._driver is not None and self._rtsi is not None
        self._driver.startServo(self._rtsi)

    def _stop_servo_loop(self) -> None:
        if self._driver is not None:
            self._driver.stopServo()

    def _move_j_blocking(self, target_joints: list[float], duration_s: float) -> None:
        assert self._driver is not None
        was_running = self._driver.status().running
        self._stop_servo_loop()
        self._driver.moveJ(list(target_joints), float(duration_s), self.config.move_j_timeout_ms)
        if was_running:
            self._start_servo_loop()

    def _raise_servo_error_if_any(self) -> None:
        if self._driver is not None:
            self._driver.check_error()

    def _is_reset_moving_locked(self, now: float) -> bool:
        return self._driver is not None and self._driver.status().reset_active

    @staticmethod
    def _interpolate_tcp_pose(start: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
        """Geodesic pose interpolation retained for the optional kinematic guards."""
        pose = np.asarray(start, dtype=np.float64).copy()
        pose[:3] += alpha * (target[:3] - start[:3])
        rotation = Rotation.from_rotvec(start[3:6])
        delta = (Rotation.from_rotvec(target[3:6]) * rotation.inv()).as_rotvec()
        pose[3:6] = (Rotation.from_rotvec(alpha * delta) * rotation).as_rotvec()
        return pose

    def _tcp_rotvec_to_feature_values(self, tcp_pose: np.ndarray) -> dict[str, float]:
        values = {
            "tcp.x": float(tcp_pose[0]),
            "tcp.y": float(tcp_pose[1]),
            "tcp.z": float(tcp_pose[2]),
        }
        quat = _rotvec_to_quaternion(tcp_pose[3:6])
        r6d = quaternion_to_rotation_6d(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
        values.update({key: float(value) for key, value in zip(TCP_ROTATION_6D_KEYS, r6d, strict=True)})
        return values

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        assert self._rtsi is not None
        obs: dict[str, Any] = {}

        if self.config.observe_tcp:
            tcp_pose = _controller_tcp_to_pose6(self._rtsi.getActualTCPPose())
            obs.update(self._tcp_rotvec_to_feature_values(tcp_pose))
        if self.config.observe_joints:
            joints = self._rtsi.getActualJointPositions()
            obs.update({key: float(value) for key, value in zip(JOINT_POSITION_KEYS, joints, strict=True)})
            joint_vel = self._rtsi.getActualJointVelocity()
            obs.update({key: float(value) for key, value in zip(JOINT_VELOCITY_KEYS, joint_vel, strict=True)})
            joint_effort = self._rtsi.getActualJointTorques()
            obs.update({key: float(value) for key, value in zip(JOINT_EFFORT_KEYS, joint_effort, strict=True)})

        if self._gripper is not None:
            obs["gripper.pos"] = self._gripper.get_gripper_position()

        for cam_name, cam in self.cameras.items():
            obs[cam_name] = cam.async_read()
        return obs

    # =========================================================================
    # Singularity-aware manipulability damping
    # =========================================================================

    def _fetch_dh(self) -> tuple[list[float], list[float], list[float]] | None:
        """Return the arm's Modified-DH ``(alpha, a, d)`` from the config override or the
        controller's primary package, or None if unavailable / unpopulated."""
        if self.config.dh_params is not None:
            alpha, a, d = self.config.dh_params
            return (list(alpha), list(a), list(d))
        assert self._cs is not None and self._driver is not None
        ki = self._cs.KinematicsInfo()
        if not self._driver.getPrimaryPackage(ki, self.config.primary_timeout_ms):
            return None
        dh = (list(ki.dh_alpha_), list(ki.dh_a_), list(ki.dh_d_))
        if any(len(v) != 6 for v in dh) or all(x == 0.0 for x in dh[0] + dh[1] + dh[2]):
            return None  # empty / unpopulated package
        return dh

    def _setup_singularity_damping(self, premove_sample) -> None:
        """Acquire DH and validate it against the live robot. Fail-safe: any problem leaves
        damping disabled (identical to the no-damping behavior)."""
        self._dh = None
        self._damping_enabled = False
        try:
            dh = self._fetch_dh()
        except Exception as exc:
            self.logger.warn(f"Singularity damping disabled: DH fetch error ({exc}).")
            return
        if dh is None:
            self.logger.warn(
                "Singularity damping disabled: no DH (controller fetch returned nothing and no dh_params override)."
            )
            return

        q1 = np.asarray(self._rtsi.getActualJointPositions(), dtype=np.float64)
        w1 = manipulability(*dh, q1)
        if not np.isfinite(w1) or w1 < 1e-6:
            self.logger.warn(
                f"Singularity damping disabled: manipulability at the start pose is degenerate "
                f"(w={w1:.3e}); the DH is likely wrong."
            )
            return

        validated = False
        if premove_sample is not None:
            q0, t0 = premove_sample
            q0 = np.asarray(q0, dtype=np.float64)
            if int(np.sum(np.abs(q0 - q1) >= 0.3)) >= 3:  # well-separated configs
                t1 = _controller_tcp_to_pose6(self._rtsi.getActualTCPPose())
                pos_drift_m, rot_drift_deg = tool_consistency(dh, q0, t0, q1, t1)
                # Gate on POSITION drift only: it validates the DH for the geometric Jacobian, which
                # is all w / the joint-velocity prediction depend on. A large ROTATION drift is a
                # benign about-flange-Z TCP-convention artifact that does not enter det(J) — log it,
                # don't disable. See manipulability.tool_consistency.
                if pos_drift_m > SELFCHECK_POS_TOL_M:
                    self.logger.warn(
                        f"Singularity damping disabled: FK self-check position drift "
                        f"{pos_drift_m * 1000:.1f}mm > {SELFCHECK_POS_TOL_M * 1000:.1f}mm "
                        f"(DH / convention mismatch)."
                    )
                    return
                if rot_drift_deg > SELFCHECK_ROT_WARN_DEG:
                    self.logger.info(
                        f"FK self-check: tool-orientation drift {rot_drift_deg:.1f}deg (about-flange-Z "
                        f"TCP convention; does NOT affect det(J)) — guard enabled on the "
                        f"position-validated DH (pos drift {pos_drift_m * 1000:.1f}mm)."
                    )
                validated = True

        self._dh = dh
        self._damping_enabled = True
        guards = []
        if self.config.singularity_w_high is not None:
            guards.append("w-damping")
        if self.config.joint_vel_limits_rad_s is not None:
            guards.append("joint-vel-limit")
        guard_str = "+".join(guards)
        if validated:
            self.logger.info(f"Kinematic scaling enabled [{guard_str}] (FK validated; start w={w1:.4f}).")
        else:
            self.logger.info(
                f"Kinematic scaling enabled [{guard_str}] (FK unvalidated by motion — relying on "
                f"w-sanity; start w={w1:.4f}). Connect with go_to_start=True for the full self-check."
            )

    def _maybe_log_w(self, w: float, s: float) -> None:
        now = time.monotonic()
        if now - self._w_log_time < 0.5:
            return
        self._w_log_time = now
        self.logger.info(f"manipulability w={w:.5f} -> damping scale s={s:.3f}")

    def _apply_kinematic_scaling(self, target: np.ndarray, held: np.ndarray) -> np.ndarray:
        """Slow the command toward ``target`` as the current config nears a singularity or as the
        predicted joint step approaches the per-joint velocity limits.

        Combines two independent, opt-in model-based guards (each gated on its own config, both
        sharing the validated DH). The final scale is the min of whichever are active, applied via
        the same ``held -> target`` interpolation, so it composes with the reach guard downstream (a
        smaller step is only ever more in-reach)."""
        assert self._dh is not None and self._rtsi is not None
        q = np.asarray(self._rtsi.getActualJointPositions(), dtype=np.float64)
        s = 1.0

        if self.config.singularity_w_high is not None:
            if self.config.singularity_directional:
                s_sing, w = directional_scale(
                    *self._dh,
                    q,
                    pose_delta(held, target),
                    self.config.singularity_w_low,
                    self.config.singularity_w_high,
                    self.config.singularity_min_scale,
                )
            else:
                w = manipulability(*self._dh, q)
                s_sing = damping_scale(
                    w,
                    self.config.singularity_w_low,
                    self.config.singularity_w_high,
                    self.config.singularity_min_scale,
                )
            if self.config.log_manipulability:
                self._maybe_log_w(w, s_sing)
            s = min(s, s_sing)

        if self.config.joint_vel_limits_rad_s is not None:
            qdot_limit = (
                np.asarray(self.config.joint_vel_limits_rad_s, dtype=np.float64) * self.config.joint_vel_limit_margin
            )
            s_jv, _ = joint_velocity_scale(
                *self._dh,
                q,
                pose_delta(held, target),
                self.config.joint_vel_horizon_s,
                qdot_limit,
                self.config.joint_vel_dls_lambda,
            )
            s = min(s, s_jv)

        if s < 1.0:
            return self._interpolate_tcp_pose(held, target, s)
        return target

    def _warn_reach_exceeded(self, target: np.ndarray) -> None:
        now = time.monotonic()
        if now - self._reach_warn_time < 1.0:
            return
        self._reach_warn_time = now
        dist = float(np.linalg.norm(target[:3]))
        self.logger.warn(
            f"Commanded TCP {dist * 1000:.0f}mm from base exceeds max_reach_radius "
            f"{self.config.max_reach_radius * 1000:.0f}mm; holding last in-reach pose "
            f"(bring the target back inside the workspace to resume)."
        )

    def _cartesian_action_to_tcp_pose(self, action: dict[str, Any]) -> np.ndarray:
        with self._servo_lock:
            last_tcp = None if self._last_tcp_command is None else self._last_tcp_command.copy()

        if self._driver is not None and self._driver.status().running:
            last_tcp = _matrix_to_pose6(self._driver.commanded_pose())

        if last_tcp is not None:
            target = last_tcp
        else:
            assert self._rtsi is not None
            target = _controller_tcp_to_pose6(self._rtsi.getActualTCPPose())

        # Last in-reach pose (before this action is applied); held if the merged
        # target leaves the workspace, so the arm never chases an unreachable target.
        held = target.copy()

        for i, key in enumerate(TCP_POSITION_KEYS):
            if key in action:
                target[i] = float(action[key])

        if any(key in action for key in TCP_ROTATION_6D_KEYS):
            if not all(key in action for key in TCP_ROTATION_6D_KEYS):
                raise ValueError("Incomplete rotation-6D action. Expected tcp.r1 through tcp.r6 together.")
            r6d = np.array([float(action[key]) for key in TCP_ROTATION_6D_KEYS], dtype=np.float64)
            # Convert the target rotation to a rotvec, then put it on the
            # same ±2π·axis branch as the rotvec we sent last tick. We use
            # **our own last-commanded rotvec** (target[3:6], seeded from
            # _last_tcp_command on entry) as the continuity anchor — NOT
            # RTSI's reported `current`. RTSI's rotvec is unstable near
            # θ≈π and can flip rx sign without the robot moving; chasing it
            # caused the prior "External Control speed limit" trips. Our
            # own rotvec stream is by construction continuous frame-to-
            # frame, so Elite SDK's IK (seeded with cmd_servo_joints) sees
            # small joint deltas and stays inside the velocity envelope.
            target_principal = _quaternion_to_rotvec(rotation_6d_to_quaternion(r6d))
            target[3:6] = _rotvec_continuity_shift(target_principal, target[3:6])

        # Kinematic scaling (singularity damping and/or predicted joint-velocity limit) pulls the
        # target toward `held` (last commanded) as the current config nears a singularity or the
        # predicted joint step nears the velocity limits. Runs BEFORE the reach guard: the scaled
        # target is closer to `held`, so it can only be more in-reach, preserving the invariant.
        if self._damping_enabled:
            target = self._apply_kinematic_scaling(target, held)

        if _reach_exceeded(target, self.config.max_reach_radius) is not None:
            self._warn_reach_exceeded(target)
            return held
        return target

    def _trace_send_action(self, action: dict[str, Any], target_tcp: np.ndarray) -> None:
        """Log enough state to diagnose joint-velocity-limit trips after the fact.

        Emits two records per send_action:
          1. ``elite-trace`` (DEBUG) — full target / current / delta dump on every
             call; goes only to the file sink.
          2. ``elite-trace-warn`` (WARN) — promoted when the per-step delta
             *between consecutive sent targets* (vs_last) exceeds the
             configured thresholds. We deliberately do NOT alarm on the
             delta against RTSI's reported current pose: RTSI's rotvec
             encoding is unstable near orientation singularities and can
             flip 2π·axis between consecutive ticks without the robot
             physically moving, which generates a constant stream of
             false positives. ``vs_last`` is host-side-only and stable.
        """
        if not self.config.trace_servoj:
            return
        try:
            current = _controller_tcp_to_pose6(self._rtsi.getActualTCPPose())
        except Exception:
            return
        last = self._last_tcp_command.copy() if self._last_tcp_command is not None else current.copy()

        d_lin_vs_current = float(np.linalg.norm(target_tcp[:3] - current[:3]))
        d_lin_vs_last = float(np.linalg.norm(target_tcp[:3] - last[:3]))

        cur_rot = Rotation.from_rotvec(current[3:6])
        tgt_rot = Rotation.from_rotvec(target_tcp[3:6])
        last_rot = Rotation.from_rotvec(last[3:6])
        d_ang_vs_current = float(np.linalg.norm((tgt_rot * cur_rot.inv()).as_rotvec()))
        d_ang_vs_last = float(np.linalg.norm((tgt_rot * last_rot.inv()).as_rotvec()))

        r1 = float(action.get("tcp.r1", float("nan")))
        r2 = float(action.get("tcp.r2", float("nan")))
        r6d_norms = (
            np.linalg.norm([action.get(f"tcp.r{i + 1}", 0.0) for i in range(3)]) if "tcp.r1" in action else float("nan")
        )

        msg = (
            f"send_action tgt=({target_tcp[0]:+.4f},{target_tcp[1]:+.4f},{target_tcp[2]:+.4f},"
            f"rv=[{target_tcp[3]:+.3f},{target_tcp[4]:+.3f},{target_tcp[5]:+.3f}]) "
            f"cur=({current[0]:+.4f},{current[1]:+.4f},{current[2]:+.4f},"
            f"rv=[{current[3]:+.3f},{current[4]:+.3f},{current[5]:+.3f}]) "
            f"d_lin(vs_cur={d_lin_vs_current * 1000:.2f}mm,vs_last={d_lin_vs_last * 1000:.2f}mm) "
            f"d_ang(vs_cur={np.rad2deg(d_ang_vs_current):.2f}deg,vs_last={np.rad2deg(d_ang_vs_last):.2f}deg) "
            f"r6d_col1_norm={r6d_norms:.3f} r1={r1:+.3f} r2={r2:+.3f}"
        )
        self.logger.debug(msg)

        # Alarm on host-side jumps only (vs_last). vs_cur deltas can be huge
        # near orientation singularities purely from RTSI's rotvec axis-sign
        # noise; alarming on that drowns the log in false warnings while the
        # robot is in fact tracking smoothly.
        if (
            self.config.trace_translation_threshold > 0 and d_lin_vs_last > self.config.trace_translation_threshold
        ) or (self.config.trace_rotation_threshold > 0 and d_ang_vs_last > self.config.trace_rotation_threshold):
            self.logger.warn(f"LARGE STEP {msg}")

    def _trace_send_action_joint(self, target_joints: list[float], current_joints: list[float]) -> None:
        """Joint-mode counterpart to ``_trace_send_action``.

        RTSI joint readings are clean (no SO(3) branch-cut noise), so unlike
        the Cartesian path we trace deltas against the actual current joints
        directly — no host-side ``_last_joint_command`` anchor needed.
        """
        if not self.config.trace_servoj:
            return
        deltas = [t - c for t, c in zip(target_joints, current_joints, strict=True)]
        max_abs_delta = max(abs(d) for d in deltas) if deltas else 0.0
        msg = (
            "joint send_action "
            f"tgt=[{','.join(f'{j:+.3f}' for j in target_joints)}] "
            f"cur=[{','.join(f'{j:+.3f}' for j in current_joints)}] "
            f"delta=[{','.join(f'{d:+.3f}' for d in deltas)}] "
            f"max_abs={max_abs_delta:.3f}rad"
        )
        self.logger.debug(msg)

        if self.config.trace_joint_threshold > 0 and max_abs_delta > self.config.trace_joint_threshold:
            self.logger.warn(f"LARGE JOINT STEP {msg}")

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        assert self._driver is not None
        self._raise_servo_error_if_any()

        sent: dict[str, Any] = {}

        if self.config.control_mode == EliteCS66RTControlMode.CARTESIAN_SERVO:
            if self._driver.status().reset_active:
                if self._gripper is not None and "gripper.pos" in action:
                    self._gripper.set_gripper_position(float(action["gripper.pos"]))
                    sent["gripper.pos"] = float(action["gripper.pos"])
                return sent or action

            target_tcp = self._cartesian_action_to_tcp_pose(action)
            self._trace_send_action(action, target_tcp)
            self._driver.submit_target(_pose6_to_matrix(target_tcp))
            self._last_tcp_command = target_tcp.copy()
            sent.update(self._tcp_rotvec_to_feature_values(target_tcp))
        else:
            if not all(key in action for key in JOINT_POSITION_KEYS):
                missing = [key for key in JOINT_POSITION_KEYS if key not in action]
                raise ValueError(f"Missing joint servo action keys: {missing}")
            target_joints = [float(action[key]) for key in JOINT_POSITION_KEYS]
            assert self._rtsi is not None
            current_joints = list(self._rtsi.getActualJointPositions())
            self._trace_send_action_joint(target_joints, current_joints)
            self._driver.submit_joints(target_joints)
            sent.update(dict(zip(JOINT_POSITION_KEYS, target_joints, strict=True)))

        if self._gripper is not None and "gripper.pos" in action:
            self._gripper.set_gripper_position(float(action["gripper.pos"]))
            sent["gripper.pos"] = float(action["gripper.pos"])

        return sent

    def disconnect(self) -> None:
        # Idempotent: re-running disconnect after a failed connect or after a
        # previous successful disconnect should be a quiet no-op, not raise.
        if not self._is_connected and self._driver is None and self._rtsi is None and self._dashboard is None:
            self.logger.warn(f"{self} is not connected, skipping disconnect.")
            return

        for cam in self.cameras.values():
            if cam.is_connected:
                cam.disconnect()

        self._stop_servo_loop()

        # Smooth return to home before tearing down the reverse socket.
        # Always attempted; on failure we log and continue with shutdown so a
        # faulted arm can't deadlock disconnect().
        if self._driver is not None and self._rtsi is not None:
            try:
                self.logger.info(
                    f"Elite CS66 returning to home_position over {self.config.home_move_duration_s:.1f}s..."
                )
                self._move_j_blocking(
                    list(self.config.home_position_rad),
                    self.config.home_move_duration_s,
                )
            except Exception as exc:
                self.logger.warn(f"Return-to-home failed; proceeding with shutdown anyway: {exc}")
            # MoveJ may have restarted the servo loop in its finally block; kill
            # it again before stopControl.
            self._stop_servo_loop()

        if self._driver is not None:
            try:
                # Clean shutdown: write idle so the controller-side script
                # ramps joint velocity to 0, then stopControl to release
                # reverse sockets so the next connect() can bind them.
                self._driver.writeIdle(self.config.command_timeout_ms)
                self._driver.stopControl(1000)
            finally:
                self._driver = None

        if self._dashboard is not None:
            try:
                self._dashboard.disconnect()
            finally:
                self._dashboard = None

        if self._rtsi is not None:
            try:
                self._rtsi.disconnect()
            finally:
                self._rtsi = None

        if self._gripper is not None:
            try:
                if self._gripper.is_connected:
                    self._gripper.disconnect()
            except Exception as exc:
                self.logger.warn(f"Gripper disconnect failed: {exc}")

        self._is_connected = False

    def get_native_status(self):
        """Native IK diagnostics, watchdog errors and planner overruns."""
        return self._driver.status() if self._driver is not None else None

    @property
    def rt_running(self) -> bool:
        return self._driver is not None and self._driver.status().running

    @property
    def rt_moving(self) -> bool:
        with self._servo_lock:
            return self._is_reset_moving_locked(time.monotonic())

    def get_current_tcp_pose_quat(self) -> np.ndarray:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        assert self._rtsi is not None
        tcp_pose = _controller_tcp_to_pose6(self._rtsi.getActualTCPPose())
        quat = _rotvec_to_quaternion(tcp_pose[3:6])
        gripper_pos = self._gripper.get_gripper_position() if self._gripper is not None else 0.0
        return np.array(
            [tcp_pose[0], tcp_pose[1], tcp_pose[2], quat[0], quat[1], quat[2], quat[3], gripper_pos],
            dtype=np.float64,
        )

    def get_current_tcp_pose_euler(self) -> np.ndarray:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        assert self._rtsi is not None
        tcp_pose = _controller_tcp_to_pose6(self._rtsi.getActualTCPPose())
        quat = _rotvec_to_quaternion(tcp_pose[3:6])
        euler = quaternion_to_euler(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
        gripper_pos = self._gripper.get_gripper_position() if self._gripper is not None else 0.0
        return np.array(
            [tcp_pose[0], tcp_pose[1], tcp_pose[2], euler[0], euler[1], euler[2], gripper_pos],
            dtype=np.float64,
        )

    def get_commanded_tcp_pose_euler(self) -> np.ndarray:
        """Current native planned TCP pose in Euler form, including gripper."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if self._last_tcp_command is None:
            return self.get_current_tcp_pose_euler()
        tcp_pose = _matrix_to_pose6(self._driver.commanded_pose())
        quat = _rotvec_to_quaternion(tcp_pose[3:6])
        euler = quaternion_to_euler(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
        gripper_pos = self._gripper.get_gripper_position() if self._gripper is not None else 0.0
        return np.array(
            [tcp_pose[0], tcp_pose[1], tcp_pose[2], euler[0], euler[1], euler[2], gripper_pos],
            dtype=np.float64,
        )

    def reset_to_initial_position(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self._raise_servo_error_if_any()
        if self.config.control_mode != EliteCS66RTControlMode.CARTESIAN_SERVO:
            self._move_j_blocking(list(self.config.start_position_rad), self.config.reset_duration_s)
        elif self._start_tcp_pose is not None and not self.rt_moving:
            self._driver.submit_target(_pose6_to_matrix(self._start_tcp_pose), self.config.reset_duration_s)
            if not self.config.use_background_servo_loop:
                while self.rt_moving:
                    self._raise_servo_error_if_any()
                    time.sleep(0.01)
