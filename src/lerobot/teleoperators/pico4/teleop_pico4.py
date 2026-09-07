#!/usr/bin/env python

# Copyright 2025 The XenseRobotics Inc. team. All rights reserved.
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
Pico4 VR teleoperator for end-effector control.

This teleoperator provides 6-DoF absolute pose control using VR controllers,
similar to Spacemouse. It outputs accumulated target_pose_6d that can be
directly sent to a Cartesian controller.

Control scheme:
- Grip button: Enable control (must be held to move robot)
- Trigger: Directly controls gripper position (trigger 0 → open, 1 → closed)
- Controller pose: Controls robot TCP pose (when grip is held)

The output format matches ARX5 SDK's spacemouse_teleop.py example:
- target_pose_6d: [x, y, z, roll, pitch, yaw] - absolute EEF pose
- gripper_pos: absolute gripper position in meters
"""

import time
from queue import Queue
from typing import Any

import numpy as np

from lerobot.teleoperators.pico4.config_pico4 import Pico4Config
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.utils.robot_utils import (
    get_logger,
    normalize_quaternion,
    quaternion_to_rotation_6d,
    slerp_quaternion,
)


class Pico4(Teleoperator):
    """
    Pico4 VR teleoperator for end-effector control.

    This teleoperator reads pose data from Pico4 VR controllers and maintains
    a target pose suitable for Cartesian control of robotic arms.

    Control scheme:
    - Grip button: Enable control (must be held to move robot, with hysteresis thresholds)
    - Trigger: Directly controls gripper position (trigger 0 → open, 1 → closed)
    - Controller pose: Controls robot TCP pose (only when grip is held)

    Position control (relative accumulation):
    - When grip is pressed, record controller position as reference
    - target_pos = start_pos + (current_pos - ref_pos) * sensitivity

    Orientation control (absolute mapping with offset):
    - When grip is pressed, calculate offset to align controller with robot orientation
    - target_quat = controller_quat_world * offset
    - This gives intuitive control: controller orientation directly maps to robot orientation

    Coordinate systems:
    - Pico4: Right-handed, X right, Y up, Z toward user
      * Origin: Set as the headset position when the Unity application starts (when Unity app launches)
      * NOT when xrt.init() is called, and NOT when clicking connect button in Unity
      * This means the coordinate system is fixed when Unity app launches, and remains constant until Unity restarts
    - World: Right-handed, X forward (away from base), Y left, Z up (Flexiv-aligned)

    Output action format (world-frame TCP, shared by Flexiv & Elite):
    - tcp.x, tcp.y, tcp.z: absolute EEF target position (meters)
    - tcp.r1-r6: absolute EEF target orientation (6D rotation representation)
    - gripper.pos: absolute gripper position (meters, from trigger)

    6D rotation representation uses the first two columns of the rotation matrix,
    which provides continuous representation without singularities.
    """

    config_class = Pico4Config
    name = "pico4"

    def __init__(self, config: Pico4Config):
        super().__init__(config)
        self.config = config
        self._is_connected = False
        self._xrt = None
        self.logger = get_logger(f"Pico4Teleop/{config.id}")

        # Target pose tracking (in world frame)
        self._target_pos: np.ndarray = np.zeros(3, dtype=np.float32)  # [x, y, z]
        self._target_quat: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # [qw, qx, qy, qz]
        self._target_gripper_pos: float = 0.0

        # Start pose (robot pose when grip is pressed/enabled, in world frame)
        self._start_pos: np.ndarray = np.zeros(3, dtype=np.float32)  # [x, y, z]
        self._start_quat: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # [qw, qx, qy, qz]

        # Reference position for relative position control (controller pos when enabled, in world frame)
        self._ref_pos: np.ndarray | None = None  # [x, y, z] in world frame

        # Quaternion offset for absolute orientation mapping
        # Calculated when enabled: offset = inv(pico4_quat_world) * robot_start_quat
        # Usage: target_quat = pico4_quat_world * offset
        # This aligns controller orientation with robot orientation at enable moment
        self._quat_offset: np.ndarray | None = None  # [qw, qx, qy, qz]

        # Window filter queues for raw Pico4 data (before coordinate transformation)
        # Filter raw pose data from Pico4 SDK
        self._raw_pos_queue: Queue = Queue(self.config.filter_window_size)  # Raw position [x, y, z] in Pico4 frame
        self._raw_quat_queue: Queue = Queue(
            self.config.filter_window_size
        )  # Raw quaternion [qx, qy, qz, qw] in Pico4 frame

        # State tracking
        self._enabled: bool = False
        self._was_enabled: bool = False  # Track previous enable state for edge detection
        self._orientation_control_active: bool = (
            True  # Whether orientation control is active (disabled if offset too large)
        )
        self._was_reset_button_pressed: bool = False  # Track previous reset button state for edge detection
        self._last_grip: float = 0.0  # Last grip value for debugging
        # Physical button states cached each get_action() call.
        # Right controller populates _last_a_button / _last_b_button.
        # Left  controller populates _last_x_button / _last_y_button.
        self._last_a_button: bool = False
        self._last_b_button: bool = False
        self._last_x_button: bool = False
        self._last_y_button: bool = False

        # Position jump filtering
        self._last_raw_pose: np.ndarray | None = None  # Last raw pose for jump detection
        self._jump_filter_count: int = 0  # Count of filtered jumps for debugging

        # Output rate limiter state
        self._last_action_time: float | None = None  # Timestamp of last get_action call
        self._prev_target_pos: np.ndarray | None = None  # Previous frame target position
        self._prev_target_quat: np.ndarray | None = None  # Previous frame target quaternion

    @property
    def is_connected(self) -> bool:
        """Check if the Pico4 VR headset is connected."""
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        """Pico4 doesn't require calibration."""
        return self._is_connected

    @property
    def action_features(self) -> dict[str, Any]:
        """
        Return action features matching world-frame TCP format.

        Returns a dictionary with dtype, shape, and names for the action space:
        - tcp.x, tcp.y, tcp.z: absolute TCP position (meters) in world frame
        - tcp.r1-r6: absolute TCP orientation (6D rotation) in world frame
        - gripper.pos: absolute gripper position (meters)
        """
        return {
            "dtype": "float32",
            "shape": (10,),
            "names": {
                "tcp.x": 0,
                "tcp.y": 1,
                "tcp.z": 2,
                "tcp.r1": 3,
                "tcp.r2": 4,
                "tcp.r3": 5,
                "tcp.r4": 6,
                "tcp.r5": 7,
                "tcp.r6": 8,
                "gripper.pos": 9,
            },
        }

    @property
    def feedback_features(self) -> dict[str, type]:
        """Pico4 doesn't support feedback."""
        return {}

    def connect(
        self, calibrate: bool = True, current_tcp_pose_quat: np.ndarray = np.zeros(8, dtype=np.float32)
    ) -> None:
        """Connect to the Pico4 VR headset via xrt SDK.

        Important: The Pico4 coordinate system origin is set when the Unity application
        starts (when Unity app launches), NOT when xrt.init() is called, and NOT when
        clicking the connect button in Unity. This means:
        - The coordinate origin is fixed when Unity app launches (as soon as Unity starts)
        - The coordinate origin remains fixed as long as Unity app is running
        - If Unity app restarts, the origin will be reset to the new headset position
        - xrt.init() only connects to the service, it does not change the coordinate origin
        - Clicking connect/disconnect in Unity does NOT change the coordinate origin

        Args:
            calibrate: Unused, kept for compatibility with Teleoperator interface
            current_tcp_pose_quat: Current TCP pose in quaternion format [x, y, z, qw, qx, qy, qz, gripper_pos]
        """
        if self._is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.logger.info("Connecting to Pico4 VR headset...")
        try:
            import xensevr_pc_service_sdk as xrt
        except ImportError as e:
            raise ImportError(
                "xensevr_pc_service_sdk is required for Pico4 teleoperator. "
                "Please install it according to your Pico4 SDK documentation."
            ) from e

        try:
            xrt.init()
            self._xrt = xrt
            self.logger.info("XenseVR SDK initialized successfully.")

            # Validate device connection - wait for controller data to be available
            # Total wait time ~3s (same as previous async version)
            time.sleep(0.5)  # Initial wait for SDK to stabilize
            max_retries = 25
            retry_interval = 0.1
            for attempt in range(max_retries):
                if self.config.use_right_controller:
                    pose = xrt.get_right_controller_pose()
                elif self.config.use_left_controller:
                    pose = xrt.get_left_controller_pose()
                else:
                    raise RuntimeError("No controller configured")

                # Check if pose data is valid
                pose_has_data = any(abs(v) > 1e-6 for v in pose)
                if pose_has_data:
                    self.logger.info(f"Controller data received (attempt {attempt + 1})")
                    break
                time.sleep(retry_interval)
            else:
                self._xrt = None
                raise DeviceNotConnectedError(
                    f"Pico4 VR device is not connected. "
                    f"All controller data is zero after {0.5 + max_retries * retry_interval:.1f}s. "
                    f"Please try: 1) Restart the VR Client app on Pico4 headset, "
                    f"2) Ensure the PC service is running, "
                    f"3) Check that controllers are powered on and paired."
                )

            # Set target pose on connect
            # Input format: [x, y, z, qw, qx, qy, qz, gripper_pos] (world frame, wxyz)
            self._target_pos = current_tcp_pose_quat[:3].copy()  # [x, y, z]
            self._target_quat = normalize_quaternion(current_tcp_pose_quat[3:7], input_format="wxyz")
            self._target_gripper_pos = current_tcp_pose_quat[7]

            # Initialize start pose (will be updated when grip is pressed)
            self._start_pos = current_tcp_pose_quat[:3].copy()

            # Initialize reference/offset (will be set when grip is first pressed)
            self._ref_pos = None
            self._quat_offset = None

            # Reset state tracking
            self._enabled = False
            self._was_enabled = False
            self._was_reset_button_pressed = False
            self._orientation_control_active = True

            self._is_connected = True
            self.logger.info(f"{self} connected successfully.")
        except RuntimeError as e:
            self._xrt = None
            raise RuntimeError(f"Failed to initialize XenseVR SDK: {e}") from e
        except DeviceNotConnectedError:
            # Re-raise DeviceNotConnectedError as-is
            raise
        except Exception as e:
            self._xrt = None
            raise DeviceNotConnectedError(
                f"Failed to connect to Pico4 VR device: {e}. "
                f"Please ensure the Pico VR service is running and the device is connected."
            ) from e

    def calibrate(self) -> None:
        """No calibration needed for Pico4."""
        pass

    def configure(self) -> None:
        """No additional configuration needed."""
        pass

    def reset_to_pose(self, pose_7d: np.ndarray, gripper_pos: float = 0.0) -> None:
        """
        Reset target pose to a specific pose (e.g., home pose).

        Args:
            pose_7d: 7D EEF pose [x, y, z, qw, qx, qy, qz] in world frame (wxyz quaternion format)
            gripper_pos: Gripper position in meters
        """
        self._target_pos = np.array(pose_7d[:3], dtype=np.float32).copy()
        self._target_quat = normalize_quaternion(pose_7d[3:7], input_format="wxyz")
        self._target_gripper_pos = float(gripper_pos)

        # Reset reference/offset - will be recalculated when grip is pressed
        self._ref_pos = None
        self._quat_offset = None

        # Reset enable state tracking - so next grip press will be detected as "just enabled"
        self._was_enabled = False
        self._enabled = False
        self._orientation_control_active = True  # Re-enable orientation control

        # Clear filter queues to avoid stale data affecting new control
        while not self._raw_pos_queue.empty():
            self._raw_pos_queue.get()
        while not self._raw_quat_queue.empty():
            self._raw_quat_queue.get()

        # Reset jump filter state
        self._last_raw_pose = None

        # Reset rate limiter state so the new pose is accepted without clamping
        self._prev_target_pos = self._target_pos.copy()
        self._prev_target_quat = self._target_quat.copy()
        self._last_action_time = None

        # Reset is invoked for both arms on a trigger edge. Keep the detailed
        # pose diagnostic in the file/debug log instead of injecting two lines
        # into the live terminal panel while it is being redrawn.
        self.logger.debug(f"Reset target pose to: pos={pose_7d[:3]}, quat={pose_7d[3:7]}, gripper={gripper_pos}")

    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions q1 * q2. Both in [qw, qx, qy, qz] format."""
        qw1, qx1, qy1, qz1 = q1
        qw2, qx2, qy2, qz2 = q2

        qw = qw1 * qw2 - qx1 * qx2 - qy1 * qy2 - qz1 * qz2
        qx = qw1 * qx2 + qx1 * qw2 + qy1 * qz2 - qz1 * qy2
        qy = qw1 * qy2 - qx1 * qz2 + qy1 * qw2 + qz1 * qx2
        qz = qw1 * qz2 + qx1 * qy2 - qy1 * qx2 + qz1 * qw2

        return np.array([qw, qx, qy, qz], dtype=np.float32)

    def _quaternion_inverse(self, q: np.ndarray) -> np.ndarray:
        """Compute quaternion inverse. Input and output in [qw, qx, qy, qz] format."""
        qw, qx, qy, qz = q
        norm_sq = qw * qw + qx * qx + qy * qy + qz * qz
        if norm_sq < 1e-10:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # Identity [qw, qx, qy, qz]
        return np.array([qw, -qx, -qy, -qz], dtype=np.float32) / norm_sq

    def _slerp_quaternion(self, q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        """Spherical Linear Interpolation (SLERP) between two quaternions."""
        return slerp_quaternion(q1, q2, t, input_format="wxyz")

    def _filter_raw_pose(self, controller_pose_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply window filter to raw Pico4 controller pose data.

        This is Step 1: Filter raw data from Pico4 SDK before coordinate transformation.

        For position: Simple moving average (arithmetic mean)
        For quaternion: Sequential SLERP interpolation through the window

        When filter_window_size=1, no filtering is applied (pass-through mode).

        Args:
            controller_pose_raw: Raw controller pose from SDK [x, y, z, qx, qy, qz, qw] in Pico4 frame
                                 Note: Pico4 SDK uses xyzw quaternion format

        Returns:
            Tuple of (filtered_pos, filtered_quat) still in Pico4 coordinate frame:
            - filtered_pos: np.ndarray [x, y, z] position in meters (Pico4 frame)
            - filtered_quat: np.ndarray [qw, qx, qy, qz] quaternion (Pico4 frame, converted to wxyz format)
        """
        # Extract position and quaternion from raw data
        pos = controller_pose_raw[:3].copy()  # [x, y, z] in Pico4 frame
        # Pico4 provides [x, y, z, qx, qy, qz, qw], convert to [qw, qx, qy, qz] for internal use
        quat = np.array(
            [
                controller_pose_raw[6],  # qw
                controller_pose_raw[3],  # qx
                controller_pose_raw[4],  # qy
                controller_pose_raw[5],  # qz
            ],
            dtype=np.float32,
        )  # [qw, qx, qy, qz] in Pico4 frame
        quat = normalize_quaternion(quat, input_format="wxyz")

        # When window_size=1, skip filtering and return raw data directly
        if self.config.filter_window_size <= 1:
            return pos, quat

        # Moving average filter for position (window filter)
        if self._raw_pos_queue.full():
            self._raw_pos_queue.get()
        self._raw_pos_queue.put(pos)
        filtered_pos = np.mean(np.array(list(self._raw_pos_queue.queue)), axis=0)

        # SLERP-based filter for quaternion
        # Method: Use SLERP between first and last quaternion in the window (midpoint)
        # This provides smooth interpolation across the entire window
        if self._raw_quat_queue.full():
            self._raw_quat_queue.get()
        self._raw_quat_queue.put(quat)

        quat_list = list(self._raw_quat_queue.queue)
        n = len(quat_list)

        if n == 1:
            filtered_quat = quat_list[0]
        elif n == 2:
            # For 2 quaternions, SLERP at midpoint (t=0.5)
            filtered_quat = self._slerp_quaternion(quat_list[0], quat_list[1], 0.5)
        else:
            # For multiple quaternions, use recursive SLERP:
            # 1. Split window into two halves
            # 2. SLERP each half to get midpoint
            # 3. SLERP the two midpoints to get final result
            mid = n // 2
            left_half = quat_list[: mid + 1]
            right_half = quat_list[mid:]

            # SLERP first half: from first to middle
            left_mid = left_half[0] if len(left_half) == 1 else self._slerp_quaternion(left_half[0], left_half[-1], 0.5)

            # SLERP second half: from middle to last
            if len(right_half) == 1:
                right_mid = right_half[0]
            else:
                right_mid = self._slerp_quaternion(right_half[0], right_half[-1], 0.5)

            # Final SLERP between the two midpoints
            filtered_quat = self._slerp_quaternion(left_mid, right_mid, 0.5)

        filtered_quat = normalize_quaternion(filtered_quat, input_format="wxyz")

        return filtered_pos, filtered_quat

    def _transform_pico_to_world_coordinate(self, pos: np.ndarray, quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Transform pose from the Pico4 controller frame into the WORLD frame.

        This is Step 2: Coordinate transformation after filtering.

        The "world" frame is the shared, gravity-aligned convention every robot now
        reports in (originally the Flexiv axes): X forward (away from base), Y left,
        Z up. Elite CS66 lifts its tilted base frame into this same convention, so
        the same transform drives any of these robots.

        Pico4:  Right-handed, X right, Y up, Z in (toward user)
        World:  Right-handed, X forward (away from base), Y left, Z up

        If the user stands in front of the robot, "in" (toward user) and "forward"
        (away from base) are opposite.

        Transformation:
        - Pico4 X (right)         -> world Y (left): negate
        - Pico4 Y (up)            -> world Z (up): same
        - Pico4 Z (in, to user)   -> world X (forward, away from base): negate

        Args:
            pos: Position in Pico4 frame [x, y, z]
            quat: Quaternion in Pico4 frame [qw, qx, qy, qz] (from _filter_raw_pose)

        Returns:
            Tuple of (transformed_pos, transformed_quat) in world frame [qw, qx, qy, qz]
        """
        # Position transformation: [Pico4_x, Pico4_y, Pico4_z] -> [world_x, world_y, world_z]
        # Pico4 X (right) -> world Y (left): negate
        # Pico4 Y (up) -> world Z (up): same
        # Pico4 Z (in, toward user) -> world X (forward, away from base): negate (opposite direction)

        # 用户站在机器人前面，面对机器人

        # Pico4 (VR手柄):               world (机器人朝向):
        #     Y (up)                       Z (up)   X (forward, away from base)
        #     |                               |     /
        #     |                               |   /
        #     |                               | /
        #     +--- X (right)     Y (left)-----+
        #    /
        #   Z (toward user)

        transformed_pos = np.array(
            [
                -pos[2],  # Pico4 Z (in) -> world X (forward, negated because opposite direction)
                -pos[0],  # Pico4 X (right) -> world Y (left, negated)
                pos[1],  # Pico4 Y (up) -> world Z (up, same)
            ],
            dtype=np.float32,
        )

        # Quaternion transformation: rotate the rotation from the Pico4 frame to the world frame
        # The transformation is a 120° rotation around axis [1, -1, -1]/√3
        # This corresponds to the position transformation matrix:
        #   [ 0  0 -1]
        #   [-1  0  0]
        #   [ 0  1  0]
        # Quaternion: q = [cos(60°), sin(60°)*axis] = [0.5, 0.5, -0.5, -0.5]
        q_frame_transform = np.array([0.5, 0.5, -0.5, -0.5], dtype=np.float32)  # [qw, qx, qy, qz]

        # Transform quaternion: q_world = q_transform * q_pico * q_transform^-1
        # q_world = q_R * q_pico * q_R^(-1)
        # This formula converts the rotation representation from the Pico4 frame to the world frame
        q_transform_inv = self._quaternion_inverse(q_frame_transform)
        q_temp = self._quaternion_multiply(q_frame_transform, quat)
        transformed_quat = self._quaternion_multiply(q_temp, q_transform_inv)
        transformed_quat = normalize_quaternion(transformed_quat, input_format="wxyz")

        return transformed_pos, transformed_quat

    def get_action(self) -> dict[str, Any]:
        """
        Get the current target pose from the Pico4 VR controller.

        Control scheme:
        - Grip: Enable control (must be held to move robot, with hysteresis thresholds)
        - Trigger: Directly controls gripper position (trigger 0 → open, 1 → closed)
        - Controller pose: Controls robot TCP pose (only when grip is held)

        Data processing pipeline:
        1. Get raw data: controller_pose, grip, trigger from Pico4 SDK
        2. Check enable state (grip > threshold)
        3. Apply window filter to raw pose data
        4. Transform from Pico4 to world frame
        5. If enabled: compute relative movement and update target pose
        6. Update gripper position from trigger value
        7. Return in world-frame TCP format

        Returns a dictionary with absolute EEF pose (matching world-frame TCP format):
        - tcp.x, tcp.y, tcp.z: absolute TCP position (meters) in world frame
        - tcp.r1-r6: absolute TCP orientation (6D rotation) in world frame
        - gripper.pos: absolute gripper position (meters)
        """
        if not self._is_connected or self._xrt is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Step 1: Get controller data from SDK (all data in one batch to minimize SDK calls)
        if self.config.use_right_controller:
            pose = self._xrt.get_right_controller_pose()
            controller_grip = float(self._xrt.get_right_grip())
            controller_trigger = float(self._xrt.get_right_trigger())
            self._last_a_button = bool(self._xrt.get_A_button())
            self._last_b_button = bool(self._xrt.get_B_button())
        elif self.config.use_left_controller:
            pose = self._xrt.get_left_controller_pose()
            controller_grip = float(self._xrt.get_left_grip())
            controller_trigger = float(self._xrt.get_left_trigger())
            self._last_x_button = bool(self._xrt.get_X_button())
            self._last_y_button = bool(self._xrt.get_Y_button())
        else:
            raise RuntimeError("No controller configured")
        controller_pose_raw = np.array(pose, dtype=np.float32)  # [x, y, z, qx, qy, qz, qw] in Pico4 frame
        self._last_grip = controller_grip

        # Step 1.5: Filter out position jumps (VR tracking glitches)
        if self._last_raw_pose is not None and self.config.position_jump_threshold > 0:
            pos_delta = np.linalg.norm(controller_pose_raw[:3] - self._last_raw_pose[:3])
            if pos_delta > self.config.position_jump_threshold:
                self._jump_filter_count += 1
                self.logger.warn(
                    f"[JUMP] Position jump #{self._jump_filter_count}: "
                    f"delta={pos_delta:.4f}m > threshold={self.config.position_jump_threshold}m, "
                    f"raw_pos={controller_pose_raw[:3]}, last_pos={self._last_raw_pose[:3]}. "
                    f"Clamping position to last frame. Auto-recovering next frame."
                )
                controller_pose_raw[:3] = self._last_raw_pose[:3]
                # Reset baseline so next frame establishes a fresh reference instead of
                # permanently clamping (which would lock translation indefinitely).
                self._last_raw_pose = None
            else:
                self._last_raw_pose = controller_pose_raw.copy()
        else:
            self._last_raw_pose = controller_pose_raw.copy()

        # Step 2: Check enable state with hysteresis (grip as enable button)
        prev_enabled = self._enabled
        if self._enabled:
            self._enabled = controller_grip > self.config.grip_disable_threshold
        else:
            self._enabled = controller_grip > self.config.grip_enable_threshold
        if self._enabled != prev_enabled:
            self.logger.debug(
                f"[ENABLE] State changed: {prev_enabled} -> {self._enabled}, "
                f"grip={controller_grip:.3f} "
                f"(enable_thresh={self.config.grip_enable_threshold}, "
                f"disable_thresh={self.config.grip_disable_threshold})"
            )

        # Step 3: Apply window filter to raw pose data (in Pico4 frame)
        raw_pos_pico = controller_pose_raw[:3].copy()
        filtered_pos_pico, filtered_quat_pico = self._filter_raw_pose(controller_pose_raw)
        filter_pos_delta = np.linalg.norm(filtered_pos_pico - raw_pos_pico)
        if filter_pos_delta > 0.01:
            self.logger.debug(
                f"[FILTER] Large filter delta: {filter_pos_delta:.4f}m, "
                f"raw={raw_pos_pico}, filtered={filtered_pos_pico}"
            )

        # Step 4: Transform from Pico4 coordinate system to world frame
        filtered_pos_world, filtered_quat_world = self._transform_pico_to_world_coordinate(
            filtered_pos_pico, filtered_quat_pico
        )

        # Step 5: Handle enable state transitions and pose updates
        # Detect rising edge: grip just pressed (transition from not enabled to enabled)
        just_enabled = self._enabled and not self._was_enabled

        if just_enabled:
            # Reset jump filter baseline so the new grip press starts fresh.
            # Without this, a prior jump clamp would lock position permanently.
            self._last_raw_pose = None

        if just_enabled or self._ref_pos is None:
            self.logger.debug(
                f"[REF_RESET] {'just_enabled' if just_enabled else 'ref_pos is None'}: "
                f"ref_pos={filtered_pos_world}, start_pos={self._target_pos}, "
                f"filtered_quat={filtered_quat_world}"
            )
            self._ref_pos = filtered_pos_world.copy()
            self._start_pos = self._target_pos.copy()

            # Save start orientation for sensitivity scaling
            self._start_quat = self._target_quat.copy()

            # Orientation: calculate offset for absolute mapping
            # offset = inv(controller_quat) * robot_quat
            # This aligns controller orientation with robot orientation at enable moment
            # Later: target_quat = controller_quat * offset
            ref_quat_inv = self._quaternion_inverse(filtered_quat_world)
            self._quat_offset = self._quaternion_multiply(ref_quat_inv, self._target_quat)
            self._quat_offset = normalize_quaternion(self._quat_offset, input_format="wxyz")

            # Check orientation offset angle (rotation difference between controller and robot)
            # θ = 2 * arccos(|qw|), where qw is the scalar part of the offset quaternion
            # For shortest path: q and -q represent the same rotation, so we choose the one with smaller angle
            # If angle > 90°, use -q to get the shorter path (180° - angle)
            offset_w = self._quat_offset[0]
            offset_angle_rad = 2.0 * np.arccos(np.clip(abs(offset_w), 0.0, 1.0))
            offset_angle_deg = np.degrees(offset_angle_rad)

            # If angle > 90°, use the shorter path by negating the quaternion
            # This ensures we always use the quaternion representation with angle <= 90°
            if offset_angle_deg > 90.0:
                # Use the shorter path: negate quaternion and recalculate angle
                self._quat_offset = -self._quat_offset
                self._quat_offset = normalize_quaternion(self._quat_offset, input_format="wxyz")
                offset_angle_rad = 2.0 * np.arccos(np.clip(self._quat_offset[0], 0.0, 1.0))
                offset_angle_deg = np.degrees(offset_angle_rad)
                self.logger.debug(
                    f"Using shorter path: offset angle = {offset_angle_deg:.1f}° (was {180.0 - offset_angle_deg:.1f}°)"
                )

            # Log offset details for debugging
            self.logger.debug(
                f"Orientation offset calculation: "
                f"controller_quat={filtered_quat_world}, "
                f"robot_quat={self._target_quat}, "
                f"offset_quat={self._quat_offset}, "
                f"offset_angle={offset_angle_deg:.1f}°"
            )

            if offset_angle_deg > self.config.orientation_offset_warning_deg:
                self.logger.warn(
                    f"Orientation offset too large: {offset_angle_deg:.1f}° > {self.config.orientation_offset_warning_deg}°. "
                    f"Please align controller with robot orientation. Orientation control DISABLED."
                )
                self._orientation_control_active = False
            else:
                self._orientation_control_active = True
                self.logger.debug(f"Orientation offset: {offset_angle_deg:.1f}°")

            self.logger.debug("Enable engaged - reference pose and orientation offset set")

        # Update previous enable state for next iteration
        self._was_enabled = self._enabled

        # Only update target pose when enabled (grip is held)
        if self._enabled:
            # === Position: Relative accumulation (always active) ===
            rel_pos = filtered_pos_world - self._ref_pos
            scaled_rel_pos = rel_pos * self.config.pos_sensitivity
            self._target_pos = self._start_pos + scaled_rel_pos

            rel_pos_norm = np.linalg.norm(rel_pos)
            if rel_pos_norm < 1e-6:
                self.logger.debug(
                    f"[POS] rel_pos is near ZERO ({rel_pos_norm:.6f}m): "
                    f"filtered={filtered_pos_world}, ref={self._ref_pos}"
                )

            # === Orientation: Absolute mapping with offset (only if offset is within threshold) ===
            if self._orientation_control_active:
                # target_quat = controller_quat_world * offset
                # The offset was calculated at enable time to align the two orientations
                # This gives intuitive control: controller orientation directly maps to robot orientation
                full_target_quat = self._quaternion_multiply(filtered_quat_world, self._quat_offset)
                full_target_quat = normalize_quaternion(full_target_quat, input_format="wxyz")

                # Apply orientation sensitivity using SLERP
                # ori_sensitivity=1.0: full tracking, ori_sensitivity=0.5: half speed
                if self.config.ori_sensitivity < 1.0:
                    # SLERP between start orientation and full target
                    self._target_quat = self._slerp_quaternion(
                        self._start_quat, full_target_quat, self.config.ori_sensitivity
                    )
                else:
                    self._target_quat = full_target_quat
            # If orientation control is disabled, target_quat stays at the value when grip was pressed
        # When not enabled, target pose stays at last position (no update)

        # Step 5.4: Quaternion hemisphere continuity.
        # q and -q encode the same rotation, but a naive slerp between them
        # walks the long-way (~360°) around the unit sphere. After grip
        # release/re-engage REF_RESET, _target_quat = controller * offset can
        # land on the opposite hemisphere from _prev_target_quat even though
        # the rotation itself barely moved. The downstream robot RT thread
        # interpolates at 1 kHz and will trip a safety fault on that ghost
        # 360° spin, killing the RT thread silently. Force continuity so
        # consecutive frames stay on the same hemisphere.
        if self._prev_target_quat is not None and np.dot(self._target_quat, self._prev_target_quat) < 0.0:
            self._target_quat = -self._target_quat

        # Step 5.5: Output rate limiter — clamp _target_pos / _target_quat
        # velocity to physically plausible human hand speed.
        now = time.time()
        if self._prev_target_pos is not None and self._last_action_time is not None:
            dt = now - self._last_action_time
            if dt > 0:
                # --- position rate limit ---
                if self.config.max_pos_velocity > 0:
                    max_delta = self.config.max_pos_velocity * dt
                    delta_pos = self._target_pos - self._prev_target_pos
                    delta_norm = np.linalg.norm(delta_pos)
                    if delta_norm > max_delta:
                        self.logger.warn(
                            f"[RATE_LIMIT] Position velocity {delta_norm / dt:.2f} m/s "
                            f"exceeds limit {self.config.max_pos_velocity} m/s, clamping."
                        )
                        self._target_pos = self._prev_target_pos + delta_pos * (max_delta / delta_norm)

                # --- orientation rate limit ---
                if self.config.max_rot_velocity > 0:
                    max_angle = self.config.max_rot_velocity * dt
                    # Angle between two quaternions: θ = 2 * arccos(|q1 · q2|)
                    dot = np.clip(abs(np.dot(self._target_quat, self._prev_target_quat)), 0.0, 1.0)
                    angle = 2.0 * np.arccos(dot)
                    if angle > max_angle:
                        self.logger.warn(
                            f"[RATE_LIMIT] Rotation velocity {np.degrees(angle / dt):.1f} deg/s "
                            f"exceeds limit {np.degrees(self.config.max_rot_velocity):.1f} deg/s, clamping."
                        )
                        t = max_angle / angle
                        self._target_quat = self._slerp_quaternion(self._prev_target_quat, self._target_quat, t)

        self._prev_target_pos = self._target_pos.copy()
        self._prev_target_quat = self._target_quat.copy()
        self._last_action_time = now

        # Step 6: Update gripper position from trigger value
        # Trigger [0, 1] maps to gripper position in [1 - gripper_width, 1].
        # Gripper convention: 1.0 = open, 0.0 = closed. So trigger=0 -> open (1.0),
        # trigger=1 -> closed (1 - gripper_width).
        self._target_gripper_pos = 1.0 - float(controller_trigger) * self.config.gripper_width

        # Step 7: Return in world-frame TCP action format with 6D rotation
        r6d = quaternion_to_rotation_6d(
            self._target_quat[0], self._target_quat[1], self._target_quat[2], self._target_quat[3]
        )
        self.logger.debug(
            f"[ACTION] pos=[{self._target_pos[0]:.4f}, {self._target_pos[1]:.4f}, {self._target_pos[2]:.4f}], "
            f"quat=[{self._target_quat[0]:.3f}, {self._target_quat[1]:.3f}, {self._target_quat[2]:.3f}, {self._target_quat[3]:.3f}], "
            f"gripper={self._target_gripper_pos:.3f}, enabled={self._enabled}, ori_active={self._orientation_control_active}"
        )
        return {
            "tcp.x": self._target_pos[0],
            "tcp.y": self._target_pos[1],
            "tcp.z": self._target_pos[2],
            "tcp.r1": r6d[0],
            "tcp.r2": r6d[1],
            "tcp.r3": r6d[2],
            "tcp.r4": r6d[3],
            "tcp.r5": r6d[4],
            "tcp.r6": r6d[5],
            "gripper.pos": self._target_gripper_pos,
        }

    def get_target_pose_array(self) -> tuple[np.ndarray, float]:
        """
        Get the current target pose as numpy array (for direct use with the robot SDK).

        Returns:
            Tuple of (tcp_pose, gripper_pos) where tcp_pose is [x, y, z, qw, qx, qy, qz] in world frame
        """
        # _target_quat is in [qw, qx, qy, qz] format
        tcp_pose = np.array(
            [
                self._target_pos[0],
                self._target_pos[1],
                self._target_pos[2],
                self._target_quat[0],  # qw
                self._target_quat[1],  # qx
                self._target_quat[2],  # qy
                self._target_quat[3],  # qz
            ],
            dtype=np.float32,
        )
        return tcp_pose, self._target_gripper_pos

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """Pico4 doesn't support feedback."""
        raise NotImplementedError("Feedback is not implemented for Pico4 teleoperator.")

    def poll_buttons(self) -> None:
        """Refresh cached physical button states without computing a full action."""
        if not self._is_connected or self._xrt is None:
            return

        if self.config.use_right_controller:
            self._last_a_button = bool(self._xrt.get_A_button())
            self._last_b_button = bool(self._xrt.get_B_button())
        if self.config.use_left_controller:
            self._last_x_button = bool(self._xrt.get_X_button())
            self._last_y_button = bool(self._xrt.get_Y_button())

    def get_reset_button(self) -> bool:
        """Get the state of the reset button with edge detection.

        Only returns True on the rising edge (button just pressed), not while held.
        This prevents multiple resets when the button is held down.

        Note: This uses the cached A button state from the last get_action()
        or poll_buttons() call to avoid additional SDK calls.

        Returns:
            True if reset button was just pressed (rising edge), False otherwise.
        """
        # Use cached button state from get_action() / poll_buttons() to avoid extra SDK calls
        current_pressed = self._last_a_button

        # Edge detection: only trigger on rising edge (was not pressed, now pressed)
        just_pressed = current_pressed and not self._was_reset_button_pressed
        self._was_reset_button_pressed = current_pressed

        return just_pressed

    def disconnect(self) -> None:
        """Disconnect from the Pico4 VR headset."""
        if not self._is_connected or self._xrt is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.logger.info("Closing XenseVR SDK...")
        try:
            self._xrt.close()
        except RuntimeError as e:
            raise RuntimeError(f"Failed to close XenseVR SDK: {e}") from e
        finally:
            self._is_connected = False
            self._xrt = None
            self.logger.info(f"{self} disconnected.")

    def __del__(self):
        """Cleanup on deletion."""
        if self._is_connected:
            try:
                self.disconnect()
            # No logging: __del__ can run during interpreter shutdown, where the
            # logger's sinks may already be gone.
            except Exception:  # nosec B110
                pass
            finally:
                self._is_connected = False
                self._xrt = None
                self.logger.info(f"{self} disconnected.")
