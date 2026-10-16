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

from dataclasses import dataclass, field  # noqa: F401

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("pico4")
@dataclass
class Pico4Config(TeleoperatorConfig):
    """Configuration for Pico4 VR teleoperator.

    This teleoperator provides 6-DoF absolute pose control using VR controllers,
    suitable for end-effector teleoperation. The output is an accumulated target_pose_6d
    that can be directly sent to a Cartesian controller.

    Attributes:
        use_left_controller: Whether to use the left controller for teleoperation.
        use_right_controller: Whether to use the right controller for teleoperation.
        pos_sensitivity: Sensitivity multiplier for position control (scales delta position).
                         1.0 = 1:1 mapping, 0.5 = half speed.
        ori_sensitivity: Sensitivity multiplier for orientation control.
                         1.0 = full tracking, 0.5 = half speed rotation.
        filter_window_size: Moving average filter window size for smoothing pose changes.
        gripper_width: Maximum gripper travel in control space [0, 1] (for clamping).
        grip_enable_threshold: Grip value (0-1) that must be exceeded to enable control.
        grip_disable_threshold: Grip value (0-1) that control must drop below to disable.
        orientation_offset_warning_deg: Warning threshold in degrees for orientation offset at enable.
                                        If controller-robot orientation difference exceeds this,
                                        a warning is logged and orientation control is disabled.
    """

    id: str = "pico4"  # Default id for Pico4 teleoperator

    use_left_controller: bool = False
    use_right_controller: bool = True
    pos_sensitivity: float = 1.0  # Scale factor for position delta (1.0 = 1:1 mapping, 0.5 = half speed)
    ori_sensitivity: float = 1.0  # Scale factor for orientation delta (1.0 = 1:1 mapping)
    filter_window_size: int = 1  # Moving average filter window size
    gripper_width: float = 1.0  # Maximum gripper position in control space [0, 1]
    # On each grip press, restart the mapping from the robot's measured TCP
    # instead of the last target, when the robot loop provides it
    # (set_grip_anchor_source). After a collision hold the last target sits
    # behind the obstacle; re-gripping then moves the arm away immediately.
    grip_anchor_to_robot: bool = True
    grip_enable_threshold: float = 0.5  # Threshold for grip to enable control (must exceed to enable)
    grip_disable_threshold: float = 0.3  # Threshold for grip to disable control (must drop below to disable)
    orientation_offset_warning_deg: float = (
        180.0  # Warning threshold for orientation offset (degrees). Set to 180 to disable check.
    )
    position_jump_threshold: float = (
        0.1  # Max allowed position change per frame (meters). Larger jumps are filtered out.
    )
    max_pos_velocity: float = 1.0  # Max allowed position velocity (m/s) for output rate limiter. 0 = disabled.
    max_rot_velocity: float = 6.28  # Max allowed angular velocity (rad/s) for output rate limiter. 0 = disabled.
