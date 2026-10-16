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

from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("bi_pico4")
@dataclass
class BiPico4Config(TeleoperatorConfig):
    """Configuration for BiPico4 dual-controller VR teleoperator.

    Uses both Pico4 VR controllers simultaneously:
    - Left controller  -> left arm  (left_tcp.*, left_gripper.pos)
    - Right controller -> right arm (right_tcp.*, right_gripper.pos)

    Output keys are the shared world-frame TCP schema (BiFlexivRizon4RT, BiEliteCS66RT).

    Attributes:
        pos_sensitivity: Position scale factor (1.0 = 1:1 mapping).
        ori_sensitivity: Orientation scale factor (1.0 = full tracking).
        filter_window_size: Moving average filter window for smoothing pose.
        left_gripper_width: Max gripper position for left arm control space.
        right_gripper_width: Max gripper position for right arm control space.
        grip_enable_threshold: Grip value to start control (hysteresis high).
        grip_disable_threshold: Grip value to stop control (hysteresis low).
        orientation_offset_warning_deg: Max allowed controller-robot orientation
            difference at enable time. If exceeded, orientation control is disabled.
        position_jump_threshold: Max allowed position change per frame (m).
    """

    id: str = "bi_pico4"

    pos_sensitivity: float = 1.0
    ori_sensitivity: float = 1.0
    filter_window_size: int = 1

    # Per-arm output rate limiter: caps how fast the target pose may translate/
    # rotate regardless of controller input. Lower these for arms with tighter
    # joint-velocity limits (e.g. Elite CS66) so VR jitter can't drive the
    # controller into a protective stop. 0 disables the cap.
    max_pos_velocity: float = 1.0  # m/s
    max_rot_velocity: float = 6.28  # rad/s

    left_gripper_width: float = 1.0
    right_gripper_width: float = 1.0

    # On each grip press, restart the mapping from the robot's measured TCP
    # instead of the last target, when the robot loop provides it
    # (set_grip_anchor_source). After a collision hold the last target sits
    # behind the obstacle; re-gripping then moves the arm away immediately.
    grip_anchor_to_robot: bool = True
    grip_enable_threshold: float = 0.5
    grip_disable_threshold: float = 0.3
    orientation_offset_warning_deg: float = 180.0
    position_jump_threshold: float = 0.1
