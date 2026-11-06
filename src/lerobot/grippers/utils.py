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

"""Factory that turns a gripper config into its driver.

Replaces the ``_make_gripper`` isinstance-chain that was copy-pasted into each
bimanual arm. Imports are deferred into each branch on purpose: the Xense and
TacCap SDKs are optional builds, so importing this module must not drag in a
native extension the host may not have. Only the branch you actually select
gets imported.
"""

from .configs import GripperConfig
from .gripper import Gripper


def make_gripper_from_config(config: GripperConfig | None) -> Gripper | None:
    """Instantiate the gripper driver matching ``config``.

    Returns ``None`` when ``config`` is ``None`` — arms use that to mean
    "no gripper on this side", so callers can pass the field straight through.

    Raises:
        ValueError: If the config's registered type has no known driver.
    """
    if config is None:
        return None

    gripper_type = config.type

    if gripper_type == "serial":
        from .serial import SerialGripper

        return SerialGripper(config)

    elif gripper_type == "taccap_follower":
        from .taccap import TaccapFollower

        return TaccapFollower(config)

    elif gripper_type == "dh-ag95":
        from .dh_ag95 import DHAG95Gripper

        return DHAG95Gripper(config)

    raise ValueError(f"Unknown gripper type {gripper_type!r} (config: {config}).")


def command_gripper_position(gripper: Gripper, position: float) -> float:
    """Submit a position and return the applied target for action recording.

    AG-95 retains the tested Dobot binary action mapping. Existing backends
    retain their continuous targets and do not need a new public method.
    """
    if gripper.config.type == "dh-ag95":
        position = gripper.config.normalize_position(position)
    gripper.set_gripper_position(position)
    return position
