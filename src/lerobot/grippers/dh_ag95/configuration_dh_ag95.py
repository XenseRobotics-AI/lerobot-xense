#!/usr/bin/env python
# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""AG-95 settings, taken from the tested BiDobotNova5DH robot configuration.

The robot-level overrides there are 33% force, a 50 Hz worker, NO periodic
position reads, and binary actions at 0.5. They override the older standalone
driver defaults (15%, 100 Hz, 20 Hz). See README.md for the provenance.
"""

import math
from dataclasses import dataclass

from ..configs import GripperConfig


@GripperConfig.register_subclass("dh-ag95")
@dataclass(kw_only=True)
class DHAG95GripperConfig(GripperConfig):
    side: str | None = None
    slave_id: int = 1
    baudrate: int = 115200
    gripper_force: int = 33
    init_open: bool = True
    init_timeout: float = 10.0
    worker_frequency: float = 50.0
    position_poll_frequency: float = 0.0
    binary_gripper_actions: bool = True
    gripper_open_threshold: float = 0.5
    auto_discover_cameras: bool = False
    enable_tactile: bool = False

    def __post_init__(self):
        if self.side not in (None, "left", "right"):
            raise ValueError("DH AG-95 side must be left, right, or None")
        if not isinstance(self.slave_id, int) or not 1 <= self.slave_id <= 247:
            raise ValueError("DH AG-95 slave_id must be in [1, 247]")
        if self.baudrate not in (9600, 19200, 38400, 115200):
            raise ValueError("DH AG-95 baudrate must be 9600, 19200, 38400, or 115200")
        if not isinstance(self.gripper_force, int) or not 20 <= self.gripper_force <= 100:
            raise ValueError("DH AG-95 gripper_force must be an integer in [20, 100] percent")
        for name in (
            "init_timeout",
            "worker_frequency",
            "position_poll_frequency",
            "gripper_open_threshold",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"DH AG-95 {name} must be finite")
        if self.init_timeout <= 0 or self.worker_frequency < 10:
            raise ValueError("DH AG-95 init_timeout must be positive and worker_frequency at least 10 Hz")
        if not 0 <= self.position_poll_frequency <= self.worker_frequency:
            raise ValueError("DH AG-95 position_poll_frequency must be in [0, worker_frequency]")
        if not 0 <= self.gripper_open_threshold <= 1:
            raise ValueError("DH AG-95 gripper_open_threshold must be in [0, 1]")
        if self.auto_discover_cameras or self.enable_tactile:
            raise ValueError("DH AG-95 has no camera discovery; configure external cameras on the robot")

    def normalize_position(self, position: float) -> float:
        """Apply the tested Dobot action mapping, without admitting NaN/inf."""
        if not math.isfinite(position) or not 0 <= position <= 1:
            raise ValueError(f"Gripper position must be finite and in [0, 1], got {position}")
        if self.binary_gripper_actions:
            return float(position >= self.gripper_open_threshold)
        return position
