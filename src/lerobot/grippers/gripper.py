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

"""Base class for gripper implementations.

Grippers are arm-agnostic: an arm robot holds one (or one per side) and drives it
through the small contract below, without knowing which backend is underneath.
Backends include ``serial`` (USB-serial parallel jaw), ``taccap_follower``
(the centric TacCap gripper), and ``dh-ag95`` (Modbus RTU via an injected transport).

Normalized position convention, uniform across every backend:

    0.0  →  fully closed
    1.0  →  fully open

Subclasses must implement connection lifecycle, ``is_connected`` and the two
position methods. ``initialize_gripper_position`` has a working default — command
a target, then poll until it is reached — that a backend overrides only if its SDK
offers a genuine blocking move.

Grippers do not carry tactile data. The visuotactile sensors on a TacCap gripper
are surfaced as ordinary cameras (auto-discovered off the gripper's USB hub), not
through the gripper object.
"""

import abc
import time

from lerobot.utils.robot_utils import get_logger

from .configs import GripperConfig

# Defaults for the blocking reach-wait in initialize_gripper_position().
DEFAULT_INIT_POSITION_TOLERANCE = 0.03
DEFAULT_INIT_TIMEOUT_S = 3.0


class Gripper(abc.ABC):
    """Base class for gripper implementations.

    Attributes:
        config: The backend-specific configuration this gripper was built from.
        logger: Per-instance logger, named after the concrete class.
    """

    config_class: type[GripperConfig]

    def __init__(self, config: GripperConfig):
        self.config = config
        self.logger = get_logger(self.__class__.__name__)

    def __str__(self) -> str:
        return self.__class__.__name__

    def __enter__(self):
        """Context manager entry — connects the gripper."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Context manager exit — disconnects, even on error."""
        self.disconnect()

    def __del__(self) -> None:
        """Destructor safety net for a gripper collected without teardown."""
        try:
            if self.is_connected:
                self.disconnect()
        except Exception:  # nosec B110 — best-effort only
            pass

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """True when the gripper is connected and ready to read/command."""

    @abc.abstractmethod
    def connect(self) -> None:
        """Open the device and make it ready for commands.

        Raises:
            DeviceAlreadyConnectedError: If already connected.
        """

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Release the device.

        Raises:
            DeviceNotConnectedError: If not connected.
        """

    # ── Position ───────────────────────────────────────────────────────────────

    @abc.abstractmethod
    def get_gripper_position(self) -> float:
        """Return the backend's jaw position, normalized to [0, 1].

        DH AG-95 with position polling disabled reports the last acknowledged
        command, as in the validated Dobot setup; see that backend's README.

        Raises:
            DeviceNotConnectedError: If not connected.
        """

    @abc.abstractmethod
    def set_gripper_position(self, normalized_pos: float) -> None:
        """Command a jaw target, normalized to [0, 1]. Non-blocking.

        Raises:
            DeviceNotConnectedError: If not connected.
            ValueError: If ``normalized_pos`` is outside [0, 1].
        """

    def initialize_gripper_position(
        self,
        normalized_pos: float,
        tolerance: float = DEFAULT_INIT_POSITION_TOLERANCE,
        timeout: float = DEFAULT_INIT_TIMEOUT_S,
    ) -> None:
        """Command a target and block until the jaw reaches it (or timeout).

        Used by the arms during homing. The default implementation polls; a backend
        whose SDK exposes a genuine blocking move should override it.

        A timeout is logged, not raised: a jaw held short of target by a
        feed-forward clamp bias is an expected steady state, not a failure.
        """
        self.set_gripper_position(normalized_pos)
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            if abs(self.get_gripper_position() - normalized_pos) <= tolerance:
                return
            time.sleep(0.02)
        self.logger.warn(
            f"Gripper did not reach init target {normalized_pos:.3f} within {timeout:.1f}s "
            f"(current={self.get_gripper_position():.3f})."
        )
