"""Elite-only Pico4 adapters for measured-TCP grip anchoring.

The CLI creates these readers only for single/bimanual Elite + Pico4 pairs.
The shared Pico4 classes and their configuration remain unchanged.
"""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np

from lerobot.teleoperators.bi_pico4.bi_pico4 import BiPico4
from lerobot.teleoperators.bi_pico4.config_bi_pico4 import BiPico4Config
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.pico4.config_pico4 import Pico4Config
from lerobot.teleoperators.pico4.teleop_pico4 import Pico4
from lerobot.utils.robot_utils import normalize_quaternion

if TYPE_CHECKING:
    from lerobot.robots.robot import Robot

TCPPose = Sequence[float] | np.ndarray


class ElitePico4(Pico4):
    """Restart each grip gesture from this Elite arm's measured world-frame TCP."""

    def __init__(self, config: Pico4Config, source: Callable[[], TCPPose]):
        super().__init__(config)
        self._grip_anchor_source = source

    def _transform_pico_to_world_coordinate(self, pos: np.ndarray, quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        world_pose = super()._transform_pico_to_world_coordinate(pos, quat)
        # Pico4.get_action updates grip hysteresis before this transform, then
        # captures the target/reference immediately afterwards. This lets the
        # Elite reader anchor on that same sample without reading the SDK twice
        # or duplicating the shared filtering, mapping and rate-limiting loop.
        if self._enabled and not self._was_enabled:
            self._anchor_target_to_robot()
        return world_pose

    def _anchor_target_to_robot(self) -> None:
        try:
            pose = np.asarray(self._grip_anchor_source(), dtype=np.float64)
        except Exception as exc:
            self.logger.warn(f"[ANCHOR] Could not read the robot TCP, keeping the last target: {exc}")
            return
        if pose.ndim != 1 or pose.size < 7 or not np.isfinite(pose[:7]).all():
            self.logger.warn(f"[ANCHOR] Invalid robot TCP {pose}, keeping the last target")
            return
        self.logger.debug(f"[ANCHOR] grip pressed: target {self._target_pos} -> robot TCP {pose[:3]}")
        self._target_pos = pose[:3].astype(np.float32)
        self._target_quat = normalize_quaternion(pose[3:7], input_format="wxyz")
        # Accept the measured pose immediately, without slewing from the old target.
        self._prev_target_pos = self._target_pos.copy()
        self._prev_target_quat = self._target_quat.copy()


class EliteBiPico4(BiPico4):
    """Keep each arm's anchor independent while sharing BiPico4's SDK lifecycle."""

    def __init__(self, config: BiPico4Config, source: Callable[[], tuple[TCPPose, TCPPose]]):
        super().__init__(config)
        # Replace only this instance's disconnected readers. The inherited
        # connect/pre_init/reset/disconnect methods continue to own both arms.
        self._left_pico4 = ElitePico4(self._left_pico4.config, lambda: source()[0])
        self._right_pico4 = ElitePico4(self._right_pico4.config, lambda: source()[1])


def make_elite_pico4_teleoperator(config: TeleoperatorConfig, robot: "Robot") -> Pico4 | BiPico4:
    """Construct an Elite-local reader, or the original reader when disabled.

    ``robot.grip_anchor_to_robot`` in the recipe controls this Elite feature;
    no field or callback API is added to the public Pico4 configurations/classes.
    The measured-pose callback is only called on grip presses after connection.
    """
    if robot.config.type == "elite_cs66_rt" and isinstance(config, Pico4Config):
        if not robot.config.grip_anchor_to_robot:
            return Pico4(config)
        return ElitePico4(config, robot.get_current_tcp_pose_quat)
    if robot.config.type == "bi_elite_cs66_rt" and isinstance(config, BiPico4Config):
        if not robot.config.grip_anchor_to_robot:
            return BiPico4(config)
        return EliteBiPico4(config, robot.get_current_tcp_pose_quat)
    raise ValueError(f"Elite Pico4 adapter does not support {robot.config.type} + {config.type}")
