"""Two independently scheduled C++ B601 arms."""

from dataclasses import dataclass, field

from lerobot.cameras.configs import CameraConfig

from ..config import RobotConfig
from ..rebot_b601_rt.config_rebot_b601_rt import RebotB601RTConfig


@RobotConfig.register_subclass("bi_rebot_b601_rt")
@dataclass
class BiRebotB601RTConfig(RobotConfig):
    left_arm_config: RebotB601RTConfig
    right_arm_config: RebotB601RTConfig
    reset_on_exit: bool = False  # disconnect: return both arms to zero unless reset=False
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        if self.left_arm_config.channel == self.right_arm_config.channel:
            raise ValueError("Each arm needs its own SocketCAN interface or DM serial device")
        names = set(self.cameras) & (set(self.left_arm_config.cameras) | set(self.right_arm_config.cameras))
        if names:
            raise ValueError(f"Camera names collide: {sorted(names)}")
