# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
Helper to recalibrate your device (robot or teleoperator).

Example:

```shell
lerobot-calibrate \
    --teleop.type=so100_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=blue
```
"""

from dataclasses import asdict, dataclass
from pprint import pformat

import draccus

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    make_robot_from_config,
)
from lerobot.robots.bi_rebot_b601_rt import BiRebotB601RTConfig
from lerobot.robots.rebot_b601_rt import RebotB601RTConfig
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    make_teleoperator_from_config,
    mock_teleop,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import get_logger
from lerobot.utils.utils import init_logging

logger = get_logger("lerobot_calibrate")


@dataclass
class CalibrateConfig:
    teleop: TeleoperatorConfig | None = None
    robot: RobotConfig | None = None

    def __post_init__(self):
        if bool(self.teleop) == bool(self.robot):
            raise ValueError("Choose either a teleop or a robot.")

        self.device = self.robot if self.robot else self.teleop


@draccus.wrap()
def calibrate(cfg: CalibrateConfig):
    init_logging()
    logger.info(pformat(asdict(cfg)))

    if isinstance(cfg.device, RebotB601RTConfig):
        from lerobot.robots.rebot_b601_rt.calibration import calibrate_b601

        calibrate_b601(cfg.device)
        return
    if isinstance(cfg.device, BiRebotB601RTConfig):
        # The generic path would connect (enabling and homing both arms) only for
        # BiRebotB601RT.calibrate() to refuse; zeroing must never move an arm.
        from lerobot.robots.bi_rebot_b601_rt.bi_calibration import calibrate_bi_b601

        calibrate_bi_b601(cfg.device)
        return

    if isinstance(cfg.device, RobotConfig):
        device = make_robot_from_config(cfg.device)
    elif isinstance(cfg.device, TeleoperatorConfig):
        device = make_teleoperator_from_config(cfg.device)

    device.connect(calibrate=False)

    try:
        device.calibrate()
    finally:
        device.disconnect()


def main():
    register_third_party_plugins()
    calibrate()


if __name__ == "__main__":
    main()
