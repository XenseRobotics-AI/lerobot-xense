"""Calibrate B601 motor zeros sequentially: right arm first, then left arm.

Run from the lerobot-xense repository root with rebot_rt and motorbridge installed:
    python -m lerobot.robots.bi_rebot_b601_rt.bi_calibration \
        --config_path=recipes/teleop/bi_rebot_b601_rt/dm-taccap00-09.yaml

    python -m lerobot.robots.bi_rebot_b601_rt.bi_calibration \
        --config_path=recipes/teleop/bi_rebot_b601_rt/dm-taccap00-10.yaml

The script reads the robot block of a dual-arm teleop or recording recipe. Camera,
teleoperator, and dataset settings are ignored; no RT controller or peripheral is started.
For platform 09, plug in the right DM adapter first, wait for enumeration, then plug
in the left adapter. Keep both connected throughout calibration. Auto-selected DM
ports are resolved once and checked again before each arm's bus is opened.

Stop other programs using either arm and support the arm being calibrated: its
motors will be disabled. Follow the prompts to place the RIGHT arm at the mechanical
zero pose specified in the B601 manual, then press ENTER. After its zero readbacks
pass and its record is saved, repeat for the LEFT arm. The mechanical zero pose is
not reset_position_rad. Separate USB/TacCap grippers are excluded by has_gripper=false.

Existing records prompt separately for each arm: ENTER keeps that record and moves
on; c followed by ENTER recalibrates that arm. A failure or Ctrl+C stops the sequence.
If the left arm fails, the completed right-arm calibration remains saved. Hardware
zero writes cannot be rolled back by this script; inspect any reported partial failure.

Record IDs match the dual-arm driver: <robot.id>_left and <robot.id>_right, under the
single-arm rebot_b601_rt calibration directory and each arm's dm/rs subdirectory.
Use --calibration_dir=/path/to/calibration to override the recipe's record directory.
"""

import argparse
from dataclasses import replace
from pathlib import Path

import draccus
import yaml

from lerobot.robots import RobotConfig
from lerobot.robots.bi_rebot_b601_rt.config_bi_rebot_b601_rt import BiRebotB601RTConfig
from lerobot.robots.rebot_b601_rt.calibration import calibrate_b601
from lerobot.robots.rebot_b601_rt.dm_serial_discovery import (
    AUTO_DM_CHANNELS,
    discover_dm_serial_devices,
    resolve_dm_serial_channel,
)
from lerobot.utils.robot_utils import get_logger

logger = get_logger("bi_b601_calibration")


def calibrate_bi_b601(config: BiRebotB601RTConfig) -> None:
    """Run the existing single-arm procedure for the right arm, then the left."""
    if not config.id:
        raise ValueError("Set robot.id to identify the left and right calibration records")
    arms = {
        side: replace(arm, id=f"{config.id}_{side}", calibration_dir=config.calibration_dir)
        for side, arm in (("right", config.right_arm_config), ("left", config.left_arm_config))
    }
    devices = None
    if any(arm.channel in AUTO_DM_CHANNELS for arm in arms.values()):
        devices = discover_dm_serial_devices()
        arms = {
            side: replace(arm, channel=resolve_dm_serial_channel(arm.channel, devices)) for side, arm in arms.items()
        }
    if Path(arms["left"].channel).resolve() == Path(arms["right"].channel).resolve():
        raise ValueError("Both arms resolve to the same communication device")
    for side, arm in arms.items():
        logger.info(f"Calibrating {side.upper()} arm: family={arm.motor_family}, channel={arm.channel}, id={arm.id}")
        calibrate_b601(arm, dm_serial_devices=devices)
    logger.info("Right and left arm calibration procedures completed")


def load_calibration_config(path: Path) -> BiRebotB601RTConfig:
    """Read only the robot configuration; calibration does not require cameras."""
    with path.open() as stream:
        recipe = yaml.safe_load(stream)
    robot = recipe.get("robot") if isinstance(recipe, dict) else None
    if not isinstance(robot, dict) or robot.get("type") != "bi_rebot_b601_rt":
        raise ValueError("Expected a recipe with robot.type=bi_rebot_b601_rt")
    robot["cameras"] = {}
    for side in ("left", "right"):
        arm = robot.get(f"{side}_arm_config")
        if not isinstance(arm, dict):
            raise ValueError(f"Missing robot.{side}_arm_config")
        arm["cameras"] = {}
    return draccus.decode(RobotConfig, robot)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate B601 motor zeros: right arm first, then left arm.")
    parser.add_argument("--config_path", required=True, type=Path, help="Dual-arm teleop or recording YAML recipe")
    parser.add_argument("--calibration_dir", type=Path, help="Override the recipe's calibration record directory")
    args = parser.parse_args()
    config = load_calibration_config(args.config_path)
    if args.calibration_dir is not None:
        config = replace(config, calibration_dir=args.calibration_dir)
    calibrate_bi_b601(config)


if __name__ == "__main__":
    main()
