"""Bimanual LeRobot facade over two native 100 Hz controllers."""

import time
from dataclasses import replace
from functools import cached_property
from pathlib import Path
from typing import Any

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.robot_utils import best_effort, get_logger

from ..rebot_b601_rt import RebotB601RT
from ..rebot_b601_rt.dm_serial_discovery import (
    AUTO_DM_CHANNELS,
    discover_dm_serial_devices,
    resolve_dm_serial_channel,
)
from ..robot import Robot
from .config_bi_rebot_b601_rt import BiRebotB601RTConfig

RobotAction = dict[str, Any]
RobotObservation = dict[str, Any]


class BiRebotB601RT(Robot):
    config_class = BiRebotB601RTConfig
    name = "bi_rebot_b601_rt"
    supports_position_hold = True

    def __init__(self, config: BiRebotB601RTConfig):
        super().__init__(config)
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self._top_camera_names = set(config.cameras)
        left_config = replace(
            config.left_arm_config,
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            cameras={**config.left_arm_config.cameras, **config.cameras},
        )
        right_config = replace(
            config.right_arm_config,
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
        )
        self.left_arm = RebotB601RT(left_config)
        self.right_arm = RebotB601RT(right_config)
        # Key cameras as get_observation() keys their frames, so a camera name
        # used on both arms (e.g. a per-arm "wrist") is kept twice, not overwritten.
        self.cameras = {
            **{k if k in self._top_camera_names else f"left_{k}": v for k, v in self.left_arm.cameras.items()},
            **{f"right_{k}": v for k, v in self.right_arm.cameras.items()},
        }

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        raise RuntimeError("B601 motor zero offsets must be set with the vendor calibration tool")

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {
            **{f"left_{k}": v for k, v in self.left_arm.action_features.items()},
            **{f"right_{k}": v for k, v in self.right_arm.action_features.items()},
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {
            **{
                k if k in self._top_camera_names else f"left_{k}": v
                for k, v in self.left_arm.observation_features.items()
            },
            **{f"right_{k}": v for k, v in self.right_arm.observation_features.items()},
        }

    @check_if_already_connected
    def connect(self, calibrate: bool = True, go_to_start: bool = True) -> None:
        connect_kwargs = {}
        channels = [self.left_arm.config.channel, self.right_arm.config.channel]
        if any(channel in AUTO_DM_CHANNELS for channel in channels):
            # Resolve both from the same snapshot before either arm is enabled.
            devices = discover_dm_serial_devices()
            left, right = (resolve_dm_serial_channel(channel, devices) for channel in channels)
            if Path(left).resolve() == Path(right).resolve():
                raise ValueError("Both arms resolve to the same communication device")
            connect_kwargs["_dm_serial_devices"] = devices
            self.logger.info(f"DM serial mapping: left={left}, right={right}; plug right first, then left")
        try:
            # Connect both arms before either begins its startup movement.
            self.left_arm.connect(calibrate=calibrate, go_to_start=False, **connect_kwargs)
            self.right_arm.connect(calibrate=calibrate, go_to_start=False, **connect_kwargs)
            if go_to_start and any(
                arm.config.reset_position_rad is not None for arm in (self.left_arm, self.right_arm)
            ):
                self.logger.info("B601 startup reset: moving both arms to the configured initial poses")
                self.reset_to_initial_position()
                while self.rt_moving:
                    time.sleep(0.01)
        except BaseException as exc:
            # A user interrupt or settling timeout after both arms connected
            # still requires the configured return to zero.
            self.disconnect(reset=go_to_start and isinstance(exc, (KeyboardInterrupt, TimeoutError)))
            raise

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        left = {
            k if k in self._top_camera_names else f"left_{k}": v for k, v in self.left_arm.get_observation().items()
        }
        right = {f"right_{k}": v for k, v in self.right_arm.get_observation().items()}
        return {**left, **right}

    @check_if_not_connected
    def get_current_tcp_pose_quat(self):
        """Each arm's measured TCP in its own base frame, for Bi-Pico4 sync."""
        return self.left_arm.get_current_tcp_pose_quat(), self.right_arm.get_current_tcp_pose_quat()

    @check_if_not_connected
    def get_commanded_tcp_pose_quat(self):
        """Each arm's current commanded TCP, for reset-to-teleop handoff."""
        return self.left_arm.get_commanded_tcp_pose_quat(), self.right_arm.get_commanded_tcp_pose_quat()

    @check_if_not_connected
    def reset_to_initial_position(self) -> None:
        if any(arm.config.reset_position_rad is None for arm in (self.left_arm, self.right_arm)):
            raise ValueError("Both arms need reset_position_rad before using the reset button")
        self._reset_to_positions(self.left_arm.config.reset_position_rad, self.right_arm.config.reset_position_rad)

    def _reset_to_positions(
        self, left_position: list[float], right_position: list[float], *, exit_reset: bool = False
    ) -> None:
        # Check both controllers before either arm starts moving.
        self.left_arm._validate_reset()
        self.right_arm._validate_reset()
        self.left_arm._reset_to_position(
            left_position,
            tolerance_rad=self.left_arm.config.exit_reset_tolerance_rad if exit_reset else None,
        )
        try:
            self.right_arm._reset_to_position(
                right_position,
                tolerance_rad=self.right_arm.config.exit_reset_tolerance_rad if exit_reset else None,
            )
        except BaseException:
            with best_effort(self.left_arm.logger, "cancelling the left arm reset"):
                self.left_arm._cancel_reset()
            raise

    @property
    def rt_moving(self) -> bool:
        # Evaluate both properties: each refreshes its native reset watchdog.
        left = self.left_arm.rt_moving
        right = self.right_arm.rt_moving
        return left or right

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        left = {k.removeprefix("left_"): v for k, v in action.items() if k.startswith("left_")}
        right = {k.removeprefix("right_"): v for k, v in action.items() if k.startswith("right_")}
        left_sent = self.left_arm.send_action(left)
        right_sent = self.right_arm.send_action(right)
        return {
            **{f"left_{k}": v for k, v in left_sent.items()},
            **{f"right_{k}": v for k, v in right_sent.items()},
        }

    def disconnect(self, *, reset: bool = True) -> None:
        """Return both arms to zero together, then stop controllers before peripheral cleanup."""
        arms = (self.left_arm, self.right_arm)
        try:
            # Gate on the arm controllers only, not cameras/grippers. Like a fault on
            # either arm, a lost controller on either arm skips the coordinated return.
            if reset and self.config.reset_on_exit and all(arm.arm_connected for arm in arms):
                self.logger.info("B601 exit reset: returning both arms to zero; press Ctrl+C again to abort")
                self._reset_to_positions([0.0] * 6, [0.0] * 6, exit_reset=True)
                while self.rt_moving:
                    time.sleep(0.01)
                self.logger.info("B601 exit reset complete; disconnecting both arms")
            elif reset and self.config.reset_on_exit and any(arm._native is not None for arm in arms):
                lost = [side for side, arm in zip(("left", "right"), arms, strict=True) if not arm.arm_connected]
                self.logger.warn(f"B601 exit reset skipped: native controller not connected on {lost} arm(s)")
        except KeyboardInterrupt:
            self.logger.warn("B601 exit reset interrupted; disconnecting both arms immediately")
        except Exception as error:
            self.logger.error(f"B601 exit reset failed; disconnecting both arms: {error}")
        finally:
            # Stop both controllers before either arm's slower camera/gripper teardown.
            for arm in arms:
                with best_effort(arm.logger, "stopping the native arm"):
                    if arm._native is not None:
                        arm._native.stop()
            try:
                self.right_arm.disconnect(reset=False)
            finally:
                self.left_arm.disconnect(reset=False)
