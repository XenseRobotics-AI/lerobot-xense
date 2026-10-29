"""Cartesian C++ RT driver configuration for the DM and RS B601 arms."""

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

from lerobot.cameras.configs import CameraConfig
from lerobot.grippers import GripperConfig
from lerobot.robots.config import RobotConfig

from .dm_serial_discovery import AUTO_DM_CHANNELS

logger = logging.getLogger(__name__)

DEFAULT_DM_TACCAP_URDF = "DM_TacCap00/urdf/rebot_b601_dm_taccap_v1_gripper.urdf"
DEFAULT_RS_TACCAP_URDF = "RS_TacCap05/urdf/rebot_b601_rs_taccap_v2_gripper.urdf"
DEFAULT_TACCAP_URDFS = {"dm": DEFAULT_DM_TACCAP_URDF, "rs": DEFAULT_RS_TACCAP_URDF}


@RobotConfig.register_subclass("rebot_b601_rt")
@dataclass
class RebotB601RTConfig(RobotConfig):
    motor_family: str = "dm"  # dm or rs
    channel: str = "can0"  # SocketCAN, serial path, or auto:first / auto:second (DM USB connection order)
    dm_serial_baud: int = 921600
    control_mode: str = "pos_vel"  # position/velocity loop, or mit
    interpolate_cartesian_targets: bool = True  # native timed Cartesian substeps, in both modes
    mit_velocity_feedforward: bool = True  # False sends MIT dq=0 while retaining position interpolation/Kd
    gravity_compensation_scale: float = 0.0  # [0, 1]; automatically set to zero in pos_vel mode
    gravity_torque_limits_nm: list[float] | None = None  # required when enabled; six FF caps, not total torque
    gravity_slew_rate_nm_s: float = 5.0  # maximum FF change per second, including startup
    gravity_vector: list[float] = field(default_factory=lambda: [0.0, 0.0, -9.81])  # URDF root frame, m/s^2
    gravity_use_joint_offsets: bool = False  # load the matching URDF's _offset.yaml holding torque biases
    motor_configs: list[dict[str, int | float | str]] = field(default_factory=list)
    urdf_path: str | None = None  # explicit override; None selects the bundled family/gripper model
    # Empty selects taccap_eef_link for bundled TacCap models, otherwise the URDF's default tool frame.
    end_frame: str | None = None
    # An external gripper uses its own driver instead of the seventh arm motor.
    # taccap_follower selects DM_TacCap00 or RS_TacCap05 by motor_family; None keeps
    # the original DM/RS model and its optional motor gripper.
    gripper: GripperConfig | None = None
    # Whether the native arm controller drives a seventh CAN motor.
    has_gripper: bool | None = None
    max_joint_speed: float = 0.5  # rad/s, fallback for all joints and native gripper speed
    max_joint_speeds: list[float] | None = None  # optional [J1, ..., J6], replaces arm scalar limit
    coordinate_cartesian_speed: bool = True  # scale TCP translation/rotation together before bounded IK
    max_cartesian_linear_speed: float | None = None  # m/s; TCP command limit, None disables
    max_cartesian_angular_speed: float | None = None  # rad/s; TCP command limit, None disables
    command_timeout_sec: float = 0.5
    reset_position_rad: list[float] | None = None  # six arm joints; startup and A-button target
    reset_on_exit: bool = False  # disconnect: return all six arm joints to zero unless reset=False
    reset_joint_speed: float = 0.5  # rad/s, additionally capped by each joint's configured limit
    reset_timeout_sec: float = 30.0
    reset_tolerance_rad: float = 0.03  # startup and A-button measured joint tolerance
    exit_reset_tolerance_rad: float = 0.03  # measured joint tolerance before disabling at zero
    trace_joint_tracking: bool = False
    trace_joint_tracking_hz: float = 2.0
    use_joint_observation: bool = False
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    @classmethod
    def get_known_choices(cls) -> dict[str, type]:
        # This concrete config also appears inside BiRebotB601RTConfig. Exposing
        # the root RobotConfig registry there recursively offers the dual-arm
        # config inside each arm while Draccus builds CLI arguments.
        return {cls.get_choice_name(cls): cls}

    def __post_init__(self):
        super().__post_init__()
        self.motor_family = self.motor_family.lower()
        if self.motor_family not in ("dm", "rs"):
            raise ValueError("motor_family must be 'dm' or 'rs'")
        self.control_mode = self.control_mode.lower()
        if self.control_mode not in ("mit", "pos_vel"):
            raise ValueError("control_mode must be 'mit' or 'pos_vel'")
        if not math.isfinite(self.gravity_compensation_scale) or not 0 <= self.gravity_compensation_scale <= 1:
            raise ValueError("gravity_compensation_scale must be finite and in [0, 1]")
        if self.control_mode == "pos_vel" and self.gravity_compensation_scale > 0:
            logger.warning(
                "B601 %s: control_mode=pos_vel automatically disables gravity feedforward "
                "(gravity_compensation_scale=%s -> 0.0).",
                self.id or self.channel,
                self.gravity_compensation_scale,
            )
            self.gravity_compensation_scale = 0.0
        if self.gravity_compensation_scale > 0 and self.gravity_torque_limits_nm is None:
            raise ValueError("Set gravity_torque_limits_nm before enabling gravity compensation")
        if self.gravity_torque_limits_nm is not None and (
            len(self.gravity_torque_limits_nm) != 6
            or any(not math.isfinite(v) or v < 0 for v in self.gravity_torque_limits_nm)
        ):
            raise ValueError("gravity_torque_limits_nm must contain six finite nonnegative values")
        if not math.isfinite(self.gravity_slew_rate_nm_s) or self.gravity_slew_rate_nm_s <= 0:
            raise ValueError("gravity_slew_rate_nm_s must be finite and positive")
        if len(self.gravity_vector) != 3 or not all(math.isfinite(v) for v in self.gravity_vector):
            raise ValueError("gravity_vector must contain three finite values")
        if self.has_gripper is None:
            self.has_gripper = self.gripper is None and not (
                self.uses_taccap_model or self.end_frame == "taccap_eef_link"
            )
        if self.gripper is not None and self.has_gripper:
            raise ValueError("An external gripper requires has_gripper=False for the native arm controller")
        if not self.channel:
            raise ValueError("channel must name a SocketCAN interface or DM serial device")
        if (self.channel == "auto" or self.channel.startswith("auto:")) and self.channel not in AUTO_DM_CHANNELS:
            raise ValueError("DM auto channel must be 'auto:first' or 'auto:second'")
        if self.motor_family == "rs" and (self.channel.startswith("/") or self.channel in AUTO_DM_CHANNELS):
            raise ValueError("RS requires a SocketCAN interface such as can0")
        if self.dm_serial_baud <= 0:
            raise ValueError("dm_serial_baud must be positive")
        if self.motor_configs and len(self.motor_configs) != 7:
            raise ValueError("motor_configs must contain seven entries")
        if any(not math.isfinite(v) or v <= 0 for v in (self.max_joint_speed, self.command_timeout_sec)):
            raise ValueError("speed and timeout must be positive and finite")
        if self.max_joint_speeds is not None and (
            len(self.max_joint_speeds) != 6 or any(not math.isfinite(v) or v <= 0 for v in self.max_joint_speeds)
        ):
            raise ValueError("max_joint_speeds must contain six finite positive speeds in rad/s")
        cartesian_limits = (self.max_cartesian_linear_speed, self.max_cartesian_angular_speed)
        if any(v is not None and (not math.isfinite(v) or v <= 0) for v in cartesian_limits):
            raise ValueError("Cartesian speed limits must be finite and positive, or None to disable")
        if any(v is not None for v in cartesian_limits) and not self.interpolate_cartesian_targets:
            raise ValueError("Cartesian speed limits require interpolate_cartesian_targets=True")
        if not math.isfinite(self.reset_joint_speed) or self.reset_joint_speed <= 0:
            raise ValueError("reset_joint_speed must be positive and finite")
        if self.reset_position_rad is not None and (
            len(self.reset_position_rad) != 6 or not all(math.isfinite(q) for q in self.reset_position_rad)
        ):
            raise ValueError("reset_position_rad must contain six finite joint angles in radians")
        if any(
            not math.isfinite(v) or v <= 0
            for v in (
                self.reset_timeout_sec,
                self.reset_tolerance_rad,
                self.exit_reset_tolerance_rad,
                self.trace_joint_tracking_hz,
            )
        ):
            raise ValueError("reset timeout, reset tolerance and tracking log rate must be positive and finite")
        if self.urdf_path is not None and not Path(self.urdf_path).is_file():
            raise ValueError(f"URDF does not exist: {self.urdf_path}")

    @property
    def default_urdf(self) -> str:
        """Bundled model selected by the mounted gripper and motor family."""
        if self.gripper is not None and self.gripper.type == "taccap_follower":
            return DEFAULT_TACCAP_URDFS[self.motor_family]
        family = self.motor_family.upper()
        return f"{family}/urdf/ReBot_Arm_{family}.urdf"

    @property
    def uses_taccap_model(self) -> bool:
        model_name = Path(self.urdf_path or self.default_urdf).name
        return any(model_name == Path(urdf).name for urdf in DEFAULT_TACCAP_URDFS.values())
