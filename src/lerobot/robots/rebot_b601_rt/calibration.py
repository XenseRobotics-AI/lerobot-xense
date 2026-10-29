"""Set B601 DM/RS motor zeros without enabling motors or changing control modes.

Usage (run from the lerobot-xense repository root with your Python environment activated):
    This module is called by lerobot-calibrate and has no standalone main entry point.
    Install the rebot_rt native SDK and the motorbridge calibration dependency first:
        python -m pip install 'motorbridge>=0.5.6,<0.6'

Platform 09: dual DM arms with separate TacCap grippers; calibrate each arm separately.
    Stop teleoperation, recording, and other programs using the arms. Support the arms:
    calibration disables the motors.
    Following the platform 09 recipe, plug in the right DM adapter first, wait for enumeration,
    then plug in the left adapter. Keep both connected: auto:first selects the right arm and
    auto:second selects the left. Alternatively, use a verified /dev/ttyACM* path for each arm.

    # Right arm: calibrate six arm joints; has_gripper=false excludes the separate USB TacCap gripper.
    lerobot-calibrate \
        --robot.type=rebot_b601_rt \
        --robot.motor_family=dm \
        --robot.channel=auto:first \
        --robot.dm_serial_baud=921600 \
        --robot.id=bi_rebot_dm_taccap00_09_right \
        --robot.has_gripper=false

    # After finishing the right arm, run calibration for the left arm.
    lerobot-calibrate \
        --robot.type=rebot_b601_rt \
        --robot.motor_family=dm \
        --robot.channel=auto:second \
        --robot.dm_serial_baud=921600 \
        --robot.id=bi_rebot_dm_taccap00_09_left \
        --robot.has_gripper=false

    Use the single-arm type rebot_b601_rt for calibration; do not pass the dual-arm teleop YAML directly.
    These IDs match the arms in the platform 09 recipe. Adapt IDs and interfaces for other platforms.
    If motor models or CAN IDs were overridden, supply matching robot.motor_configs values here too.
    For RS, use --robot.motor_family=rs and --robot.channel=can0 (replace with the actual SocketCAN interface).
    Use --robot.has_gripper=true to also zero the original seventh CAN gripper motor.

Interactive procedure:
    1. If a calibration JSON already exists, ENTER keeps it and exits; type c and press ENTER to recalibrate.
    2. After communication checks pass, manually place the arm at the mechanical zero pose specified
       in the B601 manual, then press ENTER. This is not the teleop YAML's reset_position_rad pose.
       Close the CAN gripper as well if it is included.
    3. The program sets the current motor positions as hardware zeros and requests fresh position readbacks.
       Keep supporting the arm throughout verification. The JSON is saved only if every position is within
       +/-0.01 rad. If verification fails, hardware zeros may already have changed; the JSON is not updated.

Calibration records:
    Default: HF_LEROBOT_CALIBRATION/robots/rebot_b601_rt/{dm|rs}/{id}.json, normally under
    ~/.cache/huggingface/lerobot/calibration/robots/rebot_b601_rt/ when no cache paths are overridden.
    With --robot.calibration_dir=/path/to/calibration, records go to that directory's {dm|rs}/{id}.json.
    This procedure sets motor zeros; it does not calibrate TCP/kinematics, identify gravity parameters,
    or measure joint travel. Readback checks do not verify manual alignment or persistence after a power cycle.
"""

import math
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import draccus

from lerobot.motors import MotorCalibration
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, ROBOTS
from lerobot.utils.robot_utils import get_logger

from .config_rebot_b601_rt import RebotB601RTConfig
from .dm_serial_discovery import (
    AUTO_DM_CHANNELS,
    DMSerialDevice,
    discover_dm_serial_devices,
    resolve_dm_serial_channel,
    verify_dm_serial_device,
)

logger = get_logger("b601_calibration")

# Hardware-zero readback tolerance, independent of the motion reset tolerance.
ZERO_POSITION_TOLERANCE_RAD = 0.01
DM_POSITION_REGISTER = 80  # p_m, current control position in radians
RS_POSITION_PARAMETER = 0x7019  # mechPos, mechanical position in radians


@dataclass(frozen=True)
class MotorTarget:
    name: str
    send_id: int
    feedback_id: int
    model: str


def _native_motor_defaults(motor_family: str):
    import rebot_rt

    family = rebot_rt.Family.DM if motor_family == "dm" else rebot_rt.Family.RS
    return rebot_rt.default_motor_configs(family)


def _motor_targets(config: RebotB601RTConfig) -> list[MotorTarget]:
    # Use the same native defaults and per-joint overrides as RT control.
    defaults = _native_motor_defaults(config.motor_family)
    targets = []
    for index, motor in enumerate(defaults[: 7 if config.has_gripper else 6]):
        changes = config.motor_configs[index] if config.motor_configs else {}
        target = MotorTarget(
            name=f"joint_{index + 1}" if index < 6 else "gripper",
            send_id=int(changes.get("send_id", motor.send_id)),
            feedback_id=int(changes.get("feedback_id", motor.feedback_id)),
            model=str(changes.get("model", motor.model)),
        )
        max_id = 0x7FF if config.motor_family == "dm" else 0xFF
        if not 1 <= target.send_id <= max_id or not 0 <= target.feedback_id <= max_id:
            raise ValueError(f"Invalid CAN IDs for {target.name}")
        targets.append(target)
    if len({target.send_id for target in targets}) != len(targets):
        raise ValueError("Motor send CAN IDs must be unique")
    return targets


def _open_controller(config: RebotB601RTConfig, *, dm_serial_devices: tuple[DMSerialDevice, ...] | None = None):
    try:
        from motorbridge import Controller
    except ModuleNotFoundError as error:
        if error.name != "motorbridge":
            raise
        raise ImportError(
            "B601 calibration requires motorbridge. Install the rebot-calibration extra "
            "or run: python -m pip install 'motorbridge>=0.5.6,<0.6'"
        ) from error

    channel = config.channel
    if channel in AUTO_DM_CHANNELS:
        devices = dm_serial_devices if dm_serial_devices is not None else discover_dm_serial_devices()
        channel = resolve_dm_serial_channel(channel, devices)
        verify_dm_serial_device(channel, devices)
    elif dm_serial_devices is not None and channel in {device.channel for device in dm_serial_devices}:
        # A dual-arm session pins both ports before calibrating the first arm.
        verify_dm_serial_device(channel, dm_serial_devices)
    if channel.startswith("/"):
        return Controller.from_dm_serial(serial_port=channel, baud=config.dm_serial_baud)
    return Controller(channel=channel)


def _check_motor_communication(motors: dict, targets: list[MotorTarget], config: RebotB601RTConfig) -> None:
    """Read DM ESC_ID or ping RS devices; cached motor states can be stale."""
    timeout_ms = max(1, round(config.command_timeout_sec * 1000))
    failures = []
    for target in targets:
        name, send_id, recv_id = target.name, target.send_id, target.feedback_id
        motor = motors[name]
        try:
            if config.motor_family == "dm":
                actual_id = motor.get_register_u32(8, timeout_ms=timeout_ms)
            else:
                actual_id, _ = motor.robstride_ping_host_id(recv_id, timeout_ms=timeout_ms)
            if actual_id != send_id:
                raise ValueError(f"Motor ID is {actual_id}, expected {send_id}")
        except Exception as error:
            failures.append(f"{name} (send=0x{send_id:02x}, feedback=0x{recv_id:02x}): {error}")
    if failures:
        raise ConnectionError(
            "Motor communication check failed; no zero commands were sent:\n"
            + "\n".join(failures)
            + "\nCheck power, CAN wiring and motor IDs. If the arm uses a separate USB/TacCap "
            "gripper, pass --robot.has_gripper=false."
        )


def _verify_motor_zeros(motors: dict, targets: list[MotorTarget], config: RebotB601RTConfig) -> None:
    """Request fresh position replies; get_state() alone may return a cached zero."""
    timeout_ms = max(1, round(config.command_timeout_sec * 1000))
    failures = []
    logger.info(f"Checking hardware zeros (tolerance: +/-{ZERO_POSITION_TOLERANCE_RAD:.3f} rad)")
    for target in targets:
        motor = motors[target.name]
        label = f"{target.name} (send=0x{target.send_id:02x}, feedback=0x{target.feedback_id:02x})"
        try:
            # motorbridge >=0.5.6 waits for a new register reply. DM checks the
            # reply timestamp; RS clears the parameter cache before requesting.
            # Do not fall back to the untimestamped get_state() cache on timeout.
            if config.motor_family == "dm":
                position = motor.get_register_f32(DM_POSITION_REGISTER, timeout_ms=timeout_ms)
            else:
                position = motor.robstride_get_param_f32_host_id(
                    RS_POSITION_PARAMETER, target.feedback_id, timeout_ms=timeout_ms
                )
            if not math.isfinite(position):
                raise ValueError(f"non-finite position: {position}")
            passed = abs(position) <= ZERO_POSITION_TOLERANCE_RAD
            result = f"{label}: {position:+.6f} rad ({math.degrees(position):+.3f} deg)"
            logger.info(f"Zero check {'PASS' if passed else 'FAIL'}: {result}")
            if not passed:
                failures.append(result)
        except Exception as error:
            result = f"{label}: position read failed: {error}"
            logger.warn(f"Zero check FAIL: {result}")
            failures.append(result)
    if failures:
        raise RuntimeError(
            "Hardware zero verification failed; calibration file was not updated. "
            "Zero commands have already been sent to the motors.\n"
            + "\n".join(failures)
            + "\nKeep the arm supported at the mechanical zero pose; check the reported motors and communication."
        )


def _save_calibration(path: Path, calibration: dict[str, MotorCalibration]) -> None:
    """Replace the file only after every selected motor's zero has been verified."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", dir=path.parent, suffix=".json", delete=False) as file:
            temporary = Path(file.name)
            with draccus.config_type("json"):
                draccus.dump(calibration, file, indent=4)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _close_controller(controller, motors: dict) -> None:
    # Attempt every cleanup operation even if one handle has stopped responding.
    try:
        controller.disable_all()
    except Exception as error:
        logger.warn(f"Failed to disable motors during cleanup: {error}")
    for name, motor in motors.items():
        try:
            motor.close()
        except Exception as error:
            logger.warn(f"Failed to close {name}: {error}")
    controller.close()


def calibrate_b601(config: RebotB601RTConfig, *, dm_serial_devices: tuple[DMSerialDevice, ...] | None = None) -> None:
    # A shared id such as follower1 must not reuse calibration across DM and RS.
    directory = (config.calibration_dir or HF_LEROBOT_CALIBRATION / ROBOTS / config.type) / config.motor_family
    path = directory / f"{config.id}.json"
    if path.is_file():
        choice = input(
            f"Calibration file exists: {path}\n"
            "Press ENTER to keep it, or type 'c' and press ENTER to set new hardware zeros: "
        )
        if choice.strip().lower() != "c":
            logger.info(f"Keeping calibration file {path}; no motor commands sent.")
            return

    targets = _motor_targets(config)
    controller = (
        _open_controller(config)
        if dm_serial_devices is None
        else _open_controller(config, dm_serial_devices=dm_serial_devices)
    )
    motors = {}
    try:
        add_motor = controller.add_damiao_motor if config.motor_family == "dm" else controller.add_robstride_motor
        for target in targets:
            motors[target.name] = add_motor(target.send_id, target.feedback_id, target.model)
        controller.disable_all()
        _check_motor_communication(motors, targets, config)

        gripper_instruction = " and close the CAN gripper" if "gripper" in motors else ""
        print(
            f"\nB601 {config.motor_family.upper()} calibration: set current motor positions as zero.\n"
            f"Motors are disabled. Manually move the arm to the B601 ZERO POSE{gripper_instruction}.\n"
            "Use the mechanical zero pose specified by the B601 manual.\n"
            "Keep the arm supported in that pose until zero verification finishes.\n"
            f"Selected motors: {', '.join(motors)}"
        )
        input("Press ENTER only when the arm is in the correct zero pose...")
        # The operator may have taken time to position the arm; check live communication again.
        _check_motor_communication(motors, targets, config)

        zeroed = []
        for name, motor in motors.items():
            try:
                motor.set_zero_position()
            except Exception as error:
                raise RuntimeError(
                    f"Failed to set zero for {name}. Zero commands already sent to: {zeroed}. "
                    "Calibration file was not updated; check the arm before retrying."
                ) from error
            zeroed.append(name)
            time.sleep(0.1)

        _verify_motor_zeros(motors, targets, config)

        # Preserve the legacy Seeed file format. These ranges are placeholders,
        # not measured travel limits; this procedure only sets hardware zeros.
        calibration = {
            target.name: MotorCalibration(id=target.send_id, drive_mode=0, homing_offset=0, range_min=-90, range_max=90)
            for target in targets
        }
        _save_calibration(path, calibration)
        logger.info(f"Hardware zeros verified for {len(motors)} motors. Calibration saved to {path}")
    except BaseException:
        # Keep the original calibration/communication error if cleanup also fails.
        with suppress(Exception):
            _close_controller(controller, motors)
        raise
    else:
        _close_controller(controller, motors)
