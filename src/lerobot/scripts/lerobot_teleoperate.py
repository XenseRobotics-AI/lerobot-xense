# Copyright 2026 XenseRobotics Inc. team. All rights reserved.
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
Simple script to control a robot from teleoperation.
Example (SO-101):

```shell
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --display_data=true
```

Example (ARX5 single-arm, teach mode data collection):

```shell
lerobot-teleoperate \
    --robot.type=arx5_follower \
    --robot.arm_port=can0 \
    --teleop.type=btgamepad \
    --fps=30 \
    --display_data=true
```

Example (Bimanual ARX5, teach mode data collection):

```shell
lerobot-teleoperate \
    --robot.type=bi_arx5 \
    --robot.left_config.arm_port=can0 \
    --robot.right_config.arm_port=can1 \
    --teleop.type=btgamepad \
    --fps=30
```

Example (Flexiv Rizon4 RT + Pico4):

```shell
lerobot-teleoperate \
    --robot.type=flexiv_rizon4_rt \
    --robot.robot_sn=Rizon4-063423 \
    --teleop.type=pico4 \
    --fps=60 \
    --debug_timing=true
```

Example (Flexiv Rizon4 RT + SpaceMouse):

```shell
lerobot-teleoperate \
    --robot.type=flexiv_rizon4_rt \
    --robot.robot_sn=Rizon4-063423 \
    --teleop.type=spacemouse \
    --fps=30 \
    --display_data=true
```

Example (Bimanual Flexiv Rizon4 RT + Bi-Pico4):

```shell
lerobot-teleoperate \
    --robot.type=bi_flexiv_rizon4_rt \
    --robot.left_robot_sn=Rizon4-063423 \
    --robot.right_robot_sn=Rizon4-063424 \
    --teleop.type=bi_pico4 \
    --fps=60
```

Example (Bimanual Elite CS66 RT + Bi-Pico4):

```shell
lerobot-teleoperate \
    --robot.type=bi_elite_cs66_rt \
    --robot.left_robot_ip=192.168.8.53 \
    --robot.right_robot_ip=192.168.8.223 \
    --teleop.type=bi_pico4 \
    --fps=30
```

"""

# ── TacCap SDK / OpenCV glib load-order shim ─────────────────────────────────
# opencv-python (pip) links the system libglib-2.0 (2.72), which lacks symbols
# the TacCap SDK's conda libgobject (2.86) needs. Whichever glib is loaded first
# claims the process-wide "libglib-2.0.so.0" slot, so we import xense.taccap
# BEFORE OpenCV gets pulled in (via the camera/robot imports further down),
# making the newer conda glib win. Guarded: a no-op when the SDK isn't present.
# contextlib.suppress is not usable here: this has to run before anything pulls
# in OpenCV, which is earlier than the import block below.
try:  # noqa: SIM105
    import xense.taccap  # noqa: F401
# Silent by design: the SDK being absent is the ordinary case on a host that
# does not use it, and there is no logger yet this early in the module.
except Exception:  # nosec B110
    pass

import contextlib
import time
import traceback
from dataclasses import asdict, dataclass
from pprint import pformat

import numpy as np
import rerun as rr

# Load TacCap native libs before cv2/Pillow/torchvision. Those packages may
# preload vendored JPEG/TIFF libraries that conflict with the conda OpenCV libs
# used by xense.taccap. Concretely: torchvision ships a libjpeg that claims the
# LIBJPEG_8.0 symbol version but carries none of the jpeg12_* symbols conda's
# libtiff needs, so whichever loads first wins — and when that is torchvision,
# every later `import xense.taccap` dies with an undefined-symbol ImportError.
# Importing it here, above every lerobot import, is what fixes the order; moving
# this below them puts the bug back.
with contextlib.suppress(ImportError):  # SDK absent: the TacCap paths fail later with a clear error
    import xense.taccap  # noqa: F401


from lerobot.configs import parser
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    arx5_follower,
    bi_arx5,
    bi_elite_cs66_rt,
    bi_flexiv_rizon4_rt,
    elite_cs66_rt,
    flexiv_rizon4_rt,
    make_robot_from_config,
    mock_robot,
)
from lerobot.scripts.teleop_device_loops import (
    BiPico4Policy,
    Pico4Policy,
    SpaceMousePolicy,
    run_cartesian_teleop_loop,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_pico4,
    btgamepad,
    gamepad,
    make_teleoperator_from_config,
    mock_teleop,
    pico4,
    spacemouse,
    trlc_leader,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import (
    busy_wait,
    get_logger,
    precise_sleep,
)
from lerobot.utils.utils import move_cursor_up
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

logger = get_logger("Teleoperate")


@dataclass
class TeleoperateConfig:
    # TODO: pepijn, steven: if more robots require multiple teleoperators (like lekiwi) its good to make this possibele in teleop.py and record.py with List[Teleoperator]
    teleop: TeleoperatorConfig
    robot: RobotConfig
    # Limit the maximum frames per second.
    fps: int = 60
    teleop_time_s: float | None = None
    # Display all cameras on screen
    display_data: bool = False
    # Display data on a remote Rerun server
    display_ip: str | None = None
    # Port of the remote Rerun server
    display_port: int | None = None
    # Whether to display compressed images in Rerun (JPEG) to lower memory/IPC load. Set False for lossless display.
    display_compressed_images: bool = True
    # Print per-step timing breakdown instead of action values.
    debug_timing: bool = False
    # Dryrun mode: print actions but do not send to robot
    dryrun: bool = False


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _safe_disconnect(obj, name: str) -> None:
    if obj is None:
        return
    try:
        if obj.is_connected:
            obj.disconnect()
            logger.info(f"{name} disconnected")
    except Exception as e:
        logger.error(f"Error disconnecting {name}: {e}\n{traceback.format_exc()}")


def _cleanup(robot, teleop, display_data: bool) -> None:
    if display_data:
        try:
            rr.rerun_shutdown()
        except Exception as e:
            logger.warn(f"Error shutting down rerun: {e}")
    _safe_disconnect(teleop, teleop.__class__.__name__ if teleop else "teleop")
    _safe_disconnect(robot, robot.__class__.__name__ if robot else "robot")


def _obs_ms(timing: dict, key: str) -> str:
    """One field of a robot's ``_last_obs_timing`` breakdown, as milliseconds.

    Looked up with ``.get`` rather than ``[]`` so a rig that publishes a
    different set of sources degrades to a blank column instead of raising
    inside the teleop loop.
    """
    value = timing.get(key)
    return f"{value:.1f}" if isinstance(value, (int, float)) else "--"


def _print_obs_state(
    obs: dict, display_len: int, status: str, *, extra_lines: tuple[str, ...] = ()
) -> None:
    """Print scalar observation values with a status tag (used during reset/moving)."""
    scalar_keys = [k for k, v in obs.items() if not isinstance(v, np.ndarray)]
    col = max((len(k) for k in scalar_keys), default=display_len)
    print("\n" + "-" * (col + 18))
    print(f"{'NAME':<{col}} | {'OBS':>10}  {status}")
    for k in scalar_keys:
        print(f"{k:<{col}} | {float(obs[k]):>10.4f}")
    for line in extra_lines:
        print(f"\033[2K{line}")
    # Rewind the exact panel height: blank/separator + header + N scalar rows.
    move_cursor_up(len(scalar_keys) + len(extra_lines) + 3)


# ---------------------------------------------------------------------------
# Generic teleop loop (upstream)
# ---------------------------------------------------------------------------


def teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    display_data: bool = False,
    duration: float | None = None,
    display_compressed_images: bool = True,
    debug_timing: bool = False,
):
    """
    This function continuously reads actions from a teleoperation device, processes them through optional
    pipelines, sends them to a robot, and optionally displays the robot's state. The loop runs at a
    specified frequency until a set duration is reached or it is manually interrupted.

    Args:
        teleop: The teleoperator device instance providing control actions.
        robot: The robot instance being controlled.
        fps: The target frequency for the control loop in frames per second.
        display_data: If True, fetches robot observations and displays them in the console and Rerun.
        display_compressed_images: If True, compresses images before sending them to Rerun for display.
        duration: The maximum duration of the teleoperation loop in seconds. If None, the loop runs indefinitely.
        debug_timing: If True, print per-step timing breakdown instead of action table.
    """
    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()

    while True:
        loop_start = time.perf_counter()

        # Get robot observation
        obs_t0 = time.perf_counter()
        obs = robot.get_observation()
        obs_time_ms = (time.perf_counter() - obs_t0) * 1e3

        # Get teleop action
        teleop_t0 = time.perf_counter()
        raw_action = teleop.get_action()
        teleop_time_ms = (time.perf_counter() - teleop_t0) * 1e3

        # Process teleop action through pipeline
        teleop_action = raw_action

        # Process action for robot through pipeline
        robot_action_to_send = teleop_action

        # Send processed action to robot
        send_t0 = time.perf_counter()
        _ = robot.send_action(robot_action_to_send)
        send_time_ms = (time.perf_counter() - send_t0) * 1e3

        if display_data:
            # Process robot observation through pipeline
            obs_transition = obs

            log_rerun_data(
                observation=obs_transition,
                action=teleop_action,
                compress_images=display_compressed_images,
            )

            if not debug_timing:
                print("\n" + "-" * (display_len + 10))
                print(f"{'NAME':<{display_len}} | {'VALUE':>9}")
                for motor, value in robot_action_to_send.items():
                    print(f"{motor:<{display_len}} | {value:>9.4f}")
                move_cursor_up(len(robot_action_to_send) + 3)

        dt_s = time.perf_counter() - loop_start
        precise_sleep(max(1 / fps - dt_s, 0.0))
        loop_s = time.perf_counter() - loop_start

        if debug_timing:
            print(
                f"\r\033[K"
                f"obs: {obs_time_ms:5.1f}ms | "
                f"teleop: {teleop_time_ms:5.1f}ms | "
                f"send: {send_time_ms:5.1f}ms | "
                f"loop: {loop_s * 1e3:5.1f}ms | "
                f"target: {1e3 / fps:.1f}ms | "
                f"eff: {(1 / fps) / loop_s * 100:5.1f}%",
                end="",
                flush=True,
            )
        elif not display_data:
            print(f"Teleop loop time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
            move_cursor_up(1)

        if duration is not None and time.perf_counter() - start >= duration:
            return


# ---------------------------------------------------------------------------
# Specialised teleop loops
# ---------------------------------------------------------------------------


def mock_robot_teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    display_data: bool = False,
    duration: float | None = None,
    dryrun: bool = False,
    debug_timing: bool = False,
):
    """
    Dedicated teleoperation loop for Mock Robot.

    This loop keeps the same processor pipeline as the default path but adds:
    - Action key filtering to robot.action_features
    - Dedicated timing/terminal display for mock robot teleoperation
    """
    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()
    robot_action_keys = set(robot.action_features.keys())
    warned_unmapped_keys = False

    while True:
        loop_start = time.perf_counter()

        obs_start = time.perf_counter()
        obs = robot.get_observation()
        obs_dt_ms = (time.perf_counter() - obs_start) * 1e3

        raw_action = teleop.get_action()
        teleop_action = raw_action

        # Keep only keys known by mock robot action schema.
        filtered_action = {k: v for k, v in teleop_action.items() if k in robot_action_keys}
        if not filtered_action and teleop_action:
            filtered_action = teleop_action
        elif len(filtered_action) != len(teleop_action) and not warned_unmapped_keys:
            dropped = sorted(set(teleop_action) - robot_action_keys)
            logger.warn(f"Action keys not present in mock robot action schema, dropping: {dropped}")
            warned_unmapped_keys = True

        robot_action_to_send = filtered_action

        if not dryrun:
            _ = robot.send_action(robot_action_to_send)

        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)
        loop_s = time.perf_counter() - loop_start

        if display_data:
            obs_transition = obs
            log_rerun_data(observation=obs_transition, action=teleop_action)

            ordered_keys = [k for k in robot.action_features if k in robot_action_to_send]
            ordered_keys.extend(k for k in robot_action_to_send if k not in ordered_keys)

            panel_lines = []
            panel_lines.append("-" * (display_len + 38))
            panel_lines.append(f"{'NAME':<{display_len}} | {'CMD':>8} | {'OBS':>8} | {'ERR':>8}")
            for motor in ordered_keys:
                cmd = float(robot_action_to_send[motor])
                obs_val = obs.get(motor, None)
                if obs_val is None or isinstance(obs_val, np.ndarray):
                    panel_lines.append(f"{motor:<{display_len}} | {cmd:>8.4f} | {'-':>8} | {'-':>8}")
                    continue

                obs_num = float(obs_val)
                err = cmd - obs_num
                panel_lines.append(f"{motor:<{display_len}} | {cmd:>8.4f} | {obs_num:>8.4f} | {err:>+8.4f}")

            panel_lines.append(
                f"{'timing':<{display_len}} | {'loop':>8} | {loop_s * 1e3:>6.2f}ms | {obs_dt_ms:>6.2f}ms"
            )

            print("\n".join(panel_lines), flush=True)
            move_cursor_up(len(panel_lines))

        if debug_timing and not display_data:
            dryrun_tag = " | DRYRUN" if dryrun else ""
            print(
                f"\r\033[KMOCK obs: {obs_dt_ms:5.1f}ms | loop: {loop_s * 1e3:5.1f}ms ({1 / loop_s:4.0f}Hz){dryrun_tag}",
                end="",
                flush=True,
            )
        elif not display_data:
            action_summary = " ".join(f"{k}={float(v):+.3f}" for k, v in robot_action_to_send.items())
            dryrun_tag = "[DRYRUN] " if dryrun else ""
            print(
                f"\r\033[K{dryrun_tag}{loop_s * 1e3:5.1f}ms ({1 / loop_s:4.0f}Hz) | {action_summary}",
                end="",
                flush=True,
            )

        if duration is not None and time.perf_counter() - start >= duration:
            return


def arx5_teleop_loop(
    robot: Robot,
    fps: int,
    display_data: bool = False,
    duration: float | None = None,
    debug_timing: bool = False,
):
    """
    Teleop loop for ARX5 robots (both single-arm and bimanual).

    Supports:
    - Single arm mode (arx5_follower): robot.arm
    - Bimanual mode (bi_arx5): robot.left_arm, robot.right_arm
    """
    start = time.perf_counter()
    timing_stats = {
        "robot_obs_times": [],
        "camera_obs_times": {},
        "total_obs_times": [],
        "loop_times": [],
    }

    is_bimanual = hasattr(robot, "left_arm") and hasattr(robot, "right_arm")
    is_single_arm = hasattr(robot, "arm") and not is_bimanual

    if not is_bimanual and not is_single_arm:
        raise ValueError("Robot must have either 'arm' (single) or 'left_arm'/'right_arm' (bimanual)")

    camera_keys = [key for key in robot.observation_features if not key.endswith(".pos")]
    for cam_key in camera_keys:
        timing_stats["camera_obs_times"][cam_key] = []

    while True:
        loop_start = time.perf_counter()

        obs_start = time.perf_counter()
        robot_state_start = time.perf_counter()

        if is_bimanual:
            left_joint_state = robot.left_arm.get_joint_state()
            right_joint_state = robot.right_arm.get_joint_state()
        else:
            joint_state = robot.arm.get_joint_state()

        robot_obs_time = time.perf_counter() - robot_state_start
        timing_stats["robot_obs_times"].append(robot_obs_time * 1000)

        camera_obs_start = time.perf_counter()
        camera_observations = {}
        camera_times = {}
        for cam_key, cam in robot.cameras.items():
            cam_start = time.perf_counter()
            camera_observations[cam_key] = cam.async_read()
            cam_time_ms = (time.perf_counter() - cam_start) * 1000
            camera_times[cam_key] = cam_time_ms
            timing_stats["camera_obs_times"][cam_key].append(cam_time_ms)

        total_camera_time_ms = (time.perf_counter() - camera_obs_start) * 1000

        raw_observation = {}

        if is_bimanual:
            left_pos = left_joint_state.pos().copy()
            for i in range(6):
                raw_observation[f"left_joint_{i + 1}.pos"] = float(left_pos[i])
            raw_observation["left_gripper.pos"] = float(left_joint_state.gripper_pos)

            right_pos = right_joint_state.pos().copy()
            for i in range(6):
                raw_observation[f"right_joint_{i + 1}.pos"] = float(right_pos[i])
            raw_observation["right_gripper.pos"] = float(right_joint_state.gripper_pos)
        else:
            pos = joint_state.pos().copy()
            for i in range(6):
                raw_observation[f"joint_{i + 1}.pos"] = float(pos[i])
            raw_observation["gripper.pos"] = float(joint_state.gripper_pos)

        raw_observation.update(camera_observations)

        total_obs_time = time.perf_counter() - obs_start
        timing_stats["total_obs_times"].append(total_obs_time * 1000)

        raw_action = {
            key: value
            for key, value in raw_observation.items()
            if (
                key.endswith(".pos")
                and not key.startswith("head")
                and not key.startswith("left_wrist")
                and not key.startswith("right_wrist")
            )
        }

        if display_data:
            obs_transition = raw_observation
            log_rerun_data(observation=obs_transition, action=raw_action)

            if not debug_timing:
                if is_bimanual:
                    left_motors = {k: v for k, v in raw_action.items() if k.startswith("left_")}
                    right_motors = {k: v for k, v in raw_action.items() if k.startswith("right_")}
                    col_width = 25
                    print("\n" + "-" * (col_width * 2 + 3))
                    print(f"{'LEFT ARM':<{col_width}} | {'RIGHT ARM':<{col_width}}")
                    print("-" * (col_width * 2 + 3))
                    max_motors = max(len(left_motors), len(right_motors))
                    left_items = list(left_motors.items())
                    right_items = list(right_motors.items())
                    for i in range(max_motors):
                        left_str = ""
                        right_str = ""
                        if i < len(left_items):
                            motor_name = left_items[i][0].replace("left_", "")
                            left_str = f"{motor_name}: {left_items[i][1]:>7.3f}"
                        if i < len(right_items):
                            motor_name = right_items[i][0].replace("right_", "")
                            right_str = f"{motor_name}: {right_items[i][1]:>7.3f}"
                        print(f"{left_str:<{col_width}} | {right_str:<{col_width}}")
                    move_cursor_up(max_motors + 4)
                else:
                    col_width = 20
                    print("\n" + "-" * (col_width + 12))
                    print(f"{'JOINT':<{col_width}} | {'VALUE':>7}")
                    print("-" * (col_width + 12))
                    motor_items = list(raw_action.items())
                    for motor, value in motor_items:
                        print(f"{motor:<{col_width}} | {value:>7.3f}")
                    move_cursor_up(len(motor_items) + 4)

        dt_s = time.perf_counter() - loop_start
        precise_sleep(max(1 / fps - dt_s, 0))
        loop_s = time.perf_counter() - loop_start
        timing_stats["loop_times"].append(loop_s * 1000)

        if debug_timing:
            print()
            print("TELEOP TIMING DEBUG")
            print("=" * 50)
            print(f"Robot state:     {robot_obs_time * 1000:.1f}ms")
            print(f"Total cameras:   {total_camera_time_ms:.1f}ms")
            print()
            num_cameras = len(camera_times)
            for cam_key, cam_time_ms in camera_times.items():
                speed = "SLOW" if cam_time_ms > 10 else ("MED " if cam_time_ms > 5 else "FAST")
                print(f"  {speed} {cam_key:12}: {cam_time_ms:5.1f}ms")
            print()
            print(f"Total observation: {total_obs_time * 1000:.1f}ms")
            print(f"Loop time:        {loop_s * 1000:.1f}ms")
            print(f"Target period:    {1000 / fps:.1f}ms")
            print(f"Loop efficiency:  {(1000 / fps) / (loop_s * 1000) * 100:.1f}%")
            extra_warning_lines = 0
            if total_camera_time_ms > 20:
                print()
                print(f"SLOW CAMERAS DETECTED! Total: {total_camera_time_ms:.1f}ms")
                extra_warning_lines = 2
            print("=" * 50)
            total_lines = 1 + 1 + 1 + 2 + 1 + num_cameras + 1 + 4 + extra_warning_lines + 1
            move_cursor_up(total_lines)
        else:
            if total_camera_time_ms > 20:
                print(f"SLOW CAMERAS: {total_camera_time_ms:.1f}ms")
                for cam_key, cam_time_ms in camera_times.items():
                    if cam_time_ms > 10:
                        print(f"  SLOW {cam_key}: {cam_time_ms:.1f}ms")

        if duration is not None and time.perf_counter() - start >= duration:
            if len(timing_stats["robot_obs_times"]) > 10:
                print("\n=== FINAL TIMING REPORT ===")
                all_robot = timing_stats["robot_obs_times"]
                all_total = timing_stats["total_obs_times"]
                all_loops = timing_stats["loop_times"]
                print(f"Total samples: {len(all_robot)}")
                print(f"Robot obs - avg: {sum(all_robot) / len(all_robot):.2f}ms")
                print(f"Total obs - avg: {sum(all_total) / len(all_total):.2f}ms")
                print(f"Loop time - avg: {sum(all_loops) / len(all_loops):.2f}ms")
                for cam_key, cam_times in timing_stats["camera_obs_times"].items():
                    if cam_times:
                        print(f"{cam_key} - avg: {sum(cam_times) / len(cam_times):.2f}ms")
            return


def arx5_trlc_leader_teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    display_data: bool = False,
    duration: float | None = None,
    dryrun: bool = False,
    debug_timing: bool = False,
):
    """
    Dedicated teleoperation loop for ARX5 + TRLC leader.

    TRLC leader outputs joint-space actions (`joint_i.pos` + `gripper.pos`), so this loop
    validates key compatibility with the robot's action schema and then sends commands.
    """
    # ARX5 trlc leader loop currently supports single-arm follower only.
    if hasattr(robot, "left_arm") and hasattr(robot, "right_arm"):
        raise ValueError("TRLC leader teleoperation currently supports arx5_follower only, not bi_arx5.")

    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()
    robot_action_keys = set(robot.action_features.keys())
    warned_unmapped_keys = False

    while True:
        loop_start = time.perf_counter()

        obs_start = time.perf_counter()
        obs = robot.get_observation()
        obs_dt_ms = (time.perf_counter() - obs_start) * 1e3

        raw_action = teleop.get_action()
        for k in raw_action:
            if "gripper" in k:
                raw_action[k] = (1 - raw_action[k]) * 1.57
        teleop_action = raw_action

        filtered_action = {k: v for k, v in teleop_action.items() if k in robot_action_keys}
        if len(filtered_action) != len(teleop_action) and not warned_unmapped_keys:
            dropped = sorted(set(teleop_action) - robot_action_keys)
            logger.warn(f"TRLC action keys not present in ARX5 action schema, dropping: {dropped}")
            warned_unmapped_keys = True

        if not filtered_action:
            raise ValueError(
                "No overlapping action keys between TRLC leader output and ARX5 action schema. "
                "Check robot.control_mode (TRLC requires joint-style keys like joint_i.pos, gripper.pos)."
            )

        robot_action_to_send = filtered_action

        if not dryrun:
            _ = robot.send_action(robot_action_to_send)

        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)
        loop_s = time.perf_counter() - loop_start

        if display_data:
            obs_transition = obs
            log_rerun_data(observation=obs_transition, action=teleop_action)

            ordered_keys = [k for k in robot.action_features if k in robot_action_to_send]
            ordered_keys.extend(k for k in robot_action_to_send if k not in ordered_keys)

            panel_lines = []
            panel_lines.append("-" * (display_len + 38))
            panel_lines.append(f"{'NAME':<{display_len}} | {'CMD':>8} | {'OBS':>8} | {'ERR':>8}")
            for motor in ordered_keys:
                cmd = float(robot_action_to_send[motor])
                obs_val = obs.get(motor, None)
                if obs_val is None or isinstance(obs_val, np.ndarray):
                    panel_lines.append(f"{motor:<{display_len}} | {cmd:>8.4f} | {'-':>8} | {'-':>8}")
                    continue

                obs_num = float(obs_val)
                err = cmd - obs_num
                panel_lines.append(f"{motor:<{display_len}} | {cmd:>8.4f} | {obs_num:>8.4f} | {err:>+8.4f}")

            panel_lines.append(
                f"{'timing':<{display_len}} | {'loop':>8} | {loop_s * 1e3:>6.2f}ms | {obs_dt_ms:>6.2f}ms"
            )
            print("\n".join(panel_lines), flush=True)
            move_cursor_up(len(panel_lines))
        elif debug_timing:
            dryrun_tag = " | DRYRUN" if dryrun else ""
            print(
                f"\r\033[KARX5+TRLC obs: {obs_dt_ms:5.1f}ms | loop: {loop_s * 1e3:5.1f}ms ({1 / loop_s:4.0f}Hz){dryrun_tag}",
                end="",
                flush=True,
            )
        else:
            action_summary = " ".join(f"{k}={float(v):+.3f}" for k, v in robot_action_to_send.items())
            dryrun_tag = "[DRYRUN] " if dryrun else ""
            print(
                f"\r\033[K{dryrun_tag}{loop_s * 1e3:5.1f}ms ({1 / loop_s:4.0f}Hz) | {action_summary}",
                end="",
                flush=True,
            )

        if duration is not None and time.perf_counter() - start >= duration:
            return


def spacemouse_teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    display_data: bool = False,
    duration: float | None = None,
    dryrun: bool = False,
    debug_timing: bool = False,
):
    """Flexiv Rizon4 + SpaceMouse."""
    run_cartesian_teleop_loop(
        teleop,
        robot,
        fps,
        SpaceMousePolicy(),
        display_data=display_data,
        duration=duration,
        dryrun=dryrun,
        debug_timing=debug_timing,
        honour_rt_moving=robot.name == "flexiv_rizon4_rt",
    )


def elite_cs66_rt_spacemouse_teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    display_data: bool = False,
    duration: float | None = None,
    dryrun: bool = False,
    debug_timing: bool = False,
    release_resync_idle_s: float = 0.5,
):
    """Elite CS66 + SpaceMouse.

    ``release_resync_idle_s`` is accepted for call compatibility but no longer
    drives an idle-resync here: the shared loop re-syncs whenever the arm has
    been running its own trajectory, which covers the case that mattered.
    """
    run_cartesian_teleop_loop(
        teleop,
        robot,
        fps,
        SpaceMousePolicy(),
        display_data=display_data,
        duration=duration,
        dryrun=dryrun,
        debug_timing=debug_timing,
    )


def elite_cs66_rt_pico4_teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    display_data: bool = False,
    duration: float | None = None,
    dryrun: bool = False,
    debug_timing: bool = False,
):
    """Elite CS66 + Pico4.

    Pico4 emits the 6D-rotation Cartesian schema Elite CS66 consumes, so nothing
    is converted here — the shared loop sends it through untouched.
    """
    run_cartesian_teleop_loop(
        teleop,
        robot,
        fps,
        Pico4Policy(),
        display_data=display_data,
        duration=duration,
        dryrun=dryrun,
        debug_timing=debug_timing,
    )


def pico4_teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    display_data: bool = False,
    duration: float | None = None,
    dryrun: bool = False,
    debug_timing: bool = False,
):
    """Flexiv Rizon4 + Pico4.

    ``rt_moving`` is only honoured for the RT variant: the non-RT Flexiv has no
    background servo loop to stand aside for, and reading the attribute on one
    would stall every frame.
    """
    run_cartesian_teleop_loop(
        teleop,
        robot,
        fps,
        Pico4Policy(),
        display_data=display_data,
        duration=duration,
        dryrun=dryrun,
        debug_timing=debug_timing,
        honour_rt_moving=robot.name == "flexiv_rizon4_rt",
    )


def bi_pico4_teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    display_data: bool = False,
    duration: float | None = None,
    dryrun: bool = False,
    debug_timing: bool = False,
):
    """Bimanual arm + dual Pico4 controllers."""
    run_cartesian_teleop_loop(
        teleop,
        robot,
        fps,
        BiPico4Policy(),
        display_data=display_data,
        duration=duration,
        dryrun=dryrun,
        debug_timing=debug_timing,
    )


@parser.wrap()
def teleoperate(cfg: TeleoperateConfig):
    logger.info(pformat(asdict(cfg)))
    if cfg.dryrun:
        logger.warn("DRYRUN MODE ENABLED - Actions will be printed but NOT sent to robot")

    if cfg.display_data:
        teleop_name = cfg.teleop.type if cfg.teleop else "none"
        session_name = f"teleop_{cfg.robot.type}_{teleop_name}"
        init_rerun(session_name=session_name, ip=cfg.display_ip, port=cfg.display_port)

    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    robot = None
    teleop = None

    try:
        # --- arx5_follower / bi_arx5 + spacemouse ---
        if cfg.robot.type in ("bi_arx5", "arx5_follower") and cfg.teleop.type == "spacemouse":
            mode = "bimanual" if cfg.robot.type == "bi_arx5" else "single-arm"
            logger.info(f"Detected ARX5 ({mode}) + SpaceMouse")
            robot = make_robot_from_config(cfg.robot)
            robot.connect()
            teleop = make_teleoperator_from_config(cfg.teleop)
            logger.info(f"Current TCP pose (euler+gripper): {robot.get_current_tcp_pose_euler()}")
            teleop.connect(current_tcp_pose_euler=robot.get_current_tcp_pose_euler())
            with contextlib.suppress(KeyboardInterrupt):
                spacemouse_teleop_loop(
                    teleop=teleop,
                    robot=robot,
                    fps=cfg.fps,
                    display_data=cfg.display_data,
                    duration=cfg.teleop_time_s,
                    dryrun=cfg.dryrun,
                    debug_timing=cfg.debug_timing,
                )

        # --- arx5_follower + trlc_leader ---
        elif cfg.robot.type == "arx5_follower" and cfg.teleop.type == "trlc_leader":
            logger.info("Detected ARX5 Follower + TRLC Leader")
            robot = make_robot_from_config(cfg.robot)
            robot.connect()
            teleop = make_teleoperator_from_config(cfg.teleop)
            teleop.connect()
            with contextlib.suppress(KeyboardInterrupt):
                arx5_trlc_leader_teleop_loop(
                    teleop=teleop,
                    robot=robot,
                    fps=cfg.fps,
                    display_data=cfg.display_data,
                    duration=cfg.teleop_time_s,
                    debug_timing=cfg.debug_timing,
                    dryrun=cfg.dryrun,
                )

        # --- arx5_follower / bi_arx5 (other teleops) ---
        elif cfg.robot.type in ("bi_arx5", "arx5_follower"):
            mode = "bimanual" if cfg.robot.type == "bi_arx5" else "single-arm"
            logger.info(f"Detected ARX5 ({mode}), using ARX5 teleop loop")
            robot = make_robot_from_config(cfg.robot)
            robot.connect()
            with contextlib.suppress(KeyboardInterrupt):
                arx5_teleop_loop(
                    robot=robot,
                    fps=cfg.fps,
                    display_data=cfg.display_data,
                    duration=cfg.teleop_time_s,
                    debug_timing=cfg.debug_timing,
                )

        # --- flexiv_rizon4_rt + spacemouse ---
        elif cfg.robot.type == "flexiv_rizon4_rt" and cfg.teleop.type == "spacemouse":
            logger.info("Detected Flexiv Rizon4 RT + SpaceMouse")
            robot = make_robot_from_config(cfg.robot)
            robot.connect(go_to_start=True)
            start_obs = robot.get_observation()
            tcp_keys = [k for k in start_obs if k.startswith("tcp.")]
            logger.info("Start pose: " + ", ".join(f"{k}={start_obs[k]:.6f}" for k in tcp_keys))
            teleop = make_teleoperator_from_config(cfg.teleop)
            teleop.connect(current_tcp_pose_euler=robot.get_current_tcp_pose_euler())
            try:
                spacemouse_teleop_loop(
                    teleop=teleop,
                    robot=robot,
                    fps=cfg.fps,
                    display_data=cfg.display_data,
                    duration=cfg.teleop_time_s,
                    dryrun=cfg.dryrun,
                    debug_timing=cfg.debug_timing,
                )
            except KeyboardInterrupt:
                logger.info("Teleoperation interrupted by user")

        # --- elite_cs66_rt + spacemouse ---
        elif cfg.robot.type == "elite_cs66_rt" and cfg.teleop.type == "spacemouse":
            logger.info("Detected Elite CS66 + SpaceMouse")
            robot = make_robot_from_config(cfg.robot)
            robot.connect(go_to_start=True)
            start_obs = robot.get_observation()
            tcp_keys = [k for k in start_obs if k.startswith("tcp.")]
            logger.info("Start pose: " + ", ".join(f"{k}={start_obs[k]:.6f}" for k in tcp_keys))
            teleop = make_teleoperator_from_config(cfg.teleop)
            teleop.connect(current_tcp_pose_euler=robot.get_current_tcp_pose_euler())
            try:
                elite_cs66_rt_spacemouse_teleop_loop(
                    teleop=teleop,
                    robot=robot,
                    fps=cfg.fps,
                    display_data=cfg.display_data,
                    duration=cfg.teleop_time_s,
                    dryrun=cfg.dryrun,
                    debug_timing=cfg.debug_timing,
                )
            except KeyboardInterrupt:
                logger.info("Teleoperation interrupted by user")

        # --- elite_cs66_rt + pico4 ---
        elif cfg.robot.type == "elite_cs66_rt" and cfg.teleop.type == "pico4":
            logger.info("Detected Elite CS66 + Pico4")
            robot = make_robot_from_config(cfg.robot)
            robot.connect(go_to_start=True)
            start_obs = robot.get_observation()
            tcp_keys = [k for k in start_obs if k.startswith("tcp.")]
            logger.info("Start pose: " + ", ".join(f"{k}={start_obs[k]:.6f}" for k in tcp_keys))
            teleop = make_teleoperator_from_config(cfg.teleop)
            teleop.connect(current_tcp_pose_quat=robot.get_current_tcp_pose_quat())
            try:
                elite_cs66_rt_pico4_teleop_loop(
                    teleop=teleop,
                    robot=robot,
                    fps=cfg.fps,
                    display_data=cfg.display_data,
                    duration=cfg.teleop_time_s,
                    dryrun=cfg.dryrun,
                    debug_timing=cfg.debug_timing,
                )
            except KeyboardInterrupt:
                logger.info("Teleoperation interrupted by user")

        # --- flexiv_rizon4_rt + pico4 ---
        elif cfg.robot.type == "flexiv_rizon4_rt" and cfg.teleop.type == "pico4":
            logger.info("Detected Flexiv Rizon4 RT + Pico4")
            robot = make_robot_from_config(cfg.robot)
            robot.connect(go_to_start=True)
            start_obs = robot.get_observation()
            tcp_keys = [k for k in start_obs if k.startswith("tcp.")]
            logger.info("Start pose: " + ", ".join(f"{k}={start_obs[k]:.6f}" for k in tcp_keys))
            teleop = make_teleoperator_from_config(cfg.teleop)
            teleop.connect(current_tcp_pose_quat=robot.get_current_tcp_pose_quat())
            try:
                pico4_teleop_loop(
                    teleop=teleop,
                    robot=robot,
                    fps=cfg.fps,
                    display_data=cfg.display_data,
                    duration=cfg.teleop_time_s,
                    dryrun=cfg.dryrun,
                )
            except KeyboardInterrupt:
                logger.info("Teleoperation interrupted by user")

        # --- bi_flexiv_rizon4_rt + bi_pico4 ---
        elif cfg.robot.type == "bi_flexiv_rizon4_rt" and cfg.teleop.type == "bi_pico4":
            logger.info("Detected BiFlexivRizon4RT + BiPico4")
            robot = make_robot_from_config(cfg.robot)
            teleop = make_teleoperator_from_config(cfg.teleop)

            # Bring the VR side up FIRST, then the robot. pre_init caps out at
            # ~3s (0.5s + 25 polls), while robot.connect() takes 20-40s and moves
            # both arms to the start pose partway through. Running them in
            # parallel looked like free overlap, but the executor's __exit__
            # calls shutdown(wait=True), so a pre_init failure could not
            # propagate until the robot future finished — the arms travelled for
            # a session that was already dead. Observed on the bench: pre_init
            # raised at +3s, the error surfaced 22s later, 19s after the arms had
            # moved. Losing at most 3s of overlap is the cheaper trade.
            try:
                teleop.pre_init()
                robot.connect(go_to_start=True)
            except KeyboardInterrupt:
                logger.info("Startup interrupted by user")
                raise

            left_pose, right_pose = robot.get_current_tcp_pose_quat()
            logger.info(f"Left start pose:  {left_pose}")
            logger.info(f"Right start pose: {right_pose}")
            teleop.connect(left_tcp_pose_quat=left_pose, right_tcp_pose_quat=right_pose)
            try:
                bi_pico4_teleop_loop(
                    teleop=teleop,
                    robot=robot,
                    fps=cfg.fps,
                    display_data=cfg.display_data,
                    duration=cfg.teleop_time_s,
                    dryrun=cfg.dryrun,
                    debug_timing=cfg.debug_timing,
                )
            except KeyboardInterrupt:
                logger.info("Teleoperation interrupted by user")

        # --- bi_elite_cs66_rt + bi_pico4 ---
        elif cfg.robot.type == "bi_elite_cs66_rt" and cfg.teleop.type == "bi_pico4":
            logger.info("Detected BiEliteCS66RT + BiPico4")
            robot = make_robot_from_config(cfg.robot)
            # Elite CS66 wrists have tighter joint-velocity limits than Flexiv;
            # default the VR rotation sensitivity down so controller jitter / fast
            # hand rotation can't drive a wrist-singularity over-speed trip. Lowered
            # 0.5 -> 0.3 on 2026-07-03: at 0.5 a fast wrist flick still spiked joint
            # velocity past the controller's 30 rad/s limit and dropped external
            # control ("writeServoj failed 251 ticks"). 0.3 = the arm rotates 30% of
            # the hand rotation, keeping the mapped joint velocity well under the trip.
            # An explicit --teleop.ori_sensitivity wins (only the 1.0 default is replaced).
            if cfg.teleop.ori_sensitivity == 1.0:
                cfg.teleop.ori_sensitivity = 0.3
                logger.info(
                    "BiElite: defaulting teleop ori_sensitivity to 0.3 (pass --teleop.ori_sensitivity to override)"
                )
            # The Pico4 leader's rotation rate limit defaults to the Flexiv-era 6.28 rad/s
            # (360 deg/s) — a value the 1 kHz Flexiv follower smooths away, but the 250 Hz
            # Elite cannot near a wrist singularity (a ~270 deg/s flick dropped external
            # control). Default it to 1.0 rad/s (~57 deg/s) for Elite; an explicit
            # --teleop.max_rot_velocity wins (only the untouched 6.28 default is replaced).
            if cfg.teleop.max_rot_velocity == 6.28:
                cfg.teleop.max_rot_velocity = 1.0
                logger.info(
                    "BiElite: defaulting teleop max_rot_velocity to 1.0 rad/s "
                    "(pass --teleop.max_rot_velocity to override)"
                )
            teleop = make_teleoperator_from_config(cfg.teleop)

            # Bring the VR side up FIRST, then the robot. pre_init caps out at
            # ~3s (0.5s + 25 polls), while robot.connect() takes 20-40s and moves
            # both arms to the start pose partway through. Running them in
            # parallel looked like free overlap, but the executor's __exit__
            # calls shutdown(wait=True), so a pre_init failure could not
            # propagate until the robot future finished — the arms travelled for
            # a session that was already dead. Observed on the bench: pre_init
            # raised at +3s, the error surfaced 22s later, 19s after the arms had
            # moved. Losing at most 3s of overlap is the cheaper trade.
            try:
                teleop.pre_init()
                robot.connect(go_to_start=True)
            except KeyboardInterrupt:
                logger.info("Startup interrupted by user")
                raise

            left_pose, right_pose = robot.get_current_tcp_pose_quat()
            logger.info(f"Left start pose:  {left_pose}")
            logger.info(f"Right start pose: {right_pose}")
            teleop.connect(left_tcp_pose_quat=left_pose, right_tcp_pose_quat=right_pose)
            try:
                bi_pico4_teleop_loop(
                    teleop=teleop,
                    robot=robot,
                    fps=cfg.fps,
                    display_data=cfg.display_data,
                    duration=cfg.teleop_time_s,
                    dryrun=cfg.dryrun,
                    debug_timing=cfg.debug_timing,
                )
            except KeyboardInterrupt:
                logger.info("Teleoperation interrupted by user")

        # ======================== Mock Robot ========================
        elif cfg.robot.type == "mock_robot":
            logger.info("Detected mock robot, using mock teleop loop")
            robot = make_robot_from_config(cfg.robot)
            robot.connect()
            teleop = make_teleoperator_from_config(cfg.teleop)
            teleop.connect()
            with contextlib.suppress(KeyboardInterrupt):
                mock_robot_teleop_loop(
                    teleop=teleop,
                    robot=robot,
                    fps=cfg.fps,
                    display_data=cfg.display_data,
                    duration=cfg.teleop_time_s,
                    dryrun=cfg.dryrun,
                    debug_timing=cfg.debug_timing,
                )

        # --- generic fallback ---
        else:
            teleop = make_teleoperator_from_config(cfg.teleop)
            robot = make_robot_from_config(cfg.robot)
            teleop.connect()
            robot.connect()
            with contextlib.suppress(KeyboardInterrupt):
                teleop_loop(
                    teleop=teleop,
                    robot=robot,
                    fps=cfg.fps,
                    display_data=cfg.display_data,
                    duration=cfg.teleop_time_s,
                    display_compressed_images=display_compressed_images,
                    debug_timing=cfg.debug_timing,
                )

    except Exception as e:
        logger.error(f"Error in teleoperation: {e}\n{traceback.format_exc()}")
    finally:
        _cleanup(robot, teleop, cfg.display_data)


def main():
    register_third_party_plugins()
    teleoperate()


if __name__ == "__main__":
    main()
