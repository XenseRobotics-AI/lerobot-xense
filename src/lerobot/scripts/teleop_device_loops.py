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

"""One Cartesian teleop loop, plus the per-rig differences it delegates.

Every Cartesian rig ran its own copy of the same loop — read an observation, ask
the teleoperator, honour a reset, stand aside while the arm runs its own
trajectory, send, draw a status line, sleep. The copies had drifted to 0.5-0.8
similarity, which is the worst place to be: close enough that a fix looks like it
applies everywhere, different enough that it does not.

What actually varies is small and now lives in ``DevicePolicy``:

  * how a reset is asked for — a method call, or two buttons on a rising edge
  * which pose flavour the teleoperator is re-synced from — quaternion or euler,
    one arm or two
  * where a dry-run reset snaps to — the robot's pose, or one the teleop holds
  * what the status line says

Behaviour is deliberately preserved down to the frame, including the one that
looks like an oversight: when the arm stops moving, the loop re-syncs and starts
the next frame rather than sending a target computed before the re-sync. Sending
there would command from a stale pose. See
``tests/scripts/test_teleop_loop_behaviour.py``, which pins this down.
"""

from __future__ import annotations

import time
import traceback
from typing import Any, Protocol

import numpy as np

from lerobot.grippers.taccap.taccap_follower import get_taccap_status_lines
from lerobot.robots import Robot
from lerobot.teleoperators import Teleoperator
from lerobot.utils.robot_utils import get_logger
from lerobot.utils.utils import move_cursor_up
from lerobot.utils.visualization_utils import log_rerun_data

logger = get_logger("Teleoperate")


def precise_sleep(seconds: float) -> None:
    """Imported lazily to keep this module's import graph free of the CLI."""
    from lerobot.utils.robot_utils import precise_sleep as _sleep

    _sleep(seconds)


class DevicePolicy(Protocol):
    """The rig-specific half of a Cartesian teleop loop."""

    #: Used in the reset log line, e.g. "A button" / "both buttons".
    reset_source: str

    def reset_requested(self, teleop: Teleoperator) -> bool:
        """True on the frame a reset is asked for (edge, not level)."""

    def resync(self, robot: Robot, teleop: Teleoperator) -> None:
        """Snap the teleoperator's target to where the arm actually is."""

    def dryrun_resync(self, robot: Robot, teleop: Teleoperator) -> None:
        """Same, for a dry run, where the arm is not commanded at all."""

    def status_line(self, teleop: Teleoperator, action: dict, loop_s: float, dryrun: bool) -> str:
        """The one-line readout printed when Rerun display is off."""

    def before_send(self, robot: Robot, teleop: Teleoperator, action: dict, fps: int) -> dict:
        """Last look at the action before it is sent. Returns what to send."""


# --------------------------------------------------------------------------- #
# Pose re-sync helpers — the two flavours rigs speak
# --------------------------------------------------------------------------- #


def _resync_from_quat(robot: Robot, teleop: Teleoperator) -> None:
    pose = robot.get_current_tcp_pose_quat()
    teleop.reset_to_pose(pose[:7], pose[7])


def _resync_from_euler(robot: Robot, teleop: Teleoperator) -> np.ndarray:
    # Prefer the commanded pose when the arm exposes one: it is where the
    # controller believes it is heading, which is what the teleop integrates
    # from. Near an orientation singularity the measured rotvec is unstable and
    # can read 100°+ off what has actually been commanded, which would make the
    # next send_action jump and trip the joint-velocity limit. Only elite_cs66_rt
    # and bi_elite_cs66_rt expose the commanded pose; flexiv and arx5 fall back
    # to the measured one and pay a servo cycle of lag.
    source = getattr(robot, "get_commanded_tcp_pose_euler", None) or robot.get_current_tcp_pose_euler
    pose = source()
    teleop.reset_to_pose(pose[:6], pose[6])
    return pose


def _resync_bimanual(robot: Robot, teleop: Teleoperator) -> None:
    left_pose, right_pose = robot.get_current_tcp_pose_quat()
    teleop.reset_to_pose(left_pose[:7], right_pose[:7], left_pose[7], right_pose[7])


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #


def run_cartesian_teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    policy: DevicePolicy,
    *,
    display_data: bool = False,
    duration: float | None = None,
    dryrun: bool = False,
    debug_timing: bool = False,
    honour_rt_moving: bool = True,
) -> None:
    """Drive ``robot`` from ``teleop`` at ``fps``, deferring rig quirks to ``policy``.

    Args:
        honour_rt_moving: Whether to stand aside while ``robot.rt_moving``. Rigs
            without a background servo loop have no such state; the loops that
            gated this on the robot's name pass False for anything else.
    """
    from lerobot.scripts.lerobot_teleoperate import _print_obs_state

    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()

    prev_rt_moving = False
    reset_display_cleared = False

    while True:
        loop_start = time.perf_counter()

        obs = robot.get_observation()
        raw_action = teleop.get_action()

        # --- reset ---------------------------------------------------------
        if policy.reset_requested(teleop):
            if dryrun:
                logger.info(f"[DRYRUN] Reset to initial position ({policy.reset_source})")
                policy.dryrun_resync(robot, teleop)
            elif hasattr(robot, "reset_to_initial_position"):
                try:
                    robot.reset_to_initial_position()
                    logger.info(f"Reset to initial position triggered by {policy.reset_source}")
                except Exception as e:
                    logger.error(f"Failed to reset robot position: {e}\n{traceback.format_exc()}")
            if display_data and obs is not None:
                log_rerun_data(observation=obs)
            if obs is not None:
                if not reset_display_cleared:
                    print("\033[2J\033[H", end="", flush=True)
                    reset_display_cleared = True
                _print_obs_state(obs, display_len, "RESETTING", extra_lines=get_taccap_status_lines())
            _sleep_to_fps(loop_start, fps)
            continue

        # --- the arm is running its own trajectory --------------------------
        if honour_rt_moving and getattr(robot, "rt_moving", False):
            if display_data and obs is not None:
                log_rerun_data(observation=obs)
            if obs is not None:
                if not reset_display_cleared:
                    print("\033[2J\033[H", end="", flush=True)
                    reset_display_cleared = True
                _print_obs_state(obs, display_len, "MOVING", extra_lines=get_taccap_status_lines())
            prev_rt_moving = True
            _sleep_to_fps(loop_start, fps)
            continue

        # --- it just stopped: re-sync, and start the next frame clean -------
        if prev_rt_moving:
            prev_rt_moving = False
            reset_display_cleared = False
            try:
                policy.resync(robot, teleop)
                logger.info("Teleop synced to robot pose after reset complete")
            except Exception as e:
                logger.error(f"Failed to sync teleop after reset: {e}")
            _sleep_to_fps(loop_start, fps)
            continue

        # --- normal frame ---------------------------------------------------
        action = policy.before_send(robot, teleop, raw_action, fps)
        if not dryrun:
            robot.send_action(action)

        if display_data:
            log_rerun_data(observation=obs, action=action)
            if not debug_timing:
                print("\n" + "-" * (display_len + 10))
                print(f"{'NAME':<{display_len}} | {'NORM':>7}")
                for motor, value in action.items():
                    print(f"{motor:<{display_len}} | {value:>7.4f}")
                status_lines = get_taccap_status_lines()
                for line in status_lines:
                    print(f"\033[2K{line}")
                # Rewind the exact panel height: blank/separator + header + N rows.
                # Two extra rows made the panel climb into asynchronous logs
                # and eventually overwrite its own TCP values.
                move_cursor_up(len(action) + len(status_lines) + 3)

        dt_s = time.perf_counter() - loop_start
        precise_sleep(max(1 / fps - dt_s, 0))
        loop_s = time.perf_counter() - loop_start

        if not display_data:
            print(
                f"\r\033[K{policy.status_line(teleop, action, loop_s, dryrun)}",
                end="",
                flush=True,
            )

        if duration is not None and time.perf_counter() - start >= duration:
            return


def _sleep_to_fps(loop_start: float, fps: int) -> None:
    precise_sleep(max(1 / fps - (time.perf_counter() - loop_start), 0))


# --------------------------------------------------------------------------- #
# Policies
# --------------------------------------------------------------------------- #


class _NoBeforeSend:
    """Most rigs send what the teleoperator produced, unchanged."""

    def before_send(self, robot: Robot, teleop: Teleoperator, action: dict, fps: int) -> dict:
        return action


class Pico4Policy(_NoBeforeSend):
    """Single-arm Pico4: A button resets, poses are quaternions."""

    reset_source = "A button"

    def reset_requested(self, teleop: Teleoperator) -> bool:
        return bool(teleop.get_reset_button())

    def resync(self, robot: Robot, teleop: Teleoperator) -> None:
        _resync_from_quat(robot, teleop)

    dryrun_resync = resync

    def status_line(self, teleop, action, loop_s, dryrun) -> str:
        enabled = "ENABLED" if getattr(teleop, "_enabled", False) else "DISABLED"
        ori = "ORI:ON" if getattr(teleop, "_orientation_control_active", False) else "ORI:OFF"
        grip = f"grip={getattr(teleop, '_last_grip', 0.0):.2f}"
        gripper_pos = f"gripper={action.get('gripper.pos', 0.0):.2f}"
        flag = "[DRYRUN] | " if dryrun else ""
        return f"{loop_s * 1e3:5.1f}ms ({1 / loop_s:3.0f}Hz) | {flag}{enabled} | {grip} | {gripper_pos} | {ori}"


class BiPico4Policy(Pico4Policy):
    """Bimanual Pico4: both sides re-sync together, status shows each side."""

    def resync(self, robot: Robot, teleop: Teleoperator) -> None:
        _resync_bimanual(robot, teleop)

    dryrun_resync = resync

    def status_line(self, teleop, action, loop_s, dryrun) -> str:
        def side(handle) -> str:
            on = "ON " if getattr(handle, "_enabled", False) else "OFF"
            return f"{on} grip={getattr(handle, '_last_grip', 0.0):.2f}"

        flag = "[DRYRUN] | " if dryrun else ""
        return (
            f"{loop_s * 1e3:5.1f}ms ({1 / loop_s:3.0f}Hz) | {flag}"
            f"L:{side(getattr(teleop, '_left_pico4', None))} | "
            f"R:{side(getattr(teleop, '_right_pico4', None))}"
        )


class SpaceMousePolicy:
    """SpaceMouse: both buttons on the rising edge, euler poses.

    A dry-run reset snaps to the start pose the teleoperator holds rather than
    asking the arm: the stick integrates deflection from that pose, so that is
    what "back to the start" means for this device.
    """

    reset_source = "both buttons"

    def __init__(self, release_resync_idle_s: float = 0.5) -> None:
        self._both_prev = False
        self._release_resync_idle_s = release_resync_idle_s
        self._idle_frames = 0
        self._just_resynced = False

    def reset_requested(self, teleop: Teleoperator) -> bool:
        both = teleop._spacemouse.is_left_button_pressed() and teleop._spacemouse.is_right_button_pressed()
        rising = both and not self._both_prev
        self._both_prev = both
        return rising

    def dryrun_resync(self, robot: Robot, teleop: Teleoperator) -> None:
        teleop.reset_to_pose(teleop._start_pose_6d, teleop._start_gripper_pos)

    def before_send(self, robot: Robot, teleop: Teleoperator, action: dict, fps: int) -> dict:
        """Snap the target back to the arm once the stick has been still.

        The stick integrates deflection into an absolute pose accumulator. Push
        it faster than the controller can follow and the accumulator runs ahead;
        let go, and the arm keeps drifting toward a target the operator has
        abandoned. After ``release_resync_idle_s`` of no deflection we snap the
        accumulator to where the arm actually is, so the next push starts from
        "here" — and re-ask the teleoperator, since its answer changed.
        """
        if getattr(teleop, "_enabled", True):
            self._idle_frames = 0
            self._just_resynced = False
            return action

        self._idle_frames += 1
        idle_limit = max(1, int(round(fps * self._release_resync_idle_s)))
        if self._idle_frames < idle_limit or self._just_resynced:
            return action

        try:
            _resync_from_euler(robot, teleop)
            self._just_resynced = True
            logger.info(f"Idle for {self._release_resync_idle_s:.2f}s — snapped SpaceMouse target to robot TCP")
            return teleop.get_action()
        except Exception as e:
            logger.error(f"Idle resync failed: {e}")
            return action

    def resync(self, robot: Robot, teleop: Teleoperator) -> None:
        pose = _resync_from_euler(robot, teleop)
        # Keep the accumulator's origin in step, or the next push integrates
        # from the pose the stick started at rather than from here.
        teleop._start_pose_6d = np.asarray(pose[:6]).copy()
        teleop._start_gripper_pos = pose[6]
        self._idle_frames = 0

    def status_line(self, teleop, action, loop_s, dryrun) -> str:
        pos = (action.get("tcp.x", 0.0), action.get("tcp.y", 0.0), action.get("tcp.z", 0.0))
        internal = getattr(teleop, "_target_pose_6d", None)
        rpy: tuple[Any, ...] = (
            tuple(float(internal[i]) for i in (3, 4, 5))
            if internal is not None and len(internal) >= 6
            else (0.0, 0.0, 0.0)
        )
        flag = "[DRYRUN] " if dryrun else ""
        return (
            f"{loop_s * 1e3:5.1f}ms ({1 / loop_s:3.0f}Hz) | {flag}"
            f"pos=[{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}] | "
            f"rpy=[{rpy[0]:+.3f}, {rpy[1]:+.3f}, {rpy[2]:+.3f}] | "
            f"grip={action.get('gripper.pos', 0.0):.2f}"
        )
