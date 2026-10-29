#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Test doubles for driving the device-specific teleop and record loops.

These loops only ever run against real arms, so nothing checks them. The doubles
here stand in for a robot and a teleoperator, record every call the loop makes,
and stop it after a fixed number of iterations — so a loop's behaviour becomes a
comparable trace instead of something you can only observe on a bench.

Iteration count is controlled by raising ``StopLoopError`` from ``get_observation``
rather than by the loops' own ``duration`` argument: wall-clock termination makes
the number of iterations depend on how fast the machine is, which is exactly the
kind of flakiness a characterization test cannot afford.
"""

from __future__ import annotations

import numpy as np

from lerobot.teleoperators import Teleoperator


class StopLoopError(Exception):
    """Raised by a double to end a loop after a set number of iterations."""


def run_loop(loop, *, iterations: int, **kwargs) -> None:
    """Drive ``loop`` for exactly ``iterations`` passes, then return."""
    try:
        loop(**kwargs)
    except StopLoopError:
        return
    raise AssertionError(
        f"{loop.__name__} returned on its own before {iterations} iterations — the double should have stopped it"
    )


class FakeRobot:
    """A robot that answers every call the loops make, and remembers them."""

    def __init__(
        self,
        *,
        iterations: int,
        action_features=None,
        rt_moving_frames=(),
        finished_reset_frames=(),
        name: str = "fake_robot",
        bimanual: bool = False,
    ):
        self._iterations = iterations
        self._obs_count = 0
        # Frames (0-based) on which rt_moving reports True, so a test can drive
        # the "background trajectory owns the arm" branch deterministically.
        self._rt_moving_frames = set(rt_moving_frames)
        # Frames on which a reset finishes as counted by reset_generation (the
        # Elite completion signal), whether or not rt_moving was seen True.
        self._finished_reset_frames = set(finished_reset_frames)
        self.action_features = action_features or {
            "tcp.x": float,
            "tcp.y": float,
            "tcp.z": float,
            "tcp.r1": float,
            "tcp.r2": float,
            "tcp.r3": float,
            "tcp.r4": float,
            "tcp.r5": float,
            "tcp.r6": float,
            "gripper.pos": float,
        }
        # `name` is not cosmetic: pico4_teleop_loop gates its whole rt_moving
        # branch on `robot.name == "flexiv_rizon4_rt"`.
        self.name = name
        # Bimanual arms return (left, right) from get_current_tcp_pose_quat;
        # single arms return one flat array. The loops unpack accordingly.
        self._bimanual = bimanual
        self.id = "fake"
        self.calls: list[tuple] = []

    # -- observation / action ------------------------------------------------
    def get_observation(self) -> dict:
        if self._obs_count >= self._iterations:
            raise StopLoopError
        self._obs_count += 1
        self.calls.append(("get_observation",))
        return dict.fromkeys(self.action_features, 0.0)

    def send_action(self, action: dict) -> dict:
        self.calls.append(("send_action", dict(action)))
        return action

    # -- RT state ------------------------------------------------------------
    @property
    def rt_moving(self) -> bool:
        # _obs_count has already been incremented for the current frame.
        return (self._obs_count - 1) in self._rt_moving_frames

    @property
    def reset_generation(self) -> int:
        return sum(1 for frame in self._finished_reset_frames if frame <= self._obs_count - 1)

    def reset_to_initial_position(self) -> None:
        self.calls.append(("reset_to_initial_position",))

    # -- pose readback -------------------------------------------------------
    def get_current_tcp_pose_quat(self):
        self.calls.append(("get_current_tcp_pose_quat",))
        one = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0, 0.5])
        return (one, one.copy()) if self._bimanual else one

    # spacemouse_teleop_loop (the Flexiv one) reaches for these three as well.
    _spacemouse_arm_side = "left"

    def get_start_eef_pose(self):
        self.calls.append(("get_start_eef_pose",))
        return np.zeros(7)

    def smooth_go_start(self, *args, **kwargs) -> None:
        self.calls.append(("smooth_go_start",))

    def get_current_tcp_pose_euler(self):
        self.calls.append(("get_current_tcp_pose_euler",))
        return np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.5])

    # Deliberately NOT aliased to other spellings. An alias here would let the
    # loop reach for a getter no real robot defines and still pass: the call is
    # recorded under whichever name the aliased function hardcodes, so the
    # "which pose flavour" test would assert nothing. A wrong name must raise.

    # -- trace helpers -------------------------------------------------------
    def names(self) -> list[str]:
        """Call names only — the shape of what happened, without the payloads."""
        return [c[0] for c in self.calls]

    def sent_actions(self) -> list[dict]:
        return [c[1] for c in self.calls if c[0] == "send_action"]


class FakeTeleop:
    """Baseline teleoperator: emits a fixed Cartesian action, no buttons."""

    name = "fake_teleop"

    def __init__(self, *, reset_frames=()):
        self._frame = -1
        self._reset_frames = set(reset_frames)
        self.calls: list[tuple] = []
        # Attributes the status lines read via getattr with defaults.
        self._enabled = True
        self._orientation_control_active = False
        self._last_grip = 0.25
        self._start_pose_6d = np.zeros(6)
        self._start_gripper_pos = 0.0
        self._target_pose_6d = np.zeros(6)

    def get_action(self) -> dict:
        self._frame += 1
        self.calls.append(("get_action",))
        return {
            "tcp.x": 0.11,
            "tcp.y": 0.22,
            "tcp.z": 0.33,
            "tcp.r1": 1.0,
            "tcp.r2": 0.0,
            "tcp.r3": 0.0,
            "tcp.r4": 0.0,
            "tcp.r5": 1.0,
            "tcp.r6": 0.0,
            "gripper.pos": 0.75,
        }

    def get_reset_button(self) -> bool:
        return self._frame in self._reset_frames

    def reset_to_pose(self, pose, gripper) -> None:
        self.calls.append(("reset_to_pose", tuple(np.asarray(pose).tolist()), float(gripper)))

    def names(self) -> list[str]:
        return [c[0] for c in self.calls]


class _FakeSpaceMouseDevice:
    def __init__(self, both_pressed_frames=()):
        self._frame = -1
        self._both = set(both_pressed_frames)

    def _tick(self) -> bool:
        return self._frame in self._both

    def is_left_button_pressed(self) -> bool:
        self._frame += 1  # left is polled first each pass
        return self._tick()

    def is_right_button_pressed(self) -> bool:
        return self._tick()


class FakeSpaceMouseTeleop(FakeTeleop):
    """SpaceMouse teleop: reset comes from both buttons, not a method."""

    name = "spacemouse"

    def __init__(self, *, both_pressed_frames=()):
        super().__init__()
        self._spacemouse = _FakeSpaceMouseDevice(both_pressed_frames)


class _FakePico4Side:
    def __init__(self):
        self._enabled = True
        self._last_grip = 0.4
        self._orientation_control_active = False


class FakeBiPico4Teleop(FakeTeleop):
    """Bimanual Pico4: the status line reads per-side handles."""

    name = "bi_pico4"

    def __init__(self, *, reset_frames=()):
        super().__init__(reset_frames=reset_frames)
        self._left_pico4 = _FakePico4Side()
        self._right_pico4 = _FakePico4Side()

    def get_action(self) -> dict:
        act = super().get_action()
        return {f"{side}_{k}": v for side in ("left", "right") for k, v in act.items()}

    def reset_to_pose(self, left_pose, right_pose, left_grip, right_grip) -> None:
        """Bimanual re-sync takes both sides at once, unlike the single-arm form."""
        self.calls.append(
            (
                "reset_to_pose",
                tuple(np.asarray(left_pose).tolist()),
                tuple(np.asarray(right_pose).tolist()),
                float(left_grip),
                float(right_grip),
            )
        )


class FakeDataset:
    """Records what a record loop writes, without touching disk."""

    def __init__(self, fps: int = 30, features=None):
        self.fps = fps
        self.features = features or {}
        self.frames: list[dict] = []

    def add_frame(self, frame: dict, task=None, **kwargs) -> None:
        self.frames.append(dict(frame))

    def action_keys(self) -> list[set]:
        """Per frame, which action keys were written — the shape datasets carry."""
        return [{k for k in f if k.startswith("action")} for f in self.frames]


def fresh_events() -> dict:
    """The event flags a record loop polls, all clear."""
    return {
        "exit_early": False,
        "go_start": False,
        "rerecord_episode": False,
        "stop_recording": False,
    }


# The record loops branch on `isinstance(teleop, Teleoperator)` to tell a driven
# session from an observation-only one. Registering as virtual subclasses makes
# that check pass without inheriting Teleoperator.__init__, which would create
# calibration directories on disk just to run a test.
for _double in (FakeTeleop, FakeSpaceMouseTeleop, FakeBiPico4Teleop):
    Teleoperator.register(_double)
