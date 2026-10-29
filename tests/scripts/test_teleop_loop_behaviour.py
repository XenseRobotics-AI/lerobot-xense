#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Characterization tests for the device-specific teleop loops.

These pin down what each loop *does*: the order it calls the robot in, when it
sends an action and when it declines to, what a reset trigger does, and which
pose flavour it re-syncs the teleoperator from. They record current behaviour,
not desired behaviour — their job is to make a refactor of these loops checkable
without an arm on the bench, which is the only reason the duplication between
them has survived this long.

A failure here after a refactor means robot behaviour changed. That is the
signal: decide deliberately, do not just update the expectation.
"""

import pytest

from lerobot.scripts import lerobot_teleoperate as teleop_mod
from tests.scripts.conftest_teleop_doubles import (
    FakeBiPico4Teleop,
    FakeRobot,
    FakeSpaceMouseTeleop,
    FakeTeleop,
    run_loop,
)

ITERS = 4


class Case:
    """How to assemble a double pair that a given loop will actually drive."""

    def __init__(self, loop, teleop_cls, robot_kwargs=None, reset_kwarg="reset_frames"):
        self.loop, self.teleop_cls = loop, teleop_cls
        self.robot_kwargs = robot_kwargs or {}
        self.reset_kwarg = reset_kwarg

    def robot(self, **extra):
        return FakeRobot(iterations=ITERS, **{**self.robot_kwargs, **extra})

    def teleop(self, *, reset_on=()):
        return self.teleop_cls(**({self.reset_kwarg: set(reset_on)} if reset_on else {}))

    def run(self, robot, teleop, **kwargs):
        run_loop(
            getattr(teleop_mod, self.loop),
            iterations=ITERS,
            teleop=teleop,
            robot=robot,
            fps=1000,
            display_data=False,
            **kwargs,
        )


CASES = [
    Case("elite_cs66_rt_pico4_teleop_loop", FakeTeleop),
    # pico4_teleop_loop gates its whole rt_moving branch on the robot's name.
    Case("pico4_teleop_loop", FakeTeleop, {"name": "flexiv_rizon4_rt"}),
    Case("elite_cs66_rt_spacemouse_teleop_loop", FakeSpaceMouseTeleop, reset_kwarg="both_pressed_frames"),
    Case("bi_pico4_teleop_loop", FakeBiPico4Teleop, {"bimanual": True}),
    # The Flexiv SpaceMouse loop gates on the robot name the same way.
    Case(
        "spacemouse_teleop_loop", FakeSpaceMouseTeleop, {"name": "flexiv_rizon4_rt"}, reset_kwarg="both_pressed_frames"
    ),
]
IDS = [c.loop for c in CASES]


# --------------------------------------------------------------------------- #
# The shared skeleton
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_one_observation_and_one_action_per_frame(case):
    robot, teleop = case.robot(), case.teleop()

    case.run(robot, teleop)

    assert robot.names().count("get_observation") == ITERS
    assert robot.names().count("send_action") == ITERS
    assert teleop.names().count("get_action") == ITERS


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_dryrun_reads_and_asks_but_never_commands(case):
    robot, teleop = case.robot(), case.teleop()

    case.run(robot, teleop, dryrun=True)

    assert robot.names().count("get_observation") == ITERS
    assert teleop.names().count("get_action") == ITERS
    assert "send_action" not in robot.names()


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_the_action_reaches_the_robot_unchanged(case):
    """These rigs consume the teleop schema as-is; none of them reshape it."""
    robot, teleop = case.robot(), case.teleop()
    expected = case.teleop().get_action()

    case.run(robot, teleop)

    assert robot.sent_actions() == [expected] * ITERS


# --------------------------------------------------------------------------- #
# A moving arm owns itself
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_no_action_is_sent_while_the_arm_runs_its_own_trajectory(case):
    """Sending during rt_moving would fight the background trajectory.

    Two frames are skipped for the motion itself, and a third goes to re-syncing
    the teleoperator once it ends — so with 4 frames exactly one send survives.
    That third frame is behaviour, not an accident: the loop `continue`s after
    the re-sync rather than sending a target computed before it.
    """
    robot, teleop = case.robot(rt_moving_frames={1, 2}), case.teleop()

    case.run(robot, teleop)

    assert robot.names().count("send_action") == 1
    assert robot.names().count("get_observation") == ITERS


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_the_teleop_is_resynced_once_the_trajectory_ends(case):
    """Without this the next send would jump from a stale target."""
    robot, teleop = case.robot(rt_moving_frames={1, 2}), case.teleop()

    case.run(robot, teleop)

    assert teleop.names().count("reset_to_pose") == 1


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_a_reset_seen_only_as_a_finished_count_still_resyncs(case):
    """A reset that finishes between two rt_moving reads (or fails fast) never
    reads as moving; the reset_generation counter must still trigger the re-sync,
    and the frame that re-syncs must not send a target computed before it."""
    robot, teleop = case.robot(finished_reset_frames={1}), case.teleop()

    case.run(robot, teleop)

    assert teleop.names().count("reset_to_pose") == 1
    assert robot.names().count("send_action") == ITERS - 1


@pytest.mark.parametrize(
    ("case", "expected_getter"),
    [
        (CASES[0], "get_current_tcp_pose_quat"),
        (CASES[1], "get_current_tcp_pose_quat"),
        (CASES[2], "get_current_tcp_pose_euler"),
        (CASES[3], "get_current_tcp_pose_quat"),
        (CASES[4], "get_current_tcp_pose_euler"),
    ],
    ids=IDS,
)
def test_resync_reads_the_pose_flavour_that_rig_speaks(case, expected_getter):
    """A rig re-synced from the wrong pose flavour jumps on the next send."""
    robot, teleop = case.robot(rt_moving_frames={1, 2}), case.teleop()

    case.run(robot, teleop)

    assert expected_getter in robot.names()


# --------------------------------------------------------------------------- #
# Reset triggers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_the_reset_trigger_sends_the_arm_home(case):
    robot, teleop = case.robot(), case.teleop(reset_on={1})

    case.run(robot, teleop)

    assert robot.names().count("reset_to_initial_position") == 1


def test_spacemouse_reset_fires_on_the_press_not_while_held():
    """Both buttons held across frames must trigger one reset, not one per frame."""
    case = CASES[2]
    robot, teleop = case.robot(), case.teleop(reset_on={1, 2, 3})

    case.run(robot, teleop)

    assert robot.names().count("reset_to_initial_position") == 1


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_dryrun_reset_resyncs_the_teleop_instead_of_moving_the_arm(case):
    robot, teleop = case.robot(), case.teleop(reset_on={1})

    case.run(robot, teleop, dryrun=True)

    assert "reset_to_initial_position" not in robot.names()
    assert teleop.names().count("reset_to_pose") == 1


def test_spacemouse_dryrun_reset_uses_its_stored_start_pose():
    """SpaceMouse differs from the Pico4 rigs here, and the difference is load-bearing.

    SpaceMouse integrates stick deflection from a start pose it holds itself, so a
    dry-run reset snaps back to that stored pose and never asks the robot. The
    Pico4 rigs read the arm's current TCP instead.
    """
    case = CASES[2]
    robot, teleop = case.robot(), case.teleop(reset_on={1})

    case.run(robot, teleop, dryrun=True)

    assert teleop.names().count("reset_to_pose") == 1
    assert not [n for n in robot.names() if n.startswith("get_current_tcp_pose")]


# --------------------------------------------------------------------------- #
# SpaceMouse release-drift fix
# --------------------------------------------------------------------------- #


class IdleSpaceMouseTeleop(FakeSpaceMouseTeleop):
    """A SpaceMouse whose stick is released from the start (``_enabled`` False)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._enabled = False


def test_a_released_spacemouse_stick_snaps_the_target_back_to_the_arm():
    """The release-drift fix, which a refactor can silently drop.

    The stick integrates deflection into an absolute pose accumulator. Push it
    faster than the controller follows and the accumulator runs ahead of the arm;
    let go, and the arm keeps drifting toward a target the operator abandoned.
    After a short idle the loop must snap the accumulator to the arm's real pose.

    Nothing else in this suite covers it — it was noticed missing only because
    the behaviour was written down in a docstring.
    """
    case = CASES[2]  # elite_cs66_rt_spacemouse_teleop_loop
    # fps=1000 with the 0.5s default means the snap lands well past 4 frames, so
    # drive the idle threshold down instead of running thousands of iterations.
    robot = case.robot()
    teleop = IdleSpaceMouseTeleop()

    from lerobot.scripts.teleop_device_loops import (
        SpaceMousePolicy,
        run_cartesian_teleop_loop,
    )

    run_loop(
        run_cartesian_teleop_loop,
        iterations=ITERS,
        teleop=teleop,
        robot=robot,
        fps=1000,
        policy=SpaceMousePolicy(release_resync_idle_s=0.002),  # 2 frames at 1kHz
        display_data=False,
    )

    assert teleop.names().count("reset_to_pose") >= 1, "released stick never re-synced"
    assert "get_current_tcp_pose_euler" in robot.names()


def test_an_active_spacemouse_stick_is_never_snapped():
    """Snapping mid-push would fight the operator."""
    case = CASES[2]
    robot, teleop = case.robot(), case.teleop()  # _enabled True by default

    from lerobot.scripts.teleop_device_loops import (
        SpaceMousePolicy,
        run_cartesian_teleop_loop,
    )

    run_loop(
        run_cartesian_teleop_loop,
        iterations=ITERS,
        teleop=teleop,
        robot=robot,
        fps=1000,
        policy=SpaceMousePolicy(release_resync_idle_s=0.002),
        display_data=False,
    )

    assert "reset_to_pose" not in teleop.names()
