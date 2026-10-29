#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Characterization tests for the device-specific record loops.

Same purpose as the teleop-loop suite: these two loops are 0.73 similar, have no
coverage, and only ever run against real arms. What they must not lose in a
refactor is the dataset itself — a recording that silently drops or duplicates
frames is worse than one that fails, because it is discovered at training time.
"""

import pytest

from lerobot.scripts import lerobot_record as record_mod
from tests.scripts.conftest_teleop_doubles import (
    FakeDataset,
    FakeRobot,
    FakeSpaceMouseTeleop,
    FakeTeleop,
    StopLoopError,
    fresh_events,
)

ITERS = 4
FPS = 1000


class Case:
    def __init__(self, loop, teleop_cls, reset_kwarg="reset_frames"):
        self.loop, self.teleop_cls, self.reset_kwarg = loop, teleop_cls, reset_kwarg

    def teleop(self, *, reset_on=()):
        return self.teleop_cls(**({self.reset_kwarg: set(reset_on)} if reset_on else {}))

    def run(self, robot, teleop, dataset, events, **kwargs):
        try:
            getattr(record_mod, self.loop)(
                robot=robot,
                events=events,
                fps=FPS,
                dataset=dataset,
                teleop=teleop,
                single_task="task",
                control_time_s=1000,
                **kwargs,
            )
        except StopLoopError:
            return
        raise AssertionError(f"{self.loop} ended before the double stopped it")


CASES = [
    Case("flexiv_rizon4_rt_record_loop", FakeTeleop),
    Case("elite_cs66_rt_record_loop", FakeSpaceMouseTeleop, reset_kwarg="both_pressed_frames"),
]
IDS = [c.loop for c in CASES]


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_every_frame_reaches_the_dataset(case):
    robot, dataset, events = FakeRobot(iterations=ITERS), FakeDataset(fps=FPS), fresh_events()

    case.run(robot, case.teleop(), dataset, events)

    assert len(dataset.frames) == ITERS
    assert robot.names().count("get_observation") == ITERS
    assert robot.names().count("send_action") == ITERS


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_recording_continues_while_the_arm_runs_its_own_trajectory(case):
    """The crux of these loops, and the reason they exist separately.

    While rt_moving the background trajectory owns the arm, so no action is sent —
    but the episode must stay continuous, so frames keep being written (the arm's
    current pose standing in as the action). A refactor that skipped the frame
    along with the send would punch holes in every recording taken across a reset.
    """
    robot = FakeRobot(iterations=ITERS, rt_moving_frames={1, 2})
    dataset, events = FakeDataset(fps=FPS), fresh_events()

    case.run(robot, case.teleop(), dataset, events)

    assert robot.names().count("send_action") == ITERS - 2
    assert len(dataset.frames) == ITERS, "frames must not be dropped while moving"


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_the_keyboard_go_start_event_sends_the_arm_home(case):
    robot, dataset = FakeRobot(iterations=ITERS), FakeDataset(fps=FPS)
    events = fresh_events()
    events["go_start"] = True

    case.run(robot, case.teleop(), dataset, events)

    assert robot.names().count("reset_to_initial_position") >= 1


def test_elite_record_resyncs_once_per_finished_reset_count():
    """The re-sync follows reset_generation, not a count of rt_moving reads."""
    case = CASES[1]
    robot = FakeRobot(iterations=ITERS, finished_reset_frames={1})
    dataset, events, teleop = FakeDataset(fps=FPS), fresh_events(), case.teleop()

    case.run(robot, teleop, dataset, events)

    assert teleop.names().count("reset_to_pose") == 1
    names = robot.names()
    resync_at = names.index("get_current_tcp_pose_euler")
    sends = [i for i, name in enumerate(names) if name == "send_action"]
    assert sends[0] < resync_at < sends[1], "the re-sync must precede frame 1's send"
    assert len(dataset.frames) == ITERS


def test_elite_record_also_accepts_the_spacemouse_button_reset():
    """The Elite loop adds device buttons on top of the keyboard event."""
    case = CASES[1]
    robot, dataset, events = FakeRobot(iterations=ITERS), FakeDataset(fps=FPS), fresh_events()

    case.run(robot, case.teleop(reset_on={1}), dataset, events)

    assert robot.names().count("reset_to_initial_position") == 1


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_exit_early_stops_the_episode(case):
    robot, dataset = FakeRobot(iterations=ITERS), FakeDataset(fps=FPS)
    events = fresh_events()
    events["exit_early"] = True

    # Ends on its own via the event rather than by exhausting the double.
    getattr(record_mod, case.loop)(
        robot=robot,
        events=events,
        fps=FPS,
        dataset=dataset,
        teleop=case.teleop(),
        single_task="task",
        control_time_s=1000,
    )

    assert len(dataset.frames) <= 1


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_no_dataset_means_observe_only_and_write_nothing(case):
    """The reset phase runs the same loop with dataset=None; it must not record."""
    robot, events = FakeRobot(iterations=ITERS), fresh_events()

    case.run(robot, case.teleop(), None, events)

    assert robot.names().count("get_observation") == ITERS
