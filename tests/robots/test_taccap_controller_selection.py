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

"""No-hardware coverage for TacCap SDK controller selection and YAML mapping."""

from types import SimpleNamespace

import pytest

from lerobot.grippers import TaccapFollowerConfig
from lerobot.grippers.taccap import taccap_follower as driver


class _FakeForcePositionConfig:
    pass


class _FakeControlLoop:
    def __init__(self, gripper, **kwargs):
        self.gripper = gripper
        self.kwargs = kwargs


class _FakeForcePositionController:
    def __init__(self, gripper, config):
        self.gripper = gripper
        self.config = config


def _install_fake_sdk(monkeypatch):
    fake = SimpleNamespace(
        SubmitPhase=SimpleNamespace(STREAM_LOCKED="stream_locked_enum", FREE_RUNNING="free_running_enum"),
        StallAction=SimpleNamespace(HOLD_POSITION="hold_position_enum", NONE="none_enum"),
        ControlLoop=_FakeControlLoop,
        ForcePositionConfig=_FakeForcePositionConfig,
        ForcePositionController=_FakeForcePositionController,
    )
    monkeypatch.setattr(driver, "taccap", fake)


def _follower(config):
    follower = driver.TaccapFollower(config)
    follower._gripper = SimpleNamespace(position_map=lambda: SimpleNamespace(reverse=False))
    return follower


def test_control_loop_receives_every_exposed_sdk_parameter(monkeypatch):
    _install_fake_sdk(monkeypatch)
    config = TaccapFollowerConfig(
        controller="control_loop",
        control_hz=87,
        kp=9.0,
        kd=0.8,
        feedforward_torque=-0.2,
        motor_stream_hz=73,
        submit_phase="free_running",
        max_position_torque_nm=1.1,
        rated_torque_nm=1.2,
        rated_hold_ms=31,
        rated_release_rad=0.07,
        stall_torque_nm=0.9,
        stall_vel_radps=0.12,
        stall_hold_ms=44,
        stall_action="none",
    )

    controller = _follower(config)._make_sdk_controller()

    assert controller.kwargs == {
        "hz": 87,
        "kp": 9.0,
        "kd": 0.8,
        "feedforward_torque": -0.2,
        "motor_stream_hz": 73,
        "phase": "free_running_enum",
        "max_position_torque_nm": 1.1,
        "rated_torque_nm": 1.2,
        "rated_hold_ms": 31,
        "rated_release_rad": 0.07,
        "stall_torque_nm": 0.9,
        "stall_vel_radps": 0.12,
        "stall_hold_ms": 44,
        "stall_action": "none_enum",
    }


def test_force_position_receives_every_exposed_sdk_parameter(monkeypatch):
    _install_fake_sdk(monkeypatch)
    values = {
        "close_position": 0.02,
        "close_speed_radps": 0.45,
        "grasp_torque_nm": 1.1,
        "hold_torque_limit_nm": 1.7,
        "motion_torque_limit_nm": 5.5,
        "contact_torque_nm": 0.09,
        "contact_vel_radps": 0.03,
        "contact_vel_ratio": 0.2,
        "contact_moved_rad": 0.02,
        "position_kp": 18.0,
        "position_kd": 0.9,
        "brake_distance_rad": 0.08,
        "contact_samples": 4,
        "startup_guard_ms": 275,
        "status_timeout_ms": 400,
        "motor_stream_hz": 90,
    }
    config = TaccapFollowerConfig(controller="force_position", **values)

    controller = _follower(config)._make_sdk_controller()

    assert isinstance(controller, _FakeForcePositionController)
    assert {name: getattr(controller.config, name) for name in values} == values


def test_control_loop_flips_normalized_feedforward_for_reversed_map(monkeypatch):
    _install_fake_sdk(monkeypatch)
    follower = _follower(TaccapFollowerConfig(controller="control_loop", feedforward_torque=-0.2))
    follower._gripper = SimpleNamespace(position_map=lambda: SimpleNamespace(reverse=True))

    controller = follower._make_sdk_controller()

    assert controller.kwargs["feedforward_torque"] == pytest.approx(0.2)


def test_each_controller_exposes_a_common_position_observation(monkeypatch):
    _install_fake_sdk(monkeypatch)
    observation = SimpleNamespace(position=0.42)

    control_loop = _follower(TaccapFollowerConfig(controller="control_loop"))
    control_loop._loop = SimpleNamespace(observation=lambda: observation)
    assert control_loop._latest_observation() is observation

    force_position = _follower(TaccapFollowerConfig(controller="force_position"))
    force_position._loop = SimpleNamespace(snapshot=lambda: SimpleNamespace(observation=observation))
    assert force_position._latest_observation() is observation


def test_force_position_coalesces_repeated_teleop_targets(monkeypatch):
    _install_fake_sdk(monkeypatch)
    calls = []
    follower = _follower(TaccapFollowerConfig(controller="force_position"))
    follower._is_connected = True
    follower._loop = SimpleNamespace(set_target=calls.append)

    follower.set_gripper_position(0.25)
    follower.set_gripper_position(0.25)
    follower.set_gripper_position(0.25005)
    follower.set_gripper_position(0.251)

    assert calls == [0.25, 0.251]
    follower._is_connected = False


def test_control_loop_keeps_accepting_repeated_targets(monkeypatch):
    _install_fake_sdk(monkeypatch)
    calls = []
    follower = _follower(TaccapFollowerConfig(controller="control_loop"))
    follower._is_connected = True
    follower._loop = SimpleNamespace(set_target=calls.append)

    follower.set_gripper_position(0.25)
    follower.set_gripper_position(0.25)

    assert calls == [0.25, 0.25]
    follower._is_connected = False


def test_control_loop_status_print_is_rate_limited_and_uses_cached_observation(monkeypatch):
    _install_fake_sdk(monkeypatch)
    observation = SimpleNamespace(
        position=0.25,
        raw_pos=-0.3,
        velocity=-1.2,
        torque=-0.7,
        motor_temp_c=41.0,
        age_ms=3.0,
    )
    reads = []

    def read_observation():
        reads.append(True)
        return observation

    follower = _follower(
        TaccapFollowerConfig(controller="control_loop", print_status=True, status_print_hz=5.0)
    )
    follower._is_connected = True
    follower._loop = SimpleNamespace(
        observation=read_observation,
        submit_hz=99.8,
    )
    updates = []
    monkeypatch.setattr(driver, "_set_taccap_status_line", lambda side, line: updates.append((side, line)))
    times = iter((10.0, 10.1, 10.21))
    monkeypatch.setattr(driver.time, "monotonic", lambda: next(times))

    follower.get_gripper_position()
    follower.get_gripper_position()
    follower.get_gripper_position()

    assert len(reads) == 3
    expected = (
        "L pos=0.250 raw=-0.3000rad vel=-1.20rad/s "
        "tq=-0.70Nm temp=41C age=3.0ms hz=99.8"
    )
    assert updates == [("left", expected), ("left", expected)]
    follower._is_connected = False


def test_force_position_status_print_reuses_one_snapshot(monkeypatch):
    _install_fake_sdk(monkeypatch)
    observation = SimpleNamespace(
        position=0.15,
        raw_pos=-0.18,
        velocity=-0.02,
        torque=-1.0,
        motor_temp_c=43.0,
        age_ms=4.0,
    )
    snapshot = SimpleNamespace(
        observation=observation,
        state="holding_force",
        commanded_torque_nm=1.1,
    )
    reads = []

    def read_snapshot():
        reads.append(True)
        return snapshot

    follower = _follower(TaccapFollowerConfig(controller="force_position", print_status=True))
    follower._is_connected = True
    follower._loop = SimpleNamespace(snapshot=read_snapshot)
    updates = []
    monkeypatch.setattr(driver, "_set_taccap_status_line", lambda side, line: updates.append((side, line)))
    monkeypatch.setattr(driver.time, "monotonic", lambda: 20.0)

    assert follower.get_gripper_position() == 0.15

    assert len(reads) == 1
    expected = (
        "L pos=0.150 raw=-0.1800rad vel=-0.02rad/s "
        "tq=-1.00Nm temp=43C age=4.0ms state=holding_force cmd=+1.10Nm"
    )
    assert updates == [("left", expected)]
    follower._is_connected = False


def test_disabled_status_print_does_not_touch_controller_diagnostics(monkeypatch):
    _install_fake_sdk(monkeypatch)
    observation = SimpleNamespace(position=0.4)
    follower = _follower(TaccapFollowerConfig(controller="control_loop", print_status=False))
    follower._is_connected = True
    follower._loop = SimpleNamespace(observation=lambda: observation)

    assert follower.get_gripper_position() == 0.4
    follower._is_connected = False
