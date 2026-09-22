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
    """Mirror the pybind ForcePositionConfig, which declares every field up front.

    `_make_sdk_controller` skips any field the installed native extension does
    not expose, so a fake without these attributes would silently drop the whole
    configuration instead of forwarding it.
    """

    close_speed_radps = None
    status_timeout_ms = None
    motor_stream_hz = None


class _FakeControlLoop:
    def __init__(self, gripper, **kwargs):
        self.gripper = gripper
        self.kwargs = kwargs
        self.started = False

    # start/stop 记到**夹爪的同一条时间线**上,这样 connect() 里「使能与启动
    # 谁先谁后」是真的被断言了,而不是分别检查两个计数器。
    def _rec(self, name):
        motor = getattr(getattr(self, "gripper", None), "motor", None)
        if motor is not None and hasattr(motor, "calls"):
            motor.calls.append(name)

    def start(self):
        self.started = True
        self._rec("loop.start")

    def stop(self):
        self.started = False
        self._rec("loop.stop")


class _FakeForcePositionController:
    def __init__(self, gripper, config):
        self.gripper = gripper
        self.config = config

    def start(self):
        self.started = True
        self._rec("loop.start")

    def stop(self):
        self.started = False
        self._rec("loop.stop")

    def _rec(self, name):
        motor = getattr(getattr(self, "gripper", None), "motor", None)
        if motor is not None and hasattr(motor, "calls"):
            motor.calls.append(name)


def _install_fake_sdk(monkeypatch, force_position_config=_FakeForcePositionConfig):
    fake = SimpleNamespace(
        SubmitPhase=SimpleNamespace(STREAM_LOCKED="stream_locked_enum", FREE_RUNNING="free_running_enum"),
        StallAction=SimpleNamespace(HOLD_POSITION="hold_position_enum", NONE="none_enum"),
        ControlLoop=_FakeControlLoop,
        ForcePositionConfig=force_position_config,
        ForcePositionController=_FakeForcePositionController,
        # SDK 的主入口。没有它这个 fake 就不完整 —— connect() 第一件事就是
        # taccap.FollowerGripper(device),而在补 connect 测试之前没人走到这里。
        # 惰性引用:_FakeGripper 定义在本文件更下面。
        FollowerGripper=lambda *a, **k: _FakeGripper(),
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
    # The whole surface the SDK still exposes. SDK 0.2.0 cut ForcePositionConfig
    # from sixteen fields to six: contact detection and the position gains moved
    # to detail::ForcePositionTuning, out of a caller's reach, because the MCU
    # already runs the same stall test at 500 Hz.
    # 上层只转发这三个。力矩预算及其两个天花板不再由这边配 —— 走 SDK 默认,
    # 理由见 configuration_taccap.py 里字段旁边那段。
    values = {
        "close_speed_radps": 0.45,
        "status_timeout_ms": 400,
        "motor_stream_hz": 90,
    }
    config = TaccapFollowerConfig(controller="force_position", **values)

    controller = _follower(config)._make_sdk_controller()

    assert isinstance(controller, _FakeForcePositionController)
    assert {name: getattr(controller.config, name) for name in values} == values


def test_force_position_skips_fields_the_installed_sdk_lacks(monkeypatch):
    """An older native extension missing a field must not abort controller setup."""

    class _OldForcePositionConfig:
        close_speed_radps = None

    _install_fake_sdk(monkeypatch, force_position_config=_OldForcePositionConfig)
    config = TaccapFollowerConfig(controller="force_position", close_speed_radps=0.45)

    controller = _follower(config)._make_sdk_controller()

    assert controller.config.close_speed_radps == 0.45
    # The fake declares one of the three we forward, and setup still completes
    # rather than raising AttributeError and taking both grippers down on connect.
    assert not hasattr(controller.config, "status_timeout_ms")


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

    follower = _follower(TaccapFollowerConfig(controller="control_loop", print_status=True, status_print_hz=5.0))
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
    expected = "L pos=0.250 raw=-0.3000rad vel=-1.20rad/s tq=-0.70Nm temp=41C age=3.0ms hz=99.8"
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
    expected = "L pos=0.150 raw=-0.1800rad vel=-0.02rad/s tq=-1.00Nm temp=43C age=4.0ms state=holding_force cmd=+1.10Nm"
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


def test_every_config_attribute_referenced_in_the_follower_exists():
    """`self._config.<name>` must name a real TaccapFollowerConfig field.

    This exists because removing config fields broke exactly this and no test
    noticed: the connect-time log still read `self._config.grasp_torque_nm`
    after that field was deleted, which is an AttributeError on every
    connect — and TacCap's connect() has no test coverage at all (it needs
    hardware), so the suite stayed green. Found by grep, not by tests.

    A dataclass attribute access cannot be caught by ruff, so scan for it.
    """
    import dataclasses
    import re
    from pathlib import Path

    source = (Path(__file__).resolve().parents[2] / "src/lerobot/grippers/taccap/taccap_follower.py").read_text()
    referenced = set(re.findall(r"self\._config\.([a-z_][a-z0-9_]*)", source))
    declared = {f.name for f in dataclasses.fields(TaccapFollowerConfig)}
    missing = sorted(referenced - declared)
    assert not missing, (
        f"taccap_follower.py reads {missing} off the config, but TaccapFollowerConfig does not declare them."
    )


# ---- connect() ------------------------------------------------------------
#
# 这条路径此前**一条测试都没有** —— 它要真硬件,所以从来没人测。代价已经付过:
# 删配置字段之后 connect() 的 info 日志还在读 self._config.grasp_torque_nm,
# 那是每次连接必抛的 AttributeError,而全量 830 项照样全绿。
#
# 下面几条把 connect() 里**真正有判断的**部分钉住:使能与 start 的先后、失败时
# 的回滚、未标定时的拒绝。纯日志和纯转发不测,那是噪音。


class _FakeMotor:
    def __init__(self, fail_on=None):
        self.calls = []
        self._fail_on = fail_on

    def _rec(self, name):
        self.calls.append(name)
        if self._fail_on == name:
            raise RuntimeError(f"injected failure in {name}")

    def clear_fault(self):
        self._rec("clear_fault")

    def enable(self):
        self._rec("enable")

    def disable(self):
        self._rec("disable")


class _FakeGripper:
    def __init__(self, calibrated=True, motor=None):
        self.motor = motor or _FakeMotor()
        self._calibrated = calibrated

    def get_gripper_config(self):
        return SimpleNamespace(flags=0x0001 if self._calibrated else 0x0000)

    def position_map(self):
        return SimpleNamespace(reverse=False)


def _connectable(monkeypatch, config, gripper=None, start_fails=False):
    """A TaccapFollower whose every外部依赖 is faked, ready for connect()."""
    gripper = gripper or _FakeGripper()

    class _Loop(_FakeForcePositionController):
        def start(self):
            if start_fails:
                raise RuntimeError("injected failure in start")
            super().start()

        def snapshot(self):
            return SimpleNamespace(
                observation=SimpleNamespace(position=0.5),
                grasp_torque_nm=1.1,
                hold_torque_limit_nm=1.8,
            )

    _install_fake_sdk(monkeypatch)
    monkeypatch.setattr(driver.taccap, "FollowerGripper", lambda *a, **k: gripper)
    monkeypatch.setattr(driver.taccap, "ForcePositionController", lambda g, cfg: _Loop(g, cfg))
    follower = driver.TaccapFollower(config)
    monkeypatch.setattr(follower, "_resolve_device", lambda: "/dev/fake")
    # init-open drives real motion in the base class; connect() already treats a
    # failure here as non-fatal, so neutralise it rather than fake a whole move.
    monkeypatch.setattr(follower, "initialize_gripper_position", lambda *a, **k: None)
    return follower, gripper


def test_connect_force_position_starts_the_loop_before_enabling_the_motor():
    """Ordering is load-bearing, and it differs per controller.

    ForcePositionController.start() validates the motor's persisted torque limit
    (0x700B) before any motion, so the SDK wants start-before-enable. ControlLoop
    keeps the older enable-before-start. Getting this backwards does not fail
    loudly — it just skips the check — which is exactly why it needs a test.
    """
    import pytest

    monkeypatch = pytest.MonkeyPatch()
    try:
        follower, gripper = _connectable(monkeypatch, TaccapFollowerConfig(controller="force_position"))
        follower.connect()
        assert gripper.motor.calls == ["clear_fault", "loop.start", "enable"]
        assert follower._loop.started
    finally:
        monkeypatch.undo()


def test_connect_control_loop_enables_the_motor_before_starting_the_loop():
    import pytest

    monkeypatch = pytest.MonkeyPatch()
    try:
        follower, gripper = _connectable(monkeypatch, TaccapFollowerConfig(controller="control_loop"))
        monkeypatch.setattr(driver.taccap, "ControlLoop", lambda *a, **k: _FakeControlLoop(*a, **k))
        follower.connect()
        assert gripper.motor.calls == ["clear_fault", "enable", "loop.start"]
    finally:
        monkeypatch.undo()


def test_connect_reads_the_torque_budget_off_the_snapshot_not_the_config():
    """The regression that motivated all of this.

    connect() logs the grasp torque actually in force. It used to read it from
    the config; once that field was removed the log raised AttributeError on
    every single connect, and no test noticed because connect() had none.
    """
    import pytest

    monkeypatch = pytest.MonkeyPatch()
    try:
        follower, _ = _connectable(monkeypatch, TaccapFollowerConfig(controller="force_position"))
        follower.connect()  # would raise AttributeError before the fix
        assert follower.is_connected
    finally:
        monkeypatch.undo()


def test_connect_releases_the_handle_when_the_loop_fails_to_start():
    """A half-open device must not be stranded.

    _is_connected stays False on this path, so disconnect() would refuse to run
    and the handle would live until GC — with the motor still enabled.
    """
    import pytest

    monkeypatch = pytest.MonkeyPatch()
    try:
        follower, gripper = _connectable(
            monkeypatch, TaccapFollowerConfig(controller="force_position"), start_fails=True
        )
        with pytest.raises(RuntimeError, match="injected failure in start"):
            follower.connect()
        assert "disable" in gripper.motor.calls
        assert follower._gripper is None
        assert not follower.is_connected
    finally:
        monkeypatch.undo()


def test_connect_refuses_an_uncalibrated_gripper_and_releases_it():
    """Normalized [0,1] control is meaningless without the travel span."""
    import pytest

    monkeypatch = pytest.MonkeyPatch()
    try:
        gripper = _FakeGripper(calibrated=False)
        follower, _ = _connectable(
            monkeypatch, TaccapFollowerConfig(controller="force_position", require_calibrated=True), gripper=gripper
        )
        with pytest.raises(RuntimeError, match="not calibrated"):
            follower.connect()
        assert follower._gripper is None
        assert not follower.is_connected
    finally:
        monkeypatch.undo()
