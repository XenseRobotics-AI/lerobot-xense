"""Tool bridge and Elite action/lifecycle coverage without physical hardware."""

import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from lerobot.grippers import DHAG95GripperConfig
from lerobot.robots.bi_elite_cs66_rt.bi_elite_cs66_rt import BiEliteCS66RT
from lerobot.robots.bi_elite_cs66_rt.config_bi_elite_cs66_rt import BiEliteCS66RTConfig, BiEliteCS66RTControlMode
from lerobot.robots.elite_cs66_rt import tool_rs485 as tool
from lerobot.robots.elite_cs66_rt.config_elite_cs66_rt import EliteCS66RTConfig, EliteCS66RTControlMode
from lerobot.robots.elite_cs66_rt.config_tool_rs485 import EliteToolRS485Config
from lerobot.robots.elite_cs66_rt.elite_cs66_rt import EliteCS66RT


@pytest.fixture
def bridge(monkeypatch):
    monkeypatch.setenv("ELITE_SSH_PASSWORD", "test-only-secret")
    monkeypatch.setattr(tool.shutil, "which", lambda name: "/bin/" + name)
    monkeypatch.setattr(tool.time, "sleep", lambda _: None)
    driver = Mock()
    driver.configureToolRs485.return_value = True
    driver.setToolVoltage.return_value = True
    bridge = tool.EliteToolRS485(driver, "192.0.2.1", EliteToolRS485Config(), 10.0)
    bridge._ssh = Mock(side_effect=lambda command: "ELITE_SOCAT_AVAILABLE" if "command -v socat" in command else "1234")
    connection = Mock()
    monkeypatch.setattr(tool.socket, "create_connection", Mock(side_effect=[ConnectionRefusedError(), connection]))
    return bridge, driver, connection


def test_bridge_uses_approved_settings_and_closes_its_session(bridge):
    b, driver, connection = bridge
    b.open(1, 115200)
    driver.configureToolRs485.assert_called_once_with(115200, 0, 1)
    driver.setToolVoltage.assert_called_once_with(24)
    assert b._client._timeout == 1.0
    b.close()
    connection.shutdown.assert_called_once()
    connection.close.assert_called_once()
    command = b._ssh.call_args.args[0]
    assert b._pid_file in command and "/proc/$pid/cmdline" in command
    assert "killall" not in command
    previous_calls = b._ssh.call_count
    b.close()
    assert b._ssh.call_count == previous_calls


def test_accepted_tcp_connection_is_not_assumed_to_be_an_owned_bridge(bridge, monkeypatch):
    b, driver, connection = bridge
    monkeypatch.setattr(tool.socket, "create_connection", Mock(return_value=connection))
    with pytest.raises(RuntimeError, match="listener owner is unverified"):
        b.open(1, 115200)
    driver.configureToolRs485.assert_not_called()
    driver.setToolVoltage.assert_not_called()
    b._ssh.assert_not_called()


def test_lost_launch_reply_still_attempts_session_cleanup(bridge):
    b, _, _ = bridge
    b._ssh.side_effect = ["ELITE_SOCAT_AVAILABLE", TimeoutError("launch reply lost"), ""]
    with pytest.raises(TimeoutError, match="launch reply lost"):
        b.open(1, 115200)
    assert b._ssh.call_count == 3
    assert b._pid_file in b._ssh.call_args.args[0]


def test_missing_controller_socat_fails_before_tool_configuration_or_power(bridge):
    b, driver, _ = bridge
    b._ssh.side_effect = None
    b._ssh.return_value = "ELITE_SOCAT_MISSING"
    with pytest.raises(RuntimeError, match="controller.*has no socat"):
        b.open(1, 115200)
    driver.configureToolRs485.assert_not_called()
    driver.setToolVoltage.assert_not_called()
    assert b._ssh.call_count == 1 and not b._launch_attempted
    assert b._password is None


def test_controller_ssh_failure_does_not_change_tool_configuration_or_power(bridge):
    b, driver, _ = bridge
    b._ssh.side_effect = ConnectionError("SSH authentication failed")
    with pytest.raises(ConnectionError, match="authentication failed"):
        b.open(1, 115200)
    driver.configureToolRs485.assert_not_called()
    driver.setToolVoltage.assert_not_called()
    assert not b._launch_attempted and b._password is None


def test_ssh_is_bounded_and_password_is_not_in_arguments(monkeypatch):
    b = tool.EliteToolRS485(Mock(), "192.0.2.1", EliteToolRS485Config(), 10)
    b._password = "test-only-secret"
    run = Mock(return_value=SimpleNamespace(stdout="ok"))
    monkeypatch.setattr(tool.subprocess, "run", run)
    assert b._ssh("true") == "ok"
    args, kwargs = run.call_args
    assert "test-only-secret" not in repr(args)
    assert kwargs["env"]["SSHPASS"] == "test-only-secret"
    assert kwargs["timeout"] == 10
    run.side_effect = subprocess.TimeoutExpired("ssh", 10)
    with pytest.raises(TimeoutError, match="timed out"):
        b._ssh("true")


def test_failed_gripper_connect_closes_bridge(monkeypatch):
    session = Mock()
    monkeypatch.setattr(tool, "EliteToolRS485", Mock(return_value=session))
    gripper = Mock()
    gripper.config = DHAG95GripperConfig()
    gripper.connect.side_effect = RuntimeError("init failed")
    gripper.disconnect.side_effect = session.close
    with pytest.raises(RuntimeError, match="init failed"):
        tool.connect_elite_gripper(gripper, Mock(), "192.0.2.1", EliteToolRS485Config(), 10)
    session.close.assert_called_once()


def test_partial_bridge_start_keeps_owner_when_cleanup_temporarily_fails(monkeypatch):
    from lerobot.grippers import make_gripper_from_config

    session = Mock()
    session.open.side_effect = TimeoutError("bridge startup failed")
    session.close.side_effect = [ConnectionError("SSH close failed"), None]
    monkeypatch.setattr(tool, "EliteToolRS485", Mock(return_value=session))
    gripper = make_gripper_from_config(DHAG95GripperConfig())
    with pytest.raises(TimeoutError, match="startup failed"):
        tool.connect_elite_gripper(gripper, Mock(), "192.0.2.1", EliteToolRS485Config(), 10)
    assert gripper._transport is session
    gripper.disconnect()
    assert session.close.call_count == 2 and gripper._transport is None


@pytest.fixture(params=["single", "bi"])
def robot(request):
    if request.param == "single":
        r = EliteCS66RT(EliteCS66RTConfig(control_mode=EliteCS66RTControlMode.JOINT_SERVO))
        r._rtsi = Mock()
        r._rtsi.getActualJointPositions.return_value = [0.0] * 6
        sides = [(None, "")]
    else:
        r = BiEliteCS66RT(
            BiEliteCS66RTConfig(
                control_mode=BiEliteCS66RTControlMode.JOINT_SERVO,
                left_world_rotation=np.eye(3).tolist(),
                right_world_rotation=np.eye(3).tolist(),
            )
        )
        sides = [("left", "left_"), ("right", "right_")]
        for side, _ in sides:
            r._rtsi[side] = Mock()
            r._rtsi[side].getActualJointPositions.return_value = [0.0] * 6
    for side, _ in sides:
        gripper = Mock()
        gripper.config = DHAG95GripperConfig(side=side)
        driver = Mock()
        driver.status.return_value = SimpleNamespace(running=False, error="")
        if side is None:
            r._gripper, r._driver = gripper, driver
        else:
            r._gripper[side], r._driver[side] = gripper, driver
    r._is_connected = True
    yield r, sides
    # No robot destructor should use test doubles to attempt return-to-home.
    r._is_connected = False
    if isinstance(r, BiEliteCS66RT):
        r._driver = dict.fromkeys(("left", "right"))
        r._rtsi = dict.fromkeys(("left", "right"))
    else:
        r._driver = r._rtsi = None


@pytest.mark.parametrize("during_reset", [False, True])
def test_binary_action_is_recorded_as_the_sent_endpoint(robot, during_reset):
    r, sides = robot
    action = {}
    for _, prefix in sides:
        action.update({f"{prefix}joint_{i}.pos": 0.0 for i in range(1, 7)})
        action[f"{prefix}gripper.pos"] = 0.25
    sent = r._action_during_reset(action) if during_reset else r._send_action(action)
    for side, prefix in sides:
        g = r._gripper if side is None else r._gripper[side]
        g.set_gripper_position.assert_called_once_with(0.0)
        assert sent[f"{prefix}gripper.pos"] == 0.0


def test_failed_connect_cleanup_releases_gripper_before_driver(robot):
    r, sides = robot
    events = []
    for side, prefix in sides:
        g = r._gripper if side is None else r._gripper[side]
        d = r._driver if side is None else r._driver[side]

        def close(g=g, prefix=prefix):
            events.append(prefix + "gripper")
            g._is_connected = g.is_connected = False

        g.disconnect.side_effect = close
        d.stopControl.side_effect = lambda _, prefix=prefix: events.append(prefix + "driver")
    r._cleanup_after_failed_connect()
    for _, prefix in sides:
        assert events.index(prefix + "gripper") < events.index(prefix + "driver")


def test_normal_disconnect_closes_tools_even_if_camera_cleanup_fails(robot, monkeypatch):
    r, sides = robot
    events = []
    camera = Mock()
    camera.disconnect.side_effect = OSError("camera unplugged")
    r.cameras = {"head": camera}
    monkeypatch.setattr(r, "_stop_servo_loop", Mock())
    if isinstance(r, BiEliteCS66RT):
        monkeypatch.setattr(r, "_run_arms_paired", Mock())
    else:
        monkeypatch.setattr(r, "_move_j_blocking", Mock())
    for side, prefix in sides:
        g = r._gripper if side is None else r._gripper[side]
        d = r._driver if side is None else r._driver[side]

        def close(g=g, prefix=prefix):
            events.append(prefix + "gripper")
            g._is_connected = g.is_connected = False

        g.disconnect.side_effect = close
        d.stopControl.side_effect = lambda _, prefix=prefix: events.append(prefix + "driver")
    r.disconnect()
    for _, prefix in sides:
        assert events.index(prefix + "gripper") < events.index(prefix + "driver")


def test_robot_disconnect_retries_tool_cleanup_after_arm_handles_are_gone(robot):
    r, sides = robot
    r._is_connected = False
    if isinstance(r, BiEliteCS66RT):
        r._driver = dict.fromkeys(("left", "right"))
        r._rtsi = dict.fromkeys(("left", "right"))
    else:
        r._driver = r._rtsi = None
    r.disconnect()
    for side, _ in sides:
        g = r._gripper if side is None else r._gripper[side]
        g.disconnect.assert_called_once()
