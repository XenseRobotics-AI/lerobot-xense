"""Adapter behaviour around the native libpyelite driver, with the driver mocked.

No libpyelite needed: the adapters only import it in connect(), so these run on
every host and cover the review fixes for PR #30.
"""

import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from lerobot.robots.bi_elite_cs66_rt.bi_elite_cs66_rt import BiEliteCS66RT
from lerobot.robots.bi_elite_cs66_rt.config_bi_elite_cs66_rt import BiEliteCS66RTConfig
from lerobot.robots.elite_cs66_rt.config_elite_cs66_rt import EliteCS66RTConfig
from lerobot.robots.elite_cs66_rt.elite_cs66_rt import (
    EliteCS66RT,
    _configure_native_driver,
    _derive_controller_tool,
    _pose6_to_matrix,
)
from lerobot.utils.rotation import Rotation

MEASURED_RPY = [0.4, -0.1, 0.3, 0.0, 0.0, 0.0]  # RTSI [x, y, z, roll, pitch, yaw]


def _status(running=True, error="", reset_active=False):
    return SimpleNamespace(running=running, error=error, reset_active=reset_active)


def _driver(running=True, error=""):
    driver = Mock()
    driver.status.return_value = _status(running, error)
    driver.commanded_pose.return_value = np.eye(4)
    return driver


def _rtsi():
    rtsi = Mock()
    rtsi.getActualTCPPose.return_value = list(MEASURED_RPY)
    rtsi.getActualJointPositions.return_value = [0.0] * 6
    return rtsi


# Robots built with mocked drivers but marked connected. Robot.__del__ would
# otherwise disconnect them at GC time, driving the mocks from the bimanual
# return-home thread pool and deadlocking on Mock's internal lock.
_ROBOTS: list = []


@pytest.fixture(autouse=True)
def _disconnect_mocked_robots():
    yield
    while _ROBOTS:
        _ROBOTS.pop()._is_connected = False


def _single():
    robot = EliteCS66RT(EliteCS66RTConfig(gripper=None, cameras={}))
    robot._driver, robot._rtsi = _driver(), _rtsi()
    robot._is_connected = True
    _ROBOTS.append(robot)
    return robot


def _bi():
    robot = BiEliteCS66RT(
        BiEliteCS66RTConfig(
            left_use_gripper=False,
            right_use_gripper=False,
            left_world_rotation=np.eye(3).tolist(),
            right_world_rotation=np.eye(3).tolist(),
        )
    )
    for side in ("left", "right"):
        robot._driver[side], robot._rtsi[side] = _driver(), _rtsi()
    robot._is_connected = True
    _ROBOTS.append(robot)
    return robot


def _pose(rotvec, xyz=(0.3, 0.0, 0.4)):
    return np.concatenate([xyz, rotvec]).astype(np.float64)


# ── MoveJ keeps the servo alive ─────────────────────────────────────────────


@pytest.mark.parametrize("make", [_single, _bi])
def test_failed_movej_restarts_servo_and_reraises(make):
    robot = make()
    driver = robot._driver if isinstance(robot, EliteCS66RT) else robot._driver["left"]
    driver.moveJ.side_effect = RuntimeError("MoveJ timed out")
    args = ([0.0] * 6, 1.0) if isinstance(robot, EliteCS66RT) else ("left", [0.0] * 6, 1.0)
    with pytest.raises(RuntimeError, match="MoveJ timed out"):
        robot._move_j_blocking(*args)
    driver.stopServo.assert_called_once()
    driver.startServo.assert_called_once()


def test_movej_error_wins_over_restart_error():
    robot = _single()
    robot._driver.moveJ.side_effect = RuntimeError("MoveJ timed out")
    robot._driver.startServo.side_effect = RuntimeError("RTSI timestamp is not advancing")
    with pytest.raises(RuntimeError, match="MoveJ timed out"):
        robot._move_j_blocking([0.0] * 6, 1.0)


def test_restart_failure_after_successful_movej_is_raised():
    robot = _single()
    robot._driver.startServo.side_effect = RuntimeError("RTSI timestamp is not advancing")
    with pytest.raises(RuntimeError, match="not advancing"):
        robot._move_j_blocking([0.0] * 6, 1.0)


def test_movej_drops_the_stale_streamed_target():
    robot = _single()
    robot._last_tcp_command = _pose([0.0, 0.0, 1.0])
    robot._move_j_blocking([0.0] * 6, 1.0)
    assert robot._last_tcp_command is None


# ── Partial actions merge against the last submitted target ─────────────────


def test_position_only_action_keeps_the_in_flight_rotation():
    robot = _single()
    goal = _pose([0.0, 0.0, 1.2])
    robot._last_tcp_command, robot._last_submit_time = goal.copy(), time.monotonic()
    # The planner is still rotating: its commanded pose sits halfway.
    robot._driver.commanded_pose.return_value = _pose6_to_matrix(_pose([0.0, 0.0, 0.6]))

    target = robot._cartesian_action_to_tcp_pose({"tcp.x": 0.5})

    np.testing.assert_allclose(target[:3], [0.5, 0.0, 0.4])
    np.testing.assert_allclose(Rotation.from_rotvec(target[3:]).as_matrix(), _pose6_to_matrix(goal)[:3, :3])


def test_stale_target_falls_back_to_the_commanded_pose():
    robot = _single()
    robot._last_tcp_command = _pose([0.0, 0.0, 1.2])
    robot._last_submit_time = time.monotonic() - robot.config.command_stale_timeout_s - 1.0
    commanded = _pose([0.0, 0.0, 0.6])
    robot._driver.commanded_pose.return_value = _pose6_to_matrix(commanded)
    np.testing.assert_allclose(robot._merge_base_pose(), commanded, atol=1e-12)


def test_faulted_servo_merges_against_the_measured_tcp():
    robot = _single()
    robot._last_tcp_command, robot._last_submit_time = _pose([0.0, 0.0, 1.2]), time.monotonic()
    robot._driver.status.return_value = _status(running=False, error="RTSI feedback stale")
    np.testing.assert_allclose(robot._merge_base_pose()[:3], MEASURED_RPY[:3])


def test_bimanual_position_only_action_keeps_the_in_flight_rotation():
    robot = _bi()
    goal = _pose([0.0, 0.0, 1.2])
    robot._last_tcp_command["left"], robot._last_submit_time["left"] = goal.copy(), time.monotonic()
    robot._driver["left"].commanded_pose.return_value = _pose6_to_matrix(_pose([0.0, 0.0, 0.6]))

    target = robot._cartesian_action_to_tcp_pose("left", {"left_tcp.x": 0.5})

    np.testing.assert_allclose(Rotation.from_rotvec(target[3:]).as_matrix(), _pose6_to_matrix(goal)[:3, :3])


def test_submit_records_target_and_time():
    robot = _single()
    before = time.monotonic()
    robot.send_action({"tcp.x": 0.5})
    robot._driver.submit_target.assert_called_once()
    assert robot._last_tcp_command[0] == 0.5
    assert robot._last_submit_time >= before


# ── Commanded pose getter falls back to the measured TCP ────────────────────


def test_commanded_pose_getter_falls_back_when_servo_is_down():
    robot = _single()
    robot._driver.status.return_value = _status(running=False)
    np.testing.assert_allclose(robot.get_commanded_tcp_pose_euler()[:3], MEASURED_RPY[:3])
    robot._driver.commanded_pose.assert_not_called()


def test_bimanual_commanded_pose_getter_falls_back_per_arm():
    robot = _bi()
    robot._driver["left"].status.return_value = _status(running=False, error="fault")
    robot._driver["right"].commanded_pose.return_value = _pose6_to_matrix(_pose([0.0, 0.0, 0.0], (0.1, 0.2, 0.3)))
    left, right = robot.get_commanded_tcp_pose_euler()
    np.testing.assert_allclose(left[:3], MEASURED_RPY[:3])
    np.testing.assert_allclose(right[:3], [0.1, 0.2, 0.3])
    robot._driver["left"].commanded_pose.assert_not_called()


# ── Controller TCP is derived when tool_transform is unset ──────────────────


def _cs_with_flange(flange):
    model = Mock()
    model.fk.return_value = flange
    return SimpleNamespace(PinkIKModel=Mock(return_value=model))


def _rtsi_at(tcp_matrix):
    # Inverse of the Rz(yaw) @ Ry(pitch) @ Rx(roll) that RTSI feedback encodes.
    r = tcp_matrix[:3, :3]
    roll, pitch, yaw = np.arctan2(r[2, 1], r[2, 2]), -np.arcsin(r[2, 0]), np.arctan2(r[1, 0], r[0, 0])
    rtsi = Mock()
    rtsi.getTimestamp.return_value = 12.5
    rtsi.getActualJointPositions.return_value = [0.1] * 6
    rtsi.getActualTCPPose.return_value = [*tcp_matrix[:3, 3], roll, pitch, yaw]
    return rtsi


def test_derived_tool_recovers_the_pendant_tcp():
    flange = _pose6_to_matrix(_pose([0.3, -0.2, 0.5], (0.4, -0.1, 0.5)))
    tool = _pose6_to_matrix(_pose([0.0, 0.0, 0.2], (0.0019, -0.04363, 0.21187)))
    derived = _derive_controller_tool(_cs_with_flange(flange), _rtsi_at(flange @ tool), "model.urdf")
    np.testing.assert_allclose(np.reshape(derived, (4, 4)), tool, atol=1e-9)


def test_implausible_derived_tool_is_refused():
    flange = np.eye(4)
    far = np.eye(4)
    far[:3, 3] = [0.0, 0.0, 0.9]
    with pytest.raises(RuntimeError, match="from the model flange"):
        _derive_controller_tool(_cs_with_flange(flange), _rtsi_at(far), "model.urdf")


def test_derivation_needs_a_live_rtsi_stream():
    rtsi = _rtsi_at(np.eye(4))
    rtsi.getTimestamp.return_value = 0.0
    with pytest.raises(RuntimeError, match="coherent RTSI sample"):
        _derive_controller_tool(_cs_with_flange(np.eye(4)), rtsi, "model.urdf", timeout_s=0.05)


# ── Config plumbing ─────────────────────────────────────────────────────────


def _native_cfg():
    return SimpleNamespace(ik=SimpleNamespace(), planner=SimpleNamespace(), tool=None, model_path="")


def test_write_failure_tolerance_reaches_the_native_sender():
    native = _native_cfg()
    _configure_native_driver(native, EliteCS66RTConfig(servo_failure_tolerance_ticks=250), None)
    assert native.max_consecutive_write_failures == 250


def test_recipe_speed_cap_only_tightens_the_planner():
    native = _native_cfg()
    _configure_native_driver(native, EliteCS66RTConfig(max_lin_speed=0.25, max_ang_speed=2.0), None)
    assert native.planner.max_linear_velocity == 0.25
    assert native.planner.max_angular_velocity == 1.0


def test_single_and_bimanual_planner_defaults_agree():
    single, bi = EliteCS66RTConfig(), BiEliteCS66RTConfig()
    assert single.native_max_tcp_linear_velocity == bi.native_max_tcp_linear_velocity
    assert single.native_max_tcp_angular_velocity == bi.native_max_tcp_angular_velocity


def test_servoj_time_error_says_why():
    with pytest.raises(ValueError, match="overruns its period"):
        EliteCS66RTConfig(servoj_time=0.002)


@pytest.mark.parametrize("cls", [EliteCS66RT, BiEliteCS66RT])
def test_missing_timestamp_is_explained(cls, tmp_path):
    recipe = tmp_path / "output_recipe.txt"
    recipe.write_text("actual_TCP_pose\nactual_joint_positions\nactual_joint_speeds\nactual_joint_torques\n")
    robot = _single() if cls is EliteCS66RT else _bi()
    with pytest.raises(ValueError, match="timestamp: the native servo waits for it to advance"):
        robot._validate_output_recipe(str(recipe))


def test_a_libpyelite_without_the_tolerance_field_says_to_rebuild():
    class _OldConfig:  # pybind objects reject unknown attributes the same way
        __slots__ = ("ik", "planner", "command_timeout_ms")

    native = _OldConfig()
    native.ik, native.planner = SimpleNamespace(), SimpleNamespace()
    with pytest.raises(RuntimeError, match="rebuild it"):
        _configure_native_driver(native, EliteCS66RTConfig(), None)
