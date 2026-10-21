"""Hardware-free joint reset, lifecycle and configuration contracts for Elite."""

import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from lerobot.robots.bi_elite_cs66_rt import bi_elite_cs66_rt as bi_module
from lerobot.robots.bi_elite_cs66_rt.bi_elite_cs66_rt import BiEliteCS66RT
from lerobot.robots.bi_elite_cs66_rt.config_bi_elite_cs66_rt import BiEliteCS66RTConfig
from lerobot.robots.elite_cs66_rt import elite_cs66_rt as single_module
from lerobot.robots.elite_cs66_rt.config_elite_cs66_rt import EliteCS66RTConfig
from lerobot.robots.elite_cs66_rt.elite_cs66_rt import (
    EliteCS66RT,
    _configure_native_driver,
    _validate_configuration_model,
)
from lerobot.robots.elite_cs66_rt.joint_reset import joint_sample


class Feedback:
    def __init__(self, q):
        self.q = np.asarray(q, dtype=float).copy()
        self.velocity = np.zeros(6)
        self.stamp = 1.0
        self.freeze = False
        self._second_read = False
        self.isConnected = Mock(return_value=True)
        self.connect = Mock(return_value=True)
        self.disconnect = Mock()

    def getTimestamp(self):  # noqa: N802 - SDK spelling
        if not self._second_read and not self.freeze:
            self.stamp += 0.01
        self._second_read = not self._second_read
        return self.stamp

    def getActualJointPositions(self):  # noqa: N802 - SDK spelling
        return self.q.copy()

    def getActualJointVelocity(self):  # noqa: N802 - SDK spelling
        return self.velocity.copy()

    def getActualTCPPose(self):  # noqa: N802 - SDK spelling
        return [0.4, 0.1, 0.3, 0.0, 0.0, 0.0]


def driver_for(feedback):
    driver = Mock()
    driver.status.return_value = SimpleNamespace(running=True, error="", reset_active=False)
    driver.isRobotConnected.return_value = True
    driver.commanded_pose.return_value = np.eye(4)
    driver.submit_joints.side_effect = lambda q: setattr(feedback, "q", np.array(q))
    return driver


def wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for mock reset")
        time.sleep(0.002)


@pytest.fixture(params=[False, True], ids=["single", "paired"])
def rig(request):
    cls = BiEliteCS66RTConfig if request.param else EliteCS66RTConfig
    config = cls(
        cameras={},
        reset_duration_s=0.04,
        reset_stable_dwell_s=0.02,
        reset_timeout_s=0.35,
        reset_feedback_timeout_s=0.10,
        trace_servoj=False,
    )
    if request.param:
        config.left_world_rotation = config.right_world_rotation = np.eye(3).tolist()
        robot = BiEliteCS66RT(config)
        feedback = {side: Feedback([0.1 if side == "left" else -0.2] * 6) for side in ("left", "right")}
        drivers = {side: driver_for(value) for side, value in feedback.items()}
        robot._driver, robot._rtsi = dict(drivers), dict(feedback)
        robot._q_start = {side: value.q.copy() for side, value in feedback.items()}
    else:
        robot = EliteCS66RT(config)
        feedback = {"arm": Feedback([0.1] * 6)}
        drivers = {"arm": driver_for(feedback["arm"])}
        robot._driver, robot._rtsi = drivers["arm"], feedback["arm"]
        robot._q_start = feedback["arm"].q.copy()
    robot._is_connected = True
    yield robot, drivers, feedback
    robot._joint_reset.close()
    robot._is_connected = False
    if request.param:
        robot._driver, robot._rtsi = dict.fromkeys(drivers), dict.fromkeys(drivers)
    else:
        robot._driver = robot._rtsi = None


def goals_for(robot):
    return robot._q_start if isinstance(robot, BiEliteCS66RT) else {"arm": robot._q_start}


def test_exact_saved_unwrapped_joints_and_async_status(rig):
    robot, drivers, feedback = rig
    expected = {side: q.copy() for side, q in goals_for(robot).items()}
    for feed in feedback.values():
        feed.q += 0.4
    if isinstance(robot, BiEliteCS66RT):
        robot.config.left_start_position_rad = robot.config.right_start_position_rad = [2.0] * 6
        robot._check_joint_move = Mock()
        robot._last_tcp_command = {side: np.ones(6) * 90 for side in drivers}
        robot._last_submit_time = dict.fromkeys(drivers, time.monotonic())
    else:
        robot.config.start_position_rad = [2.0] * 6
        robot._last_tcp_command = np.ones(6) * 90
        robot._last_submit_time = time.monotonic()
    robot.reset_to_initial_position()
    assert robot.rt_moving
    worker = robot._joint_reset._thread
    robot.reset_to_initial_position()
    assert robot._joint_reset._thread is worker
    # Even malformed actions cannot replace either joint or Cartesian reset.
    assert robot.send_action({"tcp.x": 90.0, "left_tcp.x": 90.0}) == {}
    wait_for(lambda: not robot._joint_reset.active)
    assert robot.rt_moving
    assert robot.rt_moving
    assert not robot.rt_moving
    assert robot.get_reset_status() == {"state": "succeeded", "active": False, "error": ""}
    for side, driver in drivers.items():
        np.testing.assert_array_equal(driver.submit_joints.call_args.args[0], expected[side])
        np.testing.assert_array_equal(goals_for(robot)[side], expected[side])
        driver.submit_target.assert_not_called()
        driver.moveJ.assert_not_called()
    if isinstance(robot, BiEliteCS66RT):
        checked = robot._check_joint_move.call_args.args[0]
        for side in drivers:
            np.testing.assert_array_equal(checked[side], expected[side])
        assert all(value is None for value in robot._last_tcp_command.values())
    else:
        assert robot._last_tcp_command is None


@pytest.mark.parametrize("mode", ["cartesian_servo", "joint_servo"])
def test_reset_mode_is_always_joint_and_suppresses_normal_sends(rig, mode):
    robot, drivers, _ = rig
    robot.config.control_mode = mode
    robot.reset_to_initial_position()
    assert robot.send_action({}) == {}
    wait_for(lambda: not robot._joint_reset.active)
    assert robot.get_reset_status()["state"] == "succeeded"
    assert all(d.submit_joints.called and not d.submit_target.called for d in drivers.values())


@pytest.mark.parametrize(
    "failure",
    ["wrong_joints", "wrapped_joints", "velocity", "stale", "disconnected", "socket", "fault", "stopped", "invalid"],
)
def test_no_tcp_only_timeout_or_fault_success(rig, failure):
    robot, drivers, feedback = rig
    side = next(iter(drivers))
    driver, feed = drivers[side], feedback[side]
    # TCP remains exactly the startup TCP for every failure.
    if failure in {"wrong_joints", "wrapped_joints"}:
        feed.q += 0.1 if failure == "wrong_joints" else 2 * np.pi
        driver.submit_joints.side_effect = None
    elif failure == "velocity":
        feed.velocity[:] = 0.1
    elif failure == "stale":
        feed.freeze = True
    elif failure == "disconnected":
        feed.isConnected.return_value = False
    elif failure == "socket":
        driver.isRobotConnected.return_value = False
    elif failure == "fault":
        driver.check_error.side_effect = RuntimeError("native fault")
    elif failure == "stopped":
        driver.status.return_value.running = False
    elif failure == "invalid":
        feed.q[0] = np.nan
    robot.reset_to_initial_position()
    wait_for(lambda: not robot._joint_reset.active)
    assert robot.get_reset_status()["state"] == "failed"
    with pytest.raises(RuntimeError, match="joint reset failed"):
        _ = robot.rt_moving
    with pytest.raises(RuntimeError, match="joint reset failed"):
        robot.send_action({})
    assert all(d.cancel.call_count >= 2 for d in drivers.values())


def test_velocity_and_angle_must_stay_settled_for_the_whole_dwell(rig):
    robot, drivers, feedback = rig
    robot.config.reset_stable_dwell_s = 0.06
    robot.config.reset_timeout_s = 1.0
    robot.reset_to_initial_position()
    wait_for(lambda: robot.get_reset_status()["state"] == "moving")
    feed = feedback[next(iter(feedback))]
    feed.velocity[:] = 0.03
    time.sleep(0.08)
    assert robot.rt_moving
    feed.velocity[:] = 0.0
    time.sleep(0.02)
    assert robot.rt_moving
    wait_for(lambda: not robot._joint_reset.active)
    assert robot.get_reset_status()["state"] == "succeeded"


def test_preflight_rejection_is_atomic_for_pair(rig):
    robot, drivers, _ = rig
    if not isinstance(robot, BiEliteCS66RT):
        pytest.skip("station exists only for paired robot")
    robot._station = Mock()
    robot._station.check_joint_path.return_value = SimpleNamespace(
        safe=False, minimum_distance=0.0, pair=("left", "right"), clearance=0.02
    )
    robot.reset_to_initial_position()
    wait_for(lambda: not robot._joint_reset.active)
    assert "station collision" in robot.get_reset_status()["error"]
    assert all(not d.submit_joints.called for d in drivers.values())
    robot._station = None


def test_submit_failure_cancels_the_whole_transaction(rig):
    robot, drivers, _ = rig
    list(drivers.values())[-1].submit_joints.side_effect = RuntimeError("submit rejected")
    robot.reset_to_initial_position()
    wait_for(lambda: not robot._joint_reset.active)
    assert robot.get_reset_status()["state"] == "failed"
    assert "submit rejected" in robot.get_reset_status()["error"]
    assert all(d.cancel.call_count >= 2 for d in drivers.values())


def test_reset_retry_after_feedback_failure(rig):
    robot, _, feedback = rig
    feed = next(iter(feedback.values()))
    feed.freeze = True
    robot.reset_to_initial_position()
    wait_for(lambda: not robot._joint_reset.active)
    assert robot.get_reset_status()["state"] == "failed"
    feed.freeze = False
    robot.reset_to_initial_position()
    wait_for(lambda: not robot._joint_reset.active)
    assert robot.get_reset_status()["state"] == "succeeded"


def test_disconnect_cancels_worker_before_teardown_and_clears_start(rig):
    robot, drivers, feedback = rig
    for feed in feedback.values():
        feed.velocity[:] = 0.1
    robot.reset_to_initial_position()
    robot.disconnect()
    assert not robot._joint_reset._thread.is_alive()
    assert robot.get_reset_status()["state"] == "cancelled"
    assert all(q is None for q in goals_for(robot).values())
    calls = [d.submit_joints.call_count for d in drivers.values()]
    time.sleep(0.02)
    assert calls == [d.submit_joints.call_count for d in drivers.values()]
    with pytest.raises(Exception, match="disconnect"):
        robot.send_action({})


def test_blocking_compatibility_waits_and_raises(rig):
    robot, _, feedback = rig
    robot.config.use_background_servo_loop = False
    robot.reset_to_initial_position()
    assert robot.get_reset_status()["state"] == "succeeded"
    assert robot.rt_moving  # Preserve the resync edge through hasattr + read.
    assert robot.rt_moving
    assert not robot.rt_moving
    next(iter(feedback.values())).freeze = True
    with pytest.raises(RuntimeError, match="stale"):
        robot.reset_to_initial_position()


@pytest.mark.parametrize("completion", ["blocking", "before_poll", "between_reads"])
def test_recording_resyncs_before_sending_after_reset(rig, monkeypatch, completion):
    robot, drivers, feedback = rig
    stale_action = robot.get_observation()
    for key in stale_action:
        if key.endswith("tcp.x"):
            stale_action[key] += 0.1
    robot.send_action(stale_action)
    for driver in drivers.values():
        driver.submit_target.reset_mock()
    teleop_action = stale_action
    events = []

    if completion == "between_reads":
        entered, release = threading.Event(), threading.Event()

        def preflight(targets):
            entered.set()
            assert release.wait(1.0)

        robot._joint_reset.start(drivers, feedback, goals_for(robot), preflight, robot._clear_reset_targets)
        assert entered.wait(1.0)
        moving = robot._joint_reset.moving
        first_read = True

        def complete_after_hasattr_read():
            nonlocal first_read
            result = moving()
            if first_read:
                first_read = False
                assert result is True
                release.set()
                robot._joint_reset._thread.join(1.0)
                assert robot.get_reset_status()["state"] == "succeeded"
            return result

        monkeypatch.setattr(robot._joint_reset, "moving", complete_after_hasattr_read)
    else:
        robot.config.use_background_servo_loop = completion != "blocking"
        robot.reset_to_initial_position()
        robot._joint_reset._thread.join(1.0)
        assert robot.get_reset_status()["state"] == "succeeded"

    prev_rt_moving = False
    moving_frames = []
    for _ in range(3):
        # Keep the recording expression verbatim: hasattr evaluates the property.
        robot_is_moving = hasattr(robot, "rt_moving") and robot.rt_moving
        moving_frames.append(robot_is_moving)
        assert robot.get_reset_status() == {"state": "succeeded", "active": False, "error": ""}
        if robot_is_moving:
            prev_rt_moving = True
        if prev_rt_moving and not robot_is_moving:
            prev_rt_moving = False
            events.append("resync")
            teleop_action = robot.get_observation()
        if not robot_is_moving:
            events.append("send")
            robot.send_action(teleop_action)
        else:
            assert all(not driver.submit_target.called for driver in drivers.values())
            if isinstance(robot, BiEliteCS66RT):
                assert all(value is None for value in robot._last_tcp_command.values())
            else:
                assert robot._last_tcp_command is None

    assert moving_frames == [True, False, False]
    assert events == ["resync", "send", "send"]
    for side, driver in drivers.items():
        assert driver.submit_target.call_count == 2
        for submission in driver.submit_target.call_args_list:
            np.testing.assert_allclose(submission.args[0][:3, 3], feedback[side].getActualTCPPose()[:3])
        assert driver.cancel.call_count == 2


def test_missing_start_never_uses_recipe_fallback(rig):
    robot, drivers, _ = rig
    if isinstance(robot, BiEliteCS66RT):
        robot._q_start["right"] = None
    else:
        robot._q_start = None
    with pytest.raises(RuntimeError, match="No measured startup"):
        robot.reset_to_initial_position()
    assert all(not d.submit_joints.called for d in drivers.values())


@pytest.mark.parametrize("go_to_start", [False, True])
def test_startup_captures_actual_joints_only_after_success(rig, monkeypatch, go_to_start):
    robot, drivers, feedback = rig
    robot._is_connected = False
    expected = {}

    def move(side, target, *args, **kwargs):
        # Simulate small measured error relative to the recipe, not exact goals.
        feedback[side].q = np.asarray(target) + 0.0001

    if isinstance(robot, BiEliteCS66RT):
        robot._connect_arm = Mock()
        monkeypatch.setattr(bi_module, "_import_elite_sdk", lambda: SimpleNamespace())
        robot._move_j_blocking = Mock(side_effect=move)
    else:
        dashboard = Mock()
        sdk = SimpleNamespace(
            RtsiIOInterface=Mock(return_value=feedback["arm"]),
            DashboardClientInterface=Mock(return_value=dashboard),
            EliteDriver=Mock(return_value=drivers["arm"]),
        )
        monkeypatch.setattr(single_module, "_import_elite_sdk", lambda: sdk)
        robot._make_driver_config = Mock()
        robot._resolve_recipe = Mock(return_value="unused.txt")
        robot._validate_output_recipe = Mock()
        robot.config.external_control_settle_s = 0
        robot._move_j_blocking = Mock(side_effect=lambda *args: move("arm", *args))
    robot.connect(go_to_start=go_to_start)
    assert robot._move_j_blocking.called == go_to_start
    for side, feed in feedback.items():
        expected[side] = feed.q.copy()
        np.testing.assert_array_equal(goals_for(robot)[side], expected[side])
        feed.q += 0.3
        robot.get_current_tcp_pose_quat()  # Grip anchoring reads measured TCP only.
        np.testing.assert_array_equal(goals_for(robot)[side], expected[side])
    # A failed subsequent startup invalidates the previous connection's targets.
    robot._is_connected = False
    robot._start_servo_loop = Mock(side_effect=RuntimeError("startup failed"))
    with pytest.raises(RuntimeError, match="startup failed"):
        robot.connect(go_to_start=False)
    assert all(q is None for q in goals_for(robot).values())


def test_command_gate_serializes_send_and_reset(rig):
    robot, drivers, _ = rig
    entered, release = threading.Event(), threading.Event()

    def slow_send(*args):
        entered.set()
        assert release.wait(1.0)

    next(iter(drivers.values())).submit_target.side_effect = slow_send
    sending = threading.Thread(target=robot.send_action, args=({},))
    sending.start()
    assert entered.wait(1.0)
    resetting = threading.Thread(target=robot.reset_to_initial_position)
    resetting.start()
    assert not robot._joint_reset.active  # Reset cannot pass the in-flight send.
    release.set()
    sending.join(1.0)
    resetting.join(1.0)
    assert not sending.is_alive() and not resetting.is_alive()
    assert robot._joint_reset.active
    assert robot.send_action({}) == {}
    wait_for(lambda: not robot._joint_reset.active)
    assert robot.get_reset_status()["state"] == "succeeded"


def test_thread_start_failure_is_latched_and_disconnect_is_safe(rig, monkeypatch):
    robot, _, _ = rig
    with monkeypatch.context() as patcher:
        patcher.setattr(threading.Thread, "start", Mock(side_effect=RuntimeError("thread start failed")))
        with pytest.raises(RuntimeError, match="thread start failed"):
            robot.reset_to_initial_position()
    assert robot.get_reset_status()["state"] == "failed"
    assert not robot._joint_reset.active
    robot.disconnect()  # No attempt to join an unstarted thread.


def test_cancel_failure_is_not_reported_as_success(rig):
    robot, drivers, _ = rig
    next(iter(drivers.values())).cancel.side_effect = RuntimeError("cancel failed")
    robot.reset_to_initial_position()
    wait_for(lambda: not robot._joint_reset.active)
    assert robot.get_reset_status()["state"] == "failed"
    assert "cancel failed" in robot.get_reset_status()["error"]
    assert all(not d.submit_joints.called for d in drivers.values())


def test_reset_cancel_during_preflight_does_not_send_late_targets(rig):
    robot, drivers, feedback = rig
    entered, release = threading.Event(), threading.Event()

    def preflight(targets):
        entered.set()
        assert release.wait(1.0)

    robot._joint_reset.start(drivers, feedback, goals_for(robot), preflight, robot._clear_reset_targets)
    assert entered.wait(1.0)
    closing = threading.Thread(target=robot._joint_reset.close)
    closing.start()
    wait_for(lambda: robot._joint_reset.closing)
    with pytest.raises(RuntimeError, match="disconnecting"):
        robot.reset_to_initial_position()
    release.set()
    closing.join(1.0)
    assert not closing.is_alive()
    assert robot.get_reset_status()["state"] == "cancelled"
    assert all(not d.submit_joints.called for d in drivers.values())


def test_station_preflight_timeout_does_not_start_motion(rig):
    robot, drivers, feedback = rig
    robot.config.reset_timeout_s = 0.08
    robot._joint_reset.start(
        drivers, feedback, goals_for(robot), lambda targets: time.sleep(0.09), robot._clear_reset_targets
    )
    wait_for(lambda: not robot._joint_reset.active)
    assert robot.get_reset_status()["state"] == "failed"
    assert "preflight" in robot.get_reset_status()["error"]
    assert all(not d.submit_joints.called for d in drivers.values())


def test_backwards_feedback_timestamp_fails(rig):
    robot, _, feedback = rig
    robot.reset_to_initial_position()
    wait_for(lambda: robot.get_reset_status()["state"] == "moving")
    next(iter(feedback.values())).stamp = 0.01
    wait_for(lambda: not robot._joint_reset.active)
    assert "backwards" in robot.get_reset_status()["error"]


@pytest.mark.parametrize("cls", [EliteCS66RTConfig, BiEliteCS66RTConfig])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"reset_joint_tolerance_rad": float("nan")},
        {"reset_velocity_tolerance_rad_s": 0},
        {"reset_stable_dwell_s": 0},
        {"reset_feedback_timeout_s": -1},
        {"reset_timeout_s": 3.1},
        {"native_configuration_margin": float("nan")},
        {"native_configuration_margin": 0},
        {"native_configuration_joint_span": np.pi},
        {"native_boundary_orientation_tolerance": 0},
        {"native_preserve_configuration": "true"},
        {"native_joint_motion_limits": False},
        {"native_max_joint_velocity": None},
        {"native_max_joint_acceleration": [1] * 5},
        {"native_max_joint_jerk": [0] * 6},
    ],
)
def test_invalid_reset_or_native_policy_config(cls, kwargs):
    with pytest.raises(ValueError):
        cls(**kwargs)


@pytest.mark.parametrize(
    "field",
    ["preserve_configuration", "configuration_margin", "configuration_joint_span", "boundary_orientation_tolerance"],
)
def test_unavailable_native_policy_field_is_explicit(field):
    planner = SimpleNamespace(
        preserve_configuration=False,
        configuration_margin=0.1,
        configuration_joint_span=3.0,
        boundary_orientation_tolerance=0.01,
    )
    delattr(planner, field)
    native = SimpleNamespace(max_consecutive_write_failures=1, planner=planner)
    with pytest.raises(RuntimeError, match=field):
        _configure_native_driver(native, EliteCS66RTConfig(), None)


def test_policy_mapping_is_explicit_without_native_extension(monkeypatch):
    delegate = Mock()
    monkeypatch.setitem(sys.modules, "libpyelite.control_profile", SimpleNamespace(configure_native_driver=delegate))
    config = EliteCS66RTConfig(
        native_configuration_margin=0.08,
        native_configuration_joint_span=2.8,
        native_boundary_orientation_tolerance=0.02,
    )
    planner = SimpleNamespace(
        preserve_configuration=False,
        configuration_margin=0.0,
        configuration_joint_span=0.0,
        boundary_orientation_tolerance=0.0,
    )
    native = SimpleNamespace(max_consecutive_write_failures=1, planner=planner)
    _configure_native_driver(native, config, None)
    delegate.assert_called_once_with(native, config, None, None, None)
    assert vars(planner) == {
        "preserve_configuration": True,
        "configuration_margin": 0.08,
        "configuration_joint_span": 2.8,
        "boundary_orientation_tolerance": 0.02,
    }


@pytest.mark.parametrize("bad", ["lower", "upper", "velocity", "shape", "order"])
def test_unbounded_or_invalid_configuration_models_rejected(bad):
    model = SimpleNamespace(lower_limits=[-6.0] * 6, upper_limits=[6.0] * 6, velocity_limits=[3.0] * 6)
    if bad == "lower":
        model.lower_limits[0] = -np.inf
    elif bad == "upper":
        model.upper_limits[0] = np.nan
    elif bad == "velocity":
        model.velocity_limits[0] = 0
    elif bad == "shape":
        model.lower_limits = [-6.0] * 7
    else:
        model.upper_limits[0] = -6.0
    cs = SimpleNamespace(PinkIKModel=Mock(return_value=model), model_path=Mock(return_value="default.urdf"))
    cfg = SimpleNamespace(model_path="assembly-right.urdf", planner=SimpleNamespace(preserve_configuration=True))
    with pytest.raises(ValueError, match="six finite bounded joints"):
        _validate_configuration_model(cs, cfg)
    cs.PinkIKModel.assert_called_once_with("assembly-right.urdf")


def test_incoherent_feedback_is_not_a_saved_start():
    feed = Feedback([0.0] * 6)
    feed.getTimestamp = Mock(side_effect=range(1, 11))
    with pytest.raises(RuntimeError, match="coherent"):
        joint_sample(feed)
