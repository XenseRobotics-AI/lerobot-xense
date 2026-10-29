"""Elite grip anchoring must leave ordinary Pico4 readers and other arms alone."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import draccus
import numpy as np
import pytest

from lerobot.robots.bi_elite_cs66_rt.config_bi_elite_cs66_rt import BiEliteCS66RTConfig
from lerobot.robots.bi_elite_cs66_rt.pico4 import EliteBiPico4, ElitePico4, make_elite_pico4_teleoperator
from lerobot.robots.elite_cs66_rt.config_elite_cs66_rt import EliteCS66RTConfig
from lerobot.teleoperators.bi_pico4.bi_pico4 import BiPico4
from lerobot.teleoperators.bi_pico4.config_bi_pico4 import BiPico4Config
from lerobot.teleoperators.pico4.config_pico4 import Pico4Config
from lerobot.teleoperators.pico4.teleop_pico4 import Pico4
from lerobot.utils.robot_utils import quaternion_to_rotation_6d

START = np.array([0.5, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0, 0.0])
MEASURED = np.array([0.55, 0.02, 0.31, 0.9238795, 0.0, 0.0, 0.3826834, 0.0])


class FakeXrt:
    def __init__(self):
        self.pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # xyzw, Pico4 frame
        self.left_grip = 0.0
        self.right_grip = 0.0
        self.left_reads = 0
        self.right_reads = 0

    def get_right_controller_pose(self):
        return list(self.pose)

    def get_left_controller_pose(self):
        return list(self.pose)

    def get_right_grip(self):
        self.right_reads += 1
        return self.right_grip

    def get_left_grip(self):
        self.left_reads += 1
        return self.left_grip

    def get_right_trigger(self):
        return 0.4

    def get_left_trigger(self):
        return 0.6

    def get_A_button(self):  # noqa: N802 - SDK method names
        return False

    get_B_button = get_A_button  # noqa: N815 - SDK method names
    get_X_button = get_A_button  # noqa: N815
    get_Y_button = get_A_button  # noqa: N815

    def close(self):
        pass


def connect_fake(teleop):
    xrt = FakeXrt()
    if isinstance(teleop, BiPico4):
        # Exercise BiPico4's inherited connect path with a pre-initialized SDK.
        teleop._xrt = xrt
        # connect() imports the SDK even when reusing _xrt. Stub that import too
        # so this test never depends on the real VR SDK being installed.
        with patch.dict(sys.modules, {"xensevr_pc_service_sdk": xrt}):
            teleop.connect(left_tcp_pose_quat=START.copy(), right_tcp_pose_quat=START.copy())
    else:
        teleop._xrt = xrt
        teleop._is_connected = True
        teleop.reset_to_pose(START)
    return xrt


def position(action, prefix=""):
    return np.array([action[f"{prefix}tcp.{axis}"] for axis in "xyz"])


def drag_then_regrip(teleop):
    xrt = connect_fake(teleop)
    xrt.right_grip = 1.0
    teleop.get_action()
    xrt.pose[0] += 0.2
    pushed = position(teleop.get_action())
    xrt.right_grip = 0.0
    teleop.get_action()
    xrt.right_grip = 1.0
    return pushed, teleop.get_action()


@pytest.mark.parametrize("limited", [False, True])
def test_regrip_uses_measured_position_and_orientation_immediately(limited, monkeypatch):
    monkeypatch.setattr("lerobot.teleoperators.pico4.teleop_pico4.time.time", lambda: 1000.0)
    source = Mock(return_value=MEASURED)
    teleop = ElitePico4(
        Pico4Config(max_pos_velocity=0.01 if limited else 0, max_rot_velocity=0.01 if limited else 0), source
    )
    _, action = drag_then_regrip(teleop)
    np.testing.assert_allclose(position(action), MEASURED[:3], atol=1e-6)
    np.testing.assert_allclose(
        [action[f"tcp.r{i}"] for i in range(1, 7)], quaternion_to_rotation_6d(*MEASURED[3:7]), atol=1e-6
    )
    assert action["gripper.pos"] == pytest.approx(0.6)  # Existing 1 - trigger mapping.
    assert source.call_count == 2
    assert teleop._xrt.right_reads == 4  # One SDK grip sample per action, including re-grip.


def test_grip_hysteresis_and_reset_read_the_tcp_only_on_a_new_gesture():
    source = Mock(return_value=MEASURED)
    teleop = ElitePico4(Pico4Config(), source)
    xrt = connect_fake(teleop)
    for grip in [0.0, 0.5, 0.6, 0.4, 0.31, 0.2, 0.4, 0.6]:
        xrt.right_grip = grip
        teleop.get_action()
    assert source.call_count == 2
    teleop.reset_to_pose(START)
    teleop.get_action()
    assert source.call_count == 3


@pytest.mark.parametrize("bad_pose", [None, 1.0, [0.0] * 6, np.zeros((7, 1)), [float("nan")] * 7])
def test_invalid_measurement_keeps_the_previous_target(bad_pose):
    teleop = ElitePico4(Pico4Config(max_pos_velocity=0, max_rot_velocity=0), lambda: bad_pose)
    pushed, action = drag_then_regrip(teleop)
    np.testing.assert_allclose(position(action), pushed, atol=1e-6)


def test_unreadable_robot_keeps_the_previous_target():
    source = Mock(side_effect=RuntimeError("RTSI down"))
    teleop = ElitePico4(Pico4Config(max_pos_velocity=0, max_rot_velocity=0), source)
    pushed, action = drag_then_regrip(teleop)
    np.testing.assert_allclose(position(action), pushed, atol=1e-6)


def test_bimanual_grips_anchor_independently():
    right_pose = MEASURED.copy()
    right_pose[1] = -0.2
    source = Mock(return_value=(MEASURED, right_pose))
    teleop = EliteBiPico4(BiPico4Config(max_pos_velocity=0, max_rot_velocity=0), source)
    xrt = connect_fake(teleop)
    xrt.left_grip = 1.0
    first = teleop.get_action()
    np.testing.assert_allclose(position(first, "left_"), MEASURED[:3], atol=1e-6)
    np.testing.assert_allclose(position(first, "right_"), START[:3], atol=1e-6)
    xrt.pose[0] += 0.05
    pushed = teleop.get_action()
    xrt.right_grip = 1.0
    regripped = teleop.get_action()
    np.testing.assert_allclose(position(regripped, "right_"), right_pose[:3], atol=1e-6)
    np.testing.assert_allclose(position(regripped, "left_"), position(pushed, "left_"), atol=1e-6)
    assert source.call_count == 2


def test_an_elite_reader_does_not_change_an_ordinary_pico4_instance():
    elite = ElitePico4(Pico4Config(), lambda: MEASURED)
    drag_then_regrip(elite)
    ordinary = Pico4(Pico4Config(max_pos_velocity=0, max_rot_velocity=0))
    pushed, action = drag_then_regrip(ordinary)
    np.testing.assert_allclose(position(action), pushed, atol=1e-6)
    assert not hasattr(ordinary, "set_grip_anchor_source")
    assert not hasattr(ordinary.config, "grip_anchor_to_robot")


@pytest.mark.parametrize(
    ("robot_cls", "teleop_cls", "ordinary_cls", "elite_cls"),
    [
        (EliteCS66RTConfig, Pico4Config, Pico4, ElitePico4),
        (BiEliteCS66RTConfig, BiPico4Config, BiPico4, EliteBiPico4),
    ],
)
@pytest.mark.parametrize("enabled", [False, True])
def test_factory_uses_the_elite_robot_setting(robot_cls, teleop_cls, ordinary_cls, elite_cls, enabled):
    robot = SimpleNamespace(
        config=draccus.decode(robot_cls, {"grip_anchor_to_robot": enabled}),
        get_current_tcp_pose_quat=Mock(),
    )
    teleop = make_elite_pico4_teleoperator(teleop_cls(), robot)
    assert type(teleop) is (elite_cls if enabled else ordinary_cls)
    robot.get_current_tcp_pose_quat.assert_not_called()  # Driver need not be connected at construction.
    if not enabled:
        xrt = connect_fake(teleop)
        xrt.left_grip = xrt.right_grip = 1.0
        teleop.get_action()
        robot.get_current_tcp_pose_quat.assert_not_called()


def test_factory_rejects_non_elite_robots():
    robot = SimpleNamespace(config=SimpleNamespace(type="flexiv_rizon4_rt"))
    with pytest.raises(ValueError, match="does not support flexiv"):
        make_elite_pico4_teleoperator(Pico4Config(), robot)


@pytest.mark.parametrize("entry", ["teleoperate", "record"])
@pytest.mark.parametrize(
    ("robot_type", "teleop_config", "elite"),
    [
        ("elite_cs66_rt", Pico4Config(), True),
        ("bi_elite_cs66_rt", BiPico4Config(), True),
        ("flexiv_rizon4_rt", Pico4Config(), False),
        ("bi_flexiv_rizon4_rt", BiPico4Config(), False),
        ("mock_robot", SimpleNamespace(type="mock_teleop"), False),
    ],
)
def test_cli_selects_the_adapter_only_for_elite_pairs(entry, robot_type, teleop_config, elite, monkeypatch):
    from lerobot.scripts import lerobot_record, lerobot_teleoperate

    module = lerobot_record if entry == "record" else lerobot_teleoperate
    robot = Mock()
    robot.get_observation.return_value = {}
    robot.get_current_tcp_pose_quat.return_value = (START, START) if robot_type.startswith("bi_") else START
    adapter_factory = Mock()
    ordinary_factory = Mock()
    monkeypatch.setattr(module, "asdict", lambda cfg: {})
    monkeypatch.setattr(module, "make_robot_from_config", lambda cfg: robot)
    monkeypatch.setattr(module, "make_teleoperator_from_config", ordinary_factory)
    monkeypatch.setattr("lerobot.robots.bi_elite_cs66_rt.pico4.make_elite_pico4_teleoperator", adapter_factory)
    cfg = SimpleNamespace(
        robot=SimpleNamespace(type=robot_type),
        teleop=teleop_config,
        display_data=False,
        display_compressed_images=False,
        dryrun=False,
        fps=30,
        teleop_time_s=0,
        debug_timing=False,
    )
    if entry == "record":
        # Stop after construction, before any dataset or hardware is opened.
        class StopAfterConstructionError(Exception):
            pass

        monkeypatch.setattr(module, "init_logging", lambda: None)
        monkeypatch.setattr(module, "_use_raw_passthrough_record", Mock(side_effect=StopAfterConstructionError))
        with pytest.raises(StopAfterConstructionError):
            module.record.__wrapped__(cfg)
    else:
        loop = Mock()
        for name in (
            "elite_cs66_rt_pico4_teleop_loop",
            "bi_pico4_teleop_loop",
            "pico4_teleop_loop",
            "mock_robot_teleop_loop",
        ):
            monkeypatch.setattr(module, name, loop)
        monkeypatch.setattr(module, "_cleanup", Mock())
        module.teleoperate.__wrapped__(cfg)
        loop.assert_called_once()
        assert loop.call_args.kwargs["teleop"] is (adapter_factory if elite else ordinary_factory).return_value
    if elite:
        adapter_factory.assert_called_once_with(teleop_config, robot)
        ordinary_factory.assert_not_called()
    else:
        ordinary_factory.assert_called_once_with(teleop_config)
        adapter_factory.assert_not_called()
