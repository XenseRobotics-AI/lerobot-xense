"""Check sequential B601 calibration with fake buses; never access hardware."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

from lerobot.robots.bi_rebot_b601_rt import BiRebotB601RTConfig, bi_calibration as dual
from lerobot.robots.rebot_b601_rt import RebotB601RTConfig, calibration as single
from lerobot.robots.rebot_b601_rt.config_rebot_b601_rt import DEFAULT_TACCAP_URDFS
from lerobot.robots.rebot_b601_rt.dm_serial_discovery import DMSerialDevice

ROOT = Path(__file__).resolve().parents[2]


def make_config(family="dm", **kwargs):
    return BiRebotB601RTConfig(
        id="bench",
        left_arm_config=RebotB601RTConfig(motor_family=family, channel="can1", has_gripper=False),
        right_arm_config=RebotB601RTConfig(motor_family=family, channel="can0", has_gripper=False),
        **kwargs,
    )


@pytest.fixture(params=["dm", "rs"])
def rig(request, tmp_path, monkeypatch):
    family = request.param
    config = make_config(family, calibration_dir=tmp_path)
    events, controllers, motors = [], {}, {}
    paths = {side: tmp_path / family / f"bench_{side}.json" for side in ("right", "left")}

    def defaults(_):
        return [
            SimpleNamespace(send_id=i, feedback_id=i + 16 if family == "dm" else 253, model="test") for i in range(1, 8)
        ]

    def open_controller(arm, **kwargs):
        side = arm.id.removeprefix("bench_")
        events.append((side, "open"))
        controller = MagicMock()
        controllers[side] = controller
        motors[side] = {}

        def add_motor(send_id, feedback_id, model):
            motor = MagicMock()
            motor.get_register_u32.return_value = send_id
            motor.robstride_ping_host_id.return_value = (send_id, feedback_id)
            motor.set_zero_position.side_effect = lambda: events.append((side, "zero"))
            motor.get_register_f32.return_value = 0.0002
            motor.robstride_get_param_f32_host_id.return_value = 0.0002
            motors[side][send_id] = motor
            return motor

        controller.add_damiao_motor.side_effect = add_motor
        controller.add_robstride_motor.side_effect = add_motor
        controller.close.side_effect = lambda: events.append((side, "close"))
        return controller

    monkeypatch.setattr(single, "_native_motor_defaults", defaults)
    opener = MagicMock(side_effect=open_controller)
    monkeypatch.setattr(single, "_open_controller", opener)
    monkeypatch.setattr(single.time, "sleep", lambda _: None)
    monkeypatch.setattr("builtins.input", lambda _: "")
    return SimpleNamespace(
        config=config, family=family, paths=paths, controllers=controllers, motors=motors, events=events, opener=opener
    )


def test_right_arm_finishes_and_closes_before_left_arm_starts(rig, monkeypatch):
    prompts = []

    def confirm(prompt):
        prompts.append(prompt)
        side = "right" if len(prompts) == 1 else "left"
        rig.controllers[side].disable_all.assert_called_once()
        for motor in rig.motors[side].values():
            motor.set_zero_position.assert_not_called()
        if side == "right":
            assert "left" not in rig.controllers
        else:
            assert rig.paths["right"].is_file()
            rig.controllers["right"].close.assert_called_once()
        return ""

    monkeypatch.setattr("builtins.input", confirm)
    dual.calibrate_bi_b601(rig.config)
    assert len(prompts) == 2
    assert rig.events.index(("right", "close")) < rig.events.index(("left", "open"))
    for side in ("right", "left"):
        record = json.loads(rig.paths[side].read_text())
        assert set(record) == {f"joint_{i}" for i in range(1, 7)}
        rig.controllers[side].enable_all.assert_not_called()
        rig.controllers[side].close.assert_called_once()
        for motor in rig.motors[side].values():
            motor.set_zero_position.assert_called_once()
            motor.enable.assert_not_called()
            motor.ensure_mode.assert_not_called()
            motor.close.assert_called_once()
    assert rig.config.right_arm_config.id is None
    assert rig.config.left_arm_config.id is None


@pytest.mark.parametrize("failure", [KeyboardInterrupt, RuntimeError])
def test_first_arm_failure_or_interrupt_never_starts_second_arm(rig, monkeypatch, failure):
    def confirm(_):
        if failure is KeyboardInterrupt:
            raise KeyboardInterrupt
        rig.motors["right"][3].set_zero_position.side_effect = RuntimeError("write failed")
        return ""

    monkeypatch.setattr("builtins.input", confirm)
    with pytest.raises(failure):
        dual.calibrate_bi_b601(rig.config)
    assert "left" not in rig.controllers
    assert not any(path.exists() for path in rig.paths.values())
    rig.controllers["right"].close.assert_called_once()
    for motor in rig.motors["right"].values():
        motor.close.assert_called_once()


def test_left_readback_failure_preserves_completed_right_and_old_left_records(rig, monkeypatch):
    rig.paths["left"].parent.mkdir()
    rig.paths["left"].write_text('{"previous": true}')

    def confirm(prompt):
        if "Calibration file exists" in prompt:
            return "c"
        if "left" in rig.motors:
            motor = rig.motors["left"][3]
            reader = motor.get_register_f32 if rig.family == "dm" else motor.robstride_get_param_f32_host_id
            reader.return_value = 0.03
        return ""

    monkeypatch.setattr("builtins.input", confirm)
    with pytest.raises(RuntimeError, match="Hardware zero verification failed"):
        dual.calibrate_bi_b601(rig.config)
    assert "joint_1" in json.loads(rig.paths["right"].read_text())
    assert rig.paths["left"].read_text() == '{"previous": true}'
    for controller in rig.controllers.values():
        controller.close.assert_called_once()


def test_keeping_right_record_still_calibrates_left(rig):
    rig.paths["right"].parent.mkdir()
    rig.paths["right"].write_text('{"previous": true}')
    dual.calibrate_bi_b601(rig.config)
    assert "right" not in rig.controllers
    assert rig.paths["right"].read_text() == '{"previous": true}'
    assert rig.paths["left"].is_file()


def test_second_open_failure_leaves_first_arm_closed_and_record_saved(rig):
    original_open = rig.opener.side_effect

    def open_controller(arm, **kwargs):
        if arm.id.endswith("_left"):
            raise ConnectionError("left disconnected")
        return original_open(arm, **kwargs)

    rig.opener.side_effect = open_controller
    with pytest.raises(ConnectionError, match="left disconnected"):
        dual.calibrate_bi_b601(rig.config)
    assert rig.paths["right"].is_file()
    assert not rig.paths["left"].exists()
    rig.controllers["right"].close.assert_called_once()


def test_auto_ports_are_pinned_from_one_snapshot_and_preserve_motor_overrides():
    config = make_config()
    config.right_arm_config.channel = "auto:first"
    config.left_arm_config.channel = "auto:second"
    config.left_arm_config.motor_configs = [{"send_id": 8, "model": "4310"}, {}, {}, {}, {}, {}, {}]
    devices = (DMSerialDevice("/dev/ttyACM4", 10), DMSerialDevice("/dev/ttyACM0", 20))
    with (
        patch.object(dual, "discover_dm_serial_devices", return_value=devices) as discover,
        patch.object(dual, "calibrate_b601") as calibrate,
    ):
        dual.calibrate_bi_b601(config)
    discover.assert_called_once()
    calls = calibrate.call_args_list
    assert [call.args[0].id for call in calls] == ["bench_right", "bench_left"]
    assert [call.args[0].channel for call in calls] == ["/dev/ttyACM4", "/dev/ttyACM0"]
    assert all(call.kwargs["dm_serial_devices"] == devices for call in calls)
    assert calls[1].args[0].motor_configs == config.left_arm_config.motor_configs
    assert config.right_arm_config.channel == "auto:first"


def test_aliases_of_the_same_serial_port_are_rejected_before_calibration(tmp_path):
    device = tmp_path / "ttyACM0"
    device.touch()
    alias = tmp_path / "right-arm"
    alias.symlink_to(device)
    config = make_config()
    config.left_arm_config.channel = str(device)
    config.right_arm_config.channel = str(alias)
    with patch.object(dual, "calibrate_b601") as calibrate, pytest.raises(ValueError, match="same communication"):
        dual.calibrate_bi_b601(config)
    calibrate.assert_not_called()


def test_resolved_auto_port_collision_is_rejected_before_calibration():
    config = make_config()
    config.right_arm_config.channel = "auto:first"
    config.left_arm_config.channel = "/dev/ttyACM0"
    with (
        patch.object(dual, "discover_dm_serial_devices", return_value=(DMSerialDevice("/dev/ttyACM0", 10),)),
        patch.object(dual, "calibrate_b601") as calibrate,
        pytest.raises(ValueError, match="same communication"),
    ):
        dual.calibrate_bi_b601(config)
    calibrate.assert_not_called()


def test_pinned_device_is_rechecked_before_opening_bus():
    config = RebotB601RTConfig(channel="/dev/ttyACM0", has_gripper=False)
    devices = (DMSerialDevice(config.channel, 10),)
    controller = MagicMock()
    with (
        patch.dict(sys.modules, {"motorbridge": SimpleNamespace(Controller=controller)}),
        patch.object(single, "verify_dm_serial_device", side_effect=RuntimeError("device changed")) as verify,
        pytest.raises(RuntimeError, match="device changed"),
    ):
        single._open_controller(config, dm_serial_devices=devices)
    verify.assert_called_once_with(config.channel, devices)
    controller.from_dm_serial.assert_not_called()


def test_missing_record_id_is_rejected_before_calibration():
    config = make_config()
    config.id = None
    with patch.object(dual, "calibrate_b601") as calibrate, pytest.raises(ValueError, match="robot.id"):
        dual.calibrate_bi_b601(config)
    calibrate.assert_not_called()


@pytest.mark.parametrize(
    "recipe", [None, {"robot": {"type": "rebot_b601_rt"}}, {"robot": {"type": "bi_rebot_b601_rt"}}]
)
def test_invalid_recipe_does_not_start_calibration(recipe, tmp_path, monkeypatch):
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(recipe))
    monkeypatch.setattr(sys, "argv", ["calibration", f"--config_path={path}"])
    with patch.object(dual, "calibrate_bi_b601") as calibrate, pytest.raises(ValueError):
        dual.main()
    calibrate.assert_not_called()


@pytest.mark.parametrize("operation", ["teleop", "record"])
@pytest.mark.parametrize("name", ["dm-taccap00-09", "dm-taccap00-10", "rs-taccap05"])
def test_loads_real_recipe_without_camera_or_native_sdks(operation, name, tmp_path, monkeypatch):
    for relative in DEFAULT_TACCAP_URDFS.values():
        model = tmp_path / "third_party/libpyrebot/rebot_rt/assets" / relative
        model.parent.mkdir(parents=True, exist_ok=True)
        model.write_text('<robot name="test"/>')
    monkeypatch.chdir(tmp_path)
    config = dual.load_calibration_config(ROOT / "recipes" / operation / "bi_rebot_b601_rt" / f"{name}.yaml")
    assert isinstance(config, BiRebotB601RTConfig)
    assert config.cameras == {}
    for arm in (config.left_arm_config, config.right_arm_config):
        assert arm.cameras == {}
        assert arm.has_gripper is False
        assert arm.gripper.type == "taccap_follower"


def test_main_reads_recipe_and_overrides_record_directory(tmp_path, monkeypatch):
    recipe = tmp_path / "recipe.yaml"
    # Keep CLI input independent of SDK assets and camera packages.
    recipe.write_text(
        yaml.safe_dump(
            {
                "robot": {
                    "type": "bi_rebot_b601_rt",
                    "id": "bench",
                    "left_arm_config": {"channel": "can1", "has_gripper": False},
                    "right_arm_config": {"channel": "can0", "has_gripper": False},
                },
                "teleop": {"type": "unused"},
                "dataset": {"repo_id": "unused"},
            }
        )
    )
    monkeypatch.setattr(
        sys, "argv", ["calibration", f"--config_path={recipe}", f"--calibration_dir={tmp_path / 'records'}"]
    )
    with patch.object(dual, "calibrate_bi_b601") as calibrate:
        dual.main()
    assert calibrate.call_args.args[0].calibration_dir == tmp_path / "records"


def test_help_does_not_import_hardware_sdks():
    code = """
import runpy
import sys
from importlib.abc import MetaPathFinder
class BlockSDKs(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ('rebot_rt', 'motorbridge', 'pyrealsense2', 'xensesdk'):
            raise AssertionError('Hardware SDK loaded during --help: ' + fullname)
sys.meta_path.insert(0, BlockSDKs())
sys.argv = ['calibration', '--help']
runpy.run_module('lerobot.robots.bi_rebot_b601_rt.bi_calibration', run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "right arm first" in result.stdout


def test_calibrate_cli_routes_bimanual_config_without_connecting_either_arm(rig, monkeypatch):
    from lerobot.scripts import lerobot_calibrate as cli

    arm_args = [
        f"--robot.{side}_arm_config.{key}={value}"
        for side, channel in (("left", "can1"), ("right", "can0"))
        for key, value in (("motor_family", rig.family), ("channel", channel), ("has_gripper", "false"))
    ]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lerobot-calibrate",
            "--robot.type=bi_rebot_b601_rt",
            "--robot.id=bench",
            f"--robot.calibration_dir={rig.config.calibration_dir}",
            *arm_args,
        ],
    )
    with (
        patch.object(cli, "register_third_party_plugins"),
        patch.object(cli, "make_robot_from_config") as robot_factory,
    ):
        cli.main()
    robot_factory.assert_not_called()
    assert [event for event in rig.events if event[1] == "open"] == [("right", "open"), ("left", "open")]
    for side in ("right", "left"):
        assert rig.paths[side].is_file()
        rig.controllers[side].enable_all.assert_not_called()
