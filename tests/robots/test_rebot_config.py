"""B601 control-mode configuration works without the optional native SDK."""

import draccus
import pytest

from lerobot.robots.bi_rebot_b601_rt import BiRebotB601RTConfig
from lerobot.robots.rebot_b601_rt import RebotB601RTConfig


@pytest.mark.parametrize("family", ["dm", "rs"])
@pytest.mark.parametrize("mode", ["pos_vel", "POS_VEL"])
@pytest.mark.parametrize("limits", [None, [1.0] * 6])
def test_position_velocity_mode_disables_recipe_gravity(family, mode, limits, caplog):
    config = RebotB601RTConfig(
        motor_family=family,
        control_mode=mode,
        gravity_compensation_scale=1.0,
        gravity_torque_limits_nm=limits,
        gravity_use_joint_offsets=True,
    )
    assert config.control_mode == "pos_vel"
    assert config.gravity_compensation_scale == 0.0
    assert "automatically disables gravity feedforward" in caplog.text
    assert config.gravity_torque_limits_nm == limits


def test_disabled_gravity_does_not_emit_a_mode_override_warning(caplog):
    config = RebotB601RTConfig(control_mode="pos_vel", gravity_compensation_scale=0.0)
    assert config.gravity_compensation_scale == 0.0
    assert not caplog.records


def test_dual_arm_mode_override_only_disables_gravity_on_the_position_velocity_arm():
    config = draccus.decode(
        BiRebotB601RTConfig,
        {
            "left_arm_config": {
                "channel": "can0",
                "control_mode": "pos_vel",
                "gravity_compensation_scale": 1.0,
            },
            "right_arm_config": {
                "channel": "can1",
                "control_mode": "mit",
                "gravity_compensation_scale": 0.5,
                "gravity_torque_limits_nm": [1.0] * 6,
            },
        },
    )
    assert config.left_arm_config.gravity_compensation_scale == 0.0
    assert config.right_arm_config.gravity_compensation_scale == 0.5


def test_mit_gravity_still_requires_torque_limits():
    with pytest.raises(ValueError, match="Set gravity_torque_limits_nm"):
        RebotB601RTConfig(control_mode="mit", gravity_compensation_scale=1.0)


@pytest.mark.parametrize("scale", [-0.1, 1.1, float("nan"), float("inf")])
def test_mode_override_does_not_hide_invalid_gravity_scales(scale):
    with pytest.raises(ValueError, match="gravity_compensation_scale must be finite"):
        RebotB601RTConfig(control_mode="pos_vel", gravity_compensation_scale=scale)
