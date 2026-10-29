"""URDF offset parsing is validated without the native SDK or model assets."""

import pytest
from defusedxml.common import EntitiesForbidden

from lerobot.robots.rebot_b601_rt.gravity_compensation import load_joint_torque_offsets


@pytest.fixture
def model_with_offsets(tmp_path):
    urdf = tmp_path / "model.urdf"
    joints = "".join(f'<joint name="joint{i}" type="revolute"/>' for i in range(6, 0, -1))
    urdf.write_text(f'<robot name="test">{joints}</robot>')
    offsets = "joints:\n" + "".join(f"  joint{i}: {{offset: {i / 100}}}\n" for i in range(6, 0, -1))
    urdf.with_name("model_offset.yaml").write_text(offsets)
    return urdf


def test_offsets_follow_native_joint_order(model_with_offsets):
    assert load_joint_torque_offsets(model_with_offsets) == [i / 100 for i in range(1, 7)]


@pytest.mark.parametrize("entity", ['"joint1"', 'SYSTEM "file:///unused-b601-entity"'])
def test_urdf_entities_are_rejected(model_with_offsets, entity):
    xml = model_with_offsets.read_text().replace('name="joint1"', 'name="&joint;"')
    model_with_offsets.write_text(f"<!DOCTYPE robot [<!ENTITY joint {entity}>]>{xml}")
    with pytest.raises(EntitiesForbidden):
        load_joint_torque_offsets(model_with_offsets)
