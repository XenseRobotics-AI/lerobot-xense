"""Read identified holding biases paired with a B601 URDF, before opening hardware."""

import math
from pathlib import Path

import yaml
from defusedxml import ElementTree


def load_joint_torque_offsets(urdf_path: str | Path) -> list[float]:
    urdf = Path(urdf_path)
    path = urdf.with_name(f"{urdf.stem}_offset.yaml")
    # B601 native joint order is joint1..joint6. Fail on a different model instead
    # of silently assigning a calibrated torque to the wrong motor.
    names = [f"joint{i}" for i in range(1, 7)]
    joints = {joint.get("name"): joint for joint in ElementTree.parse(urdf).findall("joint")}
    if any(name not in joints or joints[name].get("type") != "revolute" for name in names):
        raise ValueError(f"Identified offsets require B601 revolute joints joint1..joint6: {urdf}")
    with path.open() as stream:
        data = yaml.safe_load(stream)
    values = data.get("joints") if isinstance(data, dict) else None
    if not isinstance(values, dict) or set(values) != set(names):
        raise ValueError(f"Expected exactly joint1..joint6 in {path}")
    offsets = []
    for name in names:
        entry = values[name]
        value = entry.get("offset") if isinstance(entry, dict) else None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Expected finite offset in N m for {name} in {path}")
        offsets.append(float(value))
    return offsets
