#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Keep every recipe's `gripper:` block honest.

Each recipe lists the knobs it does not set as commented-out lines carrying that
knob's default, so the whole parameter surface is visible where you edit it. That
is documentation living next to code, which means it rots: add a field to a
gripper config, or change a default, and the comments quietly start lying.

These tests re-derive both from the dataclass and fail when they drift.
"""

import dataclasses
import re
from pathlib import Path

import pytest
import yaml

from lerobot.grippers import GripperConfig, SerialGripperConfig, TaccapFollowerConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
RECIPES = sorted((REPO_ROOT / "recipes").rglob("*.yaml"))
CLS: dict[str, type[GripperConfig]] = {
    "serial": SerialGripperConfig,
    "taccap_follower": TaccapFollowerConfig,
}

# Most TacCap recipes intentionally keep their historical ControlLoop block and
# inherit the newly exposed SDK safety/controller defaults. A recipe becomes a
# full controller reference by advertising any field in this set; from then on
# the ordinary completeness check requires the whole set. forward-01-taccap.yaml
# is the bilateral platform's canonical full reference.
TACCAP_ADVANCED_CONTROLLER_FIELDS = {
    "controller",
    "submit_phase",
    "max_position_torque_nm",
    "rated_torque_nm",
    "rated_hold_ms",
    "rated_release_rad",
    "stall_torque_nm",
    "stall_vel_radps",
    "stall_hold_ms",
    "stall_action",
    "motor_stream_hz",
    "print_status",
    "status_print_hz",
    "close_position",
    "close_speed_radps",
    "grasp_torque_nm",
    "hold_torque_limit_nm",
    "motion_torque_limit_nm",
    "contact_torque_nm",
    "contact_vel_radps",
    "contact_vel_ratio",
    "contact_moved_rad",
    "position_kp",
    "position_kd",
    "brake_distance_rad",
    "close_endpoint_tolerance_rad",
    "contact_samples",
    "startup_guard_ms",
    "status_timeout_ms",
}
TACCAP_CONTROLLER_REFERENCE = (
    REPO_ROOT / "recipes/teleop/bi_flexiv_rizon4_rt/forward-01-taccap.yaml"
)

# `# <name>: <value>` inside a gripper block, ignoring any trailing `# comment`.
COMMENTED = re.compile(r"^\s*#\s*([a-z_]+):\s*([^#\n]+?)\s*(?:#.*)?$")


def _gripper_block_lines(text: str) -> list[str]:
    m = re.search(r"^  gripper:\n((?:    .*\n|\n)+)", text, re.M)
    return m.group(1).splitlines() if m else []


def _recipes_with_grippers():
    out = []
    for path in RECIPES:
        doc = yaml.safe_load(path.read_text())
        block = (doc.get("robot") or {}).get("gripper")
        if block:
            out.append(pytest.param(path, block, id=f"{path.parent.name}/{path.name}"))
    return out


CASES = _recipes_with_grippers()


def test_some_recipes_configure_a_gripper():
    """Guard the guard: a bad glob or schema change must not silently empty this."""
    assert CASES, "no recipe with a gripper: block was found — check the discovery above"


def test_forward_01_is_the_complete_taccap_controller_reference():
    doc = yaml.safe_load(TACCAP_CONTROLLER_REFERENCE.read_text())
    block = doc["robot"]["gripper"]

    assert block["type"] == "taccap_follower"
    assert block["controller"] in {"control_loop", "force_position"}
    assert set(block) >= TACCAP_ADVANCED_CONTROLLER_FIELDS


def test_forward_01_force_position_holds_configured_torque_at_zero():
    block = yaml.safe_load(TACCAP_CONTROLLER_REFERENCE.read_text())["robot"]["gripper"]

    assert block["controller"] == "force_position"
    assert block["close_position"] == pytest.approx(0.0)
    assert block["grasp_torque_nm"] == pytest.approx(1.8)
    assert block["hold_torque_limit_nm"] >= block["grasp_torque_nm"]


@pytest.mark.parametrize(("path", "block"), CASES)
def test_every_knob_is_present_live_or_commented(path, block):
    """A knob the recipe neither sets nor documents is invisible to whoever edits
    it — which is the whole reason the commented list exists."""
    cls = CLS[block["type"]]
    lines = _gripper_block_lines(path.read_text())
    commented = {m.group(1) for line in lines if (m := COMMENTED.match(line))}
    live = set(block)

    # protocol_fixed_fields are deliberately left out: they are pinned by the wire
    # protocol or the wiring, so listing them would pad the block with knobs that
    # cannot actually be turned. The class owns that decision, not this test.
    expected = {f.name for f in dataclasses.fields(cls)} - cls.protocol_fixed_fields
    advertised = live | commented
    if cls is TaccapFollowerConfig and not advertised & TACCAP_ADVANCED_CONTROLLER_FIELDS:
        expected -= TACCAP_ADVANCED_CONTROLLER_FIELDS
    missing = expected - advertised
    assert not missing, (
        f"{path.relative_to(REPO_ROOT)}: {sorted(missing)} accepted by "
        f"{cls.__name__} but neither set nor listed as a comment. Add them "
        "(commented, at their default) so the block stays a complete reference, "
        f"or add them to {cls.__name__}.protocol_fixed_fields if they cannot be changed."
    )


@pytest.mark.parametrize(("path", "block"), CASES)
def test_commented_values_match_the_dataclass_defaults(path, block):
    """A commented value that no longer matches its default is worse than no
    comment: it reads as authoritative."""
    cls = CLS[block["type"]]
    defaults = {}
    for f in dataclasses.fields(cls):
        d = f.default if f.default is not dataclasses.MISSING else f.default_factory()
        defaults[f.name] = d

    wrong = []
    for line in _gripper_block_lines(path.read_text()):
        m = COMMENTED.match(line)
        if not m or m.group(1) not in defaults:
            continue
        name, raw = m.group(1), m.group(2)
        if yaml.safe_load(raw) != defaults[name]:
            wrong.append(f"{name}: comment says {raw!r}, default is {defaults[name]!r}")

    assert not wrong, f"{path.relative_to(REPO_ROOT)}: stale commented defaults — " + "; ".join(wrong)


@pytest.mark.parametrize(("path", "block"), CASES)
def test_protocol_fixed_fields_are_not_advertised(path, block):
    """The point of dropping them is that the block only shows knobs worth turning;
    if one creeps back in, the block starts implying it is tunable."""
    cls = CLS[block["type"]]
    lines = _gripper_block_lines(path.read_text())
    commented = {m.group(1) for line in lines if (m := COMMENTED.match(line))}
    leaked = (commented | set(block)) & cls.protocol_fixed_fields
    assert not leaked, (
        f"{path.relative_to(REPO_ROOT)}: {sorted(leaked)} is in "
        f"{cls.__name__}.protocol_fixed_fields — fixed by the protocol or the "
        "wiring — so it should not appear in the recipe block."
    )


@pytest.mark.parametrize(("path", "block"), CASES)
def test_side_is_never_pinned_in_a_bimanual_recipe(path, block):
    """`side` is stamped per arm when the shared block is cloned. Setting it in the
    recipe would give both arms the same side, and the second gripper would never
    be found."""
    if not path.parent.name.startswith("bi_"):
        return
    assert "side" not in block, (
        f"{path.relative_to(REPO_ROOT)}: bimanual recipes must not pin `side` — "
        "the arm stamps it per side when cloning the shared block."
    )
