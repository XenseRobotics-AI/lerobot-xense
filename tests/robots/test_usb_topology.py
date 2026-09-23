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
"""USB-hub grouping, and the serial-gripper camera discovery built on it.

The whole point of this machinery is that a camera never has to be named in a
config: the gripper resolves its own side, its USB hub identifies the arm, and
whatever else hangs off that hub belongs to the same arm. These tests pin the two
parts that could silently produce a *plausible but wrong* answer — sensor
ordering, and the "everything on the hub that is not tactile is the wrist camera"
inference — because either one getting it wrong mislabels a dataset in a way that
only shows up long after the recording.

Topology is faked here (no hardware). The end-to-end check runs on a bench; see
recipes/README.md.
"""

from importlib.util import find_spec

import pytest

from lerobot.grippers import SerialGripperConfig, TaccapFollowerConfig, usb_topology as topo
from tests.utils import require_package

HAS_FLEXIV = find_spec("flexiv_rt") is not None
HAS_ELITE = find_spec("elite_cs_sdk") is not None


# ── Path parsing ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("sysfs", "hub", "port"),
    [
        # Observed shapes: a gripper MCU behind a hub, and a camera two levels down.
        ("/sys/devices/pci0000:00/0000:00:14.0/usb3/3-1/3-1.1/3-1.1:1.2/tty/ttyACM1", "3-1", "3-1.1"),
        ("/sys/devices/pci0000:80/0000:80:14.0/usb3/3-7/3-7.4/3-7.4:1.0/video4linux/video6", "3-7", "3-7.4"),
        # Deeper nesting still reports the TOP hub — that is what groups one arm.
        ("/sys/devices/pci0000:00/usb1/1-3/1-3.2/1-3.2.4/1-3.2.4:1.0/video4linux/video9", "1-3", "1-3.2"),
        # Straight into a root port, no intervening hub: the root port itself is
        # the group token, and there is no sub-port to order by. Devices sharing a
        # root port would therefore group together — fine here, because a gripper
        # arm is only ever anchored via its own hub.
        ("/sys/devices/pci0000:00/usb2/2-4/2-4:1.0/video4linux/video0", "2-4", None),
    ],
)
def test_usb_hub_and_port_parsing(sysfs, hub, port):
    got_hub, got_port = topo.usb_hub_and_port(sysfs)
    assert got_hub == hub
    if port is None:
        assert got_port is None
    else:
        # The port token must start at the hub — otherwise devices on different
        # arms could be ordered against each other.
        assert got_port is not None and got_port.startswith(hub)


@pytest.mark.parametrize(
    ("cam", "expected"),
    [(6, "/dev/video6"), ("/dev/video12", "/dev/video12"), ("4", "/dev/video4")],
)
def test_video_node_accepts_both_sdk_shapes(cam, expected):
    """scanSerialNumber() returns an int index on some SDK builds, a path on others."""
    assert topo.video_node(cam) == expected


# ── Grouping and ordering ───────────────────────────────────────────────────

# Two arms, each a hub with 2 tactile sensors + 1 wrist camera, plus an unrelated
# webcam straight on a root port. Deliberately declared out of port order.
FAKE_SYSFS = {
    "/dev/video20": "/sys/devices/pci0000:00/usb3/3-1/3-1.4/3-1.4:1.0/video4linux/video20",
    "/dev/video16": "/sys/devices/pci0000:00/usb3/3-1/3-1.2/3-1.2:1.0/video4linux/video16",
    "/dev/video19": "/sys/devices/pci0000:00/usb3/3-1/3-1.3/3-1.3:1.0/video4linux/video19",
    "/dev/video4": "/sys/devices/pci0000:00/usb3/3-7/3-7.2/3-7.2:1.0/video4linux/video4",
    "/dev/video8": "/sys/devices/pci0000:00/usb3/3-7/3-7.4/3-7.4:1.0/video4linux/video8",
    "/dev/video2": "/sys/devices/pci0000:00/usb3/3-8/3-8:1.0/video4linux/video2",
}
FAKE_TACTILE = {"TAC_B": 20, "TAC_A": 16, "TAC_D": 4, "TAC_C": 8}
FAKE_VIDEO_NAMES = {
    "TAC_B": ["/dev/video20"],
    "TAC_A": ["/dev/video16"],
    "WRIST_L": ["/dev/video19"],
    "TAC_D": ["/dev/video4"],
    "TAC_C": ["/dev/video8"],
    "Some USB webcam": ["/dev/video2"],
}


@pytest.fixture
def fake_topology(monkeypatch):
    """Patch the two enumerations and the sysfs walk with the layout above."""
    monkeypatch.setattr(topo, "hub_of_video_node", lambda node: topo.usb_hub_and_port(FAKE_SYSFS[node]))

    class _Sensor:
        @staticmethod
        def scanSerialNumber():  # noqa: N802 - mirrors the SDK's name
            return dict(FAKE_TACTILE)

    monkeypatch.setitem(__import__("sys").modules, "xensesdk", type("M", (), {"Sensor": _Sensor}))
    monkeypatch.setattr(
        "lerobot.cameras.opencv.camera_opencv._parse_v4l2_devices",
        lambda: dict(FAKE_VIDEO_NAMES),
    )


def test_tactile_grouped_by_hub_and_ordered_by_port(fake_topology):
    """Sensor order must follow USB port, not enumeration order, so a hub's
    sensors read back the same way run to run in logs and errors.

    Ordering no longer *names* a pad — the ``left``/``right`` in
    ``*_tactile_<finger>`` comes from the serial's own parity
    (``camera_injection.tactile_finger``) — but a stable order is still what makes
    two discovery runs comparable when a bench is being diagnosed."""
    by_hub = topo.tactile_sns_by_hub()
    assert set(by_hub) == {"3-1", "3-7"}
    assert [sn for _, sn in by_hub["3-1"]] == ["TAC_A", "TAC_B"]  # 3-1.2 before 3-1.4
    assert [sn for _, sn in by_hub["3-7"]] == ["TAC_D", "TAC_C"]  # 3-7.2 before 3-7.4


def test_video_names_grouped_by_hub(fake_topology):
    by_hub = topo.video_names_by_hub()
    assert [n for _, n in by_hub["3-1"]] == ["TAC_A", "WRIST_L", "TAC_B"]
    assert by_hub["3-8"] == [("/dev/video2", "Some USB webcam")], (
        "a device on a root port has no hub token; it must fall back to its node rather than being grouped with an arm"
    )


# ── The wrist-camera inference ──────────────────────────────────────────────


def _infer_wrist(hub, tactile_by_hub, video_by_hub):
    """The inference under test, mirroring discover_serial_gripper_cameras."""
    tactile = {sn for _, sn in tactile_by_hub.get(hub, [])}
    return [name for _, name in video_by_hub.get(hub, []) if name not in tactile]


def test_wrist_inference_converges_to_one(fake_topology):
    t, v = topo.tactile_sns_by_hub(), topo.video_names_by_hub()
    assert _infer_wrist("3-1", t, v) == ["WRIST_L"]


def test_wrist_inference_detects_missing_and_ambiguous(fake_topology):
    """Both failure shapes must be distinguishable from success, so the caller can
    refuse rather than pick one."""
    t, v = topo.tactile_sns_by_hub(), topo.video_names_by_hub()
    # 3-7 has two tactile sensors and no other video device -> nothing to pick.
    assert _infer_wrist("3-7", t, v) == []
    # An extra camera on the arm's hub -> two candidates, not one.
    v["3-1"].append(("3-1.5", "AN_EXTRA_CAM"))
    assert sorted(_infer_wrist("3-1", t, v)) == ["AN_EXTRA_CAM", "WRIST_L"]


@require_package("xgripper")
def test_discovery_refuses_when_not_exactly_one_candidate(monkeypatch, fake_topology):
    """A wrong-but-plausible camera is worse than a failure: it silently mislabels
    a dataset. Discovery must raise, naming what it found."""
    from lerobot.grippers.serial import discovery as sd

    monkeypatch.setattr(sd, "find_port_by_side", lambda side, **kw: "/dev/ttyUSB0")
    monkeypatch.setattr(sd, "_scan_port_sns", lambda **kw: {"/dev/ttyUSB0": "000031"})
    monkeypatch.setattr(sd, "hub_of_serial_device", lambda dev: "3-7")  # hub with no wrist

    with pytest.raises(RuntimeError, match="exactly one non-tactile video device"):
        sd.discover_serial_gripper_cameras(sides=("left",))


@require_package("xgripper")
def test_discovery_resolves_a_well_formed_hub(monkeypatch, fake_topology):
    from lerobot.grippers.serial import discovery as sd

    monkeypatch.setattr(sd, "find_port_by_side", lambda side, **kw: "/dev/ttyUSB0")
    monkeypatch.setattr(sd, "_scan_port_sns", lambda **kw: {"/dev/ttyUSB0": "000031"})
    monkeypatch.setattr(sd, "hub_of_serial_device", lambda dev: "3-1")

    got = sd.discover_serial_gripper_cameras(sides=("left",))
    assert set(got) == {"left"}
    dev = got["left"]
    assert dev.usb_hub == "3-1"
    assert dev.wrist_camera_name == "WRIST_L"
    assert dev.tactile_sns == ["TAC_A", "TAC_B"]
    assert dev.gripper_sn == "000031"


@require_package("xgripper")
def test_discovery_fails_loudly_when_hub_unresolvable(monkeypatch, fake_topology):
    """A gripper plugged straight into the host has no hub to group by; that is a
    wiring problem the operator must know about, not something to work around."""
    from lerobot.grippers.serial import discovery as sd

    monkeypatch.setattr(sd, "find_port_by_side", lambda side, **kw: "/dev/ttyUSB0")
    monkeypatch.setattr(sd, "_scan_port_sns", lambda **kw: {"/dev/ttyUSB0": "000031"})
    monkeypatch.setattr(sd, "hub_of_serial_device", lambda dev: None)

    with pytest.raises(RuntimeError, match="could not resolve the USB hub"):
        sd.discover_serial_gripper_cameras(sides=("left",))


# ── Config wiring ───────────────────────────────────────────────────────────


ROBOT_TYPES = ["bi_flexiv_rizon4_rt", "bi_elite_cs66_rt"]


@pytest.mark.parametrize("robot_type", ROBOT_TYPES)
def test_both_backends_discover_their_cameras_by_default(robot_type):
    """Both backends are the same shape of device — one USB hub carrying the
    gripper, its wrist camera and its two tactile sensors — so both sniff their
    own cameras and the recipe pins only the head. Pinning SNs by hand was the
    older, worse way; a pinned SN goes stale the moment a sensor is swapped."""
    for cfg_cls, taccap_expected, serial_expected in (
        (SerialGripperConfig, False, True),
        (TaccapFollowerConfig, True, False),
    ):
        cfg = _config(robot_type)(gripper=cfg_cls())
        assert cfg.gripper.auto_discover_cameras is True
        assert cfg._autodiscover_cameras is True
        assert cfg._taccap_autodiscover is taccap_expected
        assert cfg._serial_autodiscover is serial_expected


@pytest.mark.parametrize("robot_type", ROBOT_TYPES)
def test_discovery_can_be_turned_off_per_bench(robot_type):
    """A bench with something else on the hub has to pin its cameras instead —
    discovery refuses to guess through an ambiguous hub."""
    cfg = _config(robot_type)(gripper=SerialGripperConfig(auto_discover_cameras=False))
    assert cfg._autodiscover_cameras is False
    assert cfg._serial_autodiscover is False


@pytest.mark.parametrize("robot_type", ROBOT_TYPES)
def test_no_gripper_means_no_autodiscovery(robot_type):
    """Nothing to sniff cameras off when there is no gripper."""
    cfg = _config(robot_type)(gripper=None)
    assert cfg._autodiscover_cameras is False
    assert cfg.left_gripper is None and cfg.right_gripper is None


@pytest.mark.parametrize("robot_type", ROBOT_TYPES)
def test_shared_block_is_cloned_per_side(robot_type):
    """One block in the recipe, one config per arm, each stamped with its side —
    that stamp is what lets each backend resolve which physical unit is which."""
    cfg = _config(robot_type)(gripper=TaccapFollowerConfig(close_speed_radps=2.5))
    assert cfg.left_gripper.side == "left"
    assert cfg.right_gripper.side == "right"
    assert cfg.left_gripper.close_speed_radps == cfg.right_gripper.close_speed_radps == 2.5
    assert cfg.left_gripper is not cfg.right_gripper


def _config(robot_type: str):
    if robot_type == "bi_flexiv_rizon4_rt":
        if not HAS_FLEXIV:
            pytest.skip("flexiv_rt not importable")
        from lerobot.robots.bi_flexiv_rizon4_rt.config_bi_flexiv_rizon4_rt import (
            BiFlexivRizon4RTConfig,
        )

        return BiFlexivRizon4RTConfig
    if not HAS_ELITE:
        pytest.skip("elite_cs_sdk not importable")
    from lerobot.robots.bi_elite_cs66_rt.config_bi_elite_cs66_rt import BiEliteCS66RTConfig

    return BiEliteCS66RTConfig


@require_package("xgripper")
def test_discovery_sweeps_the_serial_bus_once_for_all_sides(monkeypatch):
    """Resolving both sides must not re-scan the bus per side.

    Each sweep opens and queries every candidate serial port, so it dominates
    discovery time — two sides used to cost four sweeps (one inside
    find_port_by_side plus one to recover the SN it had already read).
    """
    from lerobot.grippers.serial import discovery as sd

    sweeps = 0

    def _counting_scan(**kw):
        nonlocal sweeps
        sweeps += 1
        return {"/dev/ttyUSB0": "000031", "/dev/ttyUSB1": "000032"}

    # Two well-formed hubs, one per side: a wrist camera plus two tactile pads.
    monkeypatch.setattr(sd, "_scan_port_sns", _counting_scan)
    monkeypatch.setattr(sd, "hub_of_serial_device", lambda dev: "3-1" if dev.endswith("0") else "3-2")
    monkeypatch.setattr(
        sd,
        "tactile_sns_by_hub",
        lambda: {
            "3-1": [("3-1.2", "TAC_A"), ("3-1.4", "TAC_B")],
            "3-2": [("3-2.2", "TAC_C"), ("3-2.4", "TAC_D")],
        },
    )
    monkeypatch.setattr(
        sd,
        "video_names_by_hub",
        lambda: {
            "3-1": [("3-1.2", "TAC_A"), ("3-1.3", "WRIST_L"), ("3-1.4", "TAC_B")],
            "3-2": [("3-2.2", "TAC_C"), ("3-2.3", "WRIST_R"), ("3-2.4", "TAC_D")],
        },
    )

    got = sd.discover_serial_gripper_cameras(sides=("left", "right"))

    assert sweeps == 1, f"expected one bus sweep for both sides, got {sweeps}"
    assert set(got) == {"left", "right"}
    # Odd SN -> left, even -> right; each side keeps its own port, hub and cameras.
    assert (got["left"].gripper_sn, got["left"].usb_hub) == ("000031", "3-1")
    assert (got["right"].gripper_sn, got["right"].usb_hub) == ("000032", "3-2")
    assert got["left"].wrist_camera_name == "WRIST_L"
    assert got["right"].wrist_camera_name == "WRIST_R"
    assert got["left"].tactile_sns == ["TAC_A", "TAC_B"]
    assert got["right"].tactile_sns == ["TAC_C", "TAC_D"]


@require_package("xgripper")
def test_scan_skips_ports_belonging_to_another_gripper_family(monkeypatch):
    """A TacCap's serial nodes must not be probed for a board SN.

    They never answer, and each one costs read_board_sn's full retry budget
    (~2.1s) before giving up. On a bench with both gripper families plugged in
    that was 8.4s of every sweep, and startup ran three sweeps.
    """
    from lerobot.grippers.serial import discovery as sd

    probed = []

    def _fake_read_board_sn(port, **kw):
        probed.append(port)
        return "000031" if port.endswith("USB0") else None

    ids = {
        "/dev/ttyACM0": ("1a86", "55d2"),  # TacCap dual-serial
        "/dev/ttyACM1": ("1a86", "55d2"),
        "/dev/ttyUSB0": ("1a86", "7523"),  # gripper board
    }
    monkeypatch.setattr(sd, "read_board_sn", _fake_read_board_sn)
    monkeypatch.setattr(sd, "usb_vid_pid", lambda port: ids.get(port))
    monkeypatch.setattr(sd, "glob", lambda pat: [p for p in ids if p.startswith(pat[:-1])])

    found = sd._scan_port_sns()

    assert probed == ["/dev/ttyUSB0"], f"probed ports that cannot answer: {probed}"
    assert found == {"/dev/ttyUSB0": "000031"}


@require_package("xgripper")
def test_scan_still_probes_an_unrecognised_usb_serial_chip(monkeypatch):
    """The filter excludes known-other devices, it is not a whitelist.

    A gripper board on an unfamiliar bridge chip has to stay discoverable — a
    whitelist would make it silently invisible instead of merely slow.
    """
    from lerobot.grippers.serial import discovery as sd

    probed = []

    def _fake_read_board_sn(port, **kw):
        probed.append(port)
        return "000031"

    monkeypatch.setattr(sd, "read_board_sn", _fake_read_board_sn)
    monkeypatch.setattr(sd, "usb_vid_pid", lambda port: ("dead", "beef"))  # unknown
    monkeypatch.setattr(sd, "glob", lambda pat: ["/dev/ttyUSB9"] if "USB" in pat else [])

    found = sd._scan_port_sns()

    assert probed == ["/dev/ttyUSB9"]
    assert found == {"/dev/ttyUSB9": "000031"}
