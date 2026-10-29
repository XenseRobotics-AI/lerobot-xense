"""B601-only nearest-hub grouping alongside the unchanged shared discovery."""

import sys
from logging import getLogger
from types import SimpleNamespace

import pytest

from lerobot.grippers import usb_topology as topo
from lerobot.robots.rebot_b601_rt import taccap_discovery as discovery


def test_shared_hub_grouping_remains_at_the_top_level():
    path = "/sys/devices/usb1/1-12/1-12.2/1-12.2.4/1-12.2.4:1.0/video4linux/video4"
    assert topo.usb_hub_and_port(path) == ("1-12", "1-12.2")
    assert discovery.nearest_hub_and_port(path) == ("1-12.2", "1-12.2.4")
    # Calling B601 discovery must not change later discovery for another robot.
    assert topo.usb_hub_and_port(path) == ("1-12", "1-12.2")


@pytest.mark.parametrize("sides", [("right",), ("left", "right")])
def test_taccap_grippers_on_a_shared_upstream_hub_keep_their_own_sensors(monkeypatch, sides):
    """Two complete TacCaps behind 1-12 must not each claim all four pads."""

    def path(port, suffix):
        return f"/sys/devices/pci0000:00/usb1/1-12/1-12.{port[0]}/1-12.{port}/1-12.{port}:1.0/{suffix}"

    nodes = {
        "ttyACM0": path("3.1", "tty/ttyACM0"),
        "ttyACM2": path("2.1", "tty/ttyACM2"),
        "video2": path("2.2", "video4linux/video2"),
        "video4": path("2.4", "video4linux/video4"),
        "video6": path("3.2", "video4linux/video6"),
        "video8": path("3.4", "video4linux/video8"),
    }
    monkeypatch.setattr(discovery, "sysfs_device_dir", lambda cls, node: nodes[node.rsplit("/", 1)[-1]])
    # Enumeration order is deliberately unrelated to hub or finger order.
    scanned = {"GSPS01A27Z0097": 8, "GSPS01A34Z0021": 4, "GSPS01A27Z0094": 6, "GSPS01A34Z0022": 2}
    monkeypatch.setitem(
        sys.modules,
        "xensesdk",
        SimpleNamespace(
            Sensor=SimpleNamespace(scanSerialNumber=lambda: scanned),
        ),
    )
    sdk = SimpleNamespace(
        find_left=lambda: SimpleNamespace(firmware_sn="TCGU01A25Z0015s", mcu_device="/dev/ttyACM0"),
        find_right=lambda: SimpleNamespace(firmware_sn="TCGU01A28Z0052s", mcu_device="/dev/ttyACM2"),
        parse_serial=lambda sn: SimpleNamespace(product="TCGU01"),
    )
    monkeypatch.setattr(discovery, "_import_taccap", lambda: sdk)

    cameras = {}
    discovery.inject_taccap_cameras(cameras, sides=sides, enable_tactile=True, logger=getLogger(__name__))
    expected = {
        "right_tactile_left": "GSPS01A34Z0021",
        "right_tactile_right": "GSPS01A34Z0022",
    }
    if "left" in sides:
        expected.update(left_tactile_left="GSPS01A27Z0097", left_tactile_right="GSPS01A27Z0094")
    assert {key: cam.serial_number for key, cam in cameras.items() if "tactile" in key} == expected
    assert cameras["right_wrist"].index_or_path == "XCA28Z0052s"
    if "left" in sides:
        assert cameras["left_wrist"].index_or_path == "XCA25Z0015s"
