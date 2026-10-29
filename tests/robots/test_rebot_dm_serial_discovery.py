"""USB order selection without serial IO or native controller dependencies."""

import subprocess
from pathlib import Path

import pytest

from lerobot.robots.rebot_b601_rt import dm_serial_discovery as discovery


def properties(timestamp="100", vendor="2e88", product="4603", interface="00"):
    return {
        "ID_VENDOR_ID": vendor,
        "ID_MODEL_ID": product,
        "ID_USB_INTERFACE_NUM": interface,
        "ID_SERIAL_SHORT": "00000000050C",  # Both arm bridges report the same serial.
        "USEC_INITIALIZED": timestamp,
    }


def mock_udev(monkeypatch, nodes):
    monkeypatch.setattr(Path, "glob", lambda self, pattern: [Path(name) for name in nodes])
    monkeypatch.setattr(discovery, "_udev_properties", lambda channel: nodes[Path(channel).name])


def test_order_uses_timestamps_despite_reversed_acm_numbers_and_other_serial_devices(monkeypatch):
    mock_udev(
        monkeypatch,
        {
            "ttyACM0": properties("200"),  # Left reused a lower ACM number.
            "ttyACM1": properties("1", "1a86", "55d4"),  # TacCap USB Dual Serial.
            "ttyACM2": properties("2", "1a86", "55d4", "02"),
            "ttyACM3": properties("3", interface="02"),
            "ttyACM12": properties("100"),  # Right connected first.
        },
    )
    devices = discovery.discover_dm_serial_devices()
    assert discovery.resolve_dm_serial_channel("auto:first", devices) == "/dev/ttyACM12"
    assert discovery.resolve_dm_serial_channel("auto:second", devices) == "/dev/ttyACM0"
    assert discovery.resolve_dm_serial_channel("can0", devices) == "can0"
    assert discovery.resolve_dm_serial_channel("/dev/explicit", devices) == "/dev/explicit"


def test_single_bridge_can_be_selected_first_but_not_second(monkeypatch):
    mock_udev(monkeypatch, {"ttyACM7": properties()})
    devices = discovery.discover_dm_serial_devices()
    assert discovery.resolve_dm_serial_channel("auto:first", devices) == "/dev/ttyACM7"
    with pytest.raises(RuntimeError, match="needs two DM bridges"):
        discovery.resolve_dm_serial_channel("auto:second", devices)


@pytest.mark.parametrize("count", [0, 3])
def test_unexpected_bridge_count_is_not_guessed(monkeypatch, count):
    mock_udev(monkeypatch, {f"ttyACM{i}": properties(str(i + 100)) for i in range(count)})
    with pytest.raises(RuntimeError, match=f"found {count}"):
        discovery.discover_dm_serial_devices()


@pytest.mark.parametrize("timestamp", [None, "", "bad", "0", "-1"])
def test_missing_or_invalid_order_metadata_is_not_replaced_with_acm_order(monkeypatch, timestamp):
    metadata = properties(timestamp)
    if timestamp is None:
        del metadata["USEC_INITIALIZED"]
    mock_udev(monkeypatch, {"ttyACM0": metadata, "ttyACM1": properties("200")})
    with pytest.raises(RuntimeError, match="initialization time"):
        discovery.discover_dm_serial_devices()


def test_tied_timestamps_require_reconnection(monkeypatch):
    mock_udev(monkeypatch, {"ttyACM0": properties(), "ttyACM1": properties()})
    with pytest.raises(RuntimeError, match="ambiguous"):
        discovery.discover_dm_serial_devices()


def test_reenumerated_or_replaced_device_is_rejected_before_open(monkeypatch):
    nodes = {"ttyACM0": properties()}
    mock_udev(monkeypatch, nodes)
    devices = discovery.discover_dm_serial_devices()
    discovery.verify_dm_serial_device("/dev/ttyACM0", devices)
    for replacement in (properties("300"), properties(vendor="1a86")):
        nodes["ttyACM0"] = replacement
        with pytest.raises(RuntimeError, match="changed during startup"):
            discovery.verify_dm_serial_device("/dev/ttyACM0", devices)


@pytest.mark.parametrize(
    "error",
    [FileNotFoundError(), subprocess.CalledProcessError(1, "udevadm"), subprocess.TimeoutExpired("udevadm", 3)],
)
def test_unavailable_udev_metadata_is_reported(monkeypatch, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(discovery.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="Cannot read udev metadata"):
        discovery._udev_properties("/dev/ttyACM0")
