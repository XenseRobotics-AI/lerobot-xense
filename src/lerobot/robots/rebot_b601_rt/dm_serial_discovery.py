"""Resolve HDSC DM bridges by udev initialization order, without opening serial ports."""

import subprocess
from dataclasses import dataclass
from pathlib import Path

AUTO_DM_CHANNELS = ("auto:first", "auto:second")
_USB_ID = ("2e88", "4603")
_ORDER_HINT = "Reconnect both DM bridges in order: right first, wait for enumeration, then left."


@dataclass(frozen=True)
class DMSerialDevice:
    channel: str
    initialized_usec: int


def _udev_properties(channel: str) -> dict[str, str]:
    try:
        result = subprocess.run(
            ["udevadm", "info", "--query=property", f"--name={channel}"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"Cannot read udev metadata for {channel}; check udev and USB connections.") from exc
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def _dm_device(channel: str, properties: dict[str, str]) -> DMSerialDevice | None:
    if (properties.get("ID_VENDOR_ID"), properties.get("ID_MODEL_ID")) != _USB_ID:
        return None
    if properties.get("ID_USB_INTERFACE_NUM") != "00":
        return None
    try:
        initialized_usec = int(properties["USEC_INITIALIZED"])
        if initialized_usec <= 0:
            raise ValueError("nonpositive timestamp")
    except (KeyError, ValueError) as exc:
        raise RuntimeError(f"Missing/invalid udev initialization time for {channel}. {_ORDER_HINT}") from exc
    return DMSerialDevice(channel, initialized_usec)


def discover_dm_serial_devices() -> tuple[DMSerialDevice, ...]:
    """Return one or two DM bridges, oldest first; ignore grippers and other ACM devices.

    This is an operator convention, not persistent arm identity. Booting with both
    bridges attached, or reconnecting just one, may change their initialization order.
    """
    devices = []
    for node in Path("/sys/class/tty").glob("ttyACM*"):
        channel = f"/dev/{node.name}"
        device = _dm_device(channel, _udev_properties(channel))
        if device is not None:
            devices.append(device)
    if not 1 <= len(devices) <= 2:
        raise RuntimeError(
            f"DM auto selection requires one or two HDSC bridges (USB 2e88:4603); found {len(devices)}. "
            f"Use explicit channel paths for other hardware or more arms. {_ORDER_HINT}"
        )
    devices.sort(key=lambda device: device.initialized_usec)
    if len(devices) == 2 and devices[0].initialized_usec == devices[1].initialized_usec:
        raise RuntimeError(f"DM bridge initialization order is ambiguous. {_ORDER_HINT}")
    return tuple(devices)


def resolve_dm_serial_channel(channel: str, devices: tuple[DMSerialDevice, ...]) -> str:
    if channel not in AUTO_DM_CHANNELS:
        return channel
    index = AUTO_DM_CHANNELS.index(channel)
    if index >= len(devices):
        raise RuntimeError(f"{channel} needs two DM bridges; found {len(devices)}. {_ORDER_HINT}")
    return devices[index].channel


def verify_dm_serial_device(channel: str, devices: tuple[DMSerialDevice, ...]) -> None:
    """Reject a disappeared/re-enumerated port before the controller opens it."""
    expected = next(device for device in devices if device.channel == channel)
    if _dm_device(channel, _udev_properties(channel)) != expected:
        raise RuntimeError(f"DM bridge {channel} changed during startup. {_ORDER_HINT}")
