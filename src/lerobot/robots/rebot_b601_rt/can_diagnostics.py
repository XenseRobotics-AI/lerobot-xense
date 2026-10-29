"""Actionable terminal hints for native B601 SocketCAN errors."""

import shlex


def can_error_hint(channel: str, error: str) -> str | None:
    # DM serial bridges do not use a SocketCAN interface.
    if channel.startswith("/"):
        return None
    message = error.lower()
    interface_missing = "unknown can interface" in message
    interface_down = "can" in message and "network is down" in message
    if interface_missing or interface_down:
        interface = shlex.quote(channel)
        status = "was not found" if interface_missing else "is DOWN"
        return (
            f"[CAN troubleshooting] Interface {channel} {status}. "
            "Exit the control program, then run these commands in a terminal:\n"
            "# The kit uses PCAN-USB; the interface normally appears as can0 or can1\n"
            "sudo modprobe peak_usb\n"
            "ip -br link\n\n"
            f"# Once {channel} appears, configure its bitrate\n"
            "# If the interface name differs, adjust these commands and the robot's channel configuration\n"
            f"sudo ip link set {interface} down 2>/dev/null\n"
            f"sudo ip link set {interface} type can bitrate 1000000\n"
            f"sudo ip link set {interface} up"
        )

    queue_blocked = "can transmit queue" in message or (
        "can write" in message and "no buffer space available" in message
    )
    feedback_missing = any(
        text in message
        for text in (
            "missing fresh feedback after fault clear",
            "missing initial feedback from motor",
            "stale initial feedback from motor",
            "stale feedback from motor",
        )
    )
    if queue_blocked or feedback_missing:
        symptom = "CAN transmit queue is blocked" if queue_blocked else "no fresh motor feedback received"
        return (
            f"[CAN troubleshooting] {channel}: {symptom}. "
            "Possible causes include an arm power issue or loose connections.\n"
            "Exit the control program, then check:\n"
            "1. The arm's 24V power supply is on and the emergency stop is released. "
            "USB/gripper detection does not confirm that the joint motors have power.\n"
            "2. The arm's power plugs, CAN plugs and adapter cables are firmly connected, "
            "with no loose or disconnected cables.\n"
            f"3. The arm's CAN cable is connected to the configured interface ({channel}), "
            "the interface is UP, and its bitrate is 1000000.\n"
            f"   ip -details -statistics link show dev {shlex.quote(channel)}\n"
            "If power and connections are correct but feedback is still missing, "
            "unplug and reconnect the USB-CAN adapter, then verify the interface name and configure CAN again."
        )
    return None
