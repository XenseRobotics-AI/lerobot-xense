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

"""Auto-discovery of the per-side devices that ship with a TacCap follower gripper.

A TacCap gripper is a self-contained USB hub carrying the gripper MCU (a serial
device), one wrist UVC camera, and two GSPS visuotactile sensors. This module
sniffs the two grippers via the TacCap SDK and resolves, for each side (left /
right), the identifiers the existing LeRobot camera stack needs:

  - wrist camera : a V4L2 device *name* (e.g. "XCA25Z0011s"), derived from the
                   gripper firmware SN (product prefix "TCGU01" → "XC"). Consumed
                   by OpenCVCameraConfig, which resolves names via v4l2-ctl.
  - tactile      : the two GSPS sensor serial numbers (e.g. "GSPS01A27Z0094"),
                   enumerated by xensesdk and matched to the *side* by USB
                   topology (same USB hub as that side's gripper MCU). Which
                   *finger* of that gripper each one sits on is a separate
                   question, answered by the serial's own trailing-digit parity
                   when the keys are built — see ``camera_injection``.

Side comes from the firmware-burned gripper SN via the TacCap SDK, never guessed.
"""

from dataclasses import dataclass, field

from lerobot.utils.robot_utils import get_logger

from ..usb_topology import hub_of_serial_device, tactile_sns_by_hub

logger = get_logger("TaccapDiscovery")

_WRIST_NAME_PREFIX = "XC"  # wrist V4L2 name = "XC" + gripper SN minus product code


@dataclass
class TaccapSideDevices:
    side: str  # "left" | "right"
    firmware_sn: str  # gripper SN, e.g. "TCGU01A25Z0011s"
    mcu_device: str  # e.g. "/dev/serial/by-id/usb-1a86_...-if02"
    usb_hub: str  # e.g. "1-3"
    wrist_camera_name: str  # e.g. "XCA25Z0011s" (V4L2 device name)
    tactile_sns: list[str] = field(default_factory=list)  # ordered GSPS serials


def _import_taccap():
    """Import the TacCap SDK, guarded with an actionable error."""
    try:
        import xense.taccap as taccap  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "xense.taccap is not importable — the TacCap SDK native extension is "
            "missing or stale. Rebuild it with `bash setup_env.sh --install` (or "
            f"`pip install third_party/taccap-gripper`). Original error: {e!r}"
        ) from e
    return taccap


def find_follower_on_side(taccap, side: str):
    """The follower gripper on ``side`` -- never a leader of the same side.

    The SDK's ``find_left()`` / ``find_right()`` without a role return the first
    unit on that side whatever its role. A leader plugged into the same host
    (teleop) is then picked whenever its CH343 serial sorts first; on the bench a
    left leader next to the left follower was avoided only by serial order.
    ``role=`` arrived after SDK 0.5.0; an older extension falls back to the
    role-less call and refuses a leader it returns, rather than open it as a
    follower.
    """
    finder = taccap.find_left if side == "left" else taccap.find_right
    try:
        return finder(role=taccap.Role.Follower)
    except (TypeError, AttributeError):
        pass  # native extension older than the role filter
    logger.warn(
        f"The installed xense.taccap has no role filter on find_{side}(); a leader on the same host "
        "could be picked. Reinstall third_party/taccap-gripper in this environment."
    )
    eps = finder()
    if str(getattr(eps, "firmware_sn", "")).lower().endswith("m"):
        raise RuntimeError(
            f"find_{side}() returned leader {eps.firmware_sn} where a follower was expected. "
            "Reinstall third_party/taccap-gripper (its find_left/find_right take role=), or unplug the leader."
        )
    return eps


def discover_taccap_sides() -> dict[str, TaccapSideDevices]:
    """Discover per-side (left/right) TacCap wrist camera + GSPS tactile SNs.

    Returns:
        {"left": TaccapSideDevices, "right": TaccapSideDevices} — only the sides
        that are actually connected are included.

    Raises:
        RuntimeError: If the SDKs are unavailable, no gripper is found, or a side's
            device topology can't be resolved.
    """
    taccap = _import_taccap()

    # --- 1. Sniff grippers per side, derive wrist name, resolve USB hub ---
    sides: dict[str, TaccapSideDevices] = {}
    for side in ("left", "right"):
        try:
            eps = find_follower_on_side(taccap, side)
        except Exception as e:
            logger.info(f"No {side} TacCap gripper found: {e}")
            continue
        parsed = taccap.parse_serial(eps.firmware_sn)
        product = parsed.product or "TCGU01"
        wrist_name = _WRIST_NAME_PREFIX + eps.firmware_sn[len(product) :]
        hub = hub_of_serial_device(eps.mcu_device)
        if hub is None:
            raise RuntimeError(
                f"Could not resolve USB hub for {side} gripper {eps.firmware_sn} "
                f"({eps.mcu_device}); cannot map its tactile sensors."
            )
        sides[side] = TaccapSideDevices(
            side=side,
            firmware_sn=eps.firmware_sn,
            mcu_device=eps.mcu_device,
            usb_hub=hub,
            wrist_camera_name=wrist_name,
        )

    if not sides:
        raise RuntimeError(
            "No TacCap grippers found via scan_grippers()/find_left()/find_right(). "
            "Check USB connection, dialout permissions, and ModemManager not grabbing "
            "the CH343 serial ports."
        )

    # --- 2. Enumerate GSPS tactile sensors, map to side by USB hub ---
    by_hub = tactile_sns_by_hub()  # {hub: [(usb_port, serial), ...]}, port-ordered

    for side, dev in sides.items():
        entries = by_hub.get(dev.usb_hub, [])  # by USB port ascending
        dev.tactile_sns = [sn for _, sn in entries]
        logger.info(
            f"[{side}] gripper={dev.firmware_sn} hub={dev.usb_hub} "
            f"wrist={dev.wrist_camera_name} tactile={dev.tactile_sns}"
        )

    return sides
