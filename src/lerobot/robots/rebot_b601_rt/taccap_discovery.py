"""B601 TacCap discovery for two gripper hubs sharing an upstream USB hub."""

import os
import re

from lerobot.grippers.camera_injection import _add_side_cameras
from lerobot.grippers.taccap.discovery import _import_taccap
from lerobot.grippers.usb_topology import sysfs_device_dir, video_node
from lerobot.utils.errors import DeviceNotConnectedError

_USB_DEVICE_RE = re.compile(r"\d+-\d+(?:\.\d+)*")


def nearest_hub_and_port(sysfs_dir: str) -> tuple[str | None, str | None]:
    devices = [part for part in sysfs_dir.split("/") if _USB_DEVICE_RE.fullmatch(part)]
    if not devices:
        return None, None
    port = devices[-1]
    if "." not in port:
        return port, None
    return port.rsplit(".", 1)[0], port


def inject_taccap_cameras(
    cameras,
    *,
    sides,
    enable_tactile,
    logger,
    undistort_wrist=False,
    fisheye_balance=0.0,
):
    # Reuse the shared SDK loader and camera settings, but keep this grouping
    # local to B601. No monkeypatching of the shared discovery functions.
    sdk = _import_taccap()
    devices = {}
    for side in sides:
        finder = sdk.find_left if side == "left" else sdk.find_right
        try:
            endpoint = finder()
        except Exception as error:
            raise DeviceNotConnectedError(f"taccap auto-discover: no {side} gripper/wrist camera found.") from error
        hub, _ = nearest_hub_and_port(sysfs_device_dir("tty", os.path.realpath(endpoint.mcu_device)))
        if hub is None:
            raise RuntimeError(f"Could not resolve USB hub for {side} gripper {endpoint.firmware_sn}")
        devices[side] = (endpoint, hub)

    by_hub = {}
    if enable_tactile:
        from xensesdk import Sensor

        for serial, camera_id in Sensor.scanSerialNumber().items():
            node = video_node(camera_id)
            hub, port = nearest_hub_and_port(sysfs_device_dir("video4linux", node))
            if hub is not None:
                by_hub.setdefault(hub, []).append((port or node, serial))

    for side, (endpoint, hub) in devices.items():
        product = sdk.parse_serial(endpoint.firmware_sn).product or "TCGU01"
        wrist_name = "XC" + endpoint.firmware_sn[len(product) :]
        tactile_sns = [sn for _, sn in sorted(by_hub.get(hub, []))]
        _add_side_cameras(
            cameras,
            side,
            wrist_name,
            tactile_sns,
            enable_tactile=enable_tactile,
            undistort=undistort_wrist,
            fisheye_balance=fisheye_balance,
            too_few_tactile_msg=(
                f"taccap auto-discover: expected 2 GSPS tactile sensors for {side}, found {tactile_sns}."
            ),
        )
        logger.info(f"[{side}] gripper={endpoint.firmware_sn} hub={hub} wrist={wrist_name} tactile={tactile_sns}")
    logger.info(f"taccap auto-discovered cameras: {sorted(cameras)}")
    return {side: endpoint.mcu_device for side, (endpoint, _) in devices.items()}
