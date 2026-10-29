#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Read-only world-frame check for the bimanual Elite CS66 station.

Connects to each Elite controller's RTSI server **read-only** (no powerOn, no
EliteDriver, no servoj) and prints the current TCP pose both in the raw (tilted)
base frame and lifted into the per-arm world frame using the EXACT same
transform the deployed driver applies in get_observation:

    R_world<-base = Rz(world_yaw)·Rz(zrot)·Rx(tilt)

with the per-arm angles taken from ``BiEliteCS66RTConfig`` (the ``diagonal``
station file). Use it to verify the inferred left-arm world_yaw=180° on-station:

  • Pose both grippers pointing the SAME physical direction (e.g. straight down,
    or both "forward" toward the workspace). Their printed WORLD approach axis
    (tool +Z) and world RPY should then match between left and right. If left is
    flipped ~180° about the vertical, the world_yaw for the left arm is wrong.
  • The "approach tilt-from-vertical" line is yaw-invariant — it should match
    across arms regardless of any residual global yaw.

This NEVER commands motion. It only opens the RTSI output stream and reads.
Safe to run while the arms are braked/idle.

Usage:
    python examples/bi_elite_cs66_rt_world_frame_check.py
    python examples/bi_elite_cs66_rt_world_frame_check.py --left-ip 192.168.8.53 \
        --right-ip 192.168.8.223 --watch
"""

import argparse
import time
from contextlib import suppress
from pathlib import Path

import numpy as np

from lerobot.robots.bi_elite_cs66_rt.bi_elite_cs66_rt import BiEliteCS66RT
from lerobot.robots.bi_elite_cs66_rt.config_bi_elite_cs66_rt import BiEliteCS66RTConfig
from lerobot.robots.elite_cs66_rt import elite_cs66_rt as _elite_mod
from lerobot.robots.elite_cs66_rt.elite_cs66_rt import _controller_tcp_to_pose6, _import_elite_sdk
from lerobot.utils.rotation import Rotation

_SIDES = ("left", "right")


def _resolve_recipe(cs, filename: str) -> str:
    """Resolve an RTSI recipe: prefer the SDK package, fall back to the elite module resource dir."""
    with suppress(Exception):
        sdk_path = Path(cs.__file__).resolve().parent / filename
        if sdk_path.exists():
            return str(sdk_path)
    module_recipe = Path(_elite_mod.__file__).resolve().parent / "resource" / filename
    if module_recipe.exists():
        return str(module_recipe)
    raise FileNotFoundError(f"Could not find RTSI recipe {filename}")


def _base_pose6_to_world(R_wb: np.ndarray, pose6: np.ndarray) -> np.ndarray:
    """Identical math to BiEliteCS66RT._base_pose6_to_world (pos rotated, orientation left-multiplied)."""
    pose6 = np.asarray(pose6, dtype=np.float64)
    pos = R_wb @ pose6[:3]
    rot = R_wb @ Rotation.from_rotvec(pose6[3:6]).as_matrix()
    return np.concatenate([pos, Rotation.from_matrix(rot).as_rotvec()])


def _read_tcp_pose(rtsi, *, settle_s: float = 2.0) -> np.ndarray:
    """Poll getActualTCPPose() until a non-zero packet arrives (or timeout).

    RTSI reports [x, y, z, roll, pitch, yaw]; the result is converted to rotvec.
    """
    deadline = time.monotonic() + settle_s
    pose = np.zeros(6, dtype=np.float64)
    while time.monotonic() < deadline:
        pose = np.asarray(rtsi.getActualTCPPose(), dtype=np.float64)
        if np.linalg.norm(pose) > 1e-9:
            return _controller_tcp_to_pose6(pose)
        time.sleep(0.05)
    return pose  # all-zero -> stream never populated (recipe/connection issue)


def _approach_tilt_from_vertical_deg(world_pose6: np.ndarray) -> float:
    """Angle (deg) between the tool +Z (approach) axis and world -Z (straight down)."""
    approach = Rotation.from_rotvec(world_pose6[3:6]).as_matrix()[:, 2]
    cos = float(np.clip(np.dot(approach, np.array([0.0, 0.0, -1.0])), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def _fmt(v: np.ndarray) -> str:
    return "[" + ", ".join(f"{x:+8.4f}" for x in v) + "]"


def _print_arm(side: str, cfg: BiEliteCS66RTConfig, R_wb: np.ndarray, base_pose: np.ndarray) -> np.ndarray:
    tilt = getattr(cfg, f"{side}_mount_tilt_deg")
    zrot = getattr(cfg, f"{side}_mount_zrot_deg")
    yaw = getattr(cfg, f"{side}_mount_world_yaw_deg")
    world_pose = _base_pose6_to_world(R_wb, base_pose)
    R_eef_world = Rotation.from_rotvec(world_pose[3:6]).as_matrix()

    print(f"\n===== {side.upper()} arm  (tilt={tilt:g}°, zrot={zrot:g}°, world_yaw={yaw:g}°) =====")
    print(f"  base  pos [m]    = {_fmt(base_pose[:3])}")
    print(f"  base  rotvec     = {_fmt(base_pose[3:6])}")
    print(
        f"  WORLD pos [m]    = {_fmt(world_pose[:3])}   (x={world_pose[0]:+.4f}  y={world_pose[1]:+.4f}  z={world_pose[2]:+.4f})"
    )
    print(f"  WORLD rotvec     = {_fmt(world_pose[3:6])}")
    print("  WORLD tool axes (columns of R_eef, expressed in world):")
    print(f"      tool +X      = {_fmt(R_eef_world[:, 0])}")
    print(f"      tool +Y      = {_fmt(R_eef_world[:, 1])}")
    print(f"      tool +Z(appr)= {_fmt(R_eef_world[:, 2])}")
    print(f"  approach tilt-from-vertical = {_approach_tilt_from_vertical_deg(world_pose):6.2f}°  (yaw-invariant)")
    return world_pose


def _print_cross_arm(left_world: np.ndarray, right_world: np.ndarray) -> None:
    """Compare the two arms' WORLD poses. Only meaningful when both arms are posed
    point-symmetrically (e.g. mirrored across the central plane).

    Expected signature for a correct shared world frame: X and Z read the SAME on
    both arms (X positive, Z negative), Y is OPPOSITE (mirrored across the Y=0 plane).
    """
    lx, ly, lz = left_world[:3]
    rx, ry, rz = right_world[:3]
    # Approach-axis angle between the two arms (≈0° if grippers point the same way).
    al = Rotation.from_rotvec(left_world[3:6]).as_matrix()[:, 2]
    ar = Rotation.from_rotvec(right_world[3:6]).as_matrix()[:, 2]
    appr_angle = np.degrees(np.arccos(float(np.clip(np.dot(al, ar), -1.0, 1.0))))

    print("\n----- CROSS-ARM (world) — meaningful only at symmetric poses -----")
    print(f"  {'axis':4} {'left':>10} {'right':>10}   {'check':<22} result")
    print(f"  {'X':4} {lx:>+10.4f} {rx:>+10.4f}   X_left - X_right = {lx - rx:+.4f}  (expect ~0, same +)")
    print(f"  {'Y':4} {ly:>+10.4f} {ry:>+10.4f}   Y_left + Y_right = {ly + ry:+.4f}  (expect ~0, opposite)")
    print(f"  {'Z':4} {lz:>+10.4f} {rz:>+10.4f}   Z_left - Z_right = {lz - rz:+.4f}  (expect ~0, same -)")
    print(f"  approach-axis angle between arms = {appr_angle:6.2f}°  (expect ~0° if both grippers point the same way)")


def main() -> None:
    cfg = BiEliteCS66RTConfig()  # loads the default station: IPs + mount angles
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--left-ip", default=cfg.left_robot_ip)
    ap.add_argument("--right-ip", default=cfg.right_robot_ip)
    ap.add_argument("--freq", type=float, default=cfg.rtsi_frequency)
    ap.add_argument("--watch", action="store_true", help="loop forever (re-read every --interval s)")
    ap.add_argument("--interval", type=float, default=1.0)
    args = ap.parse_args()

    ips = {"left": args.left_ip, "right": args.right_ip}
    R_wb = {
        side: BiEliteCS66RT._mount_rotation(
            getattr(cfg, f"{side}_mount_tilt_deg"),
            getattr(cfg, f"{side}_mount_zrot_deg"),
            getattr(cfg, f"{side}_mount_world_yaw_deg"),
        )
        for side in _SIDES
    }

    cs = _import_elite_sdk()
    out_recipe = _resolve_recipe(cs, "output_recipe.txt")
    in_recipe = _resolve_recipe(cs, "input_recipe.txt")

    rtsi: dict[str, object] = {}
    try:
        for side in _SIDES:
            print(f"Connecting RTSI (read-only) {side} @ {ips[side]}:30004 ...")
            iface = cs.RtsiIOInterface(out_recipe, in_recipe, args.freq)
            if not iface.connect(ips[side]):
                raise ConnectionError(f"Failed to connect Elite RTSI ({side}) at {ips[side]}:30004")
            rtsi[side] = iface

        while True:
            print("\n" + "=" * 72)
            print(time.strftime("%Y-%m-%d %H:%M:%S"))
            world: dict[str, np.ndarray] = {}
            for side in _SIDES:
                base_pose = _read_tcp_pose(rtsi[side])
                if np.linalg.norm(base_pose) < 1e-9:
                    print(
                        f"\n[{side}] WARNING: RTSI returned an all-zero TCP pose "
                        "(stream not populated / recipe mismatch)."
                    )
                    continue
                world[side] = _print_arm(side, cfg, R_wb[side], base_pose)
            if "left" in world and "right" in world:
                _print_cross_arm(world["left"], world["right"])
            if not args.watch:
                break
            time.sleep(args.interval)
    finally:
        for iface in rtsi.values():
            with suppress(Exception):
                iface.disconnect()


if __name__ == "__main__":
    main()
