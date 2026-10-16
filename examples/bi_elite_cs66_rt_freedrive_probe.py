#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""SDK-freedrive world-frame probe for ONE Elite CS66 arm.

Puts a single arm into **freedrive** (hand-movable) via the SDK
(``EliteDriver.writeFreedrive``) and continuously prints its TCP pose in BOTH
the raw base frame and the world frame (our current R = Rz(world_yaw)·Rz(zrot)·
Rx(tilt) from BiEliteCS66RTConfig). Hand-push the gripper along the directions
you consider forward / left / up and read which axis actually moves:

  • Δworld tells you whether our CURRENT world frame matches your intent
    (push +forward should give Δworld≈[+,0,0]; +left ≈[0,+,0]; +up ≈[0,0,+]).
  • Δbase is the ground truth: the unit direction of a physical push expressed in
    the arm's base frame. Pushing along your true forward/left/up gives the three
    columns we need to build the CORRECT world←base rotation. Report these and I
    can recompute R exactly (no more angle guessing).

Only the chosen arm is brought up (single-arm driver), so the other arm is NOT
touched. No servo loop runs — this script owns the reverse socket for freedrive.

Safety:
  • Requires REMOTE control mode on the controller (same as any SDK control).
  • The arm goes limp (freedrive) while running — SUPPORT IT / keep clear.
  • On exit (Ctrl-C) freedrive ends and the arm MoveJ's to its start pose.

Usage (Enter = re-zero the Δ baseline; Ctrl-C = quit):
    python examples/bi_elite_cs66_rt_freedrive_probe.py --arm left
"""

import argparse
import select
import sys
import time

import numpy as np

from lerobot.robots.bi_elite_cs66_rt.bi_elite_cs66_rt import BiEliteCS66RT
from lerobot.robots.bi_elite_cs66_rt.config_bi_elite_cs66_rt import BiEliteCS66RTConfig
from lerobot.robots.elite_cs66_rt import EliteCS66RT, EliteCS66RTConfig
from lerobot.robots.elite_cs66_rt.config_elite_cs66_rt import EliteCS66RTControlMode
from lerobot.robots.elite_cs66_rt.elite_cs66_rt import _controller_tcp_to_pose6, _import_elite_sdk
from lerobot.utils.robot_utils import best_effort
from lerobot.utils.rotation import Rotation


def _base_to_world(R_wb: np.ndarray, pose6: np.ndarray) -> np.ndarray:
    pose6 = np.asarray(pose6, dtype=np.float64)
    pos = R_wb @ pose6[:3]
    rot = R_wb @ Rotation.from_rotvec(pose6[3:6]).as_matrix()
    return np.concatenate([pos, Rotation.from_matrix(rot).as_rotvec()])


def _enter_pressed() -> bool:
    """Non-blocking check for an Enter keypress on stdin (Linux terminal)."""
    r, _, _ = select.select([sys.stdin], [], [], 0)
    if r:
        sys.stdin.readline()
        return True
    return False


def _fmt(v: np.ndarray) -> str:
    return "[" + ", ".join(f"{x:+8.4f}" for x in v) + "]"


def main() -> None:
    bicfg = BiEliteCS66RTConfig()  # source of per-arm IPs + mount angles
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=("left", "right"), default="left")
    ap.add_argument("--ip", default=None, help="override controller IP (else from the station file)")
    ap.add_argument("--timeout-ms", type=int, default=200, help="reverse-socket read timeout in the script")
    ap.add_argument("--send-hz", type=float, default=20.0, help="freedrive heartbeat rate")
    ap.add_argument("--print-hz", type=float, default=10.0, help="pose print + CSV log rate")
    ap.add_argument("--log", default=None, help="CSV output path (default: freedrive_<arm>_<ts>.csv in cwd)")
    args = ap.parse_args()

    side = args.arm
    ip = args.ip or getattr(bicfg, f"{side}_robot_ip")
    tilt = getattr(bicfg, f"{side}_mount_tilt_deg")
    zrot = getattr(bicfg, f"{side}_mount_zrot_deg")
    yaw = getattr(bicfg, f"{side}_mount_world_yaw_deg")
    R_wb = BiEliteCS66RT._mount_rotation(tilt, zrot, yaw)

    print(f"Freedrive probe: {side} arm @ {ip}")
    print(f"  current R from (tilt={tilt:g}°, zrot={zrot:g}°, world_yaw={yaw:g}°)")
    print("  world axes expressed in base (columns of R^T):")
    print(f"    world +X in base = {_fmt(R_wb.T @ np.array([1.0, 0, 0]))}")
    print(f"    world +Y in base = {_fmt(R_wb.T @ np.array([0, 1.0, 0]))}")
    print(f"    world +Z in base = {_fmt(R_wb.T @ np.array([0, 0, 1.0]))}")

    # Single-arm driver, no servo loop (we own the reverse socket for freedrive),
    # no gripper / cameras. Return-to-home on disconnect goes to this arm's start.
    cfg = EliteCS66RTConfig(
        robot_ip=ip,
        control_mode=EliteCS66RTControlMode.CARTESIAN_SERVO,
        use_background_servo_loop=False,
    )
    cfg.gripper = None
    cfg.cameras = {}
    cfg.start_position_rad = list(getattr(bicfg, f"{side}_start_position_rad"))
    cfg.home_position_rad = list(getattr(bicfg, f"{side}_home_position_rad"))

    print(
        "\n⚠️  The arm will go LIMP (freedrive). Support it / keep clear. "
        "Requires REMOTE mode. Ctrl-C to stop (arm returns to start)."
    )
    if input("Type 'yes' to proceed: ").strip().lower() != "yes":
        print("Aborted.")
        return
    # libpyelite does not bind writeFreedrive, and connect() always starts its native
    # servo on the reverse socket; refuse before touching the arm rather than halfway.
    if not hasattr(_import_elite_sdk()._native.NativeDriver, "writeFreedrive"):
        print("This libpyelite build has no writeFreedrive; freedrive probing is unavailable.")
        return

    log_path = args.log or f"freedrive_{side}_{int(time.time())}.csv"
    log = open(log_path, "w")  # noqa: SIM115 — held for the whole run, closed in finally
    log.write(
        "seg,t_s,base_x,base_y,base_z,base_rx,base_ry,base_rz,world_x,world_y,world_z,world_rx,world_ry,world_rz\n"
    )
    print(f"Logging to: {log_path}")

    robot = EliteCS66RT(cfg)
    robot.connect(go_to_start=False)
    cs = robot._cs
    driver = robot._driver
    rtsi = robot._rtsi
    assert driver is not None and rtsi is not None

    send_dt = 1.0 / args.send_hz
    print_dt = 1.0 / args.print_hz
    baseline: np.ndarray | None = None
    last_print = 0.0
    started = False
    fails = 0
    seg = 0  # increments each time you press Enter -> one push per segment in the CSV
    t0 = time.monotonic()

    print("\nEntering freedrive — push the gripper along your true FORWARD / LEFT / UP.")
    print("Press Enter BETWEEN pushes (re-zeros Δ AND bumps the CSV segment id).\n")
    try:
        while True:
            action = cs.FreedriveAction.FREEDRIVE_START if not started else cs.FreedriveAction.FREEDRIVE_NOOP
            if not driver.writeFreedrive(action, args.timeout_ms):
                fails += 1
                if fails > 50:
                    raise RuntimeError("writeFreedrive failed repeatedly (lost reverse socket?)")
            else:
                fails = 0
                started = True

            base = _controller_tcp_to_pose6(rtsi.getActualTCPPose())
            world = _base_to_world(R_wb, base)
            if baseline is None:
                baseline = np.concatenate([base[:3], world[:3]])

            if _enter_pressed():
                baseline = np.concatenate([base[:3], world[:3]])
                seg += 1
                print(f">>> baseline re-zeroed — segment {seg} <<<")

            now = time.monotonic()
            if now - last_print >= print_dt:
                last_print = now
                log.write(
                    f"{seg},{now - t0:.3f},"
                    + ",".join(f"{v:.5f}" for v in base)
                    + ","
                    + ",".join(f"{v:.5f}" for v in world)
                    + "\n"
                )
                log.flush()
                db = base[:3] - baseline[:3]
                dw = world[:3] - baseline[3:]
                print(
                    f"[seg {seg}] base {_fmt(base[:3])}  Δbase {_fmt(db)}   |   "
                    f"world {_fmt(world[:3])}  Δworld {_fmt(dw)}"
                )
            time.sleep(send_dt)
    except KeyboardInterrupt:
        print("\nStopping freedrive...")
    finally:
        log.close()
        with best_effort(robot.logger, "leaving freedrive"):
            driver.writeFreedrive(cs.FreedriveAction.FREEDRIVE_END, args.timeout_ms)
            driver.writeIdle(args.timeout_ms)
        robot.disconnect()
    print(f"Done. CSV saved to {log_path}")


if __name__ == "__main__":
    main()
