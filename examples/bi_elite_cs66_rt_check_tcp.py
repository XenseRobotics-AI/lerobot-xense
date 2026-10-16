#!/usr/bin/env python3
"""Read the TCP offset an Elite CS66 controller is applying, as a tool_transform.

`RtsiIOInterface.getActualTCPPose()` returns the pose of the *active TCP*
(flange composed with the pendant's TCP offset), while the libpyelite model FK of
the same joints ends at the flange. The offset between the two is exactly what the
native planner needs as ``tool_transform``; the driver derives it the same way at
connect when a recipe leaves ``tool_transform`` unset.

Run this with the arm IDLE (no teleop holding the RTSI input registers). The
printed row-major matrix can be pasted into a recipe to pin the TCP explicitly.

Usage:
    python examples/bi_elite_cs66_rt_check_tcp.py --ip 192.168.8.53
    python examples/bi_elite_cs66_rt_check_tcp.py --ip 192.168.8.53 --assembly diagonal-08 --side left
"""

import argparse
import sys
from pathlib import Path

import libpyelite
import numpy as np

from lerobot.robots.elite_cs66_rt.elite_cs66_rt import _derive_controller_tool
from lerobot.utils.rotation import Rotation

RESOURCE_DIR = Path(__file__).resolve().parents[1] / "src/lerobot/robots/elite_cs66_rt/resource"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", required=True)
    ap.add_argument("--rtsi-frequency", type=float, default=250.0)
    ap.add_argument("--assembly", help="native_assembly name; selects that station's control model")
    ap.add_argument("--side", choices=("left", "right"), help="arm within --assembly")
    args = ap.parse_args()
    if bool(args.assembly) != bool(args.side):
        ap.error("--assembly and --side go together")

    if args.assembly:
        from libpyelite.assembly import Assembly

        model = Assembly(args.assembly).control_model_path(args.side)
    else:
        model = libpyelite.model_path()

    io = libpyelite.RtsiIOInterface(
        str(RESOURCE_DIR / "output_recipe.txt"), str(RESOURCE_DIR / "input_recipe.txt"), args.rtsi_frequency
    )
    if not io.connect(args.ip):
        print(f"[ERROR] rtsi connect failed at {args.ip}:30004", file=sys.stderr)
        return 1
    try:
        tool = np.asarray(_derive_controller_tool(libpyelite, io, model), dtype=np.float64).reshape(4, 4)
    finally:
        io.disconnect()

    offset_mm = tool[:3, 3] * 1000.0
    rot_deg = np.degrees(np.linalg.norm(Rotation.from_matrix(tool[:3, :3]).as_rotvec()))
    print("\n================= TCP CHECK =================")
    print(f"  model          : {model}")
    print(f"  offset (mm)    : {np.array2string(offset_mm, precision=2)}  |offset| = {np.linalg.norm(offset_mm):.2f}")
    print(f"  rotation (deg) : {rot_deg:.2f}")
    print("  tool_transform :")
    for row in tool:
        print("    " + ", ".join(f"{v:.5f}" for v in row) + ",")
    print("============================================")
    if np.linalg.norm(offset_mm) < 1.0:
        print(">>> No tool offset active: the pendant TCP is the flange (within 1 mm).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
