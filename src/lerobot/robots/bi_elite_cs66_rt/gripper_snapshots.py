"""Measured TacCap finger state for Elite station collision checking only.

The reader is created after gripper connection, only when collision checking is
enabled. Calibration is queried once at startup; publication only reads the
SDK's existing control-loop cache. Shared gripper drivers have no snapshot API.
"""

import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lerobot.grippers.taccap.taccap_follower import TaccapFollower


@dataclass(frozen=True)
class GripperSnapshot:
    """One actual sample; retain overtravel and the monotonic sensor timestamp."""

    position: float = float("nan")
    sampled_at: float = 0.0
    sequence: int = 0
    valid: bool = False
    raw_position: float | None = None


class TaccapGripperSnapshotReader:
    """Adapt one connected TacCap session to the native station's sample format.

    Access to the backend's SDK handles stays here, inside Elite. A reader is
    tied to one connection: disconnect/reconnect cannot reuse its calibration.
    """

    def __init__(self, gripper: "TaccapFollower"):
        if gripper.config.type != "taccap_follower":
            raise ValueError("Elite collision snapshots support TacCap followers only")
        if not gripper.is_connected or gripper._gripper is None or gripper._loop is None:
            raise RuntimeError("Connect the TacCap follower before starting Elite collision snapshots")
        self._gripper = gripper
        self._hardware = gripper._gripper
        self._loop = gripper._loop
        try:
            cfg = self._hardware.get_gripper_config()
            low, high = float(cfg.min_open_rad), float(cfg.max_open_rad)
            # TacCap GripperConfig flags: bit 0 = calibrated, bit 1 = reverse.
            if not cfg.flags & 0x0001 or not math.isfinite(low) or not math.isfinite(high) or high <= low:
                raise ValueError("missing or invalid motor opening calibration")
            self._calibration = (low, high, -1.0 if cfg.flags & 0x0002 else 1.0)
        except Exception as exc:
            raise RuntimeError(f"Cannot read calibrated TacCap opening for Elite collision checking: {exc}") from exc

    def read(self) -> GripperSnapshot:
        """Read cached sensor data without USB IO or refreshing an old sample.

        SDK age_ms is recomputed from the steady-clock receive timestamp even
        while its controller is stalled. Timestamp before the FFI call so any
        contention can only make the sample look older.
        """
        if (
            not self._gripper.is_connected
            or self._gripper._gripper is not self._hardware
            or self._gripper._loop is not self._loop
        ):
            return GripperSnapshot()
        try:
            before = time.monotonic()
            snapshot = self._loop.snapshot()
            obs = snapshot.observation
            low, high, direction = self._calibration
            raw, age = float(obs.raw_pos), float(obs.age_ms) / 1000.0
            position = (direction * raw - low) / (high - low)
            valid = (
                bool(snapshot.running)
                and bool(obs.valid)
                and int(obs.seq) > 0
                and math.isfinite(raw)
                and math.isfinite(position)
                and math.isfinite(age)
                and age >= 0
            )
            return GripperSnapshot(position, before - age, int(obs.seq), valid, raw)
        except Exception:
            return GripperSnapshot()
