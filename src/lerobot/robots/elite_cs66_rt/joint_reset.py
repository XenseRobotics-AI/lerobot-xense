"""Feedback-verified joint resets shared only by the Elite adapters.

The worker supplies low-rate joint references; native senders and the station
collision guard retain ownership of real-time planning and socket writes.
"""

import math
import threading
import time
from collections.abc import Callable

import numpy as np


def validate_reset_config(config) -> None:
    for name in (
        "reset_duration_s",
        "reset_timeout_s",
        "reset_joint_tolerance_rad",
        "reset_velocity_tolerance_rad_s",
        "reset_stable_dwell_s",
        "reset_feedback_timeout_s",
        "native_configuration_joint_span",
        "native_boundary_orientation_tolerance",
    ):
        value = getattr(config, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if config.reset_timeout_s <= config.reset_duration_s + config.reset_stable_dwell_s:
        raise ValueError("reset_timeout_s must exceed reset_duration_s + reset_stable_dwell_s")
    if type(config.native_preserve_configuration) is not bool:
        raise ValueError("native_preserve_configuration must be a bool")
    if not 0 < config.native_configuration_margin < math.pi / 2:
        raise ValueError("native_configuration_margin must be in (0, pi/2)")
    if config.native_configuration_joint_span >= math.pi:
        raise ValueError("native_configuration_joint_span must be less than pi")
    if config.native_preserve_configuration and not config.native_joint_motion_limits:
        raise ValueError("native_preserve_configuration requires native_joint_motion_limits")
    if config.native_joint_motion_limits:
        for name in ("native_max_joint_velocity", "native_max_joint_acceleration", "native_max_joint_jerk"):
            values = getattr(config, name)
            if values is None or len(values) != 6 or any(not math.isfinite(v) or v <= 0 for v in values):
                raise ValueError(f"{name} must contain six finite positive limits")


def joint_sample(rtsi) -> tuple[float, np.ndarray, np.ndarray]:
    """Read one coherent, valid sample; never wrap angles or substitute zeros."""
    for _ in range(5):
        stamp = rtsi.getTimestamp()
        q = np.asarray(rtsi.getActualJointPositions(), dtype=np.float64)
        velocity = np.asarray(rtsi.getActualJointVelocity(), dtype=np.float64)
        if stamp != rtsi.getTimestamp():
            continue
        if (
            not math.isfinite(stamp)
            or stamp <= 0
            or q.shape != (6,)
            or velocity.shape != (6,)
            or not np.isfinite(q).all()
            or not np.isfinite(velocity).all()
        ):
            raise RuntimeError("Invalid Elite joint feedback")
        return stamp, q.copy(), velocity.copy()
    raise RuntimeError("Could not obtain coherent Elite joint feedback")


class JointReset:
    """One reset transaction for one or both arms, including command exclusion.

    Callers hold ``lock`` across normal sends and lifecycle transitions. The
    worker never holds it while checking paths, waiting, or joining a thread.
    Failures stay visible until a new connection or an explicit reset retry.
    """

    def __init__(self, config):
        self.config = config
        self.lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self.active = False
        self.closing = False
        self.state = "idle"
        self.error: Exception | None = None
        self._completion_reads_remaining = 0

    def prepare_connection(self) -> None:
        self.close()
        with self.lock:
            self.closing = False
            self.state = "idle"
            self.error = None
            self._completion_reads_remaining = 0

    def check_error(self) -> None:
        if self.error is not None:
            raise RuntimeError(f"Elite joint reset failed: {self.error}") from self.error

    def moving(self) -> bool:
        with self.lock:
            self.check_error()
            if self.active:
                return True
            if self._completion_reads_remaining:
                # Recording reads twice: hasattr(robot, "rt_moving") and
                # robot.rt_moving. Keep the completion edge through both reads,
                # including when the worker finishes between them.
                self._completion_reads_remaining -= 1
                return True
            return False

    def status(self) -> dict:
        with self.lock:
            return {"state": self.state, "active": self.active, "error": str(self.error) if self.error else ""}

    def close(self) -> None:
        with self.lock:
            self.closing = True
            self._cancel.set()
            worker = self._thread
        if worker is not None and worker is not threading.current_thread():
            worker.join()

    def start(self, drivers, feedback, targets, preflight: Callable, clear_targets: Callable) -> None:
        with self.lock:
            if self.closing:
                raise RuntimeError("Elite adapter is disconnecting")
            if self.active:
                return
            goals = {}
            for side, target in targets.items():
                if target is None:
                    raise RuntimeError("No measured startup joints saved; reconnect before reset")
                q = np.asarray(target, dtype=np.float64)
                if q.shape != (6,) or not np.isfinite(q).all():
                    raise RuntimeError(f"Invalid saved startup joints ({side})")
                goals[side] = q.copy()
            self.error = None
            self._cancel.clear()
            self.active = True
            self._completion_reads_remaining = 0
            self.state = "settling"
            clear_targets()
            self._thread = threading.Thread(
                target=self._run,
                args=(dict(drivers), dict(feedback), goals, preflight, clear_targets),
                name="elite-joint-reset",
                daemon=True,
            )
            try:
                self._thread.start()
            except Exception as exc:
                self._thread = None
                self.active = False
                self.state, self.error = "failed", exc
                raise
            worker = self._thread
        if not self.config.use_background_servo_loop:
            worker.join()
            self.check_error()

    def _run(self, drivers, feedback, goals, preflight, clear_targets) -> None:
        cfg = self.config
        begun = time.monotonic()
        deadline = begun + cfg.reset_timeout_s
        previous = dict.fromkeys(goals)
        changed_at = dict.fromkeys(goals, begun)
        stable_since = None
        origin = None
        motion_start = None
        endpoint_sent = False
        outcome = "cancelled"
        error = None
        try:
            # Cancel both stale Cartesian/joint targets before measuring the
            # stopped origin used for station preflight and interpolation.
            with self.lock:
                for driver in drivers.values():
                    driver.cancel()
            while not self._cancel.is_set():
                now = time.monotonic()
                if now >= deadline:
                    raise TimeoutError("Joint reset timed out before measured angle/velocity settled")
                samples = {}
                fresh = True
                settled = True
                for side, driver in drivers.items():
                    driver.check_error()
                    status = driver.status()
                    if status.error or not status.running:
                        raise RuntimeError(f"{side} native servo stopped during reset: {status.error}")
                    if not driver.isRobotConnected() or not feedback[side].isConnected():
                        raise ConnectionError(f"{side} disconnected during joint reset")
                    stamp, q, velocity = joint_sample(feedback[side])
                    if previous[side] is not None and stamp < previous[side]:
                        raise RuntimeError(f"{side} RTSI timestamp moved backwards during reset")
                    if stamp != previous[side]:
                        previous[side] = stamp
                        changed_at[side] = now
                    else:
                        fresh = False
                    if now - changed_at[side] >= cfg.reset_feedback_timeout_s:
                        raise TimeoutError(f"{side} RTSI feedback stale during joint reset")
                    samples[side] = q
                    settled &= bool(np.max(np.abs(velocity)) <= cfg.reset_velocity_tolerance_rad_s)
                    if origin is not None:
                        # Direct, unwrapped error: q and q + 2*pi are different
                        # physical return configurations even if TCPs coincide.
                        settled &= bool(np.max(np.abs(q - goals[side])) <= cfg.reset_joint_tolerance_rad)
                if not fresh or not settled:
                    stable_since = None
                elif stable_since is None:
                    stable_since = now
                if stable_since is not None and now - stable_since >= cfg.reset_stable_dwell_s:
                    if origin is None:
                        preflight({side: q.tolist() for side, q in goals.items()})
                        if self._cancel.is_set():
                            break
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Joint reset timed out during station preflight")
                        origin = samples
                        motion_start = time.monotonic()
                        stable_since = None
                        with self.lock:
                            self.state = "moving"
                    elif endpoint_sent:
                        outcome = "succeeded"
                        break
                if origin is not None:
                    alpha = min(1.0, max(0.0, (time.monotonic() - motion_start) / cfg.reset_duration_s))
                    blend = alpha**3 * (10.0 - 15.0 * alpha + 6.0 * alpha**2)
                    with self.lock:
                        if self._cancel.is_set():
                            break
                        for side, driver in drivers.items():
                            # Use an exact copy at the endpoint (no floating
                            # interpolation residue), refreshed until measured
                            # dwell completes. Native velocity/accel/jerk and
                            # station collision constraints still apply.
                            target = (
                                goals[side] if alpha == 1.0 else origin[side] + blend * (goals[side] - origin[side])
                            )
                            driver.submit_joints(target.tolist())
                        endpoint_sent = alpha == 1.0
                self._cancel.wait(min(0.01, cfg.command_stale_timeout_s / 4, cfg.reset_feedback_timeout_s / 4))
        except Exception as exc:
            outcome, error = "failed", exc
        finally:
            with self.lock:
                # Stop target refresh on success, cancellation and failure.
                # No previous teleop target may resume after this transaction.
                for driver in drivers.values():
                    try:
                        driver.cancel()
                    except Exception as exc:
                        outcome, error = "failed", error or exc
                clear_targets()
                self.state, self.error = outcome, error
                self.active = False
                # Always arm this after success: an active read could have
                # been only hasattr's discarded value. This notification does
                # not extend reset ownership or change the measured status.
                self._completion_reads_remaining = 2 if outcome == "succeeded" else 0
