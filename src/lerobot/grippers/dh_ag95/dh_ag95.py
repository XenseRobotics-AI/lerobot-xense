"""Arm-independent AG-95 driver, adapted from the validated Dobot integration.

One worker coalesces targets and optionally reads actual position. With the
tested default position_poll_frequency=0, position is the acknowledged command,
NOT a measured opening. ``read_status`` always reads hardware explicitly.
Call ``disconnect()`` explicitly or use a context manager. The running worker
holds this gripper alive, so garbage collection cannot close its transport.
"""

import math
import threading
import time
from contextlib import suppress

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..gripper import DEFAULT_INIT_POSITION_TOLERANCE, Gripper
from .configuration_dh_ag95 import DHAG95GripperConfig
from .modbus_rtu import RegisterIO

INITIALIZE = 0x0100
FORCE = 0x0101
POSITION = 0x0103
INIT_STATE = 0x0200
GRIP_STATE = 0x0201
CURRENT_POSITION = 0x0202


class DHAG95Gripper(Gripper):
    config_class = DHAG95GripperConfig

    def __init__(self, config: DHAG95GripperConfig):
        super().__init__(config)
        self._transport: RegisterIO | None = None
        self._is_connected = False
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()
        self._state_lock = threading.Lock()
        self._io_lock = threading.Lock()
        self._position = 0.0
        self._target: float | None = None
        self._sequence = 0
        self._sent_sequence = 0
        self._last_sent: float | None = None
        self._error: Exception | None = None

    def __del__(self) -> None:
        # Best effort for a bound session with no running worker, including
        # failed startup or teardown. Connected workers require explicit close.
        if getattr(self, "_transport", None) is not None:
            with suppress(Exception):
                self.disconnect()

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def position_source(self) -> str:
        return "measured" if self.config.position_poll_frequency > 0 else "commanded"

    def bind_transport(self, transport: RegisterIO) -> None:
        """Transfer ownership of a robot-provided register session before connect."""
        if self._transport is not None or self._is_connected:
            raise DeviceAlreadyConnectedError("DH AG-95 already owns a transport")
        self._transport = transport

    def _read(self, address: int) -> int:
        with self._io_lock:
            if self._transport is None:
                raise DeviceNotConnectedError("DH AG-95 transport is not bound")
            return self._transport.read_register(address)

    def _write(self, address: int, value: int) -> None:
        with self._io_lock:
            if self._transport is None:
                raise DeviceNotConnectedError("DH AG-95 transport is not bound")
            self._transport.write_register(address, value)

    def _actual_position(self) -> float:
        raw = self._read(CURRENT_POSITION)
        if not 0 <= raw <= 1000:
            raise RuntimeError(f"DH AG-95 returned invalid actual position {raw}")
        return raw / 1000.0

    def connect(self) -> None:
        if self._is_connected:
            raise DeviceAlreadyConnectedError("DH AG-95 is already connected")
        if self._transport is None:
            raise DeviceNotConnectedError("Bind a register transport before connecting DH AG-95")
        self._error = None
        self._stop.clear()
        try:
            if self._read(INIT_STATE) != 1:
                # Same full initialization as Dobot; requires unobstructed fingers.
                self._write(INITIALIZE, 0xA5)
                deadline = time.monotonic() + self.config.init_timeout
                while self._read(INIT_STATE) != 1:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("DH AG-95 hardware initialization timed out")
                    time.sleep(0.1)  # Dobot initialization poll interval.
            self._write(FORCE, self.config.gripper_force)
            self._last_sent = None
            if self.config.init_open:
                self._write(POSITION, 1000)
                self._last_sent = 1.0
                self._position = 1.0
            else:
                self._position = self._actual_position()
            if self.config.position_poll_frequency > 0:
                self._position = self._actual_position()
            self._target = self._last_sent
            self._sequence = self._sent_sequence = 0
            self._is_connected = True
            self.logger.info(
                f"DH AG-95: worker={self.config.worker_frequency:g} Hz, "
                f"poll={self.config.position_poll_frequency:g} Hz, position source={self.position_source}"
            )
            self._worker = threading.Thread(target=self._worker_loop, name=f"dh-ag95-{self.config.side}", daemon=True)
            self._worker.start()
        except BaseException:
            with suppress(Exception):
                self.disconnect()
            raise

    def _check_ready(self) -> None:
        if not self._is_connected:
            raise DeviceNotConnectedError("DH AG-95 is not connected")
        if self._error is not None:
            raise RuntimeError(f"DH AG-95 communication failed: {self._error}") from self._error

    def get_gripper_position(self) -> float:
        self._check_ready()
        with self._state_lock:
            return self._position

    def set_gripper_position(self, normalized_pos: float) -> None:
        self._check_ready()
        self._queue_target(self.config.normalize_position(normalized_pos))

    def _queue_target(self, target: float) -> int:
        with self._state_lock:
            self._target = target
            self._sequence += 1
            return self._sequence

    def _worker_loop(self) -> None:
        interval = 1.0 / self.config.worker_frequency
        poll_interval = 1.0 / self.config.position_poll_frequency if self.config.position_poll_frequency else None
        next_read = time.monotonic()
        try:
            while not self._stop.is_set():
                start = time.monotonic()
                with self._state_lock:
                    target, sequence, last_sent = self._target, self._sequence, self._last_sent
                if target is not None and sequence != self._sent_sequence:
                    changed = last_sent is None or round(target * 1000) != round(last_sent * 1000)
                    if changed:
                        self._write(POSITION, round(target * 1000))
                    with self._state_lock:
                        if changed:
                            self._last_sent = target
                            if poll_interval is None:
                                self._position = target
                        # A newer target submitted during IO retains its sequence.
                        self._sent_sequence = sequence
                if poll_interval is not None and start >= next_read:
                    actual = self._actual_position()
                    with self._state_lock:
                        self._position = actual
                    next_read = time.monotonic() + poll_interval
                self._stop.wait(max(0, interval - (time.monotonic() - start)))
        except Exception as exc:
            if not self._stop.is_set():
                self._error = exc
                self.logger.error(f"DH AG-95 communication stopped: {exc}")

    def read_status(self) -> dict[str, int | float]:
        """Explicit diagnostic read, including when periodic polling is disabled."""
        self._check_ready()
        return {
            "initialized": self._read(INIT_STATE),
            "grip_state": self._read(GRIP_STATE),
            "position": self._actual_position(),
        }

    def initialize_gripper_position(
        self,
        normalized_pos: float = 1.0,
        tolerance: float = DEFAULT_INIT_POSITION_TOLERANCE,
        timeout: float = 10.0,  # Dobot set_gripper_position_sync default.
    ) -> None:
        if not math.isfinite(timeout) or timeout <= 0 or not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("Initialization timeout must be positive and tolerance nonnegative, both finite")
        self._check_ready()
        target = self.config.normalize_position(normalized_pos)
        start_position = self._actual_position()
        closing = target < start_position - tolerance
        sequence = self._queue_target(target)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._check_ready()
            with self._state_lock:
                sent = self._sent_sequence >= sequence
                superseded = self._sequence != sequence
            if superseded:
                raise RuntimeError("DH AG-95 initialization move was superseded by a newer target")
            if sent:
                state = self._read(GRIP_STATE)
                actual = self._actual_position()
                with self._state_lock:
                    if self._sequence != sequence:
                        raise RuntimeError("DH AG-95 initialization move was superseded by a newer target")
                if state == 3:
                    raise RuntimeError("DH AG-95 object dropped during initialization move")
                # A stale 'gripped' state from a previous close cannot complete
                # a new opening move. Contact may finish a closing move only.
                if (state == 2 and closing) or (state in (1, 2) and abs(actual - target) <= tolerance):
                    return
            time.sleep(0.01)  # Dobot synchronous move poll interval.
        raise TimeoutError("DH AG-95 did not complete the initialization move")

    def disconnect(self) -> None:
        # Also handles a bound session whose connect failed; no opening command.
        self._stop.set()
        transport = self._transport
        try:
            if transport is not None:
                transport.close()
            self._transport = None
        finally:
            if self._worker is not None:
                if self._worker.ident is not None and self._worker is not threading.current_thread():
                    self._worker.join()
                self._worker = None
            self._is_connected = False
