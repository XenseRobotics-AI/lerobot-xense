"""The AG-95's FC03/FC06 subset over a transparent TCP-to-RS485 bridge.

This is RTU (including CRC), not Modbus TCP with an MBAP header. Only one
transaction may be outstanding. A broken/ambiguous stream is closed rather
than replaying a write or mistaking a late read response for another register.
"""

import socket
import struct
import threading
import time
from contextlib import suppress
from typing import Protocol


class RegisterIO(Protocol):
    def read_register(self, address: int) -> int: ...
    def write_register(self, address: int, value: int) -> None: ...
    def close(self) -> None: ...


def crc16(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0xA001 if crc & 1 else 0)
    return crc


def frame(payload: bytes) -> bytes:
    return payload + struct.pack("<H", crc16(payload))


class ModbusRTUError(OSError):
    """Invalid, exceptional, or incomplete Modbus response."""


class ModbusRTUClient:
    def __init__(self, connection: socket.socket, slave_id: int, timeout: float, baudrate: int = 115200):
        self._socket: socket.socket | None = connection
        self._slave_id = slave_id
        self._timeout = timeout
        self._transaction_lock = threading.Lock()
        # Modbus RTU t3.5, using the same character timing as the official
        # plugin's modbus_tk.utils.calculate_rtu_inter_char (11 bits/character).
        self._frame_gap = 3.5 * (0.0005 if baudrate > 19200 else 11.0 / baudrate)
        self._last_response_at = 0.0

    def close(self) -> None:
        # Deliberately does not take the transaction lock: shutdown interrupts
        # a blocked recv so the gripper worker can be joined during teardown.
        connection, self._socket = self._socket, None
        if connection is not None:
            with suppress(OSError):
                connection.shutdown(socket.SHUT_RDWR)
            connection.close()

    @staticmethod
    def _read_exact(connection: socket.socket, size: int, deadline: float) -> bytes:
        result = bytearray()
        while len(result) < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("DH AG-95 Modbus response timed out")
            connection.settimeout(remaining)
            data = connection.recv(size - len(result))
            if not data:
                raise ModbusRTUError("DH AG-95 bridge closed during a response")
            result.extend(data)
        return bytes(result)

    def _exchange(self, function: int, address: int, value: int) -> bytes:
        request = frame(struct.pack(">BBHH", self._slave_id, function, address, value))
        with self._transaction_lock:
            connection = self._socket
            if connection is None:
                raise ModbusRTUError("DH AG-95 Modbus transport is closed; reconnect explicitly")
            try:
                remaining_gap = self._frame_gap - (time.monotonic() - self._last_response_at)
                if remaining_gap > 0:
                    time.sleep(remaining_gap)
                deadline = time.monotonic() + self._timeout
                connection.settimeout(self._timeout)
                connection.sendall(request)
                header = self._read_exact(connection, 2, deadline)
                if header[0] != self._slave_id or header[1] not in (function, function | 0x80):
                    raise ModbusRTUError(f"Unexpected Modbus slave/function: {header.hex()}")
                if header[1] & 0x80:
                    response = header + self._read_exact(connection, 3, deadline)
                elif function == 3:
                    count = self._read_exact(connection, 1, deadline)
                    if count != b"\x02":
                        raise ModbusRTUError("Expected exactly one 16-bit register")
                    response = header + count + self._read_exact(connection, 4, deadline)
                else:
                    response = header + self._read_exact(connection, 6, deadline)
                if crc16(response[:-2]) != struct.unpack("<H", response[-2:])[0]:
                    raise ModbusRTUError("DH AG-95 Modbus CRC mismatch")
                if response[1] & 0x80:
                    raise ModbusRTUError(f"DH AG-95 Modbus exception {response[2]} for function {function}")
                if function == 6 and response != request:
                    raise ModbusRTUError("DH AG-95 write acknowledgement does not match the request")
                self._last_response_at = time.monotonic()
                return response
            except OSError:
                self.close()
                raise

    def read_register(self, address: int) -> int:
        return struct.unpack(">H", self._exchange(3, address, 1)[3:5])[0]

    def write_register(self, address: int, value: int) -> None:
        self._exchange(6, address, value)
