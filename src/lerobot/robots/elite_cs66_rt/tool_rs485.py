"""Elite SDK tool configuration plus its TCP-to-TCI socat bridge.

The pinned SDK's startToolRs485 reuses any existing listener and its SSH helper
has no deadline. Keep the same wire path, but manage the SSH subprocess here:
bounded waits, password in the subprocess environment, and only our own PID is
stopped. No second EliteDriver or replacement external-control program is used.
"""

import math
import os
import shutil
import socket
import subprocess
import time
import uuid
from contextlib import suppress

from lerobot.grippers.dh_ag95.modbus_rtu import ModbusRTUClient
from lerobot.utils.robot_utils import best_effort

from .config_tool_rs485 import EliteToolRS485Config

# Same boot allowance as Dobot's tested test_dh_gripper_open_close.py.
_TOOL_POWER_WAIT_S = 2.0


class EliteToolRS485:
    def __init__(self, driver, robot_ip: str, config: EliteToolRS485Config, connect_timeout: float):
        self._driver = driver
        self._ip = robot_ip
        self._config = config
        self._connect_timeout = connect_timeout
        self._client: ModbusRTUClient | None = None
        self._password: str | None = None
        self._pid_file = f"/tmp/lerobot-tool-rs485-{uuid.uuid4().hex}.pid"
        self._launch_attempted = False

    @staticmethod
    def preflight(tool_config: EliteToolRS485Config) -> None:
        if not os.environ.get(tool_config.ssh_password_env):
            raise ValueError(
                f"Set {tool_config.ssh_password_env} to the Elite controller SSH password before connecting"
            )
        if shutil.which("sshpass") is None or shutil.which("ssh") is None:
            raise RuntimeError("Elite tool RS485 requires ssh and sshpass (Ubuntu: sudo apt install sshpass)")

    def _ssh(self, command: str) -> str:
        env = os.environ.copy()
        env["SSHPASS"] = self._password
        try:
            result = subprocess.run(
                [
                    "sshpass",
                    "-e",
                    "ssh",
                    "-o",
                    "StrictHostKeyChecking=accept-new",
                    "-o",
                    f"ConnectTimeout={max(1, math.ceil(self._connect_timeout))}",
                    "-o",
                    "NumberOfPasswordPrompts=1",
                    f"root@{self._ip}",
                    command,
                ],
                env=env,
                capture_output=True,
                text=True,
                check=True,
                timeout=self._connect_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"Elite tool RS485 SSH operation timed out on {self._ip}") from exc
        except subprocess.CalledProcessError as exc:
            raise ConnectionError(f"Elite tool RS485 SSH failed on {self._ip}: {exc.stderr.strip()}") from exc
        return result.stdout.strip()

    def open(self, slave_id: int, baudrate: int) -> None:
        self.preflight(self._config)
        if not all(hasattr(self._driver, name) for name in ("configureToolRs485", "setToolVoltage")):
            raise RuntimeError("Rebuild libpyelite with tool RS485 support: ./setup_env.sh --install --elite")
        self._password = os.environ[self._config.ssh_password_env]
        port = self._config.tcp_port
        # Refuse a borrowed bridge: its settings, serial owner and lifetime are
        # unknown. The operator must also disconnect any plugin owning the TCI port.
        try:
            existing = socket.create_connection((self._ip, port), timeout=self._connect_timeout)
        except ConnectionRefusedError:
            pass
        else:
            existing.close()
            raise RuntimeError(
                f"Elite {self._ip}:{port} accepted a TCP connection before bridge startup; "
                "the listener owner is unverified. Check the controller IP and direct routing "
                "(a VPN/TUN proxy can also accept this connection). "
                "Stop an existing service only after identifying its owner."
            )
        try:
            # Use the same login-shell PATH as the SDK's launch command. Check
            # controller dependencies before changing tool mode or enabling 24 V.
            available = self._ssh(
                "bash -lc 'if command -v socat >/dev/null 2>&1; "
                'then printf "ELITE_SOCAT_AVAILABLE\\n"; else printf "ELITE_SOCAT_MISSING\\n"; fi\''
            )
            if available.splitlines()[-1:] != ["ELITE_SOCAT_AVAILABLE"]:
                raise RuntimeError(
                    f"Elite controller {self._ip} has no socat in its login-shell PATH. "
                    "The official SDK TCP-to-tool-RS485 bridge requires this controller-side program. "
                    "Have the controller administrator/vendor provision a compatible socat; "
                    "tool mode and voltage have not been changed."
                )
            if not self._driver.configureToolRs485(baudrate, 0, 1):
                raise RuntimeError("Elite tool RS485 configuration failed")
            if not self._driver.setToolVoltage(24):
                raise RuntimeError("Elite 24 V tool power command failed")
            time.sleep(_TOOL_POWER_WAIT_S)
            # Same socat command and TCI device as the official Elite SDK.
            self._launch_attempted = True
            result = self._ssh(
                "bash -lc 'socat tcp-l:"
                + str(port)
                + ",reuseaddr,fork,nodelay file:/dev/ttyTCI0,nonblock,raw,waitlock=/var/run/tty0"
                + f" >/dev/null 2>&1 </dev/null & echo $! >{self._pid_file}; cat {self._pid_file}'"
            )
            pid = int(result)
            if pid <= 1:
                raise RuntimeError("Elite bridge returned an invalid PID")
            # Wait for this process to listen. The deadline uses the existing
            # robot connect_timeout_s, not a new guessed gripper tuning value.
            deadline = time.monotonic() + self._connect_timeout
            while True:
                try:
                    connection = socket.create_connection(
                        (self._ip, port), timeout=max(0.001, deadline - time.monotonic())
                    )
                    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    self._client = ModbusRTUClient(connection, slave_id, self._config.transaction_timeout_s, baudrate)
                    break
                except ConnectionRefusedError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Elite tool RS485 bridge did not start") from None
                    time.sleep(0.1)
            self._ssh(f"kill -0 {pid}")
        except BaseException:
            with suppress(Exception):
                self.close()
            raise

    def read_register(self, address: int) -> int:
        if self._client is None:
            raise ConnectionError("Elite tool RS485 is closed")
        return self._client.read_register(address)

    def write_register(self, address: int, value: int) -> None:
        if self._client is None:
            raise ConnectionError("Elite tool RS485 is closed")
        self._client.write_register(address, value)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        if self._launch_attempted:
            port = self._config.tcp_port
            # Never kill a process solely by its port or a potentially reused PID.
            # The session-specific pid file also permits cleanup if SSH lost the
            # launch response after the process was started.
            self._ssh(
                f"if [ -r {self._pid_file} ]; then pid=$(cat {self._pid_file}); "
                'case "$pid" in \'\'|*[!0-9]*) exit 1;; esac; [ "$pid" -gt 1 ] || exit 1; '
                "if [ -r /proc/$pid/cmdline ]; then "
                "case \"$(tr '\\000' ' ' </proc/$pid/cmdline)\" in "
                f"*'socat tcp-l:{port},'*'file:/dev/ttyTCI0,'*) kill \"$pid\";; "
                "*) echo 'Bridge PID no longer belongs to this session' >&2; exit 1;; esac; fi; "
                f"rm -f {self._pid_file}; fi"
            )
            self._launch_attempted = False
        self._password = None


def connect_elite_gripper(gripper, driver, robot_ip: str, config: EliteToolRS485Config, connect_timeout: float):
    """Bind robot transport only for DH; preserve existing backend lifecycles."""
    if gripper.config.type == "dh-ag95":
        session = EliteToolRS485(driver, robot_ip, config, connect_timeout)
        # Give the robot's gripper ownership before any IO so failed startup
        # cleanup retains the session even when its first SSH close also fails.
        gripper.bind_transport(session)
        try:
            session.open(gripper.config.slave_id, gripper.config.baudrate)
            gripper.connect()
        except BaseException:
            with suppress(Exception):
                gripper.disconnect()
            raise
    else:
        gripper.connect()


def disconnect_elite_tool_gripper(gripper, logger) -> None:
    """Release arm-owned tool IO before stopping the external-control driver."""
    if gripper is not None and gripper.config.type == "dh-ag95":
        with best_effort(logger, "closing DH AG-95 and Elite tool RS485", level="warn"):
            gripper.disconnect()
