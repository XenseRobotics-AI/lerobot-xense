"""Elite-specific bridge settings; imports no hardware SDK."""

import math
from dataclasses import dataclass


@dataclass
class EliteToolRS485Config:
    # Elite SDK startToolRs485 default. Each controller has its own port.
    tcp_port: int = 54321
    ssh_password_env: str = "ELITE_SSH_PASSWORD"
    # DH.Gripper 3.1.10 daemon/main.py: RtuMaster.set_timeout(1).
    transaction_timeout_s: float = 1.0

    def __post_init__(self):
        if not isinstance(self.tcp_port, int) or not 1024 <= self.tcp_port <= 65535:
            raise ValueError("tool_rs485.tcp_port must be an unprivileged TCP port in [1024, 65535]")
        if not self.ssh_password_env or "=" in self.ssh_password_env or "\x00" in self.ssh_password_env:
            raise ValueError("tool_rs485.ssh_password_env must name an environment variable")
        if not math.isfinite(self.transaction_timeout_s) or self.transaction_timeout_s <= 0:
            raise ValueError("tool_rs485.transaction_timeout_s must be finite and positive")
