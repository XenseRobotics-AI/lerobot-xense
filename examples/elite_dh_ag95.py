#!/usr/bin/env python
"""AG-95 tool-only diagnostics on an already powered Elite controller.

No arm powerOn, brakeRelease, MoveJ, servo, or replacement main program is sent.
The tool is configured/powered at 24 V using secondary scripts. ``status`` does
not initialize or command the jaws; ``cycle`` does (including full calibration
if the gripper reports uninitialized). Each cycle closes/opens at 20%, 55%, and
100% force, then pauses for 3 s. Exit preserves the last jaw command.
"""

import argparse
import getpass
import json
import os
import socket
import sys
import time
import warnings
from contextlib import contextmanager

from lerobot.grippers import DHAG95GripperConfig, make_gripper_from_config
from lerobot.grippers.dh_ag95.dh_ag95 import CURRENT_POSITION, FORCE, GRIP_STATE, INIT_STATE
from lerobot.robots.elite_cs66_rt.config_tool_rs485 import EliteToolRS485Config
from lerobot.robots.elite_cs66_rt.tool_rs485 import EliteToolRS485

FORCE_LEVELS = (20, 55, 100)
CYCLE_PAUSE_S = 3.0


class PasswordInputError(ValueError):
    """No usable password could be read before opening the tool connection."""


@contextmanager
def _ssh_password(env_name: str, robot_ip: str):
    """Use an existing environment value, or prompt privately for this run only.

    Keep this interaction in the standalone CLI. Robot/recording backends remain
    non-interactive and require their configured password environment variable.
    """
    original = os.environ.get(env_name)
    if original:
        yield
        return
    if not sys.stdin.isatty():
        raise PasswordInputError(
            f"{env_name} is unset or empty. Set it before non-interactive runs, "
            "or run this command in a terminal to enter the controller SSH password."
        )

    print("SSH is used to start the controller's RS485-to-TCP bridge. Password input is hidden.", file=sys.stderr)
    while True:
        try:
            with warnings.catch_warnings():
                # getpass normally falls back to echoed input if it cannot
                # disable terminal echo. Refuse that fallback for credentials.
                warnings.simplefilter("error", getpass.GetPassWarning)
                password = getpass.getpass(f"Elite controller root@{robot_ip} SSH password: ")
        except EOFError:
            raise PasswordInputError("SSH password input ended; no robot connection was opened.") from None
        except getpass.GetPassWarning:
            raise PasswordInputError(
                f"Cannot hide password input in this terminal. Set {env_name} before running the command."
            ) from None
        if password:
            break
        print("SSH password cannot be empty. Try again, or press Ctrl+C to cancel.", file=sys.stderr)

    os.environ[env_name] = password
    try:
        yield
    finally:
        if original is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = original


class ToolOnlyClient:
    """The tool-only subset of the official SDK's secondary-script setup."""

    def __init__(self, ip: str):
        self.ip = ip

    def _send(self, body: str) -> bool:
        script = "sec dh_ag95_tool_setup():\n" + body + "end\n"
        # Same Primary port used by EliteDriver::sendScript. A successful send
        # is not a controller ACK; the subsequent Modbus probe verifies the path.
        with socket.create_connection((self.ip, 30001), timeout=10.0) as connection:
            connection.sendall(script.encode("ascii"))
        return True

    def configureToolRs485(self, baudrate: int, parity: int, stop_bits: int) -> bool:  # noqa: N802
        return self._send(
            f"    set_tool_analog_io_work_mode(0)\n    tool_serial_config(True,{baudrate},{parity},{stop_bits})\n"
        )

    def setToolVoltage(self, voltage: int) -> bool:  # noqa: N802
        return self._send(f"    set_tool_voltage({voltage})\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", required=True)
    parser.add_argument("--action", choices=("status", "cycle"), default="status")
    parser.add_argument("--tcp-port", type=int, default=54321)
    parser.add_argument(
        "--ssh-password-env",
        default="ELITE_SSH_PASSWORD",
        help="Use this environment variable; if unset or empty, prompt privately in an interactive terminal.",
    )
    parser.add_argument("--slave-id", type=int, default=1)
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--cycles", type=int, default=1, help="Number of 20/55/100%% force rounds (default: 1).")
    parser.add_argument(
        "--hold-s", type=float, default=1.5, help="Dwell after each completed jaw movement, in seconds."
    )
    args = parser.parse_args()
    if args.cycles < 1 or not 0 < args.hold_s < float("inf"):
        parser.error("cycles must be >= 1 and hold-s finite and positive")
    config = DHAG95GripperConfig(slave_id=args.slave_id, baudrate=args.baudrate, gripper_force=FORCE_LEVELS[0])
    bridge_config = EliteToolRS485Config(tcp_port=args.tcp_port, ssh_password_env=args.ssh_password_env)
    try:
        with _ssh_password(bridge_config.ssh_password_env, args.robot_ip):
            session = EliteToolRS485(ToolOnlyClient(args.robot_ip), args.robot_ip, bridge_config, 10.0)
            gripper = make_gripper_from_config(config)
            try:
                session.open(config.slave_id, config.baudrate)
                if args.action == "status":
                    print(
                        json.dumps(
                            {
                                "initialized": session.read_register(INIT_STATE),
                                "grip_state": session.read_register(GRIP_STATE),
                                "actual_position": session.read_register(CURRENT_POSITION) / 1000,
                            }
                        )
                    )
                else:
                    print("Cycling AG-95: uninitialized jaws perform full calibration; keep the stroke clear.")
                    gripper.bind_transport(session)
                    gripper.connect()
                    gripper.initialize_gripper_position(1.0)
                    for cycle in range(1, args.cycles + 1):
                        for force in FORCE_LEVELS:
                            session.write_register(FORCE, force)
                            force_readback = session.read_register(FORCE)
                            if force_readback != force:
                                raise RuntimeError(f"AG-95 force requested {force}%, but read back {force_readback}%")
                            print(f"Cycle {cycle}/{args.cycles}, force {force}%.", flush=True)
                            for target in (0.0, 1.0):
                                gripper.initialize_gripper_position(target)
                                time.sleep(args.hold_s)
                                print(
                                    json.dumps(
                                        {
                                            "cycle": cycle,
                                            "force_percent": force_readback,
                                            "command": target,
                                            **gripper.read_status(),
                                        }
                                    ),
                                    flush=True,
                                )
                        print(f"Cycle {cycle} complete; pausing {CYCLE_PAUSE_S:g} s.", flush=True)
                        time.sleep(CYCLE_PAUSE_S)
            finally:
                try:
                    gripper.disconnect()
                finally:
                    session.close()
    except PasswordInputError as exc:
        parser.error(str(exc))
    except KeyboardInterrupt:
        parser.exit(130, "\nCancelled.\n")


if __name__ == "__main__":
    main()
