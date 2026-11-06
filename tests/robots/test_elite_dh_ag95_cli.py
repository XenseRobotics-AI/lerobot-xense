"""Standalone CLI tests use dummy credentials and a simulated register session."""

import importlib.util
import json
import os
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

PASSWORD_ENV = "ELITE_DH_CLI_TEST_PASSWORD"
DUMMY_PASSWORD = "dummy-cli-test-only"
ROBOT_IP = "192.0.2.1"


@pytest.fixture(scope="module")
def cli():
    path = Path(__file__).resolve().parents[2] / "examples" / "elite_dh_ag95.py"
    spec = importlib.util.spec_from_file_location("elite_dh_ag95_cli_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def terminal(cli, monkeypatch):
    monkeypatch.delenv(PASSWORD_ENV, raising=False)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    prompt = Mock(return_value=DUMMY_PASSWORD)
    monkeypatch.setattr(cli.getpass, "getpass", prompt)
    return prompt


@pytest.mark.parametrize("original", [None, ""])
def test_missing_or_empty_password_prompts_for_controller_and_restores_env(
    cli, terminal, monkeypatch, capsys, original
):
    if original is not None:
        monkeypatch.setenv(PASSWORD_ENV, original)
    with cli._ssh_password(PASSWORD_ENV, ROBOT_IP):
        assert os.environ[PASSWORD_ENV] == DUMMY_PASSWORD
    terminal.assert_called_once_with(f"Elite controller root@{ROBOT_IP} SSH password: ")
    assert os.environ.get(PASSWORD_ENV) == original
    captured = capsys.readouterr()
    assert DUMMY_PASSWORD not in captured.out + captured.err


def test_existing_env_is_used_without_prompt_even_without_terminal(cli, terminal, monkeypatch):
    monkeypatch.setenv(PASSWORD_ENV, DUMMY_PASSWORD)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    with cli._ssh_password(PASSWORD_ENV, ROBOT_IP):
        assert os.environ[PASSWORD_ENV] == DUMMY_PASSWORD
    terminal.assert_not_called()
    assert os.environ[PASSWORD_ENV] == DUMMY_PASSWORD


def test_empty_input_is_retried_without_connecting(cli, terminal, capsys):
    terminal.side_effect = ["", DUMMY_PASSWORD]
    with cli._ssh_password(PASSWORD_ENV, ROBOT_IP):
        assert os.environ[PASSWORD_ENV] == DUMMY_PASSWORD
    assert terminal.call_count == 2
    assert "cannot be empty" in capsys.readouterr().err


def test_password_whitespace_is_preserved(cli, terminal):
    terminal.return_value = " " + DUMMY_PASSWORD + " "
    with cli._ssh_password(PASSWORD_ENV, ROBOT_IP):
        assert os.environ[PASSWORD_ENV] == " " + DUMMY_PASSWORD + " "


def test_prompted_password_is_removed_after_connection_error(cli, terminal):
    with pytest.raises(ConnectionError, match="offline test"), cli._ssh_password(PASSWORD_ENV, ROBOT_IP):
        raise ConnectionError("offline test")
    assert PASSWORD_ENV not in os.environ


def test_missing_env_without_terminal_does_not_prompt(cli, terminal, monkeypatch):
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    with pytest.raises(cli.PasswordInputError, match="non-interactive"), cli._ssh_password(PASSWORD_ENV, ROBOT_IP):
        pytest.fail("a session must not start without a password")
    terminal.assert_not_called()


def test_getpass_cannot_fall_back_to_echoed_input(cli, terminal):
    def warn(*args):
        warnings.warn("Cannot disable echo", cli.getpass.GetPassWarning, stacklevel=2)
        pytest.fail("GetPassWarning must interrupt before an echoed read")

    terminal.side_effect = warn
    with pytest.raises(cli.PasswordInputError, match="Cannot hide"), cli._ssh_password(PASSWORD_ENV, ROBOT_IP):
        pytest.fail("a session must not start")
    assert PASSWORD_ENV not in os.environ


@pytest.mark.parametrize(("error", "exit_code"), [(EOFError, 2), (KeyboardInterrupt, 130)])
def test_cancelled_input_exits_before_constructing_session(cli, terminal, monkeypatch, error, exit_code):
    monkeypatch.setattr(cli.sys, "argv", ["probe", "--robot-ip", ROBOT_IP, "--ssh-password-env", PASSWORD_ENV])
    session_factory = Mock()
    monkeypatch.setattr(cli, "EliteToolRS485", session_factory)
    terminal.side_effect = error
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == exit_code
    session_factory.assert_not_called()
    assert PASSWORD_ENV not in os.environ


def test_status_uses_prompted_password_through_cleanup(cli, terminal, monkeypatch, capsys):
    monkeypatch.setattr(cli.sys, "argv", ["probe", "--robot-ip", ROBOT_IP, "--ssh-password-env", PASSWORD_ENV])
    session = Mock()

    def require_password(*args):
        assert os.environ[PASSWORD_ENV] == DUMMY_PASSWORD

    session.open.side_effect = require_password
    session.close.side_effect = require_password
    session.read_register.side_effect = {cli.INIT_STATE: 1, cli.GRIP_STATE: 1, cli.CURRENT_POSITION: 600}.__getitem__
    monkeypatch.setattr(cli, "EliteToolRS485", Mock(return_value=session))
    gripper = Mock()
    monkeypatch.setattr(cli, "make_gripper_from_config", Mock(return_value=gripper))
    cli.main()
    assert PASSWORD_ENV not in os.environ
    session.open.assert_called_once_with(1, 115200)
    session.close.assert_called_once()
    gripper.connect.assert_not_called()
    captured = capsys.readouterr()
    assert json.loads(captured.out)["actual_position"] == 0.6
    assert DUMMY_PASSWORD not in captured.out + captured.err


@pytest.fixture
def force_cycle(cli, monkeypatch):
    from lerobot.grippers.dh_ag95.dh_ag95 import FORCE, POSITION

    events = []

    class RegisterSession:
        def __init__(self):
            self.registers = {cli.INIT_STATE: 1, cli.GRIP_STATE: 1, cli.CURRENT_POSITION: 1000, FORCE: 33}
            self.closed = False
            self.reject_force = None

        def open(self, slave_id, baudrate):
            events.append(("connect", slave_id, baudrate))

        def read_register(self, address):
            return self.registers[address]

        def write_register(self, address, value):
            self.registers[address] = value
            if address == FORCE:
                if value == self.reject_force:
                    self.registers[address] = value - 1
                events.append(("force", value))
            elif address == POSITION:
                force = self.registers[FORCE]
                events.append(("move", force, value))
                self.registers[cli.CURRENT_POSITION] = 420 if value == 0 else value
                self.registers[cli.GRIP_STATE] = 2 if value == 0 else 1

        def close(self):
            self.closed = True

    session = RegisterSession()
    monkeypatch.setenv(PASSWORD_ENV, DUMMY_PASSWORD)
    monkeypatch.setattr(cli, "EliteToolRS485", lambda *args: session)
    monkeypatch.setattr(cli, "time", SimpleNamespace(sleep=lambda seconds: events.append(("sleep", seconds))))
    return session, events


@pytest.mark.parametrize("cycles", [1, 2])
def test_cycle_runs_three_force_levels_and_pauses_after_each_round(cli, force_cycle, monkeypatch, capsys, cycles):
    session, events = force_cycle
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "probe",
            "--robot-ip",
            ROBOT_IP,
            "--ssh-password-env",
            PASSWORD_ENV,
            "--action",
            "cycle",
            "--cycles",
            str(cycles),
            "--hold-s",
            "0.01",
        ],
    )
    cli.main()
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    assert len(rows) == 6 * cycles
    assert [(row["cycle"], row["force_percent"], row["command"]) for row in rows] == [
        (cycle, force, target) for cycle in range(1, cycles + 1) for force in (20, 55, 100) for target in (0.0, 1.0)
    ]
    assert [event[1] for event in events if event[0] == "move" and event[2] == 0] == [20, 55, 100] * cycles
    assert [event[1] for event in events if event[0] == "sleep"] == ([0.01] * 6 + [3.0]) * cycles
    assert all(row["grip_state"] == 2 and row["position"] == 0.42 for row in rows if row["command"] == 0)
    assert all(row["position"] == 1.0 for row in rows if row["command"] == 1)
    assert session.closed


def test_cycle_stops_on_force_readback_mismatch(cli, force_cycle, monkeypatch):
    session, events = force_cycle
    session.reject_force = 55
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["probe", "--robot-ip", ROBOT_IP, "--ssh-password-env", PASSWORD_ENV, "--action", "cycle", "--cycles", "1"],
    )
    with pytest.raises(RuntimeError, match="force.*55.*54"):
        cli.main()
    assert [event[1] for event in events if event[0] == "move" and event[2] == 0] == [20]
    assert ("sleep", 3.0) not in events
    assert session.closed
