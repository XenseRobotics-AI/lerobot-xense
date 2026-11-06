"""AG-95 protocol and worker regression tests; no robot connections."""

import socket
import struct
import threading
import time

import draccus
import pytest

from lerobot.grippers import DHAG95GripperConfig, GripperConfig, command_gripper_position, make_gripper_from_config
from lerobot.grippers.dh_ag95.dh_ag95 import CURRENT_POSITION, FORCE, GRIP_STATE, INIT_STATE, INITIALIZE, POSITION
from lerobot.grippers.dh_ag95.modbus_rtu import ModbusRTUClient, ModbusRTUError, frame
from lerobot.robots.bi_elite_cs66_rt.config_bi_elite_cs66_rt import BiEliteCS66RTConfig


def wait_for(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            pytest.fail("worker did not reach the expected state")
        time.sleep(0.002)


class Registers:
    def __init__(self):
        self.values = {INIT_STATE: 1, GRIP_STATE: 1, CURRENT_POSITION: 600}
        self.writes = []
        self.reads = []
        self.closed = False
        self.fail_write = None
        self.before_write = None

    def read_register(self, address):
        self.reads.append(address)
        return self.values[address]

    def write_register(self, address, value):
        if self.before_write:
            self.before_write(address, value)
        if address == self.fail_write:
            raise OSError("write failed")
        self.writes.append((address, value))
        if address == INITIALIZE:
            self.values[INIT_STATE] = 1

    def close(self):
        self.closed = True


@pytest.fixture
def connected():
    grippers = []

    def create(**kwargs):
        gripper = make_gripper_from_config(DHAG95GripperConfig(**kwargs))
        registers = Registers()
        gripper.bind_transport(registers)
        gripper.connect()
        grippers.append(gripper)
        return gripper, registers

    yield create
    for gripper in grippers:
        gripper.disconnect()


def test_config_uses_robot_level_dobot_overrides_and_clones_sides():
    cfg = draccus.decode(GripperConfig, {"type": "dh-ag95"})
    assert cfg.gripper_force == 33
    assert cfg.worker_frequency == 50
    assert cfg.position_poll_frequency == 0
    assert cfg.binary_gripper_actions and cfg.gripper_open_threshold == 0.5
    bi = BiEliteCS66RTConfig(gripper=cfg)
    assert bi.left_gripper.side == "left" and bi.right_gripper.side == "right"
    assert cfg.side is None and bi.left_gripper is not bi.right_gripper
    assert not bi._serial_autodiscover and not bi._taccap_autodiscover


@pytest.mark.parametrize("field", ["command_epsilon", "gripper_speed"])
def test_config_rejects_unsupported_command_settings(field):
    with pytest.raises(draccus.utils.DecodingError):
        draccus.decode(GripperConfig, {"type": "dh-ag95", field: 0})


@pytest.mark.parametrize(
    "changes",
    [
        {"gripper_force": 15},
        {"worker_frequency": float("nan")},
        {"position_poll_frequency": 51},
        {"auto_discover_cameras": True},
        {"enable_tactile": True},
        {"slave_id": 0},
    ],
)
def test_invalid_config_is_rejected(changes):
    with pytest.raises(ValueError):
        DHAG95GripperConfig(**changes)


def test_connect_keeps_dobot_full_init_and_checks_writes():
    g = make_gripper_from_config(DHAG95GripperConfig())
    r = Registers()
    r.values[INIT_STATE] = 0
    r.fail_write = FORCE
    g.bind_transport(r)
    with pytest.raises(OSError, match="write failed"):
        g.connect()
    assert r.writes == [(INITIALIZE, 0xA5)]
    assert r.closed and not g.is_connected


def test_binary_mapping_and_commanded_cache_are_explicit(connected):
    g, r = connected()
    assert g.position_source == "commanded"
    assert command_gripper_position(g, 0.49) == 0.0
    wait_for(lambda: (POSITION, 0) in r.writes)
    assert g.get_gripper_position() == 0
    assert g.read_status()["position"] == 0.6
    before = r.reads.count(CURRENT_POSITION)
    time.sleep(0.06)
    assert r.reads.count(CURRENT_POSITION) == before  # No hidden periodic reads.
    assert command_gripper_position(g, 0.5) == 1.0
    wait_for(lambda: g.get_gripper_position() == 1)


def test_measured_cache_is_not_overwritten_by_a_target(connected):
    g, r = connected(position_poll_frequency=20, binary_gripper_actions=False)
    g.set_gripper_position(0.2)
    wait_for(lambda: (POSITION, 200) in r.writes)
    assert g.get_gripper_position() == 0.6
    assert g.position_source == "measured"


def test_small_continuous_moves_are_sent_and_complete(connected):
    g, r = connected(binary_gripper_actions=False)
    r.values[CURRENT_POSITION] = 1000
    original_write = r.write_register

    def reach_target(address, value):
        original_write(address, value)
        if address == POSITION:
            r.values[CURRENT_POSITION] = value

    r.write_register = reach_target
    for target in (0.96, 0.959):
        g.initialize_gripper_position(target, tolerance=0.0005, timeout=1.0)
    assert [v for a, v in r.writes if a == POSITION] == [1000, 960, 959]


@pytest.mark.parametrize("raise_inside", [False, True])
def test_context_manager_closes_transport_without_moving_jaws(raise_inside):
    g = make_gripper_from_config(DHAG95GripperConfig())
    r = Registers()
    g.bind_transport(r)
    try:
        with g:
            assert g.is_connected
            if raise_inside:
                raise RuntimeError("operation failed")
    except RuntimeError as exc:
        assert raise_inside and str(exc) == "operation failed"
    assert r.closed and not g.is_connected
    assert [v for a, v in r.writes if a == POSITION] == [1000]


def test_latest_target_survives_write_in_flight(connected):
    g, r = connected(binary_gripper_actions=False)
    entered, release = threading.Event(), threading.Event()

    def block(address, value):
        if address == POSITION and value == 200:
            entered.set()
            assert release.wait(1)

    r.before_write = block
    try:
        g.set_gripper_position(0.2)
        assert entered.wait(1)
        g.set_gripper_position(0.8)
        release.set()
        wait_for(lambda: (POSITION, 800) in r.writes)
        assert [v for a, v in r.writes if a == POSITION][-2:] == [200, 800]
    finally:
        release.set()


def test_failed_write_never_advances_cache_and_is_reported(connected):
    g, r = connected()
    r.fail_write = POSITION
    g.set_gripper_position(0)
    wait_for(lambda: g._error is not None)
    assert g._position == 1
    with pytest.raises(RuntimeError, match="communication failed"):
        g.get_gripper_position()
    with pytest.raises(RuntimeError, match="communication failed"):
        g.set_gripper_position(1)


def test_disconnect_does_not_open_and_reconnect_resets_worker_error(connected):
    g, r = connected()
    g.set_gripper_position(0)
    wait_for(lambda: g.get_gripper_position() == 0)
    before = list(r.writes)
    g.disconnect()
    assert r.closed and r.writes == before and g._worker is None
    replacement = Registers()
    g.bind_transport(replacement)
    g.connect()
    assert g.is_connected and g.get_gripper_position() == 1


def test_sync_move_uses_actual_state_even_with_polling_disabled(connected):
    g, r = connected()
    r.values[GRIP_STATE] = 3
    with pytest.raises(RuntimeError, match="dropped"):
        g.initialize_gripper_position(0, timeout=0.5)
    r.values[GRIP_STATE] = 1  # Old reached state at a different position is insufficient.
    with pytest.raises(TimeoutError):
        g.initialize_gripper_position(0, timeout=0.05)
    r.values[GRIP_STATE] = 2
    g.initialize_gripper_position(0, timeout=0.5)


def test_old_contact_state_cannot_complete_opening_move(connected):
    g, r = connected(init_open=False)
    r.values[GRIP_STATE] = 2
    r.values[CURRENT_POSITION] = 400
    with pytest.raises(TimeoutError):
        g.initialize_gripper_position(1, timeout=0.06)
    assert (POSITION, 1000) in r.writes
    r.values[CURRENT_POSITION] = 1000
    g.initialize_gripper_position(1, timeout=0.5)


def test_new_target_during_completion_read_cancels_sync_wait(connected):
    g, r = connected()
    original_read = r.read_register

    def read(address):
        if address == GRIP_STATE:
            g.set_gripper_position(1)
            return 2
        return original_read(address)

    r.read_register = read
    with pytest.raises(RuntimeError, match="superseded"):
        g.initialize_gripper_position(0, timeout=0.5)


def test_close_failure_retains_ownership_for_retry(connected):
    g, r = connected()
    attempts = []

    def close():
        attempts.append(1)
        if len(attempts) == 1:
            raise ConnectionError("SSH temporarily unavailable")
        r.closed = True

    r.close = close
    with pytest.raises(ConnectionError):
        g.disconnect()
    assert not g.is_connected and g._transport is r and g._worker is None
    g.disconnect()
    assert r.closed and g._transport is None and len(attempts) == 2


def test_failed_initialization_retains_failed_cleanup_session():
    g = make_gripper_from_config(DHAG95GripperConfig())
    r = Registers()
    r.fail_write = FORCE
    original_close = r.close
    r.close = lambda: (_ for _ in ()).throw(ConnectionError("SSH close failed"))
    g.bind_transport(r)
    with pytest.raises(OSError, match="write failed"):
        g.connect()
    assert g._transport is r and not g.is_connected
    r.close = original_close
    g.disconnect()
    assert r.closed and g._transport is None


def test_modbus_manual_known_crc_vector():
    assert frame(bytes.fromhex("01 06 01 00 00 01")) == bytes.fromhex("01 06 01 00 00 01 49 F6")


def exchange_server(response, *, fragmented=False, disconnect=False):
    client_sock, peer = socket.socketpair()
    requests = []

    def serve():
        with peer:
            data = b""
            while len(data) < 8:
                data += peer.recv(8 - len(data))
            requests.append(data)
            if not disconnect:
                payload = response(data) if callable(response) else response
                for chunk in [payload[i : i + 1] for i in range(len(payload))] if fragmented else [payload]:
                    peer.sendall(chunk)

    thread = threading.Thread(target=serve)
    thread.start()
    return ModbusRTUClient(client_sock, 1, 0.2), thread, requests


def test_modbus_fragmented_read_and_write_echo():
    client, server, requests = exchange_server(frame(bytes.fromhex("01 03 02 02 58")), fragmented=True)
    try:
        assert client.read_register(0x0202) == 600
    finally:
        client.close()
        server.join(1)
    assert requests == [frame(bytes.fromhex("01 03 02 02 00 01"))]
    client, server, _ = exchange_server(lambda request: request, fragmented=True)
    try:
        client.write_register(0x0103, 1000)
    finally:
        client.close()
        server.join(1)


@pytest.mark.parametrize(
    "response",
    [
        frame(bytes.fromhex("02 03 02 00 01")),  # Wrong slave.
        frame(bytes.fromhex("01 04 02 00 01")),  # Wrong function.
        frame(bytes.fromhex("01 03 04 00 01 00 02")),  # Wrong register count.
        bytes.fromhex("01 03 02 00 01 00 00"),  # Bad CRC.
        frame(bytes.fromhex("01 83 02")),  # Exception response.
        b"\x01",  # Truncated response.
    ],
)
def test_bad_responses_close_stream_and_are_not_replayed(response):
    client, server, requests = exchange_server(response)
    try:
        with pytest.raises(OSError):
            client.read_register(0x0202)
        with pytest.raises(ModbusRTUError, match="closed"):
            client.read_register(0x0200)
    finally:
        client.close()
        server.join(1)
    assert len(requests) == 1


def test_wrong_write_echo_is_rejected():
    client, server, _ = exchange_server(frame(struct.pack(">BBHH", 1, 6, 0x0101, 33)))
    try:
        with pytest.raises(ModbusRTUError, match="acknowledgement"):
            client.write_register(0x0103, 1000)
    finally:
        client.close()
        server.join(1)


def test_close_interrupts_waiting_transaction():
    connection, peer = socket.socketpair()
    client = ModbusRTUClient(connection, 1, 1)
    errors = []

    def read():
        try:
            client.read_register(0x0202)
        except OSError as exc:
            errors.append(exc)

    worker = threading.Thread(target=read)
    worker.start()
    peer.settimeout(1)
    assert peer.recv(8)
    client.close()
    worker.join(0.5)
    peer.close()
    assert not worker.is_alive() and errors
