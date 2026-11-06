# DH AG-95 (`type: dh-ag95`)

The Python package is `lerobot.grippers.dh_ag95`; YAML uses `dh-ag95`. The driver
implements the normal `Gripper` contract (0 closed, 1 open) without importing an
arm SDK. Elite single and bimanual adapters inject a register session before
calling its parameterless `connect()`.

## Parameter provenance

The baseline is the actual `BiDobotNova5DHConfig` overrides in
`bi_dobot_nova5_dh/config_bi_dobot_nova5_dh.py`, not the older standalone
`DHGripperIntegratedConfig` defaults. Reference checkout used for this port:
`/home/olorin/Olorin/01_Elite/bi_dobot_nova5_dh`.

| Parameter | Default | Reference |
| --- | --- | --- |
| slave_id / baudrate | 1 / 115200 | Both Dobot arms; 8N1 tool setup |
| gripper_force | 33% | Both Dobot robot overrides (standalone 15% fails its own 20–100% validation) |
| worker_frequency | 50 Hz, best effort | Both Dobot robot overrides |
| position_poll_frequency | 0 Hz | Both Dobot robot overrides |
| binary_gripper_actions / gripper_open_threshold | true / 0.5 | Dobot robot action mapping |
| init_open / init_timeout | true / 10 s | Dobot initialization |

**The default observation is an acknowledged command, not measured jaw opening.**
With `position_poll_frequency: 0`, normal operation does not issue position reads;
the cache changes only after a successful write acknowledgement. With polling
enabled explicitly, the cache contains actual register `0x0202` values. Startup
without `init_open` seeds it from hardware. `read_status()` always reads hardware,
even when periodic polling is off. Idle disconnection cannot be detected without
IO in the zero-poll mode. This limitation is inherited intentionally from Dobot.

Commands are coalesced at the worker frequency; identical quantized targets do
not cause additional writes. Every distinct register target is sent, including
small continuous moves. There is no configurable deadband or AG-95 speed setting.
Binary mapping lives in this backend and the Elite
adapters return the mapped endpoint in the action record. Set
`binary_gripper_actions: false` explicitly for continuous target positions.

## Elite reference and transport

Primary references:

- [DH AG Modbus manual V2.6](https://www.dh-robotics.com/wp-content/uploads/2022/12/AG%E7%B3%BB%E5%88%97-%E6%93%8D%E4%BD%9C%E8%AF%B4%E6%98%8E%E4%B9%A6MODBUS-%E7%89%88V2.6-%E5%A4%A7%E5%AF%B0%E6%9C%BA%E5%99%A8%E4%BA%BA.pdf), control registers and initialization.
- [Elite official DH plugin](https://docs.elibot.cn/cs/ai/c3250/5cab0/6c81c), downloaded `DH.Gripper-3.1.10.elico` on 2026-10-30.
  `daemon/main.py` uses `/dev/ttyTCI0`, 115200/8N1, a 1 s transaction timeout,
  FC03 register reads and FC16 target writes. Its blocking completion check
  sleeps 0.2 s; this is **not** a teleoperation send rate. The plugin supports
  many DH models, so its generic speed register does not establish AG-95 speed control.
- [Elite SDK startToolRs485](https://github.com/Elite-Robots/Elite_Robots_CS_SDK/blob/main/source/Elite/EliteDriver.cpp), tool configuration and `socat` forwarding to TCP 54321.

The runtime sends FC03/FC06 RTU frames with CRC through the TCP bridge, retaining
Dobot's single-register operations. There is no Modbus TCP MBAP header. One worker
owns the traffic, transaction locks serialize diagnostics, response length/CRC/
slave/function/write echo are validated, and RTU frame spacing follows Modbus
timing. TCP fragments are accumulated within one deadline.

The approved Elite-specific settings are under the **robot**, not the gripper:

```yaml
tool_rs485:
  tcp_port: 54321
  ssh_password_env: ELITE_SSH_PASSWORD
  transaction_timeout_s: 1.0
```

The tool is powered at 24 V. The 2 s boot allowance comes from Dobot's standalone
open/close probe. SSH/startup deadlines reuse the robot's `connect_timeout_s`.
Both arms use the shared settings on their own robot IP, so both can use port
54321 and slave ID 1. This version expects both controllers' SSH passwords in
the configured environment variable; different per-arm credentials need a later
configuration extension.

`libpyelite` adds only `configureToolRs485` and `setToolVoltage`, releasing the
GIL in the bindings. The former uses the official SDK's secondary setup script,
before native servo starts. The host follows the SDK's `socat` forwarding path
but owns the SSH subprocess and session-specific PID file. This adds bounded
SSH waits and avoids killing/reconfiguring a borrowed bridge. The pinned SDK's
own start/end helpers reuse an existing bridge without reconfiguring it and do
not bound SSH waits. No second EliteDriver is constructed.

The transparent bridge and the official pendant DH plugin must not own the TCI
port simultaneously. Disconnect the plugin before using this backend. A listener
already on the configured TCP port is refused; the adapter does not take it over.
The controller must provide SSH/root access and `socat`; the host needs `ssh` and
`sshpass`. SSH accepts new host keys, but rejects changed ones. A password is
passed in the subprocess environment, not argv or YAML. If the controller is
unreachable during cleanup, the session log reports the failure and the gripper
retains session ownership. Calling `disconnect()` again retries cleanup, even
after the robot handles are gone. If the host process has exited, inspect and
stop the orphan bridge before reconnecting.

Call `disconnect()` in a `finally` block, or bind the transport and use the
gripper as a context manager. Its running worker keeps the gripper alive, so
garbage collection cannot replace explicit shutdown. The destructor only offers
best-effort cleanup for sessions with no running worker.

Controller dependency checking uses the same login shell as bridge startup and
runs before changing tool mode or enabling 24 V. Missing `socat` is reported
explicitly rather than starting a nonexistent process and timing out. The
adapter does not install controller packages or modify firmware.

A TCP connect alone does not prove that the robot has a serial bridge: VPN/TUN
proxies may accept the connection even when the target is not reachable. Check
the selected bench and route before stopping any service. Bench 07's right arm
is `192.168.207.108`; bench 08's right arm is `192.168.8.223`. On the test host,
`ip route get <robot-ip>` should show the intended robot network, not a TUN route.

The [official CS Python SDK guide](https://docs.elibot.cn/cs/88fdd/4949e/1edd4)
documents `root` and the usual initial SSH password `elibot`. This is the
controller OS login, separate from the HMI operation-mode password; use the
administrator's replacement if it has been changed. The adapter does not embed
or automatically try a default password.

## Deliberate corrections to the Dobot implementation

1. Use the tested robot override force (33%), fixing the invalid standalone 15% default.
2. Check all initialization/force/open write acknowledgements. Failed setup closes the session.
3. Keep target revisions across an in-flight write, so the latest target cannot be lost.
4. With reads enabled, never overwrite actual position with a commanded target.
5. Surface worker IO errors to callers, including when polling is disabled. A
   corrupt/ambiguous TCP stream closes immediately; it is not retried blindly.
   Unlike Dobot's controller proxy, RTU read responses have no transaction ID or
   register address. Automatic replay could accept a late reply for another read
   or repeat a full-calibration write. Reconnection is explicit.
6. A blocking initialization move waits for command acknowledgement and actual
   arrival/contact; dropped-object state is an error. It does not use the
   command cache to claim completion. Opening cannot complete on an old contact
   state, and a superseding target cancels the wait. Its default timeout remains
   Dobot's 10 s.
7. Disconnect stops IO and releases the bridge without commanding the jaws open.
   This avoids releasing an object as a side effect of an error/exit.
8. Nonfinite/out-of-range actions raise an error instead of silently clipping.
9. Cleanup retains ownership on an SSH failure so disconnect can be retried;
   both tools close before either driver is released. A camera teardown failure
   does not skip tool cleanup.

Full initialization is otherwise unchanged: read `0x0200`, skip if already 1,
or write `0xA5` to `0x0100` and wait up to `init_timeout`. This moves through the
full stroke; keep it unobstructed. No flash writes or permanent baud/ID changes
are made. No force/velocity/servo tuning beyond the sourced defaults is added.

## Build and test on hardware

Initialize and rebuild the modified `third_party/libpyelite` before running the
robot adapter (`./setup_env.sh --install --elite`). The submodule changes must
travel with the parent repository; an older installed wheel is rejected when
the tool is connected. Install the host helper if missing:

```sh
sudo apt install sshpass
```

The standalone probe prompts for the controller's **root SSH password** when
`ELITE_SSH_PASSWORD` is unset or empty. Input is hidden, empty input is retried,
and Ctrl+C cancels before connecting. The entered value is used only for that
run and is not saved to a file or the parent shell. An existing nonempty
environment value takes precedence. `--ssh-password-env` can select another
variable. Non-interactive runs must set that variable and fail immediately if
it is missing; password input never falls back to visible echo.

For robot/recording processes or to reuse credentials across probe invocations,
set the environment variable explicitly in your terminal:

```sh
if elite_dh_ssh_input=$(python -c 'import getpass, warnings; warnings.simplefilter("error", getpass.GetPassWarning); print(getpass.getpass("Elite controller root SSH password: "))') && [ -n "$elite_dh_ssh_input" ]; then
    export ELITE_SSH_PASSWORD="$elite_dh_ssh_input"
fi
unset elite_dh_ssh_input
```

Failed, cancelled, or empty input leaves any previous environment value intact.

SSH authenticates bridge startup/cleanup, not the 24 V power command. A missing
environment-variable error happens locally before the probe contacts the robot;
it is not an SSH authentication rejection from the controller.

Configure the AG-95 TCP, measured installed tool mass/CoG, and 24 V supply capacity
on the controller. AG-160-95 is specified at 0.8 A rated / 1.5 A peak; matching
connector pin order alone does not establish the supply rating. Add wrist/tactile
cameras explicitly: AG-95 has no Xense USB discovery. Keep
`native_collision_profile: null`; this repository currently requires TacCap
geometry and calibrated measured finger state for station collision checking.

Start with the tool-only probe on an already powered controller (bench-08 right arm shown):

```sh
python examples/elite_dh_ag95.py --robot-ip 192.168.8.223 --action status
python examples/elite_dh_ag95.py --robot-ip 192.168.8.223 --action cycle --cycles 1
```

The probe sends only tool secondary scripts and Modbus traffic. It does not
power/brake-release/move the arm. `status` configures/powers the tool and reads
hardware; it does not issue jaw initialization/position commands (power-on
auto-initialization, if configured in the gripper, is still hardware behavior).
`cycle` initializes if necessary and starts fully open. Each round closes and
opens once at each of 20%, 55%, and 100% force, then pauses for 3 s, including
after the final round. `--cycles` counts these three-force rounds and defaults
to 1. `--cycles 10` therefore performs 30 close/open pairs. The old single
`--force` option is removed.

Each force setting is read back before movement. Each jaw movement waits for
arrival or, when closing, object-gripped feedback. `--hold-s` sets the dwell
after completion and defaults to 1.5 s. Output includes the round, force setting,
commanded target, measured `position`, and `grip_state`. A close can complete
with `grip_state: 2` and nonzero measured opening when an object blocks it.
Soft objects can continue to compress under the configured force. The force
register reports the setting, not a measured force in newtons. Interrupt/exit
preserves the last target.

Then use `recipes/teleop/bi_elite_cs66_rt/diagonal-07-dh-ag95.yaml`, after checking
its bench IPs/poses and pendant tool calibration. It inherits the existing
bench-07 arm settings and retains the pendant's payload rather than copying the
XGripper payload. The same gripper/tool blocks work in recording recipes. This
is a hardware-validation template, not a claim of tested Elite performance.

Verify both controllers independently, full initialization, gripping/drop
feedback, binary action recording, shutdown with an object held, cable loss,
and sustained dual-arm teleoperation. Measure achieved Modbus latency/rate and
native servo continuity. The reference worker's 50 Hz is a best-effort ceiling,
not a guarantee that Elite completes 50 transactions per second.

## Offline checks

Run the protocol, worker, bridge, CLI, optional-import, and recipe tests without
connecting to a robot:

```sh
python -m pytest --noconftest -q \
  tests/robots/test_dh_ag95.py \
  tests/robots/test_elite_dh_ag95.py \
  tests/robots/test_elite_dh_ag95_cli.py \
  tests/robots/test_gripper_optional_sdk_imports.py \
  tests/robots/test_recipe_gripper_blocks.py
```

These tests cover command coalescing, small continuous moves, actual-position
completion, stale contact feedback, explicit shutdown, and retryable bridge
cleanup. They do not replace the hardware checks above.
