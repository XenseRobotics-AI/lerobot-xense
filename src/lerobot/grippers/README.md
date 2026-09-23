# `grippers` — arm-agnostic gripper drivers

Grippers are their own device family, alongside `cameras/` and `motors/`: an arm
holds one (or one per side) and drives it through a small contract, without
knowing which backend is underneath. Layout mirrors `cameras/` — an ABC and a
config registry at the top, one subpackage per backend below.

```
grippers/
  gripper.py        Gripper ABC
  configs.py        GripperConfig — the draccus choice registry
  utils.py          make_gripper_from_config
  usb_topology.py   shared USB-hub helpers (used by both discoveries)
  serial/           SerialGripper   — USB-serial parallel jaw
  taccap/           TaccapFollower  — centric TacCap gripper
```

## The two backends

|                  | `serial`                            | `taccap_follower`                            |
| ---------------- | ----------------------------------- | -------------------------------------------- |
| Hardware         | parallel jaw, USB serial (XGripper) | centric TacCap gripper, FDCAN motor          |
| Control          | position + force/velocity limits    | SDK ForcePositionController (bounded torque) |
| Side resolved by | board-SN parity (odd → left)        | firmware-burned SN                           |
| On its USB hub   | wrist cam + 2 tactile               | wrist cam + 2 GSPS                           |
| SDK              | `xgripper`                          | `xense.taccap`                               |

Both SDKs are optional builds. `make_gripper_from_config` imports only the branch
it selects, so this package stays importable on a host with neither installed;
the error surfaces at `connect()` with rebuild guidance.

## Configuring one

A robot takes a single typed `gripper:` block:

```yaml
robot:
  gripper:
    type: taccap_follower
    close_speed_radps: 3.0 # setpoint-ramp rate during travel (rad/s)
    print_status: true # one live-panel row per gripper
    auto_discover_cameras: true
```

```yaml
robot:
  gripper:
    type: serial
    gripper_v_max: 100.0 # mm/s
    gripper_f_max: 30.0 # N
    gripper_min_pos: 0.0
    gripper_max_pos: 85.0
```

The block is decoded through `GripperConfig`, a `draccus.ChoiceRegistry`. That is
what makes a wrong knob **fail loudly**:

```yaml
gripper:
  type: serial
  undistort_wrist: true # -> DecodingError: unknown field
```

Before this was a typed block the same line was accepted and silently ignored,
because every backend's knobs were flattened onto the arm config as
`gripper_type` + `taccap_*` + `gripper_*`, and only the ones matching the selected
backend were read.

### Bimanual arms

Write the block **once**. Both arms are always a matched pair, so the arm config
clones it per side with `side` stamped in — that stamp is what lets each backend
work out which physical unit is which. Drop one side with
`left_use_gripper: false` / `right_use_gripper: false` without touching the block.

### Single arms

There is no side to infer, so a `serial` gripper needs `side` (or `port` / `sn`)
written in the block for the driver to find its board. `SerialGripper.connect()`
raises if it cannot resolve one.

## Cameras

`auto_discover_cameras` lives on the gripper because what is discoverable is a
property of the backend. Both are the same shape of device — one USB hub carrying
the gripper board, its wrist camera and its two tactile sensors — so both default
to **on**: the recipe pins only `head` and the robot sniffs the rest at connect.

Finding the _gripper_ is separate, always on, and not switchable: `serial` resolves
its side from the board SN's parity (odd → left), `taccap_follower` from the
firmware-burned SN. That is why no recipe pins a gripper SN.

`enable_tactile` lives here for the same reason: what a gripper carries is a
property of the gripper. Swap a TacCap for an XGripper and the sensors on the hub
change; the arm does not.

`undistort_wrist` and `fisheye_balance` are **taccap-only**, because the wrist
lens' intrinsics are burned into that gripper's own MCU flash and travel with it.
The serial family has no such record and therefore no such fields — writing them
on an XGripper block is refused at parse rather than quietly ignored. Both are
applied to the wrist camera _as it is discovered_, so `undistort_wrist` requires
`auto_discover_cameras`; a recipe that pins its cameras by hand sets `undistort:`
on the wrist camera block instead, next to the resolution it constrains.

When a gripper or its cameras do not show up, run the discovery by hand — it
prints what it found and how it classified each side:

```python
from lerobot.grippers.serial import discover_serial_gripper_sides  # port + board SN per side
from lerobot.grippers.serial import discover_serial_gripper_cameras  # + wrist/tactile on each hub
from lerobot.grippers.taccap import discover_taccap_sides  # the TacCap equivalent
```

Note that the gripper object itself never returns image or tactile data. Those
sensors are ordinary entries in the robot's `cameras`, whether pinned in the recipe
or injected by discovery.

## The contract

```python
class Gripper(abc.ABC):
    is_connected: bool                      # property
    def connect() -> None
    def disconnect() -> None
    def get_gripper_position() -> float     # normalized, 0.0 = closed, 1.0 = open
    def set_gripper_position(pos: float)    # normalized, non-blocking

    # Default implementation: command, then poll until reached. Override only if
    # the SDK offers a genuine blocking move.
    def initialize_gripper_position(pos, tolerance=0.03, timeout=3.0)
```

`0.0 = closed, 1.0 = open` is uniform across backends — an arm never has to know
the underlying units (mm of jaw travel, or radians of motor angle).
