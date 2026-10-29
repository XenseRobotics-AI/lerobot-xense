# CLI recipes

YAML configs for `lerobot-teleoperate` and `lerobot-record`. Instead of
copy-pasting a long command, store the parameters in a file here and pass its
path — both CLIs accept `--config_path` (they share the same draccus parser).

```bash
# Teleoperate
lerobot-teleoperate --config_path=recipes/teleop/bi_elite_cs66_rt/diagonal-08-xgripper.yaml

# Record
lerobot-record --config_path=recipes/record/bi_elite_cs66_rt/test-xgripper.yaml
```

CLI flags still work and **override** the YAML, so you don't edit the file for
one-off tweaks:

```bash
lerobot-record --config_path=recipes/record/bi_elite_cs66_rt/test-xgripper.yaml \
    --dataset.num_episodes=1 --resume=true

lerobot-teleoperate --config_path=recipes/teleop/bi_elite_cs66_rt/diagonal-08-xgripper.yaml --dryrun=true
```

## Layout

Split by CLI, then by robot type:

```
recipes/
  teleop/<robot_type>/<variant>.yaml    -> lerobot-teleoperate --config_path=…
  record/<robot_type>/<task>.yaml       -> lerobot-record      --config_path=…
```

```
recipes/
  teleop/
    bi_elite_cs66_rt/diagonal-08-{taccap,xgripper}.yaml   # one recipe per physical bench
    bi_flexiv_rizon4_rt/forward-04-{taccap,xgripper}.yaml
  record/
    bi_elite_cs66_rt/test-{taccap,xgripper}.yaml
    bi_flexiv_rizon4_rt/assemble_box-{taccap,xgripper}.yaml
```

## What goes in a recipe

A recipe lists **only the frequently-changed knobs**; every other field falls
back to its dataclass default, so recipes stay short. The current per-robot set
is the minimal one operators actually tune — mount + IPs, control mode, a servo
knob or two, gripper enable + force, the tactile toggle, plus the dataset fields
and top-level loop flags. To touch a rarely-changed parameter, add its line to
the recipe or pass a one-off `--flag` override; until then the dataclass default
applies. (Removing a line only changes behavior if its default differs from the
value you had — verify before trimming.)

The `robot:` and `teleop:` blocks are **identical** between a robot's record and
teleop recipe (same hardware, same teleop device) — keep the two in sync.

### Rebot platform 09: DM + TacCap00

Platform 09 uses TacCap00. Its single-arm and bimanual recipes are named
`dm-taccap00-09.yaml` under both `record/` and `teleop/`; these replace the old
`dm-taccap.yaml` filenames. The paired recipes use the same robot hardware settings.

Run from the `lerobot-xense` repository root:

```bash
lerobot-record \
  --config_path=recipes/record/bi_rebot_b601_rt/dm-taccap00-09.yaml \
  --dataset.repo_id=Xense/bi_dm_taccap00_09_pick_cube_001 \
  --dataset.single_task="Pick up the cube and place it in the box" \
  --dataset.num_episodes=10

lerobot-teleoperate \
  --config_path=recipes/teleop/bi_rebot_b601_rt/dm-taccap00-09.yaml
```

The external `head` camera is a RealSense D415, serial **249322063436** as
reported by librealsense, recording RGB at 640×480 and 30 Hz. Depth is disabled.
On the bimanual robot, `robot.cameras.head` is shared and recorded once as
`observation.images.head`; wrist and tactile cameras are still discovered from
each gripper. Adding `head` changes the dataset features: start a new dataset
instead of resuming one recorded without it.

Each arm explicitly sets `urdf_path` and `end_frame`. Paths in these recipes are
relative to the repository root and point into `third_party/libpyrebot/rebot_rt/assets`.
An absolute path to a separately installed/custom URDF also works. The selected
model is used for both kinematics and gravity compensation. With
`gravity_use_joint_offsets: true`, its matching `<URDF stem>_offset.yaml` must be
beside the URDF.

For another platform, copy and rename a recipe, update the robot/dataset IDs and
camera serial, and set **each arm's** URDF and TCP frame to match its motor family,
mount and actual payload. TacCap00 is platform 09's assembly, not a property of
all DM arms. Legacy configurations that omit `urdf_path` still use the driver's
family/gripper fallback; new platform recipes should specify the model explicitly.

### Rebot RS + TacCap05

The RS single-arm and bimanual recording/teleoperation recipes are named
`rs-taccap05.yaml`. Each arm explicitly sets
`urdf_path: third_party/libpyrebot/rebot_rt/assets/RS_TacCap05/urdf/rebot_b601_rs_taccap_v2_gripper.urdf`
and `end_frame: taccap_eef_link`. Run from the repository root, or use an absolute
URDF path. For a different mount or payload, replace the model path and TCP frame
in both the recording and teleoperation recipes.

### Rebot DM serial connection order

The DM TacCap recipes use `channel: auto:first` for the right arm and
`channel: auto:second` for the left. Connect the **right DM bridge first**, wait
until it appears under `/dev/ttyACM*`, then connect the left bridge and start
recording or teleoperation. The single-arm recipe selects the first bridge.

The driver filters HDSC bridges by USB vendor/product `2e88:4603`, excluding
TacCap and other serial devices, and sorts by udev's `USEC_INITIALIZED` timestamp.
It does not sort ACM numbers or bind to USB ports. Both arms are resolved from
one snapshot before either controller connects, and startup logs show the mapping.
Missing devices, unavailable/tied timestamps and more than two matching bridges
stop automatic selection instead of falling back to ACM numbering. No udev rules
or root privileges are needed; normal serial-device permissions still apply.

This convention does not identify the physical arms: after rebooting with both
bridges attached, reconnect **both** in right-then-left order before starting.
Also repeat that sequence if either bridge is unplugged; reconnecting only the
right bridge while leaving the left attached reverses the order. Keep the USB
connections unchanged while a program is running. Explicit `/dev/...` paths and
SocketCAN channels remain supported, including for other bridge hardware.

### Grippers

The gripper is one typed block, not a pile of flat knobs:

```yaml
robot:
  gripper:
    type: serial # or taccap_follower
    gripper_v_max: 100.0
    gripper_f_max: 30.0
  left_use_gripper: true # bimanual only — drop one side without deleting the block
  right_use_gripper: true
```

A bimanual arm writes the block once; each side gets a copy with `side` stamped in,
which is what lets the backend resolve which physical unit is which. Because the
block is decoded through the gripper registry, writing `kp:` under `type: serial`
(or misspelling a field) fails the parse instead of being quietly dropped. The two
backends and their fields are documented in
[`../src/lerobot/grippers/README.md`](../src/lerobot/grippers/README.md).

**A recipe is self-contained.** It carries the bench hardware — cameras, robot
SNs/IPs, mount geometry (`*_mount_*_deg`, `world_rotation`), `local_ip`, home and
start poses — alongside the run's tuning. One file is everything a run needs.

Precedence is lowest to highest:

```
dataclass default  <  recipe YAML  <  CLI --robot.xxx=
```

The cost of self-containment is duplication: two recipes on the same bench each
hold their own copy of that bench's hardware (today `diagonal-08` is shared by
`teleop/bi_elite_cs66_rt/diagonal-08-{taccap,xgripper}.yaml` and `record/bi_elite_cs66_rt/test-{taccap,xgripper}.yaml`).
**When a bench changes — a swapped camera, a re-mounted arm, a new controller
address — grep every recipe naming that bench and update all of them.** A missed
copy does not fail loudly; it silently records with the wrong serial.

### Cameras

A recipe pins only `head`. Both gripper backends are the same shape of device — a
USB hub carrying the gripper board, its wrist camera and its two tactile sensors —
so each sniffs its own cameras off that hub at connect
(`gripper.auto_discover_cameras`, on by default for both). Nothing a recipe could
pin is something discovery cannot work out, and a pinned SN goes stale the moment
a sensor is swapped.

Set `auto_discover_cameras: false` and pin all seven if a bench genuinely needs it
(e.g. something else shares the hub, which discovery refuses to guess through).

Discovered cameras are keyed `<side>_wrist` and `<side>_tactile_<finger>` — the
arm the gripper is on, then which jaw the pad sits on, e.g. `left_tactile_right`.
Both halves come from serial parity (odd → left, even → right), never from USB
enumeration order, so re-cabling a sensor does not move its key and these names
match what the sister repo `xense-taccap-lerobot` records. Pin the same names if
you turn discovery off.

Tactile cameras take their tuned exposure from the `XenseTactileCameraConfig`
default (`camera_properties`), so a recipe entry stays short. Pass
`camera_properties: null` on a camera to fall back to its xpack template.

> **History.** Bench hardware used to live in `stations/<robot_type>/<name>.yaml`,
> selected by a `bi_mount_type` field. That layer was removed at 3b964bc6`^`;
> the station files (including two benches that had no recipe, `forward-05` and
> `forward-dewu`, now recipes of their own) are recoverable from git history there.

Flexiv `force_control_frame` is a C++ enum with no YAML decoder, so it is set in
code (defaults to `WORLD`) and omitted from the recipe.

## YAML format

The YAML mirrors the CLI flags exactly: a dotted `--robot.servoj_gain=300`
becomes nested `robot: { servoj_gain: 300 }`, and `type:` is the discriminator
that selects the robot / teleop class. Enums use their string value
(`control_mode: cartesian_servo`).

## Naming convention

- `teleop/<robot_type>/<variant>.yaml` — one recipe **per physical bench**;
  variant = mount + bench number, e.g. `diagonal-08-xgripper.yaml`, `forward-05-taccap.yaml`.
- `record/<robot_type>/<task>.yaml` — one per dataset/campaign, e.g.
  `assemble_box-<family>.yaml`. Keep a `test-<family>.yaml` smoke-test per robot (2 short
  episodes, `push_to_hub: false`).

## Why this over copy-pasting from a markdown file

- **Single source of truth** — the recipe _is_ the runnable artifact.
- **Provenance for free** — recipes are committed, so `git log recipes/` shows
  exactly which parameters produced each dataset and when.
- **No silent mistakes** — no forgetting `--resume=false` or pasting a wrong SN.
- **Diff-friendly** — bumping `num_episodes` is a one-line, reviewable change.

The full flag reference for every robot still lives in
[`../src/lerobot/scripts/client_commands.md`](../src/lerobot/scripts/client_commands.md).
