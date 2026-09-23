# 🎯 Project Overview

🤗 This repository is a fork of [`lerobot`](https://github.com/huggingface/lerobot)
by XenseRobotics, used for Xense's multimodal tactile data acquisition system.
This branch tracks **upstream lerobot v5.1**, with Xense-specific robots
(Flexiv Rizon4 RT, Elite CS66 RT, and ARX5 — each single-arm and bimanual;
plus TacCap tactile grippers), teleoperators (Pico4 VR,
dual SpaceMouse, TRLC leader, gamepad) and tactile cameras
layered on top. For generic lerobot usage (datasets, policies, training
scripts) refer to the
[upstream README](https://github.com/huggingface/lerobot#readme).

## 🔧 Installation

Tested on Ubuntu 22.04 and 24.04, NVIDIA driver ≥ 570.144. Use
[`Mamba`](https://github.com/conda-forge/miniforge?tab=readme-ov-file#install)
(strongly recommended over plain conda — it's much faster on the
robostack-staging channel that ships ROS Humble + SOEM). v5.1 pins
**Python 3.12** and **PyTorch ≥ 2.2** with CUDA 12.8.

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

### 📦 Environment Setup

**Step 1:** 📂 Clone the repository with all submodules:

```bash
git clone \
  --recurse-submodules \
  https://github.com/XenseRobotics-AI/lerobot-xense.git
cd lerobot-xense
```

> If you already cloned without submodules, initialize them manually:

> ```bash
> git submodule update --init --recursive --progress
> ```

> **Optional — internal network (GitLab mirror).** The default clone above uses
> the public GitHub mirrors (`git@github.com:XenseRobotics-AI/*`). Company-network
> developers can instead point the submodules at the internal GitLab server;
> the internal URLs live in [`.gitmodules.gitlab`](.gitmodules.gitlab) and a
> helper repoints your **local** remotes without touching the committed
> `.gitmodules` (every submodule pin exists on both remotes):
>
> ```bash
> XENSE_GITLAB_HOST=<host> scripts/submodule-remote.sh gitlab   # `github` switches back
> git submodule update --init --recursive
> ```
>
> The GitLab host is not committed — this repository is public, so the address
> comes from `XENSE_GITLAB_HOST` instead. Ask a teammate for the value and export
> it in your shell profile.

This repository uses `third_party/` git submodules to manage hardware SDK dependencies:

| Submodule                                | Installed package                               |
| ---------------------------------------- | ----------------------------------------------- |
| `third_party/ARX5_SDK`                   | `pyarx`                                         |
| `third_party/libpyflexiv`                | `flexiv_rt`                                     |
| `third_party/XGripper`                   | `xgripper`                                      |
| `third_party/elite-robots-cs-sdk`        | Elite CS C++ SDK (builds `elite_cs_sdk`)        |
| `third_party/elite-robots-cs-sdk-python` | `elite_cs_sdk` (Elite CS Python bindings)       |
| `third_party/taccap-gripper`             | `xense.taccap` (TacCap UMI tactile gripper SDK) |

> `xensesdk` is **not** a submodule — it is installed from PyPI (`xensesdk==2.1.2`,
> the published cp312 manylinux wheel, which bundles the patched `libxense_c.so`
> flash reader). The Elite Python SDK is built against the local
> `third_party/elite-robots-cs-sdk` C++ submodule (no network fetch of the C++ source).

> **`xensevr_pc_service_sdk` (Pico4 teleop/tracker) has no submodule either.** Its
> pybind11 sources live in-repo under
> `src/lerobot/teleoperators/pico4/xensevr-pc-service-pybind/`, and the C SDK
> they link against — `PXREARobotSDK.h` plus `libPXREARobotSDK.so` — is copied
> straight out of the `xensevr-pc-service` `.deb` that `--install` fetches
> (see Step 3). The daemon in that `.deb` is what the teleop talks to at
> runtime anyway, so it was never optional; carrying a 33 MiB checkout of the
> service's Qt tree and prebuilt gRPC archives just to rebuild a library we
> were already downloading was the part that was.

**Step 2:** 🐍 Create and activate the mamba environment:

```bash
bash ./setup_env.sh --mamba lerobot-xense
mamba activate lerobot-xense
```

> The default env name baked into `conda_environment.yaml` is
> `lerobot-xense-py312`. You can pass a different name to `--mamba`,
> but the rest of this README and the openpi project assume
> `lerobot-xense-py312`.

**Step 3:** 📦 Install LeRobot-Xense and all hardware SDK bindings:

```bash
bash ./setup_env.sh --install
```

This step will:

- Update the conda environment from `conda_environment.yaml`
- Install the main package from `pyproject.toml`
- Install `xensesdk` from PyPI (`xensesdk==2.1.2`)
- Install the XenseVR PC Service daemon from its `.deb` (~116 MB, fetched from the
  [v0.2.1 release](https://github.com/XenseRobotics-AI/XenseVR-PC-Service/releases/tag/v0.2.1)
  into `/opt/apps/roboticsservice`; override with `XENSEVR_DEB_URL`, or point
  `XENSEVR_DEB` at a local file for offline installs) — only with `--pico4` / `--bi_pico4` or a full install
- Build and install all `third_party` SDK packages: `pyarx`, `flexiv_rt`, `xensevr_pc_service_sdk` (built against the `.deb`'s client SDK), `xgripper`, `elite_cs_sdk` (Elite CS — built from the C++ + Python submodules), and `xense.taccap` (TacCap UMI gripper)
- Configure SpaceMouse udev rules and HID permissions automatically

> You will be prompted for `sudo` password during installation (for ARX5 real-time capability and udev rules).

#### 🎛️ Selective hardware install (build only the SDKs you need)

By default `--install` builds **every** hardware SDK. On a station that only uses one
or two devices, pass per-hardware-family selectors after `--install` to build
**core + only those** SDKs — faster, and it skips SDKs you can't (or don't want to)
build on that host:

```bash
# core + Flexiv + TacCap only (arms auto-include the xense gripper stack)
bash ./setup_env.sh --install --flexiv --taccap

# core + Elite only
bash ./setup_env.sh --install --elite

# core only — no hardware SDK bindings
bash ./setup_env.sh --install --core

# list every selector
bash ./setup_env.sh --install --help
```

| Selector                  | Builds                                         | Robots / teleoperators enabled            |
| ------------------------- | ---------------------------------------------- | ----------------------------------------- |
| `--flexiv`, `--bi_flexiv` | `flexiv_rt` (+ `xense`)                        | `flexiv_rizon4_rt`, `bi_flexiv_rizon4_rt` |
| `--elite`, `--bi_elite`   | `elite_cs_sdk` (+ `xense`)                     | `elite_cs66_rt`, `bi_elite_cs66_rt`       |
| `--taccap`, `--bi_taccap` | `xense.taccap` (+ `xense`)                     | `taccap_follower` gripper (on any arm)    |
| `--xense`                 | `xensesdk` + `xgripper` (XGripper)             | `serial` gripper + tactile sensors        |
| `--arx5`, `--bi_arx5`     | `pyarx`                                        | `arx5_follower`, `bi_arx5`                |
| `--pico4`, `--bi_pico4`   | `xensevr_pc_service_sdk` (+ PC Service `.deb`) | `pico4`, `bi_pico4` teleop                |
| `--spacemouse`            | `pyspacemouse`                                 | `spacemouse` teleop                       |
| `--dynamixel`, `--trlc`   | `dynamixel-sdk`                                | `trlc_leader`, `bi_trlc` teleop           |
| `--all`                   | everything (explicit)                          | —                                         |
| `--core`, `--none`        | nothing (core only)                            | —                                         |
| _(no selector)_           | everything (default, backward compatible)      | —                                         |

Notes:

- **No selector = install all** — the existing `bash ./setup_env.sh --install` behavior is unchanged.
- **Arms auto-include `xense`** — `--flexiv` / `--elite` / `--taccap` also build the xense gripper stack, because those arms drive xense grippers.
- **Post-install verification** only checks the SDKs you selected.
- **Code stays compatible with a partial install.** `import lerobot` and the CLIs
  (`lerobot-teleoperate`, `lerobot-record`, …) start fine even when an SDK is absent —
  a device whose SDK isn't installed simply won't appear as a `--robot.type` /
  `--teleop.type` choice (and only errors, with a rebuild hint, if you try to construct it).

**Step 4:** ✅ Verify the installation. These checks assume a **full** `--install`; on a
selective install (e.g. `--flexiv --taccap`) only the SDKs you selected are built, so
verify just those — the installer already prints a per-SDK verification summary at the end.

```bash
python -c 'import pyarx; print("pyarx OK ->", pyarx.__file__)'
python -c 'import flexiv_rt; print("flexiv_rt OK ->", flexiv_rt.__file__)'
python -c 'import xensevr_pc_service_sdk; print("xensevr_pc_service_sdk OK ->", xensevr_pc_service_sdk.__file__)'
python -c 'import xensesdk; print("xensesdk OK ->", xensesdk.__file__)'
python -c 'import xgripper; print("xgripper OK ->", xgripper.__file__)'
python -c 'import elite_cs_sdk; print("elite_cs_sdk OK ->", elite_cs_sdk.__file__)'
python -c 'import xense.taccap; print("xense.taccap OK ->", xense.taccap.__file__)'
```

**Step 5:** 📌 **Note on FFmpeg / video:** v5.1 no longer pins `ffmpeg`
through conda (the robostack ICU pin conflicted with newer ffmpeg
builds). Video encoding/decoding is handled by `torchcodec` + `av`
wheels installed via `setup_env.sh --install`. If you need a system
ffmpeg with `libsvtav1`, install it separately (apt or upstream
static build):

```bash
# Optional: verify torchcodec wheel is loadable
python -c 'import torchcodec; print("torchcodec OK ->", torchcodec.__version__)'
```

### ARX5 Real-time Thread Permissions

The ARX5 SDK requires `CAP_SYS_NICE` on the Python interpreter for real-time CAN thread scheduling. This is handled by `setup_env.sh --install`, but can be set manually:

```bash
PY_EXE=$(python -c 'import sys, os; p = sys.executable; print(os.path.realpath(p))')
sudo setcap cap_sys_nice+ep "$PY_EXE"
getcap "$PY_EXE"  # should show: cap_sys_nice+ep
```

## 🚀 Running teleop & record (recipes)

Teleoperation and recording are driven by **recipe** YAML files under
[`recipes/`](recipes/) (see [`recipes/README.md`](recipes/README.md) for the full
guide). Pass one with `--config_path`:

```bash
lerobot-teleoperate --config_path=recipes/teleop/bi_elite_cs66_rt/diagonal-07-taccap.yaml
lerobot-record     --config_path=recipes/record/bi_flexiv_rizon4_rt/assemble_box-xgripper.yaml
```

A recipe is **self-contained**: it carries both the bench hardware (controller
IPs/SNs, camera SNs, mount geometry, per-arm home/start poses) and the run's
tuning (control mode, servo gains, guards, gripper force, dataset fields). One
file is everything a run needs; adding a bench is a new recipe, no Python change.

```bash
ls recipes/teleop/bi_flexiv_rizon4_rt/  # forward-04, forward-05, forward-dewu, diagonal-02
ls recipes/teleop/bi_elite_cs66_rt/     # diagonal-07, diagonal-08
```

Precedence is `dataclass default < recipe < CLI`, so an explicit
`--robot.left_robot_sn=…` overrides the recipe for a one-off run.

**Gripper (`gripper:`).** Every robot takes one typed gripper block; a bimanual arm
writes it once and both sides get a copy with `side` stamped in (the two are always
a matched pair). Two backends:

- `serial` — `XenseSerialGripper`, a parallel jaw over USB serial. Left/right are
  **auto-discovered by board-SN parity** (odd SN → left, even SN → right) at
  connect, so no gripper SN is configured.
- `taccap_follower` — `xense.taccap` FollowerGripper, the centric TacCap gripper
  (MIT impedance). Left/right resolved from the firmware-burned SN, and its wrist
  - GSPS tactile cameras auto-discovered at connect.

```yaml
robot:
  gripper:
    type: taccap_follower
    close_speed_radps: 3.0 # setpoint-ramp rate during travel (rad/s)
    auto_discover_cameras: true # sniff wrist + tactile off this gripper's hub
    enable_tactile: true
    undistort_wrist: true # rectify the wrist fisheye from the MCU's intrinsics
    fisheye_balance: 0.0 # 0 = calibrated focal length; 1 = 0.70x, widest FOV
```

What the gripper carries is configured on the gripper, not on the arm — swap a
TacCap for an XGripper and the sensors on the hub change while the arm does not.
`undistort_wrist` and `fisheye_balance` are taccap-only, because the wrist lens'
intrinsics live in that gripper's own MCU flash.

The block is decoded through the gripper registry, so a knob belonging to the other
backend — or a typo — is **rejected at parse time** rather than silently ignored.
That is not cosmetic here: `undistort_wrist` on an XGripper block, or with
`auto_discover_cameras` off, is refused rather than accepted and quietly ignored.
See [`src/lerobot/grippers/README.md`](src/lerobot/grippers/README.md) for the full
field lists and the driver contract.

## 🐭 SpaceMouse Teleoperation System

This project includes advanced SpaceMouse support with both single and dual-device modes for precise robotic control.

### Dependencies

**System Requirements:**

- Ubuntu 22.04 / 24.04 (tested) or other Linux distributions
- Python 3.12+
- libhidapi (installed via apt)

**Python Packages:**

- `pyspacemouse` - Modern cross-platform SpaceMouse library
- `hidapi` - Python wrapper for HID API
- `easyhid` - Easy-to-use HID library (dependency of pyspacemouse)

All Python dependencies are automatically installed by `setup_env.sh --install`.

### Permissions Setup

SpaceMouse requires proper udev rules to allow non-root access. This is configured automatically by `setup_env.sh --install` (see **Step 3** in the Installation section above).

### Testing Your SpaceMouse

After installation and permissions setup, test your SpaceMouse:

```bash
# Basic functionality test (prints real-time 6-DoF values)
python src/lerobot/teleoperators/spacemouse/examples/01_basic.py

# Device discovery (lists connected SpaceMice by path/serial)
python src/lerobot/teleoperators/spacemouse/examples/05_discovery.py
```

The test script will display real-time position (x, y, z) and orientation (roll, pitch, yaw) values as you move the SpaceMouse.

> 📝 **Note:** If you're using a 3Dconnexion Universal Receiver (wireless), you may see multiple devices listed (e.g., 14 "UniversalReceiver" entries). This is normal - the receiver exposes multiple HID interfaces for different functions. PySpaceMouse will automatically select the correct interface for 6-DoF input.

### Features

- ✅ **Modern PySpaceMouse Integration**: Uses PySpaceMouse library for cross-platform SpaceMouse support
- ✅ **No System Services Required**: Direct HID communication, no need for spacenavd daemon
- ✅ **Single Device Mode**: Traditional 6-DoF control with one SpaceMouse
- ✅ **Dual Device Mode**: Advanced left/right hand coordination for complex manipulation
- ✅ **Flexible Axis Assignment**: Configure which device controls position vs orientation
- ✅ **Independent Sensitivity**: Per-device sensitivity settings for optimal control

### Single Device Configuration

```python
from lerobot.teleoperators.spacemouse import SpacemouseConfig, SpacemouseTeleop

# Standard single SpaceMouse setup (default)
config = SpacemouseConfig(
    pos_sensitivity=0.8,  # Position control sensitivity
    ori_sensitivity=1.5,  # Orientation control sensitivity
    deadzone=0.1,  # Deadzone threshold
    frequency=200,  # Polling frequency (Hz)
)

teleop = SpacemouseTeleop(config)
```

### Dual Device Configuration

Perfect for complex robotic tasks requiring precise position and orientation control:

```python
from lerobot.teleoperators.spacemouse import SpacemouseConfig, DeviceConfig

# Left hand controls position, right hand controls orientation
config = SpacemouseConfig(
    multi_device_mode=True,
    left_device=DeviceConfig(
        device_index=0,
        enabled_axes=(True, True, True, False, False, False),  # X, Y, Z position only
        pos_sensitivity=0.8,
        ori_sensitivity=0.0,  # Disabled
    ),
    right_device=DeviceConfig(
        device_index=1,
        enabled_axes=(False, False, False, True, True, True),  # Roll, pitch, yaw only
        pos_sensitivity=0.0,  # Disabled
        ori_sensitivity=1.5,
    ),
)

teleop = SpacemouseTeleop(config)
```

### Example Configurations

See [`examples/09_custom_config.py`](src/lerobot/teleoperators/spacemouse/examples/09_custom_config.py) and [`examples/03_multi_device.py`](src/lerobot/teleoperators/spacemouse/examples/03_multi_device.py) (under `src/lerobot/teleoperators/spacemouse/`) for complete configuration examples including:

- Position/Orientation split control
- Dual-arm robot control
- Fine/Coarse movement control

### Use Cases

- 🤖 **Dual-Arm Robots**: Independent control of two robotic arms
- 🎯 **Precision Manipulation**: Decouple position and orientation control for fine tasks
- 🔄 **Complex Assembly**: Left hand positions, right hand orients components
- 🏭 **Industrial Applications**: Enhanced ergonomics and control precision

### Hardware Requirements

- **Single Mode**: Any 3Dconnexion SpaceMouse device
- **Dual Mode**: Two identical SpaceMouse devices (e.g., two SpaceNavigators)

### Supported Devices

All 3Dconnexion devices supported by PySpaceMouse:

- SpaceNavigator
- SpaceMouse Pro
- SpaceMouse Wireless
- SpaceMouse Compact
- And more...

## 🤖 Recording on Flexiv Rizon4

### Bimanual Flexiv Rizon4 RT + BiPico4 Record Controls

For `lerobot-record` with `--robot.type=bi_flexiv_rizon4_rt --teleop.type=bi_pico4`, the controller buttons are mapped as follows:

| Controller button | Keyboard equivalent | Action                                                               |
| ----------------- | ------------------- | -------------------------------------------------------------------- |
| Right `A`         | `go_start`          | Reset both arms to start pose (RT non-blocking, recording continues) |
| Left `X`          | `rerecord_episode`  | Discard current episode and re-record                                |
| Left `Y`          | `exit_early`        | Finish current episode early                                         |
| Right `B`         | `stop_recording`    | Stop the recording session                                           |

Button state is refreshed via `BiPico4.poll_buttons()` at the top of each loop iteration, before event checks. Keyboard events and controller buttons are unified into the same `events[]` checks — both are equal-priority input sources.

During RT reset, the record loop keeps running: observations are still sampled, teleop actions are still read, and the teleop pose is re-synced to the robot once the reset trajectory finishes.

## Record Loop Implementation (`flexiv_rizon4_rt_record_loop`)

This section documents the dataset construction logic in `flexiv_rizon4_rt_record_loop`, which handles both normal teleoperation recording and the RT reset trajectory recording for `bi_flexiv_rizon4_rt + bi_pico4`.

### Loop Structure

```
while timestamp < control_time_s:
    poll_buttons()              # lightweight button refresh (no pose computation)

    # Unified event checks — keyboard OR controller button, symmetric
    stop_recording  / B  →  break
    rerecord        / X  →  break
    exit_early      / Y  →  break
    go_start        / A  →  reset_to_initial_position(), recording continues

    get_observation()
    check robot_is_moving + sync teleop if reset just finished
    get_action()
    send_action / dataset write
```

### State Variables

| Variable                 | Role                                                                                                                                                                   |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `reset_triggered`        | Per-frame flag. Set `True` the frame reset is triggered. Skips `send_action` and dataset write for that frame only. Resets to `False` at the start of every iteration. |
| `prev_rt_moving`         | Edge-detection flag. Set `True` while `robot.rt_moving` is `True`. Cleared to `False` when movement stops, triggering one call to `_sync_rt_teleop_to_robot_pose()`.   |
| `prev_observation_frame` | Holds the previous frame's observation. Used by shifted-frame logic to pair `obs[t-1]` with the robot's actual position at `obs[t]` as the action.                     |

### Three Frame Modes

#### 1. Normal teleoperation (`robot_is_moving=False`, `reset_triggered=False`)

```
robot.send_action(teleop_action) → sent_action
dataset: { obs[t],  action = sent_action[t] }   # direct frame
prev_observation_frame = obs[t]
```

#### 2. Reset trigger frame (`reset_triggered=True`)

```
robot.reset_to_initial_position()   # C++ RT thread takes over arm control
send_action  → skipped
dataset      → skipped
prev_observation_frame = obs[T]     # saved as anchor for next iteration
```

`obs[T]` is intentionally not written to the dataset. It is used as `prev_observation_frame` for the first shifted frame on the next iteration, so it appears exactly once — without this skip it would appear twice (once as a direct frame, once as the prev of the first shifted frame).

#### 3. RT reset in progress (`robot_is_moving=True`)

```
send_action  → skipped (C++ RT thread drives the arm autonomously)
current_as_action = { key: obs[t][key] for key in robot.action_features }
# robot.action_features = left/right TCP pose (9D each) + gripper (1D each) = 20D total
# Iterates action_features keys only — image keys in obs[t] are excluded automatically.
dataset: { obs[t-1],  action = current_as_action }   # shifted frame
prev_observation_frame = obs[t]
```

The action is extracted from the current observation using the same keys as `robot.action_features`. This records where the robot actually moved to, not what the teleop commanded — the same shifted-frame convention used by `bi_arx5_record_loop`.

### Per-Frame Decision Table

| Frame                           | `reset_triggered` | `robot_is_moving` | `send_action` | Dataset write                        | `prev_obs` updated to |
| ------------------------------- | ----------------- | ----------------- | ------------- | ------------------------------------ | --------------------- |
| T-1 (normal teleop)             | False             | False             | ✓             | `{obs[T-1], action[T-1]}` direct     | obs[T-1]              |
| **T (reset triggered)**         | **True**          | **False**         | **skipped**   | **skipped**                          | **obs[T]**            |
| T+1 (RT moving)                 | False             | True              | skipped       | `{obs[T], state_20d[T+1]}` shifted   | obs[T+1]              |
| T+2 (RT moving)                 | False             | True              | skipped       | `{obs[T+1], state_20d[T+2]}` shifted | obs[T+2]              |
| …                               | False             | True              | skipped       | shifted                              | …                     |
| N+1 (reset done, teleop synced) | False             | False             | ✓             | `{obs[N+1], action[N+1]}` direct     | obs[N+1]              |

`state_20d[t]` = `{k: obs[t][k] for k in robot.action_features}` — left/right TCP pose (9D each) + gripper (1D each), image keys excluded.

### Complete Frame Sequence Around a Reset

```
frame T-2  normal teleop  →  dataset: { obs[T-2], action[T-2] }
frame T-1  normal teleop  →  dataset: { obs[T-1], action[T-1] },  prev=obs[T-1]
frame T    reset trigger  →  dataset: skipped,                     prev=obs[T]
frame T+1  rt_moving      →  dataset: { obs[T],   state_20d[T+1] },  prev=obs[T+1]
frame T+2  rt_moving      →  dataset: { obs[T+1], state_20d[T+2] },  prev=obs[T+2]
  ...
frame N    rt_moving      →  dataset: { obs[N-1], state_20d[N] },    prev=obs[N]
frame N+1  reset done     →  _sync_rt_teleop_to_robot_pose()
           normal teleop  →  dataset: { obs[N+1], action[N+1] }
```

### Post-Reset Teleop Sync

When `prev_rt_moving` transitions `True → False` (frame N+1), `_sync_rt_teleop_to_robot_pose()` is called once. This reads the robot's current TCP pose (now at start position) and calls `teleop.reset_to_pose()`, updating the Pico4's internal `_start_pos` reference. Without this sync the teleop would compute position deltas from the pre-reset pose, causing the arm to jump on the first grip after reset.

## 🔑 The `LeRobotDataset` format

A dataset in `LeRobotDataset` format is very simple to use. It can be loaded from a repository on the Hugging Face hub or a local folder simply with e.g. `dataset = LeRobotDataset("lerobot/aloha_static_coffee")` and can be indexed into like any Hugging Face and PyTorch dataset. For instance `dataset[0]` will retrieve a single temporal frame from the dataset containing observation(s) and an action as PyTorch tensors ready to be fed to a model.

A specificity of `LeRobotDataset` is that, rather than retrieving a single frame by its index, we can retrieve several frames based on their temporal relationship with the indexed frame, by setting `delta_timestamps` to a list of relative times with respect to the indexed frame. For example, with `delta_timestamps = {"observation.image": [-1, -0.5, -0.2, 0]}` one can retrieve, for a given index, 4 frames: 3 "previous" frames 1 second, 0.5 seconds, and 0.2 seconds before the indexed frame, and the indexed frame itself (corresponding to the 0 entry). See example [1_load_lerobot_dataset.py](https://github.com/huggingface/lerobot/blob/main/examples/dataset/load_lerobot_dataset.py) for more details on `delta_timestamps`.

Under the hood, the `LeRobotDataset` format makes use of several ways to serialize data which can be useful to understand if you plan to work more closely with this format. We tried to make a flexible yet simple dataset format that would cover most type of features and specificities present in reinforcement learning and robotics, in simulation and in real-world, with a focus on cameras and robot states but easily extended to other types of sensory inputs as long as they can be represented by a tensor.

Here are the important details and internal structure organization of a typical `LeRobotDataset` instantiated with `dataset = LeRobotDataset("lerobot/aloha_static_coffee")`. The exact features will change from dataset to dataset but not the main aspects:

```
dataset attributes:
  ├ hf_dataset: a Hugging Face dataset (backed by Arrow/parquet). Typical features example:
  │  ├ observation.images.cam_high (VideoFrame):
  │  │   VideoFrame = {'path': path to a mp4 video, 'timestamp' (float32): timestamp in the video}
  │  ├ observation.state (list of float32): position of an arm joints (for instance)
  │  ... (more observations)
  │  ├ action (list of float32): goal position of an arm joints (for instance)
  │  ├ episode_index (int64): index of the episode for this sample
  │  ├ frame_index (int64): index of the frame for this sample in the episode ; starts at 0 for each episode
  │  ├ timestamp (float32): timestamp in the episode
  │  ├ next.done (bool): indicates the end of an episode ; True for the last frame in each episode
  │  └ index (int64): general index in the whole dataset
  ├ meta: a LeRobotDatasetMetadata object containing:
  │  ├ info: a dictionary of metadata on the dataset
  │  │  ├ codebase_version (str): this is to keep track of the codebase version the dataset was created with
  │  │  ├ fps (int): frame per second the dataset is recorded/synchronized to
  │  │  ├ features (dict): all features contained in the dataset with their shapes and types
  │  │  ├ total_episodes (int): total number of episodes in the dataset
  │  │  ├ total_frames (int): total number of frames in the dataset
  │  │  ├ robot_type (str): robot type used for recording
  │  │  ├ data_path (str): formattable string for the parquet files
  │  │  └ video_path (str): formattable string for the video files (if using videos)
  │  ├ episodes: a DataFrame containing episode metadata with columns:
  │  │  ├ episode_index (int): index of the episode
  │  │  ├ tasks (list): list of tasks for this episode
  │  │  ├ length (int): number of frames in this episode
  │  │  ├ dataset_from_index (int): start index of this episode in the dataset
  │  │  └ dataset_to_index (int): end index of this episode in the dataset
  │  ├ stats: a dictionary of statistics (max, mean, min, std) for each feature in the dataset, for instance
  │  │  ├ observation.images.front_cam: {'max': tensor with same number of dimensions (e.g. `(c, 1, 1)` for images, `(c,)` for states), etc.}
  │  │  └ ...
  │  └ tasks: a DataFrame containing task information with task names as index and task_index as values
  ├ root (Path): local directory where the dataset is stored
  ├ image_transforms (Callable): optional image transformations to apply to visual modalities
  └ delta_timestamps (dict): optional delta timestamps for temporal queries
```

A `LeRobotDataset` is serialised using several widespread file formats for each of its parts, namely:

- hf_dataset stored using Hugging Face datasets library serialization to parquet
- videos are stored in mp4 format to save space
- metadata are stored in plain json/jsonl files

Dataset can be uploaded/downloaded from the HuggingFace hub seamlessly. To work on a local dataset, you can specify its location with the `root` argument if it's not in the default `~/.cache/huggingface/lerobot` location.

## 📝 Recent Updates

### SpaceMouse System Upgrade (2025-01-23)

🎉 **Major SpaceMouse System Overhaul:**

- **Modern Library Migration**: Migrated from legacy `spnav` to modern `PySpaceMouse` library
- **Cross-Platform Support**: Now supports Linux, macOS, and Windows
- **No System Dependencies**: Removed requirement for `spacenavd` system service
- **Dual-Device Support**: Revolutionary dual SpaceMouse mode for advanced manipulation
- **Flexible Configuration**: Per-device sensitivity and axis assignment
- **Hardware Independence**: Direct HID communication for better reliability

**Breaking Changes:**

- `spacenavd` service is no longer required
- Configuration options have been expanded with new dual-device parameters
- Old single-device configurations remain fully compatible

**Migration Benefits:**

- ✅ Easier setup (no system services to configure)
- ✅ Better cross-platform compatibility
- ✅ More responsive input handling
- ✅ Advanced dual-hand control capabilities
- ✅ Future-proof with active library maintenance

## Citation

If you use this codebase, please cite the original LeRobot project:

```bibtex
@misc{cadene2024lerobot,
    author = {Cadene, Remi and Alibert, Simon and Soare, Alexander and Gallouedec, Quentin and Zouitine, Adil and Palma, Steven and Kooijmans, Pepijn and Aractingi, Michel and Shukor, Mustafa and Aubakirova, Dana and Russi, Martino and Capuano, Francesco and Pascal, Caroline and Choghari, Jade and Moss, Jess and Wolf, Thomas},
    title = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
    howpublished = "\url{https://github.com/huggingface/lerobot}",
    year = {2024}
}
```

If you use this fork (LeRobot-Xense) specifically, please also cite:

```bibtex
@misc{vertax2026lerobotxense,
    author = {vertax42 and Xense Robotics Team},
    title = {LeRobot-Xense: LeRobot with Xense Tactile Robotics Support},
    howpublished = "\url{https://github.com/XenseRobotics-AI/lerobot-xense}",
    year = {2026}
}
```
