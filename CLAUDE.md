# lerobot-xense — Claude working notes

XenseRobotics Physical-AI platform: a lerobot fork carrying the arm robots
(ARX5, Flexiv Rizon4, Elite CS66) plus the Xense tactile gripper and its wrist
camera. Sister repo to `xense-taccap-lerobot`, which carries the TacCap-Gripper
handheld rig; the two share the `taccap-gripper` SDK submodule and several of
these traps.

## `import xense.taccap` must precede torchvision

torchvision ships a vendored `libjpeg` that claims the `LIBJPEG_8.0` symbol
version but carries **none** of the `jpeg12_*` symbols conda's `libtiff` needs.
Whichever loads first wins the version slot, so once torchvision is in, every
later `import xense.taccap` — which reaches libtiff via
`libopencv_videoio` → `libopencv_imgcodecs` → `libtiff` — dies with:

```text
ImportError: .../libtiff.so.6: undefined symbol: jpeg12_write_raw_data, version LIBJPEG_8.0
```

`lerobot_record.py`, `lerobot_replay.py`, `lerobot_teleoperate.py` and
`tests/conftest.py` each carry a `contextlib.suppress(ImportError)` import of
`xense.taccap` **above every lerobot import** for exactly this. Moving one below
them puts the bug back; `tests/robots/test_taccap_import_order.py` fails if that
happens.

**Why it hid for so long:** nothing fails at startup. `XenseWristCamera` imports
`FisheyeUndistorter` lazily inside `connect()` (`cameras/xense/camera_wrist.py`),
so a recording session came up fine and then died at camera-connect time. In the
test suite it looked like 11 broken fisheye tests rather than one load-order
problem.

Only the entry points that **both** pull torchvision (through `lerobot.datasets`)
and touch the SDK need the block. `lerobot-calibrate`, `-find-cameras`,
`-find-port`, `-setup-motors` and `-info` never load torchvision;
`-dataset-viz`, `-edit-dataset` and `-annotate-reward` do load it but never
import the SDK, so adding the block there would only bolt an SDK dependency onto
pure dataset tooling. Verify with a fresh interpreter before adding it anywhere
new rather than sprinkling it.

Things that do **not** work, so nobody re-tries them:

- **`LD_PRELOAD` of conda's libjpeg.** torchvision's copy is auditwheel-renamed
  (`libjpeg.4af9affd.so.8`), so it is not competing on soname but on the symbol
  version — preloading cannot outrank it.
- **Moving the lazy import in `camera_wrist.py` to module top level.** That
  module is itself imported _after_ `lerobot.datasets`, so the order is unchanged
  — it only moves the same ImportError from `connect()` to package import, which
  is strictly worse.
- **Dropping libtiff from the SDK.** It arrives through `libopencv_videoio`'s
  own `DT_NEEDED` on `libopencv_imgcodecs`; the SDK never calls an imgcodecs API
  and does not link it explicitly. Removing it means replacing `cv::VideoCapture`
  with a hand-written V4L2 capture path — and MJPEG decoding then needs a JPEG
  library anyway, so the class of conflict returns.

## Wrist camera: this repo owns the UVC device, not the SDK

`open_cameras` appears nowhere here. `XenseWristCamera` opens the device through
the LeRobot camera framework and calls `FisheyeUndistorter.apply()` itself, using
intrinsics the arm reads off the gripper's MCU and hands over with
`set_fisheye_calibration()`.

This is why the SDK's `wrist_color_mode` default (RGB since SDK `6b33678`) does
not reach us: it only applies when the SDK owns the camera. `camera_wrist.py`'s
`_postprocess_image` still receives BGR straight from OpenCV, which is exactly
what `FisheyeUndistorter.apply()` expects, and the base class converts to RGB
afterwards.

## conda and uv co-own `site-packages` — make the two versions agree

Both package managers install into the same `site-packages` and neither can read
the other's ledger: conda tracks files in `conda-meta/<pkg>.json`, uv/pip in
`<pkg>.dist-info/RECORD`. The two directions fail differently.

**uv over conda** looks clean. conda's Python packages ship a `.dist-info`, so
`uv pip install` reads it and uninstalls properly — but nothing touches
`conda-meta`, which goes on claiming the old version. Silent divergence.

**conda over uv** is the one that breaks. `mamba env update` restores only the
files in conda's own list and leaves every extra file uv added. If the two
versions moved a module into a package, the leftover package shadows the module
and the import dies. That is the whole of the 2026-08-28 outage:

```text
ImportError: cannot import name 'get_runnable_pip' from 'pip._internal.utils.misc'
```

uv had put pip 26.2.1 (`_internal/build_env/` package) over conda's 26.1.2
(`_internal/build_env.py` module). conda relinked 26.1.2, the 26.2.1 package
directory survived, and `build_env/installer.py` went looking for
`get_runnable_pip` in a `utils/misc.py` that does not have it. `mamba env update`
then died on its own `pip:` section and took the whole installer with it.
`numpy` is the same shape waiting to happen — `ctypeslib.py` became
`ctypeslib/` in 2.3.

**conda never heals this by itself.** It only relinks a package it has decided
to change; while `conda-meta` says the spec is satisfied it never looks at what
is actually on disk. Divergence therefore persists indefinitely and detonates
later, at whichever unrelated `env update` first touches that package.

So every package `conda_environment.yaml` declares must resolve to the version
the wheel stack resolves. Two pins exist for exactly this:

- `numpy>=1.26.4,<2.3` — `opencv-python` pins `numpy<2.3.0`, so uv always lands
  on 2.2.6 and conda's unbounded 2.4.6 could never survive.
- `cmake>=3.29,<4.2` — `pyproject.toml` pins `cmake<4.2.0`, so uv downgrades
  conda's 4.3.0 and overwrites `bin/{cmake,ccmake,cpack,ctest}`.

Both land on a version conda-forge also has (2.2.6, 4.1.3), so neither side
overwrites the other. `setup_env.sh` no longer runs `uv pip install --upgrade`
on anything conda owns, and `check_conda_uv_consistency` reports any violation
at the end of every install. It also sets `PYTHONNOUSERSITE=1`, because this
host's `~/.local/lib/python3.12/site-packages` shadows six env packages
(a stray `pip install --user pillow-heif`) and the system python3.12 shares that
directory, so it cannot simply be deleted.

Repairing an env that has already diverged is manual, in this order:
`mamba install --force-reinstall <pkg>`, then delete the uv-only files (RECORD
minus conda's file list) and uv's `dist-info`. A `mamba env update` that crashed
mid-transaction also leaves **duplicate `conda-meta` records**, and `mamba list`
then reports the stale one — check for two `conda-meta/<pkg>-*.json` before
believing any version number.

Things that do **not** work, so nobody re-tries them:

- **Assuming the `INSTALLER` file tells you who won.** It records who wrote the
  `dist-info`, not whose files are on disk, and `setup_env.sh` flips ownership
  twice per run (`mamba env update` first, then dozens of `uv pip install`).
  Diff conda's file list against the RECORD and check what actually imports.
- **Deleting "orphans" from a RECORD diff without checking `conda-meta`.**
  `bin/pybind11-config` shows up as uv-only and is owned by conda; deleting
  `numpy.libs/*.so` breaks the wheel numpy whose `_multiarray_umath.so` carries
  `RPATH $ORIGIN/../../numpy.libs`.
- **Treating a name collision as a conflict.** conda's `spdlog` is the C++
  library (`include/`, `lib/`) and PyPI's is a Python binding this repo really
  does import (`utils/utils.py`, `motors/damiao/damiao.py`, and two more). Zero
  overlapping files. Only flag packages whose file sets actually intersect.
- **Pinning conda up to whatever uv installed.** conda-forge lags PyPI, and it
  has none of `setuptools 84.0.0`, `wheel 0.48.0`, `pybind11 3.1.0`,
  `packaging 26.3`. For those the fix is the other direction: stop uv from
  upgrading them and let conda's version stand.
- **`uv pip install --upgrade <pkg>` to bump one package.** uv's `--upgrade`
  lifts the prefer-installed rule for the _whole_ resolution, not just the named
  packages: `--upgrade pybind11-stubgen build` walked `build`'s `packaging>=24.0`
  from conda's 26.2 to a 26.3 conda-forge does not ship (2026-09-01), and that
  is the only way `packaging` — never named anywhere in this repo — ever moved.
  `--upgrade-package <pkg>` (`-P`) is the per-package form, and a bare
  `uv pip install "<pkg>>=<min>"` already installs something newer when the
  installed copy fails the spec, so `--upgrade` is never the right flag here.

## The Pico4 client SDK comes from the `.deb`, not from a submodule

**There is no `third_party/XenseVR-PC-Service` submodule.**
`xensevr_pc_service_sdk` is built from **our** pybind under
`src/lerobot/teleoperators/pico4/xensevr-pc-service-pybind/`, and the C SDK it
links — `PXREARobotSDK.h` + `libPXREARobotSDK.so` — is copied by `setup_env.sh`
out of the installed `xensevr-pc-service` `.deb`
(`/opt/apps/roboticsservice/SDK/{include,x64}`; `arm64` on arm64 hosts). The
pybind's `include/` and `lib/` are gitignored staging, not sources.

The submodule was a 279 MiB `.git` (33 MiB with `--depth 1`) of Qt service
sources and prebuilt gRPC archives, cloned purely to rebuild a library the
`.deb` already shipped — and the `.deb` was never optional, because its daemon
is what the teleop talks to at runtime. The header in the package is
byte-identical to the one the submodule built against, and the extension built
from it exports the same 36 symbols.

**The trade:** an SDK _source_ fix now has to travel through a `.deb` release —
bump `debPack/control`, rebuild, publish, and bump `DEB_VER` in `setup_env.sh`.
Re-releasing the same version number does not work: `install_xensevr_service()`
skips a `.deb` whose version already matches what dpkg reports, so a host would
keep the old SDK and silently build its bindings against it.

Three consequences worth remembering:

- **nlohmann comes from conda** (`nlohmann_json=3.11.3` in
  `conda_environment.yaml`, `find_package`d by the pybind CMakeLists).
  `py_bindings.cpp` needs `<nlohmann/json.hpp>`; it used to resolve by accident
  because the submodule vendored a copy that `setup_env.sh` shovelled into
  `include/`. The `.deb` does not carry it.
- **`setup.py:sdk_version()` asks dpkg**, not a hardcoded string. It sat at
  `0.1.0` through the 0.2.0 bump that added the Pico camera calls, so `pip list`
  denied a capability the module had. The `.deb` is not a proxy for the SDK, it
  _is_ where the SDK came from.
- **The pybind CMakeLists no longer branches on `aarch64`.** It used to point an
  arm64 build at `include/aarch64` / `lib/aarch64`; `install_pico4()` picks
  `SDK/x64` or `SDK/arm64` itself and stages the result flat, so what lands in
  `include/` is already the right build for the host.

`install_xensevr_service()` is deliberately non-fatal on a failed download — it
warns and returns 0, and `install_pico4()` is what refuses to continue, because
the missing-SDK message there names the actual files it could not find. Also
note the `.part` staging on the curl: handing a _complete_ file to `curl -C -`
asks for a range starting at EOF, the server answers 416, and
`--retry-all-errors` then retries that five times before giving up.

To work on the C SDK itself, clone `XenseRobotics-AI/XenseVR-PC-Service`
separately.

## Repo weight: attribute it to a ref, not to a path

The repo was 324 MiB against an 8.8 MiB working tree until 2026-08-25. All of it
was history, and 278.6 MiB of that hung off a **single stale branch**,
`dev/taccap-gripper` — 7 vendored `dist/xensesdk-2.0.0-*.whl` builds (196 MiB;
93 + 68 + 11 + 11 + 8.5 + 2.9 + 2.5 compressed) plus upstream lerobot's
`tests/data/**/*.arrow` fixtures (82 MiB). `main` and every tag referenced none of
it. Deleting the branch took the repo to 44.65 MiB with no rewrite of `main`, no
force-push, and nothing for collaborators to re-clone.

The branch was the pre-slimming snapshot of the line of work that now lives in the
sister repo `xense-taccap-lerobot` (36 MiB, zero wheels and zero `tests/data` in
history — it was rewritten there). Both share root commit `007ffa89`. The sister is
ahead at `db838fb6`, which never existed here, so this branch was dead, not
diverged. Its only surviving copy is
`/home/vertax/dev-taccap-gripper-b2154ab2-20260825.bundle` (308 MB, verified,
tip `b2154ab2`, 1641 commits) — git cannot record that, because the branch is gone
from both sides.

**Diagnose per-ref before touching anything.** A largest-blob listing over `--all`
tells you _what_ is big but not _which ref keeps it alive_, and it will walk you
straight into rewriting `main` for blobs `main` never had. Ask instead what each ref
costs over `main`:

```bash
git rev-list main --objects | awk '{print $1}' | sort -u > /tmp/main_objs
for r in $(git for-each-ref --format='%(refname)' refs/remotes refs/tags); do
  git rev-list "$r" --objects 2>/dev/null | awk '{print $1}' | sort -u > /tmp/r_objs
  sz=$(comm -13 /tmp/main_objs /tmp/r_objs | git cat-file --batch-check='%(objectsize:disk)' \
       | awk '/^[0-9]+$/{s+=$1} END{printf "%.1f", s/1048576}')
  echo "$r +${sz} MiB unique-vs-main"
done
```

Then bundle the branch, delete it on both sides, `git remote prune origin`,
`git reflog expire --expire-unreachable=now --all`, and only then `git gc
--prune=now`. Skipping the reflog expiry leaves every object still reachable and
`gc` reclaims nothing.

Things that do **not** work, so nobody re-tries them:

- **Adding `dist/` to `.gitignore`.** It is already there (`.gitignore:33`) and was
  there while all 7 wheels went in — they were forced past it. Ignoring a path has
  no effect on blobs already committed, and deleting the file does not shrink the
  pack.
- **Deleting the branch on GitHub and expecting the server to shrink.** The ref goes
  away and fresh clones get the small pack, but GitHub's own pack only shrinks when
  they gc; a support ticket is the only way to force it. Local size drops
  immediately, server size does not.
- **`filter-repo` on `main` for the residual 13 MiB.** What is left is
  `libPXREARobotSDK.so` (8 MiB) and `arx5_interface.cpython-311-*.so` (5 MiB),
  historical residue from before those SDKs moved out of the tree — ARX5 to the
  `third_party/ARX5_SDK` submodule, the Pico4 client SDK to the
  `xensevr-pc-service` `.deb` (see above) — neither is in `HEAD` any more. Rewriting
  `main` breaks every outstanding clone and PR to land 13 MiB on a 45 MiB repo,
  which is already the same order as the sister repo's 36 MiB.
