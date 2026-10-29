#!/usr/bin/env bash
# Build rebot_rt for the selected Python without changing its ROS/Boost stack.
set -euo pipefail

case "${1:-}" in
    --help|-h)
        cat <<'USAGE'
Usage: bash scripts/install_rebot.sh [--skip-deps]
Install only the reBot B601 native SDK into the active Python environment.
Pinocchio and build tools live in a separate conda-forge environment.

  --skip-deps          Reuse an existing build environment without a conda solve
Environment overrides:
  REBOT_PYTHON         Target Python executable (default: python on PATH)
  REBOT_DEPS_PREFIX    Build/runtime dependency environment; keep it after install
  REBOT_BUILD_JOBS     Parallel compiler jobs (default: 2)
USAGE
        exit 0 ;;
    ""|--skip-deps) ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { echo "Too many arguments" >&2; exit 2; }
if [[ "$(uname -s)-$(uname -m)" != Linux-x86_64 ]]; then
    echo "[rebot] This installer supports Linux x86_64 (Ubuntu 22.04/24.04)." >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SDK_DIR="$REPO_ROOT/third_party/libpyrebot"
TARGET_PYTHON="$("${REBOT_PYTHON:-python}" -c 'import sys; print(sys.executable)')"
PY_VERSION="$("$TARGET_PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
DEPS_PREFIX="${REBOT_DEPS_PREFIX:-${XDG_DATA_HOME:-$HOME/.local/share}/lerobot-xense/rebot-rt-deps/py$PY_VERSION}"
DEPS_PREFIX="$(realpath -m "$DEPS_PREFIX")"
DEPS_PYTHON="$DEPS_PREFIX/bin/python"

if [[ ! -f "$SDK_DIR/CMakeLists.txt" ]]; then
    echo "[rebot] Missing submodule. Run from the repository root:" >&2
    echo "  git submodule update --init --recursive third_party/libpyrebot" >&2
    exit 1
fi

# Never solve Pinocchio 3.9 (Boost 1.88) into the target's ROS Jazzy environment.
if [[ "$(realpath -m "$DEPS_PREFIX")" == "$("$TARGET_PYTHON" -c 'import sys; print(sys.prefix)')" ]]; then
    echo "[rebot] REBOT_DEPS_PREFIX must be separate from the target Python environment." >&2
    exit 1
fi
if ! "$TARGET_PYTHON" -c 'import numpy' >/dev/null 2>&1; then
    echo "[rebot] NumPy is unavailable in $TARGET_PYTHON. Activate the installed LeRobot environment first." >&2
    exit 1
fi
CONDA_BIN="${CONDA_EXE:-$(command -v conda || command -v mamba || true)}"
if [[ -z "$CONDA_BIN" ]]; then
    echo "[rebot] Conda/mamba is required. Activate your lerobot-xense environment first." >&2
    exit 1
fi

if [[ "${1:-}" != --skip-deps ]]; then
    # Reuse a complete environment offline; an incomplete one is repaired below.
    if ! "$DEPS_PYTHON" -c '
import importlib.metadata as m, json, pathlib, sys
import pybind11, setuptools, wheel
p = pathlib.Path(sys.prefix)
assert f"{sys.version_info.major}.{sys.version_info.minor}" == sys.argv[1]
assert int(m.version("setuptools").split(".")[0]) >= 68
assert int(m.version("pip").split(".")[0]) >= 23
assert any(json.loads(f.read_text())["version"] == "3.9.0"
           for f in (p / "conda-meta").glob("libpinocchio-*.json"))
assert any(json.loads(f.read_text())["version"].startswith("12.")
           for f in (p / "conda-meta").glob("gxx_impl_linux-64-*.json"))
assert (p / "include/pinocchio/multibody/model.hpp").is_file()
assert (p / "include/eigen3/Eigen/Dense").is_file()
assert (p / "lib/libpinocchio_default.so").is_file()
assert (p / "lib/libpinocchio_parsers.so").is_file()
assert (p / "bin/cmake").is_file()
' "$PY_VERSION" 2>/dev/null; then
        conda_action=create
        [[ ! -f "$DEPS_PREFIX/conda-meta/history" ]] || conda_action=install
        # GCC 12 keeps the extension compatible with Ubuntu 22.04's libstdc++
        # when NumPy/other extensions have loaded that runtime before rebot_rt.
        "$CONDA_BIN" "$conda_action" --yes --prefix "$DEPS_PREFIX" \
            --override-channels -c conda-forge \
            "python=$PY_VERSION" libpinocchio=3.9.0 eigen=3.4.0 \
            'cmake>=3.29,<4.2' 'pybind11>=2.12' gxx_linux-64=12 \
            'setuptools>=68' wheel pip
    fi
fi

# Packaging runs in the dependency environment, so target setuptools/pip never
# need upgrading. Both interpreters must use the same extension ABI.
TARGET_ABI="$("$TARGET_PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
BUILD_ABI="$("$DEPS_PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
if [[ "$TARGET_ABI" != "$BUILD_ABI" ]]; then
    echo "[rebot] Python ABI mismatch: target=$TARGET_ABI build=$BUILD_ABI" >&2
    exit 1
fi
PYBIND11_DIR="$("$DEPS_PYTHON" -m pybind11 --cmakedir)"
RUN_IN_DEPS=("$CONDA_BIN" run --prefix "$DEPS_PREFIX")
case "$(basename "$CONDA_BIN")" in
    *mamba*) ;;  # mamba streams output by default and has no --no-capture-output
    *) RUN_IN_DEPS+=(--no-capture-output) ;;
esac

# Fresh source/build directories prevent stale CMake paths and copied binaries
# from another machine from being packaged. Do not overwrite a loaded .so.
BUILD_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/rebot-install.XXXXXX")"
trap 'rm -rf -- "$BUILD_ROOT"' EXIT
"$DEPS_PYTHON" - "$SDK_DIR" "$BUILD_ROOT/source" <<'PY'
import pathlib, shutil, sys
source, dest = map(pathlib.Path, sys.argv[1:])
dest.mkdir()
for name in ("CMakeLists.txt", "setup.py", "pyproject.toml"):
    shutil.copy2(source / name, dest / name)
for name in ("src", "include", "rebot_rt"):
    shutil.copytree(source / name, dest / name,
                    ignore=shutil.ignore_patterns("_native*.so", "__pycache__", "*.pyc"))
PY

echo "[rebot] Target Python: $TARGET_PYTHON"
echo "[rebot] Native dependencies: $DEPS_PREFIX"
"${RUN_IN_DEPS[@]}" \
    cmake -S "$BUILD_ROOT/source" -B "$BUILD_ROOT/build" \
    -DPython3_EXECUTABLE="$DEPS_PYTHON" -DPINOCCHIO_PREFIX="$DEPS_PREFIX" \
    -Dpybind11_DIR="$PYBIND11_DIR" -DCMAKE_BUILD_TYPE=Release
"${RUN_IN_DEPS[@]}" \
    cmake --build "$BUILD_ROOT/build" --parallel "${REBOT_BUILD_JOBS:-2}"
# Check the artifact in the target interpreter before replacing its installed
# SDK, including the NumPy-first import order used by LeRobot.
cat > "$BUILD_ROOT/verify.py" <<'PY'
from pathlib import Path
import math
import sys
if len(sys.argv) > 1:
    import ctypes
    ctypes.CDLL(sys.argv[1], mode=ctypes.RTLD_GLOBAL)
import numpy
import rebot_rt

for family, asset in (
    (rebot_rt.Family.DM, "DM_TacCap00/urdf/rebot_b601_dm_taccap_v1_gripper.urdf"),
    (rebot_rt.Family.RS, "RS_TacCap05/urdf/rebot_b601_rs_taccap_v2_gripper.urdf"),
):
    urdf = Path(rebot_rt.__file__).parent / "assets" / asset
    robot = rebot_rt.Robot(family, "unused", str(urdf), "taccap_eef_link", False)
    # Native JointArray has seven slots; the independent TacCap is not enabled.
    torques = robot.gravity_torques([0.0] * 7)
    assert len(torques) == 7 and all(map(math.isfinite, torques)), torques
print("[rebot] Verified native import and DM/RS gravity models:", rebot_rt.__file__)
PY
cd "$BUILD_ROOT"
RUNTIME_LIB=""
if ! PYTHONPATH="$BUILD_ROOT/source${PYTHONPATH:+:$PYTHONPATH}" \
    "$TARGET_PYTHON" "$BUILD_ROOT/verify.py" > "$BUILD_ROOT/verify.log" 2>&1; then
    # NumPy may load Ubuntu's older libstdc++ first. Prefer the target's own
    # conda runtime (also used by setup_env.sh's sitecustomize), then our isolated
    # runtime. Do not replace any library files or rely on LD_LIBRARY_PATH,
    # which is stripped when Python has cap_sys_nice.
    TARGET_BASE="$("$TARGET_PYTHON" -c 'import sys; print(sys.base_prefix)')"
    for candidate in "$TARGET_BASE/lib/libstdc++.so.6" "$DEPS_PREFIX/lib/libstdc++.so.6"; do
        [[ -f "$candidate" ]] || continue
        if PYTHONPATH="$BUILD_ROOT/source${PYTHONPATH:+:$PYTHONPATH}" \
            "$TARGET_PYTHON" "$BUILD_ROOT/verify.py" "$candidate" > "$BUILD_ROOT/verify.log" 2>&1; then
            RUNTIME_LIB="$candidate"
            break
        fi
    done
    if [[ -z "$RUNTIME_LIB" ]]; then
        cat "$BUILD_ROOT/verify.log" >&2
        exit 1
    fi
fi
cat "$BUILD_ROOT/verify.log"
"$DEPS_PYTHON" -m pip wheel "$BUILD_ROOT/source" \
    --no-build-isolation --no-deps --no-index --wheel-dir "$BUILD_ROOT/wheels"
"$DEPS_PYTHON" -m pip --python "$TARGET_PYTHON" install \
    --no-deps --no-index --force-reinstall "$BUILD_ROOT"/wheels/rebot_rt-*.whl
if [[ -n "$RUNTIME_LIB" ]]; then
    "$TARGET_PYTHON" - "$RUNTIME_LIB" <<'PY'
from pathlib import Path
import sys, sysconfig
path = Path(sysconfig.get_path("purelib")) / "lerobot_rebot_runtime.pth"
path.write_text(
    "# Managed by lerobot-xense/scripts/install_rebot.sh; required by rebot_rt.\n"
    f"import ctypes; ctypes.CDLL({sys.argv[1]!r}, mode=ctypes.RTLD_GLOBAL)\n"
)
print("[rebot] Configured C++ runtime preload:", path)
PY
fi
"$TARGET_PYTHON" "$BUILD_ROOT/verify.py"
echo "[rebot] Installed. Keep $DEPS_PREFIX: the native module uses its shared libraries."
