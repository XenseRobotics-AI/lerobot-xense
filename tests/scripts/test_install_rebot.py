"""Installer failure paths must not install stale binaries or alter the target's conda stack."""

import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]


class TestInstallRebot(unittest.TestCase):
    def test_rejects_target_environment_as_dependency_prefix(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "scripts").mkdir()
            script = root / "scripts/install_rebot.sh"
            shutil.copy2(ROOT / "scripts/install_rebot.sh", script)
            sdk = root / "third_party/libpyrebot"
            sdk.mkdir(parents=True)
            # Reach prefix validation even when CI has not checked out the SDK.
            (sdk / "CMakeLists.txt").touch()
            result = subprocess.run(
                ["bash", str(script), "--skip-deps"],
                env={**os.environ, "REBOT_PYTHON": sys.executable, "REBOT_DEPS_PREFIX": sys.prefix},
                text=True,
                capture_output=True,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be separate", result.stderr)

    def test_failed_build_never_installs_a_copied_native_binary(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "scripts").mkdir()
            script = root / "scripts/install_rebot.sh"
            shutil.copy2(ROOT / "scripts/install_rebot.sh", script)
            sdk = root / "third_party/libpyrebot"
            for name in ("src", "include", "rebot_rt"):
                (sdk / name).mkdir(parents=True)
            for name in ("CMakeLists.txt", "setup.py", "pyproject.toml"):
                (sdk / name).touch()
            (sdk / "rebot_rt/_native.cpython-stale.so").write_text("old machine's binary")
            (root / "deps/bin").mkdir(parents=True)
            (root / "deps/bin/python").symlink_to(sys.executable)
            # Only --cmakedir is needed for this failure test; no compiler/SDK is run.
            (root / "pybind11").mkdir()
            (root / "pybind11/__init__.py").touch()
            (root / "pybind11/__main__.py").write_text("print('/unused/pybind11')\n")
            conda = root / "conda"
            conda.write_text(
                "#!/bin/bash\n"
                'printf "%s\\n" "$*" >> "$REBOT_TEST_CALLS"\n'
                'if [[ "$*" == *"cmake --build"* ]]; then exit 37; fi\n'
                # The stage must contain source only, even when a .so was copied with the repo.
                "while [[ $# -gt 0 ]]; do\n"
                '  if [[ "$1" == -S ]]; then\n'
                '    compgen -G "$2/rebot_rt/_native*.so" && exit 99\n'
                '    printf "%s\\n" "$2" > "$REBOT_TEST_STAGE"\n'
                "  fi\n"
                "  shift\n"
                "done\nexit 0\n"
            )
            conda.chmod(0o755)
            result = subprocess.run(
                ["bash", str(script), "--skip-deps"],
                env={
                    **os.environ,
                    "REBOT_PYTHON": sys.executable,
                    "REBOT_DEPS_PREFIX": str(root / "deps"),
                    "CONDA_EXE": str(conda),
                    "PYTHONPATH": str(root),
                    "REBOT_TEST_CALLS": str(root / "calls"),
                    "REBOT_TEST_STAGE": str(root / "stage"),
                },
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 37, result.stdout + result.stderr)
            calls = (root / "calls").read_text().splitlines()
            self.assertEqual(len(calls), 2)
            self.assertTrue(all(f"run --prefix {root / 'deps'}" in c for c in calls))
            self.assertNotIn("pip", result.stdout + result.stderr)
            self.assertFalse(Path((root / "stage").read_text().strip()).exists())
            self.assertTrue((sdk / "rebot_rt/_native.cpython-stale.so").is_file())
