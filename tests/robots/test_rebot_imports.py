"""Other robots must remain importable even with an unusable B601 native SDK."""

import subprocess
import sys
import unittest


class TestRebotImports(unittest.TestCase):
    def test_missing_or_broken_native_sdk_has_an_install_hint(self):
        code = """
import importlib
import sys
from importlib.abc import MetaPathFinder

class BrokenNativeSDK(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'rebot_rt':
            if sys.argv[1] == 'missing':
                raise ModuleNotFoundError("No module named 'rebot_rt'")
            raise OSError('libpinocchio_default.so: cannot open shared object file')

sys.meta_path.insert(0, BrokenNativeSDK())
try:
    importlib.import_module('lerobot.robots.rebot_b601_rt.rebot_b601_rt')
except ImportError as exc:
    assert 'bash scripts/install_rebot.sh' in str(exc), str(exc)
    assert sys.executable in str(exc), str(exc)
    assert isinstance(exc.__cause__, (ImportError, OSError))
else:
    raise AssertionError('Missing SDK did not prevent driver import')
"""
        for failure in ("missing", "broken"):
            with self.subTest(failure=failure):
                result = subprocess.run(
                    [sys.executable, "-c", code, failure], capture_output=True, text=True, timeout=60
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_common_imports_leave_b601_unloaded_until_explicit_registration(self):
        code = """
import sys
from importlib.abc import MetaPathFinder
from pathlib import Path
from tempfile import TemporaryDirectory

class BrokenNativeSDK(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'rebot_rt':
            raise OSError('B601 native library must not load for other robots')

sys.meta_path.insert(0, BrokenNativeSDK())
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.robots.mock_robot import MockRobot, MockRobotConfig
with TemporaryDirectory() as directory:
    config = MockRobotConfig(calibration_dir=Path(directory))
    assert isinstance(make_robot_from_config(config), MockRobot)
assert 'lerobot.robots.rebot_b601_rt' not in sys.modules
assert 'lerobot.robots.bi_rebot_b601_rt' not in sys.modules
assert 'rebot_b601_rt' not in RobotConfig.get_known_choices()
assert 'bi_rebot_b601_rt' not in RobotConfig.get_known_choices()
from lerobot.robots import bi_rebot_b601_rt, rebot_b601_rt
assert 'rebot_b601_rt' in RobotConfig.get_known_choices()
assert 'bi_rebot_b601_rt' in RobotConfig.get_known_choices()
assert 'rebot_rt' not in sys.modules
assert 'lerobot.robots.rebot_b601_rt.rebot_b601_rt' not in sys.modules
assert 'lerobot.robots.bi_rebot_b601_rt.bi_rebot_b601_rt' not in sys.modules
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_supported_clis_register_b601_without_loading_the_native_sdk(self):
        # Each entrypoint gets a fresh process so imports in another test cannot
        # hide a missing registration before CLI configuration parsing.
        code = """
import importlib
import sys
from importlib.abc import MetaPathFinder

class BrokenNativeSDK(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'rebot_rt':
            raise OSError('B601 native library must not load during CLI registration')

sys.meta_path.insert(0, BrokenNativeSDK())
importlib.import_module('lerobot.scripts.' + sys.argv[1])
import draccus
from lerobot.robots import RobotConfig

arm = {'motor_family': 'rs'}
single = draccus.decode(RobotConfig, dict(arm, type='rebot_b601_rt'))
dual = draccus.decode(RobotConfig, {
    'type': 'bi_rebot_b601_rt',
    'left_arm_config': dict(arm, channel='can0'),
    'right_arm_config': dict(arm, channel='can1'),
})
assert single.type == 'rebot_b601_rt'
assert dual.type == 'bi_rebot_b601_rt'
assert dual.left_arm_config.channel == 'can0'
assert dual.right_arm_config.channel == 'can1'
assert 'rebot_rt' not in sys.modules
assert 'lerobot.robots.rebot_b601_rt.rebot_b601_rt' not in sys.modules
assert 'lerobot.robots.bi_rebot_b601_rt.bi_rebot_b601_rt' not in sys.modules
"""
        for entrypoint in (
            "lerobot_teleoperate",
            "lerobot_record",
            "lerobot_replay",
        ):
            with self.subTest(entrypoint=entrypoint):
                result = subprocess.run(
                    [sys.executable, "-c", code, entrypoint], capture_output=True, text=True, timeout=60
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
