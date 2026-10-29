"""Temporary model paths for B601 CLI tests that replace the robot with a double."""

from contextlib import chdir, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from lerobot.robots.rebot_b601_rt.config_rebot_b601_rt import DEFAULT_TACCAP_URDFS


@contextmanager
def isolated_rebot_assets():
    """Resolve recipe paths without a checked-out SDK or a model used for real control.

    These tests exercise configuration and CLI handoffs, not kinematics. Keep
    file validation active and leave the real checkout's assets untouched.
    """
    with TemporaryDirectory() as temp_dir:
        assets = Path(temp_dir) / "third_party/libpyrebot/rebot_rt/assets"
        for relative_path in DEFAULT_TACCAP_URDFS.values():
            urdf = assets / relative_path
            urdf.parent.mkdir(parents=True, exist_ok=True)
            urdf.write_text('<robot name="b601_test"><link name="taccap_eef_link"/></robot>\n')
        with chdir(temp_dir):
            yield
