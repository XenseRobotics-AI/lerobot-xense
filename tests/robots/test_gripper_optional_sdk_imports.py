#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Hold the gripper package to its "no hardware SDK to read a config" promise.

``lerobot.grippers`` documents that config classes carry no hardware-SDK import,
so an arm config parses and ``--robot.type`` registers on a host that has none of
the Xense / TacCap / XGripper wheels installed. Nothing enforced that, and it
silently stopped being true twice:

  - ``grippers/serial/__init__.py`` imported the driver eagerly, so the
    config-only import in ``grippers/__init__.py`` ran it and pulled in xgripper.
  - Upstream, ``xgripper`` imported a tactile-feedback controller at module
    scope, so even the pure-serial driver required xensesdk (fixed in XGripper
    39c7aad).

Neither shows up in a normal test run, because the dev machine has every SDK
installed. These tests hide them at import time to make the promise checkable.
"""

import builtins
import sys

import pytest

# Every wheel a bare host might be missing.
HARDWARE_SDKS = ("xensesdk", "xgripper", "xense.taccap", "libpyelite", "elite_cs_sdk")


@pytest.fixture
def without_hardware_sdks(monkeypatch):
    """Make importing any hardware SDK raise, as it would on a host without them."""
    import lerobot

    real_import = builtins.__import__
    # Reloading a package also replaces its attribute on the parent module.
    # Restoring sys.modules alone leaves camera/discovery tests traversing the
    # temporary package, even though Python restored the original submodules.
    parent = lerobot
    original_grippers = getattr(parent, "grippers", None)
    original_modules = set(sys.modules)

    def _blocked(name, *args, **kwargs):
        if any(name == sdk or name.startswith(sdk + ".") for sdk in HARDWARE_SDKS):
            raise ImportError(f"No module named {name!r} (blocked by test)")
        return real_import(name, *args, **kwargs)

    # Drop anything already imported, plus the gripper package itself, so the
    # import machinery actually re-runs under the block instead of hitting cache.
    for mod in list(sys.modules):
        is_sdk = any(mod == sdk or mod.startswith(sdk + ".") for sdk in HARDWARE_SDKS)
        if is_sdk or mod.startswith("lerobot.grippers"):
            monkeypatch.delitem(sys.modules, mod)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    try:
        yield
    finally:
        for name in set(sys.modules) - original_modules:
            if name.startswith("lerobot.grippers"):
                sys.modules.pop(name, None)
        if original_grippers is None:
            parent.__dict__.pop("grippers", None)
        else:
            parent.grippers = original_grippers


def test_config_classes_import_without_any_sdk(without_hardware_sdks):
    from lerobot.grippers import DHAG95GripperConfig, SerialGripperConfig, TaccapFollowerConfig

    assert SerialGripperConfig(side="left").side == "left"
    assert TaccapFollowerConfig() is not None
    assert DHAG95GripperConfig().type == "dh-ag95"


def test_gripper_types_stay_registered_without_any_sdk(without_hardware_sdks):
    """--robot.type dispatch needs the @register_subclass side effects to have run."""
    from lerobot.grippers import GripperConfig

    assert set(GripperConfig.get_known_choices()) >= {"serial", "taccap_follower", "dh-ag95"}


def test_a_recipe_gripper_block_parses_without_any_sdk(without_hardware_sdks):
    """The end the promise is really about: reading a recipe on a bare host."""
    import draccus

    from lerobot.grippers import GripperConfig

    cfg = draccus.decode(GripperConfig, {"type": "serial", "side": "left"})
    assert cfg.type == "serial"
    assert cfg.side == "left"


@pytest.mark.parametrize("module", ["lerobot.grippers.serial", "lerobot.grippers.taccap", "lerobot.grippers.dh_ag95"])
def test_backend_subpackages_import_without_any_sdk(without_hardware_sdks, module):
    """The half that keeps the laziness honest.

    Both subpackages are reachable from ``lerobot.grippers``'s config-only
    imports, so if either exported its driver eagerly this would raise — which is
    exactly the regression that made the package docstring false.
    """
    import importlib

    assert importlib.import_module(module) is not None


def test_serial_driver_reports_its_missing_sdk_clearly(without_hardware_sdks):
    """Reaching the serial driver without xgripper must fail, and say which module.

    The two backends deliberately differ here: taccap guards its SDK import and
    defers the complaint to connect(), while the serial driver imports xgripper at
    module scope. Either is fine — what matters is that neither is paid for until
    something asks for a driver.
    """
    import lerobot.grippers.serial as serial_pkg

    with pytest.raises(ImportError, match="xgripper"):
        getattr(serial_pkg, "SerialGripper")  # noqa: B009 - the getattr IS the test


def test_taccap_driver_defers_its_complaint_to_connect(without_hardware_sdks):
    """taccap_follower guards its import, so the class object itself resolves."""
    import lerobot.grippers.taccap as taccap_pkg

    driver = taccap_pkg.TaccapFollower  # must not raise
    assert driver.__name__ == "TaccapFollower"


def test_the_factory_itself_needs_no_sdk(without_hardware_sdks):
    """make_gripper_from_config defers per-branch; importing it must stay free."""
    from lerobot.grippers import make_gripper_from_config

    # None means "no gripper on this side" and must not touch any SDK.
    assert make_gripper_from_config(None) is None
