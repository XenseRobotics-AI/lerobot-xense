#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Arm-agnostic gripper drivers, shared across robot packages.

Layout mirrors ``lerobot.cameras``: a ``Gripper`` ABC and a ``GripperConfig``
choice registry at the top, one subpackage per backend underneath.

Config classes are imported eagerly — they carry no hardware-SDK import, so arm
configs and ``--robot.type`` registration keep working on a host without the
Xense/TacCap/XGripper SDKs. Driver classes are NOT imported here; get one through
``make_gripper_from_config``, which imports only the branch it needs.

That promise reaches through each backend's ``__init__`` too, which is easy to
break by adding one convenient re-export: importing a submodule runs its package
first, so an eager driver import there would drag the SDK in for anyone who only
parsed a config. ``tests/robots/test_gripper_optional_sdk_imports.py`` hides the
SDKs at import time and fails if that happens again — it has happened twice.
"""

from .configs import GripperConfig  # noqa: F401
from .dh_ag95.configuration_dh_ag95 import DHAG95GripperConfig  # noqa: F401
from .gripper import Gripper  # noqa: F401

# Config-only imports: safe without any hardware SDK present, and required so the
# @register_subclass side effects run before a config is parsed or dispatched.
from .serial.configuration_serial import SerialGripperConfig  # noqa: F401
from .taccap.configuration_taccap import TaccapFollowerConfig  # noqa: F401
from .utils import command_gripper_position, make_gripper_from_config  # noqa: F401
