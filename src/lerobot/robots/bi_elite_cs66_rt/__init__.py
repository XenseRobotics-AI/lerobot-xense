#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import contextlib

from .config_bi_elite_cs66_rt import BiEliteCS66RTConfig, BiEliteCS66RTControlMode  # noqa: F401

# Guarded: the driver pulls in the xense gripper stack (xgripper/xensesdk), so
# importing the CLI on a host without it must not crash; libpyelite itself is only
# imported once the robot is built or connected. The SDK-free config above stays
# registered as an --robot.type choice regardless.
with contextlib.suppress(ImportError):
    from .bi_elite_cs66_rt import BiEliteCS66RT  # noqa: F401
