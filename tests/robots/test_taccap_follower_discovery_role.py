#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Follower lookup never takes a leader of the same side (no hardware).

The bench this guards: left leader TCGU01A28Z0095m and left follower
TCGU01A45A0017s on one host. The SDK's role-less find_left() returns whichever
scans first.
"""

from types import SimpleNamespace

import pytest

from lerobot.grippers.taccap.discovery import find_follower_on_side

LEADER = SimpleNamespace(firmware_sn="TCGU01A28Z0095m", mcu_device="/dev/leader", role="m")
FOLLOWER_L = SimpleNamespace(firmware_sn="TCGU01A45A0017s", mcu_device="/dev/follower-l", role="s")
FOLLOWER_R = SimpleNamespace(firmware_sn="TCGU01A45A0016s", mcu_device="/dev/follower-r", role="s")
ROLE = SimpleNamespace(Follower="s", Leader="m")


def _sdk_with_role_filter(calls):
    """SDK >= the role filter: the leader scans first, as on the bench."""

    def pick(units, role=None):
        calls.append(role)
        for u in units:
            if role is None or u.role == role:
                return u
        raise OSError("no gripper")

    return SimpleNamespace(
        Role=ROLE,
        find_left=lambda role=None: pick([LEADER, FOLLOWER_L], role),
        find_right=lambda role=None: pick([FOLLOWER_R], role),
    )


def test_passes_follower_role_and_skips_leader_scanned_first():
    calls = []
    sdk = _sdk_with_role_filter(calls)
    assert find_follower_on_side(sdk, "left") is FOLLOWER_L
    assert find_follower_on_side(sdk, "right") is FOLLOWER_R
    assert calls == ["s", "s"]


def test_old_extension_returning_leader_is_refused():
    sdk = SimpleNamespace(Role=ROLE, find_left=lambda: LEADER, find_right=lambda: FOLLOWER_R)
    with pytest.raises(RuntimeError, match="returned leader TCGU01A28Z0095m"):
        find_follower_on_side(sdk, "left")


def test_old_extension_returning_follower_still_works():
    sdk = SimpleNamespace(Role=ROLE, find_left=lambda: FOLLOWER_L, find_right=lambda: FOLLOWER_R)
    assert find_follower_on_side(sdk, "left") is FOLLOWER_L


def test_no_follower_on_side_propagates_the_sdk_error():
    calls = []
    sdk = _sdk_with_role_filter(calls)
    sdk.find_left = lambda role=None: (_ for _ in ()).throw(OSError("no left-side Follower gripper"))
    with pytest.raises(OSError, match="Follower"):
        find_follower_on_side(sdk, "left")
