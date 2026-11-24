#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import numpy as np
import pytest

from lerobot.cameras.xense.camera_xense import XenseTactileCamera, to_landscape

# --- orientation ------------------------------------------------------------
# The stored tactile image must be the SDK portrait rotated 90° clockwise, the
# same as the backpack collector (taccap-core RECTIFY_ROTATE_DEG = 270). It used
# to be a transpose, which is that rotation plus a left-right flip: every image
# recorded through this driver was mirrored.


def _portrait(channels: int | None) -> np.ndarray:
    """A 700x400 SDK-shaped image whose top-left pixel is unique."""
    shape = (700, 400) if channels is None else (700, 400, channels)
    image = np.zeros(shape, dtype=np.float32)
    image[0, 0] = 1.0
    return image


@pytest.mark.parametrize("channels", [None, 3])
def test_to_landscape_is_a_clockwise_rotation_not_a_mirror(channels):
    image = _portrait(channels)
    landscape = to_landscape(image)

    assert landscape.shape[:2] == (400, 700)
    # clockwise: the portrait's top-left corner ends up at the top-right
    assert landscape[0, 699].max() == 1.0
    assert landscape[0, 0].max() == 0.0
    assert np.array_equal(landscape, np.rot90(image, -1))
    # the old transpose put it at the top-left, i.e. the mirror of this
    assert np.array_equal(np.transpose(image, (1, 0) + tuple(range(2, image.ndim))), landscape[:, ::-1])


def test_read_sensor_data_rotates_every_image_output_the_same_way():
    from lerobot.cameras.xense.configuration_xense import XenseOutputType

    rectify, depth = _portrait(3), _portrait(None)
    force = np.ones((35, 20, 3), dtype=np.float32)

    class FakeOutputType:
        """Any SDK output type the driver maps resolves to its own name."""

        def __getattr__(self, name):
            return name.lower()

    class FakeSensorModule:
        OutputType = FakeOutputType()

    class FakeSensor:
        def selectSensorInfo(self, *types):  # noqa: N802 - SDK spelling
            return (
                {"rectify": rectify, "depth": depth, "force": force}[types[0]]
                if len(types) == 1
                else (rectify, depth, force)
            )

    camera = object.__new__(XenseTactileCamera)
    camera.serial_number = "GSPS01TEST"
    camera.sensor = FakeSensor()
    camera._Sensor = FakeSensorModule
    camera._sensor_output_types_cache = None
    camera._sensor_output_types_cache_key = None
    type(camera).is_connected = property(lambda self: True)

    camera.output_types = [XenseOutputType.RECTIFY, XenseOutputType.DEPTH, XenseOutputType.FORCE]
    got_rectify, got_depth, got_force = camera._read_sensor_data()
    assert np.array_equal(got_rectify, np.rot90(rectify, -1))
    assert np.array_equal(got_depth - np.float32(0.01), np.rot90(depth, -1))
    assert got_force is force  # non-image outputs are not reoriented

    camera.output_types = [XenseOutputType.RECTIFY]
    camera._sensor_output_types_cache = None
    assert np.array_equal(camera._read_sensor_data(), np.rot90(rectify, -1))
