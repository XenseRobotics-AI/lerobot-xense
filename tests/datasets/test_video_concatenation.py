#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import tempfile
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pytest

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.utils import DEFAULT_DATA_PATH, DEFAULT_VIDEO_PATH
from lerobot.datasets.video_utils import (
    VideoTimeBaseError,
    concatenate_video_files,
    get_video_time_bases,
)
from tests.fixtures.constants import DUMMY_HWC, DUMMY_REPO_ID


def _write_video(path, num_frames, timescale, color):
    with av.open(str(path), mode="w", options={"video_track_timescale": str(timescale)}) as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width = 48
        stream.height = 32
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, timescale)
        stream.codec_context.max_b_frames = 0
        stream.codec_context.thread_count = 1
        for index in range(num_frames):
            pixels = np.full((32, 48, 3), color + index, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = index
            frame.time_base = Fraction(1, 30)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _decode_frames(path):
    with av.open(str(path), mode="r") as container:
        return [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]


@pytest.mark.parametrize("timescale", [30, 15360])
@pytest.mark.parametrize("input_type", [Path, str], ids=["paths", "strings"])
def test_repeated_concatenation_preserves_timestamps_and_frames(tmp_path, timescale, input_type):
    paths = [tmp_path / f"input-{index}.mp4" for index in range(3)]
    lengths = [7, 11, 5]
    for index, (path, length) in enumerate(zip(paths, lengths, strict=True)):
        _write_video(path, length, timescale, color=40 + index * 60)
    expected_frames = [frame for path in paths for frame in _decode_frames(path)]
    output = tmp_path / "merged.mp4"

    concatenate_video_files([input_type(path) for path in paths[:2]], output)
    # aggregate_videos appends the next input to the existing destination in place.
    concatenate_video_files([input_type(output), input_type(paths[2])], output)

    with av.open(str(output), mode="r") as container:
        stream = container.streams.video[0]
        assert stream.time_base == Fraction(1, timescale)
        assert stream.frames == sum(lengths)
        assert stream.duration * stream.time_base == Fraction(sum(lengths), 30)
        packets = [packet for packet in container.demux(stream) if packet.dts is not None]
        expected_timestamps = [index * (timescale // 30) for index in range(sum(lengths))]
        assert [packet.dts for packet in packets] == expected_timestamps
        assert [packet.pts for packet in packets] == expected_timestamps

    np.testing.assert_array_equal(_decode_frames(output), expected_frames)


def test_mismatched_time_bases_leave_existing_output_unchanged(tmp_path):
    output = tmp_path / "previously-merged.mp4"
    original = tmp_path / "original.mp4"
    _write_video(output, 7, timescale=15360, color=40)
    _write_video(original, 5, timescale=30, color=100)
    previous_bytes = output.read_bytes()

    with pytest.raises(ValueError, match="Video time_base mismatch") as error:
        concatenate_video_files([output, original], output)

    assert str(original) in str(error.value)
    assert "1/30" in str(error.value)
    assert "1/15360" in str(error.value)
    assert output.read_bytes() == previous_bytes


def _other_timescale(path):
    """A 1/N timescale that differs from the (single) video stream of ``path``."""
    (time_base,) = get_video_time_bases(path)
    return 90000 if time_base != Fraction(1, 90000) else 30


def _write_audio_only(path):
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("pcm_s16le", rate=8000)
        frame = av.AudioFrame.from_ndarray(np.zeros((1, 800), dtype=np.int16), format="s16", layout="mono")
        frame.sample_rate = 8000
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _write_two_stream_video(path, num_frames):
    # Without video_track_timescale the MP4 muxer gives each track its own timescale.
    with av.open(str(path), mode="w") as container:
        streams = []
        for rate in (30, 25):
            stream = container.add_stream("libx264", rate=rate)
            stream.width, stream.height, stream.pix_fmt = 48, 32, "yuv420p"
            stream.codec_context.max_b_frames = 0
            streams.append((stream, rate))
        for index in range(num_frames):
            for stream, rate in streams:
                frame = av.VideoFrame.from_ndarray(np.full((32, 48, 3), 40 + index, dtype=np.uint8), format="rgb24")
                frame.pts = index
                frame.time_base = Fraction(1, rate)
                for packet in stream.encode(frame):
                    container.mux(packet)
        for stream, _ in streams:
            for packet in stream.encode():
                container.mux(packet)


def test_input_without_video_stream_raises_clear_error(tmp_path):
    video = tmp_path / "video.mp4"
    audio_only = tmp_path / "audio.wav"
    _write_video(video, 3, timescale=30, color=40)
    _write_audio_only(audio_only)

    with pytest.raises(ValueError, match="No video stream found in .*audio.wav"):
        concatenate_video_files([video, audio_only], tmp_path / "merged.mp4")


def test_every_video_stream_time_base_is_checked(tmp_path):
    video = tmp_path / "two-streams.mp4"
    _write_two_stream_video(video, 3)
    with av.open(str(video)) as container:
        assert len({stream.time_base for stream in container.streams.video}) == 2  # precondition

    with pytest.raises(ValueError, match=r"Video time_base mismatch: .* \(video stream 1\)"):
        concatenate_video_files([video, video], tmp_path / "merged.mp4")
    assert not (tmp_path / "merged.mp4").exists()


def test_aggregate_rejects_time_base_mismatch_before_writing(tmp_path, lerobot_dataset_factory):
    ds_0 = lerobot_dataset_factory(
        root=tmp_path / "ds_0", repo_id=f"{DUMMY_REPO_ID}_0", total_episodes=2, total_frames=20
    )
    ds_1 = lerobot_dataset_factory(
        root=tmp_path / "ds_1", repo_id=f"{DUMMY_REPO_ID}_1", total_episodes=2, total_frames=20
    )
    # Give one video of the *later* source a different time base.
    video_key = ds_1.meta.video_keys[-1]
    episode = ds_1.meta.episodes[0]
    bad_video = ds_1.root / DEFAULT_VIDEO_PATH.format(
        video_key=video_key,
        chunk_index=episode[f"videos/{video_key}/chunk_index"],
        file_index=episode[f"videos/{video_key}/file_index"],
    )
    _write_video(bad_video, 3, _other_timescale(bad_video), color=40)
    aggr_root = tmp_path / "aggr"

    with pytest.raises(VideoTimeBaseError, match=r"Video time_base mismatch: .*ds_1"):
        aggregate_datasets(
            repo_ids=[ds_0.repo_id, ds_1.repo_id],
            roots=[ds_0.root, ds_1.root],
            aggr_repo_id=f"{DUMMY_REPO_ID}_aggr",
            aggr_root=aggr_root,
        )
    # Nothing was written, so fixing the input and re-running is not blocked by a half-built root.
    assert not aggr_root.exists()


def test_recording_starts_new_video_file_on_time_base_mismatch(tmp_path, empty_lerobot_dataset_factory):
    key = "observation.images.cam"
    features = {key: {"dtype": "video", "shape": DUMMY_HWC, "names": ["height", "width", "channels"]}}
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "rec", features=features, streaming_encoding=False)

    def record_episode():
        for _ in range(3):
            dataset.add_frame({key: np.random.rand(*DUMMY_HWC), "task": "Dummy task"})
        dataset.save_episode()

    record_episode()
    first_video = dataset.root / DEFAULT_VIDEO_PATH.format(video_key=key, chunk_index=0, file_index=0)
    timescale = _other_timescale(first_video)

    def encode_with_other_time_base(video_key, episode_index):
        path = Path(tempfile.mkdtemp(dir=dataset.root)) / f"{video_key}_{episode_index:03d}.mp4"
        _write_video(path, 3, timescale, color=100)
        return path

    # The episode's frames reach the data parquet before its video is saved, so the save must
    # not raise on a time_base mismatch: it starts a new video file instead.
    dataset._encode_temporary_episode_video = encode_with_other_time_base
    record_episode()
    dataset.finalize()

    episodes = pd.concat(
        [pd.read_parquet(path) for path in sorted((dataset.root / "meta" / "episodes").rglob("*.parquet"))]
    )
    assert episodes["episode_index"].tolist() == [0, 1]
    assert episodes[f"videos/{key}/file_index"].tolist() == [0, 1]
    assert episodes[f"videos/{key}/from_timestamp"].tolist()[1] == 0.0
    data = pd.read_parquet(dataset.root / DEFAULT_DATA_PATH.format(chunk_index=0, file_index=0))
    assert sorted(data["episode_index"].unique()) == [0, 1]
    second_video = dataset.root / DEFAULT_VIDEO_PATH.format(video_key=key, chunk_index=0, file_index=1)
    assert get_video_time_bases(second_video) == [Fraction(1, timescale)]
