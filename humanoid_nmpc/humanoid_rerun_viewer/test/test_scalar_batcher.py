"""The scalar batchers: ScalarBatcher sends each entity's rows in one send_columns call; FrameBatcher sends frames of
many entities as one record batch, which Rerun stores exactly as send_columns' chunks. Both on both timelines, in
order."""

import os
import shutil
import tempfile
import unittest
from typing import List

import numpy as np
import rerun as rr

import rrd_contents
from humanoid_rerun_viewer import bridge
from humanoid_rerun_viewer import scalar_batcher
from humanoid_rerun_viewer import scene_contract

ROBOT_TIMELINE = scene_contract.ROBOT_TIMELINE
WALL_TIMELINE = scene_contract.WALL_TIMELINE

WALL = 1.7e9


class RecordingSpy:
    """A RecordingStream stand-in that records the send_columns calls."""

    def __init__(self) -> None:
        self.calls: List[str] = []

    def send_columns(self, path: str, indexes, columns) -> None:
        del indexes, columns
        self.calls.append(path)


class ScalarBatcherTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.directory)
        self.path = os.path.join(self.directory, "scalars.rrd")
        self.recording = bridge.new_recording("test_scalar_batcher")
        self.recording.save(self.path)

    def contents(self) -> rrd_contents.RrdContents:
        self.recording.flush()
        self.recording.disconnect()
        return rrd_contents.RrdContents(self.path)

    def test_rows_arrive_in_order_on_both_timelines(self) -> None:
        batcher = scalar_batcher.ScalarBatcher(self.recording)
        for step in range(5):
            batcher.append("telemetry/a", 0.01 * step, WALL + step, [step, -step])
        batcher.append("telemetry/b", 0.0, WALL, [7.0])
        self.assertEqual(batcher.pending_rows, 6)
        self.assertEqual(batcher.flush(), 6)
        self.assertEqual(batcher.pending_rows, 0)
        self.assertEqual((batcher.rows_sent, batcher.batches_sent), (6, 2))
        contents = self.contents()
        a = contents.entities["telemetry/a"]
        self.assertEqual(a.timelines, {ROBOT_TIMELINE, WALL_TIMELINE})
        self.assertEqual(
            a.sorted_values("Scalars:scalars", ROBOT_TIMELINE),
            [[step, -step] for step in range(5)],
        )
        for actual, step in zip(a.times(ROBOT_TIMELINE), range(5)):
            self.assertAlmostEqual(actual, 0.01 * step)
        for actual, step in zip(a.times(WALL_TIMELINE), range(5)):
            self.assertAlmostEqual(actual, WALL + step, places=3)
        self.assertEqual(contents.values("telemetry/b", "Scalars:scalars"), [[7.0]])

    def test_one_call_per_entity_and_flush(self) -> None:
        spy = RecordingSpy()
        batcher = scalar_batcher.ScalarBatcher(spy)  # type: ignore[arg-type]
        for step in range(10):
            batcher.append("a", float(step), WALL, [1.0, 2.0])
            batcher.append("b", float(step), WALL, [1.0])
        batcher.flush()
        self.assertEqual(sorted(spy.calls), ["a", "b"])
        self.assertEqual(batcher.flush(), 0)

    def test_a_change_of_width_sends_the_rows_before_it(self) -> None:
        spy = RecordingSpy()
        batcher = scalar_batcher.ScalarBatcher(spy)  # type: ignore[arg-type]
        batcher.append("a", 0.0, WALL, [1.0, 2.0])
        batcher.append("a", 0.1, WALL, [1.0, 2.0, 3.0])
        self.assertEqual(spy.calls, ["a"])
        self.assertEqual(batcher.pending_rows, 1)

    def test_a_full_entity_is_sent_without_waiting(self) -> None:
        spy = RecordingSpy()
        batcher = scalar_batcher.ScalarBatcher(
            spy, max_pending_rows=3
        )  # type: ignore[arg-type]
        for step in range(7):
            batcher.append("a", float(step), WALL, [1.0])
        self.assertEqual(spy.calls, ["a", "a"])
        self.assertEqual(batcher.pending_rows, 1)

    def test_rows_without_a_robot_time_carry_the_wall_clock_only(self) -> None:
        batcher = scalar_batcher.ScalarBatcher(self.recording)
        batcher.append("status/x", None, WALL, [1.0])
        batcher.append("status/x", 2.0, WALL + 1.0, [2.0])
        batcher.flush()
        contents = self.contents()
        entity = contents.entities["status/x"]
        self.assertEqual(sorted(entity.values("Scalars:scalars")), [[1.0], [2.0]])
        self.assertEqual(entity.timelines, {ROBOT_TIMELINE, WALL_TIMELINE})
        self.assertEqual(entity.times(ROBOT_TIMELINE), [2.0])

    def test_empty_rows_are_ignored_and_options_checked(self) -> None:
        spy = RecordingSpy()
        batcher = scalar_batcher.ScalarBatcher(spy)  # type: ignore[arg-type]
        batcher.append("a", 0.0, WALL, [])
        self.assertEqual(batcher.flush(), 0)
        with self.assertRaises(ValueError):
            scalar_batcher.ScalarBatcher(spy, max_pending_rows=0)  # type: ignore[arg-type]


class FrameBatcherTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.directory)

    def recording(self, name: str):
        path = os.path.join(self.directory, f"{name}.rrd")
        recording = bridge.new_recording(f"test_frame_batcher_{name}")
        recording.save(path)
        return recording, path

    def contents(self, recording, path: str) -> rrd_contents.RrdContents:
        recording.flush()
        recording.disconnect()
        return rrd_contents.RrdContents(path)

    def test_every_entity_gets_its_rows_on_both_timelines(self) -> None:
        recording, path = self.recording("frames")
        batcher = scalar_batcher.FrameBatcher(recording)
        for step in range(5):
            batcher.append(
                0.01 * step,
                WALL + step,
                [("telemetry/a", [step, -step]), ("telemetry/b", [10.0 * step])],
            )
        self.assertEqual(batcher.pending_frames, 5)
        self.assertEqual(batcher.flush(), 10)
        self.assertEqual((batcher.rows_sent, batcher.batches_sent), (10, 1))
        self.assertEqual(batcher.flush(), 0)
        contents = self.contents(recording, path)
        a = contents.entities["telemetry/a"]
        self.assertEqual(a.timelines, {ROBOT_TIMELINE, WALL_TIMELINE})
        self.assertEqual(
            a.sorted_values("Scalars:scalars", ROBOT_TIMELINE),
            [[step, -step] for step in range(5)],
        )
        for actual, step in zip(a.times(ROBOT_TIMELINE), range(5)):
            self.assertAlmostEqual(actual, 0.01 * step)
        for actual, step in zip(a.times(WALL_TIMELINE), range(5)):
            self.assertAlmostEqual(actual, WALL + step, places=3)
        self.assertEqual(
            contents.entities["telemetry/b"].sorted_values(
                "Scalars:scalars", ROBOT_TIMELINE
            ),
            [[10.0 * step] for step in range(5)],
        )

    def test_the_chunks_equal_those_of_send_columns(self) -> None:
        frames_recording, frames_path = self.recording("frames")
        columns_recording, columns_path = self.recording("columns")
        batcher = scalar_batcher.FrameBatcher(frames_recording)
        rows = [[1.0, 2.0], [3.0, 4.0]]
        for step, values in enumerate(rows):
            batcher.append(float(step), WALL + step, [("telemetry/x", values)])
        batcher.flush()
        columns_recording.send_columns(
            "telemetry/x",
            indexes=[
                rr.TimeColumn(ROBOT_TIMELINE, duration=[0.0, 1.0]),
                rr.TimeColumn(WALL_TIMELINE, timestamp=[WALL, WALL + 1.0]),
            ],
            columns=rr.Scalars.columns(scalars=np.array(rows)),
        )
        frames = self.contents(frames_recording, frames_path).entities["telemetry/x"]
        columns = self.contents(columns_recording, columns_path).entities["telemetry/x"]
        (frames_chunk,) = frames.chunks
        (columns_chunk,) = columns.chunks
        self.assertEqual(
            frames_chunk.batch.schema.remove_metadata(),
            columns_chunk.batch.schema.remove_metadata(),
        )
        for name in columns_chunk.batch.schema.names:
            expected = dict(columns_chunk.batch.schema.field(name).metadata or {})
            actual = dict(frames_chunk.batch.schema.field(name).metadata or {})
            # Sortedness is Rerun's own bookkeeping of what it wrote.
            expected.pop(b"rerun:is_sorted", None)
            actual.pop(b"rerun:is_sorted", None)
            self.assertEqual(actual, expected, name)
        self.assertEqual(
            frames.sorted_values("Scalars:scalars", ROBOT_TIMELINE),
            columns.sorted_values("Scalars:scalars", ROBOT_TIMELINE),
        )
        self.assertEqual(frames.times(WALL_TIMELINE), columns.times(WALL_TIMELINE))

    def test_missing_rows_and_changing_widths(self) -> None:
        recording, path = self.recording("sparse")
        batcher = scalar_batcher.FrameBatcher(recording)
        batcher.append(0.0, WALL, [("telemetry/a", [1.0]), ("telemetry/b", [1.0])])
        batcher.append(0.1, WALL, [("telemetry/a", [2.0, 3.0])])
        batcher.append(0.2, WALL, [("telemetry/b", [4.0])])
        self.assertEqual(batcher.flush(), 4)
        contents = self.contents(recording, path)
        a = contents.entities["telemetry/a"]
        b = contents.entities["telemetry/b"]
        self.assertEqual(
            [
                value
                for value in a.sorted_values("Scalars:scalars", ROBOT_TIMELINE)
                if value is not None
            ],
            [[1.0], [2.0, 3.0]],
        )
        self.assertEqual(
            [
                value
                for value in b.sorted_values("Scalars:scalars", ROBOT_TIMELINE)
                if value is not None
            ],
            [[1.0], [4.0]],
        )

    def test_a_full_batcher_sends_without_waiting(self) -> None:
        recording, path = self.recording("full")
        batcher = scalar_batcher.FrameBatcher(recording, max_pending_frames=3)
        for step in range(7):
            batcher.append(float(step), WALL, [("telemetry/a", [1.0])])
        self.assertEqual(batcher.batches_sent, 2)
        self.assertEqual(batcher.pending_frames, 1)
        with self.assertRaises(ValueError):
            scalar_batcher.FrameBatcher(recording, max_pending_frames=0)
        recording.disconnect()


if __name__ == "__main__":
    unittest.main()
