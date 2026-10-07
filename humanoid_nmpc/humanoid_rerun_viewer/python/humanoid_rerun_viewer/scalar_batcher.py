# Copyright (c) 2026, Nicholas Palomo. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Batches scalar rows and sends them to Rerun in few calls.

Logging a ScalarGroup with rerun.log costs about as much as sending a batch of several rows with send_columns, so the
bridge collects rows for a flush period (50 ms by default) and then sends them together:

- ScalarBatcher: rows per entity, each entity's batch with one rerun.send_columns call. The bridge uses it for the
  status scalars, whose entities each have their own times.
- FrameBatcher: frames of rows that share their times across many entities (one TelemetrySeries is one row of each of
  its groups), sent as one Arrow record batch per flush with rerun.send_record_batch. Rerun splits the batch into one
  chunk per entity, with the same columns and metadata as send_columns writes (test_scalar_batcher checks it), for
  less than half the cost of one send_columns call per entity.

Every row carries the robot's clock and the bridge's wall clock; a row without a robot time (a status message that
arrived before any robot time was known) carries the wall clock only and is sent in a batch of its own.
"""

from collections.abc import Iterable, Sequence
import dataclasses

import numpy as np
import pyarrow as pa
import rerun as rr

from humanoid_rerun_viewer import scene_contract

ROBOT_TIMELINE = scene_contract.ROBOT_TIMELINE
WALL_TIMELINE = scene_contract.WALL_TIMELINE

# The most rows one entity buffers before it is sent without waiting for flush().
DEFAULT_MAX_PENDING_ROWS = 1000


@dataclasses.dataclass
class _Batch:
    width: int
    robot_times: list[float]
    wall_times: list[float]
    rows: list[Sequence[float]]


class ScalarBatcher:
    """Buffers rows of scalars and sends them per entity on flush().

    Not thread-safe: the bridge appends and flushes on the bus's receive thread.
    """

    def __init__(
        self,
        recording: rr.RecordingStream,
        max_pending_rows: int = DEFAULT_MAX_PENDING_ROWS,
    ) -> None:
        if max_pending_rows <= 0:
            raise ValueError("max_pending_rows must be positive")
        self._recording = recording
        self._max_pending_rows = max_pending_rows
        # Keyed by (entity path, whether the rows carry a robot time).
        self._batches: dict[tuple[str, bool], _Batch] = {}
        self._pending_rows = 0
        self.rows_sent = 0
        self.batches_sent = 0

    @property
    def pending_rows(self) -> int:
        return self._pending_rows

    def append(
        self,
        path: str,
        robot_time: float | None,
        wall_time: float,
        values: Sequence[float],
    ) -> None:
        """Buffers one row of `values` for the entity `path` at the given times.

        A row of another width than the rows already buffered for `path` sends those first.

        Args:
            path: the entity.
            robot_time: the robot's clock [s], or None when no robot time is known yet.
            wall_time: the bridge's wall clock [s since the epoch].
            values: the row; an empty one is dropped.
        """
        width = len(values)
        if width == 0:
            return
        key = (path, robot_time is not None)
        batch = self._batches.get(key)
        if batch is not None and batch.width != width:
            self._send(key, batch)
            batch = None
        if batch is None:
            batch = _Batch(width=width, robot_times=[], wall_times=[], rows=[])
            self._batches[key] = batch
        if robot_time is not None:
            batch.robot_times.append(robot_time)
        batch.wall_times.append(wall_time)
        batch.rows.append(values)
        self._pending_rows += 1
        if len(batch.rows) >= self._max_pending_rows:
            self._send(key, batch)

    def flush(self) -> int:
        """Sends every buffered row; returns how many."""
        sent = 0
        for key, batch in list(self._batches.items()):
            sent += self._send(key, batch)
        return sent

    def _send(self, key: tuple[str, bool], batch: _Batch) -> int:
        """Sends the rows of `batch`, buffered under `key`, with one send_columns call; returns how many."""
        del self._batches[key]
        count = len(batch.rows)
        if count == 0:
            return 0
        self._pending_rows -= count
        path, has_robot_time = key
        indexes = []
        if has_robot_time:
            indexes.append(
                rr.TimeColumn(
                    ROBOT_TIMELINE,
                    duration=np.asarray(batch.robot_times, dtype=np.float64),
                )
            )
        indexes.append(
            rr.TimeColumn(
                WALL_TIMELINE, timestamp=np.asarray(batch.wall_times, dtype=np.float64)
            )
        )
        self._recording.send_columns(
            path,
            indexes=indexes,
            columns=rr.Scalars.columns(
                scalars=np.asarray(batch.rows, dtype=np.float64).reshape(
                    count, batch.width
                )
            ),
        )
        self.rows_sent += count
        self.batches_sent += 1
        return count


# The Arrow metadata Rerun gives the columns of a Scalars chunk (what send_columns writes; test_scalar_batcher compares).
# LINT.IfChange(scalars_column_metadata)
_INDEX_KIND = b"index"
_SCALARS_FIELD_METADATA = {
    b"rerun:kind": b"data",
    b"rerun:archetype": b"rerun.archetypes.Scalars",
    b"rerun:component": b"Scalars:scalars",
    b"rerun:component_type": b"rerun.components.Scalar",
}
# LINT.ThenChange(//tools/python/operator_requirements.txt:rerun_version)
_SCALARS_TYPE = pa.list_(pa.float64())


def _index_field(timeline: str, arrow_type: pa.DataType) -> pa.Field:
    return pa.field(
        timeline,
        arrow_type,
        nullable=False,
        metadata={b"rerun:kind": _INDEX_KIND, b"rerun:index_name": timeline.encode()},
    )


_ROBOT_TIME_FIELD = _index_field(ROBOT_TIMELINE, pa.duration("ns"))
_WALL_TIME_FIELD = _index_field(WALL_TIMELINE, pa.timestamp("ns"))


def _nanoseconds(seconds: Sequence[float]) -> pa.Array:
    return pa.array(
        np.rint(np.asarray(seconds, dtype=np.float64) * 1e9).astype(np.int64)
    )


@dataclasses.dataclass
class _Column:
    """The rows one entity has in the frames of a FrameBatcher: their frame indices and values."""

    frames: list[int]
    values: list[Sequence[float]]
    # The width of every row, or -1 when they differ.
    width: int


class FrameBatcher:
    """Buffers frames of scalar rows that share their times and sends each flush as one Arrow record batch.

    Not thread-safe: the bridge appends and flushes on the bus's receive thread.
    """

    def __init__(
        self,
        recording: rr.RecordingStream,
        max_pending_frames: int = DEFAULT_MAX_PENDING_ROWS,
    ) -> None:
        if max_pending_frames <= 0:
            raise ValueError("max_pending_frames must be positive")
        self._recording = recording
        self._max_pending_frames = max_pending_frames
        self._robot_times: list[float] = []
        self._wall_times: list[float] = []
        self._columns: dict[str, _Column] = {}
        self._fields: dict[str, pa.Field] = {}
        self.rows_sent = 0
        self.batches_sent = 0

    @property
    def pending_frames(self) -> int:
        return len(self._robot_times)

    def append(
        self,
        robot_time: float,
        wall_time: float,
        rows: Iterable[tuple[str, Sequence[float]]],
    ) -> None:
        """Buffers one frame: for each (entity path, values) of `rows`, one row of that entity at the given times.

        An entity appears at most once per frame; one that is missing from a frame has no row at its times.

        Args:
            robot_time: the robot's clock [s].
            wall_time: the bridge's wall clock [s since the epoch].
            rows: the frame's (entity path, values) pairs.
        """
        frame = len(self._robot_times)
        self._robot_times.append(robot_time)
        self._wall_times.append(wall_time)
        for path, values in rows:
            column = self._columns.get(path)
            if column is None:
                column = _Column(frames=[], values=[], width=len(values))
                self._columns[path] = column
            elif column.width != len(values):
                column.width = -1
            column.frames.append(frame)
            column.values.append(values)
        if len(self._robot_times) >= self._max_pending_frames:
            self.flush()

    def _field(self, path: str) -> pa.Field:
        field = self._fields.get(path)
        if field is None:
            entity = path if path.startswith("/") else f"/{path}"
            metadata = dict(_SCALARS_FIELD_METADATA)
            metadata[b"rerun:entity_path"] = entity.encode()
            field = pa.field(
                f"{entity}:Scalars:scalars", _SCALARS_TYPE, metadata=metadata
            )
            self._fields[path] = field
        return field

    def _scalars_array(self, column: _Column, frames: int) -> pa.Array:
        if len(column.frames) == frames and column.width > 0:
            # Every frame has a row of the same width: one flat array and its offsets, without Python lists.
            values = np.asarray(column.values, dtype=np.float64).reshape(-1)
            offsets = np.arange(
                0, frames * column.width + 1, column.width, dtype=np.int32
            )
            return pa.ListArray.from_arrays(pa.array(offsets), pa.array(values))
        rows: list[list[float] | None] = [None] * frames
        for frame, values in zip(column.frames, column.values):
            rows[frame] = [float(value) for value in values]
        return pa.array(rows, type=_SCALARS_TYPE)

    def flush(self) -> int:
        """Sends every buffered frame; returns how many rows (entity rows, not frames)."""
        frames = len(self._robot_times)
        if frames == 0:
            return 0
        columns = {
            path: column for path, column in self._columns.items() if column.frames
        }
        robot_times, wall_times = self._robot_times, self._wall_times
        self._robot_times, self._wall_times, self._columns = [], [], {}
        if not columns:
            return 0
        fields = [_ROBOT_TIME_FIELD, _WALL_TIME_FIELD]
        arrays = [
            _nanoseconds(robot_times).cast(pa.duration("ns")),
            _nanoseconds(wall_times).cast(pa.timestamp("ns")),
        ]
        rows = 0
        for path, column in columns.items():
            fields.append(self._field(path))
            arrays.append(self._scalars_array(column, frames))
            rows += len(column.frames)
        batch = pa.RecordBatch.from_arrays(arrays, schema=pa.schema(fields))
        rr.send_record_batch(batch, recording=self._recording)
        self.rows_sent += rows
        self.batches_sent += 1
        return rows
