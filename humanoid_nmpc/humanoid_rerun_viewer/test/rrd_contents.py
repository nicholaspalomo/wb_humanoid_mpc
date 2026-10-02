"""Reads back what a recording wrote to an .rrd file, for the tests: entities, components, rows, the blueprint.

Values are converted to Python lists only when asked for, so that a recording with large meshes stays cheap to read.
The chunks of an entity are kept in file order, which is not the order of time: sorted_values() orders by a timeline.
"""

import collections
import dataclasses
from typing import Any, Dict, List, Set, Tuple

import pyarrow as pa
from rerun import chunk as rerun_chunk

# The columns of a chunk that are not components.
_CONTROL_COLUMNS = frozenset({"rerun.controls.RowId"})


@dataclasses.dataclass
class _Chunk:
    is_static: bool
    timelines: Set[str]
    batch: pa.RecordBatch


@dataclasses.dataclass
class EntityContents:
    """What one entity holds: static and temporal components, rows per component, and the timelines used."""

    static_components: Set[str] = dataclasses.field(default_factory=set)
    temporal_components: Set[str] = dataclasses.field(default_factory=set)
    rows: Dict[str, int] = dataclasses.field(
        default_factory=lambda: collections.defaultdict(int)
    )
    timelines: Set[str] = dataclasses.field(default_factory=set)
    chunks: List[_Chunk] = dataclasses.field(default_factory=list)

    @property
    def components(self) -> Set[str]:
        return self.static_components | self.temporal_components

    def columns(self, column: str) -> List[pa.Array]:
        """The arrays of a column, one per chunk that has it, in file order."""
        return [
            chunk.batch.column(column)
            for chunk in self.chunks
            if column in chunk.batch.schema.names
        ]

    def values(self, column: str) -> List[Any]:
        """Every row of a component or timeline column, in file order, as Python values."""
        result: List[Any] = []
        for array in self.columns(column):
            result.extend(array.to_pylist())
        return result

    def _timed(self, column: str, timeline: str) -> List[Tuple[float, Any]]:
        rows: List[Tuple[float, Any]] = []
        for chunk in self.chunks:
            names = chunk.batch.schema.names
            if column not in names or timeline not in names:
                continue
            times = chunk.batch.column(timeline).cast(pa.int64()).to_pylist()
            rows.extend(
                zip(
                    (nanoseconds / 1e9 for nanoseconds in times),
                    chunk.batch.column(column).to_pylist(),
                )
            )
        return rows

    def sorted_values(self, column: str, timeline: str) -> List[Any]:
        """The rows of a component that carry `timeline`, ordered by it (stable for equal times)."""
        return [
            value
            for _, value in sorted(
                self._timed(column, timeline), key=lambda row: row[0]
            )
        ]

    def times(self, timeline: str) -> List[float]:
        """Every row's time on a duration or timestamp timeline, in seconds, in time order."""
        result: List[float] = []
        for chunk in self.chunks:
            if timeline in chunk.batch.schema.names:
                result.extend(
                    nanoseconds / 1e9
                    for nanoseconds in chunk.batch.column(timeline)
                    .cast(pa.int64())
                    .to_pylist()
                )
        return sorted(result)


def _strip(entity_path: str) -> str:
    return entity_path.lstrip("/")


def _read(chunks, entities: Dict[str, EntityContents]) -> Set[str]:
    all_timelines: Set[str] = set()
    for chunk in chunks:
        entity = entities[_strip(chunk.entity_path)]
        batch = chunk.to_record_batch()
        timelines = set(chunk.timeline_names)
        entity.timelines |= timelines
        all_timelines |= timelines
        entity.chunks.append(_Chunk(chunk.is_static, timelines, batch))
        for name in batch.schema.names:
            if name in _CONTROL_COLUMNS or name in timelines:
                continue
            if chunk.is_static:
                entity.static_components.add(name)
            else:
                entity.temporal_components.add(name)
            entity.rows[name] += chunk.num_rows
    return all_timelines


class RrdContents:
    """The recording store and the blueprint store(s) of an .rrd file."""

    def __init__(self, path: str) -> None:
        reader = rerun_chunk.RrdReader(path)
        self.recordings = reader.recordings()
        self.blueprints = reader.blueprints()
        self.entities: Dict[str, EntityContents] = collections.defaultdict(
            EntityContents
        )
        self.timelines = _read(reader.stream().to_chunks(), self.entities)
        self.blueprint_entities: Dict[str, EntityContents] = collections.defaultdict(
            EntityContents
        )
        for store in self.blueprints:
            _read(reader.stream(store=store).to_chunks(), self.blueprint_entities)

    def entity_paths(self) -> Set[str]:
        return set(self.entities)

    def has(self, entity_path: str, component: str) -> bool:
        entity = self.entities.get(entity_path)
        return entity is not None and component in entity.components

    def values(self, entity_path: str, column: str) -> List[Any]:
        return self.entities[entity_path].values(column)

    def blueprint_values(self, component: str) -> List[Any]:
        """Every value of a blueprint component (e.g. "ViewBlueprint:display_name") over the blueprint stores."""
        result: List[Any] = []
        for entity in self.blueprint_entities.values():
            result.extend(entity.values(component))
        return result

    def view_names(self) -> Set[str]:
        names: Set[str] = set()
        for value in self.blueprint_values("ViewBlueprint:display_name"):
            names.update(value if isinstance(value, list) else [value])
        return names
