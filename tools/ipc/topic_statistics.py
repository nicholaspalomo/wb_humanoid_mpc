"""Statistics of the messages a listener receives: per-topic rate, period jitter and size (`hz`), and the census of
every topic seen on the bus (`list`).

Times are local receive times (time.monotonic), so the statistics describe the stream as it arrives here, after the
network: the rate and jitter of a topic published at a steady period include whatever the link adds.
"""

import collections
import dataclasses
import math
from typing import Deque, Dict, List, NamedTuple, Optional, Set, Tuple

KIBIBYTE = 1024.0


class RateSnapshot(NamedTuple):
    """The statistics of the messages in the window, at the time of the snapshot."""

    window_count: int
    rate_hz: float
    mean_period_s: float
    min_period_s: float
    max_period_s: float
    # Standard deviation of the period between consecutive messages.
    jitter_s: float
    mean_size_bytes: float
    min_size_bytes: int
    max_size_bytes: int
    bandwidth_bytes_per_s: float


class RateStatistics:
    """Rate, period jitter and message size over a sliding window of the last `window_size` messages."""

    def __init__(self, window_size: int) -> None:
        if window_size < 2:
            raise ValueError(
                f"the window must hold at least two messages, got {window_size}"
            )
        self._samples: Deque[Tuple[float, int]] = collections.deque(maxlen=window_size)
        self.total_count = 0

    def add(self, receive_time: float, size_bytes: int) -> None:
        self._samples.append((receive_time, size_bytes))
        self.total_count += 1

    def snapshot(self) -> Optional[RateSnapshot]:
        """The statistics of the window, or None until it holds two messages that arrived at different times."""
        if len(self._samples) < 2:
            return None
        times = [sample[0] for sample in self._samples]
        sizes = [sample[1] for sample in self._samples]
        periods = [later - earlier for earlier, later in zip(times, times[1:])]
        span = times[-1] - times[0]
        if span <= 0.0:
            return None
        mean_period = span / len(periods)
        variance = sum((period - mean_period) ** 2 for period in periods) / len(periods)
        return RateSnapshot(
            window_count=len(self._samples),
            rate_hz=1.0 / mean_period,
            mean_period_s=mean_period,
            min_period_s=min(periods),
            max_period_s=max(periods),
            jitter_s=math.sqrt(variance),
            mean_size_bytes=sum(sizes) / len(sizes),
            min_size_bytes=min(sizes),
            max_size_bytes=max(sizes),
            # The bytes that arrived over the span: every message but the one that opened it.
            bandwidth_bytes_per_s=sum(sizes[1:]) / span,
        )


def format_bytes(size: float) -> str:
    """A byte count with a binary unit: 512 B, 40.0 KiB, 1.2 MiB."""
    if size < KIBIBYTE:
        return f"{size:.0f} B"
    if size < KIBIBYTE * KIBIBYTE:
        return f"{size / KIBIBYTE:.1f} KiB"
    return f"{size / (KIBIBYTE * KIBIBYTE):.1f} MiB"


def format_snapshot(topic: str, snapshot: RateSnapshot) -> str:
    """Three lines: the average rate, then the periods and the window, then the message size and the bandwidth."""
    return (
        f"{topic}: average rate {snapshot.rate_hz:.3f} Hz\n"
        f"  period min {snapshot.min_period_s * 1e3:.3f} ms  max {snapshot.max_period_s * 1e3:.3f} ms  "
        f"jitter (std dev) {snapshot.jitter_s * 1e3:.3f} ms  window {snapshot.window_count}\n"
        f"  size mean {format_bytes(snapshot.mean_size_bytes)}  min {format_bytes(snapshot.min_size_bytes)}  "
        f"max {format_bytes(snapshot.max_size_bytes)}  bandwidth {format_bytes(snapshot.bandwidth_bytes_per_s)}/s"
    )


@dataclasses.dataclass
class TopicSummary:
    """What `list` reports about one topic."""

    topic: str
    type_names: Set[str]
    count: int
    first_receive_time: float
    last_receive_time: float

    @property
    def rate_hz(self) -> Optional[float]:
        span = self.last_receive_time - self.first_receive_time
        if self.count < 2 or span <= 0.0:
            return None
        return (self.count - 1) / span


class TopicCensus:
    """Every topic seen, with its type names (one, unless two publishers disagree) and how often it arrived."""

    def __init__(self) -> None:
        self._topics: Dict[str, TopicSummary] = {}

    def add(self, topic: str, type_name: str, receive_time: float) -> None:
        summary = self._topics.get(topic)
        if summary is None:
            self._topics[topic] = TopicSummary(
                topic=topic,
                type_names={type_name},
                count=1,
                first_receive_time=receive_time,
                last_receive_time=receive_time,
            )
            return
        summary.type_names.add(type_name)
        summary.count += 1
        summary.last_receive_time = receive_time

    def summaries(self) -> List[TopicSummary]:
        """The topics, sorted by name."""
        return [self._topics[topic] for topic in sorted(self._topics)]


def format_census(summaries: List[TopicSummary]) -> str:
    """A table of topic, type and rate, one topic per line."""
    rows = [("TOPIC", "TYPE", "MESSAGES", "RATE")]
    for summary in summaries:
        rate = summary.rate_hz
        rows.append(
            (
                summary.topic,
                ", ".join(sorted(summary.type_names)),
                str(summary.count),
                "-" if rate is None else f"{rate:.1f} Hz",
            )
        )
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    return "\n".join(
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip()
        for row in rows
    )
