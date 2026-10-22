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

"""Maps the bus's visualization and status messages onto a Rerun recording.

    recording = new_recording("humanoid_nmpc")
    bridge = RerunBridge(recording, robot_model)
    bridge.log_static()                    # the robot meshes of every instance, the world's axes
    bridge.handle_scene(scene)             # viz/scene: link poses and markers
    bridge.handle_telemetry(series)        # viz/telemetry: buffered, sent by flush()
    bridge.flush()                         # every ~50 ms
    bridge.handle_fsm_state(state)         # robot/fsm_state, mpc/status, robot/loop_timing: text logs and scalars

The handlers run on one thread (bus_bridge's writer thread, never the bus's receive thread, so that a sink that takes no
data cannot stop the bus). None of them raises: a message that does not fit the contract (scene_contract,
telemetry_contract) is counted under its topic and a reason in `statistics.malformed`, and logged at most once per
LOG_PERIOD_S for each topic and reason; an unexpected error is counted in `statistics.handler_errors` the same way.

A robot clock that goes back by more than NEW_RUN_REWIND_S is a new run of the robot (a restarted robot process or
simulation). Given `new_run`, the bridge then records it as a recording of its own, which the viewer shows apart from
the previous run's: one recording would put both runs on one robot_time timeline, where the time panel, which follows
the latest time, would show the previous run's end until the new run's clock passed it.

The robot's clock is the times of one stream, the clock topic: viz/telemetry, one message per robot/state sample in the
order the robot sampled them, or viz/scene while no telemetry arrives. A message of another topic never starts a run:
an MpcStatus trails the robot by a solve, and a scene of "latest" delivery may have overtaken older telemetry in the
queues. A run starts when NEW_RUN_CONFIRMATIONS messages of the clock topic in a row are behind the clock by more than
NEW_RUN_REWIND_S. Until then, every message that far behind waits (at most MAX_HELD_MESSAGES of them): once the clock
topic confirms the new run they are recorded into it, and once it shows that the robot's clock did not go back, into
the current run, at their own times. A single late message (a straggler) therefore never starts a recording.

viz/telemetry is recorded at most at `telemetry_rate` [Hz] of robot time (the first sample of every 1 / rate interval)
and only for the groups `telemetry_selection` names, the groups the plots draw: the viewer ingests every row it is
sent, whether a view shows it or not.
"""

from collections.abc import Callable, Sequence
import dataclasses
import functools
import logging
import math
import time
from typing import Any, TypeVar

from humanoid_mpc_msgs import arrows_pb2
from humanoid_mpc_msgs import fsm_state_pb2
from humanoid_mpc_msgs import line_strips_pb2
from humanoid_mpc_msgs import loop_timing_pb2
from humanoid_mpc_msgs import mpc_status_pb2
from humanoid_mpc_msgs import robot_model_instance_pb2
from humanoid_mpc_msgs import spheres_pb2
from humanoid_mpc_msgs import telemetry_series_pb2
from humanoid_mpc_msgs import visualization_scene_pb2
import numpy as np
import rerun as rr

from humanoid_mpc_ipc import topics
from humanoid_rerun_viewer import palette
from humanoid_rerun_viewer import robot_meshes
from humanoid_rerun_viewer import scalar_batcher
from humanoid_rerun_viewer import scene_contract
from humanoid_rerun_viewer import status_contract
from humanoid_rerun_viewer import telemetry_contract
from humanoid_rerun_viewer import urdf_model

MarkerKind = scene_contract.MarkerKind

_LOGGER = logging.getLogger("humanoid_rerun_viewer")

# A recurring problem is logged at most this often [s].
LOG_PERIOD_S = 5.0
# The smallest norm of a link's quaternion that is still a rotation (proto3 sends an unset one as zeros).
_MIN_QUATERNION_NORM = 1e-6
# A robot clock that goes back by more than this is a new run of the robot [s]. Smaller steps back happen within a run:
# the scene falls back from the robot/state sample's time to the older observation's when the samples stop (the
# visualization publisher's robotStateTimeout, 0.5 s), and an MpcStatus's observation time trails by a solve.
NEW_RUN_REWIND_S = 1.0
# For this long after a new run began [wall s], a message whose time is back near the previous run's is one of that
# run that was still on its way (the bus keeps the order of a topic, not across topics): it is dropped and counted.
PREVIOUS_RUN_WINDOW_S = 2.0
# How many messages of the clock topic in a row, each behind the robot's clock by more than NEW_RUN_REWIND_S, start a
# new run: one late message (a straggler) does not.
NEW_RUN_CONFIRMATIONS = 3
# The most messages that wait for the clock topic to confirm or refute a new run; one more refutes it.
MAX_HELD_MESSAGES = 64
# viz/telemetry is the clock topic while one of its messages arrived within this long [wall s]; viz/scene otherwise. A
# held message that waited this long is recorded into the current run (flush()).
CLOCK_TIMEOUT_S = 1.0
# The fraction of a telemetry interval (1 / telemetry_rate) a sample's time may fall short of the interval's start and
# still be its first: the robot samples on a period, and its times carry rounding errors.
_TELEMETRY_INTERVAL_TOLERANCE = 1e-6
# How many link names a warning lists at most.
_LISTED_LINKS = 8

# The radius and color of a marker at a path the contract does not list.
_UNLISTED_RADIUS = {
    MarkerKind.ARROWS: scene_contract.ARROW_RADIUS,
    MarkerKind.SPHERES: scene_contract.POINT_MARKER_RADIUS,
    MarkerKind.LINE_STRIPS: scene_contract.TRAJECTORY_RADIUS,
}
_UNLISTED_COLORS: tuple[palette.Rgba, ...] = (palette.with_alpha(palette.BLACK, 1.0),)


def new_recording(
    application_id: str, recording_id: str | None = None
) -> rr.RecordingStream:
    """A recording for the bridge: its own stream (not Rerun's global one), without Rerun's log_time timeline."""
    recording = rr.RecordingStream(application_id, recording_id=recording_id)
    recording.set_log_time_enabled(False)
    return recording


class MalformedMessageError(ValueError):
    """A message, or a part of one, that does not fit the contract; its text is the reason it is counted under."""


@dataclasses.dataclass
class BridgeStatistics:
    """What the bridge did, per topic."""

    # Messages each handler took, by topic.
    handled: dict[str, int] = dataclasses.field(default_factory=dict)
    # Messages or parts of them that did not fit the contract and were skipped, by (topic, reason).
    malformed: dict[tuple[str, str], int] = dataclasses.field(default_factory=dict)
    # Unexpected exceptions in a handler, by topic.
    handler_errors: dict[str, int] = dataclasses.field(default_factory=dict)
    # Link poses logged, and skipped because they had not changed.
    link_poses_logged: int = 0
    link_poses_unchanged: int = 0
    # Scalar groups of viz/telemetry buffered.
    telemetry_groups: int = 0
    # Scenes throttled due to max_scene_frequency; the last one of a stream is drawn by flush() once the interval passed.
    scenes_throttled: int = 0
    # New runs of the robot, each recorded as a recording of its own (new_run), and the previous run's messages that
    # arrived after a new run began and were dropped.
    new_runs: int = 0
    previous_run_dropped: int = 0
    # Messages behind the robot's clock by more than NEW_RUN_REWIND_S that the clock topic then showed to be late, not
    # of a new run, and that were recorded into the current run at their own times.
    late_messages: int = 0
    # viz/telemetry samples not recorded because an earlier one of their interval was (telemetry_rate).
    telemetry_decimated: int = 0

    def malformed_count(self, topic: str | None = None) -> int:
        """The malformed messages or parts counted under `topic`, or under every topic when it is None."""
        return sum(
            count
            for (malformed_topic, _), count in self.malformed.items()
            if topic is None or malformed_topic == topic
        )

    def handler_error_count(self) -> int:
        """The unexpected handler exceptions, over every topic."""
        return sum(self.handler_errors.values())


@dataclasses.dataclass(frozen=True)
class _HeldMessage:
    """A message behind the robot's clock that waits for the clock topic to confirm or refute a new run.

    Attributes:
        robot_time: the message's robot time [s].
        is_clock: whether it is of the clock topic.
        held_at: when it began to wait [wall s].
        handle: handles the message again, once it is known which run it belongs to.
    """

    robot_time: float
    is_clock: bool
    held_at: float
    handle: Callable[[], None]


@dataclasses.dataclass(frozen=True)
class _LoopTotals:
    """The counters of a LoopTiming, which are totals since the robot process started."""

    cycles: int
    overruns: int
    missed_periods: int
    telemetry_samples_dropped: int
    stale_policies_dropped: int

    @classmethod
    def of(cls, timing: loop_timing_pb2.LoopTiming) -> "_LoopTotals":
        return cls(
            cycles=timing.cycles,
            overruns=timing.overruns,
            missed_periods=timing.missed_periods,
            telemetry_samples_dropped=timing.telemetry_samples_dropped,
            stale_policies_dropped=timing.stale_policies_dropped,
        )

    def since(self, previous: "_LoopTotals | None") -> "_LoopTotals":
        """The counts since `previous`, or the totals themselves after a restarted robot process (or before any)."""
        values = dataclasses.astuple(self)
        if previous is None or any(
            now < before for now, before in zip(values, dataclasses.astuple(previous))
        ):
            return self
        return _LoopTotals(
            *(
                now - before
                for now, before in zip(values, dataclasses.astuple(previous))
            )
        )


class _RateLimitedLog:
    """Logs each kind of problem at most once per LOG_PERIOD_S."""

    def __init__(self, clock: Callable[[], float]) -> None:
        self._clock = clock
        self._last: dict[Any, float] = {}

    def warning(self, kind: Any, text: str, exc_info: bool = False) -> None:
        now = self._clock()
        if now - self._last.get(kind, -math.inf) >= LOG_PERIOD_S:
            self._last[kind] = now
            _LOGGER.warning("%s", text, exc_info=exc_info)


_Handler = TypeVar("_Handler", bound=Callable[..., None])


def _guarded(topic: str) -> Callable[[_Handler], _Handler]:
    """Counts the message under `topic`, and turns every exception of the handler into a count and a log line."""

    def decorate(handler: _Handler) -> _Handler:
        @functools.wraps(handler)
        def guarded(self: "RerunBridge", message: Any) -> None:
            statistics = self.statistics
            # pylint: disable-next=protected-access  # `self` is the RerunBridge whose own method this wraps.
            if not self._replaying:
                # A message that waited for the clock (_place_in_run) was counted when it came.
                statistics.handled[topic] = statistics.handled.get(topic, 0) + 1
            # pylint: disable-next=protected-access  # `self` is the RerunBridge whose own method this wraps.
            self._run_guarded(topic, functools.partial(handler, self), message)

        return guarded  # type: ignore[return-value]  # functools.wraps keeps the handler's signature, mypy sees it lost.

    return decorate


def _finite(value: float) -> bool:
    return math.isfinite(value)


def _listed(names: Sequence[str]) -> str:
    """The first _LISTED_LINKS of `names`, comma-separated, and how many more there are."""
    shown = ", ".join(names[:_LISTED_LINKS])
    more = len(names) - _LISTED_LINKS
    return f"{shown} and {more} more" if more > 0 else shown


def _vector3_array(vectors: Sequence[Any]) -> np.ndarray:
    """An (N, 3) array of humanoid_mpc_msgs.Vector3."""
    return np.array(
        [(vector.x, vector.y, vector.z) for vector in vectors], dtype=np.float64
    ).reshape(-1, 3)


def _colors_rgba8(colors: Sequence[Any]) -> np.ndarray:
    """An (N, 4) uint8 array of humanoid_mpc_msgs.Color (RGBA in [0, 1])."""
    rgba = np.array(
        [(color.r, color.g, color.b, color.a) for color in colors], dtype=np.float64
    ).reshape(-1, 4)
    return np.rint(np.clip(rgba, 0.0, 1.0) * 255.0).astype(np.uint8)


def _defaults_rgba8(defaults: Sequence[palette.Rgba], count: int) -> np.ndarray:
    """One default color for all elements, or the defaults cycled over `count` elements."""
    colors = [palette.to_rgba8(color) for color in defaults]
    if len(colors) == 1 or count == 0:
        return np.array(colors[:1], dtype=np.uint8).reshape(-1, 4)
    return np.array(
        [colors[index % len(colors)] for index in range(count)], dtype=np.uint8
    )


class RerunBridge:
    """Writes the bus's messages into `recording`, and each later run of the robot into a recording of `new_run`'s.

    Not thread-safe: one thread calls every method (bus_bridge's writer thread once the bus feeds it).

    Args:
        recording: where everything goes (new_recording()), attached to its sink.
        model: the robot's URDF geometry; None draws no robot (markers and plots only).
        instances: the robot instances whose meshes log_static() logs.
        clock: the wall clock [s since the epoch] of the wall_time timeline.
        max_pending_rows: per entity, before the scalar batcher sends without waiting for flush().
        max_scene_frequency: maximum scene messages to log per wall second (0.0 means unlimited).
        allowed_instances: if given, only these robot instance names are logged in viz/scene.
        new_run: makes the recording of a new run of the robot, attached to the same sink (and given the blueprint);
            None records every run into `recording`.
        telemetry_rate: the most viz/telemetry samples recorded per second of robot time [Hz]; 0.0 records every one.
        telemetry_selection: whether to record the telemetry group of a path (relative to telemetry/); None records
            every group.

    Attributes:
        model: The robot's URDF geometry; None draws no robot.
        recording: The recording of the current run.
        batcher: The status scalars' batcher of the current run.
        telemetry_batcher: The viz/telemetry frames' batcher of the current run.
    """

    def __init__(
        self,
        recording: rr.RecordingStream,
        model: urdf_model.RobotModel | None,
        instances: Sequence[
            scene_contract.RobotInstanceStyle
        ] = scene_contract.ROBOT_INSTANCES,
        clock: Callable[[], float] = time.time,
        max_pending_rows: int = scalar_batcher.DEFAULT_MAX_PENDING_ROWS,
        max_scene_frequency: float = 0.0,
        allowed_instances: Sequence[str] | None = None,
        new_run: Callable[[], rr.RecordingStream] | None = None,
        telemetry_rate: float = 0.0,
        telemetry_selection: Callable[[str], bool] | None = None,
    ) -> None:
        if not (max_scene_frequency >= 0.0 and math.isfinite(max_scene_frequency)):
            raise ValueError("max_scene_frequency must be non-negative and finite")
        if not (telemetry_rate >= 0.0 and math.isfinite(telemetry_rate)):
            raise ValueError(
                f"telemetry_rate must be non-negative and finite, got {telemetry_rate}"
            )
        self._telemetry_rate = telemetry_rate
        self._telemetry_selection = telemetry_selection
        # Per telemetry group path: whether telemetry_selection records it (asked once per path).
        self._telemetry_recorded: dict[str, bool] = {}
        # The interval (robot time * telemetry_rate, floored) of the last telemetry sample recorded; None: none yet.
        self._telemetry_interval: int | None = None
        self.recording = recording
        self._new_run = new_run
        self._max_pending_rows = max_pending_rows
        self.model = model
        self._meshes = robot_meshes.RobotMeshes(model) if model is not None else None
        if allowed_instances is not None:
            self._allowed_instances: frozenset[str] | None = frozenset(
                allowed_instances
            )
            self._instances = tuple(
                style for style in instances if style.name in self._allowed_instances
            )
        else:
            self._allowed_instances = None
            self._instances = tuple(instances)
        self._clock = clock
        self._log = _RateLimitedLog(time.monotonic)
        self._min_scene_interval = (
            1.0 / max_scene_frequency if max_scene_frequency > 0.0 else 0.0
        )
        self._last_scene_wall_time: float | None = None
        # The status scalars, each entity on its own times.
        self.batcher = scalar_batcher.ScalarBatcher(recording, max_pending_rows)
        # viz/telemetry: one frame per message, every group a row of it.
        self.telemetry_batcher = scalar_batcher.FrameBatcher(
            recording, max_pending_rows
        )
        self.statistics = BridgeStatistics()

        self._links_with_visuals = (
            model.links_with_visuals() if model is not None else frozenset()
        )
        self._urdf_links = frozenset(model.links) if model is not None else frozenset()
        # The robot instances whose link names were checked against the URDF's.
        self._link_names_checked: set[str] = set()
        self._logged_instances: set[str] = set()
        self._last_link_poses: dict[str, tuple[float, ...]] = {}
        self._last_scene_time = -math.inf
        # The latest scene the throttle held back, drawn by flush() once the interval passed; None: there is none.
        self._pending_scene: visualization_scene_pb2.VisualizationScene | None = None
        # The marker entities the last scene drew, with their kind.
        self._drawn_markers: dict[str, MarkerKind] = {}
        # Per telemetry entity: the series names its SeriesLines carry.
        self._series_names: dict[str, tuple[str, ...]] = {}
        self._status_series_logged: set[str] = set()
        self._latest_robot_time: float | None = None
        # The robot's clock: the latest time of the clock topic in the current run; None before its first message.
        self._run_time: float | None = None
        # Once a new run began: the previous run's latest robot time, and when the new run began [wall s].
        self._previous_run_time: float | None = None
        self._run_wall_start = -math.inf
        # When the last viz/telemetry message came [wall s]: it is the clock topic for CLOCK_TIMEOUT_S after.
        self._last_telemetry_wall_time = -math.inf
        # The messages behind the robot's clock that wait for the clock topic, in arrival order, and how many of them
        # are of the clock topic. True while they are handled again, which places them without waiting.
        self._held: list[_HeldMessage] = []
        self._held_clock_messages = 0
        self._replaying = False
        self._last_fsm_state: tuple[Any, ...] | None = None
        self._last_mpc_health: tuple[Any, ...] | None = None
        self._last_mpc_resets: tuple[int, int] | None = None
        # The previous LoopTiming's totals since the robot started (_LoopTotals); None before the first.
        self._last_loop_totals: _LoopTotals | None = None
        self._warned: set[Any] = set()
        self._last_report: dict[str, int] = {}

    # ------------------------------------------------------------------------------------------------------------------
    # Static data
    # ------------------------------------------------------------------------------------------------------------------

    def log_static(self) -> None:
        """Logs the world's axes (right-handed, z up) and the robot meshes of every instance of `instances`."""
        self.recording.log(
            scene_contract.WORLD_ROOT, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True
        )
        for style in self._instances:
            self._log_instance_meshes(style)

    def resend_static(self) -> None:
        """Forgets what was sent and logs the static data again, for a viewer that came back without any of it.

        The meshes, the series' names and the markers' clears are sent again as their messages come; the next scene
        logs every link pose.
        """
        self._forget_sent()
        self.log_static()

    def _forget_sent(self) -> None:
        """Forgets every entity logged so far, as for a recording that holds none of them."""
        self._logged_instances.clear()
        self._last_link_poses.clear()
        self._last_scene_time = -math.inf
        self._last_scene_wall_time = None
        self._pending_scene = None
        self._drawn_markers = {}
        self._series_names.clear()
        self._status_series_logged.clear()
        self._last_fsm_state = None
        self._last_mpc_health = None

    def _log_instance_meshes(self, style: scene_contract.RobotInstanceStyle) -> None:
        if self._meshes is None or style.name in self._logged_instances:
            return
        self._logged_instances.add(style.name)
        result = self._meshes.log_instance(self.recording, style)
        _LOGGER.info(
            "logged the robot model '%s' as %s: %d meshes, %d primitives",
            self._meshes.model.name,
            style.name,
            result.meshes,
            result.primitives,
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------------------------------------------------------

    def _count_malformed(self, topic: str, reason: str) -> None:
        key = (topic, reason)
        self.statistics.malformed[key] = self.statistics.malformed.get(key, 0) + 1
        self._log.warning(
            ("malformed", topic, reason),
            f"skipped part of a '{topic}' message: {reason}",
        )

    def _run_guarded(
        self, topic: str, work: Callable[[Any], None], message: Any
    ) -> None:
        """Runs `work` on `message`; a malformed part or an unexpected exception is counted under `topic` and logged."""
        try:
            work(message)
        except MalformedMessageError as error:
            self._count_malformed(topic, str(error))
        # pylint: disable-next=broad-exception-caught  # A handler error must not end the writer thread.
        except Exception:
            self.statistics.handler_errors[topic] = (
                self.statistics.handler_errors.get(topic, 0) + 1
            )
            self._log.warning(
                ("handler", topic),
                f"the handler of '{topic}' raised; the message was skipped",
                exc_info=True,
            )

    def _warn_once(self, key: Any, text: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            _LOGGER.warning("%s", text)

    def _is_clock_topic(self, topic: str) -> bool:
        """Whether `topic` is the robot's clock: viz/telemetry, or viz/scene while no telemetry arrives."""
        if topic == topics.VIZ_TELEMETRY:
            return True
        return (
            topic == topics.VIZ_SCENE
            and self._clock() - self._last_telemetry_wall_time >= CLOCK_TIMEOUT_S
        )

    def _place_in_run(
        self, topic: str, robot_time: float, handle: Callable[[], None]
    ) -> bool:
        """Follows the run of the robot that `robot_time` belongs to; the clock topic confirms a new one.

        A message behind the robot's clock by more than NEW_RUN_REWIND_S waits (the module's docstring): `handle` handles
        it again once the clock topic confirmed a new run or showed it late.

        Args:
            topic: the message's topic.
            robot_time: the message's robot time [s].
            handle: handles the message again; called later, after this returned False, for a message that waits.

        Returns:
            True when the caller records the message now; False for one that waits, or one of the previous run that
            arrived after the new run began, which is dropped and counted.
        """
        is_clock = self._is_clock_topic(topic)
        if topic == topics.VIZ_TELEMETRY:
            self._last_telemetry_wall_time = self._clock()
        latest = self._run_time
        if latest is None:
            if is_clock:
                self._run_time = robot_time
            return True
        if robot_time < latest - NEW_RUN_REWIND_S and not self._replaying:
            self._hold(robot_time, is_clock, handle)
            return False
        if self._is_of_previous_run(robot_time, latest):
            self.statistics.previous_run_dropped += 1
            self._log.warning(
                ("previous_run", topic),
                f"dropped a '{topic}' message of the previous run of the robot, which arrived after the new run began",
            )
            return False
        if is_clock and not self._replaying:
            # The clock is where it was: whatever waited was late, not of a new run.
            self._release_held()
        if is_clock:
            clock = self._run_time
            self._run_time = robot_time if clock is None else max(clock, robot_time)
        return True

    def _is_of_previous_run(self, robot_time: float, latest: float) -> bool:
        """Whether `robot_time` is back near the previous run's clock, shortly after a new run began."""
        return (
            self._previous_run_time is not None
            and latest + NEW_RUN_REWIND_S
            < robot_time
            <= self._previous_run_time + NEW_RUN_REWIND_S
            and self._clock() - self._run_wall_start < PREVIOUS_RUN_WINDOW_S
        )

    def _hold(
        self, robot_time: float, is_clock: bool, handle: Callable[[], None]
    ) -> None:
        """Makes a message behind the robot's clock wait; enough of the clock topic in a row start the new run."""
        self._held.append(
            _HeldMessage(
                robot_time=robot_time,
                is_clock=is_clock,
                held_at=self._clock(),
                handle=handle,
            )
        )
        if is_clock:
            self._held_clock_messages += 1
        if self._held_clock_messages >= NEW_RUN_CONFIRMATIONS:
            self._confirm_new_run()
        elif len(self._held) > MAX_HELD_MESSAGES:
            self._release_held()

    def _take_held(self) -> list[_HeldMessage]:
        held = self._held
        self._held = []
        self._held_clock_messages = 0
        return held

    def _handle_again(self, held: Sequence[_HeldMessage]) -> None:
        """Handles the messages that waited, in the order they came, without making them wait again."""
        self._replaying = True
        try:
            for message in held:
                message.handle()
        finally:
            self._replaying = False

    def _release_held(self) -> None:
        """Records the messages that waited into the current run, at their own times: they were late."""
        held = self._take_held()
        if held:
            self.statistics.late_messages += len(held)
            self._log.warning(
                "late",
                f"{len(held)} message(s) came more than {NEW_RUN_REWIND_S} s behind the robot's clock, which did not go "
                "back: recorded at their own times in the current run",
            )
            self._handle_again(held)

    def _release_stale_held(self) -> None:
        """Releases the messages that waited CLOCK_TIMEOUT_S or more: the clock topic went quiet."""
        if self._held and self._clock() - self._held[0].held_at >= CLOCK_TIMEOUT_S:
            self._release_held()

    def _confirm_new_run(self) -> None:
        """The clock topic went back: starts the new run at its first message that waited, and records them into it."""
        held = self._take_held()
        previous_time = self._run_time
        robot_time = next(message.robot_time for message in held if message.is_clock)
        if previous_time is not None:
            self._start_new_run(previous_time, robot_time)
        self._run_time = robot_time
        self._handle_again(held)

    def _start_new_run(self, previous_time: float, robot_time: float) -> None:
        """Sends what the previous run buffered, then records from a recording of new_run's with the static data again."""
        new_run = self._new_run
        if new_run is None:
            return
        self._flush_batchers()
        try:
            recording = new_run()
        # pylint: disable-next=broad-exception-caught  # A recording that cannot be made keeps the previous one.
        except Exception:
            _LOGGER.warning(
                "the robot's clock went back, but a recording of the new run could not be made; "
                "recording every run into one from now on",
                exc_info=True,
            )
            self._new_run = None
            return
        self.recording = recording
        self.batcher = scalar_batcher.ScalarBatcher(recording, self._max_pending_rows)
        self.telemetry_batcher = scalar_batcher.FrameBatcher(
            recording, self._max_pending_rows
        )
        self._forget_sent()
        self._telemetry_interval = None
        self._previous_run_time = previous_time
        self._run_wall_start = self._clock()
        self._latest_robot_time = robot_time
        self.statistics.new_runs += 1
        self.log_static()
        text = (
            f"the robot's clock went back from {previous_time:.3f} s to {robot_time:.3f} s: a new run of the robot, "
            "recorded as a recording of its own"
        )
        _LOGGER.warning("%s", text)
        self._text(status_contract.BRIDGE_LOG, robot_time, text, rr.TextLogLevel.WARN)

    def _set_time(self, robot_time: float | None, wall_time: float) -> None:
        """Sets the timelines of the following log() calls of this thread."""
        self.recording.reset_time()
        if robot_time is not None:
            self.recording.set_time(scene_contract.ROBOT_TIMELINE, duration=robot_time)
        self.recording.set_time(scene_contract.WALL_TIMELINE, timestamp=wall_time)

    # ------------------------------------------------------------------------------------------------------------------
    # viz/scene
    # ------------------------------------------------------------------------------------------------------------------

    @_guarded(topics.VIZ_SCENE)
    def handle_scene(self, scene: visualization_scene_pb2.VisualizationScene) -> None:
        """Moves the robots to the scene's link poses, draws its markers and clears the markers it no longer names.

        A scene the throttle (max_scene_frequency) holds back is kept, and flush() draws the latest such once the
        interval passed, so that the last scene of a stream is drawn too.

        Args:
            scene: The visualization scene containing robot poses and markers.
        """
        if not _finite(scene.time):
            raise MalformedMessageError("the scene's time is not finite")
        if not self._place_in_run(
            topics.VIZ_SCENE, scene.time, functools.partial(self.handle_scene, scene)
        ):
            return
        if scene.time < self._last_scene_time:
            # The robot's clock went back (a restarted simulation): poses at the new times must all be logged.
            self._last_link_poses.clear()
            self._last_scene_wall_time = None
            self._pending_scene = None
        self._last_scene_time = scene.time
        self._latest_robot_time = scene.time

        wall_now = self._clock()
        if (
            self._min_scene_interval > 0.0
            and self._last_scene_wall_time is not None
            and (wall_now - self._last_scene_wall_time) < self._min_scene_interval
        ):
            self.statistics.scenes_throttled += 1
            self._pending_scene = scene
            return
        self._draw_scene(scene, wall_now)

    def _draw_scene(
        self, scene: visualization_scene_pb2.VisualizationScene, wall_now: float
    ) -> None:
        """Logs one scene at its robot time and `wall_now`."""
        self._last_scene_wall_time = wall_now
        self._pending_scene = None
        self._set_time(scene.time, wall_now)

        for robot in scene.robots:
            self._guarded_part(topics.VIZ_SCENE, self._log_robot, robot)

        drawn: dict[str, MarkerKind] = {}
        for kind, markers, log in (
            (MarkerKind.ARROWS, scene.arrows, self._log_arrows),
            (MarkerKind.SPHERES, scene.spheres, self._log_spheres),
            (MarkerKind.LINE_STRIPS, scene.line_strips, self._log_line_strips),
        ):
            for marker in markers:
                if not scene_contract.is_valid_relative_path(marker.path):
                    self._count_malformed(
                        topics.VIZ_SCENE, f"invalid {kind.value} path '{marker.path}'"
                    )
                    continue
                if marker.path in drawn:
                    self._count_malformed(
                        topics.VIZ_SCENE,
                        f"'{marker.path}' appears twice in one scene",
                    )
                    continue
                drawn[marker.path] = kind
                if not self._guarded_part(topics.VIZ_SCENE, log, marker):
                    self._clear_marker(marker.path, kind)
        for path, kind in self._drawn_markers.items():
            if drawn.get(path) != kind:
                self._clear_marker(path, kind)
        self._drawn_markers = drawn

    def _guarded_part(self, topic: str, log: Callable[[Any], None], part: Any) -> bool:
        """Logs one part of a message; a malformed part is counted and skipped, the rest of the message is drawn."""
        try:
            log(part)
            return True
        except MalformedMessageError as error:
            self._count_malformed(topic, str(error))
            return False

    def _log_robot(self, robot: robot_model_instance_pb2.RobotModelInstance) -> None:
        """Logs the changed link poses of one robot instance, first logging its meshes if it is new."""
        name = robot.name
        if not scene_contract.is_valid_relative_path(name) or "/" in name:
            raise MalformedMessageError(f"invalid robot instance name '{name}'")
        if self._allowed_instances is not None and name not in self._allowed_instances:
            return
        if len(robot.link_names) != len(robot.link_poses):
            raise MalformedMessageError(
                f"robot '{name}' has {len(robot.link_names)} link names but {len(robot.link_poses)} poses"
            )
        if self._meshes is None:
            self._warn_once(
                "no_model",
                "no URDF was given (--urdf), so the robot instances of viz/scene are not drawn",
            )
            return
        if name not in self._logged_instances:
            style = scene_contract.instance_style(name)
            if style not in scene_contract.ROBOT_INSTANCES:
                self._warn_once(
                    ("instance", name),
                    f"robot instance '{name}' is not one of the contract's; drawing it with the default style",
                )
            self._log_instance_meshes(style)
        self._check_link_names(robot)

        for link, pose in zip(robot.link_names, robot.link_poses):
            if link not in self._links_with_visuals:
                continue
            position = pose.position
            orientation = pose.orientation
            values = (
                position.x,
                position.y,
                position.z,
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w,
            )
            if not all(math.isfinite(value) for value in values):
                raise MalformedMessageError(
                    f"robot '{name}' link '{link}' has a pose that is not finite"
                )
            norm = math.sqrt(sum(value * value for value in values[3:]))
            if norm < _MIN_QUATERNION_NORM:
                raise MalformedMessageError(
                    f"robot '{name}' link '{link}' has a zero quaternion"
                )
            path = scene_contract.link_path(name, link)
            if self._last_link_poses.get(path) == values:
                self.statistics.link_poses_unchanged += 1
                continue
            self._last_link_poses[path] = values
            self.recording.log(
                path,
                rr.Transform3D(
                    translation=[values[0], values[1], values[2]],
                    quaternion=rr.Quaternion(
                        xyzw=[
                            values[3] / norm,
                            values[4] / norm,
                            values[5] / norm,
                            values[6] / norm,
                        ]
                    ),
                ),
            )
            self.statistics.link_poses_logged += 1

    def _check_link_names(
        self, robot: robot_model_instance_pb2.RobotModelInstance
    ) -> None:
        """Counts a robot instance that names links the URDF lacks, and warns once of the visual links it never poses.

        Both mean a URDF that is not the robot's (--urdf): its links the scene does not pose stay at the world origin.
        The links the URDF has are drawn either way.

        Args:
            robot: The robot model instance to inspect for link names against the URDF.
        """
        model = self.model
        if model is None:
            return
        name = robot.name
        named = set(robot.link_names)
        unknown = sorted(named - self._urdf_links)
        if unknown:
            self._warn_once(
                ("unknown_links", name),
                f"robot '{name}' names links the URDF '{model.name}' does not have "
                f"({_listed(unknown)}); is --urdf the robot's?",
            )
            self._count_malformed(
                topics.VIZ_SCENE,
                f"robot '{name}' names links the URDF '{model.name}' does not have; is --urdf the robot's?",
            )
        if name in self._link_names_checked:
            return
        self._link_names_checked.add(name)
        unposed = sorted(self._links_with_visuals - named)
        if unposed:
            self._warn_once(
                ("unposed_links", name),
                f"robot '{name}' gives no pose for the URDF links {_listed(unposed)}, which are drawn at the world "
                "origin; is --urdf the robot's?",
            )

    def _marker_style(
        self, path: str, kind: MarkerKind
    ) -> tuple[float, tuple[palette.Rgba, ...]]:
        spec = scene_contract.MARKERS_BY_PATH.get(path)
        if spec is not None and spec.kind == kind:
            return spec.default_radius, spec.default_colors
        self._warn_once(
            ("unlisted_marker", path, kind),
            f"'{path}' ({kind.value}) is not a marker of the scene contract; drawing it with default style",
        )
        return _UNLISTED_RADIUS[kind], _UNLISTED_COLORS

    def _element_colors(
        self,
        path: str,
        colors: Sequence[Any],
        count: int,
        defaults: tuple[palette.Rgba, ...],
    ) -> np.ndarray:
        """The colors of `count` elements: the message's (one for all, or one each), else the marker's defaults."""
        if not colors:
            return _defaults_rgba8(defaults, count)
        if len(colors) not in (1, count):
            raise MalformedMessageError(
                f"'{path}' has {len(colors)} colors for {count} elements (give one, or one each)"
            )
        rgba = _colors_rgba8(colors)
        if not rgba[:, 3].any():
            self._warn_once(
                ("transparent", path),
                f"every color of '{path}' has alpha 0, so it is invisible; set Color.a",
            )
        return rgba

    def _log_arrows(self, arrows: arrows_pb2.Arrows) -> None:
        """Draws one Arrows marker, with the contract's radius and colors where the message gives none."""
        radius_default, color_defaults = self._marker_style(
            arrows.path, MarkerKind.ARROWS
        )
        origins = _vector3_array(arrows.origins)
        vectors = _vector3_array(arrows.vectors)
        if len(origins) != len(vectors):
            raise MalformedMessageError(
                f"'{arrows.path}' has {len(origins)} origins but {len(vectors)} vectors"
            )
        if not (np.isfinite(origins).all() and np.isfinite(vectors).all()):
            raise MalformedMessageError(
                f"'{arrows.path}' has coordinates that are not finite"
            )
        radius = arrows.radius if arrows.radius > 0.0 else radius_default
        self.recording.log(
            scene_contract.entity_path(scene_contract.WORLD_ROOT, arrows.path),
            rr.Arrows3D(
                origins=origins,
                vectors=vectors,
                radii=[radius],
                colors=self._element_colors(
                    arrows.path, arrows.colors, len(vectors), color_defaults
                ),
            ),
        )

    def _log_spheres(self, spheres: spheres_pb2.Spheres) -> None:
        """Draws one Spheres marker, with the contract's radius and colors where the message gives none."""
        radius_default, color_defaults = self._marker_style(
            spheres.path, MarkerKind.SPHERES
        )
        centers = _vector3_array(spheres.centers)
        count = len(centers)
        if not np.isfinite(centers).all():
            raise MalformedMessageError(
                f"'{spheres.path}' has centers that are not finite"
            )
        if not spheres.radii:
            radii = np.array([radius_default], dtype=np.float32)
        elif len(spheres.radii) in (1, count):
            radii = np.asarray(spheres.radii, dtype=np.float32)
        else:
            raise MalformedMessageError(
                f"'{spheres.path}' has {len(spheres.radii)} radii for {count} spheres (give one, or one each)"
            )
        self.recording.log(
            scene_contract.entity_path(scene_contract.WORLD_ROOT, spheres.path),
            rr.Points3D(
                positions=centers,
                radii=radii,
                colors=self._element_colors(
                    spheres.path, spheres.colors, count, color_defaults
                ),
            ),
        )

    def _log_line_strips(self, line_strips: line_strips_pb2.LineStrips) -> None:
        """Draws one LineStrips marker, with the contract's radius and colors where the message gives none."""
        radius_default, color_defaults = self._marker_style(
            line_strips.path, MarkerKind.LINE_STRIPS
        )
        strips: list[np.ndarray] = []
        defaults = _defaults_rgba8(color_defaults, len(line_strips.strips))
        colors = np.empty((len(line_strips.strips), 4), dtype=np.uint8)
        for index, strip in enumerate(line_strips.strips):
            points = _vector3_array(strip.points)
            if not np.isfinite(points).all():
                raise MalformedMessageError(
                    f"'{line_strips.path}' strip {index} has points that are not finite"
                )
            strips.append(points)
            if strip.HasField("color"):
                colors[index] = _colors_rgba8([strip.color])[0]
            else:
                colors[index] = defaults[index % len(defaults)]
        radius = line_strips.radius if line_strips.radius > 0.0 else radius_default
        self.recording.log(
            scene_contract.entity_path(scene_contract.WORLD_ROOT, line_strips.path),
            rr.LineStrips3D(strips, radii=[radius], colors=colors),
        )

    def _clear_marker(self, path: str, kind: MarkerKind) -> None:
        """Draws nothing at the marker `path` from now on (an empty archetype of its kind)."""
        entity = scene_contract.entity_path(scene_contract.WORLD_ROOT, path)
        empty = np.zeros((0, 3), dtype=np.float64)
        if kind is MarkerKind.ARROWS:
            self.recording.log(entity, rr.Arrows3D(origins=empty, vectors=empty))
        elif kind is MarkerKind.SPHERES:
            self.recording.log(entity, rr.Points3D(positions=empty))
        else:
            self.recording.log(entity, rr.LineStrips3D([]))

    # ------------------------------------------------------------------------------------------------------------------
    # viz/telemetry
    # ------------------------------------------------------------------------------------------------------------------

    @_guarded(topics.VIZ_TELEMETRY)
    def handle_telemetry(self, series: telemetry_series_pb2.TelemetrySeries) -> None:
        """Buffers the selected ScalarGroups at telemetry/<path>, at most at telemetry_rate; flush() sends them."""
        if not _finite(series.time):
            raise MalformedMessageError("the telemetry's time is not finite")
        robot_time = series.time
        if not self._place_in_run(
            topics.VIZ_TELEMETRY,
            robot_time,
            functools.partial(self.handle_telemetry, series),
        ):
            return
        self._latest_robot_time = robot_time
        if not self._takes_telemetry_sample(robot_time):
            self.statistics.telemetry_decimated += 1
            return
        wall_time = self._clock()
        rows: list[tuple[str, Sequence[float]]] = []
        paths: set[str] = set()
        for group in series.groups:
            path = group.path
            if not self._records_group(path):
                continue
            if path in paths:
                self._count_malformed(
                    topics.VIZ_TELEMETRY, f"group '{path}' appears twice in one message"
                )
                continue
            paths.add(path)
            count = len(group.values)
            if len(group.names) != count:
                # Checked on every message, not only when the group is declared: a group whose names stop matching
                # its values would be plotted under the names it was declared with.
                self._count_malformed(
                    topics.VIZ_TELEMETRY,
                    f"group '{path}' has {len(group.names)} names but {count} values",
                )
                continue
            names = self._series_names.get(path)
            if names is None or len(names) != count:
                if not self._declare_series(group):
                    continue
            rows.append((f"{scene_contract.TELEMETRY_ROOT}/{path}", group.values))
        self.telemetry_batcher.append(robot_time, wall_time, rows)
        self.statistics.telemetry_groups += len(rows)

    def _takes_telemetry_sample(self, robot_time: float) -> bool:
        """Whether the sample at `robot_time` is the first of its 1 / telemetry_rate interval (always at rate 0)."""
        if self._telemetry_rate <= 0.0:
            return True
        interval = math.floor(
            robot_time * self._telemetry_rate + _TELEMETRY_INTERVAL_TOLERANCE
        )
        if interval == self._telemetry_interval:
            return False
        # A later interval, or an earlier one after the clock went back within the run (a sample that was late).
        self._telemetry_interval = interval
        return True

    def _records_group(self, path: str) -> bool:
        """Whether telemetry_selection records the group at `path`."""
        selection = self._telemetry_selection
        if selection is None:
            return True
        recorded = self._telemetry_recorded.get(path)
        if recorded is None:
            recorded = bool(selection(path))
            self._telemetry_recorded[path] = recorded
        return recorded

    def _declare_series(self, group: Any) -> bool:
        """Logs the static SeriesLines of a group seen for the first time, or whose size changed.

        The caller checked that the group has as many names as values.

        Args:
            group: The telemetry group whose series lines are being declared.

        Returns:
            True if the series lines were successfully declared, False otherwise.
        """
        path = group.path
        if not scene_contract.is_valid_relative_path(path):
            self._count_malformed(topics.VIZ_TELEMETRY, f"invalid group path '{path}'")
            return False
        names = tuple(group.names)
        if not names:
            return False
        colors = None
        contract_group = telemetry_contract.find_group(path)
        if contract_group is None:
            self._warn_once(
                ("unlisted_group", path),
                f"telemetry group '{path}' is not in the telemetry contract; plotting it anyway",
            )
        elif not contract_group.robot_dependent and contract_group.names != names:
            self._warn_once(
                ("group_names", path),
                f"telemetry group '{path}' names {list(names)}, the contract {list(contract_group.names)}",
            )
        elif contract_group.colors:
            colors = [palette.hex_to_rgba8(color) for color in contract_group.colors]
        self.recording.log(
            f"{scene_contract.TELEMETRY_ROOT}/{path}",
            rr.SeriesLines(names=list(names), colors=colors),
            static=True,
        )
        self._series_names[path] = names
        return True

    def flush(self) -> int:
        """Draws the scene the throttle held back once its interval passed, and sends the buffered scalars.

        Messages that waited CLOCK_TIMEOUT_S for the clock topic, which went quiet, are recorded into the current run.

        Returns:
            How many scalar rows were sent.
        """
        self._release_stale_held()
        if self._pending_scene is not None and self._last_scene_wall_time is not None:
            wall_now = self._clock()
            if wall_now - self._last_scene_wall_time >= self._min_scene_interval:
                self._run_guarded(
                    topics.VIZ_SCENE,
                    functools.partial(self._draw_scene, wall_now=wall_now),
                    self._pending_scene,
                )
        return self._flush_batchers()

    def draw_pending_scene(self) -> None:
        """Draws the scene the throttle held back now, whatever the interval: the last scene before a shutdown.

        Messages that still wait for the clock topic are recorded into the current run first.
        """
        self._release_held()
        if self._pending_scene is not None:
            self._run_guarded(
                topics.VIZ_SCENE,
                functools.partial(self._draw_scene, wall_now=self._clock()),
                self._pending_scene,
            )

    def _flush_batchers(self) -> int:
        """Sends the buffered scalars; returns how many rows."""
        try:
            return self.telemetry_batcher.flush() + self.batcher.flush()
        # pylint: disable-next=broad-exception-caught  # A failed send is counted; it must not end the flush timer.
        except Exception:
            self.statistics.handler_errors["flush"] = (
                self.statistics.handler_errors.get("flush", 0) + 1
            )
            self._log.warning("flush", "sending the plots failed", exc_info=True)
            return 0

    def flush_recording(self, timeout: float) -> bool:
        """Waits up to `timeout` seconds for the recording to reach its sink; False when it did not."""
        try:
            self.recording.flush(timeout_sec=timeout)
            return True
        # pylint: disable-next=broad-exception-caught  # Rerun raises no documented type; shutdown goes on without it.
        except Exception:
            _LOGGER.warning(
                "the recording did not reach its sink within %.1f s", timeout
            )
            return False

    # ------------------------------------------------------------------------------------------------------------------
    # Status: robot/fsm_state, mpc/status, robot/loop_timing
    # ------------------------------------------------------------------------------------------------------------------

    def _status_scalars(
        self,
        series: status_contract.StatusSeries,
        robot_time: float | None,
        wall_time: float,
        values: Sequence[float],
    ) -> None:
        """Buffers one row of a status series, first logging its static SeriesLines."""
        if series.path not in self._status_series_logged:
            self._status_series_logged.add(series.path)
            self.recording.log(
                series.path, rr.SeriesLines(names=list(series.names)), static=True
            )
        self.batcher.append(series.path, robot_time, wall_time, values)

    def _text(self, path: str, robot_time: float | None, text: str, level: str) -> None:
        self._set_time(robot_time, self._clock())
        self.recording.log(path, rr.TextLog(text, level=level))

    @_guarded(topics.ROBOT_FSM_STATE)
    def handle_fsm_state(self, state: fsm_state_pb2.FsmState) -> None:
        """Logs a line when the FSM state changes (it is also re-sent at 2 Hz unchanged)."""
        key = (
            state.mode,
            state.gantry_locked,
            state.controller_resets,
            state.mpc_healthy,
        )
        if key == self._last_fsm_state:
            return
        self._last_fsm_state = key
        text = (
            f"FSM {state.mode or '(none)'}; gantry {'locked' if state.gantry_locked else 'free'}; "
            f"MPC {'healthy' if state.mpc_healthy else 'UNHEALTHY'}; controller resets {state.controller_resets}"
        )
        level = rr.TextLogLevel.INFO if state.mpc_healthy else rr.TextLogLevel.WARN
        self._text(status_contract.FSM_STATE_LOG, self._latest_robot_time, text, level)

    @_guarded(topics.MPC_STATUS)
    def handle_mpc_status(self, status: mpc_status_pb2.MpcStatus) -> None:
        """Plots the solve time, failures and observations; logs a line when the solver's health, error or resets change.

        The observation time is a robot time the bridge has seen, which FsmState and LoopTiming then take too. It trails
        the scenes' and the telemetry's by a solve, so it only moves the latest robot time forward, or back to a new run.

        Args:
            status: The MPC status message containing solve statistics and health flags.
        """
        robot_time = (
            status.observation_time if _finite(status.observation_time) else None
        )
        if robot_time is not None:
            if not self._place_in_run(
                topics.MPC_STATUS,
                robot_time,
                functools.partial(self.handle_mpc_status, status),
            ):
                return
            latest = self._latest_robot_time
            if (
                latest is None
                or robot_time > latest
                or robot_time < latest - NEW_RUN_REWIND_S
            ):
                self._latest_robot_time = robot_time
        wall_time = self._clock()
        solver = status.solver_status
        self._status_scalars(
            status_contract.MPC_SOLVE_TIME,
            robot_time,
            wall_time,
            (solver.solve_time_ms,),
        )
        self._status_scalars(
            status_contract.MPC_FAILURES,
            robot_time,
            wall_time,
            (float(solver.consecutive_failures),),
        )
        self._status_scalars(
            status_contract.MPC_OBSERVATIONS,
            robot_time,
            wall_time,
            (float(status.observations_received), float(status.observations_skipped)),
        )
        health = (solver.healthy, solver.last_error)
        if health != self._last_mpc_health:
            self._last_mpc_health = health
            if not solver.healthy:
                text = (
                    f"MPC solver UNHEALTHY after {solver.consecutive_failures} consecutive failures: "
                    f"{solver.last_error or '(no error text)'}"
                )
                level = rr.TextLogLevel.ERROR
            elif solver.last_error:
                # A solve failed, below the supervisor's limit of consecutive failures: still healthy.
                text = (
                    f"MPC solve failed ({solver.consecutive_failures} in a row; the solver is still healthy): "
                    f"{solver.last_error}"
                )
                level = rr.TextLogLevel.WARN
            else:
                text, level = "MPC solver healthy", rr.TextLogLevel.INFO
            self._text(status_contract.MPC_STATUS_LOG, robot_time, text, level)
        resets = (status.resets_served, status.full_resets_served)
        if resets != self._last_mpc_resets:
            if self._last_mpc_resets is not None:
                self._text(
                    status_contract.MPC_STATUS_LOG,
                    robot_time,
                    f"MPC served reset {status.resets_served} (full resets {status.full_resets_served})",
                    rr.TextLogLevel.INFO,
                )
            self._last_mpc_resets = resets

    @_guarded(topics.ROBOT_LOOP_TIMING)
    def handle_loop_timing(self, timing: loop_timing_pb2.LoopTiming) -> None:
        """Plots the loop's period, compute time, lateness, overruns, missed periods, drops and policy age; logs a line.

        The counters are totals since the robot process started, which the plot shows; the line warns of what they
        added since the previous report (all of them after a restarted robot process, whose totals start over).

        Args:
            timing: The loop timing message containing cycle duration and jitter statistics.
        """
        robot_time = self._latest_robot_time
        wall_time = self._clock()
        milliseconds = 1000.0
        self._status_scalars(
            status_contract.LOOP_PERIOD,
            robot_time,
            wall_time,
            (
                timing.target_period_s * milliseconds,
                timing.mean_period_s * milliseconds,
                timing.max_period_s * milliseconds,
            ),
        )
        self._status_scalars(
            status_contract.LOOP_COMPUTE_TIME,
            robot_time,
            wall_time,
            (timing.max_compute_time_s * milliseconds,),
        )
        self._status_scalars(
            status_contract.LOOP_LATENESS,
            robot_time,
            wall_time,
            (timing.max_lateness_s * milliseconds,),
        )
        self._status_scalars(
            status_contract.LOOP_EVENTS,
            robot_time,
            wall_time,
            (
                float(timing.overruns),
                float(timing.missed_periods),
                float(timing.telemetry_samples_dropped),
                float(timing.stale_policies_dropped),
            ),
        )
        # A negative age means no policy is in use; a gap in the plot says that better than a negative number.
        policy_age = timing.policy_age_s if timing.policy_age_s >= 0.0 else math.nan
        self._status_scalars(
            status_contract.POLICY_AGE, robot_time, wall_time, (policy_age,)
        )
        totals = _LoopTotals.of(timing)
        new = totals.since(self._last_loop_totals)
        self._last_loop_totals = totals
        trouble = (
            new.overruns
            or new.missed_periods
            or new.telemetry_samples_dropped
            or new.stale_policies_dropped
        )
        policy = (
            f"policy age {timing.policy_age_s:.3f} s"
            if timing.policy_age_s >= 0.0
            else "no policy"
        )
        text = (
            f"loop {timing.cycles} cycles: period target {timing.target_period_s * milliseconds:.3f} ms, "
            f"mean {timing.mean_period_s * milliseconds:.3f} ms, max {timing.max_period_s * milliseconds:.3f} ms; "
            f"compute max {timing.max_compute_time_s * milliseconds:.3f} ms; "
            f"wake-up up to {timing.max_lateness_s * milliseconds:.3f} ms late; "
            f"+{new.overruns} overruns, +{new.missed_periods} missed periods, "
            f"+{new.telemetry_samples_dropped} telemetry samples dropped, "
            f"+{new.stale_policies_dropped} stale policies dropped since the last report "
            f"({timing.overruns}, {timing.missed_periods}, {timing.telemetry_samples_dropped} and "
            f"{timing.stale_policies_dropped} since the robot started); {policy}"
        )
        level = rr.TextLogLevel.WARN if trouble else rr.TextLogLevel.DEBUG
        self._text(status_contract.LOOP_TIMING_LOG, robot_time, text, level)

    # ------------------------------------------------------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------------------------------------------------------

    def report(self, bus_rejected: int = 0, sink_dropped: int = 0) -> str | None:
        """Describes the problems since the last report in one line, which it also writes to the bridge's text log.

        Args:
            bus_rejected: how many messages the bus has rejected so far, in total; the line counts the new ones.
            sink_dropped: how many messages were dropped so far because the sink took no data, in total.

        Returns:
            The new malformed parts, handler errors, rejected and dropped messages, or None when there were none.
        """
        totals = {
            "malformed": self.statistics.malformed_count(),
            "handler errors": self.statistics.handler_error_count(),
            "rejected by the bus": bus_rejected,
            "dropped while the Rerun sink took no data": sink_dropped,
            "messages of a previous run dropped": self.statistics.previous_run_dropped,
            f"messages more than {NEW_RUN_REWIND_S:g} s behind the robot's clock": (
                self.statistics.late_messages
            ),
        }
        new = {
            name: total - self._last_report.get(name, 0)
            for name, total in totals.items()
        }
        self._last_report = totals
        problems = [f"{count} {name}" for name, count in new.items() if count > 0]
        if not problems:
            return None
        text = f'since the last report: {", ".join(problems)}'
        try:
            self._text(
                status_contract.BRIDGE_LOG,
                self._latest_robot_time,
                text,
                rr.TextLogLevel.WARN,
            )
        # pylint: disable-next=broad-exception-caught  # The report is returned even when Rerun cannot log it.
        except Exception:
            self._log.warning(
                "report", "logging the bridge's report failed", exc_info=True
            )
        return text
