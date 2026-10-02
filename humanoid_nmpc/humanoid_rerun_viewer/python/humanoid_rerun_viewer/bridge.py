"""Maps the bus's visualization and status messages onto a Rerun recording.

    recording = new_recording("humanoid_nmpc")
    bridge = RerunBridge(recording, robot_model)
    bridge.log_static()                    # the robot meshes of every instance, the world's axes
    bridge.handle_scene(scene)             # viz/scene: link poses and markers
    bridge.handle_telemetry(series)        # viz/telemetry: buffered, sent by flush()
    bridge.flush()                         # every ~50 ms
    bridge.handle_fsm_state(state)         # robot/fsm_state, mpc/status, robot/loop_timing: text logs and scalars

The handlers run on one thread (the bus's receive thread). None of them raises: a message that does not fit the
contract (scene_contract, telemetry_contract) is counted under its topic and a reason in `statistics.malformed`, and
logged at most once per LOG_PERIOD_S for each topic and reason; an unexpected error is counted in
`statistics.handler_errors` the same way.
"""

import dataclasses
import functools
import logging
import math
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, TypeVar

import numpy as np
import rerun as rr

from humanoid_mpc_msgs import arrows_pb2
from humanoid_mpc_msgs import fsm_state_pb2
from humanoid_mpc_msgs import line_strips_pb2
from humanoid_mpc_msgs import loop_timing_pb2
from humanoid_mpc_msgs import mpc_status_pb2
from humanoid_mpc_msgs import robot_model_instance_pb2
from humanoid_mpc_msgs import spheres_pb2
from humanoid_mpc_msgs import telemetry_series_pb2
from humanoid_mpc_msgs import visualization_scene_pb2

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

# The radius and color of a marker at a path the contract does not list.
_UNLISTED_RADIUS = {
    MarkerKind.ARROWS: scene_contract.ARROW_RADIUS,
    MarkerKind.SPHERES: scene_contract.POINT_MARKER_RADIUS,
    MarkerKind.LINE_STRIPS: scene_contract.TRAJECTORY_RADIUS,
}
_UNLISTED_COLORS: Tuple[palette.Rgba, ...] = (palette.with_alpha(palette.BLACK, 1.0),)


def new_recording(
    application_id: str, recording_id: Optional[str] = None
) -> rr.RecordingStream:
    """A recording for the bridge: its own stream (not Rerun's global one), without Rerun's log_time timeline."""
    recording = rr.RecordingStream(application_id, recording_id=recording_id)
    recording.set_log_time_enabled(False)
    return recording


class MalformedMessage(ValueError):
    """A message, or a part of one, that does not fit the contract; its text is the reason it is counted under."""


@dataclasses.dataclass
class BridgeStatistics:
    """What the bridge did, per topic."""

    # Messages each handler took, by topic.
    handled: Dict[str, int] = dataclasses.field(default_factory=dict)
    # Messages or parts of them that did not fit the contract and were skipped, by (topic, reason).
    malformed: Dict[Tuple[str, str], int] = dataclasses.field(default_factory=dict)
    # Unexpected exceptions in a handler, by topic.
    handler_errors: Dict[str, int] = dataclasses.field(default_factory=dict)
    # Link poses logged, and skipped because they had not changed.
    link_poses_logged: int = 0
    link_poses_unchanged: int = 0
    # Scalar groups of viz/telemetry buffered.
    telemetry_groups: int = 0

    def malformed_count(self, topic: Optional[str] = None) -> int:
        return sum(
            count
            for (malformed_topic, _), count in self.malformed.items()
            if topic is None or malformed_topic == topic
        )

    def handler_error_count(self) -> int:
        return sum(self.handler_errors.values())


class _RateLimitedLog:
    """Logs each kind of problem at most once per LOG_PERIOD_S."""

    def __init__(self, clock: Callable[[], float]) -> None:
        self._clock = clock
        self._last: Dict[Any, float] = {}

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
            statistics.handled[topic] = statistics.handled.get(topic, 0) + 1
            try:
                handler(self, message)
            except MalformedMessage as error:
                self._count_malformed(topic, str(error))
            except Exception:  # pylint: disable=broad-except
                statistics.handler_errors[topic] = (
                    statistics.handler_errors.get(topic, 0) + 1
                )
                self._log.warning(
                    ("handler", topic),
                    f"the handler of '{topic}' raised; the message was skipped",
                    exc_info=True,
                )

        return guarded  # type: ignore[return-value]

    return decorate


def _finite(value: float) -> bool:
    return math.isfinite(value)


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
    """Writes the bus's messages into `recording`.

    Args:
        recording: where everything goes (new_recording()).
        model: the robot's URDF geometry; None draws no robot (markers and plots only).
        instances: the robot instances whose meshes log_static() logs.
        clock: the wall clock [s since the epoch] of the wall_time timeline.
        max_pending_rows: per entity, before the scalar batcher sends without waiting for flush().
    """

    def __init__(
        self,
        recording: rr.RecordingStream,
        model: Optional[urdf_model.RobotModel],
        instances: Sequence[
            scene_contract.RobotInstanceStyle
        ] = scene_contract.ROBOT_INSTANCES,
        clock: Callable[[], float] = time.time,
        max_pending_rows: int = scalar_batcher.DEFAULT_MAX_PENDING_ROWS,
    ) -> None:
        self._recording = recording
        self._model = model
        self._meshes = robot_meshes.RobotMeshes(model) if model is not None else None
        self._instances = tuple(instances)
        self._clock = clock
        self._log = _RateLimitedLog(time.monotonic)
        # The status scalars, each entity on its own times.
        self._batcher = scalar_batcher.ScalarBatcher(recording, max_pending_rows)
        # viz/telemetry: one frame per message, every group a row of it.
        self._telemetry_batcher = scalar_batcher.FrameBatcher(
            recording, max_pending_rows
        )
        self.statistics = BridgeStatistics()

        self._links_with_visuals = (
            model.links_with_visuals() if model is not None else frozenset()
        )
        self._logged_instances: Set[str] = set()
        self._last_link_poses: Dict[str, Tuple[float, ...]] = {}
        self._last_scene_time = -math.inf
        # The marker entities the last scene drew, with their kind.
        self._drawn_markers: Dict[str, MarkerKind] = {}
        # Per telemetry entity: the series names its SeriesLines carry.
        self._series_names: Dict[str, Tuple[str, ...]] = {}
        self._status_series_logged: Set[str] = set()
        self._latest_robot_time: Optional[float] = None
        self._last_fsm_state: Optional[Tuple[Any, ...]] = None
        self._last_mpc_health: Optional[Tuple[Any, ...]] = None
        self._last_mpc_resets: Optional[Tuple[int, int]] = None
        self._warned: Set[Any] = set()
        self._last_report: Dict[str, int] = {}

    # ------------------------------------------------------------------------------------------------------------------
    # Static data
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def model(self) -> Optional[urdf_model.RobotModel]:
        return self._model

    @property
    def batcher(self) -> scalar_batcher.ScalarBatcher:
        """The status scalars' batcher."""
        return self._batcher

    @property
    def telemetry_batcher(self) -> scalar_batcher.FrameBatcher:
        return self._telemetry_batcher

    def log_static(self) -> None:
        """Logs the world's axes (right-handed, z up) and the robot meshes of every instance of `instances`."""
        self._recording.log(
            scene_contract.WORLD_ROOT, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True
        )
        for style in self._instances:
            self._log_instance_meshes(style)

    def _log_instance_meshes(self, style: scene_contract.RobotInstanceStyle) -> None:
        if self._meshes is None or style.name in self._logged_instances:
            return
        self._logged_instances.add(style.name)
        result = self._meshes.log_instance(self._recording, style)
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

    def _warn_once(self, key: Any, text: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            _LOGGER.warning("%s", text)

    def _set_time(self, robot_time: Optional[float], wall_time: float) -> None:
        """Sets the timelines of the following log() calls of this thread."""
        self._recording.reset_time()
        if robot_time is not None:
            self._recording.set_time(scene_contract.ROBOT_TIMELINE, duration=robot_time)
        self._recording.set_time(scene_contract.WALL_TIMELINE, timestamp=wall_time)

    # ------------------------------------------------------------------------------------------------------------------
    # viz/scene
    # ------------------------------------------------------------------------------------------------------------------

    @_guarded(topics.VIZ_SCENE)
    def handle_scene(self, scene: visualization_scene_pb2.VisualizationScene) -> None:
        """Moves the robot instances to the scene's link poses and draws its markers; clears markers it no longer
        names."""
        if not _finite(scene.time):
            raise MalformedMessage("the scene's time is not finite")
        if scene.time < self._last_scene_time:
            # The robot's clock went back (a restarted simulation): poses at the new times must all be logged.
            self._last_link_poses.clear()
        self._last_scene_time = scene.time
        self._latest_robot_time = scene.time
        self._set_time(scene.time, self._clock())

        for robot in scene.robots:
            self._guarded_part(topics.VIZ_SCENE, self._log_robot, robot)

        drawn: Dict[str, MarkerKind] = {}
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
        except MalformedMessage as error:
            self._count_malformed(topic, str(error))
            return False

    def _log_robot(self, robot: robot_model_instance_pb2.RobotModelInstance) -> None:
        name = robot.name
        if not scene_contract.is_valid_relative_path(name) or "/" in name:
            raise MalformedMessage(f"invalid robot instance name '{name}'")
        if len(robot.link_names) != len(robot.link_poses):
            raise MalformedMessage(
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
                raise MalformedMessage(
                    f"robot '{name}' link '{link}' has a pose that is not finite"
                )
            norm = math.sqrt(sum(value * value for value in values[3:]))
            if norm < _MIN_QUATERNION_NORM:
                raise MalformedMessage(
                    f"robot '{name}' link '{link}' has a zero quaternion"
                )
            path = scene_contract.link_path(name, link)
            if self._last_link_poses.get(path) == values:
                self.statistics.link_poses_unchanged += 1
                continue
            self._last_link_poses[path] = values
            self._recording.log(
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

    def _marker_style(
        self, path: str, kind: MarkerKind
    ) -> Tuple[float, Tuple[palette.Rgba, ...]]:
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
        defaults: Tuple[palette.Rgba, ...],
    ) -> np.ndarray:
        """The colors of `count` elements: the message's (one for all, or one each), else the marker's defaults."""
        if len(colors) == 0:
            return _defaults_rgba8(defaults, count)
        if len(colors) not in (1, count):
            raise MalformedMessage(
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
        radius_default, color_defaults = self._marker_style(
            arrows.path, MarkerKind.ARROWS
        )
        origins = _vector3_array(arrows.origins)
        vectors = _vector3_array(arrows.vectors)
        if len(origins) != len(vectors):
            raise MalformedMessage(
                f"'{arrows.path}' has {len(origins)} origins but {len(vectors)} vectors"
            )
        if not (np.isfinite(origins).all() and np.isfinite(vectors).all()):
            raise MalformedMessage(
                f"'{arrows.path}' has coordinates that are not finite"
            )
        radius = arrows.radius if arrows.radius > 0.0 else radius_default
        self._recording.log(
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
        radius_default, color_defaults = self._marker_style(
            spheres.path, MarkerKind.SPHERES
        )
        centers = _vector3_array(spheres.centers)
        count = len(centers)
        if not np.isfinite(centers).all():
            raise MalformedMessage(f"'{spheres.path}' has centers that are not finite")
        if len(spheres.radii) == 0:
            radii = np.array([radius_default], dtype=np.float32)
        elif len(spheres.radii) in (1, count):
            radii = np.asarray(spheres.radii, dtype=np.float32)
        else:
            raise MalformedMessage(
                f"'{spheres.path}' has {len(spheres.radii)} radii for {count} spheres (give one, or one each)"
            )
        self._recording.log(
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
        radius_default, color_defaults = self._marker_style(
            line_strips.path, MarkerKind.LINE_STRIPS
        )
        strips: List[np.ndarray] = []
        defaults = _defaults_rgba8(color_defaults, len(line_strips.strips))
        colors = np.empty((len(line_strips.strips), 4), dtype=np.uint8)
        for index, strip in enumerate(line_strips.strips):
            points = _vector3_array(strip.points)
            if not np.isfinite(points).all():
                raise MalformedMessage(
                    f"'{line_strips.path}' strip {index} has points that are not finite"
                )
            strips.append(points)
            if strip.HasField("color"):
                colors[index] = _colors_rgba8([strip.color])[0]
            else:
                colors[index] = defaults[index % len(defaults)]
        radius = line_strips.radius if line_strips.radius > 0.0 else radius_default
        self._recording.log(
            scene_contract.entity_path(scene_contract.WORLD_ROOT, line_strips.path),
            rr.LineStrips3D(strips, radii=[radius], colors=colors),
        )

    def _clear_marker(self, path: str, kind: MarkerKind) -> None:
        """Draws nothing at the marker `path` from now on (an empty archetype of its kind)."""
        entity = scene_contract.entity_path(scene_contract.WORLD_ROOT, path)
        empty = np.zeros((0, 3), dtype=np.float64)
        if kind is MarkerKind.ARROWS:
            self._recording.log(entity, rr.Arrows3D(origins=empty, vectors=empty))
        elif kind is MarkerKind.SPHERES:
            self._recording.log(entity, rr.Points3D(positions=empty))
        else:
            self._recording.log(entity, rr.LineStrips3D([]))

    # ------------------------------------------------------------------------------------------------------------------
    # viz/telemetry
    # ------------------------------------------------------------------------------------------------------------------

    @_guarded(topics.VIZ_TELEMETRY)
    def handle_telemetry(self, series: telemetry_series_pb2.TelemetrySeries) -> None:
        """Buffers every ScalarGroup at telemetry/<path>; flush() sends them."""
        if not _finite(series.time):
            raise MalformedMessage("the telemetry's time is not finite")
        robot_time = series.time
        self._latest_robot_time = robot_time
        wall_time = self._clock()
        rows: List[Tuple[str, Sequence[float]]] = []
        paths: Set[str] = set()
        for group in series.groups:
            path = group.path
            if path in paths:
                self._count_malformed(
                    topics.VIZ_TELEMETRY, f"group '{path}' appears twice in one message"
                )
                continue
            paths.add(path)
            count = len(group.values)
            names = self._series_names.get(path)
            if names is None or len(names) != count:
                if not self._declare_series(group):
                    continue
            rows.append((f"{scene_contract.TELEMETRY_ROOT}/{path}", group.values))
        self._telemetry_batcher.append(robot_time, wall_time, rows)
        self.statistics.telemetry_groups += len(rows)

    def _declare_series(self, group: Any) -> bool:
        """Logs the static SeriesLines of a group seen for the first time, or whose size changed."""
        path = group.path
        if not scene_contract.is_valid_relative_path(path):
            self._count_malformed(topics.VIZ_TELEMETRY, f"invalid group path '{path}'")
            return False
        names = tuple(group.names)
        if len(names) != len(group.values):
            self._count_malformed(
                topics.VIZ_TELEMETRY,
                f"group '{path}' has {len(names)} names but {len(group.values)} values",
            )
            return False
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
        self._recording.log(
            f"{scene_contract.TELEMETRY_ROOT}/{path}",
            rr.SeriesLines(names=list(names), colors=colors),
            static=True,
        )
        self._series_names[path] = names
        return True

    def flush(self) -> int:
        """Sends the buffered scalars; returns how many rows."""
        try:
            return self._telemetry_batcher.flush() + self._batcher.flush()
        except Exception:  # pylint: disable=broad-except
            self.statistics.handler_errors["flush"] = (
                self.statistics.handler_errors.get("flush", 0) + 1
            )
            self._log.warning("flush", "sending the plots failed", exc_info=True)
            return 0

    def flush_recording(self, timeout: float) -> bool:
        """Waits up to `timeout` seconds for the recording to reach its sink; False when it did not."""
        try:
            self._recording.flush(timeout_sec=timeout)
            return True
        except Exception:  # pylint: disable=broad-except
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
        robot_time: Optional[float],
        wall_time: float,
        values: Sequence[float],
    ) -> None:
        if series.path not in self._status_series_logged:
            self._status_series_logged.add(series.path)
            self._recording.log(
                series.path, rr.SeriesLines(names=list(series.names)), static=True
            )
        self._batcher.append(series.path, robot_time, wall_time, values)

    def _text(
        self, path: str, robot_time: Optional[float], text: str, level: str
    ) -> None:
        self._set_time(robot_time, self._clock())
        self._recording.log(path, rr.TextLog(text, level=level))

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
        """Plots the solve time, failures and observations; logs a line when the solver's health or resets change."""
        robot_time = (
            status.observation_time if _finite(status.observation_time) else None
        )
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
            if solver.healthy:
                text, level = "MPC solver healthy", rr.TextLogLevel.INFO
            else:
                text = (
                    f"MPC solver UNHEALTHY after {solver.consecutive_failures} consecutive failures: "
                    f"{solver.last_error or '(no error text)'}"
                )
                level = rr.TextLogLevel.ERROR
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
        """Plots the loop's period, compute time, lateness, overruns, missed periods, drops and policy age; logs a line."""
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
        trouble = (
            timing.overruns
            or timing.missed_periods
            or timing.telemetry_samples_dropped
            or timing.stale_policies_dropped
        )
        text = (
            f"loop {timing.cycles} cycles: period target {timing.target_period_s * milliseconds:.3f} ms, "
            f"mean {timing.mean_period_s * milliseconds:.3f} ms, max {timing.max_period_s * milliseconds:.3f} ms; "
            f"compute max {timing.max_compute_time_s * milliseconds:.3f} ms; "
            f"wake-up up to {timing.max_lateness_s * milliseconds:.3f} ms late; {timing.overruns} overruns, "
            f"{timing.missed_periods} missed periods, {timing.telemetry_samples_dropped} telemetry samples dropped, "
            f"{timing.stale_policies_dropped} stale policies dropped; "
            + (
                f"policy age {timing.policy_age_s:.3f} s"
                if timing.policy_age_s >= 0.0
                else "no policy"
            )
        )
        level = rr.TextLogLevel.WARN if trouble else rr.TextLogLevel.DEBUG
        self._text(status_contract.LOOP_TIMING_LOG, robot_time, text, level)

    # ------------------------------------------------------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------------------------------------------------------

    def report(self, bus_rejected: int = 0) -> Optional[str]:
        """A line about the problems since the last report (malformed parts, handler errors, messages the bus
        rejected), also written to the bridge's text log; None when there were none."""
        totals = {
            "malformed": self.statistics.malformed_count(),
            "handler errors": self.statistics.handler_error_count(),
            "rejected by the bus": bus_rejected,
        }
        new = {
            name: total - self._last_report.get(name, 0)
            for name, total in totals.items()
        }
        self._last_report = totals
        problems = [f"{count} {name}" for name, count in new.items() if count > 0]
        if not problems:
            return None
        text = "since the last report: " + ", ".join(problems)
        try:
            self._text(
                status_contract.BRIDGE_LOG,
                self._latest_robot_time,
                text,
                rr.TextLogLevel.WARN,
            )
        except Exception:  # pylint: disable=broad-except
            self._log.warning(
                "report", "logging the bridge's report failed", exc_info=True
            )
        return text
