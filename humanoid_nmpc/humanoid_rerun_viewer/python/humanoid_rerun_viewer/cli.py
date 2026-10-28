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

"""The command line of the Rerun bridge, which draws the 3D scene and the plots of the robot and the MPC.

    bazel run //humanoid_nmpc/humanoid_rerun_viewer -- \\
        --urdf robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf      # spawns the native viewer
    ... --rerun_sink serve_web            # web viewer at http://localhost:9090 (from the dev container's host too)
    ... --rerun_sink connect --rerun_url rerun+http://192.168.1.20:9876/proxy
    ... --rerun_sink save --rrd_path /tmp/walk.rrd --duration 30

Relative paths are looked up in the directory the bridge was started from (BUILD_WORKING_DIRECTORY under `bazel run`)
and then in the repository root. Ctrl-C (or SIGTERM) stops it cleanly: the bus is closed, what is queued is drawn, the
buffered plots are sent and the recording is flushed, within a bounded time even when the viewer takes no data.

The viewer of spawn and connect may go away. The bridge probes it (ViewerWatch): a spawned viewer that is gone (closed,
or never started) ends the run, so that whoever started the bridge, such as the GUI's button, can start another; while
a connect viewer is unreachable, the bridge drops the messages, and when it is back, it connects again and sends the
robot and the layout afresh. Exit status: 0 on success, 2 for a usage or configuration error.
"""

import argparse
from collections.abc import Callable, Sequence
import enum
import functools
import logging
import math
import os
import signal
import socket
import sys
import threading
import time
from typing import Any, NamedTuple

import rerun as rr

from humanoid_rerun_viewer import blueprint
from humanoid_rerun_viewer import bridge as bridge_module
from humanoid_rerun_viewer import bus_bridge
from humanoid_rerun_viewer import plot_config as plot_config_module
from humanoid_rerun_viewer import rerun_sinks
from humanoid_rerun_viewer import terrain_drawing
from humanoid_rerun_viewer import urdf_model
import robot_ipc

_LOGGER = logging.getLogger("humanoid_rerun_viewer")

EXIT_OK = 0
EXIT_USAGE = 2


class RunResult(NamedTuple):
    """How run() ended: the exit status, and whether the recording reached its sink at the end."""

    exit_status: int
    recording_delivered: bool = True


DEFAULT_NETWORK_CONFIG = os.path.join("config", "ipc", "network.textproto")
# The shipped plot configuration, which the launch targets give the robot of a simulation too.
# LINT.IfChange(default_plot_config)
DEFAULT_PLOT_CONFIG = os.path.join(
    "humanoid_nmpc", "humanoid_rerun_viewer", "config", "plot_config.textproto"
)
# LINT.ThenChange(//Makefile:plot_config_variable, //tools/deploy/session.sh:default_plot_config, //docker-compose.robot.sim.yaml:sim_plot_config_mount)
DEFAULT_APP_ID = "humanoid_nmpc"
DEFAULT_WEB_MAX_SCENE_FREQUENCY = 12.5
# Every flush sends one chunk per plotted entity, and the browser's cost is per chunk more than per row (README.md,
# "Cost"): 0.2 s halves the chunks of 0.1 s for a plot that trails the robot by 0.2 s more.
DEFAULT_WEB_FLUSH_PERIOD_S = 0.2
# How often the main thread checks for a stop request and the duration [s].
_POLL_PERIOD_S = 0.1
# LINT.IfChange(viewer_watch)
# How often the viewer of spawn and connect is probed, how long a probe may take, and how long it must be unreachable
# to be gone [s].
VIEWER_PROBE_PERIOD_S = 1.0
VIEWER_PROBE_TIMEOUT_S = 0.2
VIEWER_GONE_AFTER_S = 3.0
# The writer is stalled when one of its tasks has taken this long [s]: a sink that takes no data.
WRITER_STALL_S = 3.0
# LINT.ThenChange(//humanoid_nmpc/humanoid_rerun_viewer/README.md:viewer_watch)


def resolve_input_path(path: str) -> str:
    """Finds a path named on the command line: as is when absolute, else in the start directory, else in the repository.

    Args:
        path: the path as the user wrote it.

    Returns:
        The first existing candidate, or the first candidate when none exists, so that the error names the path the
        user meant.
    """
    if os.path.isabs(path):
        return path
    candidates = [
        os.path.join(
            os.environ.get("BUILD_WORKING_DIRECTORY", os.getcwd()), path
        )
    ]
    workspace = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if workspace:
        candidates.append(os.path.join(workspace, path))
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def resolve_output_path(path: str) -> str:
    """A file to write: relative to the start directory (BUILD_WORKING_DIRECTORY under `bazel run`)."""
    if os.path.isabs(path):
        return path
    return os.path.join(
        os.environ.get("BUILD_WORKING_DIRECTORY", os.getcwd()), path
    )


def build_parser() -> argparse.ArgumentParser:
    """The parser of the bridge's command line."""
    parser = argparse.ArgumentParser(
        prog="humanoid_rerun_viewer",
        description=(
            "Draws the bus's viz/scene, viz/telemetry and status messages in"
            " Rerun."
        ),
    )
    parser.add_argument(
        "--urdf",
        default="",
        help=(
            "the robot's URDF, whose meshes are drawn for every robot instance;"
            " without it no robot is drawn"
        ),
    )
    parser.add_argument(
        "--package_path",
        action="append",
        default=[],
        metavar="DIR",
        help=(
            "a directory to search for the packages of package:// URIs, besides"
            " the URDF's ancestors and robot_models/ (repeatable)"
        ),
    )
    parser.add_argument(
        "--network_config",
        default=DEFAULT_NETWORK_CONFIG,
        help="the network file of the bus (default: %(default)s)",
    )
    # LINT.IfChange(sink_flags)
    parser.add_argument(
        "--rerun_sink",
        default=rerun_sinks.DEFAULT_SINK,
        choices=rerun_sinks.sink_names(),
        help="where the recording goes (default: %(default)s)",
    )
    parser.add_argument(
        "--rerun_url",
        default=rerun_sinks.DEFAULT_CONNECT_URL,
        help="connect: the viewer's gRPC URL (default: %(default)s)",
    )
    parser.add_argument(
        "--rrd_path", default="", help="save: the .rrd file to write"
    )
    parser.add_argument(
        "--web_port",
        type=int,
        default=rerun_sinks.DEFAULT_WEB_PORT,
        help="serve_web: the web viewer's HTTP port (default: %(default)s)",
    )
    parser.add_argument(
        "--grpc_port",
        type=int,
        default=rerun_sinks.DEFAULT_GRPC_PORT,
        help=(
            "spawn: the viewer's port; serve_web: the recording's gRPC port"
            " (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--open_browser",
        action="store_true",
        help="serve_web: also open the web viewer in a browser of this machine",
    )
    parser.add_argument(
        "--viewer_executable",
        default="",
        help=(
            "spawn: the viewer to start (default: the native viewer of the"
            " rerun-sdk wheel)"
        ),
    )
    parser.add_argument(
        "--max_scene_frequency",
        type=float,
        default=None,
        help=(
            "maximum scene messages to log per wall second (default:"
            f" {DEFAULT_WEB_MAX_SCENE_FREQUENCY} for serve_web, unlimited"
            " otherwise)"
        ),
    )
    parser.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "subscribe to viz/telemetry and show plot tabs in blueprint"
            " (default: on when a plot configuration is loaded, as the shipped"
            " default is; without one, off for serve_web and on otherwise)"
        ),
    )
    parser.add_argument(
        "--plot_config",
        default=DEFAULT_PLOT_CONFIG,
        help=(
            "a plot configuration file (.textproto or text file) specifying"
            " topics/signals to plot ('none' to disable; default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--terminal_state",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "draw the end-of-trajectory robots (terminal_state,"
            " terminal_target) of viz/scene (default: the plot configuration's"
            " draw_terminal_state, which the shipped one sets false; on without"
            " a configuration)"
        ),
    )
    parser.add_argument(
        "--robot_instances",
        default="",
        help=(
            "comma-separated list of robot instances to draw (e.g. 'measured',"
            " 'measured,terminal_state'; default: the plot configuration's"
            " robot_instances, every instance when it names none)"
        ),
    )
    # LINT.ThenChange(//humanoid_nmpc/humanoid_rerun_viewer/README.md:sink_flags)
    parser.add_argument(
        "--multi_contact_scenario",
        default="",
        help=(
            "a multi-contact scenario textproto whose terrain is drawn under"
            " world/terrain, statically (the multi-contact dummy sims; default:"
            " none)"
        ),
    )
    parser.add_argument(
        "--app_id",
        default=DEFAULT_APP_ID,
        help=(
            "the Rerun application id; recordings of one id share the viewer's"
            " layout (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--follow_robot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "the 3D view's eye follows the measured robot's root link"
            " (default: on)"
        ),
    )
    parser.add_argument(
        "--flush_period",
        type=float,
        default=None,
        help=(
            "how often the buffered plots are sent [s] (default:"
            f" {DEFAULT_WEB_FLUSH_PERIOD_S} for serve_web,"
            f" {bus_bridge.DEFAULT_FLUSH_PERIOD_S} otherwise)"
        ),
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help=(
            "stop after this many seconds; 0: run until Ctrl-C (default:"
            " %(default)s)"
        ),
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="of the bridge's own messages (default: %(default)s)",
    )
    return parser


def load_model(urdf: str, package_path: Sequence[str]) -> urdf_model.RobotModel:
    """Loads the URDF's model, its meshes looked up as the module urdf_model describes; warns about meshes not drawn.

    Args:
        urdf: the URDF file, as resolve_input_path() finds it.
        package_path: more directories to search for the packages of package:// URIs.

    Returns:
        The model.

    Raises:
        urdf_model.UrdfError: the URDF cannot be read.
    """
    path = resolve_input_path(urdf)
    search_roots = tuple(
        resolve_input_path(directory) for directory in package_path
    ) + urdf_model.default_search_roots(path)
    model = urdf_model.load_urdf(path, search_roots)
    for uri in model.missing_meshes:
        _LOGGER.warning(
            "%s: mesh '%s' not found (add its package's parent with"
            " --package_path); not drawn",
            path,
            uri,
        )
    for uri in model.unsupported_meshes:
        _LOGGER.warning(
            "%s: mesh '%s' is in a format Rerun does not draw (%s) and has no"
            " twin in one; not drawn",
            path,
            uri,
            ", ".join(urdf_model.MESH_MEDIA_TYPES),
        )
    return model


def plots_enabled(
    plots_flag: bool | None,
    sink: str,
    plot_config: plot_config_module.PlotConfig | None,
) -> bool:
    """Whether the bridge subscribes to viz/telemetry and shows plot tabs.

    Args:
        plots_flag: --plots (True) or --no-plots (False); None when neither was given.
        sink: the Rerun sink's name.
        plot_config: the plot configuration, None without one (--plot_config none, or the default file missing).

    Returns:
        The flag when given; otherwise on with a plot configuration, which limits the plots to its signals, and without
        one on for every sink but serve_web, whose browser would otherwise get every telemetry series.
    """
    if plots_flag is not None:
        return plots_flag
    if plot_config is not None:
        return True
    return sink != "serve_web"


class ViewerChange(enum.Enum):
    """What ViewerWatch.poll() saw happen to the viewer."""

    GONE = "gone"
    BACK = "back"


def _probe_tcp(endpoint: tuple[str, int], timeout: float) -> None:
    """Connects to `endpoint` and closes at once; raises OSError when nothing listens there."""
    with socket.create_connection(endpoint, timeout=timeout):
        pass


class ViewerWatch:
    """Probes the viewer of a client sink (spawn, connect) and tells when it went away or came back.

    A TCP connection to the viewer's gRPC port once every VIEWER_PROBE_PERIOD_S: unreachable for VIEWER_GONE_AFTER_S
    is gone, one success is back. Not thread-safe: cli.run's main thread polls it.

    Args:
        endpoint: the viewer's (host, port).
        up: whether the viewer is up to begin with.
        probe: connects to the endpoint within a timeout [s], raising OSError when nothing listens; the tests pass a
            fake one.
        clock: monotonic seconds.

    Attributes:
        endpoint: The viewer's (host, port).
        up: Whether the viewer was up at the last poll(); only poll() changes it.
    """

    def __init__(
        self,
        endpoint: tuple[str, int],
        up: bool,
        probe: Callable[[tuple[str, int], float], None] = _probe_tcp,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.endpoint = endpoint
        self.up = up
        self._probe = probe
        self._clock = clock
        self._next_probe = clock() + VIEWER_PROBE_PERIOD_S
        self._unreachable_since: float | None = None

    def reachable(self) -> bool:
        """Probes the viewer once, now."""
        try:
            self._probe(self.endpoint, VIEWER_PROBE_TIMEOUT_S)
            return True
        except OSError:
            return False

    def poll(self) -> ViewerChange | None:
        """Probes the viewer when a probe is due; returns what changed, None when nothing did."""
        now = self._clock()
        if now < self._next_probe:
            return None
        self._next_probe = now + VIEWER_PROBE_PERIOD_S
        if self.reachable():
            self._unreachable_since = None
            if self.up:
                return None
            self.up = True
            return ViewerChange.BACK
        if self._unreachable_since is None:
            self._unreachable_since = now
        if self.up and now - self._unreachable_since >= VIEWER_GONE_AFTER_S:
            self.up = False
            return ViewerChange.GONE
        return None


def _attach_new_run(
    sink: rerun_sinks.Sink, application_id: str
) -> rr.RecordingStream:
    """A recording for a new run of the robot, attached to `sink` (the bridge's new_run)."""
    recording = bridge_module.new_recording(application_id)
    _LOGGER.info("%s", sink.attach(recording))
    return recording


def _reconnect(
    sink: rerun_sinks.Sink, the_bridge: bridge_module.RerunBridge
) -> None:
    """Attaches the bridge's recording to its viewer again and sends the static data: the writer's resume task."""
    _LOGGER.info("%s", sink.reattach(the_bridge.recording))
    the_bridge.resend_static()


def _follow_viewer(
    watch: ViewerWatch,
    sink: rerun_sinks.Sink,
    feed: bus_bridge.BusBridge,
    the_bridge: bridge_module.RerunBridge,
) -> bool:
    """Polls the viewer: pauses the feed while a connect viewer is gone, reconnects when it is back.

    Args:
        watch: The watch polling the viewer endpoint for reachable status.
        sink: The Rerun sink managing the connection to the viewer.
        feed: The bus bridge whose streaming is paused or resumed.
        the_bridge: The Rerun bridge logging the robot and scene.

    Returns:
        False when the run ends: the viewer the bridge spawned is gone.
    """
    change = watch.poll()
    host, port = watch.endpoint
    if change is ViewerChange.GONE:
        if sink.owns_viewer:
            _LOGGER.warning(
                "the spawned Rerun viewer on port %d is gone (closed, or it"
                " never started); stopping",
                port,
            )
            return False
        feed.pause(f"the Rerun viewer at {host}:{port} is unreachable")
        _LOGGER.warning(
            "the Rerun viewer at %s:%d is unreachable; dropping the messages"
            " until it is back",
            host,
            port,
        )
    elif change is ViewerChange.BACK:
        _LOGGER.info(
            "the Rerun viewer at %s:%d is back; sending it the robot and the"
            " layout again",
            host,
            port,
        )
        feed.resume(functools.partial(_reconnect, sink, the_bridge))
    return True


class _StallWarning:
    """Logs once per stall of the bridge's writer that it takes no data."""

    def __init__(self) -> None:
        self._warned = False

    def check(self, feed: bus_bridge.BusBridge) -> None:
        stalled_for = feed.writer_stalled_for()
        if stalled_for < WRITER_STALL_S:
            self._warned = False
        elif not self._warned:
            self._warned = True
            _LOGGER.warning(
                "the Rerun sink has taken no data for %.1f s (a viewer that"
                " stopped reading?); the bus's messages are dropped once the"
                " bridge's queue is full",
                stalled_for,
            )


class _StopRequest:
    """Set by SIGINT or SIGTERM; a second signal falls back to the default action (an immediate exit)."""

    def __init__(self) -> None:
        self.event = threading.Event()
        self._previous: dict[int, Any] = {}

    def install(self) -> None:
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._previous[signum] = signal.signal(signum, self._handle)

    def restore(self) -> None:
        for signum, handler in self._previous.items():
            signal.signal(signum, handler)
        self._previous.clear()

    def _handle(self, signum: int, frame: object) -> None:
        del frame
        self.event.set()
        signal.signal(signum, signal.SIG_DFL)


def _start_watch(
    sink: rerun_sinks.Sink,
    feed: bus_bridge.BusBridge,
    the_bridge: bridge_module.RerunBridge,
) -> ViewerWatch | None:
    """Logs the static data, or pauses the feed while the viewer of a connect sink is not up; the watch of its viewer.

    rerun-sdk buffers what an absent viewer does not take, the robot's meshes first: the static data waits for the
    viewer, which then gets it when it is back (_follow_viewer()). A spawned viewer counts as up once spawn returned.

    Args:
        sink: The Rerun sink managing the connection to the viewer.
        feed: The bus bridge whose streaming is paused or resumed.
        the_bridge: The Rerun bridge logging the robot and scene.

    Returns:
        The ViewerWatch instance monitoring the endpoint, or None if the sink has no viewer endpoint.
    """
    endpoint = sink.viewer_endpoint
    if endpoint is None:
        the_bridge.log_static()
        return None
    watch = ViewerWatch(endpoint, up=True)
    if sink.owns_viewer or watch.reachable():
        the_bridge.log_static()
        return watch
    host, port = endpoint
    feed.pause(f"the Rerun viewer at {host}:{port} is unreachable")
    _LOGGER.warning(
        "the Rerun viewer at %s:%d is not up; drawing nothing until it is",
        host,
        port,
    )
    return ViewerWatch(endpoint, up=False)


def run(
    args: argparse.Namespace, stop: threading.Event | None = None
) -> RunResult:
    """Runs the bridge until `stop` is set or the duration passes.

    Args:
        args: the parsed command line of build_parser(), every flag present.
        stop: set to end the run; None runs until the duration passes or Ctrl-C.

    Returns:
        The exit status, and whether the recording reached its sink.
    """
    stop = stop if stop is not None else threading.Event()
    try:
        model = load_model(args.urdf, args.package_path) if args.urdf else None
        network = robot_ipc.load_network_config(
            resolve_input_path(args.network_config)
        )
    except (
        urdf_model.UrdfError,
        robot_ipc.NetworkConfigError,
        OSError,
    ) as error:
        _LOGGER.error("%s", error)
        return RunResult(EXIT_USAGE)
    if model is None:
        _LOGGER.warning(
            "no --urdf given: the robot instances will not be drawn"
        )

    static_scene = None
    if args.multi_contact_scenario:
        try:
            terrain = terrain_drawing.load_scenario_terrain(
                resolve_input_path(args.multi_contact_scenario)
            )
        except (OSError, ValueError) as error:
            _LOGGER.error("cannot load the scenario's terrain: %s", error)
            return RunResult(EXIT_USAGE)
        static_scene = functools.partial(
            terrain_drawing.log_terrain, scene=terrain
        )

    recording = bridge_module.new_recording(args.app_id)
    plot_config: plot_config_module.PlotConfig | None = None
    plot_config_arg = args.plot_config
    if plot_config_arg and plot_config_arg.lower() not in (
        "none",
        "off",
        "false",
        "0",
    ):
        config_path = resolve_input_path(plot_config_arg)
        try:
            plot_config = plot_config_module.load_plot_config(config_path)
        except (plot_config_module.PlotConfigError, OSError) as error:
            if plot_config_arg == DEFAULT_PLOT_CONFIG and isinstance(
                error, OSError
            ):
                _LOGGER.warning(
                    "default plot config '%s' not found: %s",
                    plot_config_arg,
                    error,
                )
            else:
                _LOGGER.error("cannot load plot config: %s", error)
                return RunResult(EXIT_USAGE)

    plots = plots_enabled(args.plots, args.rerun_sink, plot_config)

    cli_robot_instances: tuple[str, ...] | None = None
    if args.robot_instances:
        names = (name.strip() for name in args.robot_instances.split(","))
        cli_robot_instances = tuple(name for name in names if name)
    allowed_instances = plot_config_module.resolve_allowed_instances(
        config=plot_config,
        cli_robot_instances=cli_robot_instances,
        cli_terminal_state=args.terminal_state,
    )

    layout = blueprint.build_blueprint(
        tracked_link=(
            model.root_link if model is not None and args.follow_robot else None
        ),
        plots=plots,
        config=plot_config,
        allowed_instances=allowed_instances,
    )
    options = rerun_sinks.SinkOptions(
        url=args.rerun_url,
        rrd_path=resolve_output_path(args.rrd_path) if args.rrd_path else "",
        grpc_port=args.grpc_port,
        web_port=args.web_port,
        open_browser=args.open_browser,
        viewer_executable=(
            resolve_input_path(args.viewer_executable)
            if args.viewer_executable
            else ""
        ),
    )
    sink = rerun_sinks.make_sink(args.rerun_sink, layout, options)
    if args.max_scene_frequency is not None:
        max_scene_frequency = args.max_scene_frequency
    elif args.rerun_sink == "serve_web":
        max_scene_frequency = DEFAULT_WEB_MAX_SCENE_FREQUENCY
    else:
        max_scene_frequency = 0.0

    if args.flush_period is not None:
        flush_period = args.flush_period
    elif args.rerun_sink == "serve_web":
        flush_period = DEFAULT_WEB_FLUSH_PERIOD_S
    else:
        flush_period = bus_bridge.DEFAULT_FLUSH_PERIOD_S

    telemetry_rate = (
        plot_config.telemetry_rate_hz
        if plot_config is not None
        else plot_config_module.DEFAULT_TELEMETRY_RATE_HZ
    )
    telemetry_selection = plot_config_module.telemetry_selection(plot_config)
    the_bridge = bridge_module.RerunBridge(
        recording,
        model,
        max_scene_frequency=max_scene_frequency,
        allowed_instances=allowed_instances,
        new_run=functools.partial(_attach_new_run, sink, args.app_id),
        telemetry_rate=telemetry_rate,
        telemetry_selection=telemetry_selection,
        static_scene=static_scene,
    )
    telemetry_enabled = plots and (
        plot_config is None
        or plot_config_module.has_telemetry_signals(plot_config)
    )
    try:
        feed = bus_bridge.BusBridge(
            the_bridge,
            network,
            flush_period=flush_period,
            telemetry=telemetry_enabled,
        )
    except (
        ValueError,
        robot_ipc.BusError,
        robot_ipc.NetworkConfigError,
    ) as error:
        _LOGGER.error("cannot subscribe to the bus: %s", error)
        return RunResult(EXIT_USAGE)
    try:
        _LOGGER.info("%s", sink.attach(recording))
    except (ValueError, RuntimeError, OSError) as error:
        _LOGGER.error(
            "cannot start the Rerun sink '%s': %s", args.rerun_sink, error
        )
        feed.bus.close()
        return RunResult(EXIT_USAGE)
    watch = _start_watch(sink, feed, the_bridge)
    _LOGGER.info(
        "subscribed to %s on DDS domain %d (peers %s)",
        ", ".join(feed.subscribed_topics),
        feed.bus.domain_id,
        ", ".join(feed.bus.discovery_peers),
    )
    if telemetry_enabled:
        _LOGGER.info(
            "recording viz/telemetry %s, %s",
            (
                f"at most at {telemetry_rate:g} Hz of robot time"
                if telemetry_rate > 0.0
                else "at every sample"
            ),
            (
                "every group"
                if telemetry_selection is None
                else "the plotted groups"
            ),
        )
    deadline = time.monotonic() + args.duration if args.duration > 0.0 else None
    stall_warning = _StallWarning()
    feed.start()
    try:
        while not stop.wait(_POLL_PERIOD_S):
            if deadline is not None and time.monotonic() >= deadline:
                break
            if watch is not None and not _follow_viewer(
                watch, sink, feed, the_bridge
            ):
                break
            stall_warning.check(feed)
    finally:
        delivered = feed.stop()
        if delivered:
            sink.disconnect_all()
        else:
            _LOGGER.warning(
                "the Rerun sink is unreachable; exiting without waiting for it"
            )
    _LOGGER.info(
        "stopped: %s",
        ", ".join(
            f"{count} {topic}"
            for topic, count in the_bridge.statistics.handled.items()
        )
        or "no messages",
    )
    return RunResult(EXIT_OK, delivered)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.getLevelNamesMapping()[args.log_level],
        format="humanoid_rerun_viewer: %(levelname)s: %(message)s",
    )
    if args.rerun_sink == "save" and not args.rrd_path:
        parser.error("--rerun_sink save needs --rrd_path")
    if args.max_scene_frequency is not None and not (
        args.max_scene_frequency >= 0.0
        and math.isfinite(args.max_scene_frequency)
    ):
        parser.error("--max_scene_frequency must be non-negative")
    # The not-form rejects NaN, which `args.flush_period <= 0.0` would let through.
    if args.flush_period is not None and not args.flush_period > 0.0:
        parser.error("--flush_period must be positive")
    stop_request = _StopRequest()
    stop_request.install()
    try:
        result = run(args, stop_request.event)
    finally:
        stop_request.restore()
    if not result.recording_delivered:
        # rerun-sdk 0.38 waits without end at exit for a gRPC sink it cannot reach; nothing more can be delivered.
        logging.shutdown()
        sys.stdout.flush()
        sys.stderr.flush()
        # pylint: disable-next=protected-access  # os._exit is public API: an exit that skips the exit handlers.
        os._exit(result.exit_status)
    return result.exit_status
