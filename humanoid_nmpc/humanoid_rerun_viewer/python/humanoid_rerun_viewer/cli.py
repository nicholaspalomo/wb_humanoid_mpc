"""The command line of the Rerun bridge, which draws the 3D scene and the plots of the robot and the MPC.

    bazel run //humanoid_nmpc/humanoid_rerun_viewer -- \\
        --urdf robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf      # spawns the native viewer
    ... --rerun_sink serve_web            # web viewer at http://localhost:9090 (from the dev container's host too)
    ... --rerun_sink connect --rerun_url rerun+http://192.168.1.20:9876/proxy
    ... --rerun_sink save --rrd_path /tmp/walk.rrd --duration 30

Relative paths are looked up in the directory the bridge was started from (BUILD_WORKING_DIRECTORY under `bazel run`)
and then in the repository root. Ctrl-C (or SIGTERM) stops it cleanly: the bus is closed, the buffered plots are sent
and the recording is flushed. Exit status: 0 on success, 2 for a usage or configuration error.
"""

import argparse
import logging
import os
import signal
import sys
import threading
import time
from typing import Any, Dict, List, NamedTuple, Optional, Sequence

import robot_ipc

from humanoid_rerun_viewer import blueprint
from humanoid_rerun_viewer import bridge as bridge_module
from humanoid_rerun_viewer import bus_bridge
from humanoid_rerun_viewer import rerun_sinks
from humanoid_rerun_viewer import urdf_model

_LOGGER = logging.getLogger("humanoid_rerun_viewer")

EXIT_OK = 0
EXIT_USAGE = 2


class RunResult(NamedTuple):
    """How run() ended: the exit status, and whether the recording reached its sink at the end."""

    exit_status: int
    recording_delivered: bool = True


DEFAULT_NETWORK_CONFIG = os.path.join("config", "ipc", "network.textproto")
DEFAULT_APP_ID = "humanoid_nmpc"
# How often the main thread checks for a stop request and the duration [s].
_POLL_PERIOD_S = 0.1


def resolve_input_path(path: str) -> str:
    """A path named on the command line: as is when absolute, else in the start directory, else in the repository.

    Returns the first existing candidate, or the first candidate when none exists, so that the error names the path the
    user meant.
    """
    if os.path.isabs(path):
        return path
    candidates = [
        os.path.join(os.environ.get("BUILD_WORKING_DIRECTORY", os.getcwd()), path)
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
    return os.path.join(os.environ.get("BUILD_WORKING_DIRECTORY", os.getcwd()), path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="humanoid_rerun_viewer",
        description="Draws the bus's viz/scene, viz/telemetry and status messages in Rerun.",
    )
    parser.add_argument(
        "--urdf",
        default="",
        help="the robot's URDF, whose meshes are drawn for every robot instance; without it no robot is drawn",
    )
    parser.add_argument(
        "--package_path",
        action="append",
        default=[],
        metavar="DIR",
        help="a directory to search for the packages of package:// URIs, besides the URDF's ancestors and "
        "robot_models/ (repeatable)",
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
    parser.add_argument("--rrd_path", default="", help="save: the .rrd file to write")
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
        help="spawn: the viewer's port; serve_web: the recording's gRPC port (default: %(default)s)",
    )
    parser.add_argument(
        "--open_browser",
        action="store_true",
        help="serve_web: also open the web viewer in a browser of this machine",
    )
    # LINT.ThenChange(//humanoid_nmpc/humanoid_rerun_viewer/README.md:sink_flags)
    parser.add_argument(
        "--app_id",
        default=DEFAULT_APP_ID,
        help="the Rerun application id; recordings of one id share the viewer's layout (default: %(default)s)",
    )
    parser.add_argument(
        "--follow_robot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="the 3D view's eye follows the measured robot's root link (default: on)",
    )
    parser.add_argument(
        "--flush_period",
        type=float,
        default=bus_bridge.DEFAULT_FLUSH_PERIOD_S,
        help="how often the buffered plots are sent [s] (default: %(default)s)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="stop after this many seconds; 0: run until Ctrl-C (default: %(default)s)",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="of the bridge's own messages (default: %(default)s)",
    )
    return parser


def load_model(urdf: str, package_path: Sequence[str]) -> urdf_model.RobotModel:
    """The URDF's model, its meshes looked up as the module urdf_model describes; warns about meshes not drawn.

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
            "%s: mesh '%s' not found (add its package's parent with --package_path); not drawn",
            path,
            uri,
        )
    for uri in model.unsupported_meshes:
        _LOGGER.warning(
            "%s: mesh '%s' is in a format Rerun does not draw (%s) and has no twin in one; not drawn",
            path,
            uri,
            ", ".join(urdf_model.MESH_MEDIA_TYPES),
        )
    return model


class _StopRequest:
    """Set by SIGINT or SIGTERM; a second signal falls back to the default action (an immediate exit)."""

    def __init__(self) -> None:
        self.event = threading.Event()
        self._previous: Dict[int, Any] = {}

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


def run(args: argparse.Namespace, stop: Optional[threading.Event] = None) -> RunResult:
    """Runs the bridge until `stop` is set or the duration passes."""
    stop = stop if stop is not None else threading.Event()
    try:
        model = load_model(args.urdf, args.package_path) if args.urdf else None
        network = robot_ipc.load_network_config(resolve_input_path(args.network_config))
    except (urdf_model.UrdfError, robot_ipc.NetworkConfigError, OSError) as error:
        _LOGGER.error("%s", error)
        return RunResult(EXIT_USAGE)
    if model is None:
        _LOGGER.warning("no --urdf given: the robot instances will not be drawn")

    recording = bridge_module.new_recording(args.app_id)
    layout = blueprint.build_blueprint(
        tracked_link=(
            model.root_link if model is not None and args.follow_robot else None
        )
    )
    options = rerun_sinks.SinkOptions(
        url=args.rerun_url,
        rrd_path=resolve_output_path(args.rrd_path) if args.rrd_path else "",
        grpc_port=args.grpc_port,
        web_port=args.web_port,
        open_browser=args.open_browser,
    )
    the_bridge = bridge_module.RerunBridge(recording, model)
    try:
        feed = bus_bridge.BusBridge(the_bridge, network, flush_period=args.flush_period)
    except (ValueError, robot_ipc.BusError, robot_ipc.NetworkConfigError) as error:
        _LOGGER.error("cannot subscribe to the bus: %s", error)
        return RunResult(EXIT_USAGE)
    try:
        _LOGGER.info(
            "%s",
            rerun_sinks.attach_sink(args.rerun_sink, recording, layout, options),
        )
    except (ValueError, RuntimeError, OSError) as error:
        _LOGGER.error("cannot start the Rerun sink '%s': %s", args.rerun_sink, error)
        feed.bus.close()
        return RunResult(EXIT_USAGE)
    the_bridge.log_static()
    _LOGGER.info(
        "subscribed to %s on %s",
        ", ".join(bus_bridge.SUBSCRIBED_TOPICS),
        ", ".join(feed.bus.subscriber_endpoints),
    )
    deadline = time.monotonic() + args.duration if args.duration > 0.0 else None
    feed.start()
    try:
        while not stop.wait(_POLL_PERIOD_S):
            if deadline is not None and time.monotonic() >= deadline:
                break
    finally:
        delivered = feed.stop()
        if delivered:
            recording.disconnect()
        else:
            _LOGGER.warning(
                "the Rerun sink is unreachable; exiting without waiting for it"
            )
    _LOGGER.info(
        "stopped: %s",
        ", ".join(
            f"{count} {topic}" for topic, count in the_bridge.statistics.handled.items()
        )
        or "no messages",
    )
    return RunResult(EXIT_OK, delivered)


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="humanoid_rerun_viewer: %(levelname)s: %(message)s",
    )
    if args.rerun_sink == "save" and not args.rrd_path:
        parser.error("--rerun_sink save needs --rrd_path")
    if args.flush_period <= 0.0:
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
        os._exit(result.exit_status)  # pylint: disable=protected-access
    return result.exit_status


if __name__ == "__main__":
    sys.exit(main())
