"""Records the robot's MPC observations from the IPC bus into a CSV file, the input of export_rollouts.py.

    bazel run //humanoid_nmpc/humanoid_common_mpc_pyutils:mpc_observation_logger -- \\
        [--network_config config/ipc/network.textproto] [--output_dir .] [--duration 0]

It subscribes to robot/mpc_observation (humanoid_mpc_msgs.MpcObservation, every message in the order it arrived) and
writes one row per observation to mpc_observation_<YYYYmmdd_HHMMSS>.csv until Ctrl-C, SIGTERM or --duration ends it:

    time, mode, x0, x1, ..., u0, u1, ...

x<i> is component i of the MPC state and u<i> of the MPC input, as the robot sent them. The names are generic because
the layout of both depends on the formulation and the robot (the centroidal MPC's state starts with the normalized
momentum, the whole-body MPC's with the generalized coordinates); the task file's ModelSettings say which component is
which. The logger only subscribes, so it runs on any machine of the network file and never disturbs the robot.
"""

import argparse
import csv
import datetime
import logging
import os
import signal
import threading
import time
from typing import List, Optional, Sequence, TextIO

import robot_ipc
from humanoid_mpc_ipc import topics
from humanoid_mpc_msgs import mpc_observation_pb2

_LOGGER = logging.getLogger("mpc_observation_logger")

DEFAULT_NETWORK_CONFIG = os.path.join("config", "ipc", "network.textproto")
FILE_PREFIX = "mpc_observation_"
TIME_COLUMN = "time"
MODE_COLUMN = "mode"
STATE_COLUMN_PREFIX = "x"
INPUT_COLUMN_PREFIX = "u"
# How often the main thread checks for a stop request and the duration [s].
_POLL_PERIOD_S = 0.1


def column_names(state_dim: int, input_dim: int) -> List[str]:
    """The CSV header of observations with a state of `state_dim` and an input of `input_dim` components."""
    return (
        [TIME_COLUMN, MODE_COLUMN]
        + [f"{STATE_COLUMN_PREFIX}{i}" for i in range(state_dim)]
        + [f"{INPUT_COLUMN_PREFIX}{i}" for i in range(input_dim)]
    )


def observation_row(message: mpc_observation_pb2.MpcObservation) -> List[float]:
    """The CSV row of one observation: time, mode, state, input."""
    observation = message.observation
    return (
        [observation.time, float(observation.mode)]
        + list(observation.state)
        + list(observation.input)
    )


class ObservationCsvWriter:
    """Writes observations as CSV rows, the header with the first one; thread-safe.

    The state and input dimensions are those of the first observation. An observation of other dimensions (a robot
    process restarted with another formulation) is not written, and counted in `skipped`.
    """

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._writer = csv.writer(stream)
        self._lock = threading.Lock()
        self._dimensions: Optional[Sequence[int]] = None
        self.written = 0
        self.skipped = 0

    def write(self, message: mpc_observation_pb2.MpcObservation) -> bool:
        """Writes the row of `message`; False when its dimensions differ from the first observation's."""
        dimensions = (len(message.observation.state), len(message.observation.input))
        with self._lock:
            if self._dimensions is None:
                self._dimensions = dimensions
                self._writer.writerow(column_names(*dimensions))
            elif dimensions != self._dimensions:
                self.skipped += 1
                return False
            self._writer.writerow(observation_row(message))
            self.written += 1
            return True

    def flush(self) -> None:
        with self._lock:
            self._stream.flush()


def log_file_name(now: datetime.datetime) -> str:
    """The name of the CSV file of a recording started at `now`, the pattern export_rollouts.py looks for."""
    return f"{FILE_PREFIX}{now.strftime('%Y%m%d_%H%M%S')}.csv"


def subscribe(bus: robot_ipc.Bus, writer: ObservationCsvWriter) -> None:
    """Hands every observation of the bus to `writer`. Call before bus.start()."""
    bus.subscribe(
        topics.ROBOT_MPC_OBSERVATION,
        mpc_observation_pb2.MpcObservation,
        writer.write,
        delivery=robot_ipc.Delivery.ALL,
    )


def _resolve(path: str) -> str:
    """A path of the command line: as is when absolute, else relative to the start directory of `bazel run`."""
    if os.path.isabs(path):
        return path
    start_directory = os.environ.get("BUILD_WORKING_DIRECTORY", os.getcwd())
    candidate = os.path.join(start_directory, path)
    workspace = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if not os.path.exists(candidate) and workspace:
        return os.path.join(workspace, path)
    return candidate


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--network_config",
        default=DEFAULT_NETWORK_CONFIG,
        help="the network file of the bus (default: %(default)s)",
    )
    parser.add_argument(
        "--output_dir",
        default=".",
        help="where the CSV file is written (default: the start directory)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="seconds to record; 0 records until Ctrl-C or SIGTERM (default: %(default)s)",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace, stop: Optional[threading.Event] = None) -> str:
    """Records until `stop` is set or the duration passes; returns the path of the CSV file."""
    stop = stop if stop is not None else threading.Event()
    network = robot_ipc.load_network_config(_resolve(args.network_config))
    output_dir = _resolve(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, log_file_name(datetime.datetime.now()))
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = ObservationCsvWriter(stream)
        # Not `with Bus(...)`: entering it starts the bus, and a running bus takes no new subscription.
        bus = robot_ipc.Bus("", network)
        try:
            subscribe(bus, writer)
            bus.start()
            _LOGGER.info(
                "recording %s from %s into %s",
                topics.ROBOT_MPC_OBSERVATION,
                ", ".join(bus.subscriber_endpoints),
                path,
            )
            deadline = time.monotonic() + args.duration if args.duration > 0.0 else None
            while not stop.wait(_POLL_PERIOD_S):
                writer.flush()
                if deadline is not None and time.monotonic() >= deadline:
                    break
        finally:
            bus.close()
        writer.flush()
    _LOGGER.info(
        "wrote %d observations to %s (%d of other dimensions skipped)",
        writer.written,
        path,
        writer.skipped,
    )
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    args = parse_args(argv)
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    try:
        run(args, stop)
    except (robot_ipc.NetworkConfigError, robot_ipc.BusError, OSError) as error:
        _LOGGER.error("%s", error)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
