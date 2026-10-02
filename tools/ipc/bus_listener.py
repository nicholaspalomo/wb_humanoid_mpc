"""A receive-only client of the IPC bus: reads the network file and connects one SUB socket to every node.

The bus (humanoid_nmpc/docs/distributed_runtime/README.md) has no broker: every publishing process binds one PUB
socket at the endpoint of its node in the network file, and a subscriber connects one SUB socket to EVERY endpoint of
that file and filters by topic. Each message is three frames: the topic, the full protobuf type name and the
serialized message.

The network file is read with the bus library's loader (robot_ipc.load_network_config), so the tool accepts exactly
the files the processes of the bus accept and reports their errors alike. The receiving half of the bus contract is
implemented here on pyzmq directly, so that the tool checks the documented framing rather than reusing robot_ipc.Bus.
It never publishes and never binds.
"""

import os
import time
from typing import List, NamedTuple, Optional, Sequence

import zmq

import robot_ipc

# Relative to the repository root (resolve_input_path), as the processes of the bus read it.
DEFAULT_NETWORK_CONFIG = os.path.join("config", "ipc", "network.textproto")

# The three frames of a bus message.
FRAME_COUNT = 3

# Messages ZeroMQ queues for this subscriber before it drops the newest; enough for a few seconds of the fastest
# stream (robot/mpc_observation, one per control cycle) while the tool formats a slow message.
DEFAULT_RECEIVE_HIGH_WATER_MARK = 10000

# A node of the network file: its name, host and port, and connect_endpoint(), where a SUB socket reaches it.
Node = robot_ipc.NodeEndpoint


class NetworkConfigError(ValueError):
    """The network file is missing or does not describe the nodes of the bus."""


def parse_network_config(text: str, source: str) -> List[Node]:
    """The nodes of the text of a network file, in file order; `source` names it in the error messages.

    Raises:
        NetworkConfigError: the file is not a valid network file; the message names its line and column, or the node.
    """
    try:
        return list(robot_ipc.parse_network_config(text, source).nodes)
    except robot_ipc.NetworkConfigError as error:
        raise NetworkConfigError(str(error)) from None


def load_network_config(path: str) -> List[Node]:
    """The nodes of the network file at `path`, in file order.

    Raises:
        NetworkConfigError: the file cannot be read or is not a valid network file.
    """
    try:
        with open(path, "r", encoding="utf-8") as stream:
            text = stream.read()
    except OSError as error:
        raise NetworkConfigError(
            f"cannot read the network file {path}: {error.strerror}"
        ) from error
    return parse_network_config(text, path)


def resolve_input_path(path: str) -> str:
    """Finds a file named on the command line.

    A relative path is looked up in the directory the tool was started from (BUILD_WORKING_DIRECTORY under `bazel
    run`, which changes into the runfiles tree) and then in the repository root (BUILD_WORKSPACE_DIRECTORY), so that
    the default network file is found from anywhere in the checkout. Returns the first existing candidate, or the
    first candidate when none exists, so that the error names the path the user meant.
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


class BusMessage(NamedTuple):
    """One message as it arrived: its three frames and the local receive time (time.monotonic, seconds)."""

    topic: str
    type_name: str
    payload: bytes
    receive_time: float


class BusListener:
    """One SUB socket connected to every endpoint of the bus, receiving one topic or all of them.

    ZeroMQ filters subscriptions by prefix, so the listener subscribes to the topic and then drops every message whose
    topic is merely longer (`robot/state` must not deliver `robot/state_estimate`). Messages that are not three frames
    are dropped and counted in `malformed_messages`.

    Usable as a context manager; the socket has LINGER 0, so closing never blocks.
    """

    def __init__(
        self,
        endpoints: Sequence[str],
        topic: Optional[str] = None,
        context: Optional[zmq.Context] = None,
        receive_high_water_mark: int = DEFAULT_RECEIVE_HIGH_WATER_MARK,
    ) -> None:
        if not endpoints:
            raise ValueError("a listener needs at least one endpoint")
        self.topic = topic
        self.endpoints = list(endpoints)
        self.malformed_messages = 0
        self._context = context if context is not None else zmq.Context.instance()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVHWM, receive_high_water_mark)
        self._socket.setsockopt(
            zmq.SUBSCRIBE, b"" if topic is None else topic.encode("utf-8")
        )
        for endpoint in self.endpoints:
            self._socket.connect(endpoint)
        self._poller = zmq.Poller()
        self._poller.register(self._socket, zmq.POLLIN)

    @classmethod
    def from_nodes(
        cls, nodes: Sequence[Node], topic: Optional[str] = None
    ) -> "BusListener":
        return cls([node.connect_endpoint() for node in nodes], topic=topic)

    def receive(self, timeout_s: float) -> Optional[BusMessage]:
        """The next message of the subscribed topic, or None when none arrived within `timeout_s` seconds."""
        deadline = time.monotonic() + max(timeout_s, 0.0)
        while True:
            remaining_ms = max(int((deadline - time.monotonic()) * 1000.0), 0)
            if not self._poller.poll(remaining_ms):
                return None
            frames = self._socket.recv_multipart(zmq.NOBLOCK)
            receive_time = time.monotonic()
            message = self._parse(frames, receive_time)
            if message is not None:
                return message
            if time.monotonic() >= deadline:
                return None

    def _parse(self, frames: List[bytes], receive_time: float) -> Optional[BusMessage]:
        if len(frames) != FRAME_COUNT:
            self.malformed_messages += 1
            return None
        try:
            topic = frames[0].decode("utf-8")
            type_name = frames[1].decode("utf-8")
        except UnicodeDecodeError:
            self.malformed_messages += 1
            return None
        if self.topic is not None and topic != self.topic:
            return None
        return BusMessage(
            topic=topic,
            type_name=type_name,
            payload=frames[2],
            receive_time=receive_time,
        )

    def close(self) -> None:
        self._socket.close()

    def __enter__(self) -> "BusListener":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
