"""Test helpers: a raw pyzmq PUB socket on an ephemeral loopback port, and a network file that names it.

The publisher frames its messages exactly as the bus does ([topic][full type name][payload], see
humanoid_nmpc/docs/distributed_runtime/README.md) without using any bus library, so the tests check the tool against
the documented contract rather than against another implementation of it. It publishes repeatedly from a thread until
stopped, which makes the tests independent of ZeroMQ's subscription latency (a SUB socket that has just connected
misses what was sent before its subscription arrived).
"""

import os
import threading
import uuid
from typing import Dict, List, Optional, Sequence

import zmq

LOOPBACK = "127.0.0.1"
DEFAULT_PERIOD_S = 0.01


def unique_topic(name: str) -> str:
    """A topic no other test publishes, so that tests sharing the loopback interface never see each other."""
    return f"test/{name}_{uuid.uuid4().hex[:8]}"


def frames(topic: str, type_name: str, payload: bytes) -> List[bytes]:
    return [topic.encode("utf-8"), type_name.encode("utf-8"), payload]


class FramePublisher:
    """A PUB socket bound to tcp://127.0.0.1:<ephemeral port> that publishes its messages round-robin."""

    def __init__(self, context: Optional[zmq.Context] = None) -> None:
        self._context = context if context is not None else zmq.Context.instance()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(f"tcp://{LOOPBACK}:*")
        endpoint = self._socket.getsockopt(zmq.LAST_ENDPOINT).decode("ascii")
        self.port = int(endpoint.rsplit(":", 1)[1])
        self.published = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(
        self, messages: Sequence[List[bytes]], period_s: float = DEFAULT_PERIOD_S
    ) -> "FramePublisher":
        """Publishes `messages` (each a list of frames) in turn every `period_s` seconds, on a thread that owns the
        socket from now on."""
        self._thread = threading.Thread(
            target=self._publish, args=(list(messages), period_s), daemon=True
        )
        self._thread.start()
        return self

    def _publish(self, messages: List[List[bytes]], period_s: float) -> None:
        index = 0
        try:
            while not self._stop.is_set():
                self._socket.send_multipart(messages[index % len(messages)])
                self.published += 1
                index += 1
                self._stop.wait(period_s)
        finally:
            self._socket.close()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        else:
            self._socket.close()

    def __enter__(self) -> "FramePublisher":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()


def write_network_file(directory: str, ports: Dict[str, int]) -> str:
    """A network file in `directory` with one loopback node per entry of `ports`; returns its path.

    The file is a textproto of robot_ipc_proto.NetworkConfig (robot_runtime/robot_ipc/proto/network_config.proto),
    written by hand rather than by the bus library, as a user would write it.
    """
    path = os.path.join(directory, "network.textproto")
    with open(path, "w", encoding="utf-8") as stream:
        for name, port in ports.items():
            stream.write(
                f'nodes {{ name: "{name}" host: "{LOOPBACK}" port: {port} }}\n'
            )
    return path


def unused_port() -> int:
    """A loopback port nothing listens on right now: bound once by ZeroMQ, then released."""
    socket = zmq.Context.instance().socket(zmq.PUB)
    try:
        socket.setsockopt(zmq.LINGER, 0)
        socket.bind(f"tcp://{LOOPBACK}:*")
        return int(
            socket.getsockopt(zmq.LAST_ENDPOINT).decode("ascii").rsplit(":", 1)[1]
        )
    finally:
        socket.close()
