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

"""Test helpers: a raw pyzmq PUB socket on an ephemeral loopback port, and a network file that names it.

The publisher frames its messages exactly as the bus does ([topic][full type name][payload], see
humanoid_nmpc/docs/distributed_runtime/README.md) without using any bus library, so the tests check the tool against
the documented contract rather than against another implementation of it. It publishes repeatedly from a thread until
stopped, which makes the tests independent of ZeroMQ's subscription latency (a SUB socket that has just connected
misses what was sent before its subscription arrived).
"""

from collections.abc import Sequence
import os
import threading
import uuid

import zmq

LOOPBACK = "127.0.0.1"
DEFAULT_PERIOD_S = 0.01


def unique_topic(name: str) -> str:
    """A topic no other test publishes, so that tests sharing the loopback interface never see each other."""
    return f"test/{name}_{uuid.uuid4().hex[:8]}"


def frames(topic: str, type_name: str, payload: bytes) -> list[bytes]:
    """The three frames of a bus message: its topic, its full type name and its payload."""
    return [topic.encode("utf-8"), type_name.encode("utf-8"), payload]


class FramePublisher:
    """A PUB socket bound to tcp://127.0.0.1:<ephemeral port> that publishes its messages round-robin."""

    def __init__(self, context: zmq.Context | None = None) -> None:
        self._context = context if context is not None else zmq.Context.instance()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(f"tcp://{LOOPBACK}:*")
        endpoint = self._socket.getsockopt(zmq.LAST_ENDPOINT).decode("ascii")
        self.port = int(endpoint.rsplit(":", 1)[1])
        self.published = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(
        self, messages: Sequence[list[bytes]], period_s: float = DEFAULT_PERIOD_S
    ) -> "FramePublisher":
        """Publishes `messages` in turn every `period_s` seconds, on a thread that owns the socket from now on.

        Args:
          messages: The messages to publish round-robin, each a list of frames (frames()).
          period_s: The time between two messages, in seconds.

        Returns:
          This publisher.
        """
        self._thread = threading.Thread(
            target=self._publish, args=(list(messages), period_s), daemon=True
        )
        self._thread.start()
        return self

    def _publish(self, messages: list[list[bytes]], period_s: float) -> None:
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
        """Stops publishing and closes the socket; waits up to 5 s for the thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        else:
            self._socket.close()

    def __enter__(self) -> "FramePublisher":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()


def write_network_file(directory: str, ports: dict[str, int]) -> str:
    """A network file in `directory` with one loopback node per entry of `ports`.

    The file is a textproto of robot_ipc_proto.NetworkConfig (robot_runtime/robot_ipc/proto/network_config.proto),
    written by hand rather than by the bus library, as a user would write it.

    Args:
      directory: Where to write the file, network.textproto.
      ports: The port of each node, by node name.

    Returns:
      The path of the file.
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
