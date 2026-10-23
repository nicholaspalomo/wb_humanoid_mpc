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

"""A receive-only client of the IPC bus: a robot_ipc.Bus that only subscribes, on the network file's DDS domain.

The bus (humanoid_nmpc/docs/distributed_runtime/README.md) has no broker: every topic is a DDS topic of the same name
whose samples are one robot_ipc::Envelope (the wire protocol's version, the writer's sequence number, the full
protobuf type name and the serialized message). The listener joins the domain of the network file as a subscriber, with
discovery sent to the host of every node, so it sees every topic whichever process publishes it; listening to every
topic, it discovers the topics from the DDS writers it sees.

The network file is read with the bus library's loader (robot_ipc.load_network_config), so the tool accepts exactly
the files the processes of the bus accept and reports their errors alike. It never publishes.
"""

import os
import queue
import time
from typing import NamedTuple

import robot_ipc

# Relative to the repository root (resolve_input_path), as the processes of the bus read it.
DEFAULT_NETWORK_CONFIG = os.path.join("config", "ipc", "network.textproto")


class NetworkConfigError(ValueError):
    """The network file is missing or does not describe the bus."""


def parse_network_config(text: str, source: str) -> robot_ipc.NetworkConfig:
    """The network of the text of a network file; `source` names it in the error messages.

    Raises:
        NetworkConfigError: the file is not a valid network file; the message names its line and column, or the entry.
    """
    try:
        return robot_ipc.parse_network_config(text, source)
    except robot_ipc.NetworkConfigError as error:
        raise NetworkConfigError(str(error)) from None


def load_network_config(path: str) -> robot_ipc.NetworkConfig:
    """The network of the network file at `path`.

    Args:
        path: The network file.

    Returns:
        Its network.

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
    the default network file is found from anywhere in the checkout.

    Args:
        path: The path as the command line gives it, absolute or relative.

    Returns:
        `path` itself when it is absolute, else the first candidate that exists, or the first candidate when none
        exists, so that the error names the path the user meant.
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
    """One message as it arrived: its topic, type name, payload and local receive time (time.monotonic, seconds)."""

    topic: str
    type_name: str
    payload: bytes
    receive_time: float


class BusListener:
    """A subscribe-only bus on the network's DDS domain, receiving one topic or all of them, queued for the caller.

    The bus's receive thread puts every message into a queue that receive() and receive_pending() take from, so the
    caller reads at its own pace. Samples of another wire protocol version, another layout or that are not a message
    of the bus are dropped and counted in `malformed_messages` (for a single topic: the bus's rejected samples).

    Usable as a context manager; closing stops the bus and deletes its DDS entities.

    Attributes:
        topic: the topic listened to; None for every topic.
        endpoints: where the listener listens, for messages: the DDS domain and its discovery peers.
    """

    def __init__(
        self, network: robot_ipc.NetworkConfig, topic: str | None = None
    ) -> None:
        """Joins the network's domain and subscribes to `topic` (every topic when None).

        Args:
            network: The network file's network.
            topic: The topic to receive, a DDS topic name; None for every topic.

        Raises:
            ValueError: the topic is not a DDS topic name.
            robot_ipc.BusError: DDS cannot join the domain.
        """
        self.topic = topic
        self.endpoints = [
            f"DDS domain {network.domain_id} (peers {', '.join(network.peer_hosts())})"
        ]
        self._messages: "queue.Queue[BusMessage]" = queue.Queue()
        self._bus = robot_ipc.Bus("", network)
        try:
            if topic is None:
                self._bus.subscribe_all_topics(self._on_any_topic)
            else:
                self._bus.subscribe_raw(topic, self._on_topic, robot_ipc.Delivery.ALL)
            self._bus.start()
        except BaseException:
            self._bus.close()
            raise

    def _on_topic(self, type_name: str, payload: bytes) -> None:
        self._on_any_topic(self.topic or "", type_name, payload)

    def _on_any_topic(self, topic: str, type_name: str, payload: bytes) -> None:
        self._messages.put(BusMessage(topic, type_name, payload, time.monotonic()))

    @property
    def malformed_messages(self) -> int:
        """The samples the bus dropped as not being a message of its wire protocol and layout."""
        return sum(
            statistics.rejected for statistics in self._bus.statistics().values()
        )

    def receive(self, timeout_s: float) -> BusMessage | None:
        """The next message, or None when none arrived within `timeout_s` seconds."""
        try:
            return self._messages.get(timeout=max(timeout_s, 0.0))
        except queue.Empty:
            return None

    def receive_pending(self) -> list[BusMessage]:
        """Every message that has already arrived, oldest first, without waiting.

        Returns:
            The messages the receive thread has queued now.
        """
        messages: list[BusMessage] = []
        while True:
            try:
                messages.append(self._messages.get_nowait())
            except queue.Empty:
                return messages

    def close(self) -> None:
        """Stops the bus and deletes its DDS entities."""
        self._bus.close()

    def __enter__(self) -> "BusListener":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
