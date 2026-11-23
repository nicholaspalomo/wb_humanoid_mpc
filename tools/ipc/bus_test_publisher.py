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

"""Test helpers: publishers of raw bus messages on the test process's own domain, and its network file.

Every test process has a domain and tag of its own (robot_ipc.isolated_test_network()), so that the tests Bazel runs at
once never see each other's samples; the network file this module writes names it, and the tool under test joins it
from that file. A publisher writes its messages repeatedly from a thread until stopped, which makes the tests
independent of when the sessions meet (a subscriber gets only what is put after its declaration reached the publisher).
RawEnvelopeWriter puts envelopes a bus would never put (another wire protocol version).
"""

from collections.abc import Sequence
import os
import threading
import time
import uuid

import zenoh

import robot_ipc
from robot_ipc import envelope
from robot_ipc import key_exprs
from robot_ipc import session_config

DEFAULT_PERIOD_S = 0.01
# The nodes of the test network, which the publishers publish as.
NODES = ("robot", "mpc")
# The writer id of the raw envelopes.
_RAW_WRITER_ID = 7


def unique_topic(name: str) -> str:
    """A topic no other test of this process publishes, so that tests never see each other's messages."""
    return f"test/{name}_{uuid.uuid4().hex[:8]}"


def network() -> robot_ipc.NetworkConfig:
    """The network of this test process: its own domain and tag, NODES on 127.0.0.1."""
    return robot_ipc.isolated_test_network(list(NODES))


def message(
    topic: str, type_name: str, payload: bytes
) -> tuple[str, str, bytes]:
    """A message of `topic` carrying `payload` as the type `type_name` names."""
    return (topic, type_name, payload)


class MessagePublisher:
    """A bus publishing as one node of the test network, writing its messages round-robin from a thread.

    Args:
        node: the node of NODES to publish as.
    """

    def __init__(self, node: str = "mpc") -> None:
        self._bus = robot_ipc.Bus(node, network())
        self._bus.start()
        self.published = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(
        self,
        messages: Sequence[tuple[str, str, bytes]],
        period_s: float = DEFAULT_PERIOD_S,
    ) -> "MessagePublisher":
        """Publishes `messages` in turn every `period_s` seconds, on a thread of its own.

        Args:
          messages: The messages to publish round-robin (message()).
          period_s: The time between two messages, in seconds.

        Returns:
          This publisher.
        """
        self._thread = threading.Thread(
            target=self._publish, args=(list(messages), period_s), daemon=True
        )
        self._thread.start()
        return self

    def _publish(
        self, messages: list[tuple[str, str, bytes]], period_s: float
    ) -> None:
        index = 0
        while not self._stop.is_set():
            topic, type_name, payload = messages[index % len(messages)]
            self._bus.publish_raw(topic, type_name, payload)
            self.published += 1
            index += 1
            self._stop.wait(period_s)

    def stop(self) -> None:
        """Stops publishing and closes the bus; waits up to 5 s for the thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._bus.close()

    def __enter__(self) -> "MessagePublisher":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()


def open_raw_session(test_network: robot_ipc.NetworkConfig) -> zenoh.Session:
    """A Zenoh session of its own in `test_network`'s domain, on the lowest free port, outside any bus.

    Args:
        test_network: The test network.

    Returns:
        The open session.

    Raises:
        RuntimeError: no port of the domain is free.
    """
    ports = session_config.tcp_port_range(test_network)
    for port in range(ports.first, ports.last + 1):
        if not session_config.is_tcp_port_free(robot_ipc.LOOPBACK_HOST, port):
            continue
        try:
            return zenoh.open(
                zenoh.Config.from_json5(
                    session_config.zenoh_session_config(test_network, "", port)
                )
            )
        except zenoh.ZError:
            continue
    raise RuntimeError(f"no free port of domain {test_network.domain_id}")


class RawEnvelopeWriter:
    """A Zenoh session of its own that puts envelopes on one topic of the test domain, outside any bus.

    Args:
        topic: the topic to put.

    Raises:
        RuntimeError: no port of the test domain is free for the session.
    """

    def __init__(self, topic: str) -> None:
        test_network = network()
        self._key_expr = key_exprs.data_key_expr(test_network, topic)
        self._session = open_raw_session(test_network)
        self._lock = threading.Lock()
        self._subscribers: set[str] = set()
        # The liveliness tokens of the topic's subscribers, of any bus and profile. A bus declares its subscriber
        # before its token, so once the token is here, so is the subscriber.
        self._tokens = self._session.liveliness().declare_subscriber(
            f"{key_exprs.ENDPOINT_KEY_EXPR_ROOT}/{test_network.domain_id}/"
            f"{key_exprs.DOMAIN_TAG_CHUNK_PREFIX}{test_network.domain_tag}/"
            f"{key_exprs.EndpointRole.SUBSCRIBER.value}/*/*/{topic}",
            self._on_token,
            history=True,
        )
        self._sequence = 0

    def _on_token(self, sample: zenoh.Sample) -> None:
        with self._lock:
            if sample.kind == zenoh.SampleKind.PUT:
                self._subscribers.add(str(sample.key_expr))
            else:
                self._subscribers.discard(str(sample.key_expr))

    def write(
        self, protocol_version: int, type_name: str, payload: bytes
    ) -> None:
        """Puts one envelope with the next sequence number."""
        self._sequence += 1
        self._session.put(
            self._key_expr,
            envelope.Envelope(
                protocol_version=protocol_version,
                writer_id=_RAW_WRITER_ID,
                sequence_number=self._sequence,
                source_timestamp_ns=time.time_ns(),
                type_name=type_name,
                payload=payload,
            ).encode(),
        )

    def matched(self) -> bool:
        """Whether a subscriber of the topic is up."""
        with self._lock:
            return bool(self._subscribers)

    def close(self) -> None:
        """Undeclares the token subscriber and closes the session, without which the process does not exit."""
        self._tokens.undeclare()
        self._session.close()

    def __enter__(self) -> "RawEnvelopeWriter":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def write_network_file(directory: str) -> str:
    """The test network's file in `directory`, as robot_ipc.format_network_config() writes it.

    Args:
      directory: Where to write the file, network.textproto.

    Returns:
      The path of the file.
    """
    path = os.path.join(directory, "network.textproto")
    with open(path, "w", encoding="utf-8") as stream:
        stream.write(robot_ipc.format_network_config(network()))
    return path
