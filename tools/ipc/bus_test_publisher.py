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

"""Test helpers: publishers of raw bus messages on the test process's own DDS domain, and its network file.

Every test process has a DDS domain and tag of its own (robot_ipc.isolated_test_network()), so that the tests Bazel
runs at once never see each other's samples; the network file this module writes names it, and the tool under test
joins it from that file. A publisher writes its messages repeatedly from a thread until stopped, which makes the tests
independent of DDS discovery (a listener of every topic discovers a topic from its writer, and gets only what is written
after that). raw_envelope_writer() writes envelopes a bus would never write (another wire protocol version).
"""

from collections.abc import Sequence
import os
import threading
import uuid

import cyclonedds.domain
import cyclonedds.pub
import cyclonedds.qos
import cyclonedds.topic

import robot_ipc

DEFAULT_PERIOD_S = 0.01
# The nodes of the test network, which the publishers publish as.
NODES = ("robot", "mpc")
# A reliable raw write waits this long for acknowledgments at most [ns].
_RAW_WRITE_BLOCKING_NS = 1_000_000_000
_RAW_WRITER_HISTORY_DEPTH = 64


def unique_topic(name: str) -> str:
    """A topic no other test of this process publishes, so that tests never see each other's messages."""
    return f"test/{name}_{uuid.uuid4().hex[:8]}"


def network() -> robot_ipc.NetworkConfig:
    """The network of this test process: its own domain and tag, NODES on 127.0.0.1."""
    return robot_ipc.isolated_test_network(list(NODES))


def message(topic: str, type_name: str, payload: bytes) -> tuple[str, str, bytes]:
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

    def _publish(self, messages: list[tuple[str, str, bytes]], period_s: float) -> None:
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


class RawEnvelopeWriter:
    """A DDS writer of robot_ipc::Envelope samples on one topic of the test domain, outside any bus.

    The domain must exist in this process already (a bus of the test network holds it), so that the writer joins it
    with the test network's configuration.

    Args:
        topic: the topic to write.
    """

    def __init__(self, topic: str) -> None:
        self._participant = cyclonedds.domain.DomainParticipant(network().domain_id)
        self._writer = cyclonedds.pub.DataWriter(
            self._participant,
            cyclonedds.topic.Topic(self._participant, topic, robot_ipc.Envelope),
            qos=cyclonedds.qos.Qos(
                cyclonedds.qos.Policy.Reliability.Reliable(
                    max_blocking_time=_RAW_WRITE_BLOCKING_NS
                ),
                cyclonedds.qos.Policy.History.KeepLast(_RAW_WRITER_HISTORY_DEPTH),
            ),
        )
        self._sequence = 0

    def write(self, protocol_version: int, type_name: str, payload: bytes) -> None:
        """Writes one envelope with the next sequence number."""
        self._sequence += 1
        self._writer.write(
            robot_ipc.Envelope(
                protocol_version,
                self._sequence,
                type_name,
                payload,
            )
        )

    def matched(self) -> bool:
        """Whether a reader has matched the writer."""
        return self._writer.get_publication_matched_status().current_count > 0


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
