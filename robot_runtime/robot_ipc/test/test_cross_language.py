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


"""The Python and the C++ bus speak one wire protocol over Cyclone DDS.

The Python bus publishes TestSample messages; the C++ echo binary (test/BusEchoMain.cpp, a typed C++ subscription)
parses each one and publishes it back with its origin set, and the Python bus parses the answer with a typed
subscription and sees its raw envelopes. Both processes generate their Cyclone DDS configuration from the same network
file, and generate the same text. A message of another type on the C++ side's topic is rejected by the C++ bus, which
proves it reads the envelope's type name. Policy-sized samples (0.75 MB) cross the processes whole over a link that
drops 2 % of its packets (Internal/Test/XmitLossiness), in datagrams of a Wi-Fi frame: the reliable_latest profile
repairs every lost fragment.
"""

from collections.abc import Callable
import functools
import os
import queue
import subprocess
import threading
import time
import unittest

from robot_ipc_test import test_event_pb2
from robot_ipc_test import test_sample_pb2

import robot_ipc

TIMEOUT = 60.0
TO_CPP = "test/to_cpp"
FROM_CPP = "test/from_cpp"
POLICY_TO_CPP = "test/policy_to_cpp"
POLICY_FROM_CPP = "test/policy_from_cpp"
# What the echo binary writes into every answer.
# LINT.IfChange(echo_origin)
ECHO_ORIGIN = "robot_ipc_bus_echo"
# LINT.ThenChange(//robot_runtime/robot_ipc/test/BusEchoMain.cpp:echo_origin)
SAMPLES = 50
POLICIES = 5
POLICY_BYTES = 768 * 1024
# A lossy Wi-Fi link: 2 % of the packets lost, one DDS fragment per frame.
LOSS_PER_MILLE = 20
WIFI_MESSAGE_SIZE_BYTES = 1400


def wait_for(condition: Callable[[], bool], timeout: float = TIMEOUT) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.001)
    return condition()


def make_sample(sequence: int) -> test_sample_pb2.TestSample:
    """A sample with a field of every kind, and contents that survive only an exact round trip."""
    return test_sample_pb2.TestSample(
        sequence=sequence,
        text=f"sample {sequence}: ünïcödé ✓",
        blob=bytes(range(256))[sequence % 7 :] + b"\x00\x00",
        value=-1.0e-300 * (sequence + 1),
        values=[0.1 * sequence, -2.5e17, 3.0],
        publisher=7,
    )


def make_policy(sequence: int) -> test_sample_pb2.TestSample:
    """A sample of a policy's size, whose bytes differ from one sequence to the next."""
    pattern = bytes((sequence + index) % 251 for index in range(4096))
    return test_sample_pb2.TestSample(
        sequence=sequence,
        blob=(pattern * (POLICY_BYTES // len(pattern) + 1))[:POLICY_BYTES],
    )


def network(
    loss_per_mille: int = 0, message_size: int = 0
) -> robot_ipc.NetworkConfig:
    isolated = robot_ipc.isolated_test_network(["python", "echo"])
    reliable_latest = robot_ipc.QosProfile.RELIABLE_LATEST
    return robot_ipc.NetworkConfig(
        nodes=isolated.nodes,
        domain_id=isolated.domain_id,
        domain_tag=isolated.domain_tag,
        topics=(
            robot_ipc.TopicQos(POLICY_TO_CPP, reliable_latest, ("python",)),
            robot_ipc.TopicQos(POLICY_FROM_CPP, reliable_latest, ("echo",)),
        ),
        transport=robot_ipc.TransportOptions(
            max_message_size_bytes=message_size,
            socket_receive_buffer_bytes=8 << 20,
            simulated_send_loss_per_mille=loss_per_mille,
        ),
    )


class EchoProcess:
    """The C++ echo binary, with its standard output read on a thread so that every read can time out."""

    def __init__(self, network_file: str) -> None:
        # pylint: disable-next=consider-using-with  # finish() and kill() end it; the test calls one of them.
        self._process = subprocess.Popen(
            [
                os.environ["BUS_ECHO"],
                f"--network_config={network_file}",
                "--node=echo",
                f"--echo={TO_CPP}={FROM_CPP},{POLICY_TO_CPP}={POLICY_FROM_CPP}",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        self._lines: "queue.Queue[str]" = queue.Queue()
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self) -> None:
        assert self._process.stdout is not None
        for line in self._process.stdout:
            self._lines.put(line.strip())

    def next_line(self) -> str:
        return self._lines.get(timeout=TIMEOUT)

    def finish(self) -> tuple[int, dict[str, dict[str, int]]]:
        """Closes its standard input, which stops it; returns its exit code and its STATS by input topic."""
        assert self._process.stdin is not None
        self._process.stdin.close()
        code = self._process.wait(timeout=TIMEOUT)
        stats = {}
        while not self._lines.empty():
            words = self._lines.get().split()
            if words and words[0] == "STATS":
                stats[words[1]] = {
                    key: int(value)
                    for key, value in (word.split("=") for word in words[2:])
                }
        return code, stats

    def kill(self) -> None:
        """Ends the process if it still runs, and closes its pipes."""
        if self._process.poll() is None:
            self._process.kill()
            self._process.wait()
        self._reader.join(timeout=TIMEOUT)
        for pipe in (self._process.stdin, self._process.stdout):
            if pipe is not None and not pipe.closed:
                pipe.close()


class CrossLanguageTestCase(unittest.TestCase):
    """The Python bus and the echo process, on NETWORK."""

    LOSS_PER_MILLE = 0
    MESSAGE_SIZE = 0

    def setUp(self) -> None:
        self.network = network(self.LOSS_PER_MILLE, self.MESSAGE_SIZE)
        network_file = os.path.join(
            os.environ["TEST_TMPDIR"], f"{self.id()}.textproto"
        )
        with open(network_file, "w", encoding="utf-8") as file:
            file.write(robot_ipc.format_network_config(self.network))
        self.bus = robot_ipc.Bus("python", self.network)
        self.addCleanup(self.bus.close)
        self.echo = EchoProcess(network_file)
        self.addCleanup(self.echo.kill)
        config = self.echo.next_line()
        self.assertTrue(config.startswith("CONFIG "), config)
        self.cpp_config = config[len("CONFIG ") :]
        self.assertEqual(self.echo.next_line(), "READY")

        self.lock = threading.Lock()
        self.echoes: dict[int, test_sample_pb2.TestSample] = {}
        self.policies: dict[int, test_sample_pb2.TestSample] = {}
        self.envelopes: list[tuple[str, str, bytes]] = []
        self.bus.subscribe(
            FROM_CPP, test_sample_pb2.TestSample, self._on_echo, "all"
        )
        self.bus.subscribe(
            POLICY_FROM_CPP, test_sample_pb2.TestSample, self._on_policy, "all"
        )
        self.bus.subscribe_all_topics(self._on_envelope)
        self.bus.start()
        self._wait_until_connected()

    def _on_echo(self, sample: test_sample_pb2.TestSample) -> None:
        with self.lock:
            self.echoes[sample.sequence] = sample

    def _on_policy(self, sample: test_sample_pb2.TestSample) -> None:
        with self.lock:
            self.policies[sample.sequence] = sample

    def _on_envelope(self, topic: str, type_name: str, payload: bytes) -> None:
        if topic == FROM_CPP:
            with self.lock:
                self.envelopes.append((topic, type_name, payload))

    def has_echo(self, sequence: int) -> bool:
        with self.lock:
            return sequence in self.echoes

    def has_policy(self, sequence: int) -> bool:
        with self.lock:
            return sequence in self.policies

    def _wait_until_connected(self) -> None:
        """Probes with sequence 0 until one comes back and the envelopes are seen: discovery has matched both ways."""

        def answered() -> bool:
            self.assertTrue(self.bus.publish(TO_CPP, make_sample(0)))
            time.sleep(0.01)
            with self.lock:
                return 0 in self.echoes and bool(self.envelopes)

        self.assertTrue(wait_for(answered), "the C++ echo never answered")


class CrossLanguageTest(CrossLanguageTestCase):

    def test_both_languages_generate_the_same_domain_configuration(
        self,
    ) -> None:
        self.assertEqual(
            self.cpp_config,
            robot_ipc.cyclone_domain_config(self.network, "echo"),
        )

    def test_python_and_cpp_carry_the_envelope_and_serialize_identically(
        self,
    ) -> None:
        sent = {
            sequence: make_sample(sequence)
            for sequence in range(1, SAMPLES + 1)
        }
        for sample in sent.values():
            self.assertTrue(self.bus.publish(TO_CPP, sample))
        self.assertTrue(wait_for(lambda: all(self.has_echo(s) for s in sent)))

        with self.lock:
            echoes = dict(self.echoes)
            envelopes = list(self.envelopes)
        for sequence, sample in sent.items():
            with self.subTest(sequence=sequence):
                echo = echoes[sequence]
                self.assertEqual(echo.origin, ECHO_ORIGIN)
                echo.ClearField("origin")
                self.assertEqual(echo, sample)
        # The C++ bus's envelopes: the full type name and the payload.
        for topic, type_name, payload in envelopes:
            self.assertEqual(topic, FROM_CPP)
            self.assertEqual(
                type_name, test_sample_pb2.TestSample.DESCRIPTOR.full_name
            )
            self.assertEqual(
                test_sample_pb2.TestSample.FromString(payload).origin,
                ECHO_ORIGIN,
            )

        code, stats = self.echo.finish()
        self.assertEqual(code, 0)
        self.assertEqual(stats[TO_CPP]["rejected"], 0)
        self.assertEqual(stats[TO_CPP]["received"], stats[TO_CPP]["delivered"])

    def test_cpp_rejects_another_type_on_its_typed_subscription(self) -> None:
        self.assertTrue(
            self.bus.publish(
                TO_CPP, test_event_pb2.TestEvent(name="not a sample")
            )
        )
        # One writer keeps its order on a reliable topic: once this echo is in, the event has reached the C++ bus.
        self.assertTrue(self.bus.publish(TO_CPP, make_sample(SAMPLES + 1)))
        self.assertTrue(wait_for(lambda: self.has_echo(SAMPLES + 1)))
        code, stats = self.echo.finish()
        self.assertEqual(code, 0)
        self.assertEqual(stats[TO_CPP]["rejected"], 1)
        self.assertEqual(
            stats[TO_CPP]["received"],
            stats[TO_CPP]["delivered"] + stats[TO_CPP]["rejected"],
        )


class LossyLinkTest(CrossLanguageTestCase):
    """Policy-sized samples across the processes over a link that loses packets."""

    LOSS_PER_MILLE = LOSS_PER_MILLE
    MESSAGE_SIZE = WIFI_MESSAGE_SIZE_BYTES

    def test_reliable_latest_carries_every_policy_whole_over_a_lossy_link(
        self,
    ) -> None:
        self.assertIn(
            f"<XmitLossiness>{LOSS_PER_MILLE}</XmitLossiness>", self.cpp_config
        )
        for sequence in range(1, POLICIES + 1):
            policy = make_policy(sequence)
            self.assertTrue(self.bus.publish(POLICY_TO_CPP, policy))
            # One at a time: KEEP_LAST 1 would replace a policy still being repaired with the next one.
            self.assertTrue(
                wait_for(functools.partial(self.has_policy, sequence)),
                f"policy {sequence} never came back",
            )
            with self.lock:
                echo = self.policies[sequence]
            self.assertEqual(echo.blob, policy.blob)
        code, stats = self.echo.finish()
        self.assertEqual(code, 0)
        self.assertEqual(stats[POLICY_TO_CPP]["delivered"], POLICIES)
        self.assertEqual(stats[POLICY_TO_CPP]["missed"], 0)


if __name__ == "__main__":
    unittest.main()
