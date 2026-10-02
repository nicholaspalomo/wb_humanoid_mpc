"""The Python and the C++ bus frame messages identically.

The Python bus publishes TestSample messages; the C++ echo binary (test/BusEchoMain.cpp, a typed C++ subscription)
parses each one and publishes it back with its origin set, and the Python bus parses the answer with a typed
subscription and sees its raw frames. A message of another type on the C++ side's topic is rejected by the C++ bus,
which proves it reads the second frame as the type name.
"""

import os
import queue
import subprocess
import threading
import time
import unittest
from typing import Callable, Dict, List, Tuple

from robot_ipc import EPHEMERAL_PORT, Bus, NetworkConfig, NodeEndpoint
from robot_ipc_test import test_event_pb2
from robot_ipc_test import test_sample_pb2

TIMEOUT = 30.0
TO_CPP = "test/to_cpp"
FROM_CPP = "test/from_cpp"
# What the echo binary writes into every answer.
# LINT.IfChange(echo_origin)
ECHO_ORIGIN = "robot_ipc_bus_echo"
# LINT.ThenChange(//robot_runtime/robot_ipc/test/BusEchoMain.cpp:echo_origin)
SAMPLES = 50


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


class EchoProcess:
    """The C++ echo binary, with its standard output read on a thread so that every read can time out."""

    def __init__(self, peer: str) -> None:
        self._process = subprocess.Popen(
            [
                os.environ["BUS_ECHO"],
                f"--peer={peer}",
                f"--input_topic={TO_CPP}",
                f"--output_topic={FROM_CPP}",
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

    def finish(self) -> Tuple[int, str]:
        """Closes its standard input, which stops it; returns its exit code and its STATS line."""
        assert self._process.stdin is not None
        self._process.stdin.close()
        code = self._process.wait(timeout=TIMEOUT)
        return code, self.next_line()

    def kill(self) -> None:
        if self._process.poll() is None:
            self._process.kill()
            self._process.wait()


def parse_stats(line: str) -> Dict[str, int]:
    """Maps "STATS received=1 delivered=1 rejected=0" to {"received": 1, "delivered": 1, "rejected": 0}."""
    words = line.split()
    assert words and words[0] == "STATS", line
    return {key: int(value) for key, value in (word.split("=") for word in words[1:])}


class CrossLanguageTest(unittest.TestCase):

    def setUp(self) -> None:
        self.bus = Bus(
            "python",
            NetworkConfig(nodes=(NodeEndpoint("python", "127.0.0.1", EPHEMERAL_PORT),)),
        )
        self.addCleanup(self.bus.close)
        self.echo = EchoProcess(self.bus.bound_endpoint)
        self.addCleanup(self.echo.kill)
        ready = self.echo.next_line().split()
        self.assertEqual(ready[0], "READY", ready)
        self.assertTrue(ready[1].startswith("tcp://127.0.0.1:"), ready)

        self.lock = threading.Lock()
        self.echoes: Dict[int, test_sample_pb2.TestSample] = {}
        self.frames: List[Tuple[str, str, bytes]] = []
        self.bus.connect(ready[1])
        self.bus.subscribe(FROM_CPP, test_sample_pb2.TestSample, self._on_echo, "all")
        self.bus.subscribe_all_topics(self._on_frames)
        self.bus.start()
        self._wait_until_connected()

    def _on_echo(self, sample: test_sample_pb2.TestSample) -> None:
        with self.lock:
            self.echoes[sample.sequence] = sample

    def _on_frames(self, topic: str, type_name: str, payload: bytes) -> None:
        if topic == FROM_CPP:
            with self.lock:
                self.frames.append((topic, type_name, payload))

    def _has_echo(self, sequence: int) -> bool:
        with self.lock:
            return sequence in self.echoes

    def _wait_until_connected(self) -> None:
        """Probes with sequence 0 until one comes back: both directions are connected and subscribed."""

        def answered() -> bool:
            self.assertTrue(self.bus.publish(TO_CPP, make_sample(0)))
            time.sleep(0.01)
            return self._has_echo(0)

        self.assertTrue(wait_for(answered), "the C++ echo never answered")

    def test_python_and_cpp_frame_and_serialize_identically(self) -> None:
        sent = {sequence: make_sample(sequence) for sequence in range(1, SAMPLES + 1)}
        for sample in sent.values():
            self.assertTrue(self.bus.publish(TO_CPP, sample))
        self.assertTrue(wait_for(lambda: all(self._has_echo(s) for s in sent)))

        with self.lock:
            echoes = dict(self.echoes)
            frames = list(self.frames)
        for sequence, sample in sent.items():
            with self.subTest(sequence=sequence):
                echo = echoes[sequence]
                self.assertEqual(echo.origin, ECHO_ORIGIN)
                echo.ClearField("origin")
                self.assertEqual(echo, sample)

        # The C++ bus's frames: the topic, the full type name, the payload.
        self.assertTrue(frames)
        for topic, type_name, payload in frames:
            self.assertEqual(topic, FROM_CPP)
            self.assertEqual(type_name, test_sample_pb2.TestSample.DESCRIPTOR.full_name)
            self.assertEqual(
                test_sample_pb2.TestSample.FromString(payload).origin, ECHO_ORIGIN
            )

        code, stats = self.echo.finish()
        self.assertEqual(code, 0)
        counts = parse_stats(stats)
        self.assertEqual(counts["rejected"], 0)
        self.assertEqual(counts["received"], counts["delivered"])

    def test_cpp_rejects_another_type_on_its_typed_subscription(self) -> None:
        self.assertTrue(
            self.bus.publish(TO_CPP, test_event_pb2.TestEvent(name="not a sample"))
        )
        # One connection keeps the order: once this sample's echo is in, the event has reached the C++ bus.
        self.assertTrue(self.bus.publish(TO_CPP, make_sample(SAMPLES + 1)))
        self.assertTrue(wait_for(lambda: self._has_echo(SAMPLES + 1)))

        code, stats = self.echo.finish()
        self.assertEqual(code, 0)
        counts = parse_stats(stats)
        self.assertEqual(counts["rejected"], 1)
        self.assertEqual(counts["received"], counts["delivered"] + counts["rejected"])


if __name__ == "__main__":
    unittest.main()
