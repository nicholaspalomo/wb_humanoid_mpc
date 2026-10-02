"""The serve_web sink: the web viewer answers HTTP and the recording is served over gRPC.

A process of its own, so that a hang of rerun-sdk's serve_grpc() (see rerun_sinks.py) shows as this test's timeout.
"""

import random
import socket
import time
import unittest
import urllib.request
from typing import Set, Tuple

from humanoid_rerun_viewer import blueprint
from humanoid_rerun_viewer import bridge
from humanoid_rerun_viewer import rerun_sinks


# The lowest port this test picks: above the bus (5600-5629), Jupyter (8888) and Rerun's defaults (9090, 9876).
_LOWEST_PORT = 20000


def _ephemeral_range() -> Tuple[int, int]:
    """The ports the kernel picks from for bind(0) and outgoing connections (ip_local_port_range)."""
    try:
        with open("/proc/sys/net/ipv4/ip_local_port_range", encoding="utf-8") as ports:
            low, high = (int(word) for word in ports.read().split())
            return low, high
    except (OSError, ValueError):
        return 32768, 60999


def unused_ports(count: int) -> Tuple[int, ...]:
    """`count` distinct loopback ports nothing is bound to, outside the kernel's ephemeral range.

    Rerun's servers bind the ports themselves, so they cannot be handed a bound socket. A port the kernel chose for a
    probe and got back could be chosen again, for another test or an outgoing connection, before Rerun binds it; one
    outside the ephemeral range is only ever bound by name, and one picked at random collides with another test's
    pick about never.
    """
    low, _ = _ephemeral_range()
    candidates = list(range(_LOWEST_PORT, max(_LOWEST_PORT + count, low)))
    random.SystemRandom().shuffle(candidates)
    chosen: Set[int] = set()
    for port in candidates:
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
        chosen.add(port)
        if len(chosen) == count:
            return tuple(chosen)
    raise RuntimeError(f"no {count} unused ports in [{_LOWEST_PORT}, {low})")


class ServeWebSinkTest(unittest.TestCase):
    def setUp(self) -> None:
        self.recording = bridge.new_recording("test_serve_web_sink")
        self.addCleanup(self.recording.disconnect)
        self.layout = blueprint.build_blueprint()

    def test_serve_web_answers_http(self) -> None:
        grpc_port, web_port = unused_ports(2)
        options = rerun_sinks.SinkOptions(grpc_port=grpc_port, web_port=web_port)
        description = rerun_sinks.attach_sink(
            "serve_web", self.recording, self.layout, options
        )
        self.assertIn(str(options.web_port), description)
        deadline = time.monotonic() + 10.0
        while True:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{options.web_port}/", timeout=2.0
                ) as response:
                    self.assertEqual(response.status, 200)
                    break
            except OSError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.05)


if __name__ == "__main__":
    unittest.main()
