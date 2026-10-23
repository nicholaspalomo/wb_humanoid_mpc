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

"""The serve_web sink: the web viewer answers HTTP and the recording is served over gRPC.

A process of its own, so that a hang of rerun-sdk's serve_grpc() (see rerun_sinks.py) shows as this test's timeout.
"""

import socket
import time
import unittest
import urllib.request

from humanoid_rerun_viewer import blueprint
from humanoid_rerun_viewer import bridge
from humanoid_rerun_viewer import rerun_sinks
import loopback_ports


def listen_on(test: unittest.TestCase, port: int) -> socket.socket:
    """A socket that listens on `port` of every interface until `test` ends, as another server would."""
    holder = socket.socket()
    test.addCleanup(holder.close)
    holder.bind(("0.0.0.0", port))
    holder.listen()
    return holder


def leave_in_time_wait(test: unittest.TestCase, port: int) -> None:
    """Leaves a closed connection on `port` in TIME_WAIT, as a bridge that just stopped does, with nothing listening.

    The listener sets SO_REUSEADDR, as Rerun's gRPC server does, and its side closes first, so its end of the
    connection, bound to `port`, lingers in TIME_WAIT. A bind without SO_REUSEADDR is then refused; the test checks
    that before it goes on.

    Args:
        test: The test case calling this helper.
        port: The port number to leave in TIME_WAIT.
    """
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", port))
        listener.listen()
        client = socket.create_connection(("127.0.0.1", port))
        accepted, _ = listener.accept()
        accepted.close()
        client.close()
    deadline = time.monotonic() + 5.0
    while True:
        with socket.socket() as plain:
            try:
                plain.bind(("0.0.0.0", port))
            except OSError:
                return
        if time.monotonic() > deadline:
            test.fail(f"port {port} did not enter TIME_WAIT")
        time.sleep(0.01)


class ServeWebSinkTest(unittest.TestCase):
    def setUp(self) -> None:
        self.recording = bridge.new_recording("test_serve_web_sink")
        self.addCleanup(self.recording.disconnect)
        self.layout = blueprint.build_blueprint()

    def attach(self, grpc_port: int, web_port: int) -> str:
        options = rerun_sinks.SinkOptions(grpc_port=grpc_port, web_port=web_port)
        return rerun_sinks.attach_sink(
            "serve_web", self.recording, self.layout, options
        )

    def test_serve_web_answers_http(self) -> None:
        ports = loopback_ports.unused_ports(2)
        options = rerun_sinks.SinkOptions(grpc_port=ports[0], web_port=ports[1])
        description = rerun_sinks.attach_sink(
            "serve_web", self.recording, self.layout, options
        )
        self.assertIn(str(options.web_port), description)
        self.assert_answers_http(options.web_port)

    def test_a_busy_grpc_port_is_refused(self) -> None:
        # rerun-sdk's serve_grpc() returns its URL even when its server cannot bind; every message would be dropped.
        ports = loopback_ports.unused_ports(2)
        grpc_port, web_port = ports[0], ports[1]
        listen_on(self, grpc_port)
        with self.assertRaises(OSError) as raised:
            self.attach(grpc_port, web_port)
        self.assertIn(str(grpc_port), str(raised.exception))
        self.assertIn("--grpc_port", str(raised.exception))

    def test_a_busy_web_port_is_refused(self) -> None:
        ports = loopback_ports.unused_ports(2)
        grpc_port, web_port = ports[0], ports[1]
        listen_on(self, web_port)
        with self.assertRaises(OSError) as raised:
            self.attach(grpc_port, web_port)
        self.assertIn(str(web_port), str(raised.exception))
        self.assertIn("--web_port", str(raised.exception))

    def test_a_port_in_time_wait_is_accepted(self) -> None:
        # A bridge started right after another stopped finds its old connections in TIME_WAIT on the gRPC port.
        ports = loopback_ports.unused_ports(2)
        grpc_port, web_port = ports[0], ports[1]
        leave_in_time_wait(self, grpc_port)
        self.assertIn(str(web_port), self.attach(grpc_port, web_port))
        self.assert_answers_http(web_port)

    def assert_answers_http(self, web_port: int) -> None:
        """Asserts that an HTTP GET request to localhost on web_port responds with HTTP 200.

        Args:
            web_port: The HTTP port to poll until it responds or times out.
        """
        deadline = time.monotonic() + 10.0
        while True:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{web_port}/", timeout=2.0
                ) as response:
                    self.assertEqual(response.status, 200)
                    break
            except OSError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.05)


if __name__ == "__main__":
    unittest.main()
