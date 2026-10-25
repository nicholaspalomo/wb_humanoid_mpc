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

"""Tests for bus_listener.py: the network file and the receive-only side of the bus over the test's DDS domain."""

from collections.abc import Callable
import os
import pathlib
import tempfile
import time
import unittest
from unittest import mock

import bus_listener
import bus_test_publisher
import robot_ipc

TYPE_NAME = "humanoid_mpc_msgs.FsmCommand"
# Generous: a loaded CI machine may take a while to discover, and a passing test returns as soon as it receives.
RECEIVE_TIMEOUT_S = 10.0


def wait_for(
    condition: Callable[[], bool], timeout_s: float = RECEIVE_TIMEOUT_S
) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.005)
    return condition()


class ParseNetworkConfigTest(unittest.TestCase):

    def test_reads_the_domain_and_the_nodes_in_file_order(self) -> None:
        network = bus_listener.parse_network_config(
            'domain_id: 3\nnodes { name: "robot" host: "127.0.0.1" }\n'
            'nodes { name: "mpc" host: "192.168.1.20" }\n',
            "network.textproto",
        )
        self.assertEqual(network.domain_id, 3)
        self.assertEqual(network.node_names(), ["robot", "mpc"])
        self.assertEqual(network.peer_hosts(), ["127.0.0.1", "192.168.1.20"])

    def test_invalid_files_are_rejected_like_the_bus_rejects_them(self) -> None:
        # The tool reads the file with the bus library's loader: what the bus refuses, the tool refuses too, with the
        # same message (its line and column, or the entry).
        cases = {
            "an unknown field": ('nodes { name: "robot" hots: "a" }', ":1:"),
            "a retired port": ('nodes { name: "robot" port: 5600 }', ":1:"),
            "no node": ("", "nodes"),
            "a host missing": ('nodes { name: "robot" }', "(robot).host"),
            "a domain out of range": (
                'nodes { name: "robot" host: "127.0.0.1" }\ndomain_id: 300',
                "domain_id",
            ),
        }
        for case, (text, expected) in cases.items():
            with self.subTest(case=case):
                with self.assertRaises(
                    bus_listener.NetworkConfigError
                ) as raised:
                    bus_listener.parse_network_config(text, "network.textproto")
                message = str(raised.exception)
                self.assertTrue(
                    message.startswith("network.textproto"), message
                )
                self.assertIn(expected, message)
                with self.assertRaises(
                    robot_ipc.NetworkConfigError
                ) as bus_raised:
                    robot_ipc.parse_network_config(text, "network.textproto")
                self.assertEqual(message, str(bus_raised.exception))


class LoadNetworkConfigTest(unittest.TestCase):

    def test_missing_and_malformed_files_raise_a_network_config_error(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            absent = os.path.join(directory, "absent.textproto")
            with self.assertRaises(bus_listener.NetworkConfigError) as raised:
                bus_listener.load_network_config(absent)
            self.assertIn(absent, str(raised.exception))
            malformed = os.path.join(directory, "malformed.textproto")
            with open(malformed, "w", encoding="utf-8") as stream:
                stream.write('nodes { name: "robot"\n')
            with self.assertRaises(bus_listener.NetworkConfigError) as raised:
                bus_listener.load_network_config(malformed)
            self.assertTrue(str(raised.exception).startswith(f"{malformed}:"))

    def test_reads_the_file_written_for_the_tests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = bus_test_publisher.write_network_file(directory)
            network = bus_listener.load_network_config(path)
        self.assertEqual(network, bus_test_publisher.network())


class ShippedNetworkFileTest(unittest.TestCase):

    def test_the_default_path_names_the_shipped_network_file_and_it_parses(
        self,
    ) -> None:
        # The test runs in the runfiles root, where config/ipc/network.textproto is a data dependency, as the tool runs
        # in the repository root.
        environment = {
            key: value
            for key, value in os.environ.items()
            if key
            not in ("BUILD_WORKING_DIRECTORY", "BUILD_WORKSPACE_DIRECTORY")
        }
        with mock.patch.dict(os.environ, environment, clear=True):
            path = bus_listener.resolve_input_path(
                bus_listener.DEFAULT_NETWORK_CONFIG
            )
        self.assertTrue(os.path.isfile(path), path)
        network = bus_listener.load_network_config(path)
        self.assertEqual(
            network.node_names(),
            ["robot", "mpc", "operator", "teleop", "config_push"],
        )
        self.assertEqual(network.domain_id, 0)


class ResolveInputPathTest(unittest.TestCase):

    def test_relative_paths_prefer_the_working_directory_then_the_workspace(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as working, tempfile.TemporaryDirectory() as workspace:
            os.makedirs(os.path.join(workspace, "config", "ipc"))
            in_workspace = os.path.join(
                workspace, "config", "ipc", "network.textproto"
            )
            pathlib.Path(in_workspace).touch()
            environment = {
                "BUILD_WORKING_DIRECTORY": working,
                "BUILD_WORKSPACE_DIRECTORY": workspace,
            }
            with mock.patch.dict(os.environ, environment):
                self.assertEqual(
                    bus_listener.resolve_input_path(
                        bus_listener.DEFAULT_NETWORK_CONFIG
                    ),
                    in_workspace,
                )
                in_working = os.path.join(working, "network.textproto")
                pathlib.Path(in_working).touch()
                self.assertEqual(
                    bus_listener.resolve_input_path("network.textproto"),
                    in_working,
                )
                # Nothing exists: the error names the path relative to where the tool was started.
                self.assertEqual(
                    bus_listener.resolve_input_path("absent.textproto"),
                    os.path.join(working, "absent.textproto"),
                )
            self.assertEqual(
                bus_listener.resolve_input_path(in_workspace), in_workspace
            )


class BusListenerTest(unittest.TestCase):

    def test_one_listener_receives_every_topic_of_every_publisher(self) -> None:
        robot_topic = bus_test_publisher.unique_topic("robot")
        mpc_topic = bus_test_publisher.unique_topic("mpc")
        with bus_test_publisher.MessagePublisher(
            "robot"
        ) as robot, bus_test_publisher.MessagePublisher("mpc") as mpc:
            robot.start(
                [bus_test_publisher.message(robot_topic, TYPE_NAME, b"r")]
            )
            mpc.start([bus_test_publisher.message(mpc_topic, TYPE_NAME, b"m")])
            with bus_listener.BusListener(
                bus_test_publisher.network()
            ) as listener:
                seen: dict[str, bytes] = {}

                def both_seen() -> bool:
                    received = listener.receive(0.05)
                    if received is not None:
                        seen[received.topic] = received.payload
                        self.assertEqual(received.type_name, TYPE_NAME)
                    return robot_topic in seen and mpc_topic in seen

                self.assertTrue(wait_for(both_seen))
                self.assertIn("DDS domain", listener.endpoints[0])
        self.assertEqual(seen[robot_topic], b"r")
        self.assertEqual(seen[mpc_topic], b"m")

    def test_a_topic_listener_receives_only_its_topic(self) -> None:
        # Every topic is a DDS topic of its own: a longer name is another topic, not a match of a prefix.
        topic = bus_test_publisher.unique_topic("state")
        with bus_test_publisher.MessagePublisher() as publisher:
            publisher.start(
                [
                    bus_test_publisher.message(
                        topic + "_estimate", TYPE_NAME, b"no"
                    ),
                    bus_test_publisher.message(topic, TYPE_NAME, b"yes"),
                ]
            )
            with bus_listener.BusListener(
                bus_test_publisher.network(), topic=topic
            ) as listener:
                received = listener.receive(RECEIVE_TIMEOUT_S)
                self.assertIsNotNone(received)
                assert received is not None  # For the type checker.
                time.sleep(0.1)
                pending = listener.receive_pending()
        for message in [received, *pending]:
            self.assertEqual(message.topic, topic)
            self.assertEqual(message.payload, b"yes")
        self.assertGreater(
            len(pending), 0, "receive_pending() hands over what has arrived"
        )

    def test_another_wire_protocol_version_is_counted_as_malformed(
        self,
    ) -> None:
        topic = bus_test_publisher.unique_topic("malformed")
        with bus_listener.BusListener(
            bus_test_publisher.network(), topic=topic
        ) as listener:
            writer = bus_test_publisher.RawEnvelopeWriter(topic)
            self.assertTrue(wait_for(writer.matched))
            writer.write(
                robot_ipc.WIRE_PROTOCOL_VERSION + 1,
                TYPE_NAME,
                b"from the future",
            )
            writer.write(robot_ipc.WIRE_PROTOCOL_VERSION, TYPE_NAME, b"ok")
            received = listener.receive(RECEIVE_TIMEOUT_S)
            self.assertIsNotNone(received)
            assert received is not None  # For the type checker.
            self.assertEqual(received.payload, b"ok")
            self.assertTrue(wait_for(lambda: listener.malformed_messages == 1))

    def test_a_silent_bus_times_out(self) -> None:
        with bus_listener.BusListener(
            bus_test_publisher.network(),
            topic=bus_test_publisher.unique_topic("silent"),
        ) as listener:
            start = time.monotonic()
            self.assertIsNone(listener.receive(0.2))
            self.assertGreaterEqual(time.monotonic() - start, 0.15)
            self.assertEqual(listener.receive_pending(), [])

    def test_a_topic_that_is_no_dds_name_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            bus_listener.BusListener(
                bus_test_publisher.network(), topic="robot-state"
            )


if __name__ == "__main__":
    unittest.main()
