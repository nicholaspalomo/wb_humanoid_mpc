"""Tests for bus_listener.py: the network file and the receive-only side of the bus contract over loopback."""

import os
import tempfile
import time
import unittest
from unittest import mock

import bus_listener
import bus_test_publisher
import robot_ipc

TYPE_NAME = "humanoid_mpc_msgs.YamlDocument"
# Generous: a loaded CI machine may take a while to connect, and a passing test returns as soon as it receives.
RECEIVE_TIMEOUT_S = 10.0


class ParseNetworkConfigTest(unittest.TestCase):

    def test_nodes_keep_file_order_and_name_tcp_endpoints(self) -> None:
        nodes = bus_listener.parse_network_config(
            'nodes { name: "robot" host: "127.0.0.1" port: 5600 }\n'
            'nodes { name: "mpc" host: "192.168.1.20" port: 5610 }\n',
            "network.textproto",
        )
        self.assertEqual([node.name for node in nodes], ["robot", "mpc"])
        self.assertEqual(
            [node.connect_endpoint() for node in nodes],
            ["tcp://127.0.0.1:5600", "tcp://192.168.1.20:5610"],
        )

    def test_a_wildcard_host_is_reached_over_loopback(self) -> None:
        nodes = bus_listener.parse_network_config(
            'nodes { name: "robot" host: "0.0.0.0" port: 5600 }\n'
            'nodes { name: "mpc" host: "*" port: 5610 bind_host: "0.0.0.0" }\n',
            "network.textproto",
        )
        self.assertEqual(
            [node.connect_endpoint() for node in nodes],
            ["tcp://127.0.0.1:5600", "tcp://127.0.0.1:5610"],
        )

    def test_invalid_files_are_rejected_like_the_bus_rejects_them(self) -> None:
        # The tool reads the file with the bus library's loader: what the bus refuses, the tool refuses too, with the
        # same message (its line and column, or the node).
        cases = {
            "an unknown field": ('nodes { name: "robot" prot: 5600 }', ":1:"),
            "the old YAML file": (
                "nodes:\n  robot: {host: 127.0.0.1, port: 5600}",
                ":2:",
            ),
            "no node": ("", "nodes"),
            "a host missing": ('nodes { name: "robot" port: 5600 }', "(robot).host"),
            "a port out of range": (
                'nodes { name: "robot" host: "127.0.0.1" port: 70000 }',
                "(robot).port",
            ),
            "a shared endpoint": (
                'nodes { name: "robot" host: "127.0.0.1" port: 5600 }\n'
                'nodes { name: "mpc" host: "127.0.0.1" port: 5600 }',
                "(mpc)",
            ),
        }
        for case, (text, expected) in cases.items():
            with self.subTest(case=case):
                with self.assertRaises(bus_listener.NetworkConfigError) as raised:
                    bus_listener.parse_network_config(text, "network.textproto")
                message = str(raised.exception)
                self.assertTrue(message.startswith("network.textproto"), message)
                self.assertIn(expected, message)
                with self.assertRaises(robot_ipc.NetworkConfigError) as bus_raised:
                    robot_ipc.parse_network_config(text, "network.textproto")
                self.assertEqual(message, str(bus_raised.exception))


class LoadNetworkConfigTest(unittest.TestCase):

    def test_missing_and_malformed_files_raise_a_network_config_error(self) -> None:
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
            path = bus_test_publisher.write_network_file(
                directory, {"robot": 5600, "mpc": 5610}
            )
            nodes = bus_listener.load_network_config(path)
        self.assertEqual([node.port for node in nodes], [5600, 5610])


class ShippedNetworkFileTest(unittest.TestCase):

    def test_the_default_path_names_the_shipped_network_file_and_it_parses(
        self,
    ) -> None:
        # The test runs in the runfiles root, where config/ipc/network.textproto is a data dependency, as the tool runs
        # in the repository root.
        environment = {
            key: value
            for key, value in os.environ.items()
            if key not in ("BUILD_WORKING_DIRECTORY", "BUILD_WORKSPACE_DIRECTORY")
        }
        with mock.patch.dict(os.environ, environment, clear=True):
            path = bus_listener.resolve_input_path(bus_listener.DEFAULT_NETWORK_CONFIG)
        self.assertTrue(os.path.isfile(path), path)
        nodes = bus_listener.load_network_config(path)
        self.assertTrue(nodes)
        self.assertEqual(nodes, list(robot_ipc.localhost_network_config().nodes))


class ResolveInputPathTest(unittest.TestCase):

    def test_relative_paths_prefer_the_working_directory_then_the_workspace(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as working, tempfile.TemporaryDirectory() as workspace:
            os.makedirs(os.path.join(workspace, "config", "ipc"))
            in_workspace = os.path.join(workspace, "config", "ipc", "network.textproto")
            open(in_workspace, "w", encoding="utf-8").close()
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
                open(in_working, "w", encoding="utf-8").close()
                self.assertEqual(
                    bus_listener.resolve_input_path("network.textproto"), in_working
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

    def test_one_socket_receives_from_every_endpoint(self) -> None:
        robot_topic = bus_test_publisher.unique_topic("robot")
        mpc_topic = bus_test_publisher.unique_topic("mpc")
        with bus_test_publisher.FramePublisher() as robot, bus_test_publisher.FramePublisher() as mpc:
            robot.start([bus_test_publisher.frames(robot_topic, TYPE_NAME, b"r")])
            mpc.start([bus_test_publisher.frames(mpc_topic, TYPE_NAME, b"m")])
            endpoints = [
                f"tcp://127.0.0.1:{robot.port}",
                f"tcp://127.0.0.1:{mpc.port}",
                # A node that is not running: connecting to it is harmless.
                f"tcp://127.0.0.1:{bus_test_publisher.unused_port()}",
            ]
            seen = set()
            with bus_listener.BusListener(endpoints) as listener:
                deadline = time.monotonic() + RECEIVE_TIMEOUT_S
                while seen != {robot_topic, mpc_topic} and time.monotonic() < deadline:
                    received = listener.receive(timeout_s=0.1)
                    if received is not None and received.topic.startswith("test/"):
                        seen.add(received.topic)
        self.assertEqual(seen, {robot_topic, mpc_topic})

    def test_a_topic_subscription_drops_longer_topics_with_the_same_prefix(
        self,
    ) -> None:
        topic = bus_test_publisher.unique_topic("state")
        with bus_test_publisher.FramePublisher() as publisher:
            publisher.start(
                [
                    bus_test_publisher.frames(topic + "_estimate", TYPE_NAME, b"no"),
                    bus_test_publisher.frames(topic, TYPE_NAME, b"yes"),
                ]
            )
            with bus_listener.BusListener(
                [f"tcp://127.0.0.1:{publisher.port}"], topic=topic
            ) as listener:
                received = [
                    listener.receive(timeout_s=RECEIVE_TIMEOUT_S) for _ in range(5)
                ]
        self.assertTrue(all(message is not None for message in received))
        self.assertEqual({message.topic for message in received}, {topic})
        self.assertEqual({message.payload for message in received}, {b"yes"})
        self.assertEqual({message.type_name for message in received}, {TYPE_NAME})

    def test_messages_that_are_not_three_frames_are_counted_and_skipped(
        self,
    ) -> None:
        topic = bus_test_publisher.unique_topic("malformed")
        with bus_test_publisher.FramePublisher() as publisher:
            publisher.start(
                [
                    [topic.encode(), b"only two frames"],
                    bus_test_publisher.frames(topic, TYPE_NAME, b"ok"),
                ]
            )
            with bus_listener.BusListener(
                [f"tcp://127.0.0.1:{publisher.port}"], topic=topic
            ) as listener:
                received = [
                    listener.receive(timeout_s=RECEIVE_TIMEOUT_S) for _ in range(3)
                ]
                malformed = listener.malformed_messages
        self.assertEqual({message.payload for message in received}, {b"ok"})
        self.assertGreater(malformed, 0)

    def test_receive_returns_none_after_the_timeout(self) -> None:
        with bus_listener.BusListener(
            [f"tcp://127.0.0.1:{bus_test_publisher.unused_port()}"],
            topic=bus_test_publisher.unique_topic("silent"),
        ) as listener:
            start = time.monotonic()
            self.assertIsNone(listener.receive(timeout_s=0.2))
            self.assertGreaterEqual(time.monotonic() - start, 0.15)

    def test_needs_an_endpoint(self) -> None:
        with self.assertRaises(ValueError):
            bus_listener.BusListener([])


if __name__ == "__main__":
    unittest.main()
