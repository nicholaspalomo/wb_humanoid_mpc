"""The Python network file loader: the properties testNetworkConfig.cpp checks for the C++ one.

A malformed textproto is an error naming its line and column, every invalid node is an error naming the node,
format_network_config() and parse_network_config() round-trip, and the shipped file is the network
localhost_network_config() builds.
"""

import os
import tempfile
import unittest

import robot_ipc
from robot_ipc import (
    EPHEMERAL_PORT,
    MAX_PORT,
    NetworkConfig,
    NetworkConfigError,
    NodeEndpoint,
)

# Relative to the test's runfiles directory (data dependency //config/ipc:network.textproto).
SHIPPED_NETWORK_FILE = "config/ipc/network.textproto"
SOURCE = "test.textproto"


def parse(text: str) -> NetworkConfig:
    return robot_ipc.parse_network_config(text, SOURCE)


class ParseTest(unittest.TestCase):

    def test_parses_the_nodes_in_file_order(self) -> None:
        config = parse(
            """
# A comment.
nodes { name: "robot" host: "127.0.0.1" port: 5600 }
nodes {
  name: "mpc"
  host: "192.168.1.20"
  port: 5610
  bind_host: "0.0.0.0"  # every interface
}
"""
        )
        self.assertEqual(config.node_names(), ["robot", "mpc"])
        robot = config.find("robot")
        self.assertEqual(robot, NodeEndpoint("robot", "127.0.0.1", 5600))
        self.assertEqual(robot.bind_endpoint(), "tcp://127.0.0.1:5600")
        self.assertEqual(robot.connect_endpoint(), "tcp://127.0.0.1:5600")
        mpc = config.find("mpc")
        self.assertIsNotNone(mpc)
        # bind_host decides where the node binds; host stays where the others connect.
        self.assertEqual(mpc.bind_endpoint(), "tcp://*:5610")
        self.assertEqual(mpc.connect_endpoint(), "tcp://192.168.1.20:5610")
        self.assertIsNone(config.find("operator"))

    def test_shipped_file_is_the_localhost_network(self) -> None:
        shipped = robot_ipc.load_network_config(SHIPPED_NETWORK_FILE)
        self.assertEqual(shipped, robot_ipc.localhost_network_config())
        robot_ipc.validate_network_config(shipped)

    def test_wildcard_host_binds_every_interface_and_is_reached_over_loopback(
        self,
    ) -> None:
        config = parse(
            """
nodes { name: "robot" host: "0.0.0.0" port: 5600 }
nodes { name: "mpc" host: "*" port: 5610 }
"""
        )
        for node in config.nodes:
            with self.subTest(node=node.name):
                self.assertTrue(robot_ipc.is_wildcard_host(node.host))
                self.assertTrue(node.bind_endpoint().startswith("tcp://*:"))
                self.assertTrue(node.connect_endpoint().startswith("tcp://127.0.0.1:"))

    def test_ephemeral_port_is_for_networks_built_in_code(self) -> None:
        config = NetworkConfig(
            nodes=(NodeEndpoint("test", "127.0.0.1", EPHEMERAL_PORT),)
        )
        robot_ipc.validate_network_config(config)
        self.assertEqual(config.nodes[0].bind_endpoint(), "tcp://127.0.0.1:*")
        # In a file, port 0 is a port left out: the other processes could not connect to it.
        with self.assertRaisesRegex(NetworkConfigError, r"nodes\[0\] \(test\)\.port"):
            parse('nodes { name: "test" host: "127.0.0.1" port: 0 }')

    def test_missing_file_raises_file_not_found(self) -> None:
        with self.assertRaises(FileNotFoundError):
            robot_ipc.load_network_config("does/not/exist.textproto")

    def test_errors_of_a_loaded_file_name_its_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "network.textproto")
            with open(path, "w", encoding="utf-8") as file:
                file.write('nodes { name: "robot" prot: 5600 }\n')
            with self.assertRaises(NetworkConfigError) as raised:
                robot_ipc.load_network_config(path)
        self.assertTrue(str(raised.exception).startswith(f"{path}:1:"))

    def test_formatted_networks_parse_back_unchanged(self) -> None:
        with_bind_host = NetworkConfig(
            nodes=(
                NodeEndpoint("robot", "192.168.1.10", 5600, bind_host="0.0.0.0"),
                NodeEndpoint("mpc-laptop_2", "laptop.local", 65535),
                NodeEndpoint("operator", "*", 1),
            )
        )
        for config in (robot_ipc.localhost_network_config(), with_bind_host):
            text = robot_ipc.format_network_config(config)
            with self.subTest(text=text):
                # One line per node.
                self.assertEqual(len(text.splitlines()), len(config.nodes))
                self.assertEqual(parse(text), config)


# (description, file, the line its error must name, a word the message must contain). The protobuf parsers of C++
# and Python word their errors differently, so the word is only set where both name the same thing. The same table as
# testNetworkConfig.cpp.
# LINT.IfChange(malformed_files)
MALFORMED_FILES = [
    (
        "an unknown field of a node",
        'nodes { name: "robot" host: "127.0.0.1" port: 5600 }\nnodes { name: "mpc" prot: 5610 }',
        2,
        "prot",
    ),
    (
        "an unknown top-level field",
        '# The nodes.\nhosts { name: "robot" }',
        2,
        "hosts",
    ),
    ("a quoted port", 'nodes {\n  name: "robot"\n  port: "5600"\n}', 3, ""),
    ("a fractional port", 'nodes {\n  name: "robot"\n  port: 5600.5\n}', 3, ""),
    ("a negative port", 'nodes {\n  name: "robot"\n  port: -1\n}', 3, ""),
    (
        "a port beyond 32 bits",
        'nodes {\n  name: "robot"\n  port: 4294967296\n}',
        3,
        "",
    ),
    ("a boolean port", 'nodes {\n  name: "robot"\n  port: true\n}', 3, ""),
    ("a port that is a word", 'nodes {\n  name: "robot"\n  port: fifty\n}', 3, ""),
    (
        "a host given twice",
        'nodes {\n  name: "robot"\n  host: "a"\n  host: "b"\n}',
        4,
        "host",
    ),
    ("an unquoted host", 'nodes {\n  name: "robot"\n  host: 127.0.0.1\n}', 3, ""),
    ("a missing colon", 'nodes { name: "robot" }\nnodes { name "mpc" }', 2, ""),
    (
        "an unclosed node",
        'nodes { name: "robot" host: "127.0.0.1" port: 5600',
        1,
        "",
    ),
    (
        "the old YAML network file",
        "nodes:\n  robot: {host: 127.0.0.1, port: 5600}",
        2,
        "",
    ),
]
# LINT.ThenChange(//robot_runtime/robot_ipc/test/testNetworkConfig.cpp:malformed_files)


# (description, file, the key the error must name). The same table as testNetworkConfig.cpp.
# LINT.IfChange(invalid_files)
INVALID_FILES = [
    ("an empty file", "", "nodes"),
    ("comments only", "# no node\n", "nodes"),
    ("a name missing", 'nodes { host: "127.0.0.1" port: 5600 }', "nodes[0].name"),
    ("a host missing", 'nodes { name: "robot" port: 5600 }', "nodes[0] (robot).host"),
    (
        "a port missing",
        'nodes { name: "robot" host: "127.0.0.1" }',
        "nodes[0] (robot).port",
    ),
    (
        "port 0",
        'nodes { name: "robot" host: "127.0.0.1" port: 0 }',
        "nodes[0] (robot).port",
    ),
    (
        "a port out of range",
        'nodes { name: "robot" host: "127.0.0.1" port: 70000 }',
        "nodes[0] (robot).port",
    ),
    (
        "a host that is an endpoint",
        'nodes { name: "robot" host: "tcp://127.0.0.1" port: 5600 }',
        "nodes[0] (robot).host",
    ),
    (
        "a host with a port",
        'nodes { name: "robot" host: "127.0.0.1:5600" port: 5600 }',
        "nodes[0] (robot).host",
    ),
    (
        "a host with a space",
        'nodes { name: "robot" host: "127.0.0.1 " port: 5600 }',
        "nodes[0] (robot).host",
    ),
    (
        "a bind_host that is an endpoint",
        'nodes { name: "robot" host: "127.0.0.1" port: 5600 bind_host: "tcp://*" }',
        "nodes[0] (robot).bind_host",
    ),
    (
        "an invalid node name",
        'nodes { name: "ro bot" host: "127.0.0.1" port: 5600 }',
        "nodes[0] (ro bot).name",
    ),
    (
        "a name used twice",
        'nodes { name: "robot" host: "127.0.0.1" port: 5600 }\n'
        'nodes { name: "robot" host: "127.0.0.1" port: 5601 }',
        "nodes[1] (robot).name",
    ),
    (
        "two nodes on one endpoint",
        'nodes { name: "robot" host: "127.0.0.1" port: 5600 }\n'
        'nodes { name: "mpc" host: "127.0.0.1" port: 5600 }',
        "nodes[1] (mpc)",
    ),
    (
        "the second node invalid",
        'nodes { name: "robot" host: "127.0.0.1" port: 5600 }\n'
        'nodes { name: "mpc" host: "127.0.0.1" port: 65536 }',
        "nodes[1] (mpc).port",
    ),
]
# LINT.ThenChange(//robot_runtime/robot_ipc/test/testNetworkConfig.cpp:invalid_files)


class MalformedFileTest(unittest.TestCase):

    def test_every_malformed_file_is_rejected_with_its_line_and_column(self) -> None:
        for description, text, line, fragment in MALFORMED_FILES:
            with self.subTest(description=description):
                with self.assertRaises(NetworkConfigError) as raised:
                    parse(text)
                message = str(raised.exception)
                # "<source>:<line>:<column>: <problem>"
                parts = message.split(":", 3)
                self.assertEqual(len(parts), 4, message)
                self.assertEqual(parts[0], SOURCE, message)
                self.assertEqual(parts[1], str(line), message)
                self.assertGreaterEqual(int(parts[2]), 1, message)
                self.assertIn(fragment, parts[3], message)


class InvalidFileTest(unittest.TestCase):

    def test_every_invalid_file_is_rejected_with_the_offending_node(self) -> None:
        for description, text, key in INVALID_FILES:
            with self.subTest(description=description):
                with self.assertRaises(NetworkConfigError) as raised:
                    parse(text)
                message = str(raised.exception)
                self.assertTrue(message.startswith(f"{SOURCE}: {key}"), message)


class ValidateTest(unittest.TestCase):

    def test_validation_of_a_code_built_network_names_the_node(self) -> None:
        with self.assertRaisesRegex(NetworkConfigError, r"^nodes\[1\] \(robot\)\.name"):
            robot_ipc.validate_network_config(
                NetworkConfig(
                    nodes=(
                        NodeEndpoint("robot", "127.0.0.1", 5600),
                        NodeEndpoint("robot", "127.0.0.1", 5601),
                    )
                )
            )
        with self.assertRaises(NetworkConfigError):
            robot_ipc.validate_network_config(NetworkConfig(nodes=()))
        for port in (MAX_PORT + 1, -1):
            with self.subTest(port=port):
                with self.assertRaisesRegex(
                    NetworkConfigError, r"^nodes\[0\] \(robot\)\.port"
                ):
                    robot_ipc.validate_network_config(
                        NetworkConfig(nodes=(NodeEndpoint("robot", "127.0.0.1", port),))
                    )
        with self.assertRaisesRegex(
            NetworkConfigError, r"^nodes\[0\] \(robot\)\.bind_host"
        ):
            robot_ipc.validate_network_config(
                NetworkConfig(
                    nodes=(
                        NodeEndpoint("robot", "127.0.0.1", 5600, bind_host="tcp://*"),
                    )
                )
            )


class DeliveryTest(unittest.TestCase):

    def test_names_round_trip_and_unknown_names_list_the_valid_ones(self) -> None:
        for delivery in robot_ipc.Delivery:
            self.assertIs(robot_ipc.parse_delivery(delivery.value), delivery)
            self.assertIs(robot_ipc.parse_delivery(delivery), delivery)
        with self.assertRaises(ValueError) as raised:
            robot_ipc.parse_delivery("newest")
        self.assertIn("latest", str(raised.exception))
        self.assertIn("all", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
