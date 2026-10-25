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


"""The Python network file loader and domain configuration: the properties testNetworkConfig.cpp checks in C++.

A malformed textproto is an error naming its line and column, every invalid node, topic or field is an error naming it,
format_network_config() and parse_network_config() round-trip, the shipped files parse, and the Cyclone DDS
configuration and ports follow from the network.
"""

import unittest

import robot_ipc
from robot_ipc import domain_config

# Relative to the test's runfiles directory (data dependencies of //config/ipc).
SHIPPED_NETWORK_FILE = "config/ipc/network.textproto"
TWO_MACHINE_NETWORK_FILE = "config/ipc/two_machine.example.textproto"
SOURCE = "test.textproto"
MAX_UDP_PORT = 65535


def parse(text: str) -> robot_ipc.NetworkConfig:
    return robot_ipc.parse_network_config(text, SOURCE)


def node(name: str, host: str = "127.0.0.1") -> robot_ipc.NodeEndpoint:
    return robot_ipc.NodeEndpoint(name, host)


class ParseTest(unittest.TestCase):

    def test_parses_the_nodes_topics_and_transport(self) -> None:
        config = parse(
            """
# A comment.
domain_id: 7
domain_tag: "lab.robot-2"
nodes { name: "robot" host: "192.168.1.10" }
nodes {
  name: "mpc"
  host: "laptop.local"  # a host name
}
topics { name: "mpc/policy" qos_profile: "reliable_latest" publishers: "mpc" }
topics { name: "robot/state" qos_profile: "latest" publishers: ["robot"] }
default_qos_profile: "latest"
transport { max_message_size_bytes: 1400 lease_duration_ms: 2500 simulated_send_loss_per_mille: 5 }
"""
        )
        self.assertEqual(config.node_names(), ["robot", "mpc"])
        self.assertEqual(config.domain_id, 7)
        self.assertEqual(config.domain_tag, "lab.robot-2")
        self.assertEqual(config.find("mpc"), node("mpc", "laptop.local"))
        self.assertIsNone(config.find("operator"))
        self.assertIs(
            config.qos_profile_of("mpc/policy"),
            robot_ipc.QosProfile.RELIABLE_LATEST,
        )
        self.assertIs(
            config.qos_profile_of("robot/state"), robot_ipc.QosProfile.LATEST
        )
        self.assertIs(
            config.qos_profile_of("unlisted/topic"), robot_ipc.QosProfile.LATEST
        )
        self.assertEqual(config.topics_published_by("mpc"), ["mpc/policy"])
        self.assertEqual(config.peer_hosts(), ["192.168.1.10", "laptop.local"])
        self.assertEqual(config.transport.max_message_size_bytes, 1400)
        self.assertEqual(config.transport.lease_duration_ms, 2500)
        self.assertEqual(config.transport.simulated_send_loss_per_mille, 5)

    def test_the_shipped_files_parse_and_the_example_is_the_shipped_network_on_two_machines(
        self,
    ) -> None:
        shipped = robot_ipc.load_network_config(SHIPPED_NETWORK_FILE)
        example = robot_ipc.load_network_config(TWO_MACHINE_NETWORK_FILE)
        self.assertTrue(
            all(
                robot_ipc.network_config.is_loopback_host(host)
                for host in shipped.peer_hosts()
            )
        )
        self.assertEqual(example.node_names(), shipped.node_names())
        self.assertEqual(example.topics, shipped.topics)
        self.assertEqual(len(example.peer_hosts()), 2)
        self.assertIs(
            shipped.qos_profile_of("mpc/policy"),
            robot_ipc.QosProfile.RELIABLE_LATEST,
        )

    def test_missing_file_raises_file_not_found(self) -> None:
        with self.assertRaises(FileNotFoundError):
            robot_ipc.load_network_config("does/not/exist.textproto")

    def test_formatted_networks_parse_back_unchanged(self) -> None:
        custom = robot_ipc.NetworkConfig(
            nodes=(
                node("robot", "192.168.1.10"),
                node("mpc-laptop_2", "laptop.local"),
            ),
            domain_id=robot_ipc.MAX_DOMAIN_ID,
            domain_tag="tag",
            topics=(
                robot_ipc.TopicQos(
                    "a/b",
                    robot_ipc.QosProfile.LATEST,
                    ("robot", "mpc-laptop_2"),
                ),
            ),
            default_qos_profile=robot_ipc.QosProfile.RELIABLE_LATEST,
            transport=robot_ipc.TransportOptions(
                socket_send_buffer_bytes=1 << 20, max_processes_per_host=4
            ),
        )
        for config in (
            robot_ipc.load_network_config(SHIPPED_NETWORK_FILE),
            custom,
        ):
            text = robot_ipc.format_network_config(config)
            self.assertEqual(parse(text), config, text)
        # One line for the domain, one per node and per topic, one for the transport.
        self.assertEqual(
            len(robot_ipc.format_network_config(custom).splitlines()),
            1 + 2 + 1 + 1,
        )


# Each: what is wrong, the text, the line the error names, a word it must contain.
# LINT.IfChange(malformed_files)
MALFORMED_FILES = [
    (
        "an unknown field of a node",
        (
            'nodes { name: "robot" host: "127.0.0.1" }\nnodes { name: "mpc"'
            ' hots: "a" }'
        ),
        2,
        "hots",
    ),
    (
        "the retired port of a node",
        'nodes {\n  name: "robot"\n  port: 5600\n}',
        3,
        "port",
    ),
    (
        "an unknown top-level field",
        '# The nodes.\nhosts { name: "robot" }',
        2,
        "hosts",
    ),
    ("a quoted domain id", 'domain_id: "7"', 1, ""),
    ("a fractional domain id", "domain_id: 7.5", 1, ""),
    (
        "a negative domain id",
        'nodes { name: "robot" host: "127.0.0.1" }\ndomain_id: -1',
        2,
        "",
    ),
    ("a domain id beyond 32 bits", "domain_id: 4294967296", 1, ""),
    ("a boolean domain id", "domain_id: true", 1, ""),
    (
        "a host given twice",
        'nodes {\n  name: "robot"\n  host: "a"\n  host: "b"\n}',
        4,
        "host",
    ),
    (
        "an unquoted host",
        'nodes {\n  name: "robot"\n  host: 127.0.0.1\n}',
        3,
        "",
    ),
    ("a missing colon", 'nodes { name: "robot" }\nnodes { name "mpc" }', 2, ""),
    ("an unclosed node", 'nodes { name: "robot" host: "127.0.0.1"', 1, ""),
]
# LINT.ThenChange(//robot_runtime/robot_ipc/test/testNetworkConfig.cpp:malformed_files)

_ROBOT = 'nodes { name: "robot" host: "127.0.0.1" }\n'

# Each: what is wrong, the text, the key the error must start with.
# LINT.IfChange(invalid_files)
INVALID_FILES = [
    ("an empty file", "", "nodes"),
    ("comments only", "# no node\n", "nodes"),
    ("a name missing", 'nodes { host: "127.0.0.1" }', "nodes[0].name"),
    ("a host missing", 'nodes { name: "robot" }', "nodes[0] (robot).host"),
    (
        "a wildcard host",
        'nodes { name: "robot" host: "0.0.0.0" }',
        "nodes[0] (robot).host",
    ),
    (
        "a host that is an endpoint",
        'nodes { name: "robot" host: "udp://127.0.0.1" }',
        "nodes[0] (robot).host",
    ),
    (
        "a host with a port",
        'nodes { name: "robot" host: "127.0.0.1:7410" }',
        "nodes[0] (robot).host",
    ),
    (
        "a host with a space",
        'nodes { name: "robot" host: "127.0.0.1 " }',
        "nodes[0] (robot).host",
    ),
    (
        "a host with a control character",
        r'nodes { name: "robot" host: "127.0.0.1\034" }',
        "nodes[0] (robot).host",
    ),
    (
        "a host with a non-ASCII space",
        r'nodes { name: "robot" host: "127.0.0.1\302\240" }',
        "nodes[0] (robot).host",
    ),
    (
        "a node name ending in a newline",
        r'nodes { name: "robot\n" host: "127.0.0.1" }',
        "nodes[0]",
    ),
    (
        "an invalid node name",
        'nodes { name: "ro bot" host: "127.0.0.1" }',
        "nodes[0] (ro bot).name",
    ),
    ("a name used twice", _ROBOT + _ROBOT, "nodes[1] (robot).name"),
    ("a domain id out of range", _ROBOT + "domain_id: 233", "domain_id"),
    ("an invalid domain tag", _ROBOT + 'domain_tag: "a b"', "domain_tag"),
    (
        "a topic that is no DDS name",
        _ROBOT + 'topics { name: "robot/a-b" qos_profile: "latest" }',
        "topics[0] (robot/a-b).name",
    ),
    (
        "a topic listed twice",
        _ROBOT
        + 'topics { name: "a" qos_profile: "latest" }\ntopics { name: "a"'
        ' qos_profile: "latest" }',
        "topics[1] (a).name",
    ),
    (
        "a topic without a profile",
        _ROBOT + 'topics { name: "a" }',
        "topics[0] (a).qos_profile",
    ),
    (
        "an unknown profile",
        _ROBOT + 'topics { name: "a" qos_profile: "fast" }',
        "topics[0] (a).qos_profile",
    ),
    (
        "a publisher that is no node",
        _ROBOT + 'topics { name: "a" qos_profile: "latest" publishers: "mpc" }',
        "topics[0] (a).publishers",
    ),
    (
        "an unknown default profile",
        _ROBOT + 'default_qos_profile: "fast"',
        "default_qos_profile",
    ),
    (
        "a message size out of range",
        _ROBOT + "transport { max_message_size_bytes: 70000 }",
        "transport.max_message_size_bytes",
    ),
    (
        "too many processes per host",
        _ROBOT + "transport { max_processes_per_host: 116 }",
        "transport.max_processes_per_host",
    ),
    (
        "a loss of every packet",
        _ROBOT + "transport { simulated_send_loss_per_mille: 1000 }",
        "transport.simulated_send_loss_per_mille",
    ),
    (
        "the second node invalid",
        _ROBOT + 'nodes { name: "mpc" host: "*" }',
        "nodes[1] (mpc).host",
    ),
]
# LINT.ThenChange(//robot_runtime/robot_ipc/test/testNetworkConfig.cpp:invalid_files)


class MalformedFileTest(unittest.TestCase):

    def test_every_malformed_file_is_rejected_with_its_line_and_column(
        self,
    ) -> None:
        for description, text, line, fragment in MALFORMED_FILES:
            with self.subTest(description=description):
                with self.assertRaises(robot_ipc.NetworkConfigError) as raised:
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

    def test_every_invalid_file_is_rejected_with_the_offending_entry(
        self,
    ) -> None:
        for description, text, key in INVALID_FILES:
            with self.subTest(description=description):
                with self.assertRaises(robot_ipc.NetworkConfigError) as raised:
                    parse(text)
                message = str(raised.exception)
                self.assertTrue(message.startswith(f"{SOURCE}: {key}"), message)


class ValidateTest(unittest.TestCase):

    def test_validation_of_a_code_built_network_names_the_entry(self) -> None:
        with self.assertRaisesRegex(
            robot_ipc.NetworkConfigError, r"^nodes\[1\] \(robot\)\.name"
        ):
            robot_ipc.validate_network_config(
                robot_ipc.NetworkConfig(nodes=(node("robot"), node("robot")))
            )
        with self.assertRaises(robot_ipc.NetworkConfigError):
            robot_ipc.validate_network_config(robot_ipc.NetworkConfig(nodes=()))
        with self.assertRaisesRegex(
            robot_ipc.NetworkConfigError, r"^domain_id"
        ):
            robot_ipc.validate_network_config(
                robot_ipc.NetworkConfig(nodes=(node("robot"),), domain_id=-1)
            )
        with self.assertRaisesRegex(
            robot_ipc.NetworkConfigError, r"^transport"
        ):
            robot_ipc.validate_network_config(
                robot_ipc.NetworkConfig(
                    nodes=(node("robot"),),
                    transport=robot_ipc.TransportOptions(
                        socket_receive_buffer_bytes=-1
                    ),
                )
            )

    def test_host_kinds(self) -> None:
        is_ipv4 = robot_ipc.network_config.is_ipv4_address
        is_loopback = robot_ipc.network_config.is_loopback_host
        self.assertTrue(is_ipv4("192.168.1.10"))
        self.assertTrue(is_ipv4("0.0.0.0"))
        for host in (
            "256.1.1.1",
            "1.2.3",
            "1.2.3.4.5",
            "1.2.3.x",
            "laptop.local",
        ):
            with self.subTest(host=host):
                self.assertFalse(is_ipv4(host))
        for host in ("127.0.0.1", "127.1.2.3", "localhost"):
            with self.subTest(host=host):
                self.assertTrue(is_loopback(host))
        for host in ("192.168.1.10", "127.example.com"):
            with self.subTest(host=host):
                self.assertFalse(is_loopback(host))


class DeliveryAndProfileTest(unittest.TestCase):

    def test_names_round_trip_and_unknown_names_list_the_valid_ones(
        self,
    ) -> None:
        for delivery in robot_ipc.Delivery:
            self.assertIs(robot_ipc.parse_delivery(delivery.value), delivery)
            self.assertIs(robot_ipc.parse_delivery(delivery), delivery)
        with self.assertRaises(ValueError) as raised:
            robot_ipc.parse_delivery("newest")
        self.assertIn("latest", str(raised.exception))
        self.assertIn("all", str(raised.exception))
        for profile in robot_ipc.QosProfile:
            self.assertIs(robot_ipc.parse_qos_profile(profile.value), profile)
        with self.assertRaises(ValueError) as raised:
            robot_ipc.parse_qos_profile("fast")
        self.assertIn("reliable_latest", str(raised.exception))

    def test_each_profile_has_its_reliability_and_history(self) -> None:
        policy = robot_ipc.qos_profile.qos_policy
        reliability = robot_ipc.qos_profile.Reliability
        self.assertIs(
            policy(robot_ipc.QosProfile.LATEST).reliability,
            reliability.BEST_EFFORT,
        )
        self.assertEqual(
            policy(robot_ipc.QosProfile.LATEST).writer_history_depth, 1
        )
        self.assertIs(
            policy(robot_ipc.QosProfile.RELIABLE_LATEST).reliability,
            reliability.RELIABLE,
        )
        self.assertEqual(
            policy(robot_ipc.QosProfile.RELIABLE_LATEST).writer_history_depth, 1
        )
        self.assertEqual(
            policy(robot_ipc.QosProfile.RELIABLE_EVENTS).writer_history_depth,
            robot_ipc.qos_profile.RELIABLE_EVENTS_HISTORY_DEPTH,
        )


class DomainConfigTest(unittest.TestCase):

    def test_the_unicast_ports_follow_the_ddsi_mapping_and_stay_below_65536(
        self,
    ) -> None:
        network = robot_ipc.NetworkConfig(nodes=(node("robot"),))
        self.assertEqual(
            robot_ipc.unicast_port_range(network),
            robot_ipc.network_config.PortRange(7410, 7449),
        )
        largest = robot_ipc.NetworkConfig(
            nodes=(node("robot"),), domain_id=robot_ipc.MAX_DOMAIN_ID
        )
        robot_ipc.validate_network_config(largest)
        self.assertLessEqual(
            robot_ipc.unicast_port_range(largest).last, MAX_UDP_PORT
        )
        with self.assertRaisesRegex(
            robot_ipc.NetworkConfigError, r"^domain_id"
        ):
            robot_ipc.validate_network_config(
                robot_ipc.NetworkConfig(
                    nodes=(node("robot"),),
                    domain_id=robot_ipc.MAX_DOMAIN_ID,
                    transport=robot_ipc.TransportOptions(
                        max_processes_per_host=115
                    ),
                )
            )

    def test_a_loopback_network_uses_the_loopback_interface_and_unicast_discovery(
        self,
    ) -> None:
        shipped = robot_ipc.load_network_config(SHIPPED_NETWORK_FILE)
        config = robot_ipc.cyclone_domain_config(shipped, "robot")
        for fragment in (
            '<NetworkInterface address="127.0.0.1"/>',
            "<AllowMulticast>false</AllowMulticast>",
            '<Peers AddLocalhost="false"><Peer Address="127.0.0.1"/></Peers>',
            "<MaxAutoParticipantIndex>19</MaxAutoParticipantIndex>",
            "<LeaseDuration>3000ms</LeaseDuration>",
            "<MaxMessageSize>65500B</MaxMessageSize>",
            '<SocketReceiveBufferSize min="default" max="8388608B"/>',
            "<WhcHigh>8388608B</WhcHigh>",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, config)
        self.assertNotIn("<Tag>", config)
        self.assertEqual(robot_ipc.cyclone_domain_config(shipped, ""), config)

    def test_a_node_uses_its_hosts_interface_or_refuses_another_machines_address(
        self,
    ) -> None:
        network = robot_ipc.NetworkConfig(
            nodes=(
                node("robot", "192.0.2.1"),
                node("mpc", "laptop.example"),
                node("local"),
            )
        )
        with self.assertRaises(robot_ipc.InterfaceError):
            domain_config.interface_address_for(network, "robot")
        self.assertEqual(
            domain_config.interface_address_for(network, "mpc"), ""
        )
        self.assertEqual(
            domain_config.interface_address_for(network, "local"), "127.0.0.1"
        )
        self.assertEqual(domain_config.interface_address_for(network, ""), "")
        config = robot_ipc.cyclone_domain_config(network, "mpc")
        self.assertIn('<NetworkInterface autodetermine="true"/>', config)
        self.assertIn(
            '<Peer Address="192.0.2.1"/><Peer Address="laptop.example"/><Peer'
            ' Address="127.0.0.1"/>',
            config,
        )

    def test_the_tag_and_the_transport_reach_the_configuration(self) -> None:
        network = robot_ipc.NetworkConfig(
            nodes=(node("robot"),),
            domain_tag="lab.robot-2",
            transport=robot_ipc.TransportOptions(
                max_processes_per_host=4,
                lease_duration_ms=1500,
                socket_send_buffer_bytes=1 << 20,
                writer_unacked_bytes_limit=1 << 22,
                simulated_send_loss_per_mille=20,
            ),
        )
        config = robot_ipc.cyclone_domain_config(network, "robot")
        for fragment in (
            "<Tag>lab.robot-2</Tag>",
            "<MaxAutoParticipantIndex>3</MaxAutoParticipantIndex>",
            "<LeaseDuration>1500ms</LeaseDuration>",
            '<SocketSendBufferSize min="default" max="1048576B"/>',
            "<WhcHigh>4194304B</WhcHigh>",
            "<Test><XmitLossiness>20</XmitLossiness></Test>",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, config)
        self.assertNotIn("MaxMessageSize", config)


if __name__ == "__main__":
    unittest.main()
