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

"""The shipped network files parse strictly, and the two-machine example is the shipped bus split over two machines.

The localhost network and the two-machine example both load with the bus's strict loader, and the example is the same
bus (the same nodes, topics, QoS profiles and domain) with the robot on a machine of its own.
"""

import os
import unittest

import robot_ipc

HERE = os.path.dirname(os.path.abspath(__file__))
LAPTOP_NODES = ("mpc", "operator", "teleop", "config_push")


def load(name: str) -> robot_ipc.NetworkConfig:
    return robot_ipc.load_network_config(os.path.join(HERE, name))


class NetworkFilesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.localhost = load("network.textproto")
        self.example = load("two_machine.example.textproto")

    def test_the_example_has_the_nodes_topics_and_domain_of_the_shipped_network(
        self,
    ) -> None:
        self.assertEqual(self.example.node_names(), self.localhost.node_names())
        self.assertEqual(self.example.topics, self.localhost.topics)
        self.assertEqual(self.example.domain_id, self.localhost.domain_id)

    def test_every_node_of_the_example_is_on_a_machine_address(self) -> None:
        for node in self.example.nodes:
            with self.subTest(node=node.name):
                self.assertTrue(
                    robot_ipc.network_config.is_ipv4_address(node.host)
                )
                self.assertFalse(
                    robot_ipc.network_config.is_loopback_host(node.host)
                )

    def test_the_robot_is_on_a_machine_of_its_own(self) -> None:
        hosts = {node.name: node.host for node in self.example.nodes}
        self.assertEqual(len({hosts[name] for name in LAPTOP_NODES}), 1)
        self.assertNotEqual(hosts["robot"], hosts["mpc"])

    def test_the_shipped_network_is_on_this_machine(self) -> None:
        self.assertTrue(
            all(
                robot_ipc.network_config.is_loopback_host(host)
                for host in self.localhost.peer_hosts()
            )
        )


if __name__ == "__main__":
    unittest.main()
