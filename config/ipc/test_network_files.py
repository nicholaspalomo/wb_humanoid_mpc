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
bus split over the robot's computer and the laptop.
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

    def test_the_example_has_the_nodes_and_ports_of_the_shipped_network(self) -> None:
        self.assertEqual(
            [(node.name, node.port) for node in self.example.nodes],
            [(node.name, node.port) for node in self.localhost.nodes],
        )

    def test_every_node_of_the_example_binds_every_interface(self) -> None:
        for node in self.example.nodes:
            with self.subTest(node=node.name):
                self.assertEqual(node.bind_host, "0.0.0.0")
                self.assertNotIn(node.host, ("0.0.0.0", "*", "127.0.0.1", "localhost"))

    def test_the_robot_is_on_a_machine_of_its_own(self) -> None:
        hosts = {node.name: node.host for node in self.example.nodes}
        self.assertEqual(len({hosts[name] for name in LAPTOP_NODES}), 1)
        self.assertNotEqual(hosts["robot"], hosts["mpc"])

    def test_the_shipped_network_is_on_this_machine(self) -> None:
        self.assertEqual({node.host for node in self.localhost.nodes}, {"127.0.0.1"})


if __name__ == "__main__":
    unittest.main()
