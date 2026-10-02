"""The shipped network files: the localhost network and the two-machine example parse strictly, and the example is the
same bus split over the robot's computer and the laptop."""

import os
import unittest

import robot_ipc

HERE = os.path.dirname(os.path.abspath(__file__))
LAPTOP_NODES = ("mpc", "operator", "teleop")


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
