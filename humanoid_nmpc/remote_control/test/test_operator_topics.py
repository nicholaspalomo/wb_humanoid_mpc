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

"""Every topic the remote control publishes or subscribes is a topics.py constant, with the README's message.

The contract is the topic table of humanoid_nmpc/docs/distributed_runtime/README.md. operator_bus.OPERATOR_TOPICS is
the remote control's copy of its rows; this test reads the README itself, so a topic, a message or a delivery changed
on either side fails here. It also checks what the code really does with those rows: the topics the operator bus and
the teleoperation publishers publish and subscribe, the topic constants of the tabs, and the nodes they publish as.
"""

import io
import os
import re
from typing import NamedTuple
import unittest
from unittest import mock

from humanoid_mpc_msgs import walking_velocity_command_pb2

from humanoid_mpc_ipc import topics
import operator_test_support
from remote_control import keyboard_walking_command_publisher
from remote_control import operator_bus
from remote_control import teleop
from remote_control.tk_app import dodgeball_tab
from remote_control.tk_app import joint_targets_tab
import robot_ipc

README = operator_test_support.repo_path(
    "humanoid_nmpc", "docs", "distributed_runtime", "README.md"
)
NETWORK_FILE = operator_test_support.repo_path("config", "ipc", "network.textproto")

# The packages of the messages on the bus: the bus messages, and the configuration files the tuning tabs publish whole.
MESSAGE_PACKAGES = ("humanoid_mpc_msgs", "humanoid_mpc_config")


class TableRow(NamedTuple):
    topic: str
    message: str
    publisher: str
    subscribers: str
    delivery: str


def read_topic_table(path: str) -> dict[str, TableRow]:
    """The rows of the README's topic table (between its LINT.IfChange(topic_table) and LINT.ThenChange), by topic."""
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    start = text.index("LINT.IfChange(topic_table)")
    end = text.index("LINT.ThenChange", start)
    rows = {}
    for line in text[start:end].splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 6 or not cells[0].startswith("`"):
            continue
        topic, message = cells[0].strip("`"), cells[1].strip("`")
        rows[topic] = TableRow(topic, message, cells[2], cells[3], cells[4])
    return rows


class TestTheReadmeTable(unittest.TestCase):
    table: dict[str, TableRow]

    @classmethod
    def setUpClass(cls):
        cls.table = read_topic_table(README)

    def test_the_table_is_read(self):
        # Positive control: the parser finds the rows, and they are the topics of topics.py.
        self.assertEqual(set(self.table), set(topics.ALL_TOPICS))

    def test_every_operator_topic_is_a_constant_with_the_readmes_message_and_delivery(
        self,
    ):
        for row in operator_bus.OPERATOR_TOPICS:
            with self.subTest(topic=row.topic):
                self.assertIn(row.topic, topics.ALL_TOPICS)
                readme = self.table[row.topic]
                self.assertIn(
                    row.message_class.DESCRIPTOR.file.package, MESSAGE_PACKAGES
                )
                self.assertEqual(row.message_class.DESCRIPTOR.name, readme.message)
                self.assertEqual(row.delivery.value, readme.delivery)

    def test_the_tuning_topics_carry_their_files_typed(self):
        # The payload is the file: the whole task and contact-planning file, the whole PD gains file. A save carries
        # the saved file's text, which the robot parses as it parses the file at start-up, and the robot answers it.
        expected = {
            topics.OPERATOR_MPC_PARAMETERS: "humanoid_mpc_config.MpcParameterUpdate",
            topics.OPERATOR_PD_GAINS: "humanoid_mpc_config.JointPdGainsFile",
            topics.OPERATOR_DODGEBALL_THROW: "humanoid_mpc_msgs.DodgeballThrow",
            topics.OPERATOR_CONFIG_SAVE: "humanoid_mpc_msgs.ConfigFileSave",
            topics.ROBOT_CONFIG_SAVE_STATUS: "humanoid_mpc_msgs.ConfigFileSaveStatus",
        }
        for topic, name in expected.items():
            with self.subTest(topic=topic):
                self.assertEqual(
                    operator_bus.operator_topic(
                        topic
                    ).message_class.DESCRIPTOR.full_name,
                    name,
                )

    def test_the_gui_is_on_the_side_the_readme_puts_it(self):
        for row in operator_bus.OPERATOR_TOPICS:
            with self.subTest(topic=row.topic):
                readme = self.table[row.topic]
                side = (
                    readme.publisher
                    if row.direction is operator_bus.Direction.PUBLISH
                    else readme.subscribers
                )
                self.assertIn("GUI", side)

    def test_every_topic_the_readme_gives_the_gui_is_in_the_table(self):
        # The GUI's subscriptions to mpc/status and robot/loop_timing are not built yet (the GUI shows neither); every
        # other topic the README has the GUI publish or subscribe is in OPERATOR_TOPICS.
        not_yet = {topics.MPC_STATUS, topics.ROBOT_LOOP_TIMING}
        expected = {
            topic
            for topic, row in self.table.items()
            if "GUI" in row.publisher or "GUI" in row.subscribers
        }
        self.assertEqual(
            {row.topic for row in operator_bus.OPERATOR_TOPICS}, expected - not_yet
        )

    def test_teleop_publishes_only_what_the_readme_lets_it(self):
        teleop_topics = {
            topic for topic, row in self.table.items() if "teleop" in row.publisher
        }
        self.assertEqual(teleop_topics, {topics.OPERATOR_WALKING_VELOCITY_COMMAND})


class TestWhatTheCodePublishes(unittest.TestCase):
    def test_the_operator_bus_publishes_and_subscribes_exactly_the_table(self):
        bus = operator_test_support.RecordingBus()
        gui_bus = operator_bus.OperatorBus(bus)
        publishers = [
            value
            for value in vars(gui_bus).values()
            if isinstance(value, operator_bus.TopicPublisher)
        ]
        published = {
            publisher.topic: publisher.message_class for publisher in publishers
        }
        # The FSM command goes through its sender.
        gui_bus.fsm_command.send("JOINT_PD")
        for topic, message in bus.published:
            published[topic] = type(message)
        expected_published = {
            row.topic: row.message_class
            for row in operator_bus.OPERATOR_TOPICS
            if row.direction is operator_bus.Direction.PUBLISH
        }
        self.assertEqual(published, expected_published)
        subscribed = {topic: entry[0] for topic, entry in bus.subscriptions.items()}
        self.assertEqual(
            subscribed,
            {
                row.topic: row.message_class
                for row in operator_bus.OPERATOR_TOPICS
                if row.direction is operator_bus.Direction.SUBSCRIBE
            },
        )

    def test_the_teleoperation_publishers_publish_the_walking_command(self):
        # The keyboard and Xbox publishers publish through teleop.connect_walking_command_publisher: a started bus of
        # the teleop node, and TopicPublisher.for_topic of the walking command on it.
        buses: dict[str, operator_test_support.RecordingBus] = {}

        def recording_bus(
            node_name: str, network: object
        ) -> operator_test_support.RecordingBus:
            del network  # Unused.
            return buses.setdefault(node_name, operator_test_support.RecordingBus())

        with mock.patch.object(robot_ipc, "Bus", recording_bus):
            publisher = teleop.connect_walking_command_publisher(NETWORK_FILE)
        self.assertEqual(list(buses), [operator_bus.TELEOP_NODE])
        bus = buses[operator_bus.TELEOP_NODE]
        self.assertIs(publisher.bus, bus)
        self.assertTrue(bus.started)
        keyboard_walking_command_publisher.KeyboardWalkingCommandPublisher(
            publisher, lambda: None, lambda: None, output=io.StringIO()
        ).tick()
        self.assertEqual(
            [(topic, type(message)) for topic, message in bus.published],
            [
                (
                    topics.OPERATOR_WALKING_VELOCITY_COMMAND,
                    walking_velocity_command_pb2.WalkingVelocityCommand,
                )
            ],
        )

    def test_the_tabs_topic_constants_are_the_topics_of_their_publishers(self):
        # The tab modules import Tk's widgets, but neither importing them nor reading their constants needs a display.
        gui_bus = operator_bus.OperatorBus(operator_test_support.RecordingBus())
        self.assertEqual(
            dodgeball_tab.DodgeballTab.TOPIC_NAME, gui_bus.dodgeball_throw.topic
        )
        self.assertEqual(
            joint_targets_tab.JointTargetsTab.TOPIC_NAME,
            gui_bus.joint_targets.topic,
        )

    def test_the_walking_command_rate_is_the_readmes(self):
        readme = read_topic_table(README)[topics.OPERATOR_WALKING_VELOCITY_COMMAND]
        with open(README, "r", encoding="utf-8") as handle:
            line = next(row for row in handle if row.startswith(f"| `{readme.topic}`"))
        rate = re.search(r"(\d+(?:\.\d+)?) Hz", line.split("|")[6])
        assert rate is not None, line
        self.assertEqual(float(rate.group(1)), teleop.WALKING_COMMAND_RATE_HZ)


class TestTheNodes(unittest.TestCase):
    def test_the_gui_teleop_and_config_push_nodes_are_in_the_shipped_network_file(self):
        network = robot_ipc.load_network_config(NETWORK_FILE)
        ports = []
        for name in (
            operator_bus.OPERATOR_NODE,
            operator_bus.TELEOP_NODE,
            operator_bus.CONFIG_PUSH_NODE,
        ):
            with self.subTest(node=name):
                node = network.find(name)
                assert node is not None, name
                ports.append(node.port)
        # Each binds its own endpoint, so push_robot_config runs next to a GUI.
        self.assertEqual(len(set(ports)), len(ports))

    def test_the_network_file_path_of_the_binaries_is_the_shipped_one(self):
        self.assertTrue(
            os.path.isfile(
                operator_test_support.repo_path(operator_bus.DEFAULT_NETWORK_CONFIG)
            )
        )


if __name__ == "__main__":
    unittest.main()
