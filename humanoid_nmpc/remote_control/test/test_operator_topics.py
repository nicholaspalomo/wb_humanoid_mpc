"""Every topic the remote control publishes or subscribes is a topics.py constant, with the README's message.

The contract is the topic table of humanoid_nmpc/docs/distributed_runtime/README.md. operator_bus.OPERATOR_TOPICS is
the remote control's copy of its rows; this test reads the README itself, so a topic, a message or a delivery changed
on either side fails here. It also checks what the code really does with those rows: the topics the operator bus and
the teleoperation publishers publish and subscribe, the topic constants of the tabs, and the nodes they publish as.
"""

import os
import re
import unittest
from typing import Dict, NamedTuple

import robot_ipc
from humanoid_mpc_ipc import topics
from humanoid_mpc_msgs import walking_velocity_command_pb2
from operator_test_support import RecordingBus, repo_path
from remote_control import teleop
from remote_control.dashboard_backend import VirtualJoystick
from remote_control.operator_bus import (
    OPERATOR_NODE,
    OPERATOR_TOPICS,
    TELEOP_NODE,
    Direction,
    OperatorBus,
    TopicPublisher,
)

README = repo_path("humanoid_nmpc", "docs", "distributed_runtime", "README.md")
NETWORK_FILE = repo_path("config", "ipc", "network.textproto")


class TableRow(NamedTuple):
    topic: str
    message: str
    publisher: str
    subscribers: str
    delivery: str


def read_topic_table(path: str) -> Dict[str, TableRow]:
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
    @classmethod
    def setUpClass(cls):
        cls.table = read_topic_table(README)

    def test_the_table_is_read(self):
        # Positive control: the parser finds the rows, and they are the topics of topics.py.
        self.assertEqual(set(self.table), set(topics.ALL_TOPICS))

    def test_every_operator_topic_is_a_constant_with_the_readmes_message_and_delivery(
        self,
    ):
        for row in OPERATOR_TOPICS:
            with self.subTest(topic=row.topic):
                self.assertIn(row.topic, topics.ALL_TOPICS)
                readme = self.table[row.topic]
                self.assertEqual(row.message_class.DESCRIPTOR.name, readme.message)
                self.assertEqual(
                    row.message_class.DESCRIPTOR.full_name,
                    f"humanoid_mpc_msgs.{readme.message}",
                )
                self.assertEqual(row.delivery.value, readme.delivery)

    def test_the_gui_is_on_the_side_the_readme_puts_it(self):
        for row in OPERATOR_TOPICS:
            with self.subTest(topic=row.topic):
                readme = self.table[row.topic]
                side = (
                    readme.publisher
                    if row.direction is Direction.PUBLISH
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
        self.assertEqual({row.topic for row in OPERATOR_TOPICS}, expected - not_yet)

    def test_teleop_publishes_only_what_the_readme_lets_it(self):
        teleop_topics = {
            topic for topic, row in self.table.items() if "teleop" in row.publisher
        }
        self.assertEqual(teleop_topics, {topics.OPERATOR_WALKING_VELOCITY_COMMAND})


class TestWhatTheCodePublishes(unittest.TestCase):
    def test_the_operator_bus_publishes_and_subscribes_exactly_the_table(self):
        bus = RecordingBus()
        operator_bus = OperatorBus(bus)
        publishers = [
            value
            for value in vars(operator_bus).values()
            if isinstance(value, TopicPublisher)
        ]
        published = {
            publisher.topic: publisher.message_class for publisher in publishers
        }
        # The FSM command goes through its sender.
        operator_bus.fsm_command.send("JOINT_PD")
        for topic, message in bus.published:
            published[topic] = type(message)
        expected_published = {
            row.topic: row.message_class
            for row in OPERATOR_TOPICS
            if row.direction is Direction.PUBLISH
        }
        self.assertEqual(published, expected_published)
        subscribed = {topic: entry[0] for topic, entry in bus.subscriptions.items()}
        self.assertEqual(
            subscribed,
            {
                row.topic: row.message_class
                for row in OPERATOR_TOPICS
                if row.direction is Direction.SUBSCRIBE
            },
        )

    def test_the_teleoperation_publishers_publish_the_walking_command(self):
        # VirtualJoystick, and the keyboard and Xbox publishers through teleop.connect_walking_command_publisher, all
        # publish through TopicPublisher.for_topic of the walking command.
        bus = RecordingBus()
        joystick = VirtualJoystick(
            publisher=TopicPublisher.for_topic(
                bus, topics.OPERATOR_WALKING_VELOCITY_COMMAND
            ),
            auto_stream=False,
        )
        self.addCleanup(joystick.shutdown)
        joystick.set_velocity(linear_x=0.5)
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
        # The tab modules import Tk's widgets but need no display to be imported.
        from remote_control.tk_app.dodgeball_tab import DodgeballTab
        from remote_control.tk_app.joint_targets_tab import JointTargetsTab

        operator_bus = OperatorBus(RecordingBus())
        self.assertEqual(DodgeballTab.TOPIC_NAME, operator_bus.dodgeball_throw.topic)
        self.assertEqual(JointTargetsTab.TOPIC_NAME, operator_bus.joint_targets.topic)

    def test_the_walking_command_rate_is_the_readmes(self):
        readme = read_topic_table(README)[topics.OPERATOR_WALKING_VELOCITY_COMMAND]
        with open(README, "r", encoding="utf-8") as handle:
            line = next(row for row in handle if row.startswith(f"| `{readme.topic}`"))
        rate = re.search(r"(\d+(?:\.\d+)?) Hz", line.split("|")[6])
        self.assertIsNotNone(rate, line)
        self.assertEqual(float(rate.group(1)), teleop.WALKING_COMMAND_RATE_HZ)


class TestTheNodes(unittest.TestCase):
    def test_the_gui_and_teleop_nodes_are_in_the_shipped_network_file(self):
        network = robot_ipc.load_network_config(NETWORK_FILE)
        self.assertIsNotNone(network.find(OPERATOR_NODE))
        self.assertIsNotNone(network.find(TELEOP_NODE))
        self.assertNotEqual(
            network.find(OPERATOR_NODE).port, network.find(TELEOP_NODE).port
        )

    def test_the_network_file_path_of_the_binaries_is_the_shipped_one(self):
        from remote_control.operator_bus import DEFAULT_NETWORK_CONFIG

        self.assertTrue(os.path.isfile(repo_path(DEFAULT_NETWORK_CONFIG)))


if __name__ == "__main__":
    unittest.main()
