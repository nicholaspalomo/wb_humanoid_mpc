"""The remote control's side of the bus: the publishers, the FSM command numbering, the FSM state mailbox, and the
operator bus itself, first over a bus that records and then over real robot_ipc buses on loopback TCP.

The loopback tests bind ephemeral ports (robot_ipc.EPHEMERAL_PORT) and connect to each other's bound endpoint, so they
never collide with a running GUI or with each other.
"""

import threading
import time
import unittest

import robot_ipc
from humanoid_mpc_ipc import topics
from humanoid_mpc_msgs import fsm_command_pb2
from humanoid_mpc_msgs import fsm_state_pb2
from humanoid_mpc_msgs import joint_targets_pb2
from humanoid_mpc_msgs import walking_velocity_command_pb2
from humanoid_mpc_msgs import yaml_document_pb2
from operator_test_support import RecordingBus
from remote_control.operator_bus import (
    OPERATOR_NODE,
    OPERATOR_TOPICS,
    Direction,
    FsmCommandSender,
    FsmStateMailbox,
    OperatorBus,
    TopicPublisher,
    joint_targets,
    operator_topic,
    walking_velocity_command,
    yaml_document,
)

TIMEOUT = 20.0


def wait_for(condition, timeout=TIMEOUT):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.005)
    return condition()


class TestTopicPublisher(unittest.TestCase):
    def test_it_publishes_its_message_on_its_topic(self):
        bus = RecordingBus()
        publisher = TopicPublisher.for_topic(bus, topics.OPERATOR_PD_GAINS)
        self.assertTrue(publisher.publish(yaml_document("kp: 1\n")))
        self.assertEqual(len(bus.published), 1)
        topic, message = bus.published[0]
        self.assertEqual(topic, topics.OPERATOR_PD_GAINS)
        self.assertEqual(message, yaml_document_pb2.YamlDocument(yaml="kp: 1\n"))

    def test_a_message_of_another_type_is_refused_before_it_is_sent(self):
        bus = RecordingBus()
        publisher = TopicPublisher.for_topic(bus, topics.OPERATOR_JOINT_TARGETS)
        with self.assertRaisesRegex(TypeError, "humanoid_mpc_msgs.JointTargets"):
            publisher.publish(yaml_document("a: 1"))
        self.assertEqual(bus.published, [])

    def test_only_the_topics_the_remote_control_publishes(self):
        with self.assertRaisesRegex(ValueError, "subscribes"):
            TopicPublisher.for_topic(RecordingBus(), topics.ROBOT_FSM_STATE)
        with self.assertRaisesRegex(ValueError, "operator/walking_velocity_command"):
            TopicPublisher.for_topic(RecordingBus(), topics.MPC_POLICY)

    def test_a_publisher_needs_a_topic_and_a_message_class(self):
        with self.assertRaises(ValueError):
            TopicPublisher(RecordingBus(), "", yaml_document_pb2.YamlDocument)
        with self.assertRaises(ValueError):
            TopicPublisher(RecordingBus(), topics.OPERATOR_PD_GAINS, str)


class TestMessageBuilders(unittest.TestCase):
    def test_the_walking_command_carries_the_four_values(self):
        self.assertEqual(
            walking_velocity_command(0.5, -0.25, 0.75, 0.8),
            walking_velocity_command_pb2.WalkingVelocityCommand(
                linear_velocity_x=0.5,
                linear_velocity_y=-0.25,
                angular_velocity_z=0.75,
                desired_pelvis_height=0.8,
            ),
        )

    def test_the_yaml_document_carries_the_text_unchanged(self):
        text = "# a comment\nQ:\n  scaling: 1e-3   # kept\n"
        self.assertEqual(yaml_document(text).yaml, text)

    def test_joint_targets_is_the_map(self):
        message = joint_targets({"l_leg_kny": 0.5, "r_leg_kny": -0.25})
        self.assertEqual(
            dict(message.positions), {"l_leg_kny": 0.5, "r_leg_kny": -0.25}
        )
        self.assertEqual(dict(joint_targets().positions), {})


class TestFsmCommandSender(unittest.TestCase):
    def _sender(self, clock_ns):
        bus = RecordingBus()
        return bus, FsmCommandSender(
            TopicPublisher.for_topic(bus, topics.OPERATOR_FSM_COMMAND),
            clock_ns=lambda: clock_ns,
        )

    def test_every_command_has_a_higher_sequence(self):
        bus, sender = self._sender(1000)
        for command in ("JOINT_PD", "LOCK_GANTRY", "JOINT_PD", "WB_MPC"):
            sender.send(command)
        messages = bus.messages_on(topics.OPERATOR_FSM_COMMAND)
        self.assertEqual(
            [message.command for message in messages],
            ["JOINT_PD", "LOCK_GANTRY", "JOINT_PD", "WB_MPC"],
        )
        sequences = [message.sequence for message in messages]
        self.assertEqual(sequences, sorted(set(sequences)))
        self.assertGreater(sequences[0], 1000)
        self.assertEqual(sender.last_sequence, sequences[-1])

    def test_a_restarted_sender_counts_on_from_the_clock(self):
        # The second run starts later on the wall clock, so its commands are above the first run's.
        _, first = self._sender(1_000)
        for _ in range(5):
            first.send("JOINT_PD")
        _, second = self._sender(2_000)
        self.assertGreater(second.send("JOINT_PD").sequence, first.last_sequence)

    def test_the_real_clock_seeds_a_sequence_above_zero(self):
        bus = RecordingBus()
        sender = FsmCommandSender(
            TopicPublisher.for_topic(bus, topics.OPERATOR_FSM_COMMAND)
        )
        self.assertGreater(sender.send("ZERO_TORQUE").sequence, 0)

    def test_concurrent_senders_never_share_a_sequence(self):
        bus, sender = self._sender(0)
        threads = [
            threading.Thread(target=lambda: [sender.send("SAFETY") for _ in range(200)])
            for _ in range(4)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        sequences = [
            message.sequence for message in bus.messages_on(topics.OPERATOR_FSM_COMMAND)
        ]
        self.assertEqual(len(sequences), 800)
        self.assertEqual(len(set(sequences)), 800)

    def test_an_empty_command_or_a_publisher_of_another_message_is_refused(self):
        _, sender = self._sender(0)
        with self.assertRaises(ValueError):
            sender.send("")
        with self.assertRaises(ValueError):
            FsmCommandSender(
                TopicPublisher.for_topic(RecordingBus(), topics.OPERATOR_PD_GAINS)
            )


class TestFsmStateMailbox(unittest.TestCase):
    def test_messages_come_out_once_in_the_order_they_arrived(self):
        mailbox = FsmStateMailbox()
        for count in range(3):
            mailbox.put(
                fsm_state_pb2.FsmState(mode="JOINT_PD", controller_resets=count)
            )
        self.assertEqual(
            [state.controller_resets for state in mailbox.take_all()], [0, 1, 2]
        )
        self.assertEqual(mailbox.take_all(), [])

    def test_a_full_mailbox_drops_the_oldest(self):
        mailbox = FsmStateMailbox(capacity=2)
        for count in range(5):
            mailbox.put(fsm_state_pb2.FsmState(mode="SAFETY", controller_resets=count))
        self.assertEqual(
            [state.controller_resets for state in mailbox.take_all()], [3, 4]
        )
        self.assertEqual(mailbox.dropped, 3)

    def test_the_capacity_must_be_positive(self):
        with self.assertRaises(ValueError):
            FsmStateMailbox(capacity=0)


class TestOperatorBusOverARecordingBus(unittest.TestCase):
    def setUp(self):
        self.bus = RecordingBus()
        self.operator_bus = OperatorBus(self.bus)

    def test_it_subscribes_to_the_robots_fsm_state_as_the_readme_says(self):
        self.assertEqual(set(self.bus.subscriptions), {topics.ROBOT_FSM_STATE})
        message_class, _, delivery = self.bus.subscriptions[topics.ROBOT_FSM_STATE]
        self.assertIs(message_class, fsm_state_pb2.FsmState)
        self.assertEqual(delivery, operator_topic(topics.ROBOT_FSM_STATE).delivery)

    def test_a_received_state_waits_in_the_mailbox_for_the_gui(self):
        state = fsm_state_pb2.FsmState(
            mode="WB_MPC", gantry_locked=False, controller_resets=2
        )
        self.bus.deliver(topics.ROBOT_FSM_STATE, state)
        self.assertEqual(self.operator_bus.take_fsm_states(), [state])
        self.assertEqual(self.operator_bus.take_fsm_states(), [])

    def test_each_publisher_publishes_its_topic(self):
        self.operator_bus.walking_velocity_command.publish(
            walking_velocity_command(0.1, 0.0, 0.0, 0.8)
        )
        self.operator_bus.fsm_command.send("JOINT_PD")
        self.operator_bus.mpc_parameters.publish(yaml_document("Q: {}\n"))
        self.operator_bus.pd_gains.publish(yaml_document("kp: 1\n"))
        self.operator_bus.joint_targets.publish(joint_targets({"j": 0.1}))
        self.operator_bus.dodgeball_throw.publish(yaml_document("dodgeball: {}\n"))
        published = {topic: type(message) for topic, message in self.bus.published}
        self.assertEqual(
            published,
            {
                topics.OPERATOR_WALKING_VELOCITY_COMMAND: walking_velocity_command_pb2.WalkingVelocityCommand,
                topics.OPERATOR_FSM_COMMAND: fsm_command_pb2.FsmCommand,
                topics.OPERATOR_MPC_PARAMETERS: yaml_document_pb2.YamlDocument,
                topics.OPERATOR_PD_GAINS: yaml_document_pb2.YamlDocument,
                topics.OPERATOR_JOINT_TARGETS: joint_targets_pb2.JointTargets,
                topics.OPERATOR_DODGEBALL_THROW: yaml_document_pb2.YamlDocument,
            },
        )
        # The table and the bus agree: every topic it publishes is a PUBLISH row, and every PUBLISH row is published.
        self.assertEqual(
            set(published),
            {
                row.topic
                for row in OPERATOR_TOPICS
                if row.direction is Direction.PUBLISH
            },
        )

    def test_start_and_close_reach_the_bus(self):
        with self.operator_bus:
            self.assertTrue(self.bus.started)
        self.assertTrue(self.bus.closed)


def _ephemeral_network(name):
    return robot_ipc.NetworkConfig(
        nodes=(robot_ipc.NodeEndpoint(name, "127.0.0.1", robot_ipc.EPHEMERAL_PORT),)
    )


class TestOperatorBusOverLoopback(unittest.TestCase):
    """The operator bus between a robot bus and an MPC bus, all three real, on loopback TCP."""

    def setUp(self):
        self.robot = robot_ipc.Bus("robot", _ephemeral_network("robot"))
        self.addCleanup(self.robot.close)
        operator = robot_ipc.Bus(OPERATOR_NODE, _ephemeral_network(OPERATOR_NODE))
        self.operator_bus = OperatorBus(operator)
        self.addCleanup(self.operator_bus.close)
        # The operator hears the robot, and the robot hears the operator.
        operator.connect(self.robot.bound_endpoint)
        self.robot.connect(operator.bound_endpoint)

        self.lock = threading.Lock()
        self.received = {}
        for topic, message_class, delivery in (
            (
                topics.OPERATOR_FSM_COMMAND,
                fsm_command_pb2.FsmCommand,
                robot_ipc.Delivery.ALL,
            ),
            (
                topics.OPERATOR_MPC_PARAMETERS,
                yaml_document_pb2.YamlDocument,
                robot_ipc.Delivery.LATEST,
            ),
            (
                topics.OPERATOR_JOINT_TARGETS,
                joint_targets_pb2.JointTargets,
                robot_ipc.Delivery.LATEST,
            ),
        ):
            self.robot.subscribe(
                topic, message_class, self._collector(topic), delivery=delivery
            )
        self.robot.start()
        self.operator_bus.start()

    def _collector(self, topic):
        def collect(message):
            with self.lock:
                self.received.setdefault(topic, []).append(message)

        return collect

    def _received(self, topic):
        with self.lock:
            return list(self.received.get(topic, []))

    def _publish_until_received(self, publish, topic):
        """Publishes until the robot has the message (ZeroMQ drops what is sent before the connection is up)."""

        def arrived():
            publish()
            time.sleep(0.01)
            return bool(self._received(topic))

        self.assertTrue(wait_for(arrived), f"nothing arrived on {topic}")
        return self._received(topic)[-1]

    def test_a_yaml_document_reaches_the_robot_byte_for_byte(self):
        text = "Q:\n  scaling: 1.5e-3   # a comment the receiver ignores\ncontactEstimator: cheater_sim\n"
        received = self._publish_until_received(
            lambda: self.operator_bus.mpc_parameters.publish(yaml_document(text)),
            topics.OPERATOR_MPC_PARAMETERS,
        )
        self.assertEqual(received.yaml, text)

    def test_joint_targets_reach_the_robot_as_a_map(self):
        targets = {"l_leg_kny": 0.5, "r_leg_kny": -0.125}
        received = self._publish_until_received(
            lambda: self.operator_bus.joint_targets.publish(joint_targets(targets)),
            topics.OPERATOR_JOINT_TARGETS,
        )
        self.assertEqual(dict(received.positions), targets)

    def test_fsm_commands_arrive_in_order_with_rising_sequences(self):
        self._publish_until_received(
            lambda: self.operator_bus.fsm_command.send("ZERO_TORQUE"),
            topics.OPERATOR_FSM_COMMAND,
        )
        for command in ("JOINT_PD", "LOCK_GANTRY", "WB_MPC"):
            self.operator_bus.fsm_command.send(command)
        self.assertTrue(
            wait_for(
                lambda: self._received(topics.OPERATOR_FSM_COMMAND)[-1].command
                == "WB_MPC"
            )
        )
        received = self._received(topics.OPERATOR_FSM_COMMAND)
        self.assertEqual(
            [message.command for message in received[-3:]],
            ["JOINT_PD", "LOCK_GANTRY", "WB_MPC"],
        )
        sequences = [message.sequence for message in received]
        self.assertEqual(sequences, sorted(sequences))
        self.assertEqual(len(set(sequences)), len(sequences))

    def test_the_robots_fsm_state_reaches_the_gui_mailbox(self):
        state = fsm_state_pb2.FsmState(
            mode="JOINT_PD", gantry_locked=True, controller_resets=4, mpc_healthy=True
        )
        taken = []

        def arrived():
            self.robot.publish(topics.ROBOT_FSM_STATE, state)
            time.sleep(0.01)
            taken.extend(self.operator_bus.take_fsm_states())
            return bool(taken)

        self.assertTrue(
            wait_for(arrived), "the FSM state never reached the operator bus"
        )
        self.assertEqual(taken[-1], state)


class TestOperatorBusConnect(unittest.TestCase):
    def test_a_node_that_is_not_in_the_network_is_refused(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "network.textproto")
            with open(path, "w") as handle:
                handle.write('nodes { name: "robot" host: "127.0.0.1" port: 5600 }\n')
            with self.assertRaisesRegex(ValueError, "operator"):
                OperatorBus.connect(path)

    def test_a_missing_network_file_is_an_error_that_names_it(self):
        with self.assertRaisesRegex(FileNotFoundError, "no_such_network.textproto"):
            OperatorBus.connect("/does/not/exist/no_such_network.textproto")


if __name__ == "__main__":
    unittest.main()
