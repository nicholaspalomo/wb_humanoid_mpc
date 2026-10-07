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

"""The remote control's side of the bus: the publishers, the FSM commands, the FSM state mailbox and the operator bus.

The operator bus is tested first over a bus that records and then over real robot_ipc buses on loopback TCP. The
loopback tests bind ephemeral ports (robot_ipc.EPHEMERAL_PORT) and connect to each other's bound endpoint, so they
never collide with a running GUI or with each other.
"""

import os
import tempfile
import threading
import time
import unittest

from humanoid_mpc_config import contact_planning_file_pb2
from humanoid_mpc_config import joint_pd_gains_file_pb2
from humanoid_mpc_config import mpc_parameter_update_pb2
from humanoid_mpc_config import task_file_pb2
from humanoid_mpc_msgs import config_file_save_pb2
from humanoid_mpc_msgs import config_file_save_status_pb2
from humanoid_mpc_msgs import dodgeball_throw_pb2
from humanoid_mpc_msgs import fsm_command_pb2
from humanoid_mpc_msgs import fsm_state_pb2
from humanoid_mpc_msgs import joint_targets_pb2
from humanoid_mpc_msgs import walking_velocity_command_pb2

from humanoid_mpc_ipc import topics
import nproto_schema
import operator_test_support
from remote_control import operator_bus
import robot_ipc

TIMEOUT = 20.0
# The identity of a task file, as the GUI's tab fills it in (robot_config_save.config_path_of()).
CONFIG_PATH = (
    "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.textproto"
)


def _gains(kp: float) -> joint_pd_gains_file_pb2.JointPdGainsFile:
    gains = joint_pd_gains_file_pb2.JointPdGainsFile()
    gains.default_gains.kp = kp
    return gains


def wait_for(condition, timeout=TIMEOUT):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.005)
    return condition()


class TestTopicPublisher(unittest.TestCase):
    def test_it_publishes_its_message_on_its_topic(self):
        bus = operator_test_support.RecordingBus()
        publisher = operator_bus.TopicPublisher.for_topic(bus, topics.OPERATOR_PD_GAINS)
        self.assertTrue(publisher.publish(_gains(1.0)))
        self.assertEqual(len(bus.published), 1)
        topic, message = bus.published[0]
        self.assertEqual(topic, topics.OPERATOR_PD_GAINS)
        self.assertEqual(message, _gains(1.0))

    def test_a_message_of_another_type_is_refused_before_it_is_sent(self):
        bus = operator_test_support.RecordingBus()
        publisher = operator_bus.TopicPublisher.for_topic(
            bus, topics.OPERATOR_JOINT_TARGETS
        )
        with self.assertRaisesRegex(TypeError, "humanoid_mpc_msgs.JointTargets"):
            publisher.publish(_gains(1.0))
        self.assertEqual(bus.published, [])

    def test_only_the_topics_the_remote_control_publishes(self):
        with self.assertRaisesRegex(ValueError, "subscribes"):
            operator_bus.TopicPublisher.for_topic(
                operator_test_support.RecordingBus(), topics.ROBOT_FSM_STATE
            )
        with self.assertRaisesRegex(ValueError, "operator/walking_velocity_command"):
            operator_bus.TopicPublisher.for_topic(
                operator_test_support.RecordingBus(), topics.MPC_POLICY
            )

    def test_a_publisher_needs_a_topic_and_a_message_class(self):
        with self.assertRaises(ValueError):
            operator_bus.TopicPublisher(
                operator_test_support.RecordingBus(),
                "",
                joint_pd_gains_file_pb2.JointPdGainsFile,
            )
        with self.assertRaises(ValueError):
            operator_bus.TopicPublisher(
                operator_test_support.RecordingBus(), topics.OPERATOR_PD_GAINS, str
            )


class TestMessageBuilders(unittest.TestCase):
    def test_the_walking_command_carries_the_four_values(self):
        self.assertEqual(
            operator_bus.walking_velocity_command(0.5, -0.25, 0.75, 0.8),
            walking_velocity_command_pb2.WalkingVelocityCommand(
                linear_velocity_x=0.5,
                linear_velocity_y=-0.25,
                angular_velocity_z=0.75,
                desired_pelvis_height=0.8,
            ),
        )

    def test_the_parameter_update_carries_both_files_whole(self):
        task = task_file_pb2.TaskFile(terrain_height=0.25)
        task.state_weights.scaling = 85.0
        planner = contact_planning_file_pb2.ContactPlanningFile()
        planner.planner.dt = 0.05
        update = operator_bus.mpc_parameter_update(
            task, planner, config_path=CONFIG_PATH
        )
        self.assertEqual(update.task, task)
        self.assertTrue(update.HasField("contact_planning"))
        self.assertEqual(update.contact_planning, planner)
        # A robot without a contact planner sends none, which the receiver tells from an empty file.
        alone = operator_bus.mpc_parameter_update(task, config_path=CONFIG_PATH)
        self.assertFalse(alone.HasField("contact_planning"))
        self.assertEqual(alone.task, task)

    def test_the_parameter_update_carries_the_task_files_identity(self):
        # A receiver that runs another configuration refuses the update (OperatorPayloadChecks.h); the parameter is
        # keyword-only and has no default, so every caller names it.
        update = operator_bus.mpc_parameter_update(
            task_file_pb2.TaskFile(), config_path=CONFIG_PATH
        )
        self.assertEqual(update.config_path, CONFIG_PATH)

    def test_the_parameter_update_carries_the_fingerprint_of_this_builds_schema(self):
        # The receivers refuse an update whose fingerprint is not their own (OperatorPayloadChecks.h).
        update = operator_bus.mpc_parameter_update(
            task_file_pb2.TaskFile(), config_path=CONFIG_PATH
        )
        self.assertEqual(
            update.schema_fingerprint,
            nproto_schema.schema_fingerprint(
                mpc_parameter_update_pb2.MpcParameterUpdate.DESCRIPTOR
            ),
        )
        self.assertRegex(update.schema_fingerprint, r"^[0-9a-f]{16}$")

    def test_joint_targets_is_the_map(self):
        message = operator_bus.joint_targets({"l_leg_kny": 0.5, "r_leg_kny": -0.25})
        self.assertEqual(
            dict(message.positions), {"l_leg_kny": 0.5, "r_leg_kny": -0.25}
        )
        self.assertEqual(dict(operator_bus.joint_targets().positions), {})


class TestFsmCommandSender(unittest.TestCase):
    def _sender(self, clock_ns):
        bus = operator_test_support.RecordingBus()
        return bus, operator_bus.FsmCommandSender(
            operator_bus.TopicPublisher.for_topic(bus, topics.OPERATOR_FSM_COMMAND),
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
        bus = operator_test_support.RecordingBus()
        sender = operator_bus.FsmCommandSender(
            operator_bus.TopicPublisher.for_topic(bus, topics.OPERATOR_FSM_COMMAND)
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
            operator_bus.FsmCommandSender(
                operator_bus.TopicPublisher.for_topic(
                    operator_test_support.RecordingBus(), topics.OPERATOR_PD_GAINS
                )
            )


class TestFsmStateMailbox(unittest.TestCase):
    def test_messages_come_out_once_in_the_order_they_arrived(self):
        mailbox = operator_bus.FsmStateMailbox()
        for count in range(3):
            mailbox.put(
                fsm_state_pb2.FsmState(mode="JOINT_PD", controller_resets=count)
            )
        self.assertEqual(
            [state.controller_resets for state in mailbox.take_all()], [0, 1, 2]
        )
        self.assertEqual(mailbox.take_all(), [])

    def test_a_full_mailbox_drops_the_oldest(self):
        mailbox = operator_bus.FsmStateMailbox(capacity=2)
        for count in range(5):
            mailbox.put(fsm_state_pb2.FsmState(mode="SAFETY", controller_resets=count))
        self.assertEqual(
            [state.controller_resets for state in mailbox.take_all()], [3, 4]
        )
        self.assertEqual(mailbox.dropped, 3)

    def test_the_capacity_must_be_positive(self):
        with self.assertRaises(ValueError):
            operator_bus.FsmStateMailbox(capacity=0)

    def test_the_save_statuses_have_a_mailbox_of_their_own(self):
        mailbox = operator_bus.ConfigSaveStatusMailbox(capacity=2)
        for sequence in (1, 2, 3):
            mailbox.put(
                config_file_save_status_pb2.ConfigFileSaveStatus(sequence=sequence)
            )
        self.assertEqual([status.sequence for status in mailbox.take_all()], [2, 3])
        self.assertEqual(mailbox.dropped, 1)
        self.assertEqual(mailbox.take_all(), [])


class TestOperatorBusOverARecordingBus(unittest.TestCase):
    def setUp(self):
        self.bus = operator_test_support.RecordingBus()
        self.gui_bus = operator_bus.OperatorBus(self.bus)

    def test_it_subscribes_to_the_robots_fsm_state_and_save_statuses_as_the_readme_says(
        self,
    ):
        self.assertEqual(
            set(self.bus.subscriptions),
            {topics.ROBOT_FSM_STATE, topics.ROBOT_CONFIG_SAVE_STATUS},
        )
        for topic, expected in (
            (topics.ROBOT_FSM_STATE, fsm_state_pb2.FsmState),
            (
                topics.ROBOT_CONFIG_SAVE_STATUS,
                config_file_save_status_pb2.ConfigFileSaveStatus,
            ),
        ):
            with self.subTest(topic=topic):
                message_class, _, delivery = self.bus.subscriptions[topic]
                self.assertIs(message_class, expected)
                self.assertEqual(delivery, operator_bus.operator_topic(topic).delivery)

    def test_a_received_save_status_waits_in_its_mailbox(self):
        status = config_file_save_status_pb2.ConfigFileSaveStatus(
            sequence=7,
            result=config_file_save_status_pb2.ConfigFileSaveStatus.RESULT_SAVED,
        )
        self.bus.deliver(topics.ROBOT_CONFIG_SAVE_STATUS, status)
        self.assertEqual(self.gui_bus.config_save_statuses.take_all(), [status])
        self.assertEqual(self.gui_bus.take_fsm_states(), [])

    def test_a_received_state_waits_in_the_mailbox_for_the_gui(self):
        state = fsm_state_pb2.FsmState(
            mode="WB_MPC", gantry_locked=False, controller_resets=2
        )
        self.bus.deliver(topics.ROBOT_FSM_STATE, state)
        self.assertEqual(self.gui_bus.take_fsm_states(), [state])
        self.assertEqual(self.gui_bus.take_fsm_states(), [])

    def test_each_publisher_publishes_its_topic(self):
        self.gui_bus.walking_velocity_command.publish(
            operator_bus.walking_velocity_command(0.1, 0.0, 0.0, 0.8)
        )
        self.gui_bus.fsm_command.send("JOINT_PD")
        self.gui_bus.mpc_parameters.publish(
            operator_bus.mpc_parameter_update(
                task_file_pb2.TaskFile(), config_path=CONFIG_PATH
            )
        )
        self.gui_bus.pd_gains.publish(_gains(1.0))
        self.gui_bus.joint_targets.publish(operator_bus.joint_targets({"j": 0.1}))
        self.gui_bus.dodgeball_throw.publish(dodgeball_throw_pb2.DodgeballThrow())
        self.gui_bus.config_save.publish(config_file_save_pb2.ConfigFileSave())
        published = {topic: type(message) for topic, message in self.bus.published}
        self.assertEqual(
            published,
            {
                topics.OPERATOR_WALKING_VELOCITY_COMMAND: walking_velocity_command_pb2.WalkingVelocityCommand,
                topics.OPERATOR_FSM_COMMAND: fsm_command_pb2.FsmCommand,
                topics.OPERATOR_MPC_PARAMETERS: mpc_parameter_update_pb2.MpcParameterUpdate,
                topics.OPERATOR_PD_GAINS: joint_pd_gains_file_pb2.JointPdGainsFile,
                topics.OPERATOR_JOINT_TARGETS: joint_targets_pb2.JointTargets,
                topics.OPERATOR_DODGEBALL_THROW: dodgeball_throw_pb2.DodgeballThrow,
                topics.OPERATOR_CONFIG_SAVE: config_file_save_pb2.ConfigFileSave,
            },
        )
        # The table and the bus agree: every topic it publishes is a PUBLISH row, and every PUBLISH row is published.
        self.assertEqual(
            set(published),
            {
                row.topic
                for row in operator_bus.OPERATOR_TOPICS
                if row.direction is operator_bus.Direction.PUBLISH
            },
        )

    def test_start_and_close_reach_the_bus(self):
        with self.gui_bus:
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
        operator = robot_ipc.Bus(
            operator_bus.OPERATOR_NODE, _ephemeral_network(operator_bus.OPERATOR_NODE)
        )
        self.gui_bus = operator_bus.OperatorBus(operator)
        self.addCleanup(self.gui_bus.close)
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
                mpc_parameter_update_pb2.MpcParameterUpdate,
                robot_ipc.Delivery.LATEST,
            ),
            (
                topics.OPERATOR_JOINT_TARGETS,
                joint_targets_pb2.JointTargets,
                robot_ipc.Delivery.LATEST,
            ),
            (
                topics.OPERATOR_CONFIG_SAVE,
                config_file_save_pb2.ConfigFileSave,
                robot_ipc.Delivery.ALL,
            ),
        ):
            self.robot.subscribe(
                topic, message_class, self._collector(topic), delivery=delivery
            )
        self.robot.start()
        self.gui_bus.start()

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

    def test_a_parameter_update_reaches_the_robot_field_for_field(self):
        task = task_file_pb2.TaskFile(contact_estimator="cheater_sim")
        task.state_weights.scaling = 1.5e-3
        update = operator_bus.mpc_parameter_update(task, config_path=CONFIG_PATH)

        def publish() -> None:
            self.gui_bus.mpc_parameters.publish(update)

        received = self._publish_until_received(publish, topics.OPERATOR_MPC_PARAMETERS)
        self.assertEqual(received, update)
        self.assertEqual(received.SerializeToString(), update.SerializeToString())

    def test_joint_targets_reach_the_robot_as_a_map(self):
        targets = {"l_leg_kny": 0.5, "r_leg_kny": -0.125}

        def publish() -> None:
            self.gui_bus.joint_targets.publish(operator_bus.joint_targets(targets))

        received = self._publish_until_received(publish, topics.OPERATOR_JOINT_TARGETS)
        self.assertEqual(dict(received.positions), targets)

    def test_fsm_commands_arrive_in_order_with_rising_sequences(self):
        self._publish_until_received(
            lambda: self.gui_bus.fsm_command.send("ZERO_TORQUE"),
            topics.OPERATOR_FSM_COMMAND,
        )
        for command in ("JOINT_PD", "LOCK_GANTRY", "WB_MPC"):
            self.gui_bus.fsm_command.send(command)

        def last_is_wb_mpc() -> bool:
            return self._received(topics.OPERATOR_FSM_COMMAND)[-1].command == "WB_MPC"

        self.assertTrue(wait_for(last_is_wb_mpc))
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
            taken.extend(self.gui_bus.take_fsm_states())
            return bool(taken)

        self.assertTrue(
            wait_for(arrived), "the FSM state never reached the operator bus"
        )
        self.assertEqual(taken[-1], state)

    def test_a_save_reaches_the_robot_and_its_answer_reaches_the_gui(self):
        save = config_file_save_pb2.ConfigFileSave(
            config_path=CONFIG_PATH, text="terrain_height: 0.0\n", sequence=42
        )
        received = self._publish_until_received(
            lambda: self.gui_bus.config_save.publish(save), topics.OPERATOR_CONFIG_SAVE
        )
        self.assertEqual(received, save)
        status = config_file_save_status_pb2.ConfigFileSaveStatus(
            sequence=received.sequence,
            result=config_file_save_status_pb2.ConfigFileSaveStatus.RESULT_SAVED,
            stored_path="/var/lib/wb-humanoid-robot/config/atlas/mpc/task.textproto",
        )
        taken = []

        def answered():
            self.robot.publish(topics.ROBOT_CONFIG_SAVE_STATUS, status)
            time.sleep(0.01)
            taken.extend(self.gui_bus.config_save_statuses.take_all())
            return bool(taken)

        self.assertTrue(wait_for(answered), "the save status never reached the GUI")
        self.assertEqual(taken[-1], status)


class TestOperatorBusConnect(unittest.TestCase):
    def test_a_node_that_is_not_in_the_network_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "network.textproto")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write('nodes { name: "robot" host: "127.0.0.1" port: 5600 }\n')
            with self.assertRaisesRegex(ValueError, "operator"):
                operator_bus.OperatorBus.connect(path)

    def test_a_missing_network_file_is_an_error_that_names_it(self):
        with self.assertRaisesRegex(FileNotFoundError, "no_such_network.textproto"):
            operator_bus.OperatorBus.connect(
                "/does/not/exist/no_such_network.textproto"
            )


if __name__ == "__main__":
    unittest.main()
