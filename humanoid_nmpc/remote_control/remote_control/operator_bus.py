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

"""The remote control's side of the IPC bus: which topics it uses, with which message, and the objects that do it.

The topics and their messages are the contract of humanoid_nmpc/docs/distributed_runtime/README.md ("Topics"); the
names are the constants of humanoid_mpc_ipc/topics.py. OPERATOR_TOPICS lists every topic the GUI and the teleoperation
publishers touch, and test/test_operator_topics.py checks each row against the README's table.

The tuning tabs publish typed messages, never text: the MPC parameters are a humanoid_mpc_config.MpcParameterUpdate
(the whole task file and contact-planning file, parsed strictly before they leave, with the task file's identity,
config_path), the PD gains a humanoid_mpc_config.JointPdGainsFile, a dodgeball throw a humanoid_mpc_msgs.DodgeballThrow.
A Save sends the text of the saved file to the robot's store as a humanoid_mpc_msgs.ConfigFileSave
(robot_config_save.py), which the robot answers with a ConfigFileSaveStatus. A receiver built from another commit that
still subscribes with another type drops the message and says so (the bus checks the type name); one with the same
type refuses a payload of another schema version: a field it does not know, or, for the MPC parameters and the saves,
a schema fingerprint other than its own.

Free of Tk, so that the tests exercise it headless:

- TopicPublisher binds one topic of a bus to its message class; it is the `publisher` a GUI tab is given, and it
  refuses a message of another type instead of sending what the receiver would reject.
- FsmCommandSender numbers the FSM commands (FsmCommand.sequence).
- FsmStateMailbox and ConfigSaveStatusMailbox hand the messages the bus's receive thread delivers to the Tk thread,
  which polls them: Tk is only ever called from its own thread.
- OperatorBus is the GUI's node on the bus: one publisher per operator topic, and the robot/fsm_state and
  robot/config_save_status subscriptions.

Nothing here calls ZeroMQ directly; robot_ipc.Bus does, on its own receive thread. Bus.publish() never waits for the
network, so publishing from a Tk callback cannot stall the GUI.
"""

import collections
from collections.abc import Callable, Mapping
import dataclasses
import enum
import threading
import time
from typing import Any, Generic, TypeVar

from google.protobuf import message as protobuf_message
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
import robot_ipc

# The node names this package publishes as; each has to be a node of the network file. The GUI publishes as the
# operator, the teleoperation publishers as teleop, and push_robot_config, which runs next to a GUI, as config_push.
# LINT.IfChange(operator_nodes)
OPERATOR_NODE = "operator"
TELEOP_NODE = "teleop"
CONFIG_PUSH_NODE = "config_push"
# LINT.ThenChange(//config/ipc/network.textproto:localhost_nodes)

# The network file the processes of one machine share (robot_runtime/robot_ipc, "The network file"); a relative path
# is resolved by config_files.resolve_input_path().
DEFAULT_NETWORK_CONFIG = "config/ipc/network.textproto"


class Direction(str, enum.Enum):
    """Whether the remote control publishes a topic or subscribes to it."""

    PUBLISH = "publish"
    SUBSCRIBE = "subscribe"


@dataclasses.dataclass(frozen=True)
class OperatorTopic:
    """One row of the README's topic table, as the remote control uses it.

    Attributes:
        topic: the topic, a constant of humanoid_mpc_ipc/topics.py.
        message_class: the generated class of the message the topic carries.
        direction: whether the remote control publishes or subscribes.
        delivery: the README's delivery of the topic: what a subscriber is handed of it.
    """

    topic: str
    message_class: type[protobuf_message.Message]
    direction: Direction
    delivery: robot_ipc.Delivery


# LINT.IfChange(operator_topics)
OPERATOR_TOPICS: tuple[OperatorTopic, ...] = (
    OperatorTopic(
        topics.OPERATOR_WALKING_VELOCITY_COMMAND,
        walking_velocity_command_pb2.WalkingVelocityCommand,
        Direction.PUBLISH,
        robot_ipc.Delivery.LATEST,
    ),
    OperatorTopic(
        topics.OPERATOR_FSM_COMMAND,
        fsm_command_pb2.FsmCommand,
        Direction.PUBLISH,
        robot_ipc.Delivery.ALL,
    ),
    OperatorTopic(
        topics.OPERATOR_MPC_PARAMETERS,
        mpc_parameter_update_pb2.MpcParameterUpdate,
        Direction.PUBLISH,
        robot_ipc.Delivery.LATEST,
    ),
    OperatorTopic(
        topics.OPERATOR_PD_GAINS,
        joint_pd_gains_file_pb2.JointPdGainsFile,
        Direction.PUBLISH,
        robot_ipc.Delivery.LATEST,
    ),
    OperatorTopic(
        topics.OPERATOR_JOINT_TARGETS,
        joint_targets_pb2.JointTargets,
        Direction.PUBLISH,
        robot_ipc.Delivery.LATEST,
    ),
    OperatorTopic(
        topics.OPERATOR_DODGEBALL_THROW,
        dodgeball_throw_pb2.DodgeballThrow,
        Direction.PUBLISH,
        robot_ipc.Delivery.ALL,
    ),
    OperatorTopic(
        topics.OPERATOR_CONFIG_SAVE,
        config_file_save_pb2.ConfigFileSave,
        Direction.PUBLISH,
        robot_ipc.Delivery.ALL,
    ),
    OperatorTopic(
        topics.ROBOT_FSM_STATE,
        fsm_state_pb2.FsmState,
        Direction.SUBSCRIBE,
        robot_ipc.Delivery.LATEST,
    ),
    OperatorTopic(
        topics.ROBOT_CONFIG_SAVE_STATUS,
        config_file_save_status_pb2.ConfigFileSaveStatus,
        Direction.SUBSCRIBE,
        robot_ipc.Delivery.ALL,
    ),
)
# LINT.ThenChange(//humanoid_nmpc/docs/distributed_runtime/README.md:topic_table)


def operator_topic(topic: str) -> OperatorTopic:
    """The row of OPERATOR_TOPICS of that topic.

    Args:
        topic: the topic, a constant of humanoid_mpc_ipc/topics.py.

    Returns:
        The row whose topic is `topic`.

    Raises:
        ValueError: the remote control does not use the topic; the message lists the ones it does.
    """
    for row in OPERATOR_TOPICS:
        if row.topic == topic:
            return row
    known = ", ".join(row.topic for row in OPERATOR_TOPICS)
    raise ValueError(
        f"the remote control does not use topic '{topic}' (it uses: {known})"
    )


class TopicPublisher:
    """Publishes one message type on one topic of a bus.

    This is the `publisher` the GUI's tabs are given: anything with publish(message). It checks the type, so that a tab
    handing it the wrong message fails where the mistake is rather than on the receiver, which would drop the message
    as carrying an unexpected type.

    Args:
        bus: anything with robot_ipc.Bus's publish(topic, message); the tests pass a bus that records.
        topic: the topic.
        message_class: the generated class of the messages the topic carries.
    """

    def __init__(
        self,
        bus: Any,
        topic: str,
        message_class: type[protobuf_message.Message],
    ) -> None:
        if not topic:
            raise ValueError("the topic is empty")
        if getattr(message_class, "DESCRIPTOR", None) is None:
            raise ValueError(f"{message_class!r} is not a generated protobuf message")
        self._bus = bus
        self._topic = topic
        self._message_class = message_class

    @classmethod
    def for_topic(cls, bus: Any, topic: str) -> "TopicPublisher":
        """The publisher of one of the topics the remote control publishes, with the message class of OPERATOR_TOPICS.

        Args:
            bus: anything with robot_ipc.Bus's publish(topic, message).
            topic: the topic, one the remote control publishes.

        Returns:
            The publisher of `topic` on `bus`.

        Raises:
            ValueError: the remote control does not publish the topic.
        """
        row = operator_topic(topic)
        if row.direction is not Direction.PUBLISH:
            raise ValueError(
                f"the remote control subscribes to '{topic}'; it does not publish it"
            )
        return cls(bus, row.topic, row.message_class)

    @property
    def bus(self) -> Any:
        return self._bus

    @property
    def topic(self) -> str:
        return self._topic

    @property
    def message_class(self) -> type[protobuf_message.Message]:
        return self._message_class

    def publish(self, message: protobuf_message.Message) -> bool:
        """Publishes `message` on the topic; never waits for the network.

        Args:
            message: the message, of the topic's message class.

        Returns:
            False when the bus dropped the message because its queue was full (robot_ipc.Bus.publish).

        Raises:
            TypeError: the message is not of the topic's type.
        """
        if not isinstance(message, self._message_class):
            raise TypeError(
                f"topic '{self._topic}' carries {self._message_class.DESCRIPTOR.full_name}, "
                f"not {type(message).__name__}"
            )
        return self._bus.publish(self._topic, message)


class FsmCommandSender:
    """Publishes FSM commands, each with a sequence number above every earlier one.

    FsmCommand.sequence lets the robot tell a repeated message from a new command. The first number is the wall clock
    in nanoseconds, so that the commands of a restarted GUI still count up from those of the run before it; after that
    the number goes up by one per command.

    Args:
        publisher: the publisher of operator/fsm_command.
        clock_ns: the clock of the first sequence number (time.time_ns); the tests pass a fixed one.
    """

    def __init__(
        self,
        publisher: TopicPublisher,
        clock_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        if publisher.message_class is not fsm_command_pb2.FsmCommand:
            raise ValueError(
                f"the publisher of '{publisher.topic}' does not carry FsmCommand"
            )
        self._publisher = publisher
        self._lock = threading.Lock()
        # The sequence of the last command sent; the first one is one above the clock.
        self._sequence = max(0, int(clock_ns()))

    @property
    def last_sequence(self) -> int:
        """The sequence number of the last command sent (before the first one: the one below it)."""
        with self._lock:
            return self._sequence

    def send(self, command: str) -> fsm_command_pb2.FsmCommand:
        """Publishes one command (a control mode name, LOCK_GANTRY or UNLOCK_GANTRY) and returns the message.

        Args:
            command: a control mode name, LOCK_GANTRY or UNLOCK_GANTRY.

        Returns:
            The message published, with its sequence number.

        Raises:
            ValueError: the command is empty.
        """
        if not command:
            raise ValueError("the FSM command is empty")
        with self._lock:
            self._sequence += 1
            message = fsm_command_pb2.FsmCommand(
                command=command, sequence=self._sequence
            )
        self._publisher.publish(message)
        return message


# The message a mailbox carries.
_MessageT = TypeVar("_MessageT", bound=protobuf_message.Message)


class Mailbox(Generic[_MessageT]):
    """Carries the messages of one subscription from the bus's receive thread to the Tk thread, in the order they arrived.

    put() is the bus callback; take_all() is called from the GUI's timer. Holding at most `capacity` messages, it
    drops the oldest when the GUI does not keep up. Thread-safe.

    Args:
        capacity: How many messages it holds; positive.
    """

    def __init__(self, capacity: int = 64) -> None:
        if capacity <= 0:
            raise ValueError("the mailbox needs a positive capacity")
        self._lock = threading.Lock()
        self._messages: collections.deque[_MessageT] = collections.deque(
            maxlen=capacity
        )
        self._dropped = 0

    def put(self, message: _MessageT) -> None:
        with self._lock:
            if len(self._messages) == self._messages.maxlen:
                self._dropped += 1
            self._messages.append(message)

    def take_all(self) -> list[_MessageT]:
        """Every message put since the last call, oldest first."""
        with self._lock:
            messages = list(self._messages)
            self._messages.clear()
        return messages

    @property
    def dropped(self) -> int:
        """Messages dropped because the mailbox was full."""
        with self._lock:
            return self._dropped


class FsmStateMailbox(Mailbox[fsm_state_pb2.FsmState]):
    """The robot's FsmState messages for the Tk thread; when it is full, the newest state is the one the GUI must show."""


class ConfigSaveStatusMailbox(
    Mailbox[config_file_save_status_pb2.ConfigFileSaveStatus]
):
    """The robot's answers to the saves (robot/config_save_status) for robot_config_save.RobotConfigSaver.poll()."""


class OperatorBus:
    """The remote control's node on the bus: a publisher for every operator topic, the robot's FSM state and its answers.

    Args:
        bus: a robot_ipc.Bus that publishes as the operator node and has not started yet (OperatorBus subscribes on
            it), or a stand-in with the same publish/subscribe/start/close.

    Attributes:
        config_save_statuses: The robot's answers to the saves (robot/config_save_status), for
            robot_config_save.RobotConfigSaver.
        walking_velocity_command: The publisher of operator/walking_velocity_command.
        fsm_command: The sender of operator/fsm_command.
        mpc_parameters: The publisher of operator/mpc_parameters.
        pd_gains: The publisher of operator/pd_gains.
        joint_targets: The publisher of operator/joint_targets.
        dodgeball_throw: The publisher of operator/dodgeball_throw.
        config_save: The publisher of operator/config_save.
    """

    def __init__(self, bus: Any) -> None:
        self._bus = bus
        self._fsm_states = FsmStateMailbox()
        self.config_save_statuses = ConfigSaveStatusMailbox()
        for topic, put in (
            (topics.ROBOT_FSM_STATE, self._fsm_states.put),
            (topics.ROBOT_CONFIG_SAVE_STATUS, self.config_save_statuses.put),
        ):
            row = operator_topic(topic)
            bus.subscribe(row.topic, row.message_class, put, delivery=row.delivery)

        self.walking_velocity_command = TopicPublisher.for_topic(
            bus, topics.OPERATOR_WALKING_VELOCITY_COMMAND
        )
        self.fsm_command = FsmCommandSender(
            TopicPublisher.for_topic(bus, topics.OPERATOR_FSM_COMMAND)
        )
        self.mpc_parameters = TopicPublisher.for_topic(
            bus, topics.OPERATOR_MPC_PARAMETERS
        )
        self.pd_gains = TopicPublisher.for_topic(bus, topics.OPERATOR_PD_GAINS)
        self.joint_targets = TopicPublisher.for_topic(
            bus, topics.OPERATOR_JOINT_TARGETS
        )
        self.dodgeball_throw = TopicPublisher.for_topic(
            bus, topics.OPERATOR_DODGEBALL_THROW
        )
        self.config_save = TopicPublisher.for_topic(bus, topics.OPERATOR_CONFIG_SAVE)

    @classmethod
    def connect(
        cls, network_config: str, node_name: str = OPERATOR_NODE
    ) -> "OperatorBus":
        """The operator bus of the network file at `network_config`, publishing as `node_name` (not started).

        Args:
            network_config: the path of the network file.
            node_name: the node of the network file the GUI publishes as.

        Returns:
            The operator bus, not started yet.

        Raises:
            OSError: the file cannot be read (FileNotFoundError: it does not exist).
            robot_ipc.NetworkConfigError: the file is invalid; the message names the line and column, or the node.
            ValueError: the node is not in the network.
            robot_ipc.BusError: the node's endpoint cannot be bound (another process publishes as it).
        """
        network = robot_ipc.load_network_config(network_config)
        return cls(robot_ipc.Bus(node_name, network))

    @property
    def bus(self) -> Any:
        return self._bus

    def take_fsm_states(self) -> list[fsm_state_pb2.FsmState]:
        """The FsmState messages received since the last call, oldest first (for the Tk thread)."""
        return self._fsm_states.take_all()

    def start(self) -> None:
        self._bus.start()

    def close(self) -> None:
        self._bus.close()

    def __enter__(self) -> "OperatorBus":
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


def walking_velocity_command(
    linear_velocity_x: float,
    linear_velocity_y: float,
    angular_velocity_z: float,
    desired_pelvis_height: float,
) -> walking_velocity_command_pb2.WalkingVelocityCommand:
    """The WalkingVelocityCommand of normalized velocities in [-1, 1] and an absolute pelvis height [m]."""
    return walking_velocity_command_pb2.WalkingVelocityCommand(
        linear_velocity_x=float(linear_velocity_x),
        linear_velocity_y=float(linear_velocity_y),
        angular_velocity_z=float(angular_velocity_z),
        desired_pelvis_height=float(desired_pelvis_height),
    )


def mpc_parameter_update(
    task: task_file_pb2.TaskFile,
    contact_planning: contact_planning_file_pb2.ContactPlanningFile | None = None,
    *,
    config_path: str,
) -> mpc_parameter_update_pb2.MpcParameterUpdate:
    """The MpcParameterUpdate of a task file and, for a robot with a contact planner, its contact-planning file.

    The update is the files (humanoid_mpc_config/README.md, "Live updates"): a block it does not carry means its
    defaults, as at start-up, so the tab hands over the whole of each file. It carries the fingerprint of this build's
    schema (nproto_schema.schema_fingerprint()), which a receiver built from another version refuses, and the task
    file's identity, which a receiver that runs another configuration refuses (a GUI opened on the centroidal G1's file
    cannot retune the whole-body G1).

    Args:
        task: The task file, with the operator's values.
        contact_planning: The contact-planning file next to it, with the operator's values; None for a robot without
            one, whose update carries none.
        config_path: The task file's identity, its path from robot_models/ on (robot_config_save.config_path_of()).

    Returns:
        The update.
    """
    update = mpc_parameter_update_pb2.MpcParameterUpdate()
    update.task.CopyFrom(task)
    if contact_planning is not None:
        update.contact_planning.CopyFrom(contact_planning)
    update.config_path = config_path
    update.schema_fingerprint = nproto_schema.schema_fingerprint(update.DESCRIPTOR)
    return update


def joint_targets(
    positions: Mapping[str, float] | None = None,
) -> joint_targets_pb2.JointTargets:
    """The JointTargets of a {joint name: position [rad]} map."""
    message = joint_targets_pb2.JointTargets()
    for name, position in (positions or {}).items():
        message.positions[str(name)] = float(position)
    return message
