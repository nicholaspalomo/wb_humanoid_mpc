"""****************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
****************************************************************************"""

"""The remote control's side of the IPC bus: which topics it uses, with which message, and the objects that do it.

The topics and their messages are the contract of humanoid_nmpc/docs/distributed_runtime/README.md ("Topics"); the
names are the constants of humanoid_mpc_ipc/topics.py. OPERATOR_TOPICS lists every topic the GUI and the teleoperation
publishers touch, and test/test_operator_topics.py checks each row against the README's table.

Free of Tk, so that the tests exercise it headless:

- TopicPublisher binds one topic of a bus to its message class; it is the `publisher` a GUI tab is given, and it
  refuses a message of another type instead of sending what the receiver would reject.
- FsmCommandSender numbers the FSM commands (FsmCommand.sequence).
- FsmStateMailbox hands the FsmState messages the bus's receive thread delivers to the Tk thread, which polls it: Tk is
  only ever called from its own thread.
- OperatorBus is the GUI's node on the bus: one publisher per operator topic and the robot/fsm_state subscription.

Nothing here calls ZeroMQ directly; robot_ipc.Bus does, on its own receive thread. Bus.publish() never waits for the
network, so publishing from a Tk callback cannot stall the GUI.
"""

import collections
import dataclasses
import enum
import threading
import time
from typing import Any, Callable, Deque, List, Mapping, Optional, Tuple, Type

from google.protobuf import message as protobuf_message

import robot_ipc
from humanoid_mpc_ipc import topics
from humanoid_mpc_msgs import fsm_command_pb2
from humanoid_mpc_msgs import fsm_state_pb2
from humanoid_mpc_msgs import joint_targets_pb2
from humanoid_mpc_msgs import walking_velocity_command_pb2
from humanoid_mpc_msgs import yaml_document_pb2

# The node names this package publishes as; both have to be nodes of the network file.
# LINT.IfChange(operator_nodes)
OPERATOR_NODE = "operator"
TELEOP_NODE = "teleop"
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
    message_class: Type[protobuf_message.Message]
    direction: Direction
    delivery: robot_ipc.Delivery


# LINT.IfChange(operator_topics)
OPERATOR_TOPICS: Tuple[OperatorTopic, ...] = (
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
        yaml_document_pb2.YamlDocument,
        Direction.PUBLISH,
        robot_ipc.Delivery.LATEST,
    ),
    OperatorTopic(
        topics.OPERATOR_PD_GAINS,
        yaml_document_pb2.YamlDocument,
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
        yaml_document_pb2.YamlDocument,
        Direction.PUBLISH,
        robot_ipc.Delivery.ALL,
    ),
    OperatorTopic(
        topics.ROBOT_FSM_STATE,
        fsm_state_pb2.FsmState,
        Direction.SUBSCRIBE,
        robot_ipc.Delivery.LATEST,
    ),
)
# LINT.ThenChange(//humanoid_nmpc/docs/distributed_runtime/README.md:topic_table)


def operator_topic(topic: str) -> OperatorTopic:
    """The row of OPERATOR_TOPICS of that topic.

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
        message_class: Type[protobuf_message.Message],
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
    def message_class(self) -> Type[protobuf_message.Message]:
        return self._message_class

    def publish(self, message: protobuf_message.Message) -> bool:
        """Publishes `message` on the topic; never waits for the network.

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


class FsmStateMailbox:
    """Carries FsmState messages from the bus's receive thread to the Tk thread, in the order they arrived.

    put() is the bus callback; take_all() is called from the GUI's timer. Holding at most `capacity` messages, it
    drops the oldest when the GUI does not keep up: the newest state is the one the GUI must show.
    """

    def __init__(self, capacity: int = 64) -> None:
        if capacity <= 0:
            raise ValueError("the mailbox needs a positive capacity")
        self._lock = threading.Lock()
        self._messages: Deque[fsm_state_pb2.FsmState] = collections.deque(
            maxlen=capacity
        )
        self._dropped = 0

    def put(self, message: fsm_state_pb2.FsmState) -> None:
        with self._lock:
            if len(self._messages) == self._messages.maxlen:
                self._dropped += 1
            self._messages.append(message)

    def take_all(self) -> List[fsm_state_pb2.FsmState]:
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


class OperatorBus:
    """The remote control's node on the bus: a publisher for every operator topic, and the robot's FSM state.

    Args:
        bus: a robot_ipc.Bus that publishes as the operator node and has not started yet (OperatorBus subscribes on
            it), or a stand-in with the same publish/subscribe/start/close.
    """

    def __init__(self, bus: Any) -> None:
        self._bus = bus
        self._fsm_states = FsmStateMailbox()
        state = operator_topic(topics.ROBOT_FSM_STATE)
        bus.subscribe(
            state.topic,
            state.message_class,
            self._fsm_states.put,
            delivery=state.delivery,
        )

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

    @classmethod
    def connect(
        cls, network_config: str, node_name: str = OPERATOR_NODE
    ) -> "OperatorBus":
        """The operator bus of the network file at `network_config`, publishing as `node_name` (not started).

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

    @property
    def fsm_states(self) -> FsmStateMailbox:
        return self._fsm_states

    def take_fsm_states(self) -> List[fsm_state_pb2.FsmState]:
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


def yaml_document(text: str) -> yaml_document_pb2.YamlDocument:
    """The YamlDocument that carries `text` unchanged."""
    return yaml_document_pb2.YamlDocument(yaml=text)


def joint_targets(
    positions: Optional[Mapping[str, float]] = None,
) -> joint_targets_pb2.JointTargets:
    """The JointTargets of a {joint name: position [rad]} map."""
    message = joint_targets_pb2.JointTargets()
    for name, position in (positions or {}).items():
        message.positions[str(name)] = float(position)
    return message
