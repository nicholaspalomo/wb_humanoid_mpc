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


"""The DDS bus of the distributed runtime, in Python: the twin of robot_ipc/Bus.h.

Every bus topic is a DDS topic of the same name whose samples are one robot_ipc::Envelope (envelope.py): the wire
protocol's version, the writer's sequence number, the full protobuf type name and the serialized message, exactly as
the C++ bus writes them. A sample of another protocol version is refused and counted as rejected.

A bus is one Cyclone DDS participant of the network's domain, configured from the network file (domain_config.py):
unicast discovery to the host of every node, the QoS of each topic by its named profile (qos_profile.py). A named bus
creates the writers of the topics the file lists its node as a publisher of when it is constructed.

One drain takes up to max_messages_per_drain samples across the topics and dispatches them in the order they were
written (their source timestamps, which every bus makes strictly increasing across its topics): every sample of an ALL
subscription, and the newest one of a LATEST subscription at the place it was written.

The DDS readers and writers belong to the bus's receive thread, which start() launches. Callbacks and periodic
callbacks run on it, one at a time. publish() may be called from any thread: it serializes on the caller and queues
the sample for the receive thread, which a guard condition wakes at once; only the receive thread writes to DDS.

    network = robot_ipc.load_network_config("config/ipc/network.textproto")
    with robot_ipc.Bus("operator", network) as bus:
        ...  # subscribe() before entering, publish() inside
"""

import collections
from collections.abc import Callable
import dataclasses
import enum
import logging
import math
import threading
import time
from typing import Any

import cyclonedds.builtin
import cyclonedds.core
import cyclonedds.domain
import cyclonedds.pub
import cyclonedds.qos
import cyclonedds.sub
import cyclonedds.topic
from google.protobuf import message as protobuf_message

from robot_ipc import domain_config
from robot_ipc import envelope
from robot_ipc import network_config
from robot_ipc import qos_profile

_LOGGER = logging.getLogger("robot_ipc")

# The receive thread logs a recurring problem at most this often, in seconds.
_LOG_PERIOD = 5.0
# The name of the DDS participant of a bus that only subscribes.
_SUBSCRIBER_PARTICIPANT_NAME = "robot_ipc_subscriber"
# A reliable write never waits for acknowledgments: at the writer's unacknowledged-bytes limit it fails at once and is
# counted as send_dropped.
_RELIABLE_WRITE_BLOCKING_TIME_NS = 0
_NANOSECONDS_PER_SECOND = 1_000_000_000
_ANY_STATE = (
    cyclonedds.core.SampleState.Any
    | cyclonedds.core.ViewState.Any
    | cyclonedds.core.InstanceState.Any
)

WIRE_PROTOCOL_VERSION = envelope.WIRE_PROTOCOL_VERSION


# LINT.IfChange(delivery_names)
class Delivery(str, enum.Enum):
    """What a subscription is handed of a topic each time the receive thread drains it."""

    # Only the newest sample of the topic; the older ones of the same drain are dropped unparsed (superseded).
    LATEST = "latest"
    # Every sample, in the order it was written.
    ALL = "all"


# LINT.ThenChange(//robot_runtime/robot_ipc/src/Delivery.cpp:delivery_names)


def parse_delivery(value: str | Delivery) -> Delivery:
    """The delivery of that name ("latest" or "all").

    Args:
        value: The name of a delivery, or a Delivery, which is returned as it is.

    Returns:
        The Delivery that `value` names.

    Raises:
        ValueError: an unknown name; the message lists the valid ones.
    """
    if isinstance(value, Delivery):
        return value
    try:
        return Delivery(value)
    except ValueError:
        valid = ", ".join(delivery.value for delivery in Delivery)
        raise ValueError(f"unknown delivery '{value}' (valid: {valid})") from None


# LINT.IfChange(topic_statistics)
@dataclasses.dataclass
class TopicStatistics:
    """What a bus did with one topic; see robot_ipc/TopicStatistics.h.

    Once a drain has dispatched, received == delivered + superseded + rejected. The last five fields are DDS's own
    account, refreshed by the receive thread at most every io_poll_period.

    Attributes:
        written: samples handed to the topic's DDS writer, not delivered ones.
        send_dropped: publish() calls refused because the queue to the receive thread was full, writes DDS refused,
            and queued samples close() discarded because the bus was not running to send them.
        received: samples taken from the topic's DDS reader.
        delivered: samples handed to the callback (including those whose callback then raised).
        superseded: LATEST only: samples replaced by a newer one of the same drain before they were parsed.
        rejected: samples of another wire protocol version, another type than the subscription's, or that did not
            parse.
        handler_errors: callback calls that raised; the bus logs and counts them and carries on.
        missed: samples a matched writer wrote that the reader never took (gaps in its sequence numbers).
        sample_lost: DDS SAMPLE_LOST of the reader.
        sample_rejected: DDS SAMPLE_REJECTED of the reader.
        incompatible_qos: DDS REQUESTED_ and OFFERED_INCOMPATIBLE_QOS of the topic's reader and writer.
        matched_readers: the DDS readers this bus's writer of the topic currently matches.
        matched_writers: the DDS writers this bus's reader of the topic currently matches.
    """

    written: int = 0
    send_dropped: int = 0
    received: int = 0
    delivered: int = 0
    superseded: int = 0
    rejected: int = 0
    handler_errors: int = 0
    missed: int = 0
    sample_lost: int = 0
    sample_rejected: int = 0
    incompatible_qos: int = 0
    matched_readers: int = 0
    matched_writers: int = 0


# LINT.ThenChange(//robot_runtime/robot_ipc/include/robot_ipc/TopicStatistics.h:topic_statistics)


class BusError(RuntimeError):
    """The bus cannot do what was asked in its current state, or cannot join its DDS domain."""


MessageCallback = Callable[[protobuf_message.Message], None]
RawCallback = Callable[[str, bytes], None]
TopicCallback = Callable[[str, str, bytes], None]


class _DomainRegistry:
    """The Cyclone DDS domains of this process, shared by its buses (robot_ipc/src/DomainRegistry.h).

    A process configures a domain once, so the buses of one domain share it: the first creates it, the last to release
    it lets it go. Thread-safe.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # domain id -> [the Domain, its configuration, its users].
        self._domains: dict[int, list[Any]] = {}

    def acquire(self, domain_id: int, config: str) -> None:
        """Creates the domain with `config`, or counts one more user of it; another configuration is refused.

        Args:
            domain_id: The DDS domain.
            config: Its Cyclone DDS configuration.

        Raises:
            BusError: the domain exists with another configuration, or DDS cannot create it.
        """
        with self._lock:
            entry = self._domains.get(domain_id)
            if entry is not None:
                if entry[1] != config:
                    raise BusError(
                        f"DDS domain {domain_id} is configured differently by another bus of this process (a bus of "
                        "another node or network); use one network file and node per process"
                    )
                entry[2] += 1
                return
            try:
                created = cyclonedds.domain.Domain(domain_id, config)
            except cyclonedds.core.DDSException as error:
                raise BusError(
                    f"cannot create DDS domain {domain_id}: {error} (configuration: {config})"
                ) from error
            self._domains[domain_id] = [created, config, 1]

    def release(self, domain_id: int) -> None:
        """Releases one use of the domain; the last release deletes it."""
        with self._lock:
            entry = self._domains.get(domain_id)
            if entry is None:
                return
            entry[2] -= 1
            if entry[2] == 0:
                del self._domains[domain_id]


# Process-wide, like the DDS domains it tracks: a domain is configured once per process.
_DOMAINS = _DomainRegistry()


@dataclasses.dataclass
class _Subscription:
    """One subscription of the bus and its DDS reader."""

    topic: str
    delivery: Delivery
    callback: Callable[..., None]
    # None for subscribe_raw(), which takes any type.
    message_class: type[protobuf_message.Message] | None
    expected_type: str
    statistics: TopicStatistics
    reader: Any = None
    # The last sequence number of each writer (its publication handle) this reader took a sample of.
    last_sequence: dict[int, int] = dataclasses.field(default_factory=dict)
    # LATEST: the index, in the drain's pending samples, of the newest sample of this subscription so far.
    latest: int | None = None


@dataclasses.dataclass
class _Writer:
    """The DDS writer of one topic, and the sequence number of its last sample."""

    writer: Any
    statistics: TopicStatistics
    last_sequence: int = 0


@dataclasses.dataclass
class _PendingMessage:
    """A sample a drain has taken and dispatches once it has taken the rest, in the order the samples were written."""

    source_timestamp: int
    # None for the subscription to every topic.
    subscription: _Subscription | None
    topic: str
    sample: envelope.Envelope
    # LATEST: a newer sample of the topic is in the same drain; this one is not dispatched.
    superseded: bool = False


@dataclasses.dataclass
class _PeriodicCallback:
    """A callback the receive thread calls every period."""

    period: float
    callback: Callable[[], None]
    next_deadline: float = 0.0


class _RateLimitedLog:
    """Logs each kind of problem at most once per _LOG_PERIOD."""

    def __init__(self) -> None:
        self._last: dict[str, float] = {}

    def warning(self, kind: str, text: str, exc_info: bool = False) -> None:
        now = time.monotonic()
        if now - self._last.get(kind, -math.inf) >= _LOG_PERIOD:
            self._last[kind] = now
            _LOGGER.warning("robot_ipc: %s", text, exc_info=exc_info)


def _qos(
    profile: qos_profile.QosProfile, depth: int, *policies: Any
) -> cyclonedds.qos.Qos:
    """The QoS of a reader or writer of `profile` with a KEEP_LAST history of `depth`, plus `policies`."""
    if qos_profile.qos_policy(profile).reliability is qos_profile.Reliability.RELIABLE:
        reliability = cyclonedds.qos.Policy.Reliability.Reliable(
            max_blocking_time=_RELIABLE_WRITE_BLOCKING_TIME_NS
        )
    else:
        reliability = cyclonedds.qos.Policy.Reliability.BestEffort
    return cyclonedds.qos.Qos(
        reliability,
        cyclonedds.qos.Policy.History.KeepLast(depth),
        cyclonedds.qos.Policy.Durability.Volatile,
        *policies,
    )


class Bus:
    """A publish/subscribe bus over Cyclone DDS, wire-compatible with the C++ robot::ipc::Bus.

    Args:
        node_name: the node this process publishes as. Empty: only subscribe.
        network: the DDS network: domain, discovery peers, topic QoS and transport.
        io_poll_period: the longest the receive thread waits, in seconds; publish(), stop(), arriving samples and
            due periodic callbacks all wake it at once.
        publish_queue_capacity: messages publish() queues before it refuses (counted as send_dropped).
        max_messages_per_drain: the most samples one drain takes, across topics, before the thread dispatches.
        latest_delivery_history_depth: the KEEP_LAST depth of the reader of a LATEST subscription (more than one, so
            that a sample of the wrong type does not displace a valid one before the drain sees both).
        all_delivery_history_depth: the KEEP_LAST depth of the reader of an ALL subscription.

    Raises:
        network_config.NetworkConfigError: the network is invalid.
        ValueError: the node is not in the network, or an option is out of range.
        domain_config.InterfaceError: the node's host is an address of another machine.
        BusError: the process's buses disagree about the domain's configuration, or DDS fails.
    """

    # LINT.IfChange(bus_defaults)
    def __init__(
        self,
        node_name: str,
        network: network_config.NetworkConfig,
        *,
        io_poll_period: float = 0.1,
        publish_queue_capacity: int = 1024,
        max_messages_per_drain: int = 4096,
        latest_delivery_history_depth: int = 4,
        all_delivery_history_depth: int = 256,
    ) -> None:
        # LINT.ThenChange(//robot_runtime/robot_ipc/include/robot_ipc/BusOptions.h:bus_defaults)
        network_config.validate_network_config(network)
        if node_name and network.find(node_name) is None:
            raise ValueError(
                f"node '{node_name}' is not in the network (nodes: {', '.join(network.node_names())})"
            )
        # The float check is negated, so that NaN fails it too (AGENTS.md, Python: negated float comparisons).
        if not io_poll_period > 0:
            raise ValueError("io_poll_period must be positive")
        if publish_queue_capacity <= 0 or max_messages_per_drain <= 0:
            raise ValueError(
                "publish_queue_capacity and max_messages_per_drain must be positive"
            )
        if latest_delivery_history_depth <= 0 or all_delivery_history_depth <= 0:
            raise ValueError(
                "latest_delivery_history_depth and all_delivery_history_depth must be positive"
            )

        self._node_name = node_name
        self._network = network
        self._io_poll_period = io_poll_period
        self._publish_queue_capacity = publish_queue_capacity
        self._max_messages_per_drain = max_messages_per_drain
        self._latest_delivery_history_depth = latest_delivery_history_depth
        self._all_delivery_history_depth = all_delivery_history_depth

        self._lifecycle_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._io_thread_ident: int | None = None
        self._stop_requested = threading.Event()
        self._closed = False

        self._subscriptions: dict[str, _Subscription] = {}
        self._all_topics_callback: TopicCallback | None = None
        # The subscription to every topic: the publications discovered so far, and a reader of each Envelope topic.
        self._publications_reader: Any = None
        self._all_topics_readers: dict[str, Any] = {}
        self._periodic_callbacks: list[_PeriodicCallback] = []
        self._periodic_callback_errors = 0

        self._queue_lock = threading.Lock()
        self._queue: collections.deque[tuple[str, str, bytes]] = collections.deque()
        self._statistics_lock = threading.Lock()
        self._statistics: dict[str, TopicStatistics] = {}
        self._log = _RateLimitedLog()
        self._last_source_timestamp = 0
        self._next_status_refresh = 0.0

        self._domain_config = domain_config.cyclone_domain_config(network, node_name)
        _DOMAINS.acquire(network.domain_id, self._domain_config)
        self._topics: dict[str, Any] = {}
        self._writers: dict[str, _Writer] = {}
        # Set below; None once close() released them.
        self._participant: Any = None
        self._waitset: Any = None
        self._wake_condition: Any = None
        try:
            self._participant = cyclonedds.domain.DomainParticipant(
                network.domain_id,
                qos=cyclonedds.qos.Qos(
                    cyclonedds.qos.Policy.EntityName(
                        node_name or _SUBSCRIBER_PARTICIPANT_NAME
                    )
                ),
            )
            self._waitset = cyclonedds.core.WaitSet(self._participant)
            self._wake_condition = cyclonedds.core.GuardCondition(self._participant)
            self._waitset.attach(self._wake_condition)
            # The writers of the topics this node publishes exist from the start, so that the subscribers have
            # discovered them before the first sample (a VOLATILE reader gets only what is written after it matched).
            for name in network.topics_published_by(node_name) if node_name else []:
                self._writer_for(name)
        except cyclonedds.core.DDSException as error:
            self._release_dds()
            raise BusError(f"cannot set up the DDS entities: {error}") from error
        except BaseException:
            self._release_dds()
            raise

    def _release_dds(self) -> None:
        """Drops the DDS entities, the participant last, and releases the domain."""
        self._writers = {}
        self._all_topics_readers = {}
        self._publications_reader = None
        for subscription in self._subscriptions.values():
            subscription.reader = None
        self._topics = {}
        self._wake_condition = None
        self._waitset = None
        self._participant = None
        _DOMAINS.release(self._network.domain_id)

    def _topic_for(self, name: str) -> Any:
        """The DDS topic of `name`, created on first use."""
        existing = self._topics.get(name)
        if existing is not None:
            return existing
        problem = network_config.topic_name_problem(name)
        if problem is not None:
            raise ValueError(problem)
        created = cyclonedds.topic.Topic(self._participant, name, envelope.Envelope)
        self._topics[name] = created
        return created

    def _writer_for(self, name: str) -> _Writer:
        """The DDS writer of topic `name`, created on first use with the topic's QoS profile."""
        existing = self._writers.get(name)
        if existing is not None:
            return existing
        profile = self._network.qos_profile_of(name)
        writer = cyclonedds.pub.DataWriter(
            self._participant,
            self._topic_for(name),
            qos=_qos(
                profile,
                qos_profile.qos_policy(profile).writer_history_depth,
                cyclonedds.qos.Policy.DataRepresentation(
                    use_xcdrv2_representation=True
                ),
                cyclonedds.qos.Policy.EntityName(self._node_name),
            ),
        )
        with self._statistics_lock:
            statistics = self._statistics_of(name)
        created = _Writer(writer=writer, statistics=statistics)
        self._writers[name] = created
        return created

    def _create_reader(self, name: str, delivery: Delivery) -> Any:
        """Creates the DDS reader of topic `name` for `delivery` and attaches it to the waitset."""
        depth = (
            self._all_delivery_history_depth
            if delivery is Delivery.ALL
            else self._latest_delivery_history_depth
        )
        reader = cyclonedds.sub.DataReader(
            self._participant,
            self._topic_for(name),
            qos=_qos(
                self._network.qos_profile_of(name),
                depth,
                cyclonedds.qos.Policy.DataRepresentation(
                    use_cdrv0_representation=True, use_xcdrv2_representation=True
                ),
            ),
        )
        self._waitset.attach(cyclonedds.core.ReadCondition(reader, _ANY_STATE))
        return reader

    # ------------------------------------------------------------------------------------------------------------------
    # Configuration (while the bus is not running)
    # ------------------------------------------------------------------------------------------------------------------

    def subscribe(
        self,
        topic: str,
        message_class: type[protobuf_message.Message],
        callback: MessageCallback,
        delivery: str | Delivery = Delivery.LATEST,
    ) -> None:
        """Hands every message of `topic` that carries `message_class` to `callback`, on the receive thread.

        A message of another type, or one that does not parse, is rejected and counted. One subscription per topic.

        Args:
            topic: The topic, a DDS topic name.
            message_class: The generated protobuf class the messages of the topic carry.
            callback: Called with each parsed message.
            delivery: Delivery.LATEST (or "latest") or Delivery.ALL (or "all").

        Raises:
            ValueError: `message_class` is not a generated message, the topic is not a DDS topic name or already
                subscribed, or the callback is not callable.
            BusError: the bus is running or closed.
        """
        descriptor = getattr(message_class, "DESCRIPTOR", None)
        if descriptor is None:
            raise ValueError(f"{message_class!r} is not a generated protobuf message")
        self._add_subscription(
            _Subscription(
                topic=topic,
                delivery=parse_delivery(delivery),
                callback=callback,
                message_class=message_class,
                expected_type=descriptor.full_name,
                statistics=TopicStatistics(),
            )
        )

    def subscribe_raw(
        self,
        topic: str,
        callback: RawCallback,
        delivery: str | Delivery = Delivery.LATEST,
    ) -> None:
        """Hands every message of `topic` to `callback(type_name, payload)`, whatever its type."""
        self._add_subscription(
            _Subscription(
                topic=topic,
                delivery=parse_delivery(delivery),
                callback=callback,
                message_class=None,
                expected_type="",
                statistics=TopicStatistics(),
            )
        )

    def subscribe_all_topics(self, callback: TopicCallback) -> None:
        """Hands every message of every topic to `callback(topic, type_name, payload)`, in order (for tools/ipc).

        The bus discovers the topics from the DDS publications it sees, and reads each one whose type is the envelope
        with a BEST_EFFORT reader, which matches every writer. It sees the messages of the topics that have their own
        subscription too, from the moment it discovered their publisher. It keeps no per-topic statistics.

        Args:
            callback: Called with the topic, the type name and the serialized message of each message.

        Raises:
            ValueError: the callback is not callable, or the bus already has a subscription to every topic.
            BusError: the bus is running or closed.
        """
        if not callable(callback):
            raise ValueError("the callback is not callable")
        self._refuse_on_io_thread("subscribe to every topic")
        with self._lifecycle_lock:
            self._check_configurable("subscribe to every topic")
            if self._all_topics_callback is not None:
                raise ValueError("the bus already has a subscription to every topic")
            self._publications_reader = cyclonedds.builtin.BuiltinDataReader(
                self._participant, cyclonedds.builtin.BuiltinTopicDcpsPublication
            )
            self._waitset.attach(
                cyclonedds.core.ReadCondition(self._publications_reader, _ANY_STATE)
            )
            self._all_topics_callback = callback

    def _add_subscription(self, subscription: _Subscription) -> None:
        """Validates `subscription` and creates its DDS reader; the bus must not be running."""
        if not isinstance(subscription.topic, str):
            raise ValueError("the topic is not a string")
        problem = network_config.topic_name_problem(subscription.topic)
        if problem is not None:
            raise ValueError(problem)
        if not callable(subscription.callback):
            raise ValueError("the callback is not callable")
        self._refuse_on_io_thread(f"subscribe to '{subscription.topic}'")
        with self._lifecycle_lock:
            self._check_configurable(f"subscribe to '{subscription.topic}'")
            if subscription.topic in self._subscriptions:
                raise ValueError(
                    f"the topic '{subscription.topic}' already has a subscription"
                )
            try:
                subscription.reader = self._create_reader(
                    subscription.topic, subscription.delivery
                )
            except cyclonedds.core.DDSException as error:
                raise BusError(
                    f"cannot create the DDS reader of '{subscription.topic}': {error}"
                ) from error
            with self._statistics_lock:
                subscription.statistics = self._statistics.setdefault(
                    subscription.topic, subscription.statistics
                )
            self._subscriptions[subscription.topic] = subscription

    def add_periodic_callback(
        self, period: float, callback: Callable[[], None]
    ) -> None:
        """Calls `callback` on the receive thread every `period` seconds, on absolute deadlines from start().

        A deadline that has passed when the call before it returns (a callback that overran its period) is skipped, to
        the next deadline still ahead, rather than called at once or in a burst.

        Args:
            period: the period in seconds, positive.
            callback: called on the receive thread, with no arguments.

        Raises:
            ValueError: the period is not positive or the callback is not callable.
            BusError: the bus is running or closed, or this is one of its callbacks.
        """
        if not period > 0:  # Negated, so that NaN fails it too.
            raise ValueError("a periodic callback needs a positive period")
        if not callable(callback):
            raise ValueError("the periodic callback is not callable")
        self._refuse_on_io_thread("add a periodic callback")
        with self._lifecycle_lock:
            self._check_configurable("add a periodic callback")
            self._periodic_callbacks.append(_PeriodicCallback(period, callback))

    def _refuse_on_io_thread(self, action: str) -> None:
        """Refuses a configuration call from a callback before it takes the lifecycle lock.

        stop() holds that lock while it joins the receive thread, so a callback waiting for it would never return.
        """
        if self.is_io_thread():
            raise BusError(
                f"cannot {action} while the bus runs; do it before start() or after stop()"
            )

    def _check_configurable(self, action: str) -> None:
        if self._closed:
            raise BusError(f"cannot {action}: the bus is closed")
        if self.is_io_thread() or self._thread is not None:
            raise BusError(
                f"cannot {action} while the bus runs; do it before start() or after stop()"
            )

    # ------------------------------------------------------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------------------------------------------------------

    def start(self) -> None:
        """Launches the receive thread. Idempotent."""
        if self.is_io_thread():
            return
        with self._lifecycle_lock:
            if self._closed:
                raise BusError("cannot start: the bus is closed")
            if self.is_running():
                return
            if self._thread is not None:
                # Asked to stop from one of its own callbacks and not joined yet.
                self._thread.join()
            self._stop_requested.clear()
            now = time.monotonic()
            for periodic in self._periodic_callbacks:
                periodic.next_deadline = now + periodic.period
            self._next_status_refresh = now
            self._thread = threading.Thread(
                target=self._run,
                name=f"robot_ipc-{self._node_name or 'subscriber'}",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        """Stops and joins the receive thread, sending what is still queued first. Idempotent.

        Called from a callback, it only asks the thread to stop after that callback; a later stop() joins it.
        """
        if self.is_io_thread():
            self._stop_requested.set()
            self._wake()
            return
        with self._lifecycle_lock:
            if self._thread is None:
                return
            self._stop_requested.set()
            self._wake()
            self._thread.join()
            self._thread = None

    def close(self) -> None:
        """Stops the bus and deletes its DDS entities; the bus cannot be started again. Idempotent.

        From the call on, publish() raises BusError. On a running bus, what was queued before is written first. On a
        bus that is not running, the queue is discarded and counted as send_dropped, as is a publish() that raced the
        call and landed after the last write. A reliable sample that DDS has not delivered yet is still sent while its
        writer lingers (Cyclone DDS's WriterLingerDuration).

        Raises:
            BusError: called from one of the bus's own callbacks, whose entities are still in use.
        """
        if self.is_io_thread():
            raise BusError(
                "close() from a callback of the bus; call stop() there instead"
            )
        with self._lifecycle_lock:
            if self._closed:
                return
            # From here publish() refuses; what was queued before goes out with the last send of stop().
            self._closed = True
        self.stop()
        self._discard_queued()
        with self._lifecycle_lock:
            self._release_dds()

    def _discard_queued(self) -> None:
        """Counts what is still queued at close() as send_dropped of its topic, and logs how many there were."""
        with self._queue_lock:
            discarded = self._queue
            self._queue = collections.deque()
        if not discarded:
            return
        with self._statistics_lock:
            for topic, _, _ in discarded:
                self._statistics_of(topic).send_dropped += 1
        self._log.warning(
            "close",
            f"close() discarded {len(discarded)} queued message(s) that the bus was not running to send",
        )

    def __enter__(self) -> "Bus":
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def is_running(self) -> bool:
        """True between start() and stop()."""
        thread = self._thread
        return (
            thread is not None
            and thread.is_alive()
            and not self._stop_requested.is_set()
        )

    def is_io_thread(self) -> bool:
        """True on the bus's receive thread, i.e. inside a callback or a periodic callback."""
        return self._io_thread_ident == threading.get_ident()

    # ------------------------------------------------------------------------------------------------------------------
    # Publishing (from any thread)
    # ------------------------------------------------------------------------------------------------------------------

    def publish(self, topic: str, message: protobuf_message.Message) -> bool:
        """Publishes `message` on `topic` from any thread; never waits for the network.

        A message published while the bus is not running stays queued until start().

        Args:
            topic: The topic to publish on.
            message: The message; it is serialized on the calling thread.

        Returns:
            False when the queue to the receive thread was full and the message was dropped (counted).

        Raises:
            BusError: the bus has no node name, so it only subscribes; or it is closed.
            ValueError: the topic is not a DDS topic name.
        """
        return self.publish_raw(
            topic, message.DESCRIPTOR.full_name, message.SerializeToString()
        )

    def publish_raw(self, topic: str, type_name: str, payload: bytes) -> bool:
        """publish() of an already serialized message of the type `type_name` names (for tools/ipc)."""
        self._check_can_publish(topic)
        with self._queue_lock:
            if self._closed:
                raise BusError(f"cannot publish on '{topic}': the bus is closed")
            if len(self._queue) >= self._publish_queue_capacity:
                with self._statistics_lock:
                    self._statistics_of(topic).send_dropped += 1
                return False
            was_empty = not self._queue
            self._queue.append((topic, type_name, bytes(payload)))
        # The receive thread empties the whole queue on one wake-up, so only the first message needs to wake it.
        if was_empty:
            self._wake()
        return True

    def publish_from_io_thread(
        self, topic: str, message: protobuf_message.Message
    ) -> None:
        """Publishes from a callback or a periodic callback, straight to the DDS writer.

        Args:
            topic: The topic to publish on.
            message: The message, written after whatever publish() queued before it.

        Raises:
            BusError: called from another thread, or the bus only subscribes.
        """
        if not self.is_io_thread():
            raise BusError(
                "publish_from_io_thread() outside the bus's receive thread; call publish()"
            )
        self._check_can_publish(topic)
        # What publish() queued before goes first, so that the messages of one thread keep their order.
        self._send_queued()
        self._send(topic, message.DESCRIPTOR.full_name, message.SerializeToString())

    def _check_can_publish(self, topic: str) -> None:
        if not self._node_name:
            raise BusError(
                "this bus has no node name, so it only subscribes and cannot publish"
            )
        if not isinstance(topic, str):
            raise ValueError("the topic is not a string")
        problem = network_config.topic_name_problem(topic)
        if problem is not None:
            raise ValueError(problem)

    def _wake(self) -> None:
        condition = self._wake_condition
        if condition is not None:
            try:
                condition.set(True)
            except cyclonedds.core.DDSException:
                # The bus is being closed: its receive thread has stopped already.
                pass

    # ------------------------------------------------------------------------------------------------------------------
    # Introspection (from any thread)
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def node_name(self) -> str:
        """The node this bus publishes as; empty when it only subscribes."""
        return self._node_name

    @property
    def domain_id(self) -> int:
        """The DDS domain of the bus."""
        return self._network.domain_id

    @property
    def discovery_peers(self) -> list[str]:
        """The hosts the bus sends its discovery packets to: the distinct hosts of the network's nodes."""
        return self._network.peer_hosts()

    @property
    def domain_config(self) -> str:
        """The Cyclone DDS configuration the bus joined its domain with."""
        return self._domain_config

    @property
    def periodic_callback_errors(self) -> int:
        """Periodic callback calls that raised."""
        return self._periodic_callback_errors

    def topic_statistics(self, topic: str) -> TopicStatistics:
        """The counters of one topic (all zero for a topic the bus has not seen)."""
        with self._statistics_lock:
            statistics = self._statistics.get(topic)
            return dataclasses.replace(statistics) if statistics else TopicStatistics()

    def statistics(self) -> dict[str, TopicStatistics]:
        """The counters of every topic the bus has published or subscribed."""
        with self._statistics_lock:
            return {
                topic: dataclasses.replace(statistics)
                for topic, statistics in self._statistics.items()
            }

    def _statistics_of(self, topic: str) -> TopicStatistics:
        """Requires _statistics_lock."""
        statistics = self._statistics.get(topic)
        if statistics is None:
            statistics = self._statistics[topic] = TopicStatistics()
        return statistics

    # ------------------------------------------------------------------------------------------------------------------
    # The receive thread
    # ------------------------------------------------------------------------------------------------------------------

    def _run(self) -> None:
        """The receive thread: waits on the waitset, writes, drains and runs the due periodic callbacks."""
        self._io_thread_ident = threading.get_ident()
        try:
            while not self._stop_requested.is_set():
                try:
                    self._waitset.wait(self._wait_timeout_ns())
                    self._wake_condition.take()
                    self._send_queued()
                    self._discover_topics()
                    self._drain()
                    now = time.monotonic()
                    self._refresh_status_if_due(now)
                    self._run_due_callbacks(now)
                # pylint: disable-next=broad-exception-caught  # The outermost loop of the receive thread.
                except Exception:
                    # Nothing escapes the receive thread.
                    self._log.warning(
                        "loop", "the receive thread caught an exception", exc_info=True
                    )
            try:
                self._send_queued()
                self._refresh_status_if_due(time.monotonic())
            # pylint: disable-next=broad-exception-caught  # The receive thread's last act; nothing escapes it.
            except Exception:
                self._log.warning(
                    "stop",
                    "sending the queued messages at stop() failed",
                    exc_info=True,
                )
        finally:
            self._io_thread_ident = None

    def _wait_timeout_ns(self) -> int:
        wait = self._io_poll_period
        now = time.monotonic()
        for periodic in self._periodic_callbacks:
            wait = min(wait, periodic.next_deadline - now)
        if wait <= 0:
            return 0
        return int(math.ceil(wait * _NANOSECONDS_PER_SECOND))

    def _send_queued(self) -> None:
        with self._queue_lock:
            if not self._queue:
                return
            sending = self._queue
            self._queue = collections.deque()
        for topic, type_name, payload in sending:
            self._send(topic, type_name, payload)

    def _next_source_timestamp(self) -> int:
        self._last_source_timestamp = max(
            time.time_ns(), self._last_source_timestamp + 1
        )
        return self._last_source_timestamp

    def _send(self, topic: str, type_name: str, payload: bytes) -> None:
        """Writes one sample without waiting, and counts it as written or dropped.

        Args:
            topic: The topic.
            type_name: The full protobuf type name of the payload.
            payload: The serialized message.
        """
        try:
            writer = self._writer_for(topic)
            sample = envelope.Envelope(
                protocol_version=envelope.WIRE_PROTOCOL_VERSION,
                sequence_number=writer.last_sequence + 1,
                type_name=type_name,
                payload=payload,
            )
            writer.writer.write(sample, timestamp=self._next_source_timestamp())
        except cyclonedds.core.DDSException as error:
            with self._statistics_lock:
                self._statistics_of(topic).send_dropped += 1
            self._log.warning(f"send:{topic}", f"writing on '{topic}' failed: {error}")
            return
        writer.last_sequence = sample.sequence_number
        with self._statistics_lock:
            writer.statistics.written += 1

    def _discover_topics(self) -> None:
        """Creates a reader for each envelope topic a newly discovered publication writes (subscribe_all_topics())."""
        if self._publications_reader is None:
            return
        for publication in self._publications_reader.take(self._max_messages_per_drain):
            name = getattr(publication, "topic_name", None)
            if (
                name is None
                or name in self._all_topics_readers
                or getattr(publication, "type_name", None)
                != envelope.ENVELOPE_TYPE_NAME
            ):
                continue
            self._all_topics_readers[name] = cyclonedds.sub.DataReader(
                self._participant,
                self._topic_for(name),
                qos=_qos(
                    qos_profile.QosProfile.LATEST,
                    self._all_delivery_history_depth,
                    cyclonedds.qos.Policy.DataRepresentation(
                        use_cdrv0_representation=True, use_xcdrv2_representation=True
                    ),
                ),
            )
            self._waitset.attach(
                cyclonedds.core.ReadCondition(
                    self._all_topics_readers[name], _ANY_STATE
                )
            )

    def _take(
        self,
        reader: Any,
        budget: int,
        topic: str,
        subscription: _Subscription | None,
        pending: list[_PendingMessage],
    ) -> int:
        """Takes up to `budget` samples of `reader` into `pending`; returns how many it took."""
        samples = reader.take(budget)
        for sample in samples:
            if not isinstance(sample, envelope.Envelope):
                # A writer's disposal or unregistration carries no sample.
                continue
            if subscription is None:
                if sample.protocol_version == envelope.WIRE_PROTOCOL_VERSION:
                    pending.append(
                        _PendingMessage(
                            sample.sample_info.source_timestamp, None, topic, sample
                        )
                    )
                continue
            self._track_sequence(
                subscription,
                sample.sample_info.publication_handle,
                sample.sequence_number,
            )
            accepted = self._accepts(subscription, sample)
            with self._statistics_lock:
                subscription.statistics.received += 1
                if not accepted:
                    subscription.statistics.rejected += 1
            if accepted:
                pending.append(
                    _PendingMessage(
                        sample.sample_info.source_timestamp, subscription, topic, sample
                    )
                )
        return len(samples)

    def _track_sequence(
        self, subscription: _Subscription, writer: int, sequence: int
    ) -> None:
        """Counts the gap between the last sequence number of `writer` and `sequence` as missed."""
        last = subscription.last_sequence.get(writer)
        subscription.last_sequence[writer] = max(sequence, last or 0)
        if last is not None and sequence > last + 1:
            with self._statistics_lock:
                subscription.statistics.missed += sequence - last - 1

    def _accepts(self, subscription: _Subscription, sample: envelope.Envelope) -> bool:
        """Whether `sample` is of this bus's wire protocol and of the subscription's type; logs why not."""
        if sample.protocol_version != envelope.WIRE_PROTOCOL_VERSION:
            self._log.warning(
                f"protocol:{subscription.topic}",
                f"a sample on '{subscription.topic}' speaks wire protocol {sample.protocol_version}, but this bus "
                f"speaks {envelope.WIRE_PROTOCOL_VERSION}; rebuild both from the same checkout",
            )
            return False
        if (
            subscription.expected_type
            and sample.type_name != subscription.expected_type
        ):
            self._log.warning(
                f"type:{subscription.topic}",
                f"'{subscription.topic}' carries a {sample.type_name}, but its subscription takes a "
                f"{subscription.expected_type}",
            )
            return False
        return True

    def _drain(self) -> None:
        """Takes up to max_messages_per_drain samples, then dispatches them in the order they were written.

        A LATEST subscription's newest sample of the drain is dispatched at the place it was written; its older ones
        are superseded. Holding back every sample until the drain ends keeps the order across topics.
        """
        pending: list[_PendingMessage] = []
        budget = self._max_messages_per_drain
        for subscription in self._subscriptions.values():
            if budget <= 0:
                break
            budget -= self._take(
                subscription.reader, budget, subscription.topic, subscription, pending
            )
        for name, reader in self._all_topics_readers.items():
            if budget <= 0:
                break
            budget -= self._take(reader, budget, name, None, pending)
        pending.sort(key=lambda message: message.source_timestamp)
        self._coalesce_latest(pending)
        for message in pending:
            if message.superseded:
                continue
            if message.subscription is None:
                self._dispatch_all_topics(message)
            else:
                self._dispatch(message.subscription, message.sample)

    def _coalesce_latest(self, pending: list[_PendingMessage]) -> None:
        """Marks every LATEST sample of the drain superseded but the newest of its topic."""
        for subscription in self._subscriptions.values():
            subscription.latest = None
        for index, message in enumerate(pending):
            owner = message.subscription
            if owner is None or owner.delivery is not Delivery.LATEST:
                continue
            if owner.latest is not None:
                pending[owner.latest].superseded = True
                with self._statistics_lock:
                    owner.statistics.superseded += 1
            owner.latest = index

    def _dispatch_all_topics(self, message: _PendingMessage) -> None:
        """Hands one sample to the subscription to every topic."""
        callback = self._all_topics_callback
        if callback is None:
            return
        try:
            callback(
                message.topic, message.sample.type_name, bytes(message.sample.payload)
            )
        # pylint: disable-next=broad-exception-caught  # A raising callback must not end the receive thread.
        except Exception:
            self._log.warning(
                "all_topics", "the callback of every topic raised", exc_info=True
            )

    def _dispatch(self, subscription: _Subscription, sample: envelope.Envelope) -> None:
        """Parses one sample of `subscription` and hands it to its callback, counting a rejection or a raise."""
        statistics = subscription.statistics
        payload = bytes(sample.payload)
        arguments: tuple[Any, ...]
        if subscription.message_class is not None:
            try:
                arguments = (subscription.message_class.FromString(payload),)
            except protobuf_message.DecodeError:
                with self._statistics_lock:
                    statistics.rejected += 1
                self._log.warning(
                    f"parse:{subscription.topic}",
                    f"a sample on '{subscription.topic}' does not parse as a {subscription.expected_type}",
                )
                return
        else:
            arguments = (sample.type_name, payload)
        with self._statistics_lock:
            statistics.delivered += 1
        try:
            subscription.callback(*arguments)
        # pylint: disable-next=broad-exception-caught  # A raising callback must not end the receive thread.
        except Exception:
            with self._statistics_lock:
                statistics.handler_errors += 1
            self._log.warning(
                f"handler:{subscription.topic}",
                f"the callback of '{subscription.topic}' raised",
                exc_info=True,
            )

    def _refresh_status_if_due(self, now: float) -> None:
        """Copies the DDS status of the readers and writers into the statistics, at most every io_poll_period."""
        if now < self._next_status_refresh:
            return
        self._next_status_refresh = now + self._io_poll_period
        requested: dict[str, int] = {}
        for subscription in self._subscriptions.values():
            reader = subscription.reader
            lost = reader.get_sample_lost_status().total_count
            rejected = reader.get_sample_rejected_status().total_count
            incompatible = reader.get_requested_incompatible_qos_status().total_count
            matched = reader.get_subscription_matched_status().current_count
            requested[subscription.topic] = incompatible
            with self._statistics_lock:
                statistics = subscription.statistics
                statistics.sample_lost = lost
                statistics.sample_rejected = rejected
                # The writer's offered count is added below, for a topic this bus also publishes.
                statistics.incompatible_qos = incompatible
                statistics.matched_writers = matched
        for name, writer in self._writers.items():
            offered = writer.writer.get_offered_incompatible_qos_status().total_count
            matched = writer.writer.get_publication_matched_status().current_count
            with self._statistics_lock:
                statistics = writer.statistics
                statistics.incompatible_qos = requested.get(name, 0) + offered
                statistics.matched_readers = matched

    def _run_due_callbacks(self, now: float) -> None:
        """Calls each periodic callback whose deadline has passed, and schedules its next call."""
        for periodic in self._periodic_callbacks:
            if now < periodic.next_deadline:
                continue
            try:
                periodic.callback()
            # pylint: disable-next=broad-exception-caught  # A raising callback must not end the receive thread.
            except Exception:
                self._periodic_callback_errors += 1
                self._log.warning(
                    "periodic", "a periodic callback raised", exc_info=True
                )
            # Against the time the callback returned, not `now`: a callback that overran its period must not be called
            # again at once. A missed deadline is skipped, to the first one still ahead on the grid of start().
            after = time.monotonic()
            periodic.next_deadline += periodic.period
            if periodic.next_deadline <= after:
                missed = math.floor((after - periodic.next_deadline) / periodic.period)
                periodic.next_deadline += (missed + 1) * periodic.period


__all__ = [
    "WIRE_PROTOCOL_VERSION",
    "Bus",
    "BusError",
    "Delivery",
    "TopicStatistics",
    "parse_delivery",
]
