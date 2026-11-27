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


"""The Zenoh bus of the distributed runtime, in Python: the twin of robot_ipc/Bus.h.

Every bus topic is a Zenoh key expression (key_exprs.py) whose samples are one envelope (envelope.py): the wire
protocol's version, the writer's id and sequence number, the source timestamp, the full protobuf type name and the
serialized message, exactly as the C++ bus writes them. A sample of another protocol version is refused and counted as
rejected.

The buses of a process share one Zenoh peer session of the network's domain, configured from the network file
(session_config.py): a TCP listener on the node's host, connections to the host of every node, the QoS of each topic by
its named profile (qos_profile.py). A named bus declares the publishers of the topics the file lists its node as a
publisher of when it is constructed, and every bus declares a liveliness token per publisher and subscriber, which the
others count (TopicStatistics.matched_readers, matched_writers).

One drain takes up to max_messages_per_drain samples across the topics and dispatches them in the order they were
written (their source timestamps, which every bus makes strictly increasing across its topics): every sample of an ALL
subscription, and the newest one of a LATEST subscription at the place it was written.

The Zenoh publishers belong to the bus's receive thread, which start() launches; Zenoh's own threads copy each arriving
sample into its subscription's queue and wake it. Callbacks and periodic callbacks run on the receive thread, one at a
time. publish() may be called from any thread: it serializes on the caller and queues the sample for the receive
thread, which it wakes at once; only the receive thread puts samples.

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
import random
import threading
import time
from typing import Any

from google.protobuf import message as protobuf_message
import zenoh

from robot_ipc import envelope
from robot_ipc import key_exprs
from robot_ipc import network_config
from robot_ipc import qos_profile
from robot_ipc import session_config

_LOGGER = logging.getLogger("robot_ipc")

# The receive thread logs a recurring problem at most this often, in seconds.
_LOG_PERIOD = 5.0
# Zenoh's own log: errors only, unless RUST_LOG says otherwise.
_ZENOH_LOG_FILTER = "error"
# The listen port of the configuration that buses sharing a session compare.
_SHARED_CONFIG_PORT = 0
# The width of a random writer or bus id, in bits.
_ID_BITS = 64

WIRE_PROTOCOL_VERSION = envelope.WIRE_PROTOCOL_VERSION

_CONGESTION_CONTROLS = {
    qos_profile.CongestionControl.DROP: zenoh.CongestionControl.DROP,
    qos_profile.CongestionControl.BLOCK: zenoh.CongestionControl.BLOCK,
}
_PRIORITIES = {
    qos_profile.Priority.INTERACTIVE_HIGH: zenoh.Priority.INTERACTIVE_HIGH,
    qos_profile.Priority.DATA: zenoh.Priority.DATA,
}


# LINT.IfChange(delivery_names)
class Delivery(str, enum.Enum):
    """What a subscription receives of a topic each time the receive thread drains it."""

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
        The Delivery that `value` specifies.

    Raises:
        ValueError: an unknown name; the message lists the valid ones.
    """
    if isinstance(value, Delivery):
        return value
    try:
        return Delivery(value)
    except ValueError:
        valid = ", ".join(delivery.value for delivery in Delivery)
        raise ValueError(
            f"unknown delivery '{value}' (valid: {valid})"
        ) from None


# LINT.IfChange(topic_statistics)
@dataclasses.dataclass
class TopicStatistics:
    """What a bus did with one topic; see robot_ipc/TopicStatistics.h.

    Once a drain has dispatched, received == delivered + superseded + rejected. The last three fields count the
    liveliness tokens of the buses (key_exprs.py), as the receive thread last saw them.

    Attributes:
        written: samples passed to the topic's Zenoh publisher, not delivered ones.
        send_dropped: publish() calls refused because the queue to the receive thread was full, puts Zenoh refused,
            and queued samples close() discarded because the bus was not running to send them.
        received: samples taken from the subscription's queue.
        delivered: samples passed to the callback (including those whose callback then raised).
        superseded: LATEST only: samples replaced by a newer one of the same drain before they were parsed.
        rejected: samples of another wire protocol version, another type than the subscription's, or that did not
            parse.
        handler_errors: callback calls that raised; the bus logs and counts them and carries on.
        missed: samples a writer wrote that the subscription never took (gaps in its sequence numbers).
        incompatible_qos: publishers and subscribers of the topic whose profile is another than this bus's.
        matched_readers: the subscribers of the topic with this bus's profile, while this bus has a publisher of it.
        matched_writers: the publishers of the topic with this bus's profile, while this bus subscribes to it.
    """

    written: int = 0
    send_dropped: int = 0
    received: int = 0
    delivered: int = 0
    superseded: int = 0
    rejected: int = 0
    handler_errors: int = 0
    missed: int = 0
    incompatible_qos: int = 0
    matched_readers: int = 0
    matched_writers: int = 0


# LINT.ThenChange(//robot_runtime/robot_ipc/include/robot_ipc/TopicStatistics.h:topic_statistics)


class BusError(RuntimeError):
    """The bus cannot do what was asked in its current state, or cannot open its Zenoh session."""


MessageCallback = Callable[[protobuf_message.Message], None]
RawCallback = Callable[[str, bytes], None]
TopicCallback = Callable[[str, str, bytes], None]


@dataclasses.dataclass
class _Session:
    """One open session of the registry."""

    session: Any
    shared_config: str
    config: str
    users: int = 1


class _SessionRegistry:
    """The Zenoh sessions of this process, one per domain, shared by its buses (robot_ipc/src/SessionRegistry.h).

    The first bus of a domain opens its session, on the lowest free port of the domain, and the last to release it
    closes it. Thread-safe.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[int, _Session] = {}
        self._log_initialized = False

    def acquire(
        self, network: network_config.NetworkConfig, node_name: str
    ) -> _Session:
        """Opens the session of `network`'s domain, or counts one more user of it; another configuration is refused.

        Args:
            network: The network.
            node_name: The node of the bus; empty for one that only subscribes.

        Returns:
            The session entry.

        Raises:
            session_config.InterfaceError: the node's host is an address of another machine.
            BusError: the session is open with another configuration, or no port of the domain is free.
        """
        shared_config = session_config.zenoh_session_config(
            network, node_name, _SHARED_CONFIG_PORT
        )
        with self._lock:
            entry = self._sessions.get(network.domain_id)
            if entry is not None:
                if entry.shared_config != shared_config:
                    raise BusError(
                        f"domain {network.domain_id} is configured differently"
                        " by another bus of this process (a bus of another node"
                        " or network); use one network file and node per"
                        " process"
                    )
                entry.users += 1
                return entry
            if not self._log_initialized:
                zenoh.init_log_from_env_or(_ZENOH_LOG_FILTER)
                self._log_initialized = True
            listen_address = session_config.interface_address_for(
                network, node_name
            )
            ports = session_config.tcp_port_range(network)
            for port in range(ports.first, ports.last + 1):
                # A taken port is skipped before Zenoh tries it, which would log an error.
                if not session_config.is_tcp_port_free(listen_address, port):
                    continue
                config = session_config.zenoh_session_config(
                    network, node_name, port
                )
                try:
                    session = zenoh.open(zenoh.Config.from_json5(config))
                except zenoh.ZError:
                    continue
                entry = _Session(session, shared_config, config)
                self._sessions[network.domain_id] = entry
                return entry
            raise BusError(
                f"no Zenoh session of domain {network.domain_id} could listen"
                f" on any TCP port of {ports.first}-{ports.last}: every port is"
                " taken on this machine (stop a process of the domain, or raise"
                " transport.max_processes_per_host), or the listen address is"
                " not this machine's"
            )

    def release(self, domain_id: int) -> None:
        """Releases one use of the domain's session; the last release closes it."""
        with self._lock:
            entry = self._sessions.get(domain_id)
            if entry is None:
                return
            entry.users -= 1
            if entry.users == 0:
                del self._sessions[domain_id]
                entry.session.close()


# Process-wide, like the sessions it tracks: one session per domain and process.
_SESSIONS = _SessionRegistry()

# The random ids of writers and buses; a module-level generator rather than one per bus, under its own lock.
_ID_LOCK = threading.Lock()
_ID_RANDOM = random.SystemRandom()


def _random_id() -> int:
    with _ID_LOCK:
        return _ID_RANDOM.getrandbits(_ID_BITS)


class _Inbox:
    """Zenoh's threads -> the receive thread: the newest `depth` items (None: all), and a wake-up of the receive thread.

    Thread-safe.
    """

    def __init__(self, depth: int | None, wake: Callable[[], None]) -> None:
        self._lock = threading.Lock()
        self._items: collections.deque[Any] = collections.deque(maxlen=depth)
        self._wake = wake

    def put(self, item: Any) -> None:
        """Appends `item`, dropping the oldest at the depth, and wakes the receive thread."""
        with self._lock:
            self._items.append(item)
        self._wake()

    def take(self, budget: int) -> list[Any]:
        """The oldest items, up to `budget`; what is left stays for the next drain."""
        with self._lock:
            count = min(budget, len(self._items))
            return [self._items.popleft() for _ in range(count)]

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


@dataclasses.dataclass
class _Subscription:
    """One subscription of the bus, its Zenoh subscriber and its queue."""

    topic: str
    delivery: Delivery
    callback: Callable[..., None]
    # None for subscribe_raw(), which takes any type.
    message_class: type[protobuf_message.Message] | None
    expected_type: str
    statistics: TopicStatistics
    profile: qos_profile.QosProfile = qos_profile.QosProfile.RELIABLE_EVENTS
    inbox: _Inbox | None = None
    subscriber: Any = None
    token: Any = None
    # The last sequence number of each writer this subscription took a sample of.
    last_sequence: dict[int, int] = dataclasses.field(default_factory=dict)
    # LATEST: the index, in the drain's pending samples, of the newest sample of this subscription so far.
    latest: int | None = None


@dataclasses.dataclass
class _Writer:
    """The Zenoh publisher of one topic, its liveliness token and the sequence number of its last sample."""

    publisher: Any
    token: Any
    profile: qos_profile.QosProfile
    writer_id: int
    statistics: TopicStatistics
    last_sequence: int = 0


@dataclasses.dataclass
class _PendingMessage:
    """A sample a drain has taken and dispatches once it has taken the rest, in the order the samples were written."""

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


class Bus:
    """A publish/subscribe bus over Zenoh, wire-compatible with the C++ robot::ipc::Bus.

    Args:
        node_name: the node this process publishes as. Empty: only subscribe.
        network: the Zenoh network: domain, peers, topic QoS and transport.
        io_poll_period: the longest the receive thread waits, in seconds; publish(), stop(), arriving samples and
            due periodic callbacks all wake it at once.
        publish_queue_capacity: messages publish() queues before it refuses (counted as send_dropped).
        max_messages_per_drain: the most samples one drain takes, across topics, before the thread dispatches.
        latest_delivery_history_depth: the samples the queue of a LATEST subscription keeps, the newest (more than
            one, so that a sample of the wrong type does not displace a valid one before the drain sees both).
        all_delivery_history_depth: the samples the queue of an ALL subscription keeps while the thread is busy.

    Raises:
        network_config.NetworkConfigError: the network is invalid.
        ValueError: the node is not in the network, or an option is out of range.
        session_config.InterfaceError: the node's host is an address of another machine.
        BusError: the process's buses disagree about the domain's configuration, or Zenoh fails.
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
                f"node '{node_name}' is not in the network (nodes:"
                f" {', '.join(network.node_names())})"
            )
        # The float check is negated, so that NaN fails it too (AGENTS.md, Python: negated float comparisons).
        if not io_poll_period > 0:
            raise ValueError("io_poll_period must be positive")
        if publish_queue_capacity <= 0 or max_messages_per_drain <= 0:
            raise ValueError(
                "publish_queue_capacity and max_messages_per_drain must be"
                " positive"
            )
        if (
            latest_delivery_history_depth <= 0
            or all_delivery_history_depth <= 0
        ):
            raise ValueError(
                "latest_delivery_history_depth and all_delivery_history_depth"
                " must be positive"
            )

        self._node_name = node_name
        self._network = network
        self._io_poll_period = io_poll_period
        self._publish_queue_capacity = publish_queue_capacity
        self._max_messages_per_drain = max_messages_per_drain
        self._latest_delivery_history_depth = latest_delivery_history_depth
        self._all_delivery_history_depth = all_delivery_history_depth
        self._bus_id = f"{_random_id():016x}"

        self._lifecycle_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._io_thread_ident: int | None = None
        self._stop_requested = threading.Event()
        self._woken = threading.Event()
        self._closed = False

        self._subscriptions: dict[str, _Subscription] = {}
        self._all_topics_callback: TopicCallback | None = None
        self._all_topics_inbox: _Inbox | None = None
        self._all_topics_subscriber: Any = None
        self._periodic_callbacks: list[_PeriodicCallback] = []
        self._periodic_callback_errors = 0

        self._queue_lock = threading.Lock()
        self._queue: collections.deque[tuple[str, str, bytes]] = (
            collections.deque()
        )
        self._statistics_lock = threading.Lock()
        self._statistics: dict[str, TopicStatistics] = {}
        self._log = _RateLimitedLog()
        self._last_source_timestamp = 0
        # Zenoh's threads -> the receive thread: (key expression, alive) of the domain's liveliness tokens.
        self._token_events = _Inbox(depth=None, wake=self._wake)
        # The live tokens of the domain by key expression (receive thread).
        self._live_tokens: dict[str, key_exprs.EndpointToken] = {}
        self._writers: dict[str, _Writer] = {}
        self._token_subscriber: Any = None

        entry = _SESSIONS.acquire(network, node_name)
        self._session: Any = entry.session
        self._session_config = entry.config
        try:
            self._token_subscriber = (
                self._session.liveliness().declare_subscriber(
                    key_exprs.all_endpoint_tokens_key_expr(network),
                    self._on_token,
                    history=True,
                )
            )
            # The publishers of the topics this node publishes exist from the start, so that the subscribers know of
            # them before the first sample (a subscriber gets only what is put after the declarations met).
            for name in (
                network.topics_published_by(node_name) if node_name else []
            ):
                self._writer_for(name)
        except zenoh.ZError as error:
            self._release_zenoh()
            raise BusError(
                f"cannot declare the Zenoh entities: {error}"
            ) from error
        except BaseException:
            self._release_zenoh()
            raise

    def _release_zenoh(self) -> None:
        """Undeclares the subscribers first, then the publishers and tokens, and releases the session."""
        if self._session is None:
            return
        undeclared = [self._token_subscriber, self._all_topics_subscriber]
        for subscription in self._subscriptions.values():
            undeclared += [subscription.subscriber, subscription.token]
            subscription.subscriber = subscription.token = None
        for writer in self._writers.values():
            undeclared += [writer.publisher, writer.token]
        for entity in undeclared:
            if entity is not None:
                entity.undeclare()
        self._token_subscriber = self._all_topics_subscriber = None
        self._writers = {}
        self._session = None
        _SESSIONS.release(self._network.domain_id)

    def _declare_token(
        self,
        role: key_exprs.EndpointRole,
        profile: qos_profile.QosProfile,
        topic: str,
    ) -> Any:
        return self._session.liveliness().declare_token(
            key_exprs.endpoint_token_key_expr(
                self._network,
                key_exprs.EndpointToken(
                    role=role, profile=profile, bus_id=self._bus_id, topic=topic
                ),
            )
        )

    def _writer_for(self, name: str) -> _Writer:
        """The Zenoh publisher of topic `name`, declared on first use with the topic's QoS profile."""
        existing = self._writers.get(name)
        if existing is not None:
            return existing
        problem = network_config.topic_name_problem(name)
        if problem is not None:
            raise ValueError(problem)
        profile = self._network.qos_profile_of(name)
        policy = qos_profile.qos_policy(profile)
        publisher = self._session.declare_publisher(
            key_exprs.data_key_expr(self._network, name),
            congestion_control=_CONGESTION_CONTROLS[policy.congestion_control],
            priority=_PRIORITIES[policy.priority],
            express=policy.express,
        )
        token = self._declare_token(
            key_exprs.EndpointRole.PUBLISHER, profile, name
        )
        with self._statistics_lock:
            statistics = self._statistics_of(name)
        created = _Writer(
            publisher=publisher,
            token=token,
            profile=profile,
            writer_id=_random_id(),
            statistics=statistics,
        )
        self._writers[name] = created
        self._count_peers()
        return created

    def _on_sample(self, inbox: _Inbox, sample: Any) -> None:
        """A subscription's sample, on a Zenoh thread."""
        inbox.put(sample.payload.to_bytes())

    def _on_all_topics_sample(self, sample: Any) -> None:
        """A sample of any topic, on a Zenoh thread (subscribe_all_topics())."""
        inbox = self._all_topics_inbox
        if inbox is not None:
            inbox.put((str(sample.key_expr), sample.payload.to_bytes()))

    def _on_token(self, sample: Any) -> None:
        """A liveliness token that appeared or went away, on a Zenoh thread."""
        self._token_events.put(
            (str(sample.key_expr), sample.kind == zenoh.SampleKind.PUT)
        )

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
        """Passes every message of `topic` that has `message_class` to `callback`, on the receive thread.

        A message of another type, or one that does not parse, is rejected and counted. One subscription per topic.

        Args:
            topic: The topic.
            message_class: The generated protobuf class the messages of the topic have.
            callback: Called with each parsed message.
            delivery: Delivery.LATEST (or "latest") or Delivery.ALL (or "all").

        Raises:
            ValueError: `message_class` is not a generated message, the topic is not a bus topic or already
                subscribed, or the callback is not callable.
            BusError: the bus is running or closed.
        """
        descriptor = getattr(message_class, "DESCRIPTOR", None)
        if descriptor is None:
            raise ValueError(
                f"{message_class!r} is not a generated protobuf message"
            )
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
        """Passes every message of `topic` to `callback(type_name, payload)`, whatever its type."""
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
        """Passes every message of every topic to `callback(topic, type_name, payload)`, in order (for tools/ipc).

        One Zenoh subscriber takes every key expression of the domain and tag, of every profile. It sees the messages of
        the topics that have their own subscription too. It keeps no per-topic statistics.

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
                raise ValueError(
                    "the bus already has a subscription to every topic"
                )
            self._all_topics_inbox = _Inbox(
                self._all_delivery_history_depth, self._wake
            )
            self._all_topics_subscriber = self._session.declare_subscriber(
                key_exprs.all_topics_key_expr(self._network),
                self._on_all_topics_sample,
            )
            self._all_topics_callback = callback

    def _add_subscription(self, subscription: _Subscription) -> None:
        """Validates `subscription` and declares its Zenoh subscriber; the bus must not be running."""
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
                    f"the topic '{subscription.topic}' already has a"
                    " subscription"
                )
            subscription.profile = self._network.qos_profile_of(
                subscription.topic
            )
            inbox = _Inbox(
                (
                    self._all_delivery_history_depth
                    if subscription.delivery is Delivery.ALL
                    else self._latest_delivery_history_depth
                ),
                self._wake,
            )
            subscription.inbox = inbox
            try:
                subscription.subscriber = self._session.declare_subscriber(
                    key_exprs.data_key_expr(self._network, subscription.topic),
                    lambda sample: self._on_sample(inbox, sample),
                )
                subscription.token = self._declare_token(
                    key_exprs.EndpointRole.SUBSCRIBER,
                    subscription.profile,
                    subscription.topic,
                )
            except zenoh.ZError as error:
                if subscription.subscriber is not None:
                    subscription.subscriber.undeclare()
                raise BusError(
                    f"cannot declare the Zenoh subscriber of"
                    f" '{subscription.topic}': {error}"
                ) from error
            with self._statistics_lock:
                subscription.statistics = self._statistics.setdefault(
                    subscription.topic, subscription.statistics
                )
            self._subscriptions[subscription.topic] = subscription
            self._count_peers()

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
                f"cannot {action} while the bus runs; do it before start() or"
                " after stop()"
            )

    def _check_configurable(self, action: str) -> None:
        if self._closed:
            raise BusError(f"cannot {action}: the bus is closed")
        if self.is_io_thread() or self._thread is not None:
            raise BusError(
                f"cannot {action} while the bus runs; do it before start() or"
                " after stop()"
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
        """Stops the bus and undeclares its Zenoh entities; the bus cannot be started again. Idempotent.

        From the call on, publish() raises BusError. On a running bus, what was queued before is put first. On a bus
        that is not running, the queue is discarded and counted as send_dropped, as is a publish() that raced the call
        and landed after the last put.

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
            self._release_zenoh()

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
            f"close() discarded {len(discarded)} queued message(s) that the bus"
            " was not running to send",
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
            ValueError: the topic is not a bus topic.
        """
        return self.publish_raw(
            topic, message.DESCRIPTOR.full_name, message.SerializeToString()
        )

    def publish_raw(self, topic: str, type_name: str, payload: bytes) -> bool:
        """publish() of an already serialized message of the type `type_name` names (for tools/ipc)."""
        self._check_can_publish(topic)
        with self._queue_lock:
            if self._closed:
                raise BusError(
                    f"cannot publish on '{topic}': the bus is closed"
                )
            if len(self._queue) >= self._publish_queue_capacity:
                with self._statistics_lock:
                    self._statistics_of(topic).send_dropped += 1
                return False
            was_empty = not self._queue
            self._queue.append((topic, type_name, bytes(payload)))
        # The receive thread empties the queue on one wake-up, so only the first message needs to wake it.
        if was_empty:
            self._wake()
        return True

    def publish_from_io_thread(
        self, topic: str, message: protobuf_message.Message
    ) -> None:
        """Publishes from a callback or a periodic callback, straight to the Zenoh publisher.

        Args:
            topic: The topic to publish on.
            message: The message, written after whatever publish() queued before it.

        Raises:
            BusError: called from another thread, or the bus only subscribes.
        """
        if not self.is_io_thread():
            raise BusError(
                "publish_from_io_thread() outside the bus's receive thread;"
                " call publish()"
            )
        self._check_can_publish(topic)
        # What publish() queued before goes first, so that the messages of one thread keep their order.
        self._send_queued()
        self._send(
            topic, message.DESCRIPTOR.full_name, message.SerializeToString()
        )

    def _check_can_publish(self, topic: str) -> None:
        if not self._node_name:
            raise BusError(
                "this bus has no node name, so it only subscribes and cannot"
                " publish"
            )
        if not isinstance(topic, str):
            raise ValueError("the topic is not a string")
        problem = network_config.topic_name_problem(topic)
        if problem is not None:
            raise ValueError(problem)

    def _wake(self) -> None:
        self._woken.set()

    # ------------------------------------------------------------------------------------------------------------------
    # Introspection (from any thread)
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def node_name(self) -> str:
        """The node this bus publishes as; empty when it only subscribes."""
        return self._node_name

    @property
    def domain_id(self) -> int:
        """The domain of the bus."""
        return self._network.domain_id

    @property
    def discovery_peers(self) -> list[str]:
        """The hosts the bus connects to: the distinct hosts of the network's nodes."""
        return self._network.peer_hosts()

    @property
    def domain_config(self) -> str:
        """The Zenoh configuration the bus's session was opened with."""
        return self._session_config

    @property
    def periodic_callback_errors(self) -> int:
        """Periodic callback calls that raised."""
        return self._periodic_callback_errors

    def topic_statistics(self, topic: str) -> TopicStatistics:
        """The counters of one topic (all zero for a topic the bus has not seen)."""
        with self._statistics_lock:
            statistics = self._statistics.get(topic)
            return (
                dataclasses.replace(statistics)
                if statistics
                else TopicStatistics()
            )

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
        """The receive thread: waits to be woken, puts, drains and runs the due periodic callbacks."""
        self._io_thread_ident = threading.get_ident()
        try:
            while not self._stop_requested.is_set():
                try:
                    self._woken.wait(self._wait_timeout())
                    self._woken.clear()
                    self._send_queued()
                    self._apply_token_events()
                    self._drain()
                    self._run_due_callbacks(time.monotonic())
                # pylint: disable-next=broad-exception-caught  # The outermost loop of the receive thread.
                except Exception:
                    # Nothing escapes the receive thread.
                    self._log.warning(
                        "loop",
                        "the receive thread caught an exception",
                        exc_info=True,
                    )
            try:
                self._send_queued()
                self._apply_token_events()
            # pylint: disable-next=broad-exception-caught  # The receive thread's last act; nothing escapes it.
            except Exception:
                self._log.warning(
                    "stop",
                    "sending the queued messages at stop() failed",
                    exc_info=True,
                )
        finally:
            self._io_thread_ident = None

    def _wait_timeout(self) -> float:
        wait = self._io_poll_period
        now = time.monotonic()
        for periodic in self._periodic_callbacks:
            wait = min(wait, periodic.next_deadline - now)
        return max(wait, 0.0)

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
        """Puts one sample, and counts it as written or dropped.

        Args:
            topic: The topic.
            type_name: The full protobuf type name of the payload.
            payload: The serialized message.
        """
        try:
            writer = self._writer_for(topic)
            sample = envelope.Envelope(
                protocol_version=envelope.WIRE_PROTOCOL_VERSION,
                writer_id=writer.writer_id,
                sequence_number=writer.last_sequence + 1,
                source_timestamp_ns=self._next_source_timestamp(),
                type_name=type_name,
                payload=payload,
            )
            writer.publisher.put(sample.encode())
        except zenoh.ZError as error:
            with self._statistics_lock:
                self._statistics_of(topic).send_dropped += 1
            self._log.warning(
                f"send:{topic}",
                f"putting a sample on '{topic}' failed: {error}",
            )
            return
        writer.last_sequence = sample.sequence_number
        with self._statistics_lock:
            writer.statistics.written += 1

    def _decode(self, topic: str, data: bytes) -> envelope.Envelope | None:
        """The envelope of `data` when it is of this bus's wire protocol; logs why not."""
        version = envelope.protocol_version_of(data)
        if version != envelope.WIRE_PROTOCOL_VERSION:
            self._log.warning(
                f"protocol:{topic}",
                f"a sample on '{topic}' speaks wire protocol {version}, but this"
                f" bus speaks {envelope.WIRE_PROTOCOL_VERSION}; rebuild both"
                " from the same checkout",
            )
            return None
        try:
            return envelope.decode(data)
        except envelope.EnvelopeError as error:
            self._log.warning(
                f"envelope:{topic}", f"a sample on '{topic}': {error}"
            )
            return None

    def _take(
        self,
        subscription: _Subscription,
        budget: int,
        pending: list[_PendingMessage],
    ) -> int:
        """Takes up to `budget` samples of `subscription`'s queue into `pending`; returns how many it took."""
        inbox = subscription.inbox
        if inbox is None:
            return 0
        taken = inbox.take(budget)
        for data in taken:
            sample = self._decode(subscription.topic, data)
            if sample is not None:
                self._track_sequence(
                    subscription, sample.writer_id, sample.sequence_number
                )
            accepted = sample is not None and self._accepts_type(
                subscription, sample
            )
            with self._statistics_lock:
                subscription.statistics.received += 1
                if not accepted:
                    subscription.statistics.rejected += 1
            if accepted and sample is not None:
                pending.append(
                    _PendingMessage(subscription, subscription.topic, sample)
                )
        if len(inbox):
            # What the budget left behind waits for the next drain, which comes at once.
            self._wake()
        return len(taken)

    def _take_all_topics(
        self, budget: int, pending: list[_PendingMessage]
    ) -> int:
        """Takes up to `budget` samples of the subscription to every topic into `pending`."""
        inbox = self._all_topics_inbox
        if inbox is None:
            return 0
        taken = inbox.take(budget)
        for key_expr, data in taken:
            key = key_exprs.parse_data_key_expr(self._network, key_expr)
            if (
                key is None
                or envelope.protocol_version_of(data) != WIRE_PROTOCOL_VERSION
            ):
                continue
            try:
                pending.append(
                    _PendingMessage(None, key.topic, envelope.decode(data))
                )
            except envelope.EnvelopeError:
                continue
        if len(inbox):
            self._wake()
        return len(taken)

    def _track_sequence(
        self, subscription: _Subscription, writer: int, sequence: int
    ) -> None:
        """Counts the gap between the last sequence number of `writer` and `sequence` as missed."""
        last = subscription.last_sequence.get(writer)
        subscription.last_sequence[writer] = max(sequence, last or 0)
        if last is not None and sequence > last + 1:
            with self._statistics_lock:
                subscription.statistics.missed += sequence - last - 1

    def _accepts_type(
        self, subscription: _Subscription, sample: envelope.Envelope
    ) -> bool:
        """Whether `sample` is of the subscription's type; logs why not."""
        if (
            subscription.expected_type
            and sample.type_name != subscription.expected_type
        ):
            self._log.warning(
                f"type:{subscription.topic}",
                f"'{subscription.topic}' carries a {sample.type_name}, but its"
                f" subscription takes a {subscription.expected_type}",
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
            budget -= self._take(subscription, budget, pending)
        if budget > 0:
            self._take_all_topics(budget, pending)
        pending.sort(key=lambda message: message.sample.source_timestamp_ns)
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
                message.topic,
                message.sample.type_name,
                message.sample.payload,
            )
        # pylint: disable-next=broad-exception-caught  # A raising callback must not end the receive thread.
        except Exception:
            self._log.warning(
                "all_topics",
                "the callback of every topic raised",
                exc_info=True,
            )

    def _dispatch(
        self, subscription: _Subscription, sample: envelope.Envelope
    ) -> None:
        """Parses one sample of `subscription` and passes it to its callback, counting a rejection or a raise."""
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
                    f"a sample on '{subscription.topic}' does not parse as a"
                    f" {subscription.expected_type}",
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

    def _apply_token_events(self) -> None:
        """Applies the liveliness changes Zenoh delivered, and counts the peers again when there were any."""
        events = self._token_events.take(len(self._token_events))
        if not events:
            return
        for key_expr, alive in events:
            if not alive:
                self._live_tokens.pop(key_expr, None)
                continue
            token = key_exprs.parse_endpoint_token_key_expr(
                self._network, key_expr
            )
            if token is not None:
                self._live_tokens[key_expr] = token
        self._count_peers()

    def _count_peers(self) -> None:
        """Counts the matching and the incompatible tokens of each topic this bus publishes or subscribes."""
        incompatible_of_writers: dict[str, int] = {}
        for name, writer in self._writers.items():
            readers = [
                token
                for token in self._live_tokens.values()
                if token.role is key_exprs.EndpointRole.SUBSCRIBER
                and token.topic == name
            ]
            matched = sum(token.profile is writer.profile for token in readers)
            incompatible_of_writers[name] = len(readers) - matched
            with self._statistics_lock:
                writer.statistics.matched_readers = matched
                writer.statistics.incompatible_qos = incompatible_of_writers[
                    name
                ]
        for subscription in self._subscriptions.values():
            writers = [
                token
                for token in self._live_tokens.values()
                if token.role is key_exprs.EndpointRole.PUBLISHER
                and token.topic == subscription.topic
            ]
            matched = sum(
                token.profile is subscription.profile for token in writers
            )
            with self._statistics_lock:
                subscription.statistics.matched_writers = matched
                # A topic this bus both publishes and subscribes counts the incompatible tokens of both roles.
                subscription.statistics.incompatible_qos = (
                    incompatible_of_writers.get(subscription.topic, 0)
                    + len(writers)
                    - matched
                )

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
                missed = math.floor(
                    (after - periodic.next_deadline) / periodic.period
                )
                periodic.next_deadline += (missed + 1) * periodic.period


__all__ = [
    "WIRE_PROTOCOL_VERSION",
    "Bus",
    "BusError",
    "Delivery",
    "TopicStatistics",
    "parse_delivery",
]
