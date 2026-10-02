"""The ZeroMQ bus of the distributed runtime, in Python: the twin of robot_ipc/Bus.h.

A bus binds one PUB socket at the endpoint of its node (unless it only subscribes) and connects one SUB socket to every
endpoint of the network, its own included. Every message is three frames: the topic, the full protobuf type name and
the serialized message, exactly as the C++ bus frames it.

Both sockets belong to the bus's receive thread, which start() launches. Callbacks and periodic callbacks run on it,
one at a time. publish() may be called from any thread: it serializes on the caller and queues the frames for the
receive thread, which an eventfd wakes at once.

    with Bus("operator", load_network_config("config/ipc/network.textproto")) as bus:
        ...  # subscribe() before entering, publish() inside
"""

import collections
import dataclasses
import enum
import logging
import math
import os
import threading
import time
import weakref
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, Type, Union

import zmq
from google.protobuf import message as protobuf_message

from robot_ipc.network_config import (
    EPHEMERAL_PORT,
    NetworkConfig,
    NetworkConfigError,
    validate_network_config,
)

_LOGGER = logging.getLogger("robot_ipc")

# The receive thread logs a recurring problem at most this often, in seconds.
_LOG_PERIOD = 5.0

# Socket options the C++ bus sets alike.
# LINT.IfChange(socket_options)
_LINGER_MS = 0
_TCP_KEEPALIVE_ON = 1
_TCP_KEEPALIVE_IDLE_S = 2
_TCP_KEEPALIVE_INTERVAL_S = 1
_TCP_KEEPALIVE_PROBES = 3
_MAX_HEARTBEAT_TTL_MS = 6553599
# LINT.ThenChange(//robot_runtime/robot_ipc/src/Bus.cpp:socket_options)


# LINT.IfChange(delivery_names)
class Delivery(str, enum.Enum):
    """What a subscription is handed of a topic each time the receive thread drains its socket."""

    # Only the newest message of the topic; the older ones of the same drain are dropped unparsed (superseded).
    LATEST = "latest"
    # Every message, in the order it arrived.
    ALL = "all"


# LINT.ThenChange(//robot_runtime/robot_ipc/src/Delivery.cpp:delivery_names)


def parse_delivery(value: Union[str, Delivery]) -> Delivery:
    """The delivery of that name ("latest" or "all").

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


@dataclasses.dataclass
class TopicStatistics:
    """What a bus did with one topic; see robot_ipc/TopicStatistics.h.

    Once a drain has dispatched, received == delivered + superseded + rejected.
    """

    # Messages written to the PUB socket (ZeroMQ may still drop one at a slow subscriber's high-water mark).
    sent: int = 0
    # publish() calls refused because the queue to the receive thread was full.
    send_dropped: int = 0
    # Messages read from the SUB socket with exactly this topic.
    received: int = 0
    # Messages handed to the callback (including those whose callback then raised).
    delivered: int = 0
    # Delivery.LATEST only: messages replaced by a newer one of the same drain before they were parsed.
    superseded: int = 0
    # Messages that were not three frames, carried another type than the subscription's, or did not parse.
    rejected: int = 0
    # Callback calls that raised; the bus logs and counts them and carries on.
    handler_errors: int = 0


class BusError(RuntimeError):
    """The bus cannot do what was asked in its current state, or cannot bind its endpoint."""


MessageCallback = Callable[[protobuf_message.Message], None]
RawCallback = Callable[[str, bytes], None]
TopicCallback = Callable[[str, str, bytes], None]


@dataclasses.dataclass
class _Subscription:
    topic: str
    delivery: Delivery
    callback: Callable[..., None]
    # None for subscribe_raw(), which takes any type.
    message_class: Optional[Type[protobuf_message.Message]]
    expected_type: bytes
    statistics: TopicStatistics
    # Delivery.LATEST: (type name, payload) of the newest message of the current drain, not yet parsed.
    latest: Optional[Tuple[bytes, bytes]] = None


@dataclasses.dataclass
class _PeriodicCallback:
    period: float
    callback: Callable[[], None]
    next_deadline: float = 0.0


class _RateLimitedLog:
    """Logs each kind of problem at most once per _LOG_PERIOD."""

    def __init__(self) -> None:
        self._last: Dict[str, float] = {}

    def warning(self, kind: str, text: str, exc_info: bool = False) -> None:
        now = time.monotonic()
        if now - self._last.get(kind, -math.inf) >= _LOG_PERIOD:
            self._last[kind] = now
            _LOGGER.warning("robot_ipc: %s", text, exc_info=exc_info)


def _milliseconds(seconds: float) -> int:
    return max(0, int(round(seconds * 1000.0)))


def _port_of(endpoint: str) -> int:
    try:
        return int(endpoint.rsplit(":", 1)[1])
    except (IndexError, ValueError):
        return EPHEMERAL_PORT


def _self_connect_endpoint(bound_endpoint: str) -> str:
    """The bound endpoint, over loopback if it is bound to every interface."""
    wildcard = "tcp://0.0.0.0:"
    if bound_endpoint.startswith(wildcard):
        return "tcp://127.0.0.1:" + bound_endpoint[len(wildcard) :]
    return bound_endpoint


class Bus:
    """A publish/subscribe bus without a broker, framed like the C++ robot::ipc::Bus.

    Args:
        node_name: the node this process publishes as; the bus binds its endpoint. Empty: only subscribe.
        network: the nodes; the SUB socket connects to every one of them, this process's own included.
        io_poll_period: the longest the receive thread waits, in seconds; publish(), stop(), arriving messages and
            due periodic callbacks all wake it at once.
        send_high_water_mark: ZMQ_SNDHWM (0: no limit).
        receive_high_water_mark: ZMQ_RCVHWM (0: no limit).
        publish_queue_capacity: messages publish() queues before it refuses (counted as send_dropped).
        max_messages_per_drain: the most messages one drain reads before the thread dispatches.
        heartbeat_interval: ZMTP heartbeat interval in seconds (0: none).
        heartbeat_timeout: a peer silent this long, in seconds, is disconnected.
        reconnect_interval: how often, in seconds, the SUB socket retries a node that is down...
        reconnect_interval_max: ...backing off up to this.

    Raises:
        NetworkConfigError: the network is invalid.
        ValueError: the node is not in the network, or an option is out of range.
        BusError: the node's endpoint cannot be bound.
    """

    # LINT.IfChange(bus_defaults)
    def __init__(
        self,
        node_name: str,
        network: NetworkConfig,
        *,
        io_poll_period: float = 0.1,
        send_high_water_mark: int = 1000,
        receive_high_water_mark: int = 1000,
        publish_queue_capacity: int = 1024,
        max_messages_per_drain: int = 4096,
        heartbeat_interval: float = 1.0,
        heartbeat_timeout: float = 3.0,
        reconnect_interval: float = 0.1,
        reconnect_interval_max: float = 1.0,
    ) -> None:
        # LINT.ThenChange(//robot_runtime/robot_ipc/include/robot_ipc/BusOptions.h:bus_defaults)
        validate_network_config(network)
        node = network.find(node_name) if node_name else None
        if node_name and node is None:
            raise ValueError(
                f"node '{node_name}' is not in the network (nodes: {', '.join(network.node_names())})"
            )
        if io_poll_period <= 0:
            raise ValueError("io_poll_period must be positive")
        if send_high_water_mark < 0 or receive_high_water_mark < 0:
            raise ValueError("the high-water marks must be >= 0 (0: no limit)")
        if publish_queue_capacity <= 0 or max_messages_per_drain <= 0:
            raise ValueError(
                "publish_queue_capacity and max_messages_per_drain must be positive"
            )
        if heartbeat_interval < 0 or heartbeat_timeout < 0:
            raise ValueError("heartbeat_interval and heartbeat_timeout must be >= 0")
        if reconnect_interval <= 0 or reconnect_interval_max < 0:
            raise ValueError(
                "reconnect_interval must be positive and reconnect_interval_max >= 0"
            )

        self._node_name = node_name
        self._io_poll_period = io_poll_period
        self._publish_queue_capacity = publish_queue_capacity
        self._max_messages_per_drain = max_messages_per_drain
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat_timeout = heartbeat_timeout
        self._reconnect_interval = reconnect_interval
        self._reconnect_interval_max = reconnect_interval_max

        self._lifecycle_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._io_thread_ident: Optional[int] = None
        self._stop_requested = threading.Event()
        self._closed = False

        self._subscriptions: Dict[bytes, _Subscription] = {}
        self._latest_subscriptions: List[_Subscription] = []
        self._all_topics_callback: Optional[TopicCallback] = None
        self._periodic_callbacks: List[_PeriodicCallback] = []
        self._periodic_callback_errors = 0

        self._queue_lock = threading.Lock()
        self._queue: Deque[Tuple[str, Tuple[bytes, bytes, bytes]]] = collections.deque()
        self._statistics_lock = threading.Lock()
        self._statistics: Dict[str, TopicStatistics] = {}
        self._log = _RateLimitedLog()

        self._bound_endpoint = ""
        self._subscriber_endpoints: List[str] = []
        self._wake_fd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
        # Closed with the bus object, not by close(): a thread that still holds the bus may call publish() after
        # close(), and a closed descriptor's number can be handed to another file or socket of the process, which a
        # wake-up would then write into. close() makes publish() refuse instead.
        self._wake_fd_finalizer = weakref.finalize(self, os.close, self._wake_fd)
        self._context = zmq.Context()
        self._publisher: Optional[zmq.Socket] = None
        try:
            if node is not None:
                self._publisher = self._context.socket(zmq.PUB)
                self._configure_socket(self._publisher)
                self._publisher.setsockopt(zmq.SNDHWM, send_high_water_mark)
                endpoint = node.bind_endpoint()
                try:
                    self._publisher.bind(endpoint)
                except zmq.ZMQError as error:
                    raise BusError(
                        f"cannot bind node '{node.name}' at {endpoint}: {error}"
                    ) from error
                self._bound_endpoint = self._publisher.getsockopt_string(
                    zmq.LAST_ENDPOINT
                )
            self._subscriber = self._context.socket(zmq.SUB)
            self._configure_socket(self._subscriber)
            self._subscriber.setsockopt(zmq.RCVHWM, receive_high_water_mark)
            # No pipe for a node that is not up: the subscriptions go out on the connection, when it completes.
            self._subscriber.setsockopt(zmq.IMMEDIATE, 1)
            for peer in network.nodes:
                if node is not None and peer.name == node.name:
                    endpoint = _self_connect_endpoint(self._bound_endpoint)
                elif peer.port == EPHEMERAL_PORT:
                    # Bound wherever its kernel chose; reachable through connect() only.
                    continue
                else:
                    endpoint = peer.connect_endpoint()
                try:
                    self._subscriber.connect(endpoint)
                except zmq.ZMQError as error:
                    raise ValueError(
                        f"cannot connect to node '{peer.name}' at {endpoint}: {error}"
                    ) from error
                self._subscriber_endpoints.append(endpoint)
        except BaseException:
            self._close_sockets()
            # Nothing else holds the bus that failed to construct.
            self._wake_fd_finalizer()
            raise

    def _configure_socket(self, socket: zmq.Socket) -> None:
        socket.setsockopt(zmq.LINGER, _LINGER_MS)
        socket.setsockopt(zmq.HEARTBEAT_IVL, _milliseconds(self._heartbeat_interval))
        socket.setsockopt(zmq.HEARTBEAT_TIMEOUT, _milliseconds(self._heartbeat_timeout))
        socket.setsockopt(
            zmq.HEARTBEAT_TTL,
            min(_milliseconds(self._heartbeat_timeout), _MAX_HEARTBEAT_TTL_MS),
        )
        socket.setsockopt(zmq.TCP_KEEPALIVE, _TCP_KEEPALIVE_ON)
        socket.setsockopt(zmq.TCP_KEEPALIVE_IDLE, _TCP_KEEPALIVE_IDLE_S)
        socket.setsockopt(zmq.TCP_KEEPALIVE_INTVL, _TCP_KEEPALIVE_INTERVAL_S)
        socket.setsockopt(zmq.TCP_KEEPALIVE_CNT, _TCP_KEEPALIVE_PROBES)
        socket.setsockopt(zmq.RECONNECT_IVL, _milliseconds(self._reconnect_interval))
        socket.setsockopt(
            zmq.RECONNECT_IVL_MAX, _milliseconds(self._reconnect_interval_max)
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Configuration (while the bus is not running)
    # ------------------------------------------------------------------------------------------------------------------

    def subscribe(
        self,
        topic: str,
        message_class: Type[protobuf_message.Message],
        callback: MessageCallback,
        delivery: Union[str, Delivery] = Delivery.LATEST,
    ) -> None:
        """Hands every message of exactly `topic` that carries `message_class` to `callback`, on the receive thread.

        A message of another type, or one that does not parse, is rejected and counted. One subscription per topic.
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
                expected_type=descriptor.full_name.encode(),
                statistics=TopicStatistics(),
            )
        )

    def subscribe_raw(
        self,
        topic: str,
        callback: RawCallback,
        delivery: Union[str, Delivery] = Delivery.LATEST,
    ) -> None:
        """Hands every message of exactly `topic` to `callback(type_name, payload)`, whatever its type."""
        self._add_subscription(
            _Subscription(
                topic=topic,
                delivery=parse_delivery(delivery),
                callback=callback,
                message_class=None,
                expected_type=b"",
                statistics=TopicStatistics(),
            )
        )

    def subscribe_all_topics(self, callback: TopicCallback) -> None:
        """Hands every message of every topic to `callback(topic, type_name, payload)`, in order (for tools/ipc).

        It sees the messages of the topics that have their own subscription too. It keeps no per-topic statistics.
        """
        if not callable(callback):
            raise ValueError("the callback is not callable")
        with self._lifecycle_lock:
            self._check_configurable("subscribe to every topic")
            if self._all_topics_callback is not None:
                raise ValueError("the bus already has a subscription to every topic")
            self._subscriber.setsockopt(zmq.SUBSCRIBE, b"")
            self._all_topics_callback = callback

    def _add_subscription(self, subscription: _Subscription) -> None:
        if not isinstance(subscription.topic, str) or not subscription.topic:
            raise ValueError(
                "the topic is empty (ZeroMQ would match every topic with it)"
            )
        if not callable(subscription.callback):
            raise ValueError("the callback is not callable")
        key = subscription.topic.encode()
        with self._lifecycle_lock:
            self._check_configurable(f"subscribe to '{subscription.topic}'")
            if key in self._subscriptions:
                raise ValueError(
                    f"the topic '{subscription.topic}' already has a subscription"
                )
            self._subscriber.setsockopt(zmq.SUBSCRIBE, key)
            with self._statistics_lock:
                subscription.statistics = self._statistics.setdefault(
                    subscription.topic, subscription.statistics
                )
            self._subscriptions[key] = subscription
            if subscription.delivery is Delivery.LATEST:
                self._latest_subscriptions.append(subscription)

    def add_periodic_callback(
        self, period: float, callback: Callable[[], None]
    ) -> None:
        """Calls `callback` on the receive thread every `period` seconds, on absolute deadlines from start()."""
        if period <= 0:
            raise ValueError("a periodic callback needs a positive period")
        if not callable(callback):
            raise ValueError("the periodic callback is not callable")
        with self._lifecycle_lock:
            self._check_configurable("add a periodic callback")
            self._periodic_callbacks.append(_PeriodicCallback(period, callback))

    def connect(self, endpoint: str) -> None:
        """Connects the SUB socket to one more endpoint, such as a node bound to EPHEMERAL_PORT."""
        with self._lifecycle_lock:
            self._check_configurable(f"connect to '{endpoint}'")
            try:
                self._subscriber.connect(endpoint)
            except zmq.ZMQError as error:
                raise ValueError(f"cannot connect to '{endpoint}': {error}") from error
            self._subscriber_endpoints.append(endpoint)

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
        """Stops the bus and closes its sockets; the bus cannot be started again. Idempotent.

        From the call on, publish() raises BusError; what was queued before it is sent first. A publish() racing the
        call may be dropped.

        Raises:
            BusError: called from one of the bus's own callbacks, whose sockets are still in use.
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
        with self._lifecycle_lock:
            self._close_sockets()

    def _close_sockets(self) -> None:
        for socket in (self._publisher, getattr(self, "_subscriber", None)):
            if socket is not None:
                socket.close(linger=_LINGER_MS)
        self._context.term()

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

        Returns:
            False when the queue to the receive thread was full and the message was dropped (counted).

        Raises:
            BusError: the bus has no node name, so it only subscribes; or it is closed.
            ValueError: the topic is empty.
        """
        return self.publish_raw(
            topic, message.DESCRIPTOR.full_name, message.SerializeToString()
        )

    def publish_raw(self, topic: str, type_name: str, payload: bytes) -> bool:
        """publish() of an already serialized message of the type `type_name` names (for tools/ipc)."""
        self._check_can_publish(topic)
        frames = (topic.encode(), type_name.encode(), bytes(payload))
        with self._queue_lock:
            if self._closed:
                raise BusError(f"cannot publish on '{topic}': the bus is closed")
            if len(self._queue) >= self._publish_queue_capacity:
                with self._statistics_lock:
                    self._statistics_of(topic).send_dropped += 1
                return False
            was_empty = not self._queue
            self._queue.append((topic, frames))
        # The receive thread empties the whole queue on one wake-up, so only the first message needs to wake it. The
        # eventfd lives as long as the bus object, so a wake-up that races close() lands in it.
        if was_empty:
            self._wake()
        return True

    def publish_from_io_thread(
        self, topic: str, message: protobuf_message.Message
    ) -> None:
        """Publishes from a callback or a periodic callback, straight onto the socket.

        Raises:
            BusError: called from another thread, or the bus only subscribes.
        """
        if not self.is_io_thread():
            raise BusError(
                "publish_from_io_thread() outside the bus's receive thread; call publish()"
            )
        self._check_can_publish(topic)
        frames = (
            topic.encode(),
            message.DESCRIPTOR.full_name.encode(),
            message.SerializeToString(),
        )
        # What publish() queued before goes first, so that the messages of one thread keep their order.
        self._send_queued()
        self._send(topic, frames)

    def _check_can_publish(self, topic: str) -> None:
        if self._publisher is None:
            raise BusError(
                "this bus has no node name, so it only subscribes and cannot publish"
            )
        if not isinstance(topic, str) or not topic:
            raise ValueError(
                "the topic is empty (ZeroMQ would match every topic with it)"
            )

    def _wake(self) -> None:
        try:
            os.eventfd_write(self._wake_fd, 1)
        except OSError:
            # The counter is full: the receive thread has a wake-up pending already.
            pass

    def _clear_wake(self) -> None:
        try:
            os.eventfd_read(self._wake_fd)
        except BlockingIOError:
            pass

    # ------------------------------------------------------------------------------------------------------------------
    # Introspection (from any thread)
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def node_name(self) -> str:
        return self._node_name

    @property
    def bound_endpoint(self) -> str:
        """The endpoint the PUB socket is bound to, e.g. "tcp://127.0.0.1:5600"; empty when it only subscribes."""
        return self._bound_endpoint

    @property
    def bound_port(self) -> int:
        """The port of bound_endpoint (the kernel's choice for EPHEMERAL_PORT), or EPHEMERAL_PORT."""
        return (
            _port_of(self._bound_endpoint) if self._bound_endpoint else EPHEMERAL_PORT
        )

    @property
    def subscriber_endpoints(self) -> List[str]:
        """The endpoints the SUB socket is connected to, in the order they were connected."""
        with self._lifecycle_lock:
            return list(self._subscriber_endpoints)

    @property
    def periodic_callback_errors(self) -> int:
        """Periodic callback calls that raised."""
        return self._periodic_callback_errors

    def topic_statistics(self, topic: str) -> TopicStatistics:
        """The counters of one topic (all zero for a topic the bus has not seen)."""
        with self._statistics_lock:
            statistics = self._statistics.get(topic)
            return dataclasses.replace(statistics) if statistics else TopicStatistics()

    def statistics(self) -> Dict[str, TopicStatistics]:
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
        self._io_thread_ident = threading.get_ident()
        poller = zmq.Poller()
        poller.register(self._subscriber, zmq.POLLIN)
        poller.register(self._wake_fd, zmq.POLLIN)
        try:
            while not self._stop_requested.is_set():
                try:
                    events = dict(poller.poll(self._poll_timeout_ms()))
                    if self._wake_fd in events:
                        self._clear_wake()
                    self._send_queued()
                    if self._subscriber in events:
                        self._drain()
                    self._run_due_callbacks(time.monotonic())
                except zmq.ContextTerminated:
                    break
                except Exception:  # pylint: disable=broad-except
                    # Nothing escapes the receive thread.
                    self._log.warning(
                        "loop", "the receive thread caught an exception", exc_info=True
                    )
            try:
                self._send_queued()
            except Exception:  # pylint: disable=broad-except
                self._log.warning(
                    "stop",
                    "sending the queued messages at stop() failed",
                    exc_info=True,
                )
        finally:
            self._io_thread_ident = None

    def _poll_timeout_ms(self) -> int:
        wait = self._io_poll_period
        now = time.monotonic()
        for periodic in self._periodic_callbacks:
            wait = min(wait, periodic.next_deadline - now)
        if wait <= 0:
            return 0
        # Rounded up, so that the thread does not wake just before a deadline and spin until it.
        return int(math.ceil(wait * 1000.0))

    def _send_queued(self) -> None:
        with self._queue_lock:
            if not self._queue:
                return
            sending = self._queue
            self._queue = collections.deque()
        for topic, frames in sending:
            self._send(topic, frames)

    def _send(self, topic: str, frames: Tuple[bytes, bytes, bytes]) -> None:
        assert self._publisher is not None
        try:
            # A PUB socket never blocks: at a subscriber's high-water mark ZeroMQ drops the message for it.
            self._publisher.send_multipart(frames, zmq.NOBLOCK)
            sent = True
        except zmq.Again:
            sent = False
        except zmq.ZMQError as error:
            sent = False
            self._log.warning("send", f"sending on '{topic}' failed: {error}")
        with self._statistics_lock:
            statistics = self._statistics_of(topic)
            if sent:
                statistics.sent += 1
            else:
                statistics.send_dropped += 1

    def _drain(self) -> None:
        for _ in range(self._max_messages_per_drain):
            try:
                frames = self._subscriber.recv_multipart(zmq.NOBLOCK)
            except zmq.Again:
                break
            if self._all_topics_callback is not None:
                self._dispatch_all_topics(frames)
            # ZeroMQ matched a prefix; the topic must match whole.
            subscription = self._subscriptions.get(frames[0])
            if subscription is None:
                continue
            with self._statistics_lock:
                subscription.statistics.received += 1
                if len(frames) != 3:
                    subscription.statistics.rejected += 1
            if len(frames) != 3:
                self._log.warning(
                    "frames",
                    f"a message on '{subscription.topic}' has {len(frames)} frames instead of three "
                    "(topic, type name, payload)",
                )
                continue
            if subscription.delivery is Delivery.ALL:
                self._dispatch(subscription, frames[1], frames[2])
                continue
            if subscription.latest is not None:
                with self._statistics_lock:
                    subscription.statistics.superseded += 1
            subscription.latest = (frames[1], frames[2])

        for subscription in self._latest_subscriptions:
            if subscription.latest is not None:
                type_name, payload = subscription.latest
                subscription.latest = None
                self._dispatch(subscription, type_name, payload)

    def _dispatch_all_topics(self, frames: List[bytes]) -> None:
        if len(frames) != 3:
            return
        callback = self._all_topics_callback
        assert callback is not None
        try:
            callback(
                frames[0].decode("utf-8", errors="replace"),
                frames[1].decode("utf-8", errors="replace"),
                frames[2],
            )
        except Exception:  # pylint: disable=broad-except
            self._log.warning(
                "all_topics", "the callback of every topic raised", exc_info=True
            )

    def _dispatch(
        self, subscription: _Subscription, type_name: bytes, payload: bytes
    ) -> None:
        statistics = subscription.statistics
        if subscription.expected_type and type_name != subscription.expected_type:
            with self._statistics_lock:
                statistics.rejected += 1
            self._log.warning(
                f"type:{subscription.topic}",
                f"'{subscription.topic}' carries a {type_name.decode(errors='replace')}, but its subscription "
                f"takes a {subscription.expected_type.decode()}",
            )
            return
        arguments: Tuple[Any, ...]
        if subscription.message_class is not None:
            try:
                arguments = (subscription.message_class.FromString(payload),)
            except protobuf_message.DecodeError:
                with self._statistics_lock:
                    statistics.rejected += 1
                self._log.warning(
                    f"parse:{subscription.topic}",
                    f"a message on '{subscription.topic}' does not parse as a {subscription.expected_type.decode()}",
                )
                return
        else:
            arguments = (type_name.decode("utf-8", errors="replace"), payload)
        with self._statistics_lock:
            statistics.delivered += 1
        try:
            subscription.callback(*arguments)
        except Exception:  # pylint: disable=broad-except
            with self._statistics_lock:
                statistics.handler_errors += 1
            self._log.warning(
                f"handler:{subscription.topic}",
                f"the callback of '{subscription.topic}' raised",
                exc_info=True,
            )

    def _run_due_callbacks(self, now: float) -> None:
        for periodic in self._periodic_callbacks:
            if now < periodic.next_deadline:
                continue
            try:
                periodic.callback()
            except Exception:  # pylint: disable=broad-except
                self._periodic_callback_errors += 1
                self._log.warning(
                    "periodic", "a periodic callback raised", exc_info=True
                )
            periodic.next_deadline += periodic.period
            if periodic.next_deadline <= now:
                # A whole period late: skip the missed calls instead of bursting through them.
                periodic.next_deadline = now + periodic.period


__all__ = [
    "Bus",
    "BusError",
    "Delivery",
    "NetworkConfigError",
    "TopicStatistics",
    "parse_delivery",
]
