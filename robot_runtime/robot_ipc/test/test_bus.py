"""The Python bus over loopback TCP: the properties testBus.cpp checks for the C++ bus.

Exact topic matching, Delivery.LATEST against Delivery.ALL, publishing from several threads, processes that start in
either order or restart, a prompt stop(), the publish queue's drop counting, rejection of malformed messages,
callbacks that raise, and decode(). Every publisher binds an ephemeral port (EPHEMERAL_PORT) that the subscriber's
network then names, so concurrent tests never collide on a port.
"""

import gc
import os
import threading
import time
import unittest
from typing import Callable, List, Tuple

import zmq

import robot_ipc
from robot_ipc import (
    EPHEMERAL_PORT,
    Bus,
    BusError,
    Delivery,
    NetworkConfig,
    NetworkConfigError,
    NodeEndpoint,
)
from robot_ipc_test import test_event_pb2
from robot_ipc_test import test_sample_pb2

TIMEOUT = 20.0
PUBLISHER = "publisher"
PROBE_TOPIC = "test/probe"


def wait_for(condition: Callable[[], bool], timeout: float = TIMEOUT) -> bool:
    """Polls `condition` until it holds or `timeout` seconds pass."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.001)
    return condition()


def loopback_network(name: str, port: int) -> NetworkConfig:
    return NetworkConfig(nodes=(NodeEndpoint(name, "127.0.0.1", port),))


def make_sample(sequence: int, publisher: int = 0) -> test_sample_pb2.TestSample:
    return test_sample_pb2.TestSample(
        sequence=sequence, publisher=publisher, text="sample", values=[float(sequence)]
    )


class Collector:
    """Collects what a callback receives, on the receive thread, for the test thread to read."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._samples: List[test_sample_pb2.TestSample] = []

    def __call__(self, sample: test_sample_pb2.TestSample) -> None:
        with self._lock:
            self._samples.append(sample)

    def samples(self) -> List[test_sample_pb2.TestSample]:
        with self._lock:
            return list(self._samples)

    def size(self) -> int:
        with self._lock:
            return len(self._samples)

    def contains(self, sequence: int) -> bool:
        with self._lock:
            return any(sample.sequence == sequence for sample in self._samples)


class BusTestCase(unittest.TestCase):
    """Creates buses that close when the test ends."""

    def publisher(self, port: int = EPHEMERAL_PORT, **options) -> Bus:
        bus = Bus(PUBLISHER, loopback_network(PUBLISHER, port), **options)
        self.addCleanup(bus.close)
        return bus

    def subscriber(self, publisher_port: int, **options) -> Bus:
        bus = Bus("", loopback_network(PUBLISHER, publisher_port), **options)
        self.addCleanup(bus.close)
        return bus

    @staticmethod
    def subscribe_probe(bus: Bus) -> None:
        bus.subscribe_raw(PROBE_TOPIC, lambda type_name, payload: None)

    def wait_until_connected(self, publisher: Bus, subscriber: Bus) -> None:
        """Publishes probes until the subscriber has received more than before (ZeroMQ's "slow joiner")."""
        before = subscriber.topic_statistics(PROBE_TOPIC).received
        probe = test_event_pb2.TestEvent(name="probe")

        def connected() -> bool:
            publisher.publish(PROBE_TOPIC, probe)
            time.sleep(0.005)
            return subscriber.topic_statistics(PROBE_TOPIC).received > before

        self.assertTrue(wait_for(connected), "the subscriber never connected")

    def connected_pair(self, **options) -> Tuple[Bus, Bus]:
        publisher = self.publisher(**options)
        subscriber = self.subscriber(publisher.bound_port, **options)
        self.subscribe_probe(subscriber)
        return publisher, subscriber

    def start_and_connect(self, publisher: Bus, subscriber: Bus) -> None:
        publisher.start()
        subscriber.start()
        self.wait_until_connected(publisher, subscriber)


class CreateTest(BusTestCase):

    def test_rejects_a_node_outside_the_network_and_lists_the_nodes(self) -> None:
        with self.assertRaises(ValueError) as raised:
            Bus("ghost", robot_ipc.localhost_network_config())
        self.assertIn("ghost", str(raised.exception))
        self.assertIn("robot", str(raised.exception))

    def test_rejects_an_invalid_network_and_options(self) -> None:
        with self.assertRaises(NetworkConfigError):
            Bus("", NetworkConfig(nodes=()))
        with self.assertRaises(ValueError):
            Bus(
                "",
                loopback_network(PUBLISHER, EPHEMERAL_PORT),
                publish_queue_capacity=0,
            )

    def test_reports_a_taken_endpoint(self) -> None:
        first = self.publisher()
        with self.assertRaises(BusError) as raised:
            Bus("second", loopback_network("second", first.bound_port))
        self.assertIn("second", str(raised.exception))
        self.assertIn(first.bound_endpoint, str(raised.exception))

    def test_ephemeral_port_is_reported_and_its_own_subscriber_connects_to_it(
        self,
    ) -> None:
        bus = self.publisher()
        self.assertEqual(bus.node_name, PUBLISHER)
        self.assertGreater(bus.bound_port, EPHEMERAL_PORT)
        self.assertTrue(bus.bound_endpoint.startswith("tcp://127.0.0.1:"))
        self.assertEqual(bus.subscriber_endpoints, [bus.bound_endpoint])

    def test_subscribe_only_bus_binds_nothing_and_cannot_publish(self) -> None:
        bus = self.subscriber(publisher_port=5999)
        self.assertEqual(bus.node_name, "")
        self.assertEqual(bus.bound_endpoint, "")
        self.assertEqual(bus.bound_port, EPHEMERAL_PORT)
        self.assertEqual(bus.subscriber_endpoints, ["tcp://127.0.0.1:5999"])
        with self.assertRaises(BusError):
            bus.publish("test/topic", make_sample(0))

    def test_publishing_on_a_closed_bus_is_refused(self) -> None:
        bus = self.publisher()
        bus.start()
        bus.close()
        with self.assertRaises(BusError) as raised:
            bus.publish("test/topic", make_sample(0))
        self.assertIn("closed", str(raised.exception))
        with self.assertRaises(BusError):
            bus.publish_raw("test/topic", "robot_ipc_test.TestSample", b"")

    def test_the_wake_up_descriptor_lives_as_long_as_the_bus_object(self) -> None:
        # A thread that still holds a closed bus may publish into it at any time. Were the eventfd closed with the
        # sockets, its number could already belong to another file or socket of the process, which the wake-up of
        # that publish would write eight bytes into.
        bus = Bus(PUBLISHER, loopback_network(PUBLISHER, EPHEMERAL_PORT))
        bus.start()
        wake_fd = bus._wake_fd  # pylint: disable=protected-access
        bus.close()
        os.fstat(wake_fd)  # Still the bus's: no other descriptor can take its number.
        del bus
        gc.collect()
        with self.assertRaises(OSError):
            os.fstat(wake_fd)

    def test_a_publish_racing_close_either_succeeds_before_it_or_is_refused(
        self,
    ) -> None:
        # A thread that publishes on its own, as the operator tools' streaming threads do, while the bus closes.
        for _ in range(20):
            bus = self.publisher()
            bus.start()
            outcomes: List[str] = []

            def stream() -> None:
                while True:
                    try:
                        bus.publish("test/topic", make_sample(0))
                    except BusError:
                        outcomes.append("refused")
                        return
                    except Exception as error:  # pylint: disable=broad-except
                        outcomes.append(repr(error))
                        return

            streamer = threading.Thread(target=stream)
            streamer.start()
            time.sleep(0.001)
            bus.close()
            streamer.join(timeout=TIMEOUT)
            self.assertFalse(streamer.is_alive())
            self.assertEqual(outcomes, ["refused"])

    def test_close_releases_the_port(self) -> None:
        with Bus(PUBLISHER, loopback_network(PUBLISHER, EPHEMERAL_PORT)) as bus:
            port = bus.bound_port
            self.assertTrue(bus.is_running())
        self.assertFalse(bus.is_running())
        with self.assertRaises(BusError):
            bus.start()
        self.publisher(port=port)


class LifecycleTest(BusTestCase):

    def test_configuration_is_refused_while_running_and_accepted_after_stop(
        self,
    ) -> None:
        bus = self.publisher()
        collector = Collector()
        with self.assertRaises(ValueError):
            bus.subscribe("", test_sample_pb2.TestSample, collector, "all")
        with self.assertRaises(ValueError):
            bus.subscribe("test/a", test_sample_pb2.TestSample, None, "all")
        with self.assertRaises(ValueError):
            bus.subscribe("test/a", test_sample_pb2.TestSample, collector, "newest")
        with self.assertRaises(ValueError):
            bus.add_periodic_callback(0.0, lambda: None)
        bus.subscribe("test/a", test_sample_pb2.TestSample, collector, "all")
        with self.assertRaises(ValueError):
            bus.subscribe("test/a", test_sample_pb2.TestSample, collector, "latest")

        self.assertFalse(bus.is_running())
        bus.start()
        bus.start()
        self.assertTrue(bus.is_running())
        with self.assertRaises(BusError):
            bus.subscribe("test/b", test_sample_pb2.TestSample, collector, "all")
        with self.assertRaises(BusError):
            bus.add_periodic_callback(0.001, lambda: None)
        with self.assertRaises(BusError):
            bus.connect("tcp://127.0.0.1:5999")

        bus.stop()
        bus.stop()
        self.assertFalse(bus.is_running())
        bus.subscribe("test/b", test_sample_pb2.TestSample, collector, Delivery.ALL)
        with self.assertRaises(ValueError):
            bus.connect("not an endpoint")

    def test_stop_is_prompt_whatever_the_poll_period(self) -> None:
        bus = self.publisher(io_poll_period=60.0)
        bus.start()
        time.sleep(0.02)
        start = time.monotonic()
        bus.stop()
        self.assertLess(time.monotonic() - start, 1.0)
        bus.start()
        time.sleep(0.02)
        start = time.monotonic()
        bus.close()
        self.assertLess(time.monotonic() - start, 1.0)

    def test_stop_from_a_callback_ends_the_loop_and_a_later_stop_joins(self) -> None:
        bus = self.publisher()
        self.subscribe_probe(bus)
        calls: List[int] = []

        def stop_on_first(sample: test_sample_pb2.TestSample) -> None:
            calls.append(sample.sequence)
            bus.stop()

        bus.subscribe("test/stop", test_sample_pb2.TestSample, stop_on_first, "all")
        bus.start()
        self.wait_until_connected(bus, bus)
        bus.publish("test/stop", make_sample(1))
        self.assertTrue(wait_for(lambda: bool(calls)))
        self.assertTrue(wait_for(lambda: not bus.is_running()))
        bus.stop()
        bus.start()
        self.assertTrue(bus.is_running())

    def test_close_from_a_callback_is_refused_and_the_bus_carries_on(self) -> None:
        bus = self.publisher()
        self.subscribe_probe(bus)
        bus.subscribe(
            "test/close", test_sample_pb2.TestSample, lambda sample: bus.close(), "all"
        )
        bus.start()
        self.wait_until_connected(bus, bus)
        bus.publish("test/close", make_sample(1))
        self.assertTrue(
            wait_for(lambda: bus.topic_statistics("test/close").handler_errors == 1)
        )
        self.assertTrue(bus.is_running())


class DeliveryTest(BusTestCase):

    def test_a_process_receives_its_own_topics_and_callbacks_publish_from_the_io_thread(
        self,
    ) -> None:
        bus = self.publisher()
        self.subscribe_probe(bus)
        on_io_thread: List[bool] = []
        pongs = Collector()

        def answer(ping: test_sample_pb2.TestSample) -> None:
            on_io_thread.append(bus.is_io_thread())
            pong = test_sample_pb2.TestSample()
            pong.CopyFrom(ping)
            pong.origin = "pong"
            bus.publish_from_io_thread("test/pong", pong)

        bus.subscribe("test/ping", test_sample_pb2.TestSample, answer, "all")
        bus.subscribe("test/pong", test_sample_pb2.TestSample, pongs, "all")
        bus.start()
        self.wait_until_connected(bus, bus)

        with self.assertRaises(BusError):
            bus.publish_from_io_thread("test/pong", make_sample(0))
        bus.publish("test/ping", make_sample(7))
        self.assertTrue(wait_for(lambda: pongs.size() == 1))
        self.assertEqual(pongs.samples()[0].sequence, 7)
        self.assertEqual(pongs.samples()[0].origin, "pong")
        self.assertEqual(on_io_thread, [True])
        self.assertFalse(bus.is_io_thread())

    def test_topics_match_exactly_not_by_prefix(self) -> None:
        publisher, subscriber = self.connected_pair()
        collector = Collector()
        subscriber.subscribe("test/a", test_sample_pb2.TestSample, collector, "all")
        self.start_and_connect(publisher, subscriber)

        # ZeroMQ passes every topic that starts with "test/a"; only "test/a" itself may be delivered.
        publisher.publish("test/ab", make_sample(100))
        publisher.publish("test/a", make_sample(1))
        publisher.publish("test/a/b", make_sample(101))
        publisher.publish("test/", make_sample(102))
        # One connection keeps the order, so once this one is in, all of the above have arrived.
        publisher.publish("test/a", make_sample(2))
        self.assertTrue(wait_for(lambda: collector.contains(2)))

        self.assertEqual([sample.sequence for sample in collector.samples()], [1, 2])
        self.assertEqual(subscriber.topic_statistics("test/a").received, 2)
        self.assertEqual(subscriber.topic_statistics("test/ab").received, 0)
        self.assertEqual(publisher.topic_statistics("test/ab").sent, 1)

    def test_latest_hands_over_only_the_newest_of_a_burst_and_all_hands_over_every_message(
        self,
    ) -> None:
        burst = 500
        options = dict(
            send_high_water_mark=0,
            receive_high_water_mark=0,
            publish_queue_capacity=4 * burst,
        )
        publisher, subscriber = self.connected_pair(**options)
        latest = Collector()
        everything = Collector()
        subscriber.subscribe(
            "test/latest", test_sample_pb2.TestSample, latest, Delivery.LATEST
        )
        subscriber.subscribe("test/all", test_sample_pb2.TestSample, everything, "all")
        self.start_and_connect(publisher, subscriber)

        # With the subscriber's thread stopped, the burst piles up in its socket, so that the next drain holds many
        # messages of each topic.
        subscriber.stop()
        for sequence in range(burst):
            self.assertTrue(publisher.publish("test/latest", make_sample(sequence)))
            self.assertTrue(publisher.publish("test/all", make_sample(sequence)))
        self.assertTrue(
            wait_for(lambda: publisher.topic_statistics("test/all").sent == burst)
        )
        time.sleep(0.3)
        subscriber.start()

        self.assertTrue(wait_for(lambda: everything.size() == burst))
        self.assertTrue(wait_for(lambda: latest.contains(burst - 1)))

        self.assertEqual(
            [sample.sequence for sample in everything.samples()], list(range(burst))
        )
        all_statistics = subscriber.topic_statistics("test/all")
        self.assertEqual(all_statistics.received, burst)
        self.assertEqual(all_statistics.delivered, burst)
        self.assertEqual(all_statistics.superseded, 0)

        sequences = [sample.sequence for sample in latest.samples()]
        self.assertLess(len(sequences), burst)
        self.assertEqual(sequences, sorted(set(sequences)))
        self.assertEqual(sequences[-1], burst - 1)
        latest_statistics = subscriber.topic_statistics("test/latest")
        self.assertEqual(latest_statistics.received, burst)
        self.assertEqual(latest_statistics.delivered, len(sequences))
        self.assertEqual(
            latest_statistics.received,
            latest_statistics.delivered
            + latest_statistics.superseded
            + latest_statistics.rejected,
        )
        self.assertEqual(latest_statistics.rejected, 0)

    def test_publishing_from_several_threads_keeps_the_order_of_each_thread(
        self,
    ) -> None:
        threads, per_thread = 4, 250
        options = dict(
            send_high_water_mark=0,
            receive_high_water_mark=0,
            publish_queue_capacity=threads * per_thread,
        )
        publisher, subscriber = self.connected_pair(**options)
        collector = Collector()
        subscriber.subscribe(
            "test/threads", test_sample_pb2.TestSample, collector, "all"
        )
        self.start_and_connect(publisher, subscriber)

        def publish_all(index: int) -> None:
            for sequence in range(per_thread):
                publisher.publish("test/threads", make_sample(sequence, index))

        workers = [
            threading.Thread(target=publish_all, args=(index,))
            for index in range(threads)
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()

        self.assertTrue(wait_for(lambda: collector.size() == threads * per_thread))
        next_sequence = [0] * threads
        for sample in collector.samples():
            self.assertEqual(sample.sequence, next_sequence[sample.publisher])
            next_sequence[sample.publisher] = sample.sequence + 1
        statistics = publisher.topic_statistics("test/threads")
        self.assertEqual(statistics.sent, threads * per_thread)
        self.assertEqual(statistics.send_dropped, 0)

    def test_rejects_malformed_messages_and_carries_on(self) -> None:
        # A bare ZeroMQ publisher, to send what a Bus never would.
        context = zmq.Context()
        raw = context.socket(zmq.PUB)
        raw.setsockopt(zmq.LINGER, 0)
        self.addCleanup(context.term)
        self.addCleanup(raw.close)
        raw.bind("tcp://127.0.0.1:*")

        subscriber = self.subscriber(publisher_port=5999)
        subscriber.connect(raw.getsockopt_string(zmq.LAST_ENDPOINT))
        collector = Collector()
        self.subscribe_probe(subscriber)
        subscriber.subscribe("test/typed", test_sample_pb2.TestSample, collector, "all")
        subscriber.start()

        def probed() -> bool:
            raw.send_multipart([PROBE_TOPIC.encode(), b"probe", b""])
            time.sleep(0.005)
            return subscriber.topic_statistics(PROBE_TOPIC).received > 0

        self.assertTrue(wait_for(probed))

        sample_type = test_sample_pb2.TestSample.DESCRIPTOR.full_name.encode()
        event_type = test_event_pb2.TestEvent.DESCRIPTOR.full_name.encode()
        event = test_event_pb2.TestEvent(name="not a sample").SerializeToString()
        raw.send_multipart([b"test/typed", event_type, event])
        # A length-delimited field (text) that claims five bytes and carries two.
        raw.send_multipart([b"test/typed", sample_type, b"\x12\x05ab"])
        raw.send_multipart([b"test/typed", sample_type])
        raw.send_multipart(
            [b"test/typed", sample_type, make_sample(0).SerializeToString(), b"extra"]
        )
        raw.send_multipart(
            [b"test/typed", sample_type, make_sample(5).SerializeToString()]
        )
        self.assertTrue(wait_for(lambda: collector.size() == 1))

        self.assertEqual(collector.samples()[0].sequence, 5)
        statistics = subscriber.topic_statistics("test/typed")
        self.assertEqual(statistics.received, 5)
        self.assertEqual(statistics.rejected, 4)
        self.assertEqual(statistics.delivered, 1)

    def test_a_raising_callback_is_counted_and_the_bus_carries_on(self) -> None:
        publisher, subscriber = self.connected_pair()
        collector = Collector()

        def unwelcome_first(sample: test_sample_pb2.TestSample) -> None:
            if sample.sequence == 0:
                raise RuntimeError("the first sample is unwelcome")
            collector(sample)

        subscriber.subscribe(
            "test/raises", test_sample_pb2.TestSample, unwelcome_first, "all"
        )
        self.start_and_connect(publisher, subscriber)
        publisher.publish("test/raises", make_sample(0))
        publisher.publish("test/raises", make_sample(1))
        self.assertTrue(wait_for(lambda: collector.size() == 1))
        statistics = subscriber.topic_statistics("test/raises")
        self.assertEqual(statistics.handler_errors, 1)
        self.assertEqual(statistics.delivered, 2)
        self.assertTrue(subscriber.is_running())

    def test_periodic_callbacks_run_on_the_io_thread_and_may_publish(self) -> None:
        publisher, subscriber = self.connected_pair()
        collector = Collector()
        subscriber.subscribe(
            "test/periodic", test_sample_pb2.TestSample, collector, "all"
        )
        calls: List[bool] = []

        def tick() -> None:
            calls.append(publisher.is_io_thread())
            publisher.publish_from_io_thread("test/periodic", make_sample(len(calls)))

        def fail() -> None:
            raise RuntimeError("a failing callback")

        publisher.add_periodic_callback(0.002, tick)
        publisher.add_periodic_callback(0.005, fail)
        self.start_and_connect(publisher, subscriber)

        before = len(calls)
        self.assertTrue(wait_for(lambda: len(calls) >= before + 20))
        self.assertTrue(wait_for(lambda: collector.size() >= 10))
        self.assertTrue(all(calls))
        self.assertGreater(publisher.periodic_callback_errors, 0)
        self.assertTrue(publisher.is_running())

    def test_subscribe_all_topics_sees_every_message_with_its_type_name(self) -> None:
        publisher, subscriber = self.connected_pair()
        seen: List[Tuple[str, str, bytes]] = []
        lock = threading.Lock()

        def record(topic: str, type_name: str, payload: bytes) -> None:
            with lock:
                seen.append((topic, type_name, payload))

        subscriber.subscribe_all_topics(record)
        self.start_and_connect(publisher, subscriber)
        publisher.publish("test/one", make_sample(1))
        publisher.publish("test/two", test_event_pb2.TestEvent(name="two"))

        def both_seen() -> bool:
            with lock:
                return {"test/one", "test/two"} <= {entry[0] for entry in seen}

        self.assertTrue(wait_for(both_seen))
        with lock:
            by_topic = {
                topic: (type_name, payload) for topic, type_name, payload in seen
            }
        type_name, payload = by_topic["test/two"]
        self.assertEqual(type_name, "robot_ipc_test.TestEvent")
        self.assertEqual(
            robot_ipc.decode(type_name, payload), test_event_pb2.TestEvent(name="two")
        )


class ConnectionTest(BusTestCase):

    def _subscriber_before_publisher_receives(self, port: int) -> bool:
        subscriber = self.subscriber(port)
        collector = Collector()
        self.subscribe_probe(subscriber)
        subscriber.subscribe("test/late", test_sample_pb2.TestSample, collector)
        subscriber.start()
        time.sleep(0.05)
        try:
            publisher = self.publisher(port=port)
        except BusError:
            # Another process took the port in between; the caller retries with a fresh one.
            return False
        publisher.start()
        self.wait_until_connected(publisher, subscriber)
        publisher.publish("test/late", make_sample(3))
        self.assertTrue(wait_for(lambda: collector.contains(3)))
        return True

    def test_subscriber_started_before_the_publisher_receives_once_it_appears(
        self,
    ) -> None:
        for _ in range(3):
            # A port the kernel just handed out and that is free again.
            probe = Bus(PUBLISHER, loopback_network(PUBLISHER, EPHEMERAL_PORT))
            port = probe.bound_port
            probe.close()
            if self._subscriber_before_publisher_receives(port):
                return
        self.fail("every port was taken before the publisher could bind it")

    def test_a_restarted_publisher_is_reconnected(self) -> None:
        publisher = Bus(PUBLISHER, loopback_network(PUBLISHER, EPHEMERAL_PORT))
        self.addCleanup(publisher.close)
        port = publisher.bound_port
        subscriber = self.subscriber(port)
        collector = Collector()
        self.subscribe_probe(subscriber)
        subscriber.subscribe(
            "test/restart", test_sample_pb2.TestSample, collector, "all"
        )
        self.start_and_connect(publisher, subscriber)
        publisher.publish("test/restart", make_sample(1))
        self.assertTrue(wait_for(lambda: collector.contains(1)))

        # The process dies and comes back on its port.
        publisher.close()
        publisher = self.publisher(port=port)
        publisher.start()
        self.wait_until_connected(publisher, subscriber)
        publisher.publish("test/restart", make_sample(2))
        self.assertTrue(wait_for(lambda: collector.contains(2)))

    def test_the_publish_queue_drops_at_capacity_counts_and_sends_the_rest_on_start(
        self,
    ) -> None:
        capacity, published = 8, 20
        publisher = self.publisher(publish_queue_capacity=capacity)
        # Not started: nothing drains the queue.
        accepted = [
            publisher.publish("test/queue", make_sample(sequence))
            for sequence in range(published)
        ]
        self.assertEqual(accepted.count(False), published - capacity)
        statistics = publisher.topic_statistics("test/queue")
        self.assertEqual(statistics.send_dropped, published - capacity)
        self.assertEqual(statistics.sent, 0)

        publisher.start()
        self.assertTrue(
            wait_for(lambda: publisher.topic_statistics("test/queue").sent == capacity)
        )
        self.assertIn("test/queue", publisher.statistics())


class DecodeTest(unittest.TestCase):

    def test_decodes_by_type_name(self) -> None:
        sample = make_sample(42)
        for type_name in ("robot_ipc_test.TestSample", b"robot_ipc_test.TestSample"):
            with self.subTest(type_name=type_name):
                self.assertEqual(
                    robot_ipc.decode(type_name, sample.SerializeToString()), sample
                )
        self.assertIs(
            robot_ipc.message_class("robot_ipc_test.TestSample"),
            test_sample_pb2.TestSample,
        )

    def test_an_unknown_type_names_itself(self) -> None:
        with self.assertRaises(robot_ipc.UnknownMessageTypeError) as raised:
            robot_ipc.decode("robot_ipc_test.NoSuchMessage", b"")
        self.assertIn("robot_ipc_test.NoSuchMessage", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
