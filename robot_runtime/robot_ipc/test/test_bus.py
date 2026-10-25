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


"""The Python bus over Cyclone DDS: the properties testBus.cpp checks for the C++ bus.

Topics kept apart, Delivery.LATEST against Delivery.ALL, the order of a drain across topics, publishing from several
threads, processes that start in either order or restart, a prompt stop(), the publish queue's drop counting, the
counters' invariants, rejection of malformed samples and of another wire protocol version, callbacks that raise, a
policy-sized sample, the subscription to every topic, and decode(). Every test process has a DDS domain and tag of its
own (robot_ipc.isolated_test_network()), so tests that run at once never see each other's samples.
"""

from collections.abc import Callable
import os
import threading
import time
import unittest

import cyclonedds.domain
import cyclonedds.pub
import cyclonedds.qos
import cyclonedds.sub
import cyclonedds.topic
from robot_ipc_test import test_event_pb2
from robot_ipc_test import test_sample_pb2

import robot_ipc

TIMEOUT = 20.0
PUBLISHER = "publisher"
SUBSCRIBER = "subscriber"
PROBE_TOPIC = "test/probe"
POLICY_BYTES = 768 * 1024


def wait_for(condition: Callable[[], bool], timeout: float = TIMEOUT) -> bool:
    """Polls `condition` until it holds or `timeout` seconds pass."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.001)
    return condition()


def isolated_network(
    topics: tuple[robot_ipc.TopicQos, ...] = (),
) -> robot_ipc.NetworkConfig:
    network = robot_ipc.isolated_test_network([PUBLISHER, SUBSCRIBER])
    return robot_ipc.NetworkConfig(
        nodes=network.nodes,
        domain_id=network.domain_id,
        domain_tag=network.domain_tag,
        topics=topics,
    )


def make_sample(
    sequence: int, publisher: int = 0
) -> test_sample_pb2.TestSample:
    return test_sample_pb2.TestSample(
        sequence=sequence,
        publisher=publisher,
        text="sample",
        values=[float(sequence)],
    )


def subscribe_probe(bus: robot_ipc.Bus) -> None:
    bus.subscribe_raw(PROBE_TOPIC, lambda type_name, payload: None)


def matched_writers(
    bus: robot_ipc.Bus, topic: str, count: int = 1
) -> Callable[[], bool]:
    """A condition: the bus's reader of `topic` matches `count` writers."""
    return lambda: bus.topic_statistics(topic).matched_writers == count


def matched_readers(
    bus: robot_ipc.Bus, topic: str, count: int = 1
) -> Callable[[], bool]:
    """A condition: the bus's writer of `topic` matches `count` readers."""
    return lambda: bus.topic_statistics(topic).matched_readers == count


def written(bus: robot_ipc.Bus, topic: str, count: int) -> Callable[[], bool]:
    """A condition: the bus has written `count` samples of `topic`."""
    return lambda: bus.topic_statistics(topic).written == count


def accounted_for(statistics: robot_ipc.TopicStatistics) -> bool:
    """The counters' invariant once a drain has dispatched."""
    return statistics.received == (
        statistics.delivered + statistics.superseded + statistics.rejected
    )


class Collector:
    """Collects what a callback receives, on the receive thread, for the test thread to read."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._samples: list[test_sample_pb2.TestSample] = []

    def __call__(self, sample: test_sample_pb2.TestSample) -> None:
        with self._lock:
            self._samples.append(sample)

    def samples(self) -> list[test_sample_pb2.TestSample]:
        with self._lock:
            return list(self._samples)

    def size(self) -> int:
        with self._lock:
            return len(self._samples)

    def contains(self, sequence: int) -> bool:
        with self._lock:
            return any(sample.sequence == sequence for sample in self._samples)


class OrderLog:
    """Records what the callbacks of several topics receive, in call order, as "topic:sequence"."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: list[str] = []

    def callback(
        self, topic: str
    ) -> Callable[[test_sample_pb2.TestSample], None]:
        def record(sample: test_sample_pb2.TestSample) -> None:
            with self._lock:
                self._entries.append(f"{topic}:{sample.sequence}")

        return record

    def entries(self) -> list[str]:
        with self._lock:
            return list(self._entries)


class BusTestCase(unittest.TestCase):
    """Creates buses that close when the test ends."""

    def publisher(
        self, network: robot_ipc.NetworkConfig | None = None, **options
    ) -> robot_ipc.Bus:
        bus = robot_ipc.Bus(PUBLISHER, network or isolated_network(), **options)
        self.addCleanup(bus.close)
        return bus

    def subscriber(
        self, network: robot_ipc.NetworkConfig | None = None, **options
    ) -> robot_ipc.Bus:
        bus = robot_ipc.Bus("", network or isolated_network(), **options)
        self.addCleanup(bus.close)
        return bus

    def wait_until_connected(
        self, publisher: robot_ipc.Bus, subscriber: robot_ipc.Bus
    ) -> None:
        """Publishes probes until the subscriber has received more than before: DDS discovery has matched them."""
        before = subscriber.topic_statistics(PROBE_TOPIC).received
        probe = test_event_pb2.TestEvent(name="probe")

        def connected() -> bool:
            publisher.publish(PROBE_TOPIC, probe)
            time.sleep(0.005)
            return subscriber.topic_statistics(PROBE_TOPIC).received > before

        self.assertTrue(wait_for(connected), "the subscriber never matched")

    def connected_pair(self, **options) -> tuple[robot_ipc.Bus, robot_ipc.Bus]:
        publisher = self.publisher(**options)
        subscriber = self.subscriber(**options)
        subscribe_probe(subscriber)
        return publisher, subscriber

    def start_and_connect(
        self, publisher: robot_ipc.Bus, subscriber: robot_ipc.Bus
    ) -> None:
        publisher.start()
        subscriber.start()
        self.wait_until_connected(publisher, subscriber)


class CreateTest(BusTestCase):

    def test_rejects_a_node_outside_the_network_and_lists_the_nodes(
        self,
    ) -> None:
        with self.assertRaises(ValueError) as raised:
            robot_ipc.Bus("ghost", isolated_network())
        self.assertIn("ghost", str(raised.exception))
        self.assertIn(PUBLISHER, str(raised.exception))

    def test_rejects_an_invalid_network_and_options(self) -> None:
        with self.assertRaises(robot_ipc.NetworkConfigError):
            robot_ipc.Bus("", robot_ipc.NetworkConfig(nodes=()))
        for options in (
            {"publish_queue_capacity": 0},
            {"latest_delivery_history_depth": 0},
            {"io_poll_period": float("nan")},
        ):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    robot_ipc.Bus("", isolated_network(), **options)

    def test_a_node_on_another_machine_is_refused(self) -> None:
        # 192.0.2.1 (TEST-NET-1, RFC 5737) is no address of this machine.
        network = isolated_network()
        foreign = robot_ipc.NetworkConfig(
            nodes=(robot_ipc.NodeEndpoint(PUBLISHER, "192.0.2.1"),),
            domain_id=network.domain_id,
            domain_tag=network.domain_tag,
        )
        with self.assertRaises(robot_ipc.InterfaceError) as raised:
            robot_ipc.Bus(PUBLISHER, foreign)
        self.assertIn("192.0.2.1", str(raised.exception))

    def test_the_buses_of_a_process_share_a_domain_and_refuse_another_configuration(
        self,
    ) -> None:
        first = robot_ipc.Bus(PUBLISHER, isolated_network())
        network = isolated_network()
        other_tag = robot_ipc.NetworkConfig(
            nodes=network.nodes,
            domain_id=network.domain_id,
            domain_tag="another_tag",
        )
        with self.assertRaises(robot_ipc.BusError):
            robot_ipc.Bus("", other_tag)
        first.close()
        # Once its last bus is gone, the domain may be configured anew.
        robot_ipc.Bus("", other_tag).close()

    def test_a_subscribe_only_bus_reports_its_domain_and_cannot_publish(
        self,
    ) -> None:
        bus = self.subscriber()
        self.assertEqual(bus.node_name, "")
        self.assertEqual(bus.domain_id, isolated_network().domain_id)
        self.assertEqual(bus.discovery_peers, ["127.0.0.1"])
        self.assertIn(
            f"<Tag>{isolated_network().domain_tag}</Tag>", bus.domain_config
        )
        with self.assertRaises(robot_ipc.BusError):
            bus.publish("test/topic", make_sample(0))

    def test_the_environment_does_not_configure_the_bus(self) -> None:
        # BUILD.bazel sets CYCLONEDDS_URI to an interface that does not exist; the bus passes its own configuration.
        self.assertIn("no_such_interface", os.environ["CYCLONEDDS_URI"])
        bus = self.publisher()
        subscribe_probe(bus)
        bus.start()
        self.wait_until_connected(bus, bus)

    def test_a_node_creates_the_writers_of_its_topics_up_front(self) -> None:
        network = isolated_network(
            (
                robot_ipc.TopicQos(
                    "test/declared",
                    robot_ipc.QosProfile.RELIABLE_EVENTS,
                    (PUBLISHER,),
                ),
            )
        )
        subscriber = self.subscriber(network)
        collector = Collector()
        subscriber.subscribe(
            "test/declared", test_sample_pb2.TestSample, collector, "all"
        )
        subscriber.start()
        publisher = self.publisher(network)
        publisher.start()
        self.assertTrue(
            wait_for(matched_writers(subscriber, "test/declared", 1))
        )
        self.assertTrue(
            wait_for(matched_readers(publisher, "test/declared", 1))
        )
        self.assertEqual(publisher.topic_statistics("test/declared").written, 0)
        publisher.publish("test/declared", make_sample(1))
        self.assertTrue(wait_for(lambda: collector.contains(1)))

    def test_publishing_on_a_closed_bus_is_refused(self) -> None:
        bus = self.publisher()
        bus.start()
        bus.close()
        with self.assertRaises(robot_ipc.BusError) as raised:
            bus.publish("test/topic", make_sample(0))
        self.assertIn("closed", str(raised.exception))
        with self.assertRaises(robot_ipc.BusError):
            bus.publish_raw("test/topic", "robot_ipc_test.TestSample", b"")

    def test_a_publish_racing_close_either_succeeds_before_it_or_is_refused(
        self,
    ) -> None:
        # A thread that publishes on its own, as the operator tools' streaming threads do, while the bus closes.
        def stream(bus: robot_ipc.Bus, outcomes: list[str]) -> None:
            while True:
                try:
                    bus.publish("test/topic", make_sample(0))
                except robot_ipc.BusError:
                    outcomes.append("refused")
                    return
                # pylint: disable-next=broad-exception-caught  # Any other exception is recorded, and fails the test.
                except Exception as error:
                    outcomes.append(repr(error))
                    return

        for _ in range(20):
            bus = self.publisher()
            bus.start()
            outcomes: list[str] = []
            streamer = threading.Thread(target=stream, args=(bus, outcomes))
            streamer.start()
            time.sleep(0.001)
            bus.close()
            streamer.join(timeout=TIMEOUT)
            self.assertFalse(streamer.is_alive())
            self.assertEqual(outcomes, ["refused"])

    def test_close_counts_what_a_bus_that_is_not_running_never_sent(
        self,
    ) -> None:
        never_started = self.publisher()
        self.assertTrue(never_started.publish("test/close", make_sample(1)))
        never_started.close()
        statistics = never_started.topic_statistics("test/close")
        self.assertEqual(statistics.send_dropped, 1)
        self.assertEqual(statistics.written, 0)
        stopped = self.publisher()
        stopped.start()
        stopped.stop()
        self.assertTrue(stopped.publish("test/close", make_sample(2)))
        stopped.close()
        self.assertEqual(stopped.topic_statistics("test/close").send_dropped, 1)
        self.assertEqual(stopped.topic_statistics("test/close").written, 0)

    def test_a_closed_bus_cannot_start_and_its_domain_is_reusable(self) -> None:
        with robot_ipc.Bus(PUBLISHER, isolated_network()) as bus:
            self.assertTrue(bus.is_running())
        self.assertFalse(bus.is_running())
        with self.assertRaises(robot_ipc.BusError):
            bus.start()
        self.publisher()


class LifecycleTest(BusTestCase):

    def test_configuration_is_refused_while_running_and_accepted_after_stop(
        self,
    ) -> None:
        bus = self.publisher()
        collector = Collector()
        for topic in ("", "test/a-b", "1test"):
            with self.subTest(topic=topic):
                with self.assertRaises(ValueError):
                    bus.subscribe(topic, test_sample_pb2.TestSample, collector)
        with self.assertRaises(ValueError):
            bus.publish("test.a", make_sample(0))
        with self.assertRaises(ValueError):
            bus.add_periodic_callback(0.0, lambda: None)
        bus.subscribe("test/a", test_sample_pb2.TestSample, collector, "all")
        with self.assertRaises(ValueError):
            bus.subscribe("test/a", test_sample_pb2.TestSample, collector)

        self.assertFalse(bus.is_running())
        bus.start()
        bus.start()
        self.assertTrue(bus.is_running())
        with self.assertRaises(robot_ipc.BusError):
            bus.subscribe("test/b", test_sample_pb2.TestSample, collector)
        with self.assertRaises(robot_ipc.BusError):
            bus.add_periodic_callback(0.001, lambda: None)

        bus.stop()
        bus.stop()
        self.assertFalse(bus.is_running())
        bus.subscribe("test/b", test_sample_pb2.TestSample, collector)

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
        self.assertLess(time.monotonic() - start, 2.0)

    def test_stop_from_a_callback_ends_the_loop_and_a_later_stop_joins(
        self,
    ) -> None:
        bus = self.publisher()
        subscribe_probe(bus)
        calls = []

        def stop(sample: test_sample_pb2.TestSample) -> None:
            calls.append(sample)
            bus.stop()

        bus.subscribe("test/stop", test_sample_pb2.TestSample, stop, "all")
        bus.start()
        self.wait_until_connected(bus, bus)
        bus.publish("test/stop", make_sample(1))
        self.assertTrue(wait_for(lambda: bool(calls)))
        self.assertTrue(wait_for(lambda: not bus.is_running()))
        bus.stop()
        bus.start()
        self.assertTrue(bus.is_running())

    def test_close_from_a_callback_is_refused_and_the_bus_carries_on(
        self,
    ) -> None:
        bus = self.publisher()
        outcomes: list[str] = []

        def close() -> None:
            if outcomes:
                return
            try:
                bus.close()
                outcomes.append("closed")
            except robot_ipc.BusError:
                outcomes.append("refused")

        bus.add_periodic_callback(0.005, close)
        bus.start()
        self.assertTrue(wait_for(lambda: bool(outcomes)))
        self.assertEqual(outcomes, ["refused"])
        self.assertTrue(bus.is_running())

    def test_a_callback_configuring_during_a_stop_does_not_deadlock_it(
        self,
    ) -> None:
        bus = self.publisher()
        entered = threading.Event()
        go = threading.Event()
        outcome: list[str] = []
        ran: list[bool] = []

        def configure() -> None:
            if ran:
                return
            ran.append(True)
            entered.set()
            go.wait()
            try:
                bus.subscribe_raw("test/late", lambda type_name, payload: None)
                outcome.append("subscribed")
            except robot_ipc.BusError:
                outcome.append("refused")

        bus.add_periodic_callback(0.01, configure)
        bus.start()
        entered.wait()
        stopper = threading.Thread(target=bus.stop)
        stopper.start()
        self.assertTrue(wait_for(lambda: not bus.is_running()))
        go.set()
        stopper.join(timeout=2.0)
        self.assertFalse(
            stopper.is_alive(), "stop() deadlocked with a callback"
        )
        self.assertEqual(outcome, ["refused"])

    def test_a_periodic_callback_that_overran_its_period_is_not_called_again_at_once(
        self,
    ) -> None:
        bus = self.publisher()
        period, overrun = 0.02, 0.07
        starts: list[float] = []
        ends: list[float] = []

        def callback() -> None:
            starts.append(time.monotonic())
            if len(starts) == 1:
                time.sleep(overrun)
            ends.append(time.monotonic())

        bus.add_periodic_callback(period, callback)
        bus.start()
        self.assertTrue(wait_for(lambda: len(starts) >= 3))
        bus.stop()
        self.assertGreaterEqual(starts[1] - ends[0], 0.005)
        self.assertLess(starts[1] - ends[0], period + 0.015)


class DeliveryTest(BusTestCase):

    def test_a_process_receives_its_own_topics_and_callbacks_publish_from_the_io_thread(
        self,
    ) -> None:
        bus = self.publisher()
        pongs = Collector()
        on_io_thread: list[bool] = []

        def ping(sample: test_sample_pb2.TestSample) -> None:
            on_io_thread.append(bus.is_io_thread())
            pong = test_sample_pb2.TestSample()
            pong.CopyFrom(sample)
            pong.origin = "pong"
            bus.publish_from_io_thread("test/pong", pong)

        subscribe_probe(bus)
        bus.subscribe("test/ping", test_sample_pb2.TestSample, ping, "all")
        bus.subscribe("test/pong", test_sample_pb2.TestSample, pongs, "all")
        bus.start()
        self.wait_until_connected(bus, bus)
        with self.assertRaises(robot_ipc.BusError):
            bus.publish_from_io_thread("test/pong", make_sample(0))
        bus.publish("test/ping", make_sample(7))
        self.assertTrue(wait_for(lambda: pongs.size() == 1))
        self.assertEqual(pongs.samples()[0].sequence, 7)
        self.assertEqual(pongs.samples()[0].origin, "pong")
        self.assertTrue(all(on_io_thread))
        self.assertFalse(bus.is_io_thread())

    def test_every_topic_is_a_dds_topic_of_its_own(self) -> None:
        publisher, subscriber = self.connected_pair()
        collector = Collector()
        subscriber.subscribe(
            "test/a", test_sample_pb2.TestSample, collector, "all"
        )
        self.start_and_connect(publisher, subscriber)
        publisher.publish("test/ab", make_sample(100))
        publisher.publish("test/a", make_sample(1))
        publisher.publish("test/a/b", make_sample(101))
        publisher.publish("test/a", make_sample(2))
        self.assertTrue(wait_for(lambda: collector.contains(2)))
        self.assertEqual(
            [sample.sequence for sample in collector.samples()], [1, 2]
        )
        self.assertEqual(subscriber.topic_statistics("test/ab").received, 0)
        self.assertEqual(publisher.topic_statistics("test/ab").written, 1)

    def test_latest_hands_over_only_the_newest_of_a_burst_and_all_hands_over_every_message(
        self,
    ) -> None:
        burst = 300
        publisher, subscriber = self.connected_pair(
            publish_queue_capacity=4 * burst,
            all_delivery_history_depth=2 * burst,
        )
        latest = Collector()
        everything = Collector()
        subscriber.subscribe(
            "test/latest", test_sample_pb2.TestSample, latest, "latest"
        )
        subscriber.subscribe(
            "test/all", test_sample_pb2.TestSample, everything, "all"
        )
        self.start_and_connect(publisher, subscriber)
        publisher.publish("test/latest", make_sample(0))
        self.assertTrue(wait_for(lambda: latest.contains(0)))

        subscriber.stop()
        for sequence in range(1, burst + 1):
            publisher.publish("test/latest", make_sample(sequence))
            publisher.publish("test/all", make_sample(sequence))
        self.assertTrue(wait_for(written(publisher, "test/all", burst)))
        time.sleep(0.3)
        subscriber.start()
        self.assertTrue(wait_for(lambda: everything.size() == burst))
        self.assertTrue(wait_for(lambda: latest.contains(burst)))

        self.assertEqual(
            [sample.sequence for sample in everything.samples()],
            list(range(1, burst + 1)),
        )
        all_statistics = subscriber.topic_statistics("test/all")
        self.assertEqual(all_statistics.received, burst)
        self.assertEqual(all_statistics.missed, 0)
        sequences = [sample.sequence for sample in latest.samples()]
        self.assertLess(len(sequences), burst)
        self.assertEqual(sequences, sorted(sequences))
        self.assertEqual(sequences[-1], burst)
        latest_statistics = subscriber.topic_statistics("test/latest")
        self.assertTrue(accounted_for(latest_statistics), latest_statistics)
        self.assertEqual(
            latest_statistics.received + latest_statistics.missed,
            publisher.topic_statistics("test/latest").written,
        )
        self.assertGreater(latest_statistics.missed, 0)

    def test_wrong_typed_message_does_not_displace_valid_message_in_latest_delivery(
        self,
    ) -> None:
        publisher, subscriber = self.connected_pair()
        latest = Collector()
        subscriber.subscribe("test/latest", test_sample_pb2.TestSample, latest)
        self.start_and_connect(publisher, subscriber)
        subscriber.stop()
        publisher.publish("test/latest", make_sample(42))
        publisher.publish("test/latest", test_event_pb2.TestEvent(name="event"))
        self.assertTrue(wait_for(written(publisher, "test/latest", 2)))
        time.sleep(0.05)
        subscriber.start()
        self.assertTrue(wait_for(lambda: latest.size() == 1))
        self.assertEqual(latest.samples()[0].sequence, 42)
        statistics = subscriber.topic_statistics("test/latest")
        self.assertEqual(
            (
                statistics.received,
                statistics.delivered,
                statistics.rejected,
                statistics.superseded,
            ),
            (2, 1, 1, 0),
        )

    def test_publishing_from_several_threads_keeps_the_order_of_each_thread(
        self,
    ) -> None:
        threads, per_thread = 4, 250
        publisher, subscriber = self.connected_pair(
            publish_queue_capacity=threads * per_thread,
            all_delivery_history_depth=2 * threads * per_thread,
        )
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
        self.assertTrue(
            wait_for(lambda: collector.size() == threads * per_thread)
        )
        next_sequence = [0] * threads
        for sample in collector.samples():
            self.assertEqual(sample.sequence, next_sequence[sample.publisher])
            next_sequence[sample.publisher] = sample.sequence + 1
        statistics = publisher.topic_statistics("test/threads")
        self.assertEqual(statistics.written, threads * per_thread)
        self.assertEqual(statistics.send_dropped, 0)

    def test_rejects_malformed_samples_and_another_protocol_version_and_carries_on(
        self,
    ) -> None:
        subscriber = self.subscriber()
        collector = Collector()
        subscriber.subscribe(
            "test/typed", test_sample_pb2.TestSample, collector, "all"
        )
        subscriber.start()
        # A raw DDS writer in the bus's domain, to write what a Bus never would.
        participant = cyclonedds.domain.DomainParticipant(
            isolated_network().domain_id
        )
        writer = cyclonedds.pub.DataWriter(
            participant,
            cyclonedds.topic.Topic(
                participant, "test/typed", robot_ipc.Envelope
            ),
            qos=cyclonedds.qos.Qos(
                cyclonedds.qos.Policy.Reliability.Reliable(
                    max_blocking_time=10**9
                ),
                cyclonedds.qos.Policy.History.KeepLast(64),
            ),
        )
        self.assertTrue(wait_for(matched_writers(subscriber, "test/typed", 1)))
        sample_type = "robot_ipc_test.TestSample"
        version = robot_ipc.WIRE_PROTOCOL_VERSION
        for sequence, (protocol, type_name, payload) in enumerate(
            (
                (
                    version,
                    "robot_ipc_test.TestEvent",
                    test_event_pb2.TestEvent(name="x").SerializeToString(),
                ),
                # A length-delimited field (text) that claims five bytes and carries two.
                (version, sample_type, b"\x12\x05ab"),
                (version - 1, sample_type, make_sample(1).SerializeToString()),
                (version + 1, sample_type, make_sample(2).SerializeToString()),
                (version, sample_type, make_sample(5).SerializeToString()),
            ),
            start=1,
        ):
            writer.write(
                robot_ipc.Envelope(protocol, sequence, type_name, payload)
            )
        self.assertTrue(wait_for(lambda: collector.size() == 1))
        self.assertEqual(collector.samples()[0].sequence, 5)
        statistics = subscriber.topic_statistics("test/typed")
        self.assertEqual(
            (
                statistics.received,
                statistics.rejected,
                statistics.delivered,
                statistics.missed,
            ),
            (5, 4, 1, 0),
        )
        self.assertTrue(subscriber.is_running())
        del writer, participant

    def test_the_bus_writes_the_envelope_of_its_wire_protocol(self) -> None:
        publisher = self.publisher()
        participant = cyclonedds.domain.DomainParticipant(
            isolated_network().domain_id
        )
        reader = cyclonedds.sub.DataReader(
            participant,
            cyclonedds.topic.Topic(
                participant, "test/version", robot_ipc.Envelope
            ),
            qos=cyclonedds.qos.Qos(
                cyclonedds.qos.Policy.Reliability.Reliable(
                    max_blocking_time=10**9
                )
            ),
        )
        publisher.start()
        received: list[robot_ipc.Envelope] = []

        def two_received() -> bool:
            publisher.publish("test/version", make_sample(3))
            time.sleep(0.005)
            received.extend(
                sample
                for sample in reader.take(16)
                if isinstance(sample, robot_ipc.Envelope)
            )
            return len(received) >= 2

        self.assertTrue(wait_for(two_received))
        for envelope in received:
            self.assertEqual(
                envelope.protocol_version, robot_ipc.WIRE_PROTOCOL_VERSION
            )
            self.assertEqual(envelope.type_name, "robot_ipc_test.TestSample")
            self.assertEqual(
                test_sample_pb2.TestSample.FromString(
                    bytes(envelope.payload)
                ).sequence,
                3,
            )
        sequences = [envelope.sequence_number for envelope in received]
        self.assertEqual(
            sequences, list(range(sequences[0], sequences[0] + len(sequences)))
        )
        del reader, participant

    def test_one_drain_dispatches_in_write_order_across_topics(self) -> None:
        publisher, subscriber = self.connected_pair()
        log = OrderLog()
        subscriber.subscribe(
            "test/a", test_sample_pb2.TestSample, log.callback("a"), "all"
        )
        subscriber.subscribe(
            "test/b", test_sample_pb2.TestSample, log.callback("b"), "latest"
        )
        subscriber.subscribe(
            "test/c", test_sample_pb2.TestSample, log.callback("c"), "all"
        )
        self.start_and_connect(publisher, subscriber)
        subscriber.stop()
        for topic, sequence in (
            ("a", 0),
            ("b", 0),
            ("c", 0),
            ("b", 1),
            ("a", 1),
            ("c", 1),
        ):
            publisher.publish(f"test/{topic}", make_sample(sequence))
        self.assertTrue(wait_for(written(publisher, "test/c", 2)))
        time.sleep(0.3)
        subscriber.start()
        self.assertTrue(wait_for(lambda: len(log.entries()) == 5))
        self.assertEqual(log.entries(), ["a:0", "c:0", "b:1", "a:1", "c:1"])
        self.assertEqual(subscriber.topic_statistics("test/b").superseded, 1)
        self.assertTrue(accounted_for(subscriber.topic_statistics("test/b")))

    def test_a_policy_sized_sample_arrives_whole(self) -> None:
        network = isolated_network(
            (
                robot_ipc.TopicQos(
                    "test/policy",
                    robot_ipc.QosProfile.RELIABLE_LATEST,
                    (PUBLISHER,),
                ),
            )
        )
        publisher = self.publisher(network)
        subscriber = self.subscriber(network)
        collector = Collector()
        subscriber.subscribe(
            "test/policy", test_sample_pb2.TestSample, collector
        )
        publisher.start()
        subscriber.start()
        self.assertTrue(wait_for(matched_writers(subscriber, "test/policy", 1)))
        text = "".join(
            chr(ord("a") + index % 26) for index in range(POLICY_BYTES)
        )
        publisher.publish(
            "test/policy", test_sample_pb2.TestSample(sequence=9, text=text)
        )
        self.assertTrue(wait_for(lambda: collector.size() == 1))
        self.assertEqual(collector.samples()[0].text, text)

    def test_a_raising_callback_is_counted_and_the_bus_carries_on(self) -> None:
        publisher, subscriber = self.connected_pair()
        collector = Collector()

        def callback(sample: test_sample_pb2.TestSample) -> None:
            if sample.sequence == 0:
                raise RuntimeError("the first sample is unwelcome")
            collector(sample)

        subscriber.subscribe(
            "test/raises", test_sample_pb2.TestSample, callback, "all"
        )
        self.start_and_connect(publisher, subscriber)
        publisher.publish("test/raises", make_sample(0))
        publisher.publish("test/raises", make_sample(1))
        self.assertTrue(wait_for(lambda: collector.size() == 1))
        statistics = subscriber.topic_statistics("test/raises")
        self.assertEqual(statistics.handler_errors, 1)
        self.assertEqual(statistics.delivered, 2)
        self.assertTrue(subscriber.is_running())

    def test_periodic_callbacks_run_on_the_io_thread_and_may_publish(
        self,
    ) -> None:
        publisher, subscriber = self.connected_pair()
        collector = Collector()
        subscriber.subscribe(
            "test/periodic", test_sample_pb2.TestSample, collector, "all"
        )
        calls: list[bool] = []

        def tick() -> None:
            calls.append(publisher.is_io_thread())
            publisher.publish_from_io_thread(
                "test/periodic", make_sample(len(calls))
            )

        def fail() -> None:
            raise RuntimeError("a failing callback")

        publisher.add_periodic_callback(0.002, tick)
        publisher.add_periodic_callback(0.005, fail)
        self.start_and_connect(publisher, subscriber)
        self.assertTrue(wait_for(lambda: collector.size() >= 10))
        self.assertTrue(all(calls))
        self.assertGreater(publisher.periodic_callback_errors, 0)

    def test_subscribe_all_topics_sees_every_message_with_its_type_name(
        self,
    ) -> None:
        publisher, subscriber = self.connected_pair()
        seen: list[tuple[str, str, bytes]] = []
        lock = threading.Lock()

        def record(topic: str, type_name: str, payload: bytes) -> None:
            with lock:
                seen.append((topic, type_name, payload))

        subscriber.subscribe_all_topics(record)
        self.start_and_connect(publisher, subscriber)

        # The subscription discovers each topic from its writer, so it sees the samples written after that.
        def both_seen() -> bool:
            publisher.publish("test/one", make_sample(1))
            publisher.publish("test/two", test_event_pb2.TestEvent(name="two"))
            time.sleep(0.01)
            with lock:
                return {"test/one", "test/two"} <= {entry[0] for entry in seen}

        self.assertTrue(wait_for(both_seen))
        with lock:
            by_topic = {
                topic: (type_name, payload)
                for topic, type_name, payload in seen
            }
        type_name, payload = by_topic["test/two"]
        self.assertEqual(type_name, "robot_ipc_test.TestEvent")
        self.assertEqual(
            robot_ipc.decode(type_name, payload),
            test_event_pb2.TestEvent(name="two"),
        )


class ConnectionTest(BusTestCase):

    def test_subscriber_started_before_the_publisher_receives_once_it_appears(
        self,
    ) -> None:
        subscriber = self.subscriber()
        collector = Collector()
        subscribe_probe(subscriber)
        subscriber.subscribe("test/late", test_sample_pb2.TestSample, collector)
        subscriber.start()
        time.sleep(0.05)
        self.assertEqual(
            subscriber.topic_statistics("test/late").matched_writers, 0
        )
        publisher = self.publisher()
        publisher.start()
        self.wait_until_connected(publisher, subscriber)
        publisher.publish("test/late", make_sample(3))
        self.assertTrue(wait_for(lambda: collector.contains(3)))

    def test_a_restarted_publisher_is_matched_again_and_its_gone_writer_unmatched(
        self,
    ) -> None:
        publisher, subscriber = self.connected_pair()
        collector = Collector()
        subscriber.subscribe(
            "test/restart", test_sample_pb2.TestSample, collector, "all"
        )
        self.start_and_connect(publisher, subscriber)
        publisher.publish("test/restart", make_sample(1))
        self.assertTrue(wait_for(lambda: collector.contains(1)))
        self.assertTrue(
            wait_for(matched_writers(subscriber, "test/restart", 1))
        )
        publisher.close()
        self.assertTrue(
            wait_for(matched_writers(subscriber, "test/restart", 0))
        )
        restarted = self.publisher()
        restarted.start()
        self.wait_until_connected(restarted, subscriber)
        restarted.publish("test/restart", make_sample(2))
        self.assertTrue(wait_for(lambda: collector.contains(2)))
        self.assertEqual(subscriber.topic_statistics("test/restart").missed, 0)

    def test_the_publish_queue_drops_at_capacity_counts_and_sends_the_rest_on_start(
        self,
    ) -> None:
        capacity, published = 8, 20
        publisher = self.publisher(publish_queue_capacity=capacity)
        accepted = [
            publisher.publish("test/queue", make_sample(n))
            for n in range(published)
        ]
        self.assertEqual(accepted.count(False), published - capacity)
        self.assertEqual(
            publisher.topic_statistics("test/queue").send_dropped,
            published - capacity,
        )
        publisher.start()
        self.assertTrue(wait_for(written(publisher, "test/queue", capacity)))


class EnvelopeTest(unittest.TestCase):

    def test_every_encoding_round_trips_and_garbage_is_refused(self) -> None:
        envelope = robot_ipc.Envelope(2, 7, "a.B", b"\x00\x01payload")
        for version_2 in (True, False):
            with self.subTest(version_2=version_2):
                data = envelope.serialize(use_version_2=version_2)
                self.assertEqual(robot_ipc.Envelope.deserialize(data), envelope)
        for garbage in (
            b"",
            b"\x00\x07",
            b"\x00\x09\x00\x00" + bytes(16),
            b"\x00\x07\x00\x00\x02",
        ):
            with self.subTest(garbage=garbage):
                with self.assertRaises(robot_ipc.envelope.EnvelopeError):
                    robot_ipc.Envelope.deserialize(garbage)


class DecodeTest(unittest.TestCase):

    def test_decodes_by_type_name(self) -> None:
        sample = make_sample(42)
        for type_name in (
            "robot_ipc_test.TestSample",
            b"robot_ipc_test.TestSample",
        ):
            with self.subTest(type_name=type_name):
                self.assertEqual(
                    robot_ipc.decode(type_name, sample.SerializeToString()),
                    sample,
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
