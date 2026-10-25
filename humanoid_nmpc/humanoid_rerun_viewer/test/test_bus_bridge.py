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


"""The BusBridge's writer thread: a handler that blocks (a Rerun sink that takes no data) never blocks the bus.

A stand-in bridge, whose handlers record what they are given and can be made to block, is fed over a DDS bus of the test process.
"""

from collections.abc import Callable
import threading
import time
import unittest

from humanoid_mpc_msgs import fsm_state_pb2
from humanoid_mpc_msgs import mpc_status_pb2

from humanoid_mpc_ipc import topics
from humanoid_rerun_viewer import bus_bridge
import robot_ipc

TIMEOUT_S = 10.0
PUBLISHER = "viz"


def wait_for(condition: Callable[[], bool], timeout: float = TIMEOUT_S) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


class StandInBridge:
    """The RerunBridge methods the BusBridge calls, recording what they are given; handle_mpc_status waits for `release`."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.release.set()
        self.lock = threading.Lock()
        self.statuses: list[float] = []
        self.threads: set[str] = set()
        self.reconnects = 0
        self.events: list[str] = []
        self.flushes = 0
        self.delivered = True

    def record(self, event: str) -> None:
        with self.lock:
            self.events.append(event)
            self.threads.add(threading.current_thread().name)

    def handle_mpc_status(self, status: mpc_status_pb2.MpcStatus) -> None:
        self.release.wait()
        with self.lock:
            self.statuses.append(status.observation_time)
        self.record("status")

    def handle_scene(self, scene) -> None:
        del scene  # Unused.
        self.record("scene")

    def handle_telemetry(self, series) -> None:
        del series  # Unused.
        self.record("telemetry")

    def handle_fsm_state(self, state) -> None:
        del state  # Unused.
        self.record("fsm")

    def handle_loop_timing(self, timing) -> None:
        del timing  # Unused.
        self.record("loop_timing")

    def flush(self) -> int:
        with self.lock:
            self.flushes += 1
        return 0

    def draw_pending_scene(self) -> None:
        self.record("pending_scene")

    def report(
        self, bus_rejected: int = 0, sink_dropped: int = 0
    ) -> str | None:
        del bus_rejected, sink_dropped  # Unused.
        return None

    def flush_recording(self, timeout: float) -> bool:
        del timeout  # Unused.
        return self.delivered


def isolated_network() -> robot_ipc.NetworkConfig:
    """This test process's own DDS domain (robot_ipc.isolated_test_network()), which every bus of it shares."""
    return robot_ipc.isolated_test_network([PUBLISHER])


class WriterThreadTest(unittest.TestCase):
    def setUp(self) -> None:
        self.publisher = robot_ipc.Bus(PUBLISHER, isolated_network())
        self.addCleanup(self.publisher.close)
        self.publisher.start()
        self.bridge = StandInBridge()
        # Whatever a test leaves blocked is released at the end, so that its writer thread ends.
        self.addCleanup(self.bridge.release.set)
        self.time = 0.0

    def feed(self, **options) -> bus_bridge.BusBridge:
        feed = bus_bridge.BusBridge(
            self.bridge,  # type: ignore[arg-type]  # A stand-in with the methods the BusBridge calls.
            isolated_network(),
            **options,
        )
        self.addCleanup(feed.stop, recording_flush_timeout=0.1)
        return feed

    def publish_status(self) -> None:
        self.time += 0.01
        self.publisher.publish(
            topics.MPC_STATUS,
            mpc_status_pb2.MpcStatus(observation_time=self.time),
        )

    def publish_until(self, condition: Callable[[], bool]) -> bool:
        """Publishes an MpcStatus every 10 ms until `condition` holds, which absorbs DDS discovery."""

        def published() -> bool:
            if condition():
                return True
            self.publish_status()
            return False

        return wait_for(published)

    def test_the_handlers_run_on_the_writer_thread_in_order(self) -> None:
        feed = self.feed()
        feed.start()
        self.assertTrue(
            self.publish_until(lambda: len(self.bridge.statuses) >= 20)
        )
        self.assertTrue(feed.stop())
        self.assertEqual(self.bridge.threads, {"rerun_bridge_writer"})
        self.assertEqual(self.bridge.statuses, sorted(self.bridge.statuses))
        self.assertGreater(self.bridge.flushes, 0)

    def test_a_blocked_handler_does_not_block_the_bus_or_the_stop(self) -> None:
        # A Rerun sink that takes no data blocks the handler's log() call; the bus keeps receiving and drops what the
        # queue cannot hold, and stop() returns within its bound, reporting that nothing reached the sink.
        self.bridge.release.clear()
        feed = self.feed(max_queued_messages=4)
        feed.start()
        self.assertTrue(
            self.publish_until(lambda: feed.dropped() >= 10),
            f"dropped {feed.dropped()}",
        )
        self.assertEqual(
            feed.dropped_by_topic(), {topics.MPC_STATUS: feed.dropped()}
        )
        self.assertTrue(wait_for(lambda: feed.writer_stalled_for() > 0.2))
        stopped: list[bool] = []

        def stop() -> None:
            stopped.append(feed.stop(recording_flush_timeout=0.1))

        stopper = threading.Thread(target=stop)
        started = time.monotonic()
        stopper.start()
        stopper.join(timeout=bus_bridge.SHUTDOWN_DRAIN_TIMEOUT_S + TIMEOUT_S)
        self.assertFalse(
            stopper.is_alive(), "stop() waited for the blocked handler"
        )
        self.assertLess(
            time.monotonic() - started,
            bus_bridge.SHUTDOWN_DRAIN_TIMEOUT_S + 2.0,
        )
        self.assertEqual(stopped, [False])

    def test_a_latest_topic_keeps_only_its_newest_message_waiting(self) -> None:
        self.bridge.release.clear()
        feed = self.feed(max_queued_messages=8)
        feed.start()
        self.assertTrue(
            self.publish_until(lambda: feed.writer_stalled_for() > 0.05)
        )
        for mode in ("A", "B", "C", "D", "E", "F"):
            self.publisher.publish(
                topics.ROBOT_FSM_STATE, fsm_state_pb2.FsmState(mode=mode)
            )
            time.sleep(0.05)
        # Six states, which a queue of eight would hold: one entry waits, for the newest.
        self.assertEqual(feed.dropped(), 0)
        self.bridge.release.set()
        self.assertTrue(wait_for(lambda: "fsm" in self.bridge.events))
        time.sleep(0.2)
        self.assertEqual(self.bridge.events.count("fsm"), 1)

    def test_a_pause_drops_and_a_resume_reconnects_before_any_message(
        self,
    ) -> None:
        feed = self.feed()
        feed.pause("the viewer is unreachable")
        feed.start()
        self.assertTrue(self.publish_until(lambda: feed.dropped() >= 5))
        self.assertEqual(self.bridge.statuses, [])

        def reconnect() -> None:
            self.bridge.record("reconnect")

        feed.resume(reconnect)
        self.assertTrue(
            self.publish_until(lambda: len(self.bridge.statuses) >= 3)
        )
        self.assertFalse(feed.paused)
        self.assertEqual(self.bridge.events[0], "reconnect")
        # An unreachable viewer at stop is not waited for.
        feed.pause("the viewer is unreachable")
        self.assertFalse(feed.stop(recording_flush_timeout=0.1))

    def test_a_resume_that_a_later_pause_overtook_keeps_it(self) -> None:
        feed = self.feed()
        feed.start()
        feed.pause("first")
        feed.resume(lambda: None)
        feed.pause("second")
        time.sleep(0.3)
        self.assertTrue(feed.paused)

    def test_stop_without_start_still_flushes(self) -> None:
        feed = self.feed()
        self.assertTrue(feed.stop(recording_flush_timeout=0.1))
        self.assertEqual(self.bridge.flushes, 1)
        self.assertIn("pending_scene", self.bridge.events)

    def test_the_queue_must_hold_a_message(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "max_queued_messages must be positive"
        ):
            bus_bridge.BusBridge(
                self.bridge,  # type: ignore[arg-type]  # A stand-in with the methods the BusBridge calls.
                isolated_network(),
                max_queued_messages=0,
            )


if __name__ == "__main__":
    unittest.main()
