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

"""Feeds a RerunBridge from the IPC bus: the subscriptions, the writer thread, the periodic flush and report, start and stop.

The bridge only subscribes (it publishes as no node), so it can run on any machine of the network file; DDS discovery
matches it with every publisher, so the publishers may start, stop and restart in any order.

| Topic | Message | Delivery | Handler |
|---|---|---|---|
| viz/scene | VisualizationScene | latest | RerunBridge.handle_scene |
| viz/telemetry | TelemetrySeries | all (every sample is plotted) | RerunBridge.handle_telemetry |
| robot/fsm_state | FsmState | latest | RerunBridge.handle_fsm_state |
| mpc/status | MpcStatus | all (every solve time is plotted) | RerunBridge.handle_mpc_status |
| robot/loop_timing | LoopTiming | all | RerunBridge.handle_loop_timing |

The bus's receive thread never calls Rerun. Its callbacks queue each message, at most MAX_QUEUED_MESSAGES, and a writer
thread of the BusBridge runs the bridge's handlers, the periodic flush and the report. A topic of "latest" delivery
keeps one message waiting, the newest, as the bus does. A sink that takes no data blocks only the writer: rerun-sdk
0.38 buffers about 100 MiB for a viewer that is not there or stopped reading, and then blocks every log() call. The
bus then keeps receiving, a message that finds the queue full is dropped and counted, and stop() waits for the writer
a bounded time. While the viewer is unreachable (pause()), messages are dropped before they reach the bridge, so that
rerun-sdk buffers nothing for it; resume() reconnects on the writer thread.
"""

from collections.abc import Callable
import functools
import logging
import math
import queue
import threading
import time
from typing import Any

from humanoid_mpc_msgs import fsm_state_pb2
from humanoid_mpc_msgs import loop_timing_pb2
from humanoid_mpc_msgs import mpc_status_pb2
from humanoid_mpc_msgs import telemetry_series_pb2
from humanoid_mpc_msgs import visualization_scene_pb2

from humanoid_mpc_ipc import topics
from humanoid_rerun_viewer import bridge as bridge_module
import robot_ipc

_LOGGER = logging.getLogger("humanoid_rerun_viewer")

# LINT.IfChange(bus_defaults)
DEFAULT_FLUSH_PERIOD_S = 0.05
DEFAULT_REPORT_PERIOD_S = 5.0
# The last flush may wait this long for the sink [s]; a viewer that went away must not hang the exit.
SHUTDOWN_FLUSH_TIMEOUT_S = 5.0
# How many messages may wait for the writer thread, about 2 s of the bus's messages; one more is dropped and counted.
MAX_QUEUED_MESSAGES = 512
# At stop, how long the writer may take for the messages still queued [s], before the last flush.
SHUTDOWN_DRAIN_TIMEOUT_S = 2.0
# LINT.ThenChange(//humanoid_nmpc/humanoid_rerun_viewer/README.md:bus_defaults)

# stop() waits for the writer this much longer than its drain and its last flush may take [s].
_JOIN_MARGIN_S = 1.0

_Task = Callable[[], Any]


def _next_deadline(deadline: float, period: float, now: float) -> float:
    """The deadline after `deadline`; a whole period late, one period from `now` instead of a burst of missed ones."""
    deadline += period
    return deadline if deadline > now else now + period


class BusBridge:
    """Subscribes `bridge` to the topics of every node of `network`, and runs it on a writer thread of its own.

    Thread-safe: the bus's receive thread queues the messages, the writer thread runs the bridge, and another thread
    (cli.run's) may call pause(), resume(), writer_stalled_for(), dropped() and stop().

    Args:
        bridge: the handlers, which only the writer thread calls once start() ran (stop() without start() calls them on
            its caller's thread).
        network: the nodes to connect to.
        flush_period: how often the buffered plots are sent [s].
        report_period: how often problems since the last report are logged [s].
        telemetry: whether to subscribe to viz/telemetry.
        max_queued_messages: how many messages may wait for the writer thread.

    Attributes:
        bus: The bus the bridge is subscribed on.
        subscribed_topics: The topics subscribed to, in subscription order.
    """

    def __init__(
        self,
        bridge: bridge_module.RerunBridge,
        network: robot_ipc.NetworkConfig,
        flush_period: float = DEFAULT_FLUSH_PERIOD_S,
        report_period: float = DEFAULT_REPORT_PERIOD_S,
        telemetry: bool = True,
        max_queued_messages: int = MAX_QUEUED_MESSAGES,
    ) -> None:
        # The not-form rejects NaN, which `flush_period <= 0.0` would let through.
        if not (flush_period > 0.0 and report_period > 0.0):
            raise ValueError(
                f"the flush and report periods must be positive, got {flush_period} and {report_period}"
            )
        if max_queued_messages <= 0:
            raise ValueError(
                f"max_queued_messages must be positive, got {max_queued_messages}"
            )
        self._bridge = bridge
        self._flush_period = flush_period
        self._report_period = report_period
        self._queue: queue.Queue[tuple[str, _Task]] = queue.Queue(
            maxsize=max_queued_messages
        )
        self._lock = threading.Lock()
        # Under _lock. Messages dropped, by topic. Why messages are dropped before they reach the bridge, None while
        # they are not; pause() counts its calls, so that a resume() a later pause() overtook does not end that one.
        # The task to run on the writer thread before it takes messages again. When the writer began its current task
        # (time.monotonic()), None while it waits for one. Whether the recording reached its sink at stop().
        self._dropped: dict[str, int] = {}
        # Under _lock: per topic of "latest" delivery, its newest message not yet handled, None when there is none.
        # The queue holds one entry for a topic whose message waits here, which hands the writer whatever is newest.
        self._latest: dict[str, _Task | None] = {}
        self._paused_reason: str | None = None
        self._pauses = 0
        self._resume_task: tuple[int, _Task] | None = None
        self._busy_since: float | None = None
        self._delivered = True
        self._last_drop_log = -math.inf
        self._recording_flush_timeout = SHUTDOWN_FLUSH_TIMEOUT_S
        self._stop_writer = threading.Event()
        self._writer = threading.Thread(
            target=self._write, name="rerun_bridge_writer", daemon=True
        )
        self._started = False
        self._stopped = False
        self.bus = robot_ipc.Bus("", network)
        subscriptions: list[
            tuple[str, Any, Callable[[Any], None], robot_ipc.Delivery]
        ] = [
            (
                topics.VIZ_SCENE,
                visualization_scene_pb2.VisualizationScene,
                bridge.handle_scene,
                robot_ipc.Delivery.LATEST,
            ),
            (
                topics.ROBOT_FSM_STATE,
                fsm_state_pb2.FsmState,
                bridge.handle_fsm_state,
                robot_ipc.Delivery.LATEST,
            ),
            (
                topics.MPC_STATUS,
                mpc_status_pb2.MpcStatus,
                bridge.handle_mpc_status,
                robot_ipc.Delivery.ALL,
            ),
            (
                topics.ROBOT_LOOP_TIMING,
                loop_timing_pb2.LoopTiming,
                bridge.handle_loop_timing,
                robot_ipc.Delivery.ALL,
            ),
        ]
        if telemetry:
            subscriptions.insert(
                1,
                (
                    topics.VIZ_TELEMETRY,
                    telemetry_series_pb2.TelemetrySeries,
                    bridge.handle_telemetry,
                    robot_ipc.Delivery.ALL,
                ),
            )
        self.subscribed_topics = tuple(
            subscription[0] for subscription in subscriptions
        )
        try:
            for topic, message_class, handler, delivery in subscriptions:
                if delivery is robot_ipc.Delivery.LATEST:
                    self._latest[topic] = None
                    callback = functools.partial(self._enqueue_latest, topic, handler)
                else:
                    callback = functools.partial(self._enqueue, topic, handler)
                self.bus.subscribe(topic, message_class, callback, delivery)
        except BaseException:
            self.bus.close()
            raise

    def bus_rejected(self) -> int:
        """Messages of the subscribed topics the bus rejected: wrong type, not three frames, or not parsing."""
        statistics = self.bus.statistics()
        return sum(
            statistics[topic].rejected
            for topic in self.subscribed_topics
            if topic in statistics
        )

    def dropped(self) -> int:
        """Messages dropped so far because the sink took no data or its viewer was unreachable."""
        with self._lock:
            return sum(self._dropped.values())

    def dropped_by_topic(self) -> dict[str, int]:
        """dropped(), by topic."""
        with self._lock:
            return dict(self._dropped)

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._paused_reason is not None

    def writer_stalled_for(self) -> float:
        """How long the writer thread has been in its current task [s], 0 while it waits for one."""
        with self._lock:
            busy_since = self._busy_since
        return 0.0 if busy_since is None else time.monotonic() - busy_since

    def pause(self, reason: str) -> None:
        """Drops every message, counted under `reason`, until resume(): a viewer that is unreachable gets nothing.

        rerun-sdk buffers what a viewer that is not there does not take, up to about 100 MiB, and then blocks; while
        paused the bridge sends it nothing, so a viewer that comes back is sent the robot afresh (resume()).

        Args:
            reason: The reason string describing why streaming is paused.
        """
        with self._lock:
            self._paused_reason = reason
            self._pauses += 1
            self._resume_task = None
            for topic in self._latest:
                self._latest[topic] = None

    def resume(self, reconnect: _Task) -> None:
        """Runs `reconnect` on the writer thread, then takes messages again; called after start().

        Args:
            reconnect: attaches the recording to the viewer again and sends it the static data
                (RerunBridge.resend_static()); it runs before any message that arrives after it.
        """
        with self._lock:
            self._resume_task = (self._pauses, reconnect)

    def _enqueue(
        self, topic: str, handler: Callable[[Any], None], message: Any
    ) -> None:
        """The bus's callback, on its receive thread: queues the message for the writer, or drops and counts it."""
        with self._lock:
            paused_reason = self._paused_reason
        if paused_reason is not None:
            self._count_drop(topic, paused_reason)
            return
        try:
            self._queue.put_nowait((topic, functools.partial(handler, message)))
        except queue.Full:
            self._count_drop(
                topic,
                "the Rerun sink takes no data (a viewer that stopped reading?), so the bridge's queue is full",
            )

    def _enqueue_latest(
        self, topic: str, handler: Callable[[Any], None], message: Any
    ) -> None:
        """The bus's callback of a "latest" topic: the message replaces the one waiting, or is queued if none waits."""
        with self._lock:
            paused_reason = self._paused_reason
            waiting = self._latest[topic] is not None
            self._latest[topic] = (
                None
                if paused_reason is not None
                else functools.partial(handler, message)
            )
        if paused_reason is not None:
            self._count_drop(topic, paused_reason)
            return
        if waiting:
            return
        try:
            self._queue.put_nowait((topic, functools.partial(self._take_latest, topic)))
        except queue.Full:
            with self._lock:
                self._latest[topic] = None
            self._count_drop(
                topic,
                "the Rerun sink takes no data (a viewer that stopped reading?), so the bridge's queue is full",
            )

    def _take_latest(self, topic: str) -> None:
        """The writer's entry of a "latest" topic: handles the newest message that waits."""
        with self._lock:
            task = self._latest[topic]
            self._latest[topic] = None
        if task is not None:
            task()

    def _count_drop(self, topic: str, reason: str) -> None:
        now = time.monotonic()
        with self._lock:
            self._dropped[topic] = self._dropped.get(topic, 0) + 1
            total = sum(self._dropped.values())
            log = now - self._last_drop_log >= bridge_module.LOG_PERIOD_S
            if log:
                self._last_drop_log = now
        if log:
            _LOGGER.warning(
                "dropping the bus's messages: %s (%d dropped so far)", reason, total
            )

    def report(self) -> str | None:
        text = self._bridge.report(
            bus_rejected=self.bus_rejected(), sink_dropped=self.dropped()
        )
        if text is not None:
            _LOGGER.warning("%s", text)
        return text

    def start(self) -> None:
        """Starts the writer thread and the bus's receive thread."""
        self._started = True
        self._writer.start()
        self.bus.start()

    def _write(self) -> None:
        """The writer thread: the queued handlers, the periodic flush and report, then the last flush at stop()."""
        now = time.monotonic()
        next_flush = now + self._flush_period
        next_report = now + self._report_period
        while not self._stop_writer.is_set():
            self._run_resume_task()
            wait = max(0.0, min(next_flush, next_report) - time.monotonic())
            try:
                topic, task = self._queue.get(timeout=wait)
            except queue.Empty:
                pass
            else:
                self._run_message(topic, task)
            now = time.monotonic()
            if now >= next_flush:
                self._run_task(self._bridge.flush)
                next_flush = _next_deadline(next_flush, self._flush_period, now)
            if now >= next_report:
                self._run_task(self.report)
                next_report = _next_deadline(next_report, self._report_period, now)
        self._finish()

    def _run_resume_task(self) -> None:
        with self._lock:
            pending = self._resume_task
            self._resume_task = None
        if pending is None:
            return
        pauses, reconnect = pending
        self._run_task(reconnect)
        with self._lock:
            if self._pauses == pauses:
                self._paused_reason = None

    def _run_message(self, topic: str, task: _Task) -> None:
        with self._lock:
            paused_reason = self._paused_reason
        if paused_reason is not None:
            # Queued before pause(): the viewer it was for is gone.
            self._count_drop(topic, paused_reason)
            return
        self._run_task(task)

    def _run_task(self, task: _Task) -> None:
        """Runs one task of the writer; the time it takes is what writer_stalled_for() measures."""
        with self._lock:
            self._busy_since = time.monotonic()
        try:
            task()
        # pylint: disable-next=broad-exception-caught  # The writer thread's outermost call; the bridge's own raise nothing.
        except Exception:
            _LOGGER.warning("a task of the bridge's writer raised", exc_info=True)
        finally:
            with self._lock:
                self._busy_since = None

    def _finish(self) -> None:
        """The last work after the bus closed: the messages still queued, the throttled scene, the plots, the report.

        Then flushes the recording, unless the viewer is unreachable, which nothing reaches.
        """
        deadline = time.monotonic() + SHUTDOWN_DRAIN_TIMEOUT_S
        while time.monotonic() < deadline:
            try:
                topic, task = self._queue.get_nowait()
            except queue.Empty:
                break
            self._run_message(topic, task)
        if not self.paused:
            self._run_task(self._bridge.draw_pending_scene)
        self._run_task(self._bridge.flush)
        self._run_task(self.report)
        if self.paused:
            delivered = False
        else:
            delivered = self._bridge.flush_recording(self._recording_flush_timeout)
        with self._lock:
            self._delivered = delivered

    def stop(self, recording_flush_timeout: float = SHUTDOWN_FLUSH_TIMEOUT_S) -> bool:
        """Closes the bus, draws what is still queued, sends the buffered plots and flushes the recording. Idempotent.

        Args:
            recording_flush_timeout: how long the recording may take to reach its sink [s].

        Returns:
            False when the recording did not reach its sink within the timeout, the viewer was unreachable, or the
            writer was still blocked in rerun-sdk after the drain and the flush may take: the caller must then not
            disconnect the recording or wait for it, which rerun-sdk 0.38 does without end.
        """
        if self._stopped:
            with self._lock:
                return self._delivered
        self._stopped = True
        # The receive thread only queues, so the bus closes whatever the sink does.
        self.bus.close()
        self._recording_flush_timeout = recording_flush_timeout
        if not self._started:
            self._finish()
        else:
            self._stop_writer.set()
            self._writer.join(
                timeout=SHUTDOWN_DRAIN_TIMEOUT_S
                + recording_flush_timeout
                + _JOIN_MARGIN_S
            )
            if self._writer.is_alive():
                _LOGGER.warning(
                    "the Rerun sink took no data for the last %.1f s; stopping without it",
                    self.writer_stalled_for(),
                )
                with self._lock:
                    self._delivered = False
        with self._lock:
            return self._delivered

    def __enter__(self) -> "BusBridge":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()
