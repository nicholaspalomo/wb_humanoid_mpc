"""Feeds a RerunBridge from the IPC bus: the subscriptions, the periodic flush and report, start and stop.

The bridge only subscribes (it binds no node), so it can run on any machine of the network file and connects to every
node. ZeroMQ reconnects on its own, so the publishers may start, stop and restart in any order.

| Topic | Message | Delivery | Handler |
|---|---|---|---|
| viz/scene | VisualizationScene | latest | RerunBridge.handle_scene |
| viz/telemetry | TelemetrySeries | all (every sample is plotted) | RerunBridge.handle_telemetry |
| robot/fsm_state | FsmState | latest | RerunBridge.handle_fsm_state |
| mpc/status | MpcStatus | all (every solve time is plotted) | RerunBridge.handle_mpc_status |
| robot/loop_timing | LoopTiming | all | RerunBridge.handle_loop_timing |
"""

import logging
from typing import Optional

import robot_ipc
from humanoid_mpc_ipc import topics
from humanoid_mpc_msgs import fsm_state_pb2
from humanoid_mpc_msgs import loop_timing_pb2
from humanoid_mpc_msgs import mpc_status_pb2
from humanoid_mpc_msgs import telemetry_series_pb2
from humanoid_mpc_msgs import visualization_scene_pb2

from humanoid_rerun_viewer import bridge as bridge_module

_LOGGER = logging.getLogger("humanoid_rerun_viewer")

# LINT.IfChange(bus_defaults)
DEFAULT_FLUSH_PERIOD_S = 0.05
DEFAULT_REPORT_PERIOD_S = 5.0
# The last flush may wait this long for the sink [s]; a viewer that went away must not hang the exit.
SHUTDOWN_FLUSH_TIMEOUT_S = 5.0
# LINT.ThenChange(//humanoid_nmpc/humanoid_rerun_viewer/README.md:bus_defaults)

SUBSCRIBED_TOPICS = (
    topics.VIZ_SCENE,
    topics.VIZ_TELEMETRY,
    topics.ROBOT_FSM_STATE,
    topics.MPC_STATUS,
    topics.ROBOT_LOOP_TIMING,
)


class BusBridge:
    """Subscribes `bridge` to the topics of every node of `network`.

    Args:
        bridge: the handlers.
        network: the nodes to connect to.
        flush_period: how often the buffered plots are sent [s].
        report_period: how often problems since the last report are logged [s].
    """

    def __init__(
        self,
        bridge: bridge_module.RerunBridge,
        network: robot_ipc.NetworkConfig,
        flush_period: float = DEFAULT_FLUSH_PERIOD_S,
        report_period: float = DEFAULT_REPORT_PERIOD_S,
    ) -> None:
        if flush_period <= 0.0 or report_period <= 0.0:
            raise ValueError("the flush and report periods must be positive")
        self._bridge = bridge
        self._bus = robot_ipc.Bus("", network)
        self._stopped = False
        self._delivered = True
        try:
            self._bus.subscribe(
                topics.VIZ_SCENE,
                visualization_scene_pb2.VisualizationScene,
                bridge.handle_scene,
                robot_ipc.Delivery.LATEST,
            )
            self._bus.subscribe(
                topics.VIZ_TELEMETRY,
                telemetry_series_pb2.TelemetrySeries,
                bridge.handle_telemetry,
                robot_ipc.Delivery.ALL,
            )
            self._bus.subscribe(
                topics.ROBOT_FSM_STATE,
                fsm_state_pb2.FsmState,
                bridge.handle_fsm_state,
                robot_ipc.Delivery.LATEST,
            )
            self._bus.subscribe(
                topics.MPC_STATUS,
                mpc_status_pb2.MpcStatus,
                bridge.handle_mpc_status,
                robot_ipc.Delivery.ALL,
            )
            self._bus.subscribe(
                topics.ROBOT_LOOP_TIMING,
                loop_timing_pb2.LoopTiming,
                bridge.handle_loop_timing,
                robot_ipc.Delivery.ALL,
            )
            self._bus.add_periodic_callback(flush_period, self._flush)
            self._bus.add_periodic_callback(report_period, self.report)
        except BaseException:
            self._bus.close()
            raise

    @property
    def bus(self) -> robot_ipc.Bus:
        return self._bus

    def bus_rejected(self) -> int:
        """Messages of the subscribed topics the bus rejected: wrong type, not three frames, or not parsing."""
        statistics = self._bus.statistics()
        return sum(
            statistics[topic].rejected
            for topic in SUBSCRIBED_TOPICS
            if topic in statistics
        )

    def _flush(self) -> None:
        self._bridge.flush()

    def report(self) -> Optional[str]:
        text = self._bridge.report(bus_rejected=self.bus_rejected())
        if text is not None:
            _LOGGER.warning("%s", text)
        return text

    def start(self) -> None:
        self._bus.start()

    def stop(self, recording_flush_timeout: float = SHUTDOWN_FLUSH_TIMEOUT_S) -> bool:
        """Closes the bus, sends what is still buffered and flushes the recording. Idempotent.

        Returns:
            False when the recording did not reach its sink within the timeout (a viewer that is gone): the caller must
            then not disconnect the recording or wait for it, which rerun-sdk 0.38 does without end.
        """
        if self._stopped:
            return self._delivered
        self._stopped = True
        self._bus.close()
        self._bridge.flush()
        self.report()
        self._delivered = self._bridge.flush_recording(recording_flush_timeout)
        return self._delivered

    def __enter__(self) -> "BusBridge":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()
