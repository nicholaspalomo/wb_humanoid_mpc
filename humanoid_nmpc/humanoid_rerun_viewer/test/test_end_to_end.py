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

"""The bridge on a loopback bus: messages published with robot_ipc end up in an .rrd file, and the binary stops cleanly.

Concurrent tests never collide on a port. Every publisher binds an ephemeral port (EPHEMERAL_PORT) and the bridge, or
the binary's network file, is then given the port the kernel chose; a publisher that restarts binds that port again,
which the connections of its first run keep out of the kernel's ephemeral choices while they linger in TIME_WAIT. A
network file that names a node nobody runs names a port this test holds bound, without listening, until it ends.
Publishers repeat until the bridge has what it needs, which absorbs ZeroMQ's asynchronous connect.
"""

from collections.abc import Callable
import os
import shutil
import signal
import socket
import subprocess
import tempfile
import time
import unittest

from humanoid_mpc_msgs import fsm_state_pb2
from humanoid_mpc_msgs import loop_timing_pb2
from humanoid_mpc_msgs import mpc_status_pb2

from humanoid_mpc_ipc import topics
from humanoid_rerun_viewer import blueprint
from humanoid_rerun_viewer import bridge
from humanoid_rerun_viewer import bus_bridge
from humanoid_rerun_viewer import scene_contract
from humanoid_rerun_viewer import status_contract
from humanoid_rerun_viewer import telemetry_contract
from humanoid_rerun_viewer import urdf_model
import robot_ipc
import rrd_contents
import synthetic_messages

TIMEOUT_S = 30.0
PUBLISHER = "viz"
LINKS = synthetic_messages.link_names_with_visuals()


def wait_for(condition: Callable[[], bool], timeout: float = TIMEOUT_S) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


def reserve_unused_port(test: unittest.TestCase) -> int:
    """A loopback port nothing listens on and nothing else can bind until `test` ends: a connect to it is refused.

    The socket stays bound without listening. Binding the port number and closing the socket instead would hand the
    number back to the kernel, which may give it to another test, or to an outgoing connection, before it is used.
    """
    reservation = socket.socket()
    test.addCleanup(reservation.close)
    reservation.bind(("127.0.0.1", 0))
    return reservation.getsockname()[1]


def loopback_network(port: int) -> robot_ipc.NetworkConfig:
    return robot_ipc.NetworkConfig(
        nodes=(robot_ipc.NodeEndpoint(PUBLISHER, "127.0.0.1", port),)
    )


class Publisher:
    """Publishes a scene, a telemetry sample and the three status messages per step, with the robot's clock."""

    def __init__(self, bus: robot_ipc.Bus) -> None:
        self.bus = bus
        self.time = 0.0

    def step(self) -> None:
        """Advances the robot's clock by 10 ms and publishes one message on each topic the bridge subscribes to."""
        self.time += 0.01
        self.bus.publish(
            topics.VIZ_SCENE, synthetic_messages.full_scene(self.time, LINKS)
        )
        self.bus.publish(
            topics.VIZ_TELEMETRY, synthetic_messages.full_telemetry(self.time)
        )
        self.bus.publish(
            topics.ROBOT_FSM_STATE,
            fsm_state_pb2.FsmState(mode="WB_MPC", mpc_healthy=True),
        )
        status = mpc_status_pb2.MpcStatus(observation_time=self.time)
        status.solver_status.healthy = True
        status.solver_status.solve_time_ms = 4.0
        self.bus.publish(topics.MPC_STATUS, status)
        self.bus.publish(
            topics.ROBOT_LOOP_TIMING,
            loop_timing_pb2.LoopTiming(target_period_s=0.002, policy_age_s=0.01),
        )


def telemetry_beyond(
    publisher: Publisher, handled: dict[str, int], count: int
) -> Callable[[], bool]:
    """A wait_for() condition: publishes one step, then whether the bridge handled more than `count` telemetry samples."""

    def received() -> bool:
        publisher.step()
        return handled.get(topics.VIZ_TELEMETRY, 0) > count

    return received


class LoopbackBusTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.directory)
        self.publisher_bus = robot_ipc.Bus(
            PUBLISHER, loopback_network(robot_ipc.EPHEMERAL_PORT)
        )
        self.addCleanup(self.publisher_bus.close)
        self.publisher_bus.start()

    def test_published_messages_reach_the_recording(self) -> None:
        rrd_path = os.path.join(self.directory, "loopback.rrd")
        recording = bridge.new_recording("test_end_to_end")
        recording.save(rrd_path)
        recording.send_blueprint(blueprint.build_blueprint("base_link"))
        model = urdf_model.load_urdf(
            synthetic_messages.write_sample_package(self.directory)
        )
        the_bridge = bridge.RerunBridge(recording, model)
        the_bridge.log_static()
        feed = bus_bridge.BusBridge(
            the_bridge,
            loopback_network(self.publisher_bus.bound_port),
            flush_period=0.02,
        )
        self.addCleanup(feed.stop)
        feed.start()

        publisher = Publisher(self.publisher_bus)
        handled = the_bridge.statistics.handled

        def enough() -> bool:
            publisher.step()
            # A FsmState on viz/telemetry, which delivers every message: the bus rejects it, the bridge never sees it.
            self.publisher_bus.publish(topics.VIZ_TELEMETRY, fsm_state_pb2.FsmState())
            return (
                handled.get(topics.VIZ_SCENE, 0) >= 3
                and handled.get(topics.VIZ_TELEMETRY, 0) >= 20
                and all(
                    handled.get(topic, 0) >= 1
                    for topic in (
                        topics.ROBOT_FSM_STATE,
                        topics.MPC_STATUS,
                        topics.ROBOT_LOOP_TIMING,
                    )
                )
                and feed.bus_rejected() >= 1
            )

        self.assertTrue(wait_for(enough), dict(handled))
        feed.stop()
        recording.disconnect()
        self.assertEqual(the_bridge.statistics.handler_error_count(), 0)
        self.assertEqual(the_bridge.statistics.malformed_count(), 0)

        contents = rrd_contents.RrdContents(rrd_path)
        for style in scene_contract.ROBOT_INSTANCES:
            for link in LINKS:
                self.assertTrue(
                    contents.has(
                        scene_contract.link_path(style.name, link),
                        "Transform3D:translation",
                    )
                )
            self.assertTrue(
                contents.has(
                    scene_contract.visual_path(style.name, "base_link", 0),
                    "Asset3D:blob",
                )
            )
        for marker in scene_contract.MARKERS:
            self.assertIn(
                scene_contract.entity_path(scene_contract.WORLD_ROOT, marker.path),
                contents.entities,
            )
        for group in telemetry_contract.all_groups(synthetic_messages.FRAMES):
            path = f"{scene_contract.TELEMETRY_ROOT}/{group.path}"
            self.assertTrue(contents.has(path, "Scalars:scalars"), path)
            self.assertTrue(contents.has(path, "SeriesLines:names"), path)
        for path in (
            status_contract.FSM_STATE_LOG,
            status_contract.MPC_STATUS_LOG,
            status_contract.LOOP_TIMING_LOG,
            status_contract.MPC_SOLVE_TIME.path,
            status_contract.LOOP_PERIOD.path,
        ):
            self.assertIn(path, contents.entities)
        # Every telemetry sample that arrived is in the file, none twice.
        rows = contents.entities[
            f"{scene_contract.TELEMETRY_ROOT}/base_pose/position_x"
        ].rows["Scalars:scalars"]
        self.assertEqual(rows, handled[topics.VIZ_TELEMETRY])
        self.assertIn(blueprint.SCENE_VIEW_NAME, contents.view_names())

    def test_a_restarted_publisher_is_picked_up_again(self) -> None:
        # ZeroMQ reconnects on its own: the bridge needs no restart when the visualization publisher restarts. The
        # first run binds the port the kernel chooses; the restart binds it again.
        port = self.publisher_bus.bound_port
        recording = bridge.new_recording("test_end_to_end_restart")
        recording.save(os.path.join(self.directory, "restart.rrd"))
        the_bridge = bridge.RerunBridge(recording, None)
        feed = bus_bridge.BusBridge(the_bridge, loopback_network(port))
        self.addCleanup(feed.stop)
        feed.start()
        handled = the_bridge.statistics.handled
        for restart in range(2):
            if restart == 0:
                publisher_bus = self.publisher_bus
            else:
                publisher_bus = robot_ipc.Bus(PUBLISHER, loopback_network(port))
                self.addCleanup(publisher_bus.close)
                publisher_bus.start()
            before = handled.get(topics.VIZ_TELEMETRY, 0)
            self.assertTrue(
                wait_for(
                    telemetry_beyond(Publisher(publisher_bus), handled, before + 5)
                ),
                f"restart {restart}: {dict(handled)}",
            )
            publisher_bus.close()
        feed.stop()
        recording.disconnect()
        self.assertEqual(the_bridge.statistics.handler_error_count(), 0)


class BinaryTest(unittest.TestCase):
    """The py_binary itself: it bridges until a signal, then exits 0 with a complete recording."""

    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.directory)
        self.binary = os.path.abspath(os.environ["RERUN_VIEWER_BINARY"])
        self.urdf = synthetic_messages.write_sample_package(self.directory)

    def write_network_file(self, port: int) -> str:
        path = os.path.join(self.directory, "network.textproto")
        with open(path, "w", encoding="utf-8") as network_file:
            network_file.write(
                f'nodes {{ name: "{PUBLISHER}" host: "127.0.0.1" port: {port} }}\n'
            )
        return path

    def run_until_signal(self, signum: int) -> None:
        """Runs the binary against a publisher until it is sent `signum`; checks that it exits 0 with a recording."""
        publisher_bus = robot_ipc.Bus(
            PUBLISHER, loopback_network(robot_ipc.EPHEMERAL_PORT)
        )
        self.addCleanup(publisher_bus.close)
        publisher_bus.start()
        port = publisher_bus.bound_port
        rrd_path = os.path.join(self.directory, f"signal_{signum}.rrd")
        log_path = os.path.join(self.directory, f"signal_{signum}.log")
        with open(log_path, "w", encoding="utf-8") as log:
            # pylint: disable-next=consider-using-with  # The process outlives this block; kill_if_running() ends it.
            process = subprocess.Popen(
                [
                    self.binary,
                    f"--urdf={self.urdf}",
                    f"--network_config={self.write_network_file(port)}",
                    "--rerun_sink=save",
                    f"--rrd_path={rrd_path}",
                    "--flush_period=0.02",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )

        def kill_if_running() -> None:
            if process.poll() is None:
                process.kill()
                process.wait()

        self.addCleanup(kill_if_running)

        def log_text() -> str:
            with open(log_path, encoding="utf-8") as log_file:
                return log_file.read()

        publisher = Publisher(publisher_bus)
        self.assertTrue(
            wait_for(
                lambda: "subscribed to" in log_text() or process.poll() is not None
            ),
            log_text(),
        )
        # Long enough for the subscriber to connect and take a few messages.
        for _ in range(100):
            publisher.step()
            time.sleep(0.01)
        process.send_signal(signum)
        self.assertEqual(process.wait(timeout=TIMEOUT_S), 0, log_text())
        self.assertIn("stopped:", log_text())
        contents = rrd_contents.RrdContents(rrd_path)
        self.assertTrue(
            contents.has(
                scene_contract.visual_path(scene_contract.MEASURED, "base_link", 0),
                "Asset3D:blob",
            )
        )
        self.assertTrue(
            contents.has(
                scene_contract.link_path(scene_contract.MEASURED, "base_link"),
                "Transform3D:translation",
            ),
            log_text(),
        )
        self.assertTrue(
            contents.has("telemetry/base_pose/position_x", "Scalars:scalars")
        )

    def test_sigint_stops_it_cleanly(self) -> None:
        self.run_until_signal(signal.SIGINT)

    def test_sigterm_stops_it_cleanly(self) -> None:
        self.run_until_signal(signal.SIGTERM)

    def run_binary(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [self.binary, *arguments],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
            check=False,
        )

    def test_configuration_errors_exit_2(self) -> None:
        network = self.write_network_file(reserve_unused_port(self))
        save = ["--rerun_sink=save", f"--rrd_path={self.directory}/x.rrd"]
        for arguments, expected in (
            (["--rerun_sink=save", f"--network_config={network}"], "--rrd_path"),
            (["--rerun_sink=no_such_sink"], "invalid choice"),
            (
                [f"--network_config={self.directory}/missing.textproto", *save],
                "missing",
            ),
            (
                [
                    f"--urdf={self.directory}/missing.urdf",
                    f"--network_config={network}",
                    *save,
                ],
                "missing.urdf",
            ),
        ):
            with self.subTest(arguments=arguments):
                result = self.run_binary(*arguments)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn(expected, result.stderr)

    def test_a_viewer_that_is_not_there_does_not_hang_the_exit(self) -> None:
        network = self.write_network_file(reserve_unused_port(self))
        started = time.monotonic()
        result = self.run_binary(
            f"--network_config={network}",
            "--rerun_sink=connect",
            f"--rerun_url=rerun+http://127.0.0.1:{reserve_unused_port(self)}/proxy",
            "--duration=0.5",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("unreachable", result.stderr)
        # About 5 s of rerun-sdk waiting for the viewer, the run, and at most 5 s of the last flush.
        self.assertLess(time.monotonic() - started, TIMEOUT_S)

    def test_a_duration_ends_the_run(self) -> None:
        network = self.write_network_file(reserve_unused_port(self))
        rrd_path = os.path.join(self.directory, "duration.rrd")
        result = self.run_binary(
            f"--network_config={network}",
            "--rerun_sink=save",
            f"--rrd_path={rrd_path}",
            "--duration=0.5",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("no --urdf", result.stderr)
        self.assertTrue(rrd_contents.RrdContents(rrd_path).blueprints)


if __name__ == "__main__":
    unittest.main()
