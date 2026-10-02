"""The C++ visualization publisher, the bus and the Rerun bridge together: the contracts' entity paths are in the .rrd.

The driver (VisualizationPublisherDriver.cpp) publishes the G1's scene and telemetry, computed by the publisher from
synthetic observations, policies and robot/state samples, on a bus bound to an ephemeral port that it prints. The
bridge subscribes to it, as on the laptop, with its save sink, and the recording is read back.
"""

import os
import shutil
import subprocess
import tempfile
import time
import unittest
from typing import Callable, Dict, List, Set

import robot_ipc
from humanoid_mpc_msgs import telemetry_series_pb2

import rrd_contents
from humanoid_mpc_ipc import topics
from humanoid_rerun_viewer import bridge
from humanoid_rerun_viewer import bus_bridge
from humanoid_rerun_viewer import scene_contract
from humanoid_rerun_viewer import telemetry_contract
from humanoid_rerun_viewer import urdf_model

TIMEOUT_S = 60.0
DRIVER = os.environ["VISUALIZATION_PUBLISHER_DRIVER"]
URDF = "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf"
# The telemetryFrames of the G1's task file (robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.yaml).
FRAMES = ("foot_l_contact", "foot_r_contact", "pelvis", "torso_link")


def wait_for(condition: Callable[[], bool], timeout: float = TIMEOUT_S) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return condition()


class EndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = tempfile.mkdtemp()
        rrd_path = os.path.join(cls.directory, "visualization.rrd")
        recording = bridge.new_recording("test_visualization_publisher")
        recording.save(rrd_path)
        cls.model = urdf_model.load_urdf(URDF)
        cls.bridge = bridge.RerunBridge(recording, cls.model)
        cls.bridge.log_static()

        # The driver's log goes to a file, so that a full pipe can never hold it up.
        driver_log_path = os.path.join(cls.directory, "driver.log")
        driver_log = open(driver_log_path, "w", encoding="utf-8")
        driver = subprocess.Popen(
            [DRIVER, "--duration=60s"],
            stdout=subprocess.PIPE,
            stderr=driver_log,
            text=True,
        )
        try:
            line = driver.stdout.readline()
            if not line.startswith("PORT "):
                driver.kill()
                driver.wait(timeout=10)
                with open(driver_log_path, encoding="utf-8") as log:
                    raise AssertionError(
                        f"the driver printed {line!r}; its log: {log.read()}"
                    )
            port = int(line.split()[1])
            network = robot_ipc.NetworkConfig(
                nodes=(robot_ipc.NodeEndpoint("visualization", "127.0.0.1", port),)
            )
            feed = bus_bridge.BusBridge(cls.bridge, network, flush_period=0.02)
            feed.start()
            # The raw messages too, for what an .rrd file does not keep: the order of the groups.
            cls.series: List[telemetry_series_pb2.TelemetrySeries] = []

            def keep(message: telemetry_series_pb2.TelemetrySeries) -> None:
                if len(cls.series) < 3:
                    copy = telemetry_series_pb2.TelemetrySeries()
                    copy.CopyFrom(message)
                    cls.series.append(copy)

            listener = robot_ipc.Bus("", network)
            listener.subscribe(
                topics.VIZ_TELEMETRY,
                telemetry_series_pb2.TelemetrySeries,
                keep,
                delivery=robot_ipc.Delivery.ALL,
            )
            listener.start()
            handled = cls.bridge.statistics.handled
            cls.received = wait_for(
                lambda: handled.get(topics.VIZ_SCENE, 0) >= 5
                and handled.get(topics.VIZ_TELEMETRY, 0) >= 50
            )
            wait_for(lambda: len(cls.series) >= 3)
            listener.close()
            feed.stop()
            # After the bridge has stopped: what it handled is what the recording holds.
            cls.handled: Dict[str, int] = dict(handled)
        finally:
            driver.terminate()
            driver.wait(timeout=10)
            driver.stdout.close()
            driver_log.close()
        recording.disconnect()
        cls.contents = rrd_contents.RrdContents(rrd_path)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.directory)

    def test_the_bridge_received_scenes_and_telemetry(self) -> None:
        self.assertTrue(self.received, self.handled)
        self.assertEqual(
            self.bridge.statistics.malformed_count(),
            0,
            self.bridge.statistics.malformed,
        )
        self.assertEqual(self.bridge.statistics.handler_error_count(), 0)

    def test_every_robot_instance_moves_every_link_with_visuals(self) -> None:
        links = self.model.links_with_visuals()
        self.assertGreater(len(links), 20)
        for style in scene_contract.ROBOT_INSTANCES:
            for link in links:
                self.assertTrue(
                    self.contents.has(
                        scene_contract.link_path(style.name, link),
                        "Transform3D:translation",
                    ),
                    f"{style.name}/{link}",
                )

    def test_every_marker_of_the_contract_is_drawn(self) -> None:
        for marker in scene_contract.MARKERS:
            path = scene_contract.entity_path(scene_contract.WORLD_ROOT, marker.path)
            self.assertIn(path, self.contents.entities, path)
        drawn = {
            "world/markers/contact_forces": "Arrows3D:vectors",
            "world/markers/corner_forces": "Arrows3D:vectors",
            "world/markers/center_of_pressure": "Points3D:positions",
            "world/plan/footholds": "Points3D:positions",
            "world/plan/end_effectors": "LineStrips3D:strips",
            "world/plan/base": "LineStrips3D:strips",
            "world/plan/com": "LineStrips3D:strips",
        }
        for path, component in drawn.items():
            self.assertTrue(self.contents.has(path, component), f"{path} {component}")
        self.assertTrue(
            any(
                len(vectors) > 0
                for vectors in self.contents.values(
                    "world/markers/contact_forces", "Arrows3D:vectors"
                )
            )
        )

    def test_the_telemetry_is_exactly_the_contracts_groups(self) -> None:
        plotted: Set[str] = {
            path[len(scene_contract.TELEMETRY_ROOT) + 1 :]
            for path in self.contents.entities
            if path.startswith(scene_contract.TELEMETRY_ROOT + "/")
            and self.contents.has(path, "Scalars:scalars")
        }
        expected = {group.path for group in telemetry_contract.all_groups(FRAMES)}
        self.assertEqual(plotted, expected)
        for group in telemetry_contract.all_groups(FRAMES):
            path = f"{scene_contract.TELEMETRY_ROOT}/{group.path}"
            names = self.contents.values(path, "SeriesLines:names")
            self.assertTrue(names, path)
            if not group.robot_dependent:
                self.assertEqual(tuple(names[-1]), group.names, path)
        # Every series that arrived is in the file.
        rows = self.contents.entities[
            f"{scene_contract.TELEMETRY_ROOT}/base_pose/position_x"
        ].rows["Scalars:scalars"]
        self.assertEqual(rows, self.handled[topics.VIZ_TELEMETRY])

    def test_a_message_carries_the_contracts_groups_in_its_order(self) -> None:
        self.assertEqual(len(self.series), 3)
        groups = telemetry_contract.all_groups(FRAMES)
        for series in self.series:
            self.assertEqual(
                [group.path for group in series.groups],
                [group.path for group in groups],
            )
            for message_group, contract_group in zip(series.groups, groups):
                self.assertEqual(
                    len(message_group.names),
                    len(message_group.values),
                    contract_group.path,
                )
                if not contract_group.robot_dependent:
                    self.assertEqual(tuple(message_group.names), contract_group.names)
        times = [series.time for series in self.series]
        self.assertEqual(times, sorted(times))


if __name__ == "__main__":
    unittest.main()
