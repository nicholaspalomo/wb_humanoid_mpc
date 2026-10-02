"""Tests for model_sandbox.py: the joint positions and the scene they make, when the scene goes out, the slider ranges,
the binary's scene over a loopback bus, and the slider window where Tk can open one."""

import math
import os
import socket
import tempfile
import threading
import unittest
from typing import List

import robot_ipc
from humanoid_mpc_ipc import topics
from humanoid_mpc_msgs import visualization_scene_pb2
from operator_test_support import requires_display

from humanoid_rerun_viewer import model_sandbox
from humanoid_rerun_viewer import scene_contract
from humanoid_rerun_viewer import urdf_kinematics

_RUNFILES_ROOT = os.path.join(os.environ.get("TEST_SRCDIR", ""), "_main")
G1_URDF = os.path.join(
    _RUNFILES_ROOT, "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf"
)

TWO_JOINTS = """<robot name="two">
  <link name="base"/><link name="upper"/><link name="lower"/>
  <joint name="hip" type="revolute">
    <parent link="base"/><child link="upper"/><axis xyz="0 1 0"/><limit lower="-1" upper="1"/>
  </joint>
  <joint name="spin" type="continuous">
    <parent link="upper"/><child link="lower"/><origin xyz="0 0 -0.5"/><axis xyz="0 0 1"/>
  </joint>
</robot>
"""


def two_joint_scene() -> model_sandbox.SandboxScene:
    return model_sandbox.SandboxScene(
        urdf_kinematics.parse_kinematic_tree(TWO_JOINTS, "two.urdf")
    )


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class SandboxSceneTest(unittest.TestCase):
    def test_the_joints_start_nominal_and_are_clamped(self) -> None:
        scene = two_joint_scene()
        self.assertEqual(scene.positions(), {"hip": 0.0, "spin": 0.0})
        self.assertEqual(scene.set_position("hip", 3.0), 1.0)
        self.assertEqual(scene.set_position("spin", 7.0), 7.0)
        self.assertEqual(scene.positions(), {"hip": 1.0, "spin": 7.0})
        scene.reset()
        self.assertEqual(scene.positions(), {"hip": 0.0, "spin": 0.0})

    def test_what_is_not_a_joint_position_is_refused(self) -> None:
        scene = two_joint_scene()
        with self.assertRaisesRegex(KeyError, "the joints are: hip, spin"):
            scene.set_position("knee", 0.0)
        with self.assertRaises(ValueError):
            scene.set_position("hip", math.nan)

    def test_the_version_counts_changes_only(self) -> None:
        scene = two_joint_scene()
        start = scene.version
        scene.set_position("hip", 0.0)
        scene.reset()
        self.assertEqual(scene.version, start)
        scene.set_position("hip", 0.5)
        scene.set_position("hip", 0.5)
        self.assertEqual(scene.version, start + 1)
        scene.reset()
        self.assertEqual(scene.version, start + 2)

    def test_the_scene_is_the_measured_instance_with_every_link(self) -> None:
        scene = two_joint_scene()
        scene.set_position("hip", 0.5)
        message = scene.scene(2.5)
        self.assertEqual(message.time, 2.5)
        self.assertEqual(len(message.robots), 1)
        robot = message.robots[0]
        self.assertEqual(robot.name, scene_contract.MEASURED)
        self.assertEqual(list(robot.link_names), ["base", "upper", "lower"])
        self.assertEqual(len(robot.link_poses), 3)
        lower = robot.link_poses[2]
        # The hip pitches the leg about +y: the lower link's origin swings from below the hip toward -x.
        self.assertAlmostEqual(lower.position.x, -0.5 * math.sin(0.5))
        self.assertAlmostEqual(lower.position.z, -0.5 * math.cos(0.5))
        norm = math.sqrt(
            lower.orientation.x**2
            + lower.orientation.y**2
            + lower.orientation.z**2
            + lower.orientation.w**2
        )
        self.assertAlmostEqual(norm, 1.0)
        # No markers: the bridge draws the robot alone.
        self.assertEqual(len(message.arrows) + len(message.spheres), 0)


class ScenePublisherTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = two_joint_scene()
        self.clock = FakeClock()
        self.sent: List[visualization_scene_pb2.VisualizationScene] = []
        self.accept = True

        def publish(message: visualization_scene_pb2.VisualizationScene) -> bool:
            if self.accept:
                self.sent.append(message)
            return self.accept

        self.publisher = model_sandbox.ScenePublisher(
            self.scene, publish, republish_period=1.0, clock=self.clock
        )

    def test_it_publishes_on_a_change_and_every_period(self) -> None:
        self.assertTrue(self.publisher.tick())
        self.assertFalse(self.publisher.tick())
        self.clock.now += 0.5
        self.scene.set_position("hip", 0.2)
        self.assertTrue(self.publisher.tick())
        self.clock.now += 0.9
        self.assertFalse(self.publisher.tick())
        self.clock.now += 0.2
        self.assertTrue(self.publisher.tick())
        self.assertEqual(len(self.sent), 3)
        for message, expected in zip(self.sent, (0.0, 0.5, 1.6)):
            self.assertAlmostEqual(message.time, expected)
        self.assertEqual(self.publisher.published, 3)

    def test_a_refused_scene_goes_out_on_the_next_tick(self) -> None:
        self.accept = False
        self.assertFalse(self.publisher.tick())
        self.accept = True
        self.assertTrue(self.publisher.tick())
        self.assertEqual(len(self.sent), 1)

    def test_the_period_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            model_sandbox.ScenePublisher(self.scene, lambda message: True, 0.0)


class SliderRangeTest(unittest.TestCase):
    def test_limits_or_one_turn(self) -> None:
        tree = urdf_kinematics.parse_kinematic_tree(TWO_JOINTS, "two.urdf")
        hip, spin = tree.movable_joints()
        self.assertEqual(model_sandbox.slider_range(hip), (-1.0, 1.0))
        self.assertEqual(model_sandbox.slider_range(spin), (-math.pi, math.pi))

    def test_the_joint_sources_are_named(self) -> None:
        self.assertEqual(set(model_sandbox.JOINT_SOURCES), {"nominal", "sliders"})


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class BinaryTest(unittest.TestCase):
    def test_the_nominal_pose_of_the_g1_reaches_the_bus(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            network_file = os.path.join(directory, "network.textproto")
            with open(network_file, "w") as stream:
                stream.write(
                    f'nodes {{ name: "mpc" host: "127.0.0.1" port: {free_port()} }}\n'
                )
            network = robot_ipc.load_network_config(network_file)
            received: List[visualization_scene_pb2.VisualizationScene] = []
            arrived = threading.Event()

            def on_scene(message: visualization_scene_pb2.VisualizationScene) -> None:
                copy = visualization_scene_pb2.VisualizationScene()
                copy.CopyFrom(message)
                received.append(copy)
                arrived.set()

            listener = robot_ipc.Bus("", network)
            listener.subscribe(
                topics.VIZ_SCENE,
                visualization_scene_pb2.VisualizationScene,
                on_scene,
                delivery="all",
            )
            with listener:
                status: List[int] = []
                runner = threading.Thread(
                    target=lambda: status.append(
                        model_sandbox.main(
                            [
                                "--urdf",
                                G1_URDF,
                                "--network_config",
                                network_file,
                                "--joint_source",
                                "nominal",
                                "--republish_period",
                                "0.1",
                                "--duration",
                                "3",
                            ]
                        )
                    )
                )
                runner.start()
                self.assertTrue(arrived.wait(timeout=10.0), "no viz/scene arrived")
                runner.join(timeout=20.0)
            self.assertEqual(status, [model_sandbox.EXIT_OK])
            tree = urdf_kinematics.load_kinematic_tree(G1_URDF)
            robot = received[-1].robots[0]
            self.assertEqual(robot.name, scene_contract.MEASURED)
            self.assertEqual(list(robot.link_names), list(tree.links))
            root = robot.link_poses[list(tree.links).index(tree.root_link)]
            self.assertEqual(
                (root.position.x, root.position.y, root.position.z), (0, 0, 0)
            )

    def test_a_missing_urdf_is_a_usage_error(self) -> None:
        self.assertEqual(
            model_sandbox.main(
                ["--urdf", "/nonexistent.urdf", "--joint_source", "nominal"]
            ),
            model_sandbox.EXIT_USAGE,
        )


@requires_display
class SliderWindowTest(unittest.TestCase):
    def test_a_slider_per_joint_moves_the_scene(self) -> None:
        import tkinter

        root = tkinter.Tk()
        root.withdraw()
        try:
            scene = two_joint_scene()
            window = model_sandbox.SliderWindow(scene, root)
            self.assertEqual(set(window.sliders), {"hip", "spin"})
            self.assertEqual(window.sliders["hip"].cget("from"), -1.0)
            window.sliders["hip"].set(0.4)
            root.update()
            self.assertAlmostEqual(scene.positions()["hip"], 0.4, places=2)
            window.reset()
            root.update()
            self.assertEqual(scene.positions()["hip"], 0.0)
            self.assertEqual(float(window.sliders["hip"].get()), 0.0)
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
