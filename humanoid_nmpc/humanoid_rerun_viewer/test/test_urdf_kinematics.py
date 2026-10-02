"""Tests for urdf_kinematics.py: the forward kinematics of a small URDF against hand-computed poses, rigid-body
properties on every shipped robot at random joint positions, and the URDFs it refuses."""

import math
import os
import unittest
from typing import Dict

import numpy as np

from humanoid_rerun_viewer import urdf_kinematics
from humanoid_rerun_viewer import urdf_model

_RUNFILES_ROOT = os.path.join(os.environ.get("TEST_SRCDIR", ""), "_main")
SHIPPED_URDFS = (
    "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf",
    "robot_models/engineai_sa01/engineai_sa01_description/urdf/zq_sa01.urdf",
    "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf",
    "robot_models/unitree_r1/unitree_r1_description/urdf/R1.urdf",
)

# base --(shoulder: revolute about z, 1 m along x)--> arm --(slide: prismatic along x)--> hand --(fixed)--> tool,
# and a finger that mimics the shoulder at half its angle.
ARM = """<robot name="arm">
  <link name="base"/>
  <link name="arm"/>
  <link name="hand"/>
  <link name="tool"/>
  <link name="finger"/>
  <joint name="shoulder" type="revolute">
    <parent link="base"/><child link="arm"/>
    <origin xyz="1 0 0"/>
    <axis xyz="0 0 2"/>
    <limit lower="-2" upper="2"/>
  </joint>
  <joint name="slide" type="prismatic">
    <parent link="arm"/><child link="hand"/>
    <origin xyz="1 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="0.5" upper="1.0"/>
  </joint>
  <joint name="mount" type="fixed">
    <parent link="hand"/><child link="tool"/>
    <origin xyz="0 0 0.5" rpy="0 0 1.5707963267948966"/>
  </joint>
  <joint name="finger" type="continuous">
    <parent link="hand"/><child link="finger"/>
    <axis xyz="0 0 1"/>
    <mimic joint="shoulder" multiplier="0.5" offset="0.1"/>
  </joint>
</robot>
"""


def runfile(path: str) -> str:
    candidate = os.path.join(_RUNFILES_ROOT, path)
    if os.path.exists(candidate):
        return candidate
    return os.path.join(os.path.dirname(__file__), "..", "..", "..", path)


def yaw_of(quaternion) -> float:
    x, y, z, w = quaternion
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class SmallTreeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tree = urdf_kinematics.parse_kinematic_tree(ARM, "arm.urdf")

    def test_the_tree_and_its_settable_joints(self) -> None:
        self.assertEqual(self.tree.root_link, "base")
        self.assertEqual(
            [joint.name for joint in self.tree.movable_joints()], ["shoulder", "slide"]
        )
        # 0 is outside the slide's limits: the nominal position is the nearest limit.
        self.assertEqual(self.tree.nominal_positions(), {"shoulder": 0.0, "slide": 0.5})

    def test_the_poses_at_the_nominal_positions(self) -> None:
        poses = self.tree.link_poses(self.tree.nominal_positions())
        np.testing.assert_allclose(poses["base"][0], (0, 0, 0), atol=1e-12)
        np.testing.assert_allclose(poses["arm"][0], (1, 0, 0), atol=1e-12)
        np.testing.assert_allclose(poses["hand"][0], (2.5, 0, 0), atol=1e-12)
        np.testing.assert_allclose(poses["tool"][0], (2.5, 0, 0.5), atol=1e-12)
        self.assertAlmostEqual(yaw_of(poses["tool"][1]), math.pi / 2)
        self.assertAlmostEqual(yaw_of(poses["finger"][1]), 0.1)

    def test_the_poses_with_the_shoulder_turned(self) -> None:
        poses = self.tree.link_poses({"shoulder": math.pi / 2, "slide": 0.75})
        # The arm turns about its own origin; what hangs on it swings around that origin.
        np.testing.assert_allclose(poses["arm"][0], (1, 0, 0), atol=1e-12)
        np.testing.assert_allclose(poses["hand"][0], (1, 1.75, 0), atol=1e-12)
        np.testing.assert_allclose(poses["tool"][0], (1, 1.75, 0.5), atol=1e-12)
        self.assertAlmostEqual(yaw_of(poses["arm"][1]), math.pi / 2)
        self.assertAlmostEqual(abs(yaw_of(poses["tool"][1])), math.pi)
        self.assertAlmostEqual(
            yaw_of(poses["finger"][1]), math.pi / 2 + math.pi / 4 + 0.1
        )

    def test_positions_are_clamped_and_missing_ones_are_nominal(self) -> None:
        poses = self.tree.link_poses({"slide": 5.0})
        np.testing.assert_allclose(poses["hand"][0], (3.0, 0, 0), atol=1e-12)
        np.testing.assert_allclose(poses["arm"][0], (1, 0, 0), atol=1e-12)

    def test_the_rotation_convention_is_the_bridges(self) -> None:
        # urdf_model.quaternion_from_rpy draws the visuals; the link poses must use the same convention.
        for rpy in ((0.3, -0.2, 1.1), (math.pi / 2, 0.0, 0.0), (0.0, 1.4, -2.9)):
            with self.subTest(rpy=rpy):
                ours = urdf_kinematics.quaternion_from_rotation(
                    urdf_kinematics.rotation_from_rpy(*rpy)
                )
                theirs = urdf_model.quaternion_from_rpy(*rpy)
                self.assertAlmostEqual(abs(float(np.dot(ours, theirs))), 1.0)


class ShippedRobotTest(unittest.TestCase):
    def test_every_link_of_every_robot_gets_a_rigid_pose(self) -> None:
        generator = np.random.default_rng(seed=7)
        for path in SHIPPED_URDFS:
            with self.subTest(urdf=path):
                tree = urdf_kinematics.load_kinematic_tree(runfile(path))
                self.assertEqual(
                    tree.root_link, urdf_model.load_urdf(runfile(path)).root_link
                )
                self.assertGreater(len(tree.movable_joints()), 10)
                for joint in tree.movable_joints():
                    self.assertEqual(
                        joint.clamp(tree.nominal_positions()[joint.name]),
                        tree.nominal_positions()[joint.name],
                    )
                positions: Dict[str, float] = {
                    joint.name: float(generator.uniform(-3.0, 3.0))
                    for joint in tree.movable_joints()
                }
                poses = tree.link_poses(positions)
                self.assertEqual(set(poses), set(tree.links))
                for link, (translation, quaternion) in poses.items():
                    self.assertTrue(np.all(np.isfinite(translation)), link)
                    self.assertAlmostEqual(float(np.linalg.norm(quaternion)), 1.0)
                # A joint moves its child about the joint's own origin: the distance from the parent's origin to
                # the child's is that of the joint origin, whatever the positions.
                for joint in tree.joints:
                    if joint.type in urdf_kinematics.PRISMATIC_TYPES:
                        continue
                    distance = np.linalg.norm(
                        np.subtract(poses[joint.child][0], poses[joint.parent][0])
                    )
                    self.assertAlmostEqual(
                        float(distance),
                        float(np.linalg.norm(joint.origin_translation)),
                        places=9,
                    )


class RefusedTest(unittest.TestCase):
    def refuse(self, text: str, pattern: str) -> None:
        with self.assertRaisesRegex(urdf_model.UrdfError, pattern):
            urdf_kinematics.parse_kinematic_tree(text, "bad.urdf")

    def test_what_is_not_one_tree_is_refused(self) -> None:
        self.refuse(
            '<robot name="r"><link name="a"/><link name="b"/></robot>',
            r"one tree.*\['a', 'b'\]",
        )
        self.refuse(
            '<robot name="r"><link name="a"/><link name="b"/><link name="c"/>'
            '<joint name="j" type="fixed"><parent link="a"/><child link="c"/></joint>'
            '<joint name="k" type="fixed"><parent link="b"/><child link="c"/></joint></robot>',
            "child of joints 'j' and 'k'",
        )
        self.refuse(
            '<robot name="r"><link name="a"/>'
            '<joint name="j" type="fixed"><parent link="a"/><child link="z"/></joint></robot>',
            "unknown link 'z'",
        )

    def test_malformed_joints_are_refused(self) -> None:
        link = '<link name="a"/><link name="b"/>'
        ends = '<parent link="a"/><child link="b"/>'
        self.refuse(
            f'<robot name="r">{link}<joint name="j" type="hinge">{ends}</joint></robot>',
            "unknown type 'hinge'",
        )
        self.refuse(
            f'<robot name="r">{link}<joint name="j" type="revolute">{ends}<axis xyz="0 0 0"/></joint></robot>',
            "axis is zero",
        )
        self.refuse(
            f'<robot name="r">{link}<joint name="j" type="fixed">{ends}<origin xyz="1 x 0"/></joint></robot>',
            r'xyz="1 x 0" is not 3 numbers',
        )
        self.refuse(
            f'<robot name="r">{link}<joint name="j" type="revolute">{ends}<mimic joint="none"/></joint></robot>',
            "mimics 'none'",
        )
        self.refuse("<robot", "bad.urdf")
        self.refuse("<model/>", "not <robot>")


if __name__ == "__main__":
    unittest.main()
