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
"""Every robot's default stance stands on flat ground, and its MPC contact rectangle lies on the MuJoCo sole.

The default posture - the reference file's default_base_height and default_joint_state, which the MPC returns to, and
the task file's initial_state, which it starts from - must put each foot's contact frame on the terrain (z = 0) with the
sole level. A frame above the ground makes every stance a planned free fall of the base by that height onto the real
ground, a frame below it a penetration the contact constraints fight; a tilted sole a planned contact on one edge. The
R1 shipped with its contact frames 6.2 mm above the ground and its soles tilted by 0.85 degrees: its hip pitch joint's
frame is rolled by 25 degrees, so the ankle pitch that levels the foot is not minus the hip's.

The tolerances: 1 mm of height, a fifth of the 5 mm radius of the sole spheres and the size of the MPC's touch-down
height offset; and 0.1 degree of tilt, which across the longest sole here (0.25 m) is a height difference of 0.4 mm
from heel to toe, inside the 1 mm.

The contact rectangle (task file contacts.contact_rectangle, in the contact frame) bounds the center of pressure the MPC
may plan. The simulated foot touches the ground at its sole spheres only, so the center of pressure it can produce is
inside the convex hull of their centers; a rectangle reaching beyond it plans a center of pressure the foot cannot hold.
Each corner must lie inside that hull, and each sphere's bottom on the contact frame's plane (the 1 mm again), on both
feet of every robot.

The forward kinematics are this file's own, of the URDF that Pinocchio loads for the MPC (a contact frame is its parent
joint's frame translated by contacts.contact_frame_translation, unrotated, as createPinocchioModel() adds it), and the
spheres are MuJoCo's, of the MJCF the simulator loads; the two frames are checked to coincide first.
"""

import math
import os
import unittest
from xml.etree import ElementTree

from humanoid_mpc_config import reference_file_pb2
from humanoid_mpc_config import task_file_pb2
import mujoco
import numpy as np

import nproto_textproto

# [m] How far a contact frame of the default stance may be from the ground, and a sole sphere's bottom from its plane.
HEIGHT_TOLERANCE_M = 1e-3
# [deg] How far the sole's normal may be from the vertical.
TILT_TOLERANCE_DEG = 0.1
# [m] How far the URDF's and the MJCF's frame of a foot may be apart at the zero configuration (six digits in the files).
FRAME_TOLERANCE_M = 1e-5

# Each robot configuration: its description's URDF and MJCF, and its config directory, relative to robot_models/.
_ROBOTS = {
    "drc_atlas": (
        "drc_atlas/drc_atlas_description/urdf/atlas",
        "drc_atlas/drc_atlas_centroidal_mpc/config",
    ),
    "engineai_sa01": (
        "engineai_sa01/engineai_sa01_description/urdf/zq_sa01",
        "engineai_sa01/engineai_sa01_centroidal_mpc/config",
    ),
    "unitree_g1": (
        "unitree_g1/g1_description/urdf/g1_29dof",
        "unitree_g1/g1_centroidal_mpc/config",
    ),
    "unitree_g1_wb_mpc": (
        "unitree_g1/g1_description/urdf/g1_29dof",
        "unitree_g1/g1_wb_mpc/config",
    ),
    "unitree_r1": (
        "unitree_r1/unitree_r1_description/urdf/R1",
        "unitree_r1/unitree_r1_centroidal_mpc/config",
    ),
}

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def _rotation_about(axis: np.ndarray, angle: float) -> np.ndarray:
    """Returns the rotation by `angle` [rad] about the unit vector `axis` (Rodrigues)."""
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return (
        np.eye(3)
        + math.sin(angle) * cross
        + (1.0 - math.cos(angle)) * cross @ cross
    )


def _rotation_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Returns Rz(yaw) Ry(pitch) Rx(roll): the URDF's fixed-axis roll-pitch-yaw, and the ZYX Euler angles of a state."""
    return (
        _rotation_about(np.array([0.0, 0.0, 1.0]), yaw)
        @ _rotation_about(np.array([0.0, 1.0, 0.0]), pitch)
        @ _rotation_about(np.array([1.0, 0.0, 0.0]), roll)
    )


def _attribute(element: ElementTree.Element, name: str) -> str:
    """Returns the attribute `name` of `element`, failing with the element named when it is missing."""
    value = element.get(name)
    if value is None:
        raise ValueError(
            f"<{element.tag} name='{element.get('name')}'> has no {name}=..."
        )
    return value


def _child_attribute(element: ElementTree.Element, tag: str, name: str) -> str:
    """Returns the attribute `name` of `element`'s child `tag`, failing with the element named when either is missing."""
    child = element.find(tag)
    if child is None:
        raise ValueError(
            f"<{element.tag} name='{element.get('name')}'> has no <{tag}>"
        )
    return _attribute(child, name)


class _UrdfJoint:
    """A URDF joint: its parent and child links, its origin and its axis.

    Attributes:
        name: The joint's name.
        kind: The URDF type: revolute, continuous, prismatic or fixed.
        parent: The parent link's name.
        child: The child link's name.
        translation: The origin's translation in the parent link's frame [m].
        rotation: The origin's rotation in the parent link's frame.
        axis: The unit axis in the joint's (the child link's) frame.
    """

    def __init__(self, element: ElementTree.Element):
        self.name = _attribute(element, "name")
        self.kind = _attribute(element, "type")
        self.parent = _child_attribute(element, "parent", "link")
        self.child = _child_attribute(element, "child", "link")
        origin = element.find("origin")
        xyz = origin.get("xyz") if origin is not None else None
        rpy = origin.get("rpy") if origin is not None else None
        self.translation = (
            np.array([float(v) for v in xyz.split()]) if xyz else np.zeros(3)
        )
        self.rotation = _rotation_from_rpy(
            *([float(v) for v in rpy.split()] if rpy else [0.0, 0.0, 0.0])
        )
        axis = element.find("axis")
        direction = (
            np.array([float(v) for v in _attribute(axis, "xyz").split()])
            if axis is not None
            else np.array([1.0, 0.0, 0.0])
        )
        self.axis = direction / np.linalg.norm(direction)


def _joint_frame(
    joints: dict[str, _UrdfJoint],
    joint_name: str,
    base: tuple[np.ndarray, np.ndarray],
    positions: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Returns the world (translation, rotation) of joint `joint_name`'s frame, its child link's.

    Args:
        joints: The URDF's joints by name.
        joint_name: The joint whose frame to return.
        base: The world (translation, rotation) of the root link.
        positions: The joint positions [rad or m] by name; a joint not in it is at zero.

    Returns:
        The frame's translation [m] and rotation in the world.
    """
    by_child = {joint.child: joint for joint in joints.values()}
    chain = []
    link = joints[joint_name].child
    while link in by_child:
        chain.append(by_child[link])
        link = by_child[link].parent
    translation, rotation = base
    for joint in reversed(chain):
        translation = translation + rotation @ joint.translation
        rotation = rotation @ joint.rotation
        value = positions.get(joint.name, 0.0)
        if joint.kind in ("revolute", "continuous"):
            rotation = rotation @ _rotation_about(joint.axis, value)
        elif joint.kind == "prismatic":
            translation = translation + rotation @ (value * joint.axis)
    return translation, rotation


def _paths(robot: str) -> tuple[str, str, str, str]:
    """Returns the URDF, MJCF, reference file and task file paths of `robot`."""
    description, config = _ROBOTS[robot]
    return (
        os.path.join(_ROOT, f"{description}.urdf"),
        os.path.join(_ROOT, f"{description}.xml"),
        os.path.join(_ROOT, config, "command", "reference.textproto"),
        os.path.join(_ROOT, config, "mpc", "task.textproto"),
    )


def _load(
    robot: str,
) -> tuple[
    dict[str, _UrdfJoint],
    reference_file_pb2.ReferenceFile,
    task_file_pb2.TaskFile,
]:
    """Returns `robot`'s URDF joints, reference file and task file."""
    urdf, _, reference_path, task_path = _paths(robot)
    parsed = [
        _UrdfJoint(element)
        for element in ElementTree.parse(urdf).getroot().findall("joint")
    ]
    joints = {joint.name: joint for joint in parsed}
    reference = nproto_textproto.load_textproto(
        reference_path, reference_file_pb2.ReferenceFile
    )
    task = nproto_textproto.load_textproto(task_path, task_file_pb2.TaskFile)
    return joints, reference, task


def _contact_translation(task: task_file_pb2.TaskFile) -> np.ndarray:
    """Returns contacts.contact_frame_translation, the contact frame in its parent joint's frame [m]."""
    translation = task.contacts.contact_frame_translation
    return np.array([translation.x, translation.y, translation.z])


class DefaultStanceTest(unittest.TestCase):
    """The default stance of every robot configuration: its feet flat on the ground."""

    def _assert_stands(
        self,
        robot: str,
        what: str,
        base: tuple[np.ndarray, np.ndarray],
        positions: dict[str, float],
    ) -> None:
        """Asserts that both contact frames of `robot` with its base at `base` and its joints at `positions` stand.

        Args:
            robot: The robot configuration, a key of _ROBOTS.
            what: Which stance this is, for the messages.
            base: The world (translation, rotation) of the root link.
            positions: The joint positions by name; a joint not in it is at zero.
        """
        joints, _, task = _load(robot)
        translation = _contact_translation(task)
        parents = list(task.model_settings.contact_parent_joint_names)
        self.assertEqual(len(parents), 2, f"{robot}: two feet")
        for parent in parents:
            origin, rotation = _joint_frame(joints, parent, base, positions)
            contact = origin + rotation @ translation
            tilt = math.degrees(math.acos(min(1.0, rotation[2, 2])))
            label = f"{robot} {what}, the contact frame on {parent}"
            self.assertLess(
                abs(contact[2]),
                HEIGHT_TOLERANCE_M,
                f"{label} is {contact[2] * 1000:.2f} mm above the ground",
            )
            self.assertLess(
                tilt,
                TILT_TOLERANCE_DEG,
                f"{label}: the sole is tilted by {tilt:.3f} deg",
            )

    def test_the_reference_default_stance_stands_on_the_ground(self):
        for robot in _ROBOTS:
            _, reference, _ = _load(robot)
            positions = {
                entry.joint: entry.value
                for entry in reference.default_joint_state
            }
            base = (
                np.array([0.0, 0.0, reference.default_base_height]),
                np.eye(3),
            )
            self._assert_stands(robot, "reference default", base, positions)

    def test_the_task_initial_state_stands_on_the_ground(self):
        for robot in _ROBOTS:
            _, _, task = _load(robot)
            state = task.initial_state
            positions = {
                entry.joint: entry.value for entry in state.joint_positions
            }
            orientation = state.base_orientation
            base = (
                np.array(
                    [
                        state.base_position.x,
                        state.base_position.y,
                        state.base_position.z,
                    ]
                ),
                _rotation_from_rpy(
                    orientation.roll, orientation.pitch, orientation.yaw
                ),
            )
            self._assert_stands(robot, "initial_state", base, positions)

    def test_the_initial_state_is_the_reference_default(self):
        # The MPC starts where it returns to: the same base height and joints.
        for robot in _ROBOTS:
            _, reference, task = _load(robot)
            self.assertAlmostEqual(
                task.initial_state.base_position.z,
                reference.default_base_height,
                delta=1e-9,
                msg=robot,
            )
            initial = {
                entry.joint: entry.value
                for entry in task.initial_state.joint_positions
            }
            for entry in reference.default_joint_state:
                self.assertAlmostEqual(
                    initial.get(entry.joint, 0.0),
                    entry.value,
                    delta=1e-9,
                    msg=f"{robot} {entry.joint}",
                )


def _cross(a: np.ndarray, b: np.ndarray) -> float:
    """Returns the z component of the cross product of the 2D vectors `a` and `b`."""
    return float(a[0] * b[1] - a[1] * b[0])


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Returns the convex hull of the 2D `points`, counter-clockwise (Andrew's monotone chain)."""
    ordered = sorted((float(point[0]), float(point[1])) for point in points)
    lower: list[tuple[float, float]] = []
    upper: list[tuple[float, float]] = []
    for point in ordered:
        while (
            len(lower) >= 2
            and _cross(
                np.subtract(lower[-1], lower[-2]), np.subtract(point, lower[-2])
            )
            <= 0.0
        ):
            lower.pop()
        lower.append(point)
    for point in reversed(ordered):
        while (
            len(upper) >= 2
            and _cross(
                np.subtract(upper[-1], upper[-2]), np.subtract(point, upper[-2])
            )
            <= 0.0
        ):
            upper.pop()
        upper.append(point)
    return np.array(lower[:-1] + upper[:-1])


def _inside(point: np.ndarray, hull: np.ndarray) -> bool:
    """Returns whether `point` is inside or on the counter-clockwise convex polygon `hull` (to round-off)."""
    return all(
        _cross(hull[(i + 1) % len(hull)] - hull[i], point - hull[i]) >= -1e-12
        for i in range(len(hull))
    )


class ContactRectangleTest(unittest.TestCase):
    """The MPC's contact rectangle of every robot against the MuJoCo soles."""

    def test_the_hull_algorithm_on_a_known_polygon(self):
        hull = _convex_hull(
            np.array(
                [[0.0, 0.0], [2.0, 0.0], [1.0, 1.0], [2.0, 2.0], [0.0, 2.0]]
            )
        )
        self.assertEqual(len(hull), 4, "the interior point is not a vertex")
        self.assertTrue(_inside(np.array([1.0, 1.0]), hull))
        self.assertTrue(
            _inside(np.array([2.0, 2.0]), hull), "a vertex is on the hull"
        )
        self.assertFalse(_inside(np.array([2.1, 1.0]), hull))

    def test_the_rectangle_is_inside_the_sole_spheres_of_both_feet(self):
        for robot in _ROBOTS:
            joints, _, task = _load(robot)
            _, mjcf, _, _ = _paths(robot)
            model = mujoco.MjModel.from_xml_path(mjcf)
            data = mujoco.MjData(model)
            mujoco.mj_kinematics(model, data)
            translation = _contact_translation(task)
            rectangle = task.contacts.contact_rectangle
            corners = np.array(
                [
                    [rectangle.x_min, rectangle.y_min],
                    [rectangle.x_max, rectangle.y_min],
                    [rectangle.x_max, rectangle.y_max],
                    [rectangle.x_min, rectangle.y_max],
                ]
            )
            # The MJCF free body sits where the URDF's root link does at qpos0.
            root_body = model.jnt_bodyid[0]
            base = (
                data.xpos[root_body].copy(),
                data.xmat[root_body].reshape(3, 3).copy(),
            )
            for parent in task.model_settings.contact_parent_joint_names:
                joint = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, parent
                )
                self.assertGreaterEqual(
                    joint, 0, f"{robot}: the MJCF has no joint {parent}"
                )
                body = model.jnt_bodyid[joint]
                self.assertTrue(
                    np.allclose(model.jnt_pos[joint], 0.0),
                    f"{robot} {parent}: the joint is at its body's origin",
                )
                origin, rotation = _joint_frame(joints, parent, base, {})
                self.assertLess(
                    np.linalg.norm(data.xpos[body] - origin),
                    FRAME_TOLERANCE_M,
                    f"{robot} {parent}: frame position",
                )
                self.assertLess(
                    np.abs(data.xmat[body].reshape(3, 3) - rotation).max(),
                    FRAME_TOLERANCE_M,
                    f"{robot} {parent}: frame rotation",
                )
                spheres = [
                    geom
                    for geom in range(model.ngeom)
                    if model.geom_bodyid[geom] == body
                    and model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_SPHERE
                    and (
                        model.geom_contype[geom] != 0
                        or model.geom_conaffinity[geom] != 0
                    )
                ]
                self.assertGreaterEqual(
                    len(spheres),
                    3,
                    f"{robot} {parent}: a sole of contact spheres",
                )
                centers = np.array(
                    [model.geom_pos[geom] - translation for geom in spheres]
                )
                for geom, center in zip(spheres, centers):
                    bottom = center[2] - model.geom_size[geom][0]
                    self.assertLess(
                        abs(bottom),
                        HEIGHT_TOLERANCE_M,
                        f"{robot} {parent}: a sole sphere's bottom is"
                        f" {bottom * 1000:.2f} mm off the contact frame's"
                        " plane",
                    )
                hull = _convex_hull(centers[:, :2])
                for corner in corners:
                    self.assertTrue(
                        _inside(corner, hull),
                        f"{robot} {parent}: the contact rectangle's corner"
                        f" {corner.tolist()} is outside the sole spheres' hull"
                        f" {np.round(hull, 4).tolist()}",
                    )


if __name__ == "__main__":
    unittest.main()
