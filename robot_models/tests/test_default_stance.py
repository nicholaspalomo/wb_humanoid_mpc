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
"""Every robot's default stance stands on flat ground, and its MPC contacts lie on the MuJoCo feet.

The default posture - the reference file's nom_pos_base_rt_surface_ewrt_world_z and default_joint_state, which the MPC returns to, and
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
Each corner must lie inside that hull, and each sphere's bottom on the contact frame's plane (the 1 mm again), on every
6-DoF contact of every robot.

A 3-DoF (point) contact (task file model_settings.contacts_3dof), such as a quadruped's foot, carries a force and no
moment, so it has no sole to level and no rectangle. Its MuJoCo foot is exactly one sphere in the parent joint's body or
a body below it, and the sphere's lowest point along the link's z axis (its center minus its radius, in the link frame)
is the contact frame, within the 1 mm in each axis. A second sphere would be a second contact point the MPC does not
plan for, and a sphere away from the frame puts the simulated contact where the MPC does not expect it. The contact
frame stands on the ground in the default stances like a sole's frame; its tilt is not checked.

A rolling contact (task file model_settings.contacts_rolling), a wheel, carries a force at the wheel's lowest point, which
moves over the rim as the wheel turns: its frame is the wheel center (pos_wheel_rt_link_ewrt_link on the wheel's own
joint), and its contact point is the center a wheel radius below, on flat ground (humanoid_nmpc/docs/rolling_contact).
That point stands on the ground in the default stances within the 1 mm. Its MuJoCo wheel is exactly one collision geom in
the wheel joint's body and the bodies below it, a cylinder coaxial with the wheel joint's axis, centered at the wheel
center and of the task file's wheel_radius, each within the 1 mm: another geom would roll on another radius or touch the
ground where the MPC plans no contact.

The forward kinematics are this file's own, of the URDF that Pinocchio loads for the MPC (a contact frame is its parent
joint's frame translated by its pos_contact_rt_link_ewrt_link, unrotated, as createPinocchioModel() adds it), and the
spheres are MuJoCo's, of the MJCF the simulator loads; the two frames are checked to coincide first. The contacts are
read in the MPC's order: model_settings.contacts_3dof, each with its parent_joint_name and pos_contact_rt_link_ewrt_link,
then model_settings.contacts_rolling, each with its parent_joint_name, pos_wheel_rt_link_ewrt_link and wheel_radius, then
the 6-DoF contacts on model_settings.contact_parent_joint_names at the shared contacts.pos_contact_rt_link_ewrt_link.
"""

import dataclasses
import enum
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
    "anymal_b": (
        "anybotics_anymal_b/anymal_b_description/urdf/anymal_b",
        "anybotics_anymal_b/anymal_b_centroidal_mpc/config",
    ),
    "anymal_c_wheels": (
        "anybotics_anymal_c_wheels/anymal_c_wheels_description/urdf/anymal_c_wheels",
        "anybotics_anymal_c_wheels/anymal_c_wheels_centroidal_mpc/config",
    ),
    "drc_atlas": (
        "drc_atlas/drc_atlas_description/urdf/atlas",
        "drc_atlas/drc_atlas_centroidal_mpc/config",
    ),
    "engineai_sa01": (
        "engineai_sa01/engineai_sa01_description/urdf/zq_sa01",
        "engineai_sa01/engineai_sa01_centroidal_mpc/config",
    ),
    "tron1": (
        "limx_tron1/tron1_description/urdf/tron1",
        "limx_tron1/tron1_centroidal_mpc/config",
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


def _matrix_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
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
        _matrix_from_axis_angle(np.array([0.0, 0.0, 1.0]), yaw)
        @ _matrix_from_axis_angle(np.array([0.0, 1.0, 0.0]), pitch)
        @ _matrix_from_axis_angle(np.array([1.0, 0.0, 0.0]), roll)
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
        origin_translation: The origin's translation in the parent link's frame [m].
        origin_rotmat: The origin's rotation matrix in the parent link's frame.
        axis_joint_ewrt_link: The unit axis in the joint's (the child link's) frame.
    """

    def __init__(self, element: ElementTree.Element):
        self.name = _attribute(element, "name")
        self.kind = _attribute(element, "type")
        self.parent = _child_attribute(element, "parent", "link")
        self.child = _child_attribute(element, "child", "link")
        origin = element.find("origin")
        xyz = origin.get("xyz") if origin is not None else None
        rpy = origin.get("rpy") if origin is not None else None
        self.origin_translation = (
            np.array([float(v) for v in xyz.split()]) if xyz else np.zeros(3)
        )
        self.origin_rotmat = _rotation_from_rpy(
            *([float(v) for v in rpy.split()] if rpy else [0.0, 0.0, 0.0])
        )
        axis = element.find("axis")
        raw_axis_joint_ewrt_link = (
            np.array([float(v) for v in _attribute(axis, "xyz").split()])
            if axis is not None
            else np.array([1.0, 0.0, 0.0])
        )
        self.axis_joint_ewrt_link = raw_axis_joint_ewrt_link / np.linalg.norm(
            raw_axis_joint_ewrt_link
        )


def _compute_pose_link_rt_world(
    joints: dict[str, _UrdfJoint],
    joint_name: str,
    pose_base_rt_world: tuple[np.ndarray, np.ndarray],
    joint_positions: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Returns the world (translation, rotation) of joint `joint_name`'s frame, its child link's.

    Args:
        joints: The URDF's joints by name.
        joint_name: The joint whose frame to return.
        pose_base_rt_world: The world (translation, rotation) of the root link.
        joint_positions: The joint positions [rad or m] by name; a joint not in it is at zero.

    Returns:
        The frame's translation [m] and rotation in the world.
    """
    by_child = {joint.child: joint for joint in joints.values()}
    chain = []
    link = joints[joint_name].child
    while link in by_child:
        chain.append(by_child[link])
        link = by_child[link].parent
    pos_link_rt_world_ewrt_world, rotmat_link_rt_world = pose_base_rt_world
    for joint in reversed(chain):
        pos_link_rt_world_ewrt_world = (
            pos_link_rt_world_ewrt_world
            + rotmat_link_rt_world @ joint.origin_translation
        )
        rotmat_link_rt_world = rotmat_link_rt_world @ joint.origin_rotmat
        value = joint_positions.get(joint.name, 0.0)
        if joint.kind in ("revolute", "continuous"):
            rotmat_link_rt_world = (
                rotmat_link_rt_world
                @ _matrix_from_axis_angle(joint.axis_joint_ewrt_link, value)
            )
        elif joint.kind == "prismatic":
            pos_link_rt_world_ewrt_world = (
                pos_link_rt_world_ewrt_world
                + rotmat_link_rt_world @ (value * joint.axis_joint_ewrt_link)
            )
    return pos_link_rt_world_ewrt_world, rotmat_link_rt_world


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


class _ContactKind(enum.Enum):
    """The kind of a contact of the MPC's contact set (ModelSettings::ContactKind)."""

    POINT = "point"
    ROLLING = "rolling"
    SOLE = "sole"


@dataclasses.dataclass(frozen=True)
class _Contact:
    """One contact of the MPC's contact set.

    Attributes:
        parent_joint: The joint whose frame the contact frame hangs from.
        pos_contact_rt_link_ewrt_link: The contact frame in its parent joint's frame [m], shape (3,); a rolling
            contact's is the wheel center.
        kind: A 3-DoF point contact, a rolling contact (a wheel) or a 6-DoF sole.
        wheel_radius: [m] The wheel's radius of a rolling contact; 0 for the others.
    """

    parent_joint: str
    pos_contact_rt_link_ewrt_link: np.ndarray
    kind: _ContactKind
    wheel_radius: float = 0.0


def _xyz(xyz) -> np.ndarray:
    """Returns an Xyz message as an array, shape (3,)."""
    return np.array([xyz.x, xyz.y, xyz.z])


def _contacts(task: task_file_pb2.TaskFile) -> list[_Contact]:
    """Returns the task file's contacts in the MPC's order: the point contacts, the rolling ones, then the 6-DoF ones."""
    contacts = [
        _Contact(
            parent_joint=point.parent_joint_name,
            pos_contact_rt_link_ewrt_link=_xyz(
                point.pos_contact_rt_link_ewrt_link
            ),
            kind=_ContactKind.POINT,
        )
        for point in task.model_settings.contacts_3dof
    ]
    contacts.extend(
        _Contact(
            parent_joint=wheel.parent_joint_name,
            pos_contact_rt_link_ewrt_link=_xyz(
                wheel.pos_wheel_rt_link_ewrt_link
            ),
            kind=_ContactKind.ROLLING,
            wheel_radius=wheel.wheel_radius,
        )
        for wheel in task.model_settings.contacts_rolling
    )
    pos_contact_rt_link_ewrt_link = _xyz(
        task.contacts.pos_contact_rt_link_ewrt_link
    )
    contacts.extend(
        _Contact(
            parent_joint=parent,
            pos_contact_rt_link_ewrt_link=pos_contact_rt_link_ewrt_link,
            kind=_ContactKind.SOLE,
        )
        for parent in task.model_settings.contact_parent_joint_names
    )
    return contacts


class DefaultStanceTest(unittest.TestCase):
    """The default stance of every robot configuration: its feet flat on the ground."""

    def _assert_stands(
        self,
        robot: str,
        what: str,
        pose_base_rt_world: tuple[np.ndarray, np.ndarray],
        joint_positions: dict[str, float],
    ) -> None:
        """Asserts that every contact frame of `robot` at `pose_base_rt_world` with its joints at `joint_positions` stands.

        Args:
            robot: The robot configuration, a key of _ROBOTS.
            what: Which stance this is, for the messages.
            pose_base_rt_world: The world (translation, rotation) of the root link.
            joint_positions: The joint positions by name; a joint not in it is at zero.
        """
        joints, _, task = _load(robot)
        contacts = _contacts(task)
        self.assertTrue(contacts, f"{robot}: the task file has no contacts")
        for contact in contacts:
            pos_link_rt_world_ewrt_world, rotmat_link_rt_world = (
                _compute_pose_link_rt_world(
                    joints,
                    contact.parent_joint,
                    pose_base_rt_world,
                    joint_positions,
                )
            )
            # A rolling contact's frame is the wheel center; its contact point is the wheel's lowest point, a radius
            # below on flat ground.
            pos_contact_rt_world_ewrt_world = (
                pos_link_rt_world_ewrt_world
                + rotmat_link_rt_world @ contact.pos_contact_rt_link_ewrt_link
                - np.array([0.0, 0.0, contact.wheel_radius])
            )
            tiltangle_link_rt_world = math.acos(
                min(1.0, rotmat_link_rt_world[2, 2])
            )
            label = (
                f"{robot} {what}, the contact point on {contact.parent_joint}"
            )
            self.assertLess(
                abs(pos_contact_rt_world_ewrt_world[2]),
                HEIGHT_TOLERANCE_M,
                f"{label} is {pos_contact_rt_world_ewrt_world[2] * 1000:.2f} mm"
                " above the ground",
            )
            # A point contact has no sole to level, and a wheel turns.
            if contact.kind != _ContactKind.SOLE:
                continue
            self.assertLess(
                math.degrees(tiltangle_link_rt_world),
                TILT_TOLERANCE_DEG,
                f"{label}: the sole is tilted by"
                f" {math.degrees(tiltangle_link_rt_world):.3f} deg",
            )

    def test_the_reference_default_stance_stands_on_the_ground(self):
        for robot in _ROBOTS:
            _, reference, _ = _load(robot)
            joint_positions = {
                entry.joint: entry.value
                for entry in reference.default_joint_state
            }
            pose_base_rt_world = (
                np.array(
                    [0.0, 0.0, reference.nom_pos_base_rt_surface_ewrt_world_z]
                ),
                np.eye(3),
            )
            self._assert_stands(
                robot, "reference default", pose_base_rt_world, joint_positions
            )

    def test_the_task_initial_state_stands_on_the_ground(self):
        for robot in _ROBOTS:
            _, _, task = _load(robot)
            state = task.initial_state
            joint_positions = {
                entry.joint: entry.value for entry in state.joint_positions
            }
            eulerzyx_base_rt_world = state.eulerzyx_base_rt_world
            pose_base_rt_world = (
                np.array(
                    [
                        state.pos_base_rt_world_ewrt_world.x,
                        state.pos_base_rt_world_ewrt_world.y,
                        state.pos_base_rt_world_ewrt_world.z,
                    ]
                ),
                _rotation_from_rpy(
                    eulerzyx_base_rt_world.roll,
                    eulerzyx_base_rt_world.pitch,
                    eulerzyx_base_rt_world.yaw,
                ),
            )
            self._assert_stands(
                robot, "initial_state", pose_base_rt_world, joint_positions
            )

    def test_the_initial_state_is_the_reference_default(self):
        # The MPC starts where it returns to: the same base height and joints.
        for robot in _ROBOTS:
            _, reference, task = _load(robot)
            self.assertAlmostEqual(
                task.initial_state.pos_base_rt_world_ewrt_world.z,
                reference.nom_pos_base_rt_surface_ewrt_world_z,
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


def _subtree(model: mujoco.MjModel, body: int) -> set[int]:
    """Returns `body` and every body below it."""
    bodies = {body}
    for child in range(body + 1, model.nbody):
        if model.body_parentid[child] in bodies:
            bodies.add(child)
    return bodies


def _collision_spheres(model: mujoco.MjModel, bodies: set[int]) -> list[int]:
    """Returns the geoms of `bodies` that are spheres taking part in collisions."""
    return [
        geom
        for geom in range(model.ngeom)
        if model.geom_bodyid[geom] in bodies
        and model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_SPHERE
        and (model.geom_contype[geom] != 0 or model.geom_conaffinity[geom] != 0)
    ]


def _point_contact_violations(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body: int,
    pos_contact_rt_link_ewrt_link: np.ndarray,
) -> list[str]:
    """Returns why the foot of a point contact on `body` breaks the one-sphere rule (the module docstring), or nothing.

    Args:
        model: The compiled scene.
        data: Its kinematics, already computed (mj_kinematics).
        body: The body of the contact's parent joint.
        pos_contact_rt_link_ewrt_link: The contact frame in that body's frame [m], shape (3,).
    """
    spheres = _collision_spheres(model, _subtree(model, body))
    if len(spheres) != 1:
        return [
            f"{len(spheres)} collision spheres in the body and the bodies below"
            " it; a point contact's foot is exactly one"
        ]
    geom = spheres[0]
    rotmat_link_rt_world = data.xmat[body].reshape(3, 3)
    pos_sphere_rt_contact_ewrt_link = (
        rotmat_link_rt_world.T @ (data.geom_xpos[geom] - data.xpos[body])
        - pos_contact_rt_link_ewrt_link
    )
    # The sphere's lowest point along the link's z axis, relative to the contact frame.
    err_pos_sphere_rt_contact_ewrt_link = (
        pos_sphere_rt_contact_ewrt_link
        - np.array([0.0, 0.0, model.geom_size[geom][0]])
    )
    if np.abs(err_pos_sphere_rt_contact_ewrt_link).max() >= HEIGHT_TOLERANCE_M:
        return [
            "the sphere's lowest point is"
            f" {np.round(err_pos_sphere_rt_contact_ewrt_link * 1000, 2).tolist()} mm"
            " from the contact frame in the link's axes"
        ]
    return []


def _collision_geoms(model: mujoco.MjModel, bodies: set[int]) -> list[int]:
    """Returns the geoms of `bodies` that take part in collisions, of any type."""
    return [
        geom
        for geom in range(model.ngeom)
        if model.geom_bodyid[geom] in bodies
        and (model.geom_contype[geom] != 0 or model.geom_conaffinity[geom] != 0)
    ]


def _rolling_contact_violations(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint: int,
    pos_wheel_rt_link_ewrt_link: np.ndarray,
    wheel_radius: float,
) -> list[str]:
    """Returns why the wheel of a rolling contact on `joint` breaks the one-cylinder rule (the module docstring), or nothing.

    Args:
        model: The compiled scene.
        data: Its kinematics, already computed (mj_kinematics).
        joint: The wheel's own joint.
        pos_wheel_rt_link_ewrt_link: The wheel center in the joint's body frame [m], shape (3,).
        wheel_radius: [m] The task file's wheel radius.
    """
    body = model.jnt_bodyid[joint]
    geoms = _collision_geoms(model, _subtree(model, body))
    if len(geoms) != 1:
        return [
            f"{len(geoms)} collision geoms in the wheel's body and the bodies"
            " below it; a wheel is exactly one"
        ]
    geom = geoms[0]
    if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_CYLINDER:
        return ["the wheel's collision geom is not a cylinder"]
    violations = []
    rotmat_link_rt_world = data.xmat[body].reshape(3, 3)
    # The cylinder's axis is its frame's z axis; the hinge's axis is in its body's frame.
    axis_geom_ewrt_link = (
        rotmat_link_rt_world.T @ data.geom_xmat[geom].reshape(3, 3)[:, 2]
    )
    axis_joint_ewrt_link = model.jnt_axis[joint] / np.linalg.norm(
        model.jnt_axis[joint]
    )
    if 1.0 - abs(float(axis_geom_ewrt_link @ axis_joint_ewrt_link)) > 1e-9:
        violations.append(
            f"the cylinder's axis {np.round(axis_geom_ewrt_link, 4).tolist()} is"
            f" not the wheel joint's {np.round(axis_joint_ewrt_link, 4).tolist()}"
        )
    err_pos_geom_rt_wheel_ewrt_link = (
        rotmat_link_rt_world.T @ (data.geom_xpos[geom] - data.xpos[body])
        - pos_wheel_rt_link_ewrt_link
    )
    if np.abs(err_pos_geom_rt_wheel_ewrt_link).max() >= HEIGHT_TOLERANCE_M:
        violations.append(
            "the cylinder's center is"
            f" {np.round(err_pos_geom_rt_wheel_ewrt_link * 1000, 2).tolist()} mm"
            " from the wheel center in the link's axes"
        )
    if abs(model.geom_size[geom][0] - wheel_radius) >= HEIGHT_TOLERANCE_M:
        violations.append(
            f"the cylinder's radius is {model.geom_size[geom][0]:.4f} m, the"
            f" task file's wheel_radius {wheel_radius:.4f} m"
        )
    return violations


# A wheel on a hinge about y below a free body, for the self-tests of the rolling-contact rule: {geoms} is replaced by
# the wheel's geoms. The wheel center is the hinge's body origin.
_WHEEL_MJCF = """<mujoco><worldbody>
  <body name="shank"><freejoint/><inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
    <body name="wheel" pos="0 0 -0.3"><joint name="wheel" type="hinge" axis="0 1 0"/>
      <inertial pos="0 0 0" mass="0.2" diaginertia="0.001 0.002 0.001"/>
      {geoms}
    </body>
  </body></worldbody></mujoco>"""

# A body on a free joint with a foot below it, for the self-tests of the point-contact rule: {spheres} is replaced by
# the foot's geoms.
_POINT_FOOT_MJCF = """<mujoco><worldbody>
  <body name="shank"><freejoint/><inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
    <geom type="capsule" fromto="0 0 0 0 0 -0.2" size="0.01"/>
    {spheres}
  </body></worldbody></mujoco>"""


class ContactRectangleTest(unittest.TestCase):
    """The MPC's contacts of every robot against the MuJoCo feet: soles' rectangles, point feet's spheres, wheels."""

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

    def _point_violations(self, spheres: str) -> list[str]:
        """Returns the point-contact rule's findings on _POINT_FOOT_MJCF with `spheres`, the contact at (0, 0, -0.3)."""
        model = mujoco.MjModel.from_xml_string(
            _POINT_FOOT_MJCF.format(spheres=spheres)
        )
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        return _point_contact_violations(
            model, data, 1, np.array([0.0, 0.0, -0.3])
        )

    def test_the_point_rule_on_known_feet(self):
        self.assertEqual(
            self._point_violations(
                '<geom type="sphere" size="0.02" pos="0 0 -0.28"/>'
            ),
            [],
            "one sphere whose lowest point is the contact frame",
        )
        self.assertEqual(
            self._point_violations(
                '<body pos="0 0 -0.1"><geom type="sphere" size="0.02"'
                ' pos="0 0 -0.18"/></body>'
            ),
            [],
            "the sphere may be in a body below the contact's",
        )
        self.assertEqual(
            self._point_violations(
                '<geom type="sphere" size="0.02" pos="0 0 -0.28"/>'
                '<geom type="sphere" size="0.02" pos="0 0 -0.28"'
                ' contype="0" conaffinity="0"/>'
            ),
            [],
            "a visual sphere takes no part in collisions",
        )
        self.assertTrue(
            self._point_violations(
                '<geom type="sphere" size="0.02" pos="0 0 -0.28"/>'
                '<geom type="sphere" size="0.02" pos="0.05 0 -0.28"/>'
            ),
            "two spheres are two contact points",
        )
        self.assertTrue(
            self._point_violations(
                '<geom type="sphere" size="0.02" pos="0 0 -0.278"/>'
            ),
            "the lowest point 2 mm above the contact frame",
        )
        self.assertTrue(
            self._point_violations(
                '<geom type="sphere" size="0.02" pos="0.002 0 -0.28"/>'
            ),
            "the lowest point 2 mm beside the contact frame",
        )
        self.assertTrue(
            self._point_violations(""), "a point contact without a sphere"
        )

    def _rolling_violations(self, geoms: str) -> list[str]:
        """Returns the rolling-contact rule's findings on _WHEEL_MJCF with `geoms`, a 0.1 m wheel at the hinge."""
        model = mujoco.MjModel.from_xml_string(_WHEEL_MJCF.format(geoms=geoms))
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        return _rolling_contact_violations(
            model,
            data,
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "wheel"),
            np.zeros(3),
            wheel_radius=0.1,
        )

    def test_the_rolling_rule_on_known_wheels(self):
        self.assertEqual(
            self._rolling_violations(
                '<geom type="cylinder" size="0.1 0.02" zaxis="0 1 0"/>'
            ),
            [],
            "one cylinder on the hinge's axis, at the center, of the radius",
        )
        self.assertEqual(
            self._rolling_violations(
                '<geom type="cylinder" size="0.1 0.02" zaxis="0 -1 0"/>'
                '<geom type="sphere" size="0.05" contype="0" conaffinity="0"/>'
            ),
            [],
            "either direction of the axis, and a visual geom takes no part",
        )
        self.assertTrue(
            self._rolling_violations('<geom type="cylinder" size="0.1 0.02"/>'),
            "a cylinder about z, across the hinge",
        )
        self.assertTrue(
            self._rolling_violations(
                '<geom type="cylinder" size="0.098 0.02" zaxis="0 1 0"/>'
            ),
            "a radius 2 mm short of the task file's",
        )
        self.assertTrue(
            self._rolling_violations(
                '<geom type="cylinder" size="0.1 0.02" zaxis="0 1 0"'
                ' pos="0.002 0 0"/>'
            ),
            "a center 2 mm off the wheel center",
        )
        self.assertTrue(
            self._rolling_violations(
                '<geom type="cylinder" size="0.1 0.02" zaxis="0 1 0"/>'
                '<geom type="sphere" size="0.1"/>'
            ),
            "a second collision geom",
        )
        self.assertTrue(
            self._rolling_violations('<geom type="sphere" size="0.1"/>'),
            "a sphere does not roll about one axis",
        )

    def _assert_sole(
        self,
        label: str,
        model: mujoco.MjModel,
        body: int,
        pos_contact_rt_link_ewrt_link: np.ndarray,
        pos_corners_rt_contact_ewrt_contact_xy: np.ndarray,
    ) -> None:
        """Asserts that the sole spheres of a 6-DoF contact on `body` are level and hold the contact rectangle."""
        spheres = _collision_spheres(model, {body})
        self.assertGreaterEqual(
            len(spheres),
            3,
            f"{label}: a sole of contact spheres",
        )
        pos_spheres_rt_contact_ewrt_link = np.array(
            [
                model.geom_pos[geom] - pos_contact_rt_link_ewrt_link
                for geom in spheres
            ]
        )
        for geom, pos_sphere_rt_contact_ewrt_link in zip(
            spheres, pos_spheres_rt_contact_ewrt_link
        ):
            # The sphere's lowest point: its center's z minus its radius.
            min_pos_sphere_rt_contact_ewrt_link_z = (
                pos_sphere_rt_contact_ewrt_link[2] - model.geom_size[geom][0]
            )
            self.assertLess(
                abs(min_pos_sphere_rt_contact_ewrt_link_z),
                HEIGHT_TOLERANCE_M,
                f"{label}: a sole sphere's bottom is"
                f" {min_pos_sphere_rt_contact_ewrt_link_z * 1000:.2f} mm off the contact frame's"
                " plane",
            )
        hull = _convex_hull(pos_spheres_rt_contact_ewrt_link[:, :2])
        for (
            pos_corner_rt_contact_ewrt_contact_xy
        ) in pos_corners_rt_contact_ewrt_contact_xy:
            self.assertTrue(
                _inside(pos_corner_rt_contact_ewrt_contact_xy, hull),
                f"{label}: the contact rectangle's corner"
                f" {pos_corner_rt_contact_ewrt_contact_xy.tolist()} is outside the sole spheres' hull"
                f" {np.round(hull, 4).tolist()}",
            )

    def test_every_contact_lies_on_its_mujoco_foot(self):
        for robot in _ROBOTS:
            joints, _, task = _load(robot)
            _, mjcf, _, _ = _paths(robot)
            model = mujoco.MjModel.from_xml_path(mjcf)
            data = mujoco.MjData(model)
            mujoco.mj_kinematics(model, data)
            rectangle = task.contacts.contact_rectangle
            pos_corners_rt_contact_ewrt_contact_xy = np.array(
                [
                    [
                        rectangle.min_pos_corners_rt_contact_ewrt_contact_x,
                        rectangle.min_pos_corners_rt_contact_ewrt_contact_y,
                    ],
                    [
                        rectangle.max_pos_corners_rt_contact_ewrt_contact_x,
                        rectangle.min_pos_corners_rt_contact_ewrt_contact_y,
                    ],
                    [
                        rectangle.max_pos_corners_rt_contact_ewrt_contact_x,
                        rectangle.max_pos_corners_rt_contact_ewrt_contact_y,
                    ],
                    [
                        rectangle.min_pos_corners_rt_contact_ewrt_contact_x,
                        rectangle.max_pos_corners_rt_contact_ewrt_contact_y,
                    ],
                ]
            )
            # The MJCF free body sits where the URDF's root link does at qpos0.
            root_body = model.jnt_bodyid[0]
            pose_base_rt_world = (
                data.xpos[root_body].copy(),
                data.xmat[root_body].reshape(3, 3).copy(),
            )
            for contact in _contacts(task):
                parent = contact.parent_joint
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
                pos_link_rt_world_ewrt_world, rotmat_link_rt_world = (
                    _compute_pose_link_rt_world(
                        joints, parent, pose_base_rt_world, {}
                    )
                )
                self.assertLess(
                    np.linalg.norm(
                        data.xpos[body] - pos_link_rt_world_ewrt_world
                    ),
                    FRAME_TOLERANCE_M,
                    f"{robot} {parent}: frame position",
                )
                self.assertLess(
                    np.abs(
                        data.xmat[body].reshape(3, 3) - rotmat_link_rt_world
                    ).max(),
                    FRAME_TOLERANCE_M,
                    f"{robot} {parent}: frame rotation",
                )
                if contact.kind == _ContactKind.ROLLING:
                    self.assertEqual(
                        _rolling_contact_violations(
                            model,
                            data,
                            joint,
                            contact.pos_contact_rt_link_ewrt_link,
                            contact.wheel_radius,
                        ),
                        [],
                        f"{robot} {parent}: the rolling contact's wheel",
                    )
                    continue
                if contact.kind == _ContactKind.POINT:
                    self.assertEqual(
                        _point_contact_violations(
                            model,
                            data,
                            body,
                            contact.pos_contact_rt_link_ewrt_link,
                        ),
                        [],
                        f"{robot} {parent}: the point contact's foot",
                    )
                    continue
                self._assert_sole(
                    f"{robot} {parent}",
                    model,
                    body,
                    contact.pos_contact_rt_link_ewrt_link,
                    pos_corners_rt_contact_ewrt_contact_xy,
                )


if __name__ == "__main__":
    unittest.main()
