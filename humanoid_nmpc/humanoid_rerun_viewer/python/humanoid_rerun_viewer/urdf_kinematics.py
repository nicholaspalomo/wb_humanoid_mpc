"""The kinematic tree of a URDF and its forward kinematics: the world pose of every link at given joint positions.

    tree = load_kinematic_tree("robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf")
    poses = tree.link_poses(tree.nominal_positions())     # link name -> (translation, quaternion x y z w)

The model sandbox (model_sandbox.py) draws a URDF with it where no MPC computes the link poses. It reads what the
poses need: the joints' type, parent and child link, origin, axis and limits. Mimic joints follow the joint they mimic.
The root link (the link that is no joint's child) is at the world origin; a floating or planar joint is held at its
origin, since it has no single position.
"""

import dataclasses
import math
import xml.etree.ElementTree as element_tree
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from humanoid_rerun_viewer import urdf_model

Translation = Tuple[float, float, float]
QuaternionXyzw = Tuple[float, float, float, float]
LinkPose = Tuple[Translation, QuaternionXyzw]

# The joint types whose position is one number, and those that move nothing.
REVOLUTE_TYPES = ("revolute", "continuous")
PRISMATIC_TYPES = ("prismatic",)
MOVABLE_TYPES = REVOLUTE_TYPES + PRISMATIC_TYPES
FIXED_TYPES = ("fixed", "floating", "planar")
_KNOWN_TYPES = MOVABLE_TYPES + FIXED_TYPES


@dataclasses.dataclass(frozen=True)
class Mimic:
    """The joint's position is `multiplier * position(joint) + offset`."""

    joint: str
    multiplier: float = 1.0
    offset: float = 0.0


@dataclasses.dataclass(frozen=True)
class Joint:
    """One <joint> of a URDF.

    Attributes:
        name: the joint's name.
        type: revolute, continuous, prismatic, fixed, floating or planar.
        parent, child: the links it connects.
        origin_translation, origin_rotation: the child's joint frame in the parent link's frame at position 0.
        axis: the unit axis, in the joint frame.
        lower, upper: the position limits; None for a continuous joint or one without limits.
        mimic: the joint it follows, if any.
    """

    name: str
    type: str
    parent: str
    child: str
    origin_translation: np.ndarray
    origin_rotation: np.ndarray
    axis: np.ndarray
    lower: Optional[float] = None
    upper: Optional[float] = None
    mimic: Optional[Mimic] = None

    @property
    def movable(self) -> bool:
        return self.type in MOVABLE_TYPES

    def clamp(self, position: float) -> float:
        """`position` within the joint's limits."""
        if self.lower is not None:
            position = max(position, self.lower)
        if self.upper is not None:
            position = min(position, self.upper)
        return position


def rotation_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF's fixed-axis roll, pitch, yaw as a rotation matrix: Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def rotation_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' formula: the rotation by `angle` about the unit vector `axis`."""
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def quaternion_from_rotation(rotation: np.ndarray) -> QuaternionXyzw:
    """The unit quaternion (x, y, z, w) of a rotation matrix, with w >= 0."""
    m = rotation
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = 2.0 * math.sqrt(trace + 1.0)
        w, x = 0.25 * s, (m[2, 1] - m[1, 2]) / s
        y, z = (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w, x = (m[2, 1] - m[1, 2]) / s, 0.25 * s
        y, z = (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w, x = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s
        y, z = 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w, x = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s
        y, z = (m[1, 2] + m[2, 1]) / s, 0.25 * s
    quaternion = np.array([x, y, z, w])
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    return tuple(float(value) for value in quaternion)  # type: ignore[return-value]


class KinematicTree:
    """The links and joints of a URDF, and their forward kinematics.

    Args:
        name: the robot's name.
        links: every link, in document order.
        joints: every joint, in document order.
        source: the file, for error messages.

    Raises:
        urdf_model.UrdfError: a joint names an unknown link, a link has two parents, the links are not one tree, or a
            mimic joint follows a joint that is not movable.
    """

    def __init__(
        self, name: str, links: Sequence[str], joints: Sequence[Joint], source: str
    ) -> None:
        self.name = name
        self.links: Tuple[str, ...] = tuple(links)
        self.joints: Tuple[Joint, ...] = tuple(joints)
        known = set(self.links)
        parent_joint: Dict[str, Joint] = {}
        for joint in self.joints:
            for link in (joint.parent, joint.child):
                if link not in known:
                    raise urdf_model.UrdfError(
                        f"{source}: joint '{joint.name}' names the unknown link '{link}'"
                    )
            if joint.child in parent_joint:
                raise urdf_model.UrdfError(
                    f"{source}: link '{joint.child}' is the child of joints '{parent_joint[joint.child].name}' and "
                    f"'{joint.name}'"
                )
            parent_joint[joint.child] = joint
        roots = [link for link in self.links if link not in parent_joint]
        if len(roots) != 1:
            raise urdf_model.UrdfError(
                f"{source}: the links must form one tree; the links without a parent are {roots}"
            )
        self.root_link = roots[0]
        children: Dict[str, List[Joint]] = {link: [] for link in self.links}
        for joint in self.joints:
            children[joint.parent].append(joint)
        # Parents before children, so that link_poses() is one pass.
        self._order: List[Joint] = []
        stack = [self.root_link]
        while stack:
            link = stack.pop()
            for joint in reversed(children[link]):
                self._order.append(joint)
                stack.append(joint.child)
        if len(self._order) != len(self.joints):
            raise urdf_model.UrdfError(f"{source}: the joints contain a cycle")
        by_name = {joint.name: joint for joint in self.joints}
        for joint in self.joints:
            if joint.mimic is not None:
                followed = by_name.get(joint.mimic.joint)
                if followed is None or not followed.movable:
                    raise urdf_model.UrdfError(
                        f"{source}: joint '{joint.name}' mimics '{joint.mimic.joint}', which is no movable joint"
                    )

    def movable_joints(self) -> Tuple[Joint, ...]:
        """The joints an operator sets: movable and not mimicking another, in document order."""
        return tuple(
            joint for joint in self.joints if joint.movable and joint.mimic is None
        )

    def nominal_positions(self) -> Dict[str, float]:
        """Every settable joint at 0, clamped to its limits (as joint_state_publisher_gui starts)."""
        return {joint.name: joint.clamp(0.0) for joint in self.movable_joints()}

    def link_poses(self, positions: Mapping[str, float]) -> Dict[str, LinkPose]:
        """The world pose of every link, the root at the origin. A settable joint missing from `positions` is at its
        nominal position; a position outside the limits is clamped."""
        nominal = self.nominal_positions()
        by_name = {joint.name: joint for joint in self.joints}

        def position_of(joint: Joint) -> float:
            if joint.mimic is not None:
                followed = by_name[joint.mimic.joint]
                return (
                    joint.mimic.multiplier * position_of(followed) + joint.mimic.offset
                )
            return joint.clamp(float(positions.get(joint.name, nominal[joint.name])))

        rotations: Dict[str, np.ndarray] = {self.root_link: np.eye(3)}
        translations: Dict[str, np.ndarray] = {self.root_link: np.zeros(3)}
        for joint in self._order:
            rotation = rotations[joint.parent] @ joint.origin_rotation
            translation = (
                translations[joint.parent]
                + rotations[joint.parent] @ joint.origin_translation
            )
            if joint.type in REVOLUTE_TYPES:
                rotation = rotation @ rotation_about_axis(
                    joint.axis, position_of(joint)
                )
            elif joint.type in PRISMATIC_TYPES:
                translation = translation + rotation @ (joint.axis * position_of(joint))
            rotations[joint.child] = rotation
            translations[joint.child] = translation
        return {
            link: (
                tuple(float(value) for value in translations[link]),  # type: ignore[misc]
                quaternion_from_rotation(rotations[link]),
            )
            for link in self.links
        }


def _floats(
    element: Optional[element_tree.Element],
    attribute: str,
    default: Sequence[float],
    where: str,
    source: str,
) -> np.ndarray:
    if element is None or element.get(attribute) is None:
        return np.array(default, dtype=float)
    text = element.get(attribute, "")
    try:
        values = np.array([float(value) for value in text.split()], dtype=float)
    except ValueError:
        values = np.array([])
    if values.shape != (len(default),) or not np.all(np.isfinite(values)):
        raise urdf_model.UrdfError(
            f'{source}: {where}: {attribute}="{text}" is not {len(default)} numbers'
        )
    return values


def _optional_float(
    element: Optional[element_tree.Element], attribute: str, where: str, source: str
) -> Optional[float]:
    if element is None or element.get(attribute) is None:
        return None
    try:
        value = float(element.get(attribute, ""))
    except ValueError:
        value = math.nan
    if not math.isfinite(value):
        raise urdf_model.UrdfError(
            f'{source}: {where}: {attribute}="{element.get(attribute)}" is not a number'
        )
    return value


def parse_kinematic_tree(text: str, source: str) -> KinematicTree:
    """The kinematic tree of a URDF's text; `source` names it in errors."""
    try:
        robot = element_tree.fromstring(text)
    except element_tree.ParseError as error:
        raise urdf_model.UrdfError(f"{source}: {error}") from error
    if robot.tag != "robot":
        raise urdf_model.UrdfError(
            f"{source}: the root element is <{robot.tag}>, not <robot>"
        )
    links = [link.get("name", "") for link in robot.findall("link")]
    joints = []
    for element in robot.findall("joint"):
        name = element.get("name", "")
        where = f"joint '{name}'"
        joint_type = element.get("type", "")
        if joint_type not in _KNOWN_TYPES:
            raise urdf_model.UrdfError(
                f"{source}: {where}: unknown type '{joint_type}'; the types are {', '.join(_KNOWN_TYPES)}"
            )
        parent = element.find("parent")
        child = element.find("child")
        if parent is None or child is None:
            raise urdf_model.UrdfError(
                f"{source}: {where}: needs a <parent> and a <child>"
            )
        origin = element.find("origin")
        translation = _floats(origin, "xyz", (0.0, 0.0, 0.0), where, source)
        rpy = _floats(origin, "rpy", (0.0, 0.0, 0.0), where, source)
        axis = _floats(element.find("axis"), "xyz", (1.0, 0.0, 0.0), where, source)
        norm = float(np.linalg.norm(axis))
        if joint_type in MOVABLE_TYPES and norm == 0.0:
            raise urdf_model.UrdfError(f"{source}: {where}: the axis is zero")
        limit = element.find("limit")
        lower = upper = None
        if joint_type in ("revolute", "prismatic"):
            lower = _optional_float(limit, "lower", where, source)
            upper = _optional_float(limit, "upper", where, source)
        mimic_element = element.find("mimic")
        mimic = None
        if mimic_element is not None:
            multiplier = _optional_float(mimic_element, "multiplier", where, source)
            offset = _optional_float(mimic_element, "offset", where, source)
            mimic = Mimic(
                joint=mimic_element.get("joint", ""),
                multiplier=1.0 if multiplier is None else multiplier,
                offset=0.0 if offset is None else offset,
            )
        joints.append(
            Joint(
                name=name,
                type=joint_type,
                parent=parent.get("link", ""),
                child=child.get("link", ""),
                origin_translation=translation,
                origin_rotation=rotation_from_rpy(*rpy),
                axis=axis / norm if norm > 0.0 else axis,
                lower=lower,
                upper=upper,
                mimic=mimic,
            )
        )
    return KinematicTree(robot.get("name", ""), links, joints, source)


def load_kinematic_tree(path: str) -> KinematicTree:
    """The kinematic tree of the URDF at `path`."""
    try:
        with open(path, "r", encoding="utf-8") as stream:
            text = stream.read()
    except OSError as error:
        raise urdf_model.UrdfError(
            f"cannot read the URDF {path}: {error.strerror}"
        ) from error
    return parse_kinematic_tree(text, path)
