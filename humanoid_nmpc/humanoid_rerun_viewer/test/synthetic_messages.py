"""Synthetic viz/scene and viz/telemetry messages that follow the contracts, and a small URDF package, for the tests.

full_telemetry() is also a reference for a producer: it fills every group of telemetry_contract for a robot.
"""

import math
import os
import struct
from typing import Iterable, List, Sequence, Tuple

from humanoid_mpc_msgs import arrows_pb2
from humanoid_mpc_msgs import line_strips_pb2
from humanoid_mpc_msgs import robot_model_instance_pb2
from humanoid_mpc_msgs import scalar_group_pb2
from humanoid_mpc_msgs import spheres_pb2
from humanoid_mpc_msgs import telemetry_series_pb2
from humanoid_mpc_msgs import visualization_scene_pb2

from humanoid_rerun_viewer import scene_contract
from humanoid_rerun_viewer import telemetry_contract

JOINTS = ("left_hip_pitch_joint", "left_knee_joint", "right_hip_pitch_joint")
DOFS = ("base_x", "base_y", "base_z", "base_yaw", "base_pitch", "base_roll") + JOINTS
FRAMES = ("foot_l_contact", "foot_r_contact", "pelvis")
STATE_DIM = 12
INPUT_DIM = 9


def robot_dependent_names(
    group: telemetry_contract.SeriesGroup,
    joints: Sequence[str] = JOINTS,
    dofs: Sequence[str] = DOFS,
    state_dim: int = STATE_DIM,
    input_dim: int = INPUT_DIM,
) -> Tuple[str, ...]:
    """The names a producer gives a robot-dependent group, by its names rule."""
    if group.names_rule == telemetry_contract.JOINT_NAMES_RULE:
        return tuple(joints)
    if group.names_rule == telemetry_contract.DOF_NAMES_RULE:
        return tuple(dofs)
    if group.names_rule == telemetry_contract.STATE_NAMES_RULE:
        return tuple(f"x{index}" for index in range(state_dim))
    if group.names_rule == telemetry_contract.INPUT_NAMES_RULE:
        return tuple(f"u{index}" for index in range(input_dim))
    raise ValueError(f"no names for the rule of '{group.path}'")


def full_telemetry(
    time: float,
    joints: Sequence[str] = JOINTS,
    frames: Sequence[str] = FRAMES,
) -> telemetry_series_pb2.TelemetrySeries:
    """A TelemetrySeries with every group of the contract; the values are deterministic functions of time."""
    series = telemetry_series_pb2.TelemetrySeries(time=time)
    for group_index, group in enumerate(telemetry_contract.all_groups(frames)):
        names = group.names or robot_dependent_names(group, joints)
        series.groups.append(
            scalar_group_pb2.ScalarGroup(
                path=group.path,
                names=names,
                values=[
                    math.sin(time + 0.1 * group_index + 0.01 * index)
                    for index in range(len(names))
                ],
            )
        )
    return series


def _pose(
    robot: robot_model_instance_pb2.RobotModelInstance, x: float, yaw: float
) -> None:
    pose = robot.link_poses.add()
    pose.position.x, pose.position.y, pose.position.z = x, 0.1, 0.8
    pose.orientation.w = math.cos(yaw / 2.0)
    pose.orientation.z = math.sin(yaw / 2.0)


def robot_instance(
    name: str, links: Iterable[str], time: float
) -> robot_model_instance_pb2.RobotModelInstance:
    robot = robot_model_instance_pb2.RobotModelInstance(name=name)
    for index, link in enumerate(links):
        robot.link_names.append(link)
        _pose(robot, x=0.1 * time + 0.01 * index, yaw=0.05 * index)
    return robot


def full_scene(
    time: float, links: Sequence[str], arrows_count: int = 2
) -> visualization_scene_pb2.VisualizationScene:
    """A scene with the three robot instances and every marker of the contract."""
    scene = visualization_scene_pb2.VisualizationScene(time=time)
    for style in scene_contract.ROBOT_INSTANCES:
        scene.robots.append(robot_instance(style.name, links, time))
    for marker in scene_contract.MARKERS:
        if marker.kind is scene_contract.MarkerKind.ARROWS:
            arrows = arrows_pb2.Arrows(path=marker.path)
            for index in range(arrows_count):
                origin = arrows.origins.add()
                origin.x, origin.y, origin.z = 0.1 * index, 0.0, 0.0
                vector = arrows.vectors.add()
                vector.z = 1.0 + index
            scene.arrows.append(arrows)
        elif marker.kind is scene_contract.MarkerKind.SPHERES:
            spheres = spheres_pb2.Spheres(path=marker.path)
            for index in range(3):
                center = spheres.centers.add()
                center.x, center.y = 0.2 * index, 0.1
            scene.spheres.append(spheres)
        else:
            line_strips = line_strips_pb2.LineStrips(path=marker.path)
            for strip_index in range(2):
                strip = line_strips.strips.add()
                for index in range(4):
                    point = strip.points.add()
                    point.x, point.y, point.z = 0.1 * index, 0.1 * strip_index, 0.0
            scene.line_strips.append(line_strips)
    return scene


# ======================================================================================================================
# A URDF package on disk
# ======================================================================================================================

SAMPLE_PACKAGE = "sample_robot_description"
SAMPLE_LINKS = ("base_link", "thigh", "shin", "foot", "sensor_mount")


def write_binary_stl(path: str) -> None:
    """A tetrahedron as a binary STL."""
    vertices = [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.1)]
    faces = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
    with open(path, "wb") as stl:
        stl.write(b"test tetrahedron".ljust(80, b" "))
        stl.write(struct.pack("<I", len(faces)))
        for face in faces:
            stl.write(struct.pack("<3f", 0.0, 0.0, 0.0))
            for vertex in face:
                stl.write(struct.pack("<3f", *vertices[vertex]))
            stl.write(struct.pack("<H", 0))


def sample_urdf_text(package: str = SAMPLE_PACKAGE) -> str:
    """A five-link URDF: an upper-case .STL mesh with a scale and an origin, a named and an inline material, a box, a
    cylinder and a sphere, a .dae with an .stl twin, and a link without visuals."""
    return f"""<?xml version="1.0"?>
<robot name="sample_robot">
  <material name="steel"><color rgba="0.5 0.5 0.6 1"/></material>
  <link name="base_link">
    <visual>
      <origin xyz="0 0 0.1" rpy="0 0 1.5707963267948966"/>
      <geometry><mesh filename="package://{package}/meshes/base.STL" scale="0.001 0.001 0.001"/></geometry>
      <material name="steel"/>
    </visual>
  </link>
  <link name="thigh">
    <visual>
      <geometry><box size="0.1 0.2 0.3"/></geometry>
      <material name="red"><color rgba="1 0 0 1"/></material>
    </visual>
    <visual>
      <geometry><cylinder radius="0.05" length="0.4"/></geometry>
    </visual>
  </link>
  <link name="shin">
    <visual><geometry><sphere radius="0.03"/></geometry></visual>
  </link>
  <link name="foot">
    <visual><geometry><mesh filename="package://{package}/meshes/foot.dae"/></geometry></visual>
  </link>
  <link name="sensor_mount"/>
  <joint name="hip" type="revolute"><parent link="base_link"/><child link="thigh"/></joint>
  <joint name="knee" type="revolute"><parent link="thigh"/><child link="shin"/></joint>
  <joint name="ankle" type="revolute"><parent link="shin"/><child link="foot"/></joint>
  <joint name="mount" type="fixed"><parent link="base_link"/><child link="sensor_mount"/></joint>
</robot>
"""


def write_sample_package(directory: str, package: str = SAMPLE_PACKAGE) -> str:
    """Writes <directory>/<package>/{urdf/test.urdf, meshes/base.STL, meshes/foot.dae, meshes/foot.stl}; returns the
    URDF's path."""
    root = os.path.join(directory, package)
    os.makedirs(os.path.join(root, "urdf"), exist_ok=True)
    os.makedirs(os.path.join(root, "meshes"), exist_ok=True)
    write_binary_stl(os.path.join(root, "meshes", "base.STL"))
    write_binary_stl(os.path.join(root, "meshes", "foot.stl"))
    with open(os.path.join(root, "meshes", "foot.dae"), "w", encoding="utf-8") as dae:
        dae.write("<COLLADA/>\n")
    urdf_path = os.path.join(root, "urdf", "test.urdf")
    with open(urdf_path, "w", encoding="utf-8") as urdf:
        urdf.write(sample_urdf_text(package))
    return urdf_path


def link_names_with_visuals() -> List[str]:
    return ["base_link", "thigh", "shin", "foot"]
