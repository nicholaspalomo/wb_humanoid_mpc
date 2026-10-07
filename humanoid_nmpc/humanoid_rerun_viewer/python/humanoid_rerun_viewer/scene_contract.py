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

"""The 3D scene's entity paths: the contract between the visualization publisher (viz/scene) and the bridge.

Every path a VisualizationScene names is relative to WORLD_ROOT: an Arrows message with path "markers/contact_forces"
is drawn at the entity world/markers/contact_forces. The robot model instances are drawn under ROBOTS_ROOT, one entity
per URDF link (world/robots/<instance>/<link>, whose Transform3D is the link's world pose) with the link's visuals as
its children (world/robots/<instance>/<link>/visual_<i>, static, at the visual origin of the URDF).

The markers listed in MARKERS are the ones the visualization publisher draws. Their defaults (radius and colors) apply
when a message leaves them unset: a radius of 0, or no color. They keep the look the ROS-era 3D view had: OCS2's
palette, 0.01 m arrow shafts with 0.02 m heads, 0.01 m wide lines.
"""

import dataclasses
import enum
import re

from humanoid_rerun_viewer import palette

# The roots of the recording: the 3D scene, the plots of viz/telemetry, and the state of the robot, the MPC and the
# bridge itself.
WORLD_ROOT = "world"
ROBOTS_ROOT = "world/robots"
TELEMETRY_ROOT = "telemetry"
STATUS_ROOT = "status"

# The timelines: the robot's clock (the time fields of the messages, seconds as a duration), and the bridge's wall clock
# when the message arrived (a timestamp). Rerun's own log_time is switched off, since wall_time replaces it everywhere.
ROBOT_TIMELINE = "robot_time"
WALL_TIMELINE = "wall_time"

# A relative entity path of the contract: names of letters, digits, '_', '-' and '.', joined by '/'.
_PATH = re.compile(r"^[A-Za-z0-9_.\-]+(/[A-Za-z0-9_.\-]+)*$")


def is_valid_relative_path(path: str) -> bool:
    """True for a path such as "markers/contact_forces": not empty, no leading or trailing '/', no '.' or '..'."""
    if not _PATH.match(path):
        return False
    return all(part not in (".", "..") for part in path.split("/"))


def entity_path(root: str, relative_path: str) -> str:
    """The entity path of `relative_path` under `root`, e.g. ("world", "plan/com") -> "world/plan/com"."""
    return f"{root}/{relative_path}"


def instance_path(instance: str) -> str:
    """world/robots/<instance>: the root of one copy of the robot model."""
    return f"{ROBOTS_ROOT}/{instance}"


def link_path(instance: str, link: str) -> str:
    """world/robots/<instance>/<link>: carries the link's world pose."""
    return f"{ROBOTS_ROOT}/{instance}/{link}"


def visual_path(instance: str, link: str, index: int) -> str:
    """world/robots/<instance>/<link>/visual_<index>: the index-th <visual> of the link, in URDF order."""
    return f"{ROBOTS_ROOT}/{instance}/{link}/visual_{index}"


@dataclasses.dataclass(frozen=True)
class RobotInstanceStyle:
    """How one copy of the robot model is drawn.

    Attributes:
        name: the RobotModelInstance.name of viz/scene.
        tint: the color every visual of the instance takes; None: the URDF's own material colors.
        alpha: the opacity of the instance.
        visible_by_default: False for an instance the blueprint hides until the user shows it.
    """

    name: str
    tint: palette.Rgb | None
    alpha: float
    visible_by_default: bool


# The instances the bridge loads the URDF's meshes for at start; an instance of another name gets DEFAULT_INSTANCE_STYLE
# when it first appears.
# LINT.IfChange(robot_instances)
MEASURED = "measured"
TERMINAL_STATE = "terminal_state"
TERMINAL_TARGET = "terminal_target"
ROBOT_INSTANCES: tuple[RobotInstanceStyle, ...] = (
    RobotInstanceStyle(
        name=MEASURED,
        tint=None,
        alpha=0.4,
        visible_by_default=True,
    ),
    RobotInstanceStyle(
        name=TERMINAL_STATE,
        tint=palette.BLUE,
        alpha=0.4,
        visible_by_default=True,
    ),
    RobotInstanceStyle(
        name=TERMINAL_TARGET,
        tint=palette.GREEN,
        alpha=0.3,
        visible_by_default=False,
    ),
)
# LINT.ThenChange(//humanoid_nmpc/humanoid_rerun_viewer/README.md:robot_instances, //humanoid_nmpc/docs/distributed_runtime/README.md:rerun_mapping, //humanoid_nmpc/humanoid_mpc_msgs/robot_model_instance.proto, //humanoid_nmpc/humanoid_common_mpc_app/visualization/include/humanoid_common_mpc_app/visualization/SceneContract.h:robot_instances)

DEFAULT_INSTANCE_STYLE = RobotInstanceStyle(
    name="",
    tint=palette.BLACK,
    alpha=0.4,
    visible_by_default=True,
)


def instance_style(name: str) -> RobotInstanceStyle:
    """The style of the instance `name`: its entry of ROBOT_INSTANCES, or DEFAULT_INSTANCE_STYLE under that name."""
    for style in ROBOT_INSTANCES:
        if style.name == name:
            return style
    return dataclasses.replace(DEFAULT_INSTANCE_STYLE, name=name)


class MarkerKind(str, enum.Enum):
    """The VisualizationScene field a marker arrives in, and the Rerun archetype it becomes."""

    ARROWS = "arrows"  # humanoid_mpc_msgs.Arrows -> rerun.Arrows3D
    SPHERES = "spheres"  # humanoid_mpc_msgs.Spheres -> rerun.Points3D with radii
    LINE_STRIPS = "line_strips"  # humanoid_mpc_msgs.LineStrips -> rerun.LineStrips3D


@dataclasses.dataclass(frozen=True)
class MarkerSpec:
    """One marker of the scene.

    Attributes:
        path: relative to WORLD_ROOT, as the message's `path` field names it.
        kind: the field of VisualizationScene that carries it.
        default_radius: the radius [m] when the message's is 0 or missing. For arrows it is Rerun's arrow radius: the
            shaft is drawn with half of it, the head with all of it, so 0.01 is a 0.01 m shaft and a 0.02 m head.
        default_colors: the colors when the message has none; one for all, or one per element (cycled).
        visible_by_default: False for a marker the blueprint hides until the user shows it.
    """

    path: str
    kind: MarkerKind
    default_radius: float
    default_colors: tuple[palette.Rgba, ...]
    visible_by_default: bool


def _opaque(*colors: palette.Rgb) -> tuple[palette.Rgba, ...]:
    return tuple(palette.with_alpha(color, 1.0) for color in colors)


# Arrow shaft diameter, as Rerun's arrow radius.
ARROW_RADIUS = 0.01
# Half of the 0.01 m width of the plan's trajectory lines.
TRAJECTORY_RADIUS = 0.005
# Half of the 0.03 m diameter of the center-of-pressure and foothold points.
POINT_MARKER_RADIUS = 0.015

# LINT.IfChange(markers)
MARKERS: tuple[MarkerSpec, ...] = (
    MarkerSpec(
        path="markers/contact_forces",
        kind=MarkerKind.ARROWS,
        default_radius=ARROW_RADIUS,
        default_colors=_opaque(palette.GREEN),
        visible_by_default=True,
    ),
    MarkerSpec(
        path="markers/center_of_pressure",
        kind=MarkerKind.SPHERES,
        default_radius=POINT_MARKER_RADIUS,
        default_colors=_opaque(palette.GREEN),
        visible_by_default=True,
    ),
    MarkerSpec(
        path="markers/corner_forces",
        kind=MarkerKind.ARROWS,
        default_radius=ARROW_RADIUS,
        default_colors=_opaque(palette.BLUE),
        visible_by_default=False,
    ),
    MarkerSpec(
        path="plan/end_effectors",
        kind=MarkerKind.LINE_STRIPS,
        default_radius=TRAJECTORY_RADIUS,
        default_colors=_opaque(*palette.CONTACT_COLORS),
        visible_by_default=True,
    ),
    MarkerSpec(
        path="plan/base",
        kind=MarkerKind.LINE_STRIPS,
        default_radius=TRAJECTORY_RADIUS,
        default_colors=_opaque(palette.RED),
        visible_by_default=True,
    ),
    MarkerSpec(
        path="plan/com",
        kind=MarkerKind.LINE_STRIPS,
        default_radius=TRAJECTORY_RADIUS,
        default_colors=_opaque(palette.YELLOW),
        visible_by_default=True,
    ),
    MarkerSpec(
        path="plan/footholds",
        kind=MarkerKind.SPHERES,
        default_radius=POINT_MARKER_RADIUS,
        default_colors=_opaque(*palette.CONTACT_COLORS),
        visible_by_default=True,
    ),
    MarkerSpec(
        path="markers/collision_spheres",
        kind=MarkerKind.SPHERES,
        default_radius=POINT_MARKER_RADIUS,
        default_colors=(palette.with_alpha(palette.RED, 0.5),),
        visible_by_default=False,
    ),
)
# LINT.ThenChange(//humanoid_nmpc/humanoid_rerun_viewer/README.md:markers, //humanoid_nmpc/docs/distributed_runtime/README.md:rerun_mapping, //humanoid_nmpc/humanoid_common_mpc_app/visualization/include/humanoid_common_mpc_app/visualization/SceneContract.h:markers)

MARKERS_BY_PATH: dict[str, MarkerSpec] = {marker.path: marker for marker in MARKERS}


def hidden_by_default() -> tuple[str, ...]:
    """The entity paths the blueprint hides in the 3D view until the user shows them: clutter most of the time."""
    instances = tuple(
        instance_path(style.name)
        for style in ROBOT_INSTANCES
        if not style.visible_by_default
    )
    markers = tuple(
        entity_path(WORLD_ROOT, marker.path)
        for marker in MARKERS
        if not marker.visible_by_default
    )
    return instances + markers
