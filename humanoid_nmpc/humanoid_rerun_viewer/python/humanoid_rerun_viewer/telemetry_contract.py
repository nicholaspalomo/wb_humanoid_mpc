"""The plots' paths and series names: the contract between the producer of viz/telemetry and the bridge's blueprint.

Every humanoid_mpc_msgs.ScalarGroup of a TelemetrySeries becomes the entity telemetry/<path> with one series per name.
This module lists every group the visualization publisher sends and the plot tabs the blueprint builds from them. The
series replace the ROS 2 telemetry topics of the ROS-era telemetry publishers, and each group's description names the
topic it replaces; the panel tabs keep the layout of the ROS-era plots, panel for panel and in the same colors:

- PANEL_GROUPS: one small group per panel, whose names are the panel's curves ("measured" and "reference", or "mpc"
  and "measured"). The blueprint plots each panel from its group alone. The values also appear in the complete groups; the duplication costs a few doubles per message and
  keeps every panel independent of the robot's joint names.
- The complete groups: every joint (JOINT_GROUPS), every generalized coordinate, velocity and force (DOF_GROUPS), both
  contact wrenches (CONTACT_WRENCH_GROUPS), every tracked frame (frame_groups(), one set per frame of the task file's
  telemetryFrames) and the MPC observation (MPC_OBSERVATION_GROUPS). Together with PANEL_GROUPS they carry every series
  the ROS-era publishers published, except the orientation quaternions, which the Euler angles next to them (and the 3D
  scene) show, and add the optimized plan, which those publishers did not publish.

All values are SI (m, rad, s, N, N m) in the world frame unless a group says otherwise. One message is one robot/state
sample, and every source is taken at the sample's time: "measured" is the sample, "reference" the MPC's target
trajectory (the old "mpc/desired" topics were the reference, not the plan), "plan" the latest MPC policy's state and
input, "mpc" the plan's contact wrenches, "target" the joint action the robot applied. The producer is the C++
visualization publisher (humanoid_nmpc/humanoid_common_mpc_app/visualization). The README of this package tabulates the
contract.
"""

import dataclasses
import re
from typing import Dict, Iterable, Optional, Tuple

from humanoid_rerun_viewer import palette

# The names of the panel groups' series.
MEASURED = "measured"
REFERENCE = "reference"
TARGET = "target"
MPC = "mpc"
PLAN = "plan"
MEASURED_VS_REFERENCE = (MEASURED, REFERENCE)

# The names of the complete groups' series.
POSE_NAMES = ("x", "y", "z", "roll", "pitch", "yaw")
TWIST_NAMES = (
    "linear_x",
    "linear_y",
    "linear_z",
    "angular_x",
    "angular_y",
    "angular_z",
)
WRENCH_NAMES = (
    "force_x",
    "force_y",
    "force_z",
    "torque_x",
    "torque_y",
    "torque_z",
)
FORCE_NAMES = ("force_x", "force_y", "force_z")

# Contact 0 and 1 of ModelSettings::contactNames (CONTACT_LEFT_INDEX, CONTACT_RIGHT_INDEX).
LEFT = "left"
RIGHT = "right"

# How the producer names the series of a group whose names depend on the robot.
JOINT_NAMES_RULE = "the robot's joints, ModelSettings::fullJointNames in order (the order of the old joint_states topic)"
DOF_NAMES_RULE = (
    "base_x, base_y, base_z, base_yaw, base_pitch, base_roll (getBaseDofNames()), then "
    "ModelSettings::mpcModelJointNames in order"
)
STATE_NAMES_RULE = "x0, x1, ...: one per component of the MPC state"
INPUT_NAMES_RULE = "u0, u1, ...: one per component of the MPC input"

# A curve of a robot-dependent group names its series by position: "[0]" is the group's first name.
_INDEX_SERIES = re.compile(r"^\[(\d+)\]$")


@dataclasses.dataclass(frozen=True)
class SeriesGroup:
    """One ScalarGroup of viz/telemetry.

    Attributes:
        path: relative to the telemetry root; the entity is telemetry/<path>.
        names: the series, in the order of ScalarGroup.values; empty when they depend on the robot (names_rule).
        unit: of every value.
        description: what the values are, and the ROS 2 topic they replace.
        colors: "#rrggbb" per name (palette.py's curve colors, for the panel groups); empty: Rerun picks.
        names_rule: how the producer names the series when `names` is empty.
    """

    path: str
    names: Tuple[str, ...]
    unit: str
    description: str
    colors: Tuple[str, ...] = ()
    names_rule: str = ""

    @property
    def robot_dependent(self) -> bool:
        """True when the series names depend on the robot (and `names_rule` says how)."""
        return not self.names

    def has_series(self, series: str) -> bool:
        """True when `series` is one of the names, or "[i]" (the i-th name) for a robot-dependent group."""
        if self.robot_dependent:
            return _INDEX_SERIES.match(series) is not None
        return series in self.names

    def color_of(self, series: str) -> Optional[str]:
        """The color of `series`, or None when the group leaves the colors to Rerun."""
        if not self.colors or series not in self.names:
            return None
        return self.colors[self.names.index(series)]


@dataclasses.dataclass(frozen=True)
class Curve:
    """One curve of a panel: the series it plots.

    Attributes:
        path: the group that carries it.
        series: its name in the group; "[i]" for the i-th name of a robot-dependent group.
    """

    path: str
    series: str


@dataclasses.dataclass(frozen=True)
class Panel:
    """One plot of the blueprint.

    Attributes:
        title: the view's name.
        paths: what the view plots, relative to the telemetry root; an entry ending in "/**" plots a subtree.
        curves: the curves of a panel of PANEL_TABS, each a series of one of `paths`; empty for the panels of the
            complete groups, which plot whole groups.
    """

    title: str
    paths: Tuple[str, ...]
    curves: Tuple[Curve, ...] = ()


@dataclasses.dataclass(frozen=True)
class Tab:
    """One tab of plots: rows top to bottom, panels left to right."""

    title: str
    rows: Tuple[Tuple[Panel, ...], ...]

    def panels(self) -> Tuple[Panel, ...]:
        return tuple(panel for row in self.rows for panel in row)


# ======================================================================================================================
# The groups
# ======================================================================================================================


def _pair(
    path: str,
    unit: str,
    description: str,
    colors: Tuple[str, str],
    names: Tuple[str, str] = MEASURED_VS_REFERENCE,
) -> SeriesGroup:
    return SeriesGroup(
        path=path, names=names, unit=unit, description=description, colors=colors
    )


_POSITION_COLORS = (palette.PLOT_BLUE, palette.PLOT_RED)
_ANGLE_COLORS = (palette.PLOT_GREEN, palette.PLOT_ORANGE)

_BASE_POSITION = (
    "measured: root position (robot/base_pose); reference: base position of the target trajectory "
    "(mpc/target_base_pose)"
)
_BASE_EULER = (
    "measured: root orientation as roll, pitch, yaw (robot/base_euler); reference: the target trajectory's "
    "(mpc/target_base_euler)"
)
_BASE_LINEAR_VELOCITY = (
    "measured: root linear velocity (robot/base_twist); reference: base CoM velocity of the target trajectory "
    "(mpc/target_base_twist)"
)
_BASE_ANGULAR_VELOCITY = (
    "measured: root angular velocity (robot/base_twist); reference: 0, the target trajectory carries none "
    "(mpc/target_base_twist)"
)
_NORMAL_FORCE = (
    "mpc: force z of the plan's contact wrench, the MPC policy's at the sample's time (mpc/contact_wrench/<side>); "
    "measured: the force sensor's (sensors/contact_wrench/<side>)"
)
_TANGENTIAL_FORCE = (
    "mpc_x, mpc_y: the policy's; measured_x, measured_y: the force sensor's"
)
_GENERALIZED_COORDINATE = (
    "generalized coordinate (robot/generalized_coordinates/<dof>); reference: of the target trajectory "
    "(mpc/desired/generalized_coordinates/<dof>)"
)
_GENERALIZED_VELOCITY = (
    "generalized velocity, for an angle the Euler angle rate (robot/generalized_velocities/<dof>); reference: of the "
    "target trajectory (mpc/desired/generalized_velocities/<dof>)"
)
_FOOT_ACCELERATION = (
    "linear z of the classical acceleration of the contact frame, LOCAL_WORLD_ALIGNED (robot/frames/<frame>/accel); "
    "reference: of the target trajectory (mpc/desired/frames/<frame>/accel)"
)
_FOOT_VELOCITY = (
    "linear z of the contact frame's velocity, LOCAL_WORLD_ALIGNED (robot/frames/<frame>/twist); reference: of the "
    "target trajectory (mpc/desired/frames/<frame>/twist)"
)

# LINT.IfChange(panel_groups)
PANEL_GROUPS: Tuple[SeriesGroup, ...] = (
    # Base Pose & Euler
    _pair("base_pose/position_x", "m", _BASE_POSITION, _POSITION_COLORS),
    _pair("base_pose/position_y", "m", _BASE_POSITION, _POSITION_COLORS),
    _pair("base_pose/position_z", "m", _BASE_POSITION, _POSITION_COLORS),
    _pair("base_pose/roll", "rad", _BASE_EULER, _ANGLE_COLORS),
    _pair("base_pose/pitch", "rad", _BASE_EULER, _ANGLE_COLORS),
    _pair("base_pose/yaw", "rad", _BASE_EULER, _ANGLE_COLORS),
    # Base Twist
    _pair("base_twist/linear_x", "m/s", _BASE_LINEAR_VELOCITY, _POSITION_COLORS),
    _pair("base_twist/linear_y", "m/s", _BASE_LINEAR_VELOCITY, _POSITION_COLORS),
    _pair("base_twist/linear_z", "m/s", _BASE_LINEAR_VELOCITY, _POSITION_COLORS),
    _pair("base_twist/angular_x", "rad/s", _BASE_ANGULAR_VELOCITY, _ANGLE_COLORS),
    _pair("base_twist/angular_y", "rad/s", _BASE_ANGULAR_VELOCITY, _ANGLE_COLORS),
    _pair("base_twist/angular_z", "rad/s", _BASE_ANGULAR_VELOCITY, _ANGLE_COLORS),
    # Contact Forces
    _pair(
        "contact_forces/left_normal",
        "N",
        _NORMAL_FORCE,
        (palette.PLOT_RED, palette.PLOT_BLUE),
        names=(MPC, MEASURED),
    ),
    _pair(
        "contact_forces/right_normal",
        "N",
        _NORMAL_FORCE,
        (palette.PLOT_RED, palette.PLOT_BLUE),
        names=(MPC, MEASURED),
    ),
    SeriesGroup(
        path="contact_forces/left_tangential",
        names=("mpc_x", "measured_x", "mpc_y", "measured_y"),
        unit="N",
        description=_TANGENTIAL_FORCE,
        colors=(
            palette.PLOT_ORANGE,
            palette.PLOT_GREEN,
            palette.PLOT_PURPLE,
            palette.PLOT_BROWN,
        ),
    ),
    SeriesGroup(
        path="contact_forces/right_tangential",
        names=("mpc_x", "measured_x", "mpc_y", "measured_y"),
        unit="N",
        description=_TANGENTIAL_FORCE,
        colors=(
            palette.PLOT_ORANGE,
            palette.PLOT_GREEN,
            palette.PLOT_PURPLE,
            palette.PLOT_BROWN,
        ),
    ),
    # Generalized Coordinates (Pinocchio)
    _pair(
        "generalized_base/position_z", "m", _GENERALIZED_COORDINATE, _POSITION_COLORS
    ),
    _pair("generalized_base/pitch", "rad", _GENERALIZED_COORDINATE, _ANGLE_COLORS),
    _pair("generalized_base/roll", "rad", _GENERALIZED_COORDINATE, _ANGLE_COLORS),
    _pair(
        "generalized_base/velocity_z", "m/s", _GENERALIZED_VELOCITY, _POSITION_COLORS
    ),
    _pair("generalized_base/pitch_rate", "rad/s", _GENERALIZED_VELOCITY, _ANGLE_COLORS),
    _pair("generalized_base/roll_rate", "rad/s", _GENERALIZED_VELOCITY, _ANGLE_COLORS),
    # Frame Kinematics & Acceleration (contact 0 is the left foot, contact 1 the right)
    _pair(
        "foot_kinematics/left_acceleration_z",
        "m/s^2",
        _FOOT_ACCELERATION,
        _POSITION_COLORS,
    ),
    _pair(
        "foot_kinematics/right_acceleration_z",
        "m/s^2",
        _FOOT_ACCELERATION,
        _POSITION_COLORS,
    ),
    _pair("foot_kinematics/left_velocity_z", "m/s", _FOOT_VELOCITY, _ANGLE_COLORS),
    _pair("foot_kinematics/right_velocity_z", "m/s", _FOOT_VELOCITY, _ANGLE_COLORS),
)
# LINT.ThenChange(//humanoid_nmpc/humanoid_rerun_viewer/README.md:panel_groups, //humanoid_nmpc/humanoid_common_mpc_app/visualization/src/TelemetryBuilder.cpp:telemetry_groups)


def _robot_dependent(
    path: str, unit: str, description: str, names_rule: str
) -> SeriesGroup:
    return SeriesGroup(
        path=path, names=(), unit=unit, description=description, names_rule=names_rule
    )


# LINT.IfChange(complete_groups)
JOINT_GROUPS: Tuple[SeriesGroup, ...] = (
    _robot_dependent(
        "joints/position/measured",
        "rad",
        "measured joint position (joint_states.position)",
        JOINT_NAMES_RULE,
    ),
    _robot_dependent(
        "joints/position/target",
        "rad",
        "position target of the joint action applied, the measured position without one "
        "(mpc/joint_targets.position)",
        JOINT_NAMES_RULE,
    ),
    _robot_dependent(
        "joints/velocity/measured",
        "rad/s",
        "measured joint velocity (joint_states.velocity)",
        JOINT_NAMES_RULE,
    ),
    _robot_dependent(
        "joints/velocity/target",
        "rad/s",
        "velocity target of the joint action applied, 0 without one (mpc/joint_targets.velocity)",
        JOINT_NAMES_RULE,
    ),
    _robot_dependent(
        "joints/effort/measured",
        "N m",
        "effort applied: feed-forward + kp (q_des - q) + kd (qd_des - qd), 0 without an action "
        "(joint_states.effort)",
        JOINT_NAMES_RULE,
    ),
    _robot_dependent(
        "joints/effort/target",
        "N m",
        "feed-forward effort of the joint action applied (mpc/joint_targets.effort)",
        JOINT_NAMES_RULE,
    ),
)

DOF_GROUPS: Tuple[SeriesGroup, ...] = (
    _robot_dependent(
        "dofs/position/measured",
        "m, rad",
        "generalized coordinates: base position, Euler ZYX angles, MPC joints (robot/generalized_coordinates)",
        DOF_NAMES_RULE,
    ),
    _robot_dependent(
        "dofs/position/reference",
        "m, rad",
        "generalized coordinates of the target trajectory (mpc/desired/generalized_coordinates)",
        DOF_NAMES_RULE,
    ),
    _robot_dependent(
        "dofs/position/plan",
        "m, rad",
        "generalized coordinates of the MPC policy's state (new: the ROS-era publishers sent no plan)",
        DOF_NAMES_RULE,
    ),
    _robot_dependent(
        "dofs/velocity/measured",
        "m/s, rad/s",
        "generalized velocities: base linear velocity, Euler angle rates, joints (robot/generalized_velocities)",
        DOF_NAMES_RULE,
    ),
    _robot_dependent(
        "dofs/velocity/reference",
        "m/s, rad/s",
        "generalized velocities of the target trajectory (mpc/desired/generalized_velocities)",
        DOF_NAMES_RULE,
    ),
    _robot_dependent(
        "dofs/velocity/plan",
        "m/s, rad/s",
        "generalized velocities of the MPC policy's state and input (new)",
        DOF_NAMES_RULE,
    ),
    _robot_dependent(
        "dofs/force/measured",
        "N, N m",
        "generalized forces: 0 for the base, the effort applied for a joint (robot/generalized_forces)",
        DOF_NAMES_RULE,
    ),
    _robot_dependent(
        "dofs/force/reference",
        "N, N m",
        "generalized forces: 0 for the base, the feed-forward effort for a joint (mpc/desired/generalized_forces)",
        DOF_NAMES_RULE,
    ),
)

CONTACT_WRENCH_GROUPS: Tuple[SeriesGroup, ...] = tuple(
    group
    for side in (LEFT, RIGHT)
    for group in (
        SeriesGroup(
            path=f"contact_wrenches/{side}/{MPC}",
            names=WRENCH_NAMES,
            unit="N, N m",
            description=f"the plan's contact wrench, the MPC policy's at the sample's time (mpc/contact_wrench/{side})",
        ),
        SeriesGroup(
            path=f"contact_wrenches/{side}/{MEASURED}",
            names=FORCE_NAMES,
            unit="N",
            description=f"the force sensor's force (sensors/contact_wrench/{side})",
        ),
    )
)

MPC_OBSERVATION_GROUPS: Tuple[SeriesGroup, ...] = (
    _robot_dependent(
        "mpc_observation/state",
        "",
        "the observation's MPC state (mpc/observation.state)",
        STATE_NAMES_RULE,
    ),
    _robot_dependent(
        "mpc_observation/input",
        "",
        "the MPC policy's input at the observation's time (mpc/observation.input)",
        INPUT_NAMES_RULE,
    ),
    SeriesGroup(
        path="mpc_observation/mode",
        names=("mode",),
        unit="",
        description="the observation's mode number (mpc/observation.mode)",
    ),
)

# The groups of one tracked frame: frames/<kind>/<frame>/<source>, kind first so that a view can plot one kind of
# every frame. <frame> is a frame of the task file's telemetryFrames (the contact frames when it lists none).
FRAME_KINDS: Dict[str, Tuple[Tuple[str, ...], str, str]] = {
    "pose": (
        POSE_NAMES,
        "m, rad",
        "world position and roll, pitch, yaw (robot/frames/<frame>/pose and /euler)",
    ),
    "twist": (
        TWIST_NAMES,
        "m/s, rad/s",
        "frame velocity, LOCAL_WORLD_ALIGNED (robot/frames/<frame>/twist)",
    ),
    "acceleration": (
        TWIST_NAMES,
        "m/s^2, rad/s^2",
        "classical acceleration, LOCAL_WORLD_ALIGNED (robot/frames/<frame>/accel)",
    ),
    "wrench": (
        WRENCH_NAMES,
        "N, N m",
        "measured: the force sensor's wrench for a contact frame; reference: the contact wrench of the target "
        "trajectory's input; plan: the MPC policy's; 0 for a frame that is not a contact (robot/frames/<frame>/wrench)",
    ),
}
FRAME_SOURCES = (MEASURED, REFERENCE, PLAN)
# LINT.ThenChange(//humanoid_nmpc/humanoid_rerun_viewer/README.md:complete_groups, //humanoid_nmpc/humanoid_common_mpc_app/visualization/src/TelemetryBuilder.cpp:telemetry_groups)


def frame_groups(frame: str) -> Tuple[SeriesGroup, ...]:
    """The groups of the tracked frame `frame`: measured, of the target trajectory (reference) and of the plan."""
    return tuple(
        SeriesGroup(
            path=f"frames/{kind}/{frame}/{source}",
            names=names,
            unit=unit,
            description=description,
        )
        for kind, (names, unit, description) in FRAME_KINDS.items()
        for source in FRAME_SOURCES
    )


_FRAME_PATH = re.compile(
    r"^frames/(?P<kind>[a-z]+)/(?P<frame>[A-Za-z0-9_.\-]+)/(?P<source>[a-z]+)$"
)

# Every group whose path does not depend on the robot.
STATIC_GROUPS: Tuple[SeriesGroup, ...] = (
    PANEL_GROUPS
    + JOINT_GROUPS
    + DOF_GROUPS
    + CONTACT_WRENCH_GROUPS
    + MPC_OBSERVATION_GROUPS
)
_STATIC_GROUPS_BY_PATH: Dict[str, SeriesGroup] = {
    group.path: group for group in STATIC_GROUPS
}


def find_group(path: str) -> Optional[SeriesGroup]:
    """The group of the contract at `path` (a frame group for any frame name), or None."""
    group = _STATIC_GROUPS_BY_PATH.get(path)
    if group is not None:
        return group
    match = _FRAME_PATH.match(path)
    if match is None or match.group("kind") not in FRAME_KINDS:
        return None
    if match.group("source") not in FRAME_SOURCES:
        return None
    for frame_group in frame_groups(match.group("frame")):
        if frame_group.path == path:
            return frame_group
    return None


def all_groups(frames: Iterable[str]) -> Tuple[SeriesGroup, ...]:
    """Every group a producer that tracks `frames` sends."""
    return STATIC_GROUPS + tuple(
        group for frame in frames for group in frame_groups(frame)
    )


# ======================================================================================================================
# The tabs
# ======================================================================================================================


def _pair_panel(
    title: str,
    path: str,
    names: Tuple[str, str] = MEASURED_VS_REFERENCE,
) -> Panel:
    """A panel of one group of two series, "measured" against "reference" unless `names` says otherwise."""
    return Panel(
        title=title,
        paths=(path,),
        curves=(Curve(path, names[0]), Curve(path, names[1])),
    )


def _contact_normal_panel(title: str, side: str) -> Panel:
    return _pair_panel(title, f"contact_forces/{side}_normal", names=(MPC, MEASURED))


def _contact_tangential_panel(title: str, side: str) -> Panel:
    path = f"contact_forces/{side}_tangential"
    return Panel(
        title=title,
        paths=(path,),
        curves=tuple(
            Curve(path, series)
            for series in ("mpc_x", "measured_x", "mpc_y", "measured_y")
        ),
    )


# The six tabs of the panel groups, one small plot per panel. The joint panels plot every joint of the robot, and the
# curves name the first of them.
# LINT.IfChange(panel_tabs)
PANEL_TABS: Tuple[Tab, ...] = (
    Tab(
        title="Base Pose & Euler",
        rows=(
            (
                _pair_panel("Base Pos X [m]", "base_pose/position_x"),
                _pair_panel("Base Pos Y [m]", "base_pose/position_y"),
                _pair_panel("Base Pos Z (Height) [m]", "base_pose/position_z"),
            ),
            (
                _pair_panel("Base Roll [rad]", "base_pose/roll"),
                _pair_panel("Base Pitch [rad]", "base_pose/pitch"),
                _pair_panel("Base Yaw [rad]", "base_pose/yaw"),
            ),
        ),
    ),
    Tab(
        title="Base Twist (Linear & Angular)",
        rows=(
            (
                _pair_panel("Linear Vel X [m/s]", "base_twist/linear_x"),
                _pair_panel("Linear Vel Y [m/s]", "base_twist/linear_y"),
                _pair_panel("Linear Vel Z [m/s]", "base_twist/linear_z"),
            ),
            (
                _pair_panel(
                    "Angular Vel X (Roll Rate) [rad/s]", "base_twist/angular_x"
                ),
                _pair_panel(
                    "Angular Vel Y (Pitch Rate) [rad/s]", "base_twist/angular_y"
                ),
                _pair_panel("Angular Vel Z (Yaw Rate) [rad/s]", "base_twist/angular_z"),
            ),
        ),
    ),
    Tab(
        title="Contact Forces (MPC vs Measured)",
        rows=(
            (
                _contact_normal_panel("Left Foot Normal Force Fz [N]", LEFT),
                _contact_normal_panel("Right Foot Normal Force Fz [N]", RIGHT),
            ),
            (
                _contact_tangential_panel("Left Foot Tangential Force Fx/Fy [N]", LEFT),
                _contact_tangential_panel(
                    "Right Foot Tangential Force Fx/Fy [N]", RIGHT
                ),
            ),
        ),
    ),
    Tab(
        title="Joint Dynamics",
        rows=(
            (
                Panel(
                    title="Joint Positions & Targets",
                    paths=("joints/position/measured", "joints/position/target"),
                    curves=(
                        Curve("joints/position/measured", "[0]"),
                        Curve("joints/position/target", "[0]"),
                    ),
                ),
            ),
            (
                Panel(
                    title="Joint Velocities",
                    paths=("joints/velocity/measured", "joints/velocity/target"),
                    curves=(Curve("joints/velocity/measured", "[0]"),),
                ),
            ),
            (
                Panel(
                    title="Joint Torques / Effort [N*m]",
                    paths=("joints/effort/measured", "joints/effort/target"),
                    curves=(
                        Curve("joints/effort/measured", "[0]"),
                        Curve("joints/effort/target", "[0]"),
                    ),
                ),
            ),
        ),
    ),
    Tab(
        title="Generalized Coordinates (Pinocchio)",
        rows=(
            (
                _pair_panel("Base Z Height [m]", "generalized_base/position_z"),
                _pair_panel("Base Pitch [rad]", "generalized_base/pitch"),
                _pair_panel("Base Roll [rad]", "generalized_base/roll"),
            ),
            (
                _pair_panel("Base Vz [m/s]", "generalized_base/velocity_z"),
                _pair_panel("Base Pitch Rate [rad/s]", "generalized_base/pitch_rate"),
                _pair_panel("Base Roll Rate [rad/s]", "generalized_base/roll_rate"),
            ),
        ),
    ),
    Tab(
        title="Frame Kinematics & Acceleration",
        rows=(
            (
                _pair_panel(
                    "Left Foot Linear Accel Z [m/s^2]",
                    "foot_kinematics/left_acceleration_z",
                ),
                _pair_panel(
                    "Right Foot Linear Accel Z [m/s^2]",
                    "foot_kinematics/right_acceleration_z",
                ),
            ),
            (
                _pair_panel(
                    "Left Foot Twist Z [m/s]", "foot_kinematics/left_velocity_z"
                ),
                _pair_panel(
                    "Right Foot Twist Z [m/s]", "foot_kinematics/right_velocity_z"
                ),
            ),
        ),
    ),
)
# LINT.ThenChange(//humanoid_nmpc/humanoid_rerun_viewer/README.md:panel_groups)

# The tabs of the complete groups.
COMPLETE_TABS: Tuple[Tab, ...] = (
    Tab(
        title="Generalized (all DOFs)",
        rows=(
            (
                Panel(
                    "Generalized Coordinates",
                    (
                        "dofs/position/measured",
                        "dofs/position/reference",
                        "dofs/position/plan",
                    ),
                ),
            ),
            (
                Panel(
                    "Generalized Velocities",
                    (
                        "dofs/velocity/measured",
                        "dofs/velocity/reference",
                        "dofs/velocity/plan",
                    ),
                ),
            ),
            (
                Panel(
                    "Generalized Forces",
                    ("dofs/force/measured", "dofs/force/reference"),
                ),
            ),
        ),
    ),
    Tab(
        title="Frames",
        rows=(
            (
                Panel("Frame Poses", ("frames/pose/**",)),
                Panel("Frame Twists", ("frames/twist/**",)),
            ),
            (
                Panel("Frame Accelerations", ("frames/acceleration/**",)),
                Panel("Frame Wrenches", ("frames/wrench/**",)),
            ),
        ),
    ),
    Tab(
        title="Contact Wrenches",
        rows=(
            (
                Panel(
                    "Left Contact Wrench",
                    (
                        f"contact_wrenches/{LEFT}/{MPC}",
                        f"contact_wrenches/{LEFT}/{MEASURED}",
                    ),
                ),
                Panel(
                    "Right Contact Wrench",
                    (
                        f"contact_wrenches/{RIGHT}/{MPC}",
                        f"contact_wrenches/{RIGHT}/{MEASURED}",
                    ),
                ),
            ),
        ),
    ),
    Tab(
        title="MPC Observation",
        rows=(
            (Panel("MPC State", ("mpc_observation/state",)),),
            (
                Panel("MPC Input", ("mpc_observation/input",)),
                Panel("Mode", ("mpc_observation/mode",)),
            ),
        ),
    ),
)

TABS: Tuple[Tab, ...] = PANEL_TABS + COMPLETE_TABS
