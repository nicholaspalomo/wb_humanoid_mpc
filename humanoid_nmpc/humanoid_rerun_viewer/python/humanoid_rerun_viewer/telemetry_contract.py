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

"""The plots' paths and series names: the contract between the producer of viz/telemetry and the bridge's blueprint.

Every humanoid_mpc_msgs.ScalarGroup of a TelemetrySeries becomes the entity telemetry/<path> with one series per name.
This module lists every group the visualization publisher sends and the plot tabs the blueprint builds from them, and
each group's description says what its values are:

- PANEL_GROUPS: one small group per panel, whose names are the panel's curves ("measured" and "reference", or "mpc"
  and "measured"). The blueprint plots each panel from its group alone. The values also appear in the complete groups;
  the duplication costs a few doubles per message and keeps every panel independent of the robot's joint names.
- The complete groups: every joint (JOINT_GROUPS), every generalized coordinate, velocity and force (DOF_GROUPS), both
  contact wrenches (CONTACT_WRENCH_GROUPS), every tracked frame (frame_groups(), one set per frame of the task file's
  telemetry_frames), the MPC observation (MPC_OBSERVATION_GROUPS), the optimized plan's included, and the latest MPC
  solution's performance index and cost and constraint terms (MPC_SOLUTION_GROUPS). The orientation quaternions are not
  plotted: the Euler angles next to them (and the 3D scene) show them.
- The balance panels plot all three sources: the whole-body CoM, the DCM / ICP xi = c + c_dot / omega with
  omega = sqrt(g / z0) from the model, the ZMP of the contact wrenches and the ACoM attitude.

The MPC's state holds the base orientation as a quaternion and its velocity as the body angular velocity w_B. The plots
show the ZYX Euler angles (roll, pitch, yaw) of the quaternion, their rates T_B(Theta)^-1 w_B, which are display values
only and NaN, a gap in the plot, within about 1e-6 rad of a pitch of +-90 degrees, and w_B itself: the "Base Pose &
Euler" tab has a row of each rate below the Euler angles.

All values are SI (m, rad, s, N, N m) in the world frame unless a group says otherwise. One message is one robot/state
sample, and every source is taken at the sample's time: "measured" is the sample, "reference" the MPC's target
trajectory (not the plan), "plan" the latest MPC policy's state and input, "mpc" the plan's contact wrenches, "target"
the joint action the robot applied. The producer is the C++ visualization publisher
(humanoid_nmpc/humanoid_common_mpc_app/visualization). The README of this package tabulates the contract.
"""

from collections.abc import Iterable
import dataclasses
import re

from humanoid_rerun_viewer import palette

# The names of the panel groups' series.
MEASURED = "measured"
REFERENCE = "reference"
MPC = "mpc"
PLAN = "plan"
MEASURED_VS_REFERENCE = (MEASURED, REFERENCE)
MEASURED_REFERENCE_PLAN = (MEASURED, REFERENCE, PLAN)

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

# Contact 0 and 1 of ModelSettings::contactNames (kContactLeftIndex, kContactRightIndex).
LEFT = "left"
RIGHT = "right"

# How the producer names the series of a group whose names depend on the robot.
JOINT_NAMES_RULE = "the robot's joints, ModelSettings::fullJointNames in order"
DOF_NAMES_RULE = (
    "base_x, base_y, base_z, base_yaw, base_pitch, base_roll"
    " (getBaseDofNames(): the base's Euler angles in place of its quaternion),"
    " then ModelSettings::mpcModelJointNames in order"
)
DOF_VELOCITY_NAMES_RULE = (
    "base_vx, base_vy, base_vz, base_wx, base_wy, base_wz"
    " (getBaseVelocityDofNames(): the world linear and the body angular"
    " velocity), then ModelSettings::mpcModelJointNames in order"
)
COST_TERM_NAMES_RULE = (
    "<collection>/<term> per cost term of the solver's problem, in the order it"
    " was added: intermediate/, state/, soft/, state_soft/ (at the solution's"
    " first node), final/, final_soft/ (at its last node)"
)
CONSTRAINT_TERM_NAMES_RULE = (
    "<collection>/<term> per constraint term of the solver's problem:"
    " equality/, state_equality/, inequality/, state_inequality/ (at the"
    " solution's first node), final_equality/, final_inequality/ (at its last"
    " node)"
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
        description: what the values are.
        colors: "#rrggbb" per name (palette.py's curve colors, for the panel groups); empty: Rerun picks.
        names_rule: how the producer names the series when `names` is empty.
    """

    path: str
    names: tuple[str, ...]
    unit: str
    description: str
    colors: tuple[str, ...] = ()
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
    paths: tuple[str, ...]
    curves: tuple[Curve, ...] = ()


@dataclasses.dataclass(frozen=True)
class Tab:
    """One tab of plots: rows top to bottom, panels left to right."""

    title: str
    rows: tuple[tuple[Panel, ...], ...]

    def panels(self) -> tuple[Panel, ...]:
        panels: list[Panel] = []
        for row in self.rows:
            panels.extend(row)
        return tuple(panels)


# ======================================================================================================================
# The groups
# ======================================================================================================================


def _pair(
    path: str,
    unit: str,
    description: str,
    colors: tuple[str, str],
    names: tuple[str, str] = MEASURED_VS_REFERENCE,
) -> SeriesGroup:
    return SeriesGroup(
        path=path,
        names=names,
        unit=unit,
        description=description,
        colors=colors,
    )


_POSITION_COLORS = (palette.PLOT_BLUE, palette.PLOT_RED)
_ANGLE_COLORS = (palette.PLOT_GREEN, palette.PLOT_ORANGE)
_SOURCE_COLORS = (palette.PLOT_BLUE, palette.PLOT_RED, palette.PLOT_PURPLE)


def _triple(path: str, unit: str, description: str) -> SeriesGroup:
    """A panel group of the three sources: measured, reference and plan."""
    return SeriesGroup(
        path=path,
        names=MEASURED_REFERENCE_PLAN,
        unit=unit,
        description=description,
        colors=_SOURCE_COLORS,
    )


_COM_POSITION = (
    "the whole-body center of mass (Pinocchio's, of the source's q), world"
    " frame, at its height. measured: of the sample; reference: of the target"
    " trajectory; plan: of the MPC policy"
)
_DCM = (
    "the divergent component of motion (instantaneous capture point) xi = c +"
    " c_dot / omega, omega = sqrt(g / z0), g the Pinocchio model's gravity, z0"
    " the nominal state's CoM height above its feet"
    " (computeComHeightAboveFeet); NaN without a nominal state. Sources as com/"
)
_ZMP = (
    "the zero moment point of the source's contact wrenches at its contact"
    " frames, on the plane of their normal-force-weighted height; NaN below 1 N"
    " of net normal force. measured: the force sensors' forces, which carry no"
    " torque, so the normal-force-weighted mean of the contact frames;"
    " reference: of the target trajectory's input; plan: of the MPC policy's"
    " input"
)
_ACOM = (
    "the attitude of the angular center of mass"
    " (AngularCenterOfMass::acomAttitude() of the base quaternion and the"
    " joints) as ZYX Euler angles, for display; NaN for a robot without an ACoM"
    " network. Sources as com/"
)

_BASE_POSITION = (
    "measured: root position; reference: base position of the target trajectory"
)
_BASE_EULER = (
    "measured: root orientation as roll, pitch, yaw; reference: the target"
    " trajectory's"
)
_BASE_LINEAR_VELOCITY = (
    "measured: root linear velocity; reference: base CoM velocity of the target"
    " trajectory"
)
_BASE_EULER_RATE = (
    "the ZYX Euler angle rates of the body angular velocity, T_B(Theta)^-1 w_B,"
    " for display; NaN within about 1e-6 rad of a pitch of +-90 degrees."
    " measured: of the sample; reference: of the target trajectory, 0 where it"
    " carries no velocity"
)
_BASE_BODY_ANGULAR_VELOCITY = (
    "the body angular velocity w_B, in the base frame. measured: the sample's"
    " angvel_base_rt_world_ewrt_base; reference: the target trajectory's,"
    " Pinocchio's v[3:6], 0 where it carries no velocity"
)
_BASE_ANGULAR_VELOCITY = (
    "the root's angular velocity in the world frame, R(xi) w_B. measured: of"
    " the sample; reference: of the target trajectory, 0 where it carries no"
    " velocity"
)
_NORMAL_FORCE = (
    "mpc: force z of the plan's contact wrench, the MPC policy's at the"
    " sample's time; measured: the force sensor's"
)
_TANGENTIAL_FORCE = (
    "mpc_x, mpc_y: the policy's; measured_x, measured_y: the force sensor's"
)
_GENERALIZED_COORDINATE = (
    "generalized coordinate, for an angle the base's Euler angle in place of"
    " its quaternion; reference: of the target trajectory"
)
_GENERALIZED_VELOCITY = (
    "generalized velocity, for an angle the Euler angle rate, as"
    " base_pose/<angle>_rate; reference: of the target trajectory"
)
_FOOT_ACCELERATION = (
    "linear z of the classical acceleration of the contact frame,"
    " LOCAL_WORLD_ALIGNED; reference: of the target trajectory"
)
_FOOT_VELOCITY = (
    "linear z of the contact frame's velocity, LOCAL_WORLD_ALIGNED; reference:"
    " of the target trajectory"
)

# LINT.IfChange(panel_groups)
PANEL_GROUPS: tuple[SeriesGroup, ...] = (
    # Base Pose & Euler
    _pair("base_pose/position_x", "m", _BASE_POSITION, _POSITION_COLORS),
    _pair("base_pose/position_y", "m", _BASE_POSITION, _POSITION_COLORS),
    _pair("base_pose/position_z", "m", _BASE_POSITION, _POSITION_COLORS),
    _pair("base_pose/roll", "rad", _BASE_EULER, _ANGLE_COLORS),
    _pair("base_pose/pitch", "rad", _BASE_EULER, _ANGLE_COLORS),
    _pair("base_pose/yaw", "rad", _BASE_EULER, _ANGLE_COLORS),
    _pair("base_pose/roll_rate", "rad/s", _BASE_EULER_RATE, _ANGLE_COLORS),
    _pair("base_pose/pitch_rate", "rad/s", _BASE_EULER_RATE, _ANGLE_COLORS),
    _pair("base_pose/yaw_rate", "rad/s", _BASE_EULER_RATE, _ANGLE_COLORS),
    _pair(
        "base_pose/angular_velocity_x",
        "rad/s",
        _BASE_BODY_ANGULAR_VELOCITY,
        _ANGLE_COLORS,
    ),
    _pair(
        "base_pose/angular_velocity_y",
        "rad/s",
        _BASE_BODY_ANGULAR_VELOCITY,
        _ANGLE_COLORS,
    ),
    _pair(
        "base_pose/angular_velocity_z",
        "rad/s",
        _BASE_BODY_ANGULAR_VELOCITY,
        _ANGLE_COLORS,
    ),
    # Base Twist
    _pair(
        "base_twist/linear_x", "m/s", _BASE_LINEAR_VELOCITY, _POSITION_COLORS
    ),
    _pair(
        "base_twist/linear_y", "m/s", _BASE_LINEAR_VELOCITY, _POSITION_COLORS
    ),
    _pair(
        "base_twist/linear_z", "m/s", _BASE_LINEAR_VELOCITY, _POSITION_COLORS
    ),
    _pair(
        "base_twist/angular_x", "rad/s", _BASE_ANGULAR_VELOCITY, _ANGLE_COLORS
    ),
    _pair(
        "base_twist/angular_y", "rad/s", _BASE_ANGULAR_VELOCITY, _ANGLE_COLORS
    ),
    _pair(
        "base_twist/angular_z", "rad/s", _BASE_ANGULAR_VELOCITY, _ANGLE_COLORS
    ),
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
        "generalized_base/position_z",
        "m",
        _GENERALIZED_COORDINATE,
        _POSITION_COLORS,
    ),
    _pair(
        "generalized_base/pitch", "rad", _GENERALIZED_COORDINATE, _ANGLE_COLORS
    ),
    _pair(
        "generalized_base/roll", "rad", _GENERALIZED_COORDINATE, _ANGLE_COLORS
    ),
    _pair(
        "generalized_base/velocity_z",
        "m/s",
        _GENERALIZED_VELOCITY,
        _POSITION_COLORS,
    ),
    _pair(
        "generalized_base/pitch_rate",
        "rad/s",
        _GENERALIZED_VELOCITY,
        _ANGLE_COLORS,
    ),
    _pair(
        "generalized_base/roll_rate",
        "rad/s",
        _GENERALIZED_VELOCITY,
        _ANGLE_COLORS,
    ),
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
    _pair(
        "foot_kinematics/left_velocity_z", "m/s", _FOOT_VELOCITY, _ANGLE_COLORS
    ),
    _pair(
        "foot_kinematics/right_velocity_z", "m/s", _FOOT_VELOCITY, _ANGLE_COLORS
    ),
    # Balance (CoM, DCM, ZMP, ACoM)
    _triple("com/position_x", "m", _COM_POSITION),
    _triple("com/position_y", "m", _COM_POSITION),
    _triple("com/position_z", "m", _COM_POSITION),
    _triple("dcm/x", "m", _DCM),
    _triple("dcm/y", "m", _DCM),
    _triple("zmp/x", "m", _ZMP),
    _triple("zmp/y", "m", _ZMP),
    _triple("acom/roll", "rad", _ACOM),
    _triple("acom/pitch", "rad", _ACOM),
    _triple("acom/yaw", "rad", _ACOM),
)
# LINT.ThenChange(//humanoid_nmpc/humanoid_rerun_viewer/README.md:panel_groups, //humanoid_nmpc/humanoid_common_mpc_app/visualization/src/TelemetryBuilder.cpp:telemetry_groups)


def _robot_dependent(
    path: str, unit: str, description: str, names_rule: str
) -> SeriesGroup:
    return SeriesGroup(
        path=path,
        names=(),
        unit=unit,
        description=description,
        names_rule=names_rule,
    )


# LINT.IfChange(complete_groups)
JOINT_GROUPS: tuple[SeriesGroup, ...] = (
    _robot_dependent(
        "joints/position/measured",
        "rad",
        "measured joint position",
        JOINT_NAMES_RULE,
    ),
    _robot_dependent(
        "joints/position/target",
        "rad",
        "position target of the joint action applied, the measured position"
        " without one",
        JOINT_NAMES_RULE,
    ),
    _robot_dependent(
        "joints/velocity/measured",
        "rad/s",
        "measured joint velocity",
        JOINT_NAMES_RULE,
    ),
    _robot_dependent(
        "joints/velocity/target",
        "rad/s",
        "velocity target of the joint action applied, 0 without one",
        JOINT_NAMES_RULE,
    ),
    _robot_dependent(
        "joints/effort/measured",
        "N m",
        "effort applied: feed-forward + kp (q_des - q) + kd (qd_des - qd), 0"
        " without an action",
        JOINT_NAMES_RULE,
    ),
    _robot_dependent(
        "joints/effort/target",
        "N m",
        "feed-forward effort of the joint action applied",
        JOINT_NAMES_RULE,
    ),
)

DOF_GROUPS: tuple[SeriesGroup, ...] = (
    _robot_dependent(
        "dofs/position/measured",
        "m, rad",
        "generalized coordinates for display: base position, the Euler ZYX"
        " angles of the base quaternion, MPC joints",
        DOF_NAMES_RULE,
    ),
    _robot_dependent(
        "dofs/position/reference",
        "m, rad",
        "generalized coordinates of the target trajectory",
        DOF_NAMES_RULE,
    ),
    _robot_dependent(
        "dofs/position/plan",
        "m, rad",
        "generalized coordinates of the MPC policy's state",
        DOF_NAMES_RULE,
    ),
    _robot_dependent(
        "dofs/velocity/measured",
        "m/s, rad/s",
        "generalized velocities: base world linear velocity, base body angular"
        " velocity, joints",
        DOF_VELOCITY_NAMES_RULE,
    ),
    _robot_dependent(
        "dofs/velocity/reference",
        "m/s, rad/s",
        "generalized velocities of the target trajectory",
        DOF_VELOCITY_NAMES_RULE,
    ),
    _robot_dependent(
        "dofs/velocity/plan",
        "m/s, rad/s",
        "generalized velocities of the MPC policy's state and input",
        DOF_VELOCITY_NAMES_RULE,
    ),
    _robot_dependent(
        "dofs/force/measured",
        "N, N m",
        "generalized forces, dual to the velocities: 0 for the base, the effort"
        " applied for a joint",
        DOF_VELOCITY_NAMES_RULE,
    ),
    _robot_dependent(
        "dofs/force/reference",
        "N, N m",
        "generalized forces: 0 for the base, the feed-forward effort for a"
        " joint",
        DOF_VELOCITY_NAMES_RULE,
    ),
)


def _contact_wrench_groups(side: str) -> tuple[SeriesGroup, SeriesGroup]:
    """The groups of one foot's contact wrench: the plan's, and the force sensor's."""
    return (
        SeriesGroup(
            path=f"contact_wrenches/{side}/{MPC}",
            names=WRENCH_NAMES,
            unit="N, N m",
            description=(
                "the plan's contact wrench, the MPC policy's at the sample's"
                " time"
            ),
        ),
        SeriesGroup(
            path=f"contact_wrenches/{side}/{MEASURED}",
            names=FORCE_NAMES,
            unit="N",
            description="the force sensor's force",
        ),
    )


CONTACT_WRENCH_GROUPS: tuple[SeriesGroup, ...] = _contact_wrench_groups(
    LEFT
) + _contact_wrench_groups(RIGHT)

MPC_OBSERVATION_GROUPS: tuple[SeriesGroup, ...] = (
    _robot_dependent(
        "mpc_observation/state",
        "",
        "the observation's MPC state",
        STATE_NAMES_RULE,
    ),
    _robot_dependent(
        "mpc_observation/input",
        "",
        "the MPC policy's input at the observation's time",
        INPUT_NAMES_RULE,
    ),
    SeriesGroup(
        path="mpc_observation/mode",
        names=("mode",),
        unit="",
        description="the observation's mode number",
    ),
)

MPC_SOLUTION_GROUPS: tuple[SeriesGroup, ...] = (
    SeriesGroup(
        path="mpc_solution/performance",
        names=(
            "merit",
            "cost",
            "dynamics_violation_sse",
            "equality_constraints_sse",
            "inequality_constraints_sse",
            "equality_lagrangian",
            "inequality_lagrangian",
        ),
        unit="",
        description=(
            "the solver's performance index of the latest solution, over its"
            " horizon (OCS2's PerformanceIndex); NaN before the first"
        ),
    ),
    _robot_dependent(
        "mpc_solution/costs",
        "",
        "per cost term of the latest solution, evaluated on the solver thread"
        " with the solver's own problem: an intermediate term's L(t0, x0, u0)"
        " at the first node (a rate, per second; 0 while inactive), a final"
        " term's Phi(tf, xf); NaN before the first solution and for a term the"
        " problem no longer has",
        COST_TERM_NAMES_RULE,
    ),
    _robot_dependent(
        "mpc_solution/constraints",
        "",
        "per constraint term of the latest solution, at the nodes of"
        " mpc_solution/costs: the 2-norm of its violation, |g| for an equality"
        " g = 0, |min(g, 0)| for an inequality g >= 0; 0 while inactive",
        CONSTRAINT_TERM_NAMES_RULE,
    ),
)

# The groups of one tracked frame: frames/<kind>/<frame>/<source>, kind first so that a view can plot one kind of
# every frame. <frame> is a frame of the task file's telemetry_frames (the contact frames when it lists none).
FRAME_KINDS: dict[str, tuple[tuple[str, ...], str, str]] = {
    "pose": (
        POSE_NAMES,
        "m, rad",
        "world position and roll, pitch, yaw",
    ),
    "twist": (
        TWIST_NAMES,
        "m/s, rad/s",
        "frame velocity, LOCAL_WORLD_ALIGNED",
    ),
    "acceleration": (
        TWIST_NAMES,
        "m/s^2, rad/s^2",
        "classical acceleration, LOCAL_WORLD_ALIGNED",
    ),
    "wrench": (
        WRENCH_NAMES,
        "N, N m",
        (
            "measured: the force sensor's wrench for a contact frame;"
            " reference: the contact wrench of the target trajectory's input;"
            " plan: the MPC policy's; 0 for a frame that is not a contact"
        ),
    ),
}
FRAME_SOURCES = (MEASURED, REFERENCE, PLAN)
# LINT.ThenChange(//humanoid_nmpc/humanoid_rerun_viewer/README.md:complete_groups, //humanoid_nmpc/humanoid_common_mpc_app/visualization/src/TelemetryBuilder.cpp:telemetry_groups)


def frame_groups(frame: str) -> tuple[SeriesGroup, ...]:
    """The groups of the tracked frame `frame`: measured, of the target trajectory (reference) and of the plan."""
    groups: list[SeriesGroup] = []
    for kind, (names, unit, description) in FRAME_KINDS.items():
        groups.extend(
            SeriesGroup(
                path=f"frames/{kind}/{frame}/{source}",
                names=names,
                unit=unit,
                description=description,
            )
            for source in FRAME_SOURCES
        )
    return tuple(groups)


_FRAME_PATH = re.compile(
    r"^frames/(?P<kind>[a-z]+)/(?P<frame>[A-Za-z0-9_.\-]+)/(?P<source>[a-z]+)$"
)

# Every group whose path does not depend on the robot.
STATIC_GROUPS: tuple[SeriesGroup, ...] = (
    PANEL_GROUPS
    + JOINT_GROUPS
    + DOF_GROUPS
    + CONTACT_WRENCH_GROUPS
    + MPC_OBSERVATION_GROUPS
    + MPC_SOLUTION_GROUPS
)
_STATIC_GROUPS_BY_PATH: dict[str, SeriesGroup] = {
    group.path: group for group in STATIC_GROUPS
}


def find_group(path: str) -> SeriesGroup | None:
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


def all_groups(frames: Iterable[str]) -> tuple[SeriesGroup, ...]:
    """Every group a producer that tracks `frames` sends."""
    groups = STATIC_GROUPS
    for frame in frames:
        groups += frame_groups(frame)
    return groups


# ======================================================================================================================
# The tabs
# ======================================================================================================================


def _pair_panel(
    title: str,
    path: str,
    names: tuple[str, str] = MEASURED_VS_REFERENCE,
) -> Panel:
    """A panel of one group of two series, "measured" against "reference" unless `names` says otherwise."""
    return Panel(
        title=title,
        paths=(path,),
        curves=(Curve(path, names[0]), Curve(path, names[1])),
    )


def _triple_panel(title: str, path: str) -> Panel:
    """A panel of one group of the three sources: measured, reference and plan."""
    return Panel(
        title=title,
        paths=(path,),
        curves=tuple(Curve(path, name) for name in MEASURED_REFERENCE_PLAN),
    )


def _contact_normal_panel(title: str, side: str) -> Panel:
    return _pair_panel(
        title, f"contact_forces/{side}_normal", names=(MPC, MEASURED)
    )


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


# The seven tabs of the panel groups, one small plot per panel. The joint panels plot every joint of the robot, and the
# curves name the first of them. "Base Pose & Euler" shows the base's Euler angles, their rates and its body angular
# velocity, so that a plot configuration that selects it alone shows all three.
# LINT.IfChange(panel_tabs)
PANEL_TABS: tuple[Tab, ...] = (
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
            (
                _pair_panel("Euler Roll Rate [rad/s]", "base_pose/roll_rate"),
                _pair_panel("Euler Pitch Rate [rad/s]", "base_pose/pitch_rate"),
                _pair_panel("Euler Yaw Rate [rad/s]", "base_pose/yaw_rate"),
            ),
            (
                _pair_panel(
                    "Body Angular Vel X [rad/s]", "base_pose/angular_velocity_x"
                ),
                _pair_panel(
                    "Body Angular Vel Y [rad/s]", "base_pose/angular_velocity_y"
                ),
                _pair_panel(
                    "Body Angular Vel Z [rad/s]", "base_pose/angular_velocity_z"
                ),
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
                    "Angular Vel X (world) [rad/s]", "base_twist/angular_x"
                ),
                _pair_panel(
                    "Angular Vel Y (world) [rad/s]", "base_twist/angular_y"
                ),
                _pair_panel(
                    "Angular Vel Z (world) [rad/s]", "base_twist/angular_z"
                ),
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
                _contact_tangential_panel(
                    "Left Foot Tangential Force Fx/Fy [N]", LEFT
                ),
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
                    paths=(
                        "joints/position/measured",
                        "joints/position/target",
                    ),
                    curves=(
                        Curve("joints/position/measured", "[0]"),
                        Curve("joints/position/target", "[0]"),
                    ),
                ),
            ),
            (
                Panel(
                    title="Joint Velocities",
                    paths=(
                        "joints/velocity/measured",
                        "joints/velocity/target",
                    ),
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
                _pair_panel(
                    "Base Pitch Rate [rad/s]", "generalized_base/pitch_rate"
                ),
                _pair_panel(
                    "Base Roll Rate [rad/s]", "generalized_base/roll_rate"
                ),
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
                    "Right Foot Twist Z [m/s]",
                    "foot_kinematics/right_velocity_z",
                ),
            ),
        ),
    ),
    Tab(
        title="Balance (CoM, DCM, ZMP, ACoM)",
        rows=(
            (
                _triple_panel("CoM X [m]", "com/position_x"),
                _triple_panel("CoM Y [m]", "com/position_y"),
                _triple_panel("CoM Height Z [m]", "com/position_z"),
            ),
            (
                _triple_panel("DCM X [m]", "dcm/x"),
                _triple_panel("DCM Y [m]", "dcm/y"),
                _triple_panel("ZMP X [m]", "zmp/x"),
                _triple_panel("ZMP Y [m]", "zmp/y"),
            ),
            (
                _triple_panel("ACoM Roll [rad]", "acom/roll"),
                _triple_panel("ACoM Pitch [rad]", "acom/pitch"),
                _triple_panel("ACoM Yaw [rad]", "acom/yaw"),
            ),
        ),
    ),
)
# LINT.ThenChange(//humanoid_nmpc/humanoid_rerun_viewer/README.md:panel_groups)

# The tabs of the complete groups.
COMPLETE_TABS: tuple[Tab, ...] = (
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
        title="MPC Solution",
        rows=(
            (Panel("Performance Index", ("mpc_solution/performance",)),),
            (Panel("Cost Terms", ("mpc_solution/costs",)),),
            (Panel("Constraint Violations", ("mpc_solution/constraints",)),),
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

TABS: tuple[Tab, ...] = PANEL_TABS + COMPLETE_TABS
