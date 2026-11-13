#!/usr/bin/env python3
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

"""Derives a robot's `locomotion_heuristics` parameter block from its URDF, its task file and its gait schedule.

The coefficients of Bledt's regularization heuristics (legged_nmpc/docs/locomotion_heuristics/README.md) are not
free numbers: most of them follow from the robot's mass and geometry, from the cadence of the gait it is walking
with, and from the command limits the operator can reach. This script does that arithmetic, so that the block in a
robot's `config/mpc/task.textproto` can be regenerated rather than maintained by hand, and so that the derivation of
every number is visible next to the number. It prints the blocks as textproto, ready to paste into the task file's
`locomotion_heuristics { ... }`.

    python3 tools/locomotion_heuristics/derive_parameters.py --robot drc_atlas
    python3 tools/locomotion_heuristics/derive_parameters.py --robot drc_atlas --gait walk
    python3 tools/locomotion_heuristics/derive_parameters.py --task-file <path> --urdf <path> --reference-file <path>
    python3 tools/locomotion_heuristics/derive_parameters.py --robot drc_atlas --check

WHAT IT DOES NOT DO. It does not write the task file. Every number it prints is a STARTING POINT to sweep in
simulation, not a tuned value - the dissertation's own procedure is to add one heuristic at a time and measure the
viable operating region after each (section 4.3, figure 4-13) - and pasting a generated block over a tuned one would
throw that work away. `--check` compares the shipped block against the derivation and reports the differences, which
is the way to notice that a gait or a geometry has moved out from under a coefficient.

THE THREE GROUPS IT PRINTS.
  * Derived in closed form, no fitting: `capture_point`, `high_speed_turning`, `hip_centered_stepping`,
    `impulse_scaling`, `centripetal_acceleration`, and the clamps. These follow from geometry, from the gait, or from
    a textbook expression, and this script is the authority for them.
  * Sized from the command limits: `orientation_compensation_parameters.pitch_per_forward_velocity_gain`,
    `height_compensation.height_per_speed_gain`, `translational_stepping`, `in_place_turning`. These depend on a design
    choice (how much lean, how much crouch), so the script takes it from an explicit, printed assumption.
  * Genuinely fitted, and left at zero: the rest of Table C.2. They are what Bledt's extraction framework produces
    from data (chapter 4), and a number invented here would be a guess wearing a derivation's clothes.
"""

import argparse
import dataclasses
import math
import os
import re
import sys
from typing import Any

import numpy as np

from config_textproto import textproto_document

try:
    import pinocchio as pin
except ImportError:  # pragma: no cover - the dev container always has it
    sys.exit(
        "pinocchio is not importable. Run this inside the dev container:\n "
        " docker compose exec app python3"
        " tools/locomotion_heuristics/derive_parameters.py --robot drc_atlas"
    )

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

#: Where a robot keeps its files, by the name used with --robot. Every centroidal MPC package in robot_models is in
#: ROBOTS or in UNDERIVED_ROBOTS (packages whose coefficients the two-foot formulas cannot derive, such as ANYmal B,
#: Tron1 and the wheeled ANYmal C);
#: the whole-body ones are absent, because WBMpcInterface never builds the heuristic layer.
#:
#: NOT a Bazel target. Pinocchio reaches Python through robotpkg's bindings (`/opt/openrobots`, Python 3.12, on the
#: PYTHONPATH of the dev image and of setup_env.sh) while Bazel's toolchain is a hermetic 3.11 that cannot see them, so
#: this runs under the dev container's system interpreter like tools/hooks/format_code.py does.
#: `make test-heuristic-parameters` is the entry point.
# LINT.IfChange(derive_parameters_robots)
ROBOTS: dict[str, dict[str, str]] = {
    "drc_atlas": {
        "mpc": "robot_models/drc_atlas/drc_atlas_centroidal_mpc",
        "urdf": "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf",
    },
    "engineai_sa01": {
        "mpc": "robot_models/engineai_sa01/engineai_sa01_centroidal_mpc",
        "urdf": "robot_models/engineai_sa01/engineai_sa01_description/urdf/zq_sa01.urdf",
    },
    "unitree_g1": {
        "mpc": "robot_models/unitree_g1/g1_centroidal_mpc",
        "urdf": "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf",
    },
    "unitree_r1": {
        "mpc": "robot_models/unitree_r1/unitree_r1_centroidal_mpc",
        "urdf": "robot_models/unitree_r1/unitree_r1_description/urdf/R1.urdf",
    },
}
#: The centroidal packages whose coefficients this script does not derive: its formulas are written for two feet (the
#: foot separation, the step width, the bipeds' gait table, footholds fixed in the world). ANYmal B plans on four point
#: contacts, and Tron1 and the wheeled ANYmal C on two and four rolling contacts whose stance points move with the base
#: (the wheeled ANYmal C also lifts them in its turning_trot). Their task files are still held to the default-off rule
#: (test_derive_parameters.py).
UNDERIVED_ROBOTS: dict[str, str] = {
    "anymal_b": "robot_models/anybotics_anymal_b/anymal_b_centroidal_mpc",
    "tron1": "robot_models/limx_tron1/tron1_centroidal_mpc",
    "anymal_c_wheels": "robot_models/anybotics_anymal_c_wheels/anymal_c_wheels_centroidal_mpc",
}
# LINT.ThenChange(//legged_nmpc/docs/locomotion_heuristics/README.md:derive_parameters_usage, //Makefile:derive_heuristic_parameters_usage, //legged_nmpc/legged_centroidal_mpc/test/testNominalPendulum.cpp:nominal_pendulum_robots)

# [m/s^2] The robot models' gravity: Pinocchio's default, which every shipped URDF and MuJoCo scene keeps
# (legged_common_mpc_app/robot testShippedRobotGravity), and which the MPC reads from the model.
GRAVITY = 9.81

# The gait table every centroidal MPC reads, from the repository root.
GAIT_FILE = "legged_nmpc/legged_common_mpc/config/command/gait.textproto"

# The fields of the reference, task and contact-planning files derive_parameters() reads, which the schema-less reader
# sees only where a file writes them: the reference's command limits, which the stack requires (an absent one is
# reported, and a stand-in assumed), and the defaults of the schema for the fields a contact-planning file may leave out,
# which the stack then runs. A field renamed or a default changed in the schemas must change these.
# LINT.IfChange(derive_parameters_inputs)
REQUIRED_COMMAND_LIMITS = (
    "cmd_max_linvel_base_rt_world_ewrt_heading_x",
    "cmd_max_angvel_base_rt_world_ewrt_world_z",
    "nom_pos_base_rt_surface_ewrt_world_z",
)
FOOT_SEPARATION_DEFAULTS = {
    "max_norm_pos_lfoot_rt_rfoot_ewrt_heading_x": 0.5,
    "min_pos_lfoot_rt_rfoot_ewrt_heading_y": 0.15,
    "max_pos_lfoot_rt_rfoot_ewrt_heading_y": 0.45,
}
HLIP_STEP_WIDTH_DEFAULT = 0.25
NOMINAL_FOOTHOLD_STEP_WIDTH_DEFAULT = 0.0
# LINT.ThenChange(//legged_nmpc/legged_mpc_config/reference_file.proto:derive_parameters_command_limits, //legged_nmpc/legged_mpc_config/reference_file.proto:derive_parameters_base_height, //legged_nmpc/legged_mpc_config/contact_planning_file.proto:derive_parameters_foot_separation, //legged_nmpc/legged_mpc_config/contact_planning_file.proto:derive_parameters_hlip_step_width, //legged_nmpc/legged_mpc_config/nominal_foothold_config.proto:derive_parameters_step_width)

# The contact planner's file of a robot, beside its task file.
# LINT.IfChange(contact_planning_file_name)
CONTACT_PLANNING_FILE = "config/mpc/contact_planning.textproto"
# LINT.ThenChange(//legged_nmpc/legged_common_mpc/include/legged_common_mpc/config/ConfigFiles.h:contact_planning_file_name)

# The cost of the task file's `costs` list that replaces base-pose tracking with CoM + ACoM tracking, and the retired
# top-level boolean it replaced. MpcFormulationConfig.cpp normalizes cost names the same way (case, '_', '-' and ' '
# ignored), so either spelling of the name is recognized here too.
# LINT.IfChange(com_and_acom_tracking_cost_name)
COM_AND_ACOM_TRACKING_COST = "com_and_acom_tracking_cost"
RETIRED_ACOM_KEY = "use_com_and_acom_tracking"
# LINT.ThenChange(//legged_nmpc/legged_common_mpc/src/common/MpcFormulationConfig.cpp:com_and_acom_tracking_cost_name)

# The top-level key that names the contact input parameterization, the name whose contact input lives in the local
# contact frame, and the retired boolean it replaced (which the MPC refuses at start-up, whatever its value).
# LINT.IfChange(contact_input_parameterization_name)
CONTACT_INPUT_PARAMETERIZATION_KEY = "contact_input_parameterization"
BASIS_VECTOR_CONTACT_INPUTS = "basis_vectors"
RETIRED_BASIS_KEY = "use_contact_basis_vector_inputs"
# LINT.ThenChange(//legged_nmpc/legged_common_mpc/include/legged_common_mpc/common/ContactInputParameterization.h:contact_input_parameterization_names)

# The top-level key that names where the mode schedule and the footholds come from, the name of the online contact
# planner (which supplies the footholds itself, so a foothold heuristic is refused beside it), the name of the gait
# schedule (the default, and what every robot ships), and the retired boolean the key replaced (which the MPC refuses
# at start-up, whatever its value).
# LINT.IfChange(contact_schedule_source_name)
CONTACT_SCHEDULE_SOURCE_KEY = "contact_schedule_source"
CONTACT_PLANNER_SCHEDULE_SOURCE = "contact_planner"
GAIT_SCHEDULE_SOURCE = "gait_schedule"
RETIRED_CONTACT_PLANNING_KEY = "use_contact_planning"
# LINT.ThenChange(//legged_nmpc/legged_common_mpc/include/legged_common_mpc/common/MpcFormulationConfig.h:contact_schedule_source_names)

# The cost of the task file's `costs` list that ends the horizon on the DCM (capture point), and the retired top-level
# boolean it replaced (which the MPC refuses at start-up, whatever its value).
# LINT.IfChange(retired_formulation_keys)
DCM_TERMINAL_COST = "dcm_terminal_cost"
RETIRED_DCM_TERMINAL_COST_KEY = "use_dcm_terminal_cost"
# LINT.ThenChange(//legged_nmpc/legged_mpc_config/task_file.proto:retired_formulation_keys)

# ---------------------------------------------------------------------------------------------------------------
# Assumptions the sized-from-the-command-limits group rests on. They are constants HERE, and printed with the
# result, rather than buried in the expressions, because they are the only judgment calls in the script and the
# first thing anyone should disagree with.
# ---------------------------------------------------------------------------------------------------------------

#: [rad] How far the base pitches into the command at the maximum forward stick,
#: cmd_max_linvel_base_rt_world_ewrt_heading_x. A few degrees carries the center of mass towards the leading edge of
#: the support (dissertation section 4.3) and keeps the body close to level, which on Bledt's model was also what kept
#: its small-roll-and-pitch approximation valid; the control model here carries the full base rotation as a quaternion
#: and makes no such approximation.
DES_ERR_EULERZYX_BASE_RT_WORLD_PITCH = math.radians(2.8)

#: [m] How far the base is lowered at the maximum forward stick, cmd_max_linvel_base_rt_world_ewrt_heading_x: the
#: nominal minus the desired height, so positive lowers the base (the opposite sign of the pitch lean above, which is
#: desired minus nominal). Buys the swing leg vertical room inside the same joint range, and costs knee torque headroom.
DES_ERR_POS_BASE_RT_SURFACE_EWRT_WORLD_Z = 0.03

#: [rad] Pelvic obliquity of the periodic roll: the stance-side hip rises, the swing-side hip drops. Human gait runs
#: 4-8 degrees; this is deliberately conservative.
AMPLITUDE_EULERZYX_BASE_RT_WORLD_ROLL = math.radians(1.7)

#: [rad] The clamp on each of the roll and pitch the base-pose reference may ask for (orientation_compensation_parameters'
#: max_offset_eulerzyx_base_rt_world): 8.6 degrees, three times the lean at full stick, so the clamp stays off the derived lean and bounds
#: only what a stick the operator has not rate limited, or a fitted coefficient, could otherwise ask a walking base to
#: hold.
MAX_OFFSET_EULERZYX_BASE_RT_WORLD = math.radians(8.6)

#: Fraction of one step along the heading the capture point may consume.
CAPTURE_POINT_STEP_FRACTION = 0.4

#: Fraction of the friction budget the centripetal force reference may use. The clamp is on the total horizontal force,
#: which the stance feet share as they share the weight, so it is this fraction of each foot's own budget mu W / n in
#: single and in double support alike.
CENTRIPETAL_FRICTION_FRACTION = 0.4

#: Margin below the smallest duty factor of any shipped gait, for the 1/beta clamp.
DUTY_FACTOR_MARGIN = 0.075


@dataclasses.dataclass
class RobotGeometry:
    """What the URDF says, evaluated at the nominal standing posture of the task file's `initial_state`."""

    total_mass: float  # [kg]
    #: [m] center of mass above the mean foot height
    pos_com_rt_feetmean_ewrt_world_z: float
    #: [m] (x, y) of each foot's hip relative to the base, in the heading frame
    pos_hips_rt_base_ewrt_heading_xy: list[tuple[float, float]] = (
        dataclasses.field(default_factory=list)
    )
    hip_joint_names: list[str] = dataclasses.field(default_factory=list)
    #: [m] lateral distance in the heading frame, the absolute value, between the first two CONTACT FRAMES at the
    #: nominal posture (left first, as contact_parent_joint_names lists them) - the stance the robot actually stands in,
    #: which is wider than the hip joints are apart wherever the legs splay.
    nom_norm_pos_lfoot_rt_rfoot_ewrt_heading_y: float = 0.0

    @property
    def total_weight(self) -> float:
        """[N] The robot's weight."""
        return self.total_mass * GRAVITY

    @property
    def max_norm_pos_hips_rt_base_ewrt_heading_y(self) -> float:
        """[m] The largest lateral offset of a hip from the base, the absolute value; 0 without hips."""
        return (
            max(abs(y) for _, y in self.pos_hips_rt_base_ewrt_heading_xy)
            if self.pos_hips_rt_base_ewrt_heading_xy
            else 0.0
        )


@dataclasses.dataclass
class GaitCadence:
    """The timing of one named gait of `legged_common_mpc/config/command/gait.textproto`."""

    name: str
    stride_duration: float  # [s] one full left-right cycle
    step_duration: float  # [s] between consecutive touch-downs
    duty_factor: float  # fraction of the stride each foot is on the ground
    has_double_support: bool


def load_textproto(path: str) -> dict[str, Any]:
    """A configuration textproto at `path`, read without its schema, as nested dicts.

    The interpreter this runs on (robotpkg's, for pinocchio) has no protobuf, so the file is read with the standard
    library's textproto_document, which knows syntax and not schemas. Every field is a key: a scalar its value, a block
    a dict, and a field the file gives more than once, or as a `[...]` list, the list of its values. A repeated field
    the file happens to give once reads as that one value, so read repeated fields with as_list(). The strict parse
    against the schema is the C++ stack's and the tests' (every shipped file is parsed strictly there).

    Args:
      path: The file.

    Returns:
      Its fields; an empty file is an empty dict.

    Raises:
      OSError: The file cannot be read.
      textproto_document.DocumentError: It is not a textproto.
    """
    return _block_fields(textproto_document.load(path).root)


def parse_textproto(text: str, source: str = "<text>") -> dict[str, Any]:
    """The textproto `text` as load_textproto() reads a file; `source` names it in errors (DocumentError)."""
    return _block_fields(textproto_document.parse(text, source).root)


def _block_fields(block: textproto_document.BlockNode) -> dict[str, Any]:
    collected: dict[str, list[Any]] = {}
    listed: set[str] = set()
    for field in block.fields:
        if isinstance(field.value, textproto_document.ListNode):
            listed.add(field.name)
            values = [_node_value(element) for element in field.value.elements]
        else:
            values = [_node_value(field.value)]
        collected.setdefault(field.name, []).extend(values)
    return {
        name: values if (len(values) != 1 or name in listed) else values[0]
        for name, values in collected.items()
    }


def _node_value(
    node: textproto_document.ScalarNode | textproto_document.BlockNode,
) -> Any:
    if isinstance(node, textproto_document.BlockNode):
        return _block_fields(node)
    return node.value()


def as_list(value: Any) -> list[Any]:
    """A repeated field of load_textproto(): its values, however many the file gives (none: an empty list)."""
    if value is None:
        return []
    return list(value) if isinstance(value, list) else [value]


def gait_table(gait_file: dict[str, Any]) -> dict[str, Any]:
    """The gait file as the gait functions read it: `gait_list`, and each gait by its name.

    Args:
      gait_file: The gait file (load_textproto()), whose `gaits` are {name, mode_sequence, switching_times} entries.

    Returns:
      {"gait_list": [names], <name>: {"mode_sequence": [...], "switching_times": [...]}, ...}.
    """
    table: dict[str, Any] = {
        "gait_list": [str(name) for name in as_list(gait_file.get("gait_list"))]
    }
    for gait in as_list(gait_file.get("gaits")):
        table[str(gait.get("name", ""))] = {
            "mode_sequence": [
                str(mode) for mode in as_list(gait.get("mode_sequence"))
            ],
            "switching_times": [
                float(time) for time in as_list(gait.get("switching_times"))
            ],
        }
    return table


# ---------------------------------------------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------------------------------------------


def derive_geometry(
    urdf_path: str, task: dict, model_settings: dict
) -> RobotGeometry:
    """The C++ deriveLocomotionHeuristicModelParameters(), in Python and against the same nominal state.

    `pos_com_rt_feetmean_ewrt_world_z` is the C++ computeProxyPosComRtFeetmeanEwrtWorldZ() at `initial_state`: the
    center of mass above the MEAN height of the contact frames (the sole centers). It is the one pendulum length of this
    controller - a `dcm_terminal_cost` or a contact_planning.textproto `shared` block without
    `nom_pos_com_rt_feetmean_ewrt_world_z` resolves to it at start-up - and
    legged_centroidal_mpc:testNominalPendulum checks that every derived number a task file ships agrees with the C++
    side of it.

    Mirrors it deliberately, including the walk up the kinematic tree to the last joint before the floating base:
    that joint is the hip however the URDF happens to name it, which is why neither this nor the C++ looks for a name.

    The REDUCED model is what has to be built, not the URDF's own. `model_settings.fixed_joint_names` locks joints out
    of the MPC model, and those joints have no entry in `initial_state`, whose joints are named: each joint of the
    reduced model takes its own entry (on the DRC Atlas the four wrist joints are locked, and the posture of the legs
    is read by their names wherever the wrists were).

    Args:
      urdf_path: The robot's URDF.
      task: The task file.
      model_settings: The task file's `model_settings`, which names the locked joints and the contact parent joints.

    Returns:
      The geometry at the nominal posture; raises SystemExit when no contact parent joint exists in the URDF.
    """
    full_model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
    locked = [
        full_model.getJointId(name)
        for name in as_list(model_settings.get("fixed_joint_names"))
        if full_model.existJointName(name)
    ]
    model = (
        pin.buildReducedModel(full_model, locked, pin.neutral(full_model))
        if locked
        else full_model
    )
    data = model.createData()

    # The nominal posture: the base pose and the joints of `initial_state`, by name. Its joints are the crouch the
    # robot stands in - not the URDF's neutral, where the legs are straight - and a joint it does not name is at 0.
    # names[0] is "universe" and names[1] the floating base, so the MPC joints start at 2.
    mpc_joints: list[str] = list(model.names[2:])
    initial_state = task.get("initial_state", {}) or {}
    pos_base_rt_world_ewrt_world = (
        initial_state.get("pos_base_rt_world_ewrt_world", {}) or {}
    )
    eulerzyx_base_rt_world = (
        initial_state.get("eulerzyx_base_rt_world", {}) or {}
    )
    joint_positions = {
        str(entry.get("joint", "")): float(entry.get("value", 0.0))
        for entry in as_list(initial_state.get("joint_positions"))
    }
    pos_base_rt_world_ewrt_world_xy = np.array(
        [
            float(pos_base_rt_world_ewrt_world.get("x", 0.0)),
            float(pos_base_rt_world_ewrt_world.get("y", 0.0)),
        ]
    )
    eulerzyx_base_rt_world_yaw = float(eulerzyx_base_rt_world.get("yaw", 0.0))

    q = pin.neutral(model)
    q[0:3] = np.array(
        [
            pos_base_rt_world_ewrt_world_xy[0],
            pos_base_rt_world_ewrt_world_xy[1],
            float(pos_base_rt_world_ewrt_world.get("z", 0.0)),
        ]
    )
    quat_base_rt_world = pin.Quaternion(
        pin.rpy.rpyToMatrix(
            float(eulerzyx_base_rt_world.get("roll", 0.0)),
            float(eulerzyx_base_rt_world.get("pitch", 0.0)),
            eulerzyx_base_rt_world_yaw,
        )
    )
    q[3:7] = np.array(
        [
            quat_base_rt_world.x,
            quat_base_rt_world.y,
            quat_base_rt_world.z,
            quat_base_rt_world.w,
        ]
    )
    for name in mpc_joints:
        joint = model.getJointId(name)
        if model.joints[joint].nq == 1:
            q[model.joints[joint].idx_q] = joint_positions.get(name, 0.0)

    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    pin.centerOfMass(model, data, q, False)

    # The contact frames are NOT in the URDF: createPinocchioModel() adds them at run time, one per entry of
    # `contact_parent_joint_names`, offset from that joint by `contacts.pos_contact_rt_link_ewrt_link`. So the foot position
    # is computed here the same way rather than looked up by name.
    pos_contact_rt_link_ewrt_link = np.array(
        [
            float(
                (
                    task.get("contacts", {}).get(
                        "pos_contact_rt_link_ewrt_link", {}
                    )
                    or {}
                ).get(axis, 0.0)
            )
            for axis in ("x", "y", "z")
        ]
    )
    parent_joints = as_list(model_settings.get("contact_parent_joint_names"))
    pos_feet_rt_world_ewrt_world_z = [
        float(
            (data.oMi[model.getJointId(name)] * pos_contact_rt_link_ewrt_link)[
                2
            ]
        )
        for name in parent_joints
        if model.existJointName(name)
    ]
    if not pos_feet_rt_world_ewrt_world_z:
        raise SystemExit(
            "none of model_settings.contact_parent_joint_names %s exist in %s"
            % (parent_joints, os.path.basename(urdf_path))
        )
    pos_feetmean_rt_world_ewrt_world_z = float(
        np.mean(pos_feet_rt_world_ewrt_world_z)
    )

    pos_feet_rt_world_ewrt_world = [
        np.array(
            data.oMi[model.getJointId(name)] * pos_contact_rt_link_ewrt_link
        )
        for name in parent_joints
        if model.existJointName(name)
    ]
    # The heading frame is the base's twist heading (MpcRobotModelBase::getYawBaseRtWorld(), the C++
    # deriveLocomotionHeuristicModelParameters' rotation), 2 atan2(q_z, q_w) wrapped; it equals the ZYX yaw only while
    # the base is level.
    yaw_base_rt_world = 2.0 * math.atan2(
        quat_base_rt_world.z, quat_base_rt_world.w
    )
    yaw_base_rt_world = math.atan2(
        math.sin(yaw_base_rt_world), math.cos(yaw_base_rt_world)
    )
    rotmat_world_rt_heading = np.array(
        [
            [
                math.cos(yaw_base_rt_world),
                math.sin(yaw_base_rt_world),
            ],
            [
                -math.sin(yaw_base_rt_world),
                math.cos(yaw_base_rt_world),
            ],
        ]
    )
    nom_norm_pos_lfoot_rt_rfoot_ewrt_heading_y = 0.0
    if len(pos_feet_rt_world_ewrt_world) > 1:
        nom_pos_lfoot_rt_rfoot_ewrt_heading_xy = rotmat_world_rt_heading @ (
            pos_feet_rt_world_ewrt_world[0][0:2]
            - pos_feet_rt_world_ewrt_world[1][0:2]
        )
        nom_norm_pos_lfoot_rt_rfoot_ewrt_heading_y = float(
            abs(nom_pos_lfoot_rt_rfoot_ewrt_heading_xy[1])
        )
    geometry = RobotGeometry(
        total_mass=float(pin.computeTotalMass(model)),
        pos_com_rt_feetmean_ewrt_world_z=float(
            data.com[0][2] - pos_feetmean_rt_world_ewrt_world_z
        ),
        nom_norm_pos_lfoot_rt_rfoot_ewrt_heading_y=nom_norm_pos_lfoot_rt_rfoot_ewrt_heading_y,
    )

    for name in parent_joints:
        if not model.existJointName(name):
            geometry.pos_hips_rt_base_ewrt_heading_xy.append((0.0, 0.0))
            geometry.hip_joint_names.append("(not found: %s)" % name)
            continue
        joint = model.getJointId(name)
        while (
            model.parents[joint] > 1
        ):  # stop at the first child of the floating base: the hip
            joint = model.parents[joint]
        pos_hip_rt_base_ewrt_heading_xy = rotmat_world_rt_heading @ (
            np.array(data.oMi[joint].translation[0:2])
            - pos_base_rt_world_ewrt_world_xy
        )
        geometry.pos_hips_rt_base_ewrt_heading_xy.append(
            (
                float(pos_hip_rt_base_ewrt_heading_xy[0]),
                float(pos_hip_rt_base_ewrt_heading_xy[1]),
            )
        )
        geometry.hip_joint_names.append(model.names[joint])
    return geometry


# ---------------------------------------------------------------------------------------------------------------
# Gait
# ---------------------------------------------------------------------------------------------------------------


def gait_cadence(gaits: dict, name: str) -> GaitCadence:
    """Duty factor and step period of one gait, read the way the mode schedule defines them.

    The mode names say which foot is IN CONTACT - `LF` is contact flags {true, false} - which is the thing most
    easily got backwards when reading the gait file.

    Args:
      gaits: The gait file's table (gait_table()).
      name: The gait, a key of `gaits`; raises SystemExit for an unknown one.

    Returns:
      The cadence of the gait.
    """
    if name not in gaits:
        raise SystemExit(
            "unknown gait '%s'; the gait file lists: %s"
            % (name, ", ".join(gaits.get("gait_list", [])))
        )
    modes = gaits[name]["mode_sequence"]
    times = gaits[name]["switching_times"]
    stride = float(times[-1]) - float(times[0])
    contact = {"left": 0.0, "right": 0.0}
    for index, mode in enumerate(modes):
        duration = float(times[index + 1]) - float(times[index])
        left = mode in ("LF", "STANCE")
        right = mode in ("RF", "STANCE")
        contact["left"] += duration if left else 0.0
        contact["right"] += duration if right else 0.0
    duty = (
        min(contact["left"], contact["right"]) / stride if stride > 0 else 1.0
    )
    # Two touch-downs per stride in any alternating gait, so the step period is half the stride. Counting them from
    # the sequence would miscount a stride that begins mid-swing.
    return GaitCadence(
        name=name,
        stride_duration=stride,
        step_duration=stride / 2.0,
        duty_factor=duty,
        has_double_support=any(mode == "STANCE" for mode in modes),
    )


def smallest_duty_factor(gaits: dict) -> tuple[str, float]:
    """The lowest duty factor over every named gait: what the 1/beta clamp has to stay below."""
    worst_name, worst = "", 1.0
    for name in gaits.get("gait_list", []):
        if name not in gaits:
            continue
        try:
            cadence = gait_cadence(gaits, name)
        except (KeyError, IndexError, ZeroDivisionError):
            continue
        if 0.0 < cadence.duty_factor < worst:
            worst_name, worst = name, cadence.duty_factor
    return worst_name, worst


def largest_weight_compensation_ratio(gaits: dict) -> tuple[str, float]:
    """The largest W/(F beta) that any shipped gait asks for, as a multiple of instantaneous weight compensation.

    With n feet down the reference is W/(F beta) per foot against a baseline of W/n, a ratio of n/(F beta) with F = 2,
    so a gait's largest ratio is in its phase with the most feet down: 1/beta in double support, 1/(2 beta) in single
    support. The double support of the slowest-stepping walking gait usually binds, but not always: a gait without
    double support whose duty factor is below 1/2 (a run, a hop) asks for more than weight compensation in single
    support too, and is counted here.

    Args:
      gaits: The gait file's table (gait_table()).

    Returns:
      The binding gait and its ratio; ("", 1.0) when no gait asks for more than weight compensation.
    """
    worst_name, worst = "", 1.0
    for name in gaits.get("gait_list", []):
        if name not in gaits:
            continue
        try:
            cadence = gait_cadence(gaits, name)
        except (KeyError, IndexError, ZeroDivisionError):
            continue
        if cadence.duty_factor <= 0.0:
            continue
        most_feet_down = 2.0 if cadence.has_double_support else 1.0
        ratio = most_feet_down / (2.0 * cadence.duty_factor)
        if ratio > worst:
            worst_name, worst = name, ratio
    return worst_name, worst


# ---------------------------------------------------------------------------------------------------------------
# The coefficients
# ---------------------------------------------------------------------------------------------------------------


def nominal_stance_width(
    nom_pos_lfoot_rt_rfoot_ewrt_heading_y: float, geometry: RobotGeometry
) -> float:
    """[m] The lateral distance between the feet that the robot walks with, which the stance-sized coefficients keep.

    The task file's nominal_foothold.nom_pos_lfoot_rt_rfoot_ewrt_heading_y where it sets one; otherwise the stance of
    the nominal posture, the contact frames' lateral distance, because a robot with no nominal step stands and walks in
    that one; otherwise the hips' own width.

    Args:
      nom_pos_lfoot_rt_rfoot_ewrt_heading_y: nominal_foothold.nom_pos_lfoot_rt_rfoot_ewrt_heading_y [m], 0 when the file sets none.
      geometry: The robot's geometry at the nominal posture.

    Returns:
      The stance width, positive for any robot with hips or a stance.
    """
    if nom_pos_lfoot_rt_rfoot_ewrt_heading_y > 0.0:
        return nom_pos_lfoot_rt_rfoot_ewrt_heading_y
    if geometry.nom_norm_pos_lfoot_rt_rfoot_ewrt_heading_y > 0.0:
        return geometry.nom_norm_pos_lfoot_rt_rfoot_ewrt_heading_y
    return 2.0 * geometry.max_norm_pos_hips_rt_base_ewrt_heading_y


def friction_coefficient(task: dict) -> tuple[float, str | None]:
    """The friction coefficient of the cone the robot's task file actually lists, and the key it came from.

    The robots differ: Atlas and SA01 list `contact_wrench_cone` (contacts.contact_wrench_cone_soft_constraint), G1 and
    R1 `friction_force_cone` (contacts.friction_force_cone_soft_constraint, 0.4). Reading one key for all of them handed
    G1 and R1 the other cone's default.

    Args:
      task: The task file.

    Returns:
      The coefficient and the key it came from; (0.5, None) when neither cone is listed.
    """
    listed = as_list(task.get("soft_constraints"))
    contacts = task.get("contacts", {}) or {}
    for cone, key in (
        ("contact_wrench_cone", "contact_wrench_cone_soft_constraint"),
        ("friction_force_cone", "friction_force_cone_soft_constraint"),
    ):
        if cone in listed and "friction_coefficient" in (
            contacts.get(key) or {}
        ):
            return (
                float(contacts[key]["friction_coefficient"]),
                "contacts.%s.friction_coefficient" % key,
            )
    return 0.5, None


def derive_parameters(
    geometry: RobotGeometry,
    cadence: GaitCadence,
    limits: dict,
    task: dict,
    planning: dict,
    gaits: dict,
) -> tuple[dict[str, dict[str, float]], list[str]]:
    """Every coefficient, with the one-line derivation of each returned alongside it."""
    notes: list[str] = []
    out: dict[str, dict[str, float]] = {}

    for field in REQUIRED_COMMAND_LIMITS:
        if field not in limits:
            notes.append(
                "WARNING the reference file has no %s, which the MPC requires"
                " (it refuses the file); a stand-in is assumed below" % field
            )
    cmd_max_linvel_base_rt_world_ewrt_heading_x = float(
        limits.get("cmd_max_linvel_base_rt_world_ewrt_heading_x", 1.0)
    )
    cmd_max_angvel_base_rt_world_ewrt_world_z = float(
        limits.get("cmd_max_angvel_base_rt_world_ewrt_world_z", 1.0)
    )
    # None when the reference file sets no nominal base height.
    nom_pos_base_rt_surface_ewrt_world_z: float | None = (
        float(limits["nom_pos_base_rt_surface_ewrt_world_z"])
        if "nom_pos_base_rt_surface_ewrt_world_z" in limits
        else None
    )
    friction, friction_source = friction_coefficient(task)
    if friction_source is None:
        notes.append(
            "WARNING the task file lists neither contact_wrench_cone nor"
            " friction_force_cone in soft_constraints, so there is no friction"
            " coefficient to size the centripetal clamp against; %.2f is"
            " assumed." % friction
        )
    nom_pos_lfoot_rt_rfoot_ewrt_heading_y = float(
        task.get("nominal_foothold", {}).get(
            "nom_pos_lfoot_rt_rfoot_ewrt_heading_y",
            NOMINAL_FOOTHOLD_STEP_WIDTH_DEFAULT,
        )
    )
    foot_separation = planning.get("foot_separation", {}) or {}
    # A robot with a contact-planning file runs the schema's default of a field the file leaves out; one without has no
    # planner bound at all (0 below: none).
    separation_defaults = (
        FOOT_SEPARATION_DEFAULTS
        if planning
        else dict.fromkeys(FOOT_SEPARATION_DEFAULTS, 0.0)
    )
    # foot_separation.max_norm_pos_lfoot_rt_rfoot_ewrt_heading_x, i.e. the REACHABILITY bound, not the hlip block's
    # field of that name, which is the closed-form planner's own clip on the step it emits. With no planner running it
    # is the reach that bounds how far a landing target may be pushed.
    max_norm_pos_lfoot_rt_rfoot_ewrt_heading_x = float(
        foot_separation.get(
            "max_norm_pos_lfoot_rt_rfoot_ewrt_heading_x",
            separation_defaults["max_norm_pos_lfoot_rt_rfoot_ewrt_heading_x"],
        )
    )
    max_pos_lfoot_rt_rfoot_ewrt_heading_y = float(
        foot_separation.get(
            "max_pos_lfoot_rt_rfoot_ewrt_heading_y",
            separation_defaults["max_pos_lfoot_rt_rfoot_ewrt_heading_y"],
        )
    )
    min_pos_lfoot_rt_rfoot_ewrt_heading_y = float(
        foot_separation.get(
            "min_pos_lfoot_rt_rfoot_ewrt_heading_y",
            separation_defaults["min_pos_lfoot_rt_rfoot_ewrt_heading_y"],
        )
    )
    hlip_nom_pos_lfoot_rt_rfoot_ewrt_heading_y = float(
        (planning.get("hlip", {}) or {}).get(
            "nom_pos_lfoot_rt_rfoot_ewrt_heading_y",
            HLIP_STEP_WIDTH_DEFAULT if planning else 0.0,
        )
    )

    # THE PENDULUM LENGTH. Every linear-inverted-pendulum expression in this controller - the DCM terminal cost, the
    # contact planner, and the two heuristics below - uses ONE length, or they disagree about where the robot is
    # heading. A nom_pos_com_rt_feetmean_ewrt_world_z left out of either file (both shipped robots that have the blocks)
    # is the model's, computed
    # above the way the C++ computeProxyPosComRtFeetmeanEwrtWorldZ() computes it; a positive value is an explicit override, and wins, and a
    # disagreement between it and the model is reported rather than silently resolved: it means the controller is
    # reasoning about a pendulum the robot does not have.
    cfg_pos_com_rt_feetmean_ewrt_world_z = float(
        task.get("dcm_terminal_cost", {}).get(
            "nom_pos_com_rt_feetmean_ewrt_world_z", 0.0
        )
    ) or float(
        (planning.get("shared", {}) or {}).get(
            "nom_pos_com_rt_feetmean_ewrt_world_z", 0.0
        )
    )
    lip_pos_com_rt_feetmean_ewrt_world_z = (
        cfg_pos_com_rt_feetmean_ewrt_world_z
        if cfg_pos_com_rt_feetmean_ewrt_world_z > 0
        else geometry.pos_com_rt_feetmean_ewrt_world_z
    )
    if (
        cfg_pos_com_rt_feetmean_ewrt_world_z > 0
        and abs(
            cfg_pos_com_rt_feetmean_ewrt_world_z
            - geometry.pos_com_rt_feetmean_ewrt_world_z
        )
        > 0.05 * geometry.pos_com_rt_feetmean_ewrt_world_z
    ):
        notes.append(
            "WARNING the configured LIP height (%.4f m,"
            " dcm_terminal_cost.nom_pos_com_rt_feetmean_ewrt_world_z) and the model's center of mass"
            " above the feet (%.4f m at initial_state) differ by %.0f%%. The"
            " configured one is used below so that the capture point, the DCM"
            " terminal cost and the contact planner agree, but one of the two"
            " is wrong."
            % (
                cfg_pos_com_rt_feetmean_ewrt_world_z,
                geometry.pos_com_rt_feetmean_ewrt_world_z,
                100.0
                * abs(
                    cfg_pos_com_rt_feetmean_ewrt_world_z
                    - geometry.pos_com_rt_feetmean_ewrt_world_z
                )
                / geometry.pos_com_rt_feetmean_ewrt_world_z,
            )
        )

    # The keys below are the fields of the schema's heuristic blocks (locomotion_heuristics_config.proto), one for one:
    # a key this prints that the schema does not have would be refused when pasted into the task file.
    # LINT.IfChange(derived_heuristic_keys)
    # ---- base pose ----
    pitch_per_forward_velocity_gain = 0.0
    if cmd_max_linvel_base_rt_world_ewrt_heading_x > 0:
        pitch_per_forward_velocity_gain = (
            DES_ERR_EULERZYX_BASE_RT_WORLD_PITCH
            / cmd_max_linvel_base_rt_world_ewrt_heading_x
        )
    out["orientation_compensation_parameters"] = {
        # 0, and deliberately: at a CONSTANT sidestep velocity there is no net lateral force to lean against, and
        # what a biped's roll really does in single support is oscillate, which is periodic_orientation's job.
        "roll_per_lateral_velocity_gain": 0.0,
        "offset_eulerzyx_base_rt_world_roll": 0.0,
        "pitch_per_forward_velocity_gain": round(
            pitch_per_forward_velocity_gain, 4
        ),
        "offset_eulerzyx_base_rt_world_pitch": 0.0,
        "max_offset_eulerzyx_base_rt_world": round(
            MAX_OFFSET_EULERZYX_BASE_RT_WORLD, 3
        ),
    }
    notes.append(
        "orientation_compensation_parameters.pitch_per_forward_velocity_gain = %.1f deg / %.2f"
        " m/s = %.4f rad s/m  (POSITIVE is nose-down in this Euler-ZYX layout)"
        % (
            math.degrees(DES_ERR_EULERZYX_BASE_RT_WORLD_PITCH),
            cmd_max_linvel_base_rt_world_ewrt_heading_x,
            pitch_per_forward_velocity_gain,
        )
    )
    notes.append(
        "orientation_compensation_parameters.roll_per_lateral_velocity_gain left at 0: no"
        " steady lateral lean to fit"
    )

    out["periodic_orientation_parameters"] = {
        "amplitude_eulerzyx_base_rt_world_roll": round(
            AMPLITUDE_EULERZYX_BASE_RT_WORLD_ROLL, 3
        ),
        "roll_phase_rate": round(2.0 * math.pi, 7),
        # Phase [0, 0.5) is LF = LEFT STANCE, so its midpoint is 0.25; a positive roll raises the left (stance) side;
        # and sin(2*pi*0.25 + 0) is its maximum. So offset 0 already drops the hip on the SWING side. Nothing to fit.
        "roll_phase_offset": 0.0,
        # 0: pitch oscillates once per STEP and where its extremum falls is a property of this robot's mass
        # distribution, not of the gait's geometry. A sinusoid at the wrong phase is worse than none.
        "amplitude_eulerzyx_base_rt_world_pitch": 0.0,
        "pitch_phase_rate": round(4.0 * math.pi, 7),
        "pitch_phase_offset": 0.0,
    }
    notes.append(
        "periodic_orientation_parameters.amplitude_eulerzyx_base_rt_world_roll = %.1f deg of pelvic obliquity;"
        " phase 0 peaks at mid LEFT stance"
        % math.degrees(AMPLITUDE_EULERZYX_BASE_RT_WORLD_ROLL)
    )
    notes.append(
        "periodic_orientation_parameters.amplitude_eulerzyx_base_rt_world_pitch left at 0: its phase needs data,"
        " not arithmetic"
    )

    height_per_speed_gain = 0.0
    if cmd_max_linvel_base_rt_world_ewrt_heading_x > 0:
        height_per_speed_gain = (
            -DES_ERR_POS_BASE_RT_SURFACE_EWRT_WORLD_Z
            / cmd_max_linvel_base_rt_world_ewrt_heading_x
        )
    # The height the crouch percentage is reported against: the reference file's nominal base height, or without one
    # the CoM's height above the feet, a different quantity (the CoM, not the base, and above the feet rather than the
    # ground) that is close enough for a percentage that is only reported.
    if nom_pos_base_rt_surface_ewrt_world_z is not None:
        crouch_percent_denominator = nom_pos_base_rt_surface_ewrt_world_z
        crouch_percent_denominator_name = "nominal base height"
    else:
        crouch_percent_denominator = geometry.pos_com_rt_feetmean_ewrt_world_z
        crouch_percent_denominator_name = (
            "CoM height above the feet (no nominal base height)"
        )
    crouch_percent = 0.0
    if crouch_percent_denominator > 0:
        crouch_percent = (
            100.0
            * DES_ERR_POS_BASE_RT_SURFACE_EWRT_WORLD_Z
            / crouch_percent_denominator
        )
    out["height_compensation"] = {
        "height_per_speed_squared_gain": 0.0,
        "height_per_speed_gain": round(height_per_speed_gain, 4),
        "offset_pos_base_rt_world_ewrt_world_z": 0.0,
        "max_offset_pos_base_rt_world_ewrt_world_z": round(
            max(0.05, 1.5 * DES_ERR_POS_BASE_RT_SURFACE_EWRT_WORLD_Z), 3
        ),
    }
    notes.append(
        "height_compensation.height_per_speed_gain = -%.3f m / %.2f m/s = %.4f "
        " (%.1f%% of the %.4f m %s)"
        % (
            DES_ERR_POS_BASE_RT_SURFACE_EWRT_WORLD_Z,
            cmd_max_linvel_base_rt_world_ewrt_heading_x,
            height_per_speed_gain,
            crouch_percent,
            crouch_percent_denominator,
            crouch_percent_denominator_name,
        )
    )

    # ---- foothold ----
    # This heuristic REPLACES the step-width anchor, so the scale that changes nothing about the nominal geometry is
    # the one reproducing the stance the robot already walks with. Listing it at 1.0 would resize the stance the
    # moment it is switched on, which is a different gait rather than a correction.
    # Twice the largest lateral hip offset from the base: the hips' lateral distance of a symmetric robot.
    nom_norm_pos_lhip_rt_rhip_ewrt_heading_y = (
        2.0 * geometry.max_norm_pos_hips_rt_base_ewrt_heading_y
    )
    stance_nom_pos_lfoot_rt_rfoot_ewrt_heading_y = nominal_stance_width(
        nom_pos_lfoot_rt_rfoot_ewrt_heading_y, geometry
    )
    lateral_scale = 1.0
    if nom_norm_pos_lhip_rt_rhip_ewrt_heading_y > 0:
        lateral_scale = (
            stance_nom_pos_lfoot_rt_rfoot_ewrt_heading_y
            / nom_norm_pos_lhip_rt_rhip_ewrt_heading_y
        )
    out["hip_centered_stepping"] = {
        "lateral_scale": round(lateral_scale, 3),
        "longitudinal_scale": 1.0,
    }
    if nom_pos_lfoot_rt_rfoot_ewrt_heading_y > 0:
        lateral_scale_note = (
            "hip_centered_stepping.lateral_scale = nominal_foothold.nom_pos_lfoot_rt_rfoot_ewrt_heading_y"
            " %.3f / hip width %.3f = %.3f%s"
        )
    else:
        lateral_scale_note = (
            "hip_centered_stepping.lateral_scale = nominal stance %.3f / hip"
            " width %.3f = %.3f: nominal_foothold.nom_pos_lfoot_rt_rfoot_ewrt_heading_y is 0, so the"
            " stance to reproduce is the one the robot stands in%s"
        )
    planner_comparison = ""
    if (
        hlip_nom_pos_lfoot_rt_rfoot_ewrt_heading_y > 0
        and nom_norm_pos_lhip_rt_rhip_ewrt_heading_y > 0
    ):
        planner_comparison = (
            "  (%.2f matches the %.3f m stance of the nominal posture, %.2f the"
            " planner's %.2f m orbit)"
            % (
                geometry.nom_norm_pos_lfoot_rt_rfoot_ewrt_heading_y
                / nom_norm_pos_lhip_rt_rhip_ewrt_heading_y,
                geometry.nom_norm_pos_lfoot_rt_rfoot_ewrt_heading_y,
                hlip_nom_pos_lfoot_rt_rfoot_ewrt_heading_y
                / nom_norm_pos_lhip_rt_rhip_ewrt_heading_y,
                hlip_nom_pos_lfoot_rt_rfoot_ewrt_heading_y,
            )
        )
    notes.append(
        lateral_scale_note
        % (
            stance_nom_pos_lfoot_rt_rfoot_ewrt_heading_y,
            nom_norm_pos_lhip_rt_rhip_ewrt_heading_y,
            lateral_scale,
            planner_comparison,
        )
    )
    if (
        abs(
            geometry.pos_hips_rt_base_ewrt_heading_xy[0][0]
            if geometry.pos_hips_rt_base_ewrt_heading_xy
            else 0.0
        )
        < 1e-6
    ):
        notes.append(
            "hip_centered_stepping.longitudinal_scale is inert here: the hip"
            " joints sit at x = 0"
        )

    # The lateral room is asymmetric: the shipped stance may already be AT
    # foot_separation.max_pos_lfoot_rt_rfoot_ewrt_heading_y, in which case the only room is inward. Take the tighter of
    # the two bounds so the clamp fits whichever way the step is pushed. The inward room is the nominal stance width
    # minus its lower bound, the outward room its upper bound minus the nominal width [m].
    inward_err_pos_lfoot_rt_rfoot_ewrt_heading_y = 0.0
    outward_err_pos_lfoot_rt_rfoot_ewrt_heading_y = 0.0
    if nom_pos_lfoot_rt_rfoot_ewrt_heading_y > 0:
        inward_err_pos_lfoot_rt_rfoot_ewrt_heading_y = max(
            nom_pos_lfoot_rt_rfoot_ewrt_heading_y
            - min_pos_lfoot_rt_rfoot_ewrt_heading_y,
            0.0,
        )
        outward_err_pos_lfoot_rt_rfoot_ewrt_heading_y = max(
            max_pos_lfoot_rt_rfoot_ewrt_heading_y
            - nom_pos_lfoot_rt_rfoot_ewrt_heading_y,
            0.0,
        )
    max_norm_offset_pos_foot_rt_world_ewrt_world_xy = 0.25
    if max_norm_pos_lfoot_rt_rfoot_ewrt_heading_x > 0:
        max_norm_offset_pos_foot_rt_world_ewrt_world_xy = (
            CAPTURE_POINT_STEP_FRACTION
            * max_norm_pos_lfoot_rt_rfoot_ewrt_heading_x
        )
    out["capture_point"] = {
        "gain": 1.0,
        "override_pos_com_rt_feetmean_ewrt_world_z": round(
            lip_pos_com_rt_feetmean_ewrt_world_z, 4
        ),
        "max_norm_offset_pos_foot_rt_world_ewrt_world_xy": round(
            max_norm_offset_pos_foot_rt_world_ewrt_world_xy, 3
        ),
    }
    notes.append(
        "capture_point: omega = sqrt(%.2f / %.4f) = %.3f rad/s, so the offset"
        " is %.4f s per m/s of velocity error"
        % (
            GRAVITY,
            lip_pos_com_rt_feetmean_ewrt_world_z,
            math.sqrt(GRAVITY / lip_pos_com_rt_feetmean_ewrt_world_z),
            math.sqrt(lip_pos_com_rt_feetmean_ewrt_world_z / GRAVITY),
        )
    )
    if max_norm_pos_lfoot_rt_rfoot_ewrt_heading_x > 0:
        notes.append(
            "capture_point.max_norm_offset_pos_foot_rt_world_ewrt_world_xy = %.0f%% of"
            " foot_separation.max_norm_pos_lfoot_rt_rfoot_ewrt_heading_x %.2f = %.3f m"
            % (
                100 * CAPTURE_POINT_STEP_FRACTION,
                max_norm_pos_lfoot_rt_rfoot_ewrt_heading_x,
                max_norm_offset_pos_foot_rt_world_ewrt_world_xy,
            )
        )
    else:
        notes.append(
            "capture_point.max_norm_offset_pos_foot_rt_world_ewrt_world_xy = %.3f m, the fallback: there is no"
            " contact_planning.textproto with a foot_separation.max_norm_pos_lfoot_rt_rfoot_ewrt_heading_x"
            " to take %.0f%% of"
            % (
                max_norm_offset_pos_foot_rt_world_ewrt_world_xy,
                100 * CAPTURE_POINT_STEP_FRACTION,
            )
        )
    if (
        nom_pos_lfoot_rt_rfoot_ewrt_heading_y > 0
        and outward_err_pos_lfoot_rt_rfoot_ewrt_heading_y
        < max_norm_offset_pos_foot_rt_world_ewrt_world_xy
    ):
        notes.append(
            "WARNING the lateral room is asymmetric: the %.2f m nominal stance"
            " leaves %.3f m outward to foot_separation.max_pos_lfoot_rt_rfoot_ewrt_heading_y %.2f but %.3f m"
            " inward to foot_separation.min_pos_lfoot_rt_rfoot_ewrt_heading_y %.2f, so a capture step pushed OUTWARD can exceed"
            " the planner's own bound. Narrow nominal_foothold.nom_pos_lfoot_rt_rfoot_ewrt_heading_y, or"
            " accept that this clamp only fits inward."
            % (
                nom_pos_lfoot_rt_rfoot_ewrt_heading_y,
                outward_err_pos_lfoot_rt_rfoot_ewrt_heading_y,
                max_pos_lfoot_rt_rfoot_ewrt_heading_y,
                inward_err_pos_lfoot_rt_rfoot_ewrt_heading_y,
                min_pos_lfoot_rt_rfoot_ewrt_heading_y,
            )
        )

    # Raibert's rule: the foot lands half a stance of travel ahead of the hip it lands under. The foothold anchor is the
    # base PREDICTED at touch-down (SwitchedModelReferenceManager::getNomPosFootRtWorldEwrtWorldXy), i.e. the hip at the moment of
    # landing, so the lead is a1 = T_stance / 2. Expressed per unit of the stance the foot is about to begin - the
    # stance fraction, 1/2 - rather than as a constant, because the gait scheduler changes the stance duration with the
    # commanded speed (slow_walk ... run) and one constant is right in only one of those bands. Bledt's constant a1 is
    # left at zero.
    stance = cadence.duty_factor * cadence.stride_duration
    out["translational_stepping"] = {
        "forward_per_forward_velocity_gain": 0.0,
        "forward_stance_fraction": 0.5,
        "offset_pos_foot_rt_world_ewrt_heading_x": 0.0,
        "lateral_per_lateral_velocity_gain": 0.0,
        "lateral_stance_fraction": 0.5,
        "offset_pos_foot_rt_world_ewrt_heading_y": 0.0,
    }
    notes.append(
        "translational_stepping stance fraction = 1/2 (Raibert): a1 = T_stance"
        " / 2, e.g. %.3f s on `%s` (stance %.2f s of a %.2f s stride), and it"
        " follows the gait the scheduler selects"
        % (stance / 2.0, cadence.name, stance, cadence.stride_duration)
    )

    # The same half-stance rule applied to the speed the HIP travels at rather than the body: a foot at lever arm r
    # from the center moves at r * psidot under a body yaw rate, so the lead is r * psidot * T_stance / 2, i.e. a
    # stance lever of r / 2 per unit of stance duration.
    pos_lfoot_rt_feetmean_ewrt_heading_y = (
        stance_nom_pos_lfoot_rt_rfoot_ewrt_heading_y / 2.0
    )
    out["in_place_turning"] = {
        "forward_per_yaw_rate": 0.0,
        "forward_stance_lever": round(
            pos_lfoot_rt_feetmean_ewrt_heading_y / 2.0, 4
        ),
        "offset_pos_foot_rt_world_ewrt_heading_x": 0.0,
        "lateral_per_yaw_rate": 0.0,
        "offset_pos_foot_rt_world_ewrt_heading_y": 0.0,
    }
    notes.append(
        "in_place_turning.forward_stance_lever = (lever r = %.3f m) / 2 = %.4f"
        " m/rad per second of stance, i.e. %.3f m per foot at the %.2f rad/s"
        " stick on `%s`"
        % (
            pos_lfoot_rt_feetmean_ewrt_heading_y,
            pos_lfoot_rt_feetmean_ewrt_heading_y / 2.0,
            pos_lfoot_rt_feetmean_ewrt_heading_y
            / 2.0
            * stance
            * cmd_max_angvel_base_rt_world_ewrt_world_z,
            cmd_max_angvel_base_rt_world_ewrt_world_z,
            cadence.name,
        )
    )

    # Closed form, not fitted: Bledt equation 4.31 places the foot along the resultant of gravity and the centripetal
    # acceleration, which displaces it by (z_com / g) * (pdot x omega). The SAME coefficient on both axes, because
    # the two entries are components of one vector.
    lean_coefficient = lip_pos_com_rt_feetmean_ewrt_world_z / GRAVITY
    out["high_speed_turning"] = {
        "forward_per_cross_term": round(lean_coefficient, 4),
        "offset_pos_foot_rt_world_ewrt_heading_x": 0.0,
        "lateral_per_cross_term": round(lean_coefficient, 4),
        "offset_pos_foot_rt_world_ewrt_heading_y": 0.0,
    }
    notes.append(
        "high_speed_turning = z_com / g = %.4f / %.2f = %.4f s^2 (Bledt eq."
        " 4.31); throws the foot %.3f m outwards at the (%.2f m/s, %.2f rad/s)"
        " stick"
        % (
            lip_pos_com_rt_feetmean_ewrt_world_z,
            GRAVITY,
            lean_coefficient,
            lean_coefficient
            * cmd_max_linvel_base_rt_world_ewrt_heading_x
            * cmd_max_angvel_base_rt_world_ewrt_world_z,
            cmd_max_linvel_base_rt_world_ewrt_heading_x,
            cmd_max_angvel_base_rt_world_ewrt_world_z,
        )
    )

    # ---- wrench ----
    worst_gait, worst_duty = smallest_duty_factor(gaits)
    duty_clamp = max(0.05, round(worst_duty - DUTY_FACTOR_MARGIN, 2))
    ratio_gait, ratio = largest_weight_compensation_ratio(gaits)
    out["impulse_scaling"] = {
        "scale": 1.0,
        "minimum_duty_factor": duty_clamp,
        "maximum_weight_compensation_ratio": round(
            max(2.0, math.ceil(ratio * 10.0) / 10.0), 2
        ),
    }
    notes.append(
        "impulse_scaling.minimum_duty_factor = %.3f, just below the smallest of"
        " any shipped gait (`%s`, beta = %.3f)"
        % (duty_clamp, worst_gait, worst_duty)
    )
    notes.append(
        "impulse_scaling.maximum_weight_compensation_ratio covers the binding gait `%s`,"
        " which asks for %.2f x weight compensation" % (ratio_gait, ratio)
    )

    # The stance feet together carry the whole weight, so their friction budget is mu * W in total and mu * W / n each;
    # the clamp is on the total and is shared like the weight, so a fraction of mu * W is the same fraction of every
    # foot's budget, and the reference stays inside the cone the wrench-cone barrier enforces rather than fighting it.
    # The largest centripetal force the command can ask for, m v psidot at the largest speed and yaw rate, per weight.
    cmd_max_wnorm_force_com_ewrt_world_xy = (
        geometry.total_mass
        * cmd_max_linvel_base_rt_world_ewrt_heading_x
        * cmd_max_angvel_base_rt_world_ewrt_world_z
        / geometry.total_weight
    )
    # max_force_com_ewrt_world_xy is left out (it has no 0 "off"): the ratio of weight below is the clamp.
    out["centripetal_acceleration_parameters"] = {
        "scale": 1.0,
        "max_wnorm_force_com_ewrt_world_xy": round(
            CENTRIPETAL_FRICTION_FRACTION * friction, 3
        ),
    }
    notes.append(
        "centripetal_acceleration_parameters.max_wnorm_force_com_ewrt_world_xy = %.0f%% of the"
        " mu = %.2f friction budget (%s) = %.3f W (%.0f N); the largest"
        " legitimate demand is m*v*psidot = %.0f N = %.3f W"
        % (
            100 * CENTRIPETAL_FRICTION_FRACTION,
            friction,
            friction_source or "assumed",
            CENTRIPETAL_FRICTION_FRACTION * friction,
            CENTRIPETAL_FRICTION_FRACTION * friction * geometry.total_weight,
            cmd_max_wnorm_force_com_ewrt_world_xy * geometry.total_weight,
            cmd_max_wnorm_force_com_ewrt_world_xy,
        )
    )
    # LINT.ThenChange(//tools/locomotion_heuristics/test_derive_parameters.py:expected_heuristic_names, //legged_nmpc/legged_mpc_config/locomotion_heuristics_config.proto:locomotion_heuristic_keys)
    return out, notes


# ---------------------------------------------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------------------------------------------


def format_block(parameters: dict[str, dict[str, float]]) -> str:
    """The parameter blocks as textproto, in the order the task files and knownHeuristicNames() list them.

    Args:
      parameters: The coefficients of each heuristic (derive_parameters()).

    Returns:
      The blocks, `  <heuristic> {` ... `  }` each, indented to paste into the task file's
      `locomotion_heuristics { ... }`.
    """
    lines: list[str] = []
    for name, values in parameters.items():
        lines.append("  %s {" % name)
        for key, value in values.items():
            # Integers are printed with a decimal point, so that a coefficient of exactly 1 does not read as a count.
            # Ten significant digits, so that a printed block pasted into the task file reads back as exactly the
            # derived value and `--check` finds no difference (%g kept six, and turned 2*pi into a false positive).
            text = "%.10g" % value if value != int(value) else ("%.1f" % value)
            lines.append("    %s: %s" % (key, text))
        lines.append("  }")
    return "\n".join(lines)


def _normalized_cost_name(name: str) -> str:
    """A cost name as MpcFormulationConfig.cpp compares it: lower case, without '_', '-' or ' '."""
    return "".join(c for c in name.lower() if c not in "_- ")


def lists_com_and_acom_tracking(task: dict) -> bool:
    """Whether the task file's `costs` list names the CoM + ACoM tracking cost (which zeroes the base-pose weights)."""
    costs = as_list(task.get("costs"))
    wanted = _normalized_cost_name(COM_AND_ACOM_TRACKING_COST)
    return any(
        isinstance(cost, str) and _normalized_cost_name(cost) == wanted
        for cost in costs
    )


def _snake_case(name: str) -> str:
    """A field name as the strict parser matches a retired one: snake_case, whatever case it was written in."""
    return re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", name).lower()


def carries(task: dict, retired: str) -> bool:
    """Whether the task file carries the retired field `retired`, in either spelling (the MPC refuses both)."""
    return any(_snake_case(name) == retired for name in task)


def formulation_warnings(task: dict) -> list[str]:
    """The report lines about formulation settings that decide whether the heuristics reach the solver at all.

    They are reported because the failure is silent: the block is read and the layer is built either way.

    Args:
      task: The task file.

    Returns:
      The lines, ready to print; empty when nothing stands in the heuristics' way.
    """
    lines: list[str] = []
    if carries(task, RETIRED_ACOM_KEY):
        lines.append(
            "# ERROR   %s is a retired key: the MPC refuses this task file at"
            " start-up. List %s under costs"
            % (RETIRED_ACOM_KEY, COM_AND_ACOM_TRACKING_COST)
        )
        lines.append("#         instead where it was true, and delete the key.")
    if lists_com_and_acom_tracking(task):
        lines.append(
            "# WARNING costs lists %s, so the cost factory zeroes the base-pose"
            " weights and the three" % COM_AND_ACOM_TRACKING_COST
        )
        lines.append(
            "#         base_pose heuristics are INERT on this robot as shipped."
        )
    if carries(task, RETIRED_CONTACT_PLANNING_KEY):
        lines.append(
            "# ERROR   %s is a retired key: the MPC refuses this task file at"
            " start-up. Write" % RETIRED_CONTACT_PLANNING_KEY
        )
        lines.append(
            '#         %s: "%s" where it was true, and delete the key.'
            % (CONTACT_SCHEDULE_SOURCE_KEY, CONTACT_PLANNER_SCHEDULE_SOURCE)
        )
    if task.get(CONTACT_SCHEDULE_SOURCE_KEY) == CONTACT_PLANNER_SCHEDULE_SOURCE:
        lines.append(
            "# WARNING %s is %s, so the foothold heuristics are REJECTED at"
            " start-up: the planner"
            % (CONTACT_SCHEDULE_SOURCE_KEY, CONTACT_PLANNER_SCHEDULE_SOURCE)
        )
        lines.append("#         supplies footholds itself.")
    if carries(task, RETIRED_DCM_TERMINAL_COST_KEY):
        lines.append(
            "# ERROR   %s is a retired key: the MPC refuses this task file at"
            " start-up. List %s under costs"
            % (RETIRED_DCM_TERMINAL_COST_KEY, DCM_TERMINAL_COST)
        )
        lines.append(
            "#         in place of terminal_cost where it was true, and delete"
            " the key."
        )
    if carries(task, RETIRED_BASIS_KEY):
        lines.append(
            "# ERROR   %s is a retired key: the MPC refuses this task file at"
            " start-up. Write" % RETIRED_BASIS_KEY
        )
        lines.append(
            '#         %s: "%s" where it was true, and delete the key.'
            % (CONTACT_INPUT_PARAMETERIZATION_KEY, BASIS_VECTOR_CONTACT_INPUTS)
        )
    if (
        task.get(CONTACT_INPUT_PARAMETERIZATION_KEY)
        == BASIS_VECTOR_CONTACT_INPUTS
    ):
        lines.append(
            "# NOTE    %s is %s, so centripetal_acceleration - the one"
            " heuristic whose"
            % (CONTACT_INPUT_PARAMETERIZATION_KEY, BASIS_VECTOR_CONTACT_INPUTS)
        )
        lines.append(
            "#         force is horizontal - puts the contact-force reference"
            " on the forward-kinematics path."
        )
    weights = (task.get("task_space_foot_cost", {}) or {}).get(
        "weights", {}
    ) or {}
    if (
        float(weights.get("pos_x", 0.0)) == 0.0
        and float(weights.get("pos_y", 0.0)) == 0.0
    ):
        lines.append(
            "# NOTE    task_space_foot_cost.weights.pos_x and pos_y are both 0,"
            " so a foothold heuristic's landing"
        )
        lines.append(
            "#         target is computed and then multiplied by zero. Raise"
            " them before sweeping one."
        )
    return lines


# LINT.IfChange(parameter_block_names)
#: The heuristics whose parameter block is not named after them alone: their names hold a frame-dependent quantity
#: word (legged_nmpc/docs/frame_naming/README.md), so their blocks take `_parameters`.
_SUFFIXED_PARAMETER_BLOCKS = frozenset(
    {
        "orientation_compensation",
        "periodic_orientation",
        "centripetal_acceleration",
    }
)
# LINT.ThenChange(//legged_nmpc/legged_common_mpc/src/locomotion_heuristics/LocomotionHeuristicFormulation.cpp:parameter_block_names)


def parameter_block_name(name: str) -> str:
    """The `locomotion_heuristics` field that holds the parameters of the heuristic `name`, as parameterBlockName() in C++.

    Args:
      name: A heuristic's registry name.

    Returns:
      The name, with `_parameters` where it holds a quantity word.
    """
    return f"{name}_parameters" if name in _SUFFIXED_PARAMETER_BLOCKS else name


def shipped_parameters(task: dict) -> dict[str, dict[str, float]]:
    """The heuristic blocks of the task file's `locomotion_heuristics`, by block name, as floats."""
    block = task.get("locomotion_heuristics", {}) or {}
    return {
        name: {key: float(value) for key, value in (values or {}).items()}
        for name, values in block.items()
        if name
        not in (
            "base_pose_heuristics",
            "foothold_heuristics",
            "wrench_heuristics",
        )
        and isinstance(values, dict)
    }


def compare(
    derived: dict[str, dict[str, float]], shipped: dict[str, dict[str, float]]
) -> int:
    """Prints every coefficient whose shipped value differs from the derivation, or that the task file lacks.

    NOT a test: a coefficient that has been swept in simulation SHOULD differ from its starting point.
    This exists so that a gait or a geometry moving out from under a number is
    noticed, and so that the differences can be read as a list of what has been tuned.

    Args:
      derived: The derived coefficients (derive_parameters).
      shipped: The task file's coefficients (shipped_parameters).

    Returns:
      The number of differences.
    """
    differences = 0
    for name, values in derived.items():
        for key, value in values.items():
            current = shipped.get(name, {}).get(key)
            if current is None:
                print(
                    "  %-28s %-26s absent from the task file (derived %g)"
                    % (name, key, value)
                )
                differences += 1
            elif abs(current - value) > 1e-6:
                print(
                    "  %-28s %-26s shipped %-14.10g derived %.10g"
                    % (name, key, current, value)
                )
                differences += 1
    return differences


def main() -> int:
    """Prints the derived block, or with --check its differences from the shipped one; returns the exit status."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--robot",
        choices=sorted(ROBOTS),
        help="a robot of robot_models/, which supplies every path",
    )
    parser.add_argument(
        "--task-file", help="config/mpc/task.textproto (overrides --robot)"
    )
    parser.add_argument("--urdf", help="the robot's URDF (overrides --robot)")
    parser.add_argument(
        "--reference-file",
        help="config/command/reference.textproto (overrides --robot)",
    )
    parser.add_argument(
        "--planning-file",
        help="config/mpc/contact_planning.textproto (overrides --robot)",
    )
    parser.add_argument(
        "--gait",
        default="trot",
        help=(
            "the gait of the gait file the stepping coefficients are sized for"
            " (default: trot). The step period is what sets them, so a robot"
            " walking with another gait needs them re-derived."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "compare the shipped block against the derivation and report the"
            " differences instead of printing it"
        ),
    )
    args = parser.parse_args()

    if not args.robot and not (args.task_file and args.urdf):
        parser.error("give --robot, or both --task-file and --urdf")

    paths = ROBOTS.get(args.robot, {})
    mpc_dir = os.path.join(REPO_ROOT, paths.get("mpc", "")) if paths else ""
    task_file = args.task_file or os.path.join(
        mpc_dir, "config/mpc/task.textproto"
    )
    urdf_file = args.urdf or os.path.join(REPO_ROOT, paths.get("urdf", ""))
    reference_file = args.reference_file or os.path.join(
        mpc_dir, "config/command/reference.textproto"
    )
    planning_file = args.planning_file or os.path.join(
        mpc_dir, CONTACT_PLANNING_FILE
    )
    gait_file = os.path.join(REPO_ROOT, GAIT_FILE)

    for label, path in (
        ("task file", task_file),
        ("URDF", urdf_file),
        ("reference file", reference_file),
    ):
        if not os.path.isfile(path):
            print("%s not found: %s" % (label, path), file=sys.stderr)
            return 1

    try:
        task = load_textproto(task_file)
        limits = load_textproto(reference_file)
        planning = (
            load_textproto(planning_file)
            if os.path.isfile(planning_file)
            else {}
        )
        gaits = gait_table(load_textproto(gait_file))
    except (OSError, textproto_document.DocumentError) as error:
        print(error, file=sys.stderr)
        return 1
    model_settings = task.get("model_settings", {}) or {}

    geometry = derive_geometry(urdf_file, task, model_settings)
    cadence = gait_cadence(gaits, args.gait)
    parameters, notes = derive_parameters(
        geometry, cadence, limits, task, planning, gaits
    )

    print(
        "# Robot:      %s"
        % (
            args.robot
            or os.path.basename(os.path.dirname(os.path.dirname(task_file)))
        )
    )
    print("# URDF:       %s" % os.path.relpath(urdf_file, REPO_ROOT))
    print("# Task file:  %s" % os.path.relpath(task_file, REPO_ROOT))
    print("#")
    print("# From the model, at the nominal posture of `initial_state`:")
    print(
        "#   total mass            %.3f kg  (weight %.1f N)"
        % (geometry.total_mass, geometry.total_weight)
    )
    print(
        "#   CoM above the feet    %.4f m"
        % geometry.pos_com_rt_feetmean_ewrt_world_z
    )
    for index, ((x, y), name) in enumerate(
        zip(geometry.pos_hips_rt_base_ewrt_heading_xy, geometry.hip_joint_names)
    ):
        print(
            "#   hip %d (%-12s) (%+.4f, %+.4f) m in the base frame"
            % (index, name, x, y)
        )
    print(
        "#   nominal stance        %.4f m between the contact frames"
        % geometry.nom_norm_pos_lfoot_rt_rfoot_ewrt_heading_y
    )
    print("#")
    print(
        "# Gait `%s`: stride %.2f s, step %.2f s, duty factor %.3f, %s"
        % (
            cadence.name,
            cadence.stride_duration,
            cadence.step_duration,
            cadence.duty_factor,
            (
                "with double support"
                if cadence.has_double_support
                else "no double support"
            ),
        )
    )
    print("#")
    for note in notes:
        print("# %s" % note)
    print("#")

    # The formulation settings that decide whether any of this reaches the solver.
    for line in formulation_warnings(task):
        print(line)

    if args.check:
        print()
        print(
            "# Differences between the shipped block and this derivation (a"
            " swept coefficient SHOULD differ):"
        )
        differences = compare(parameters, shipped_parameters(task))
        print("#   %d difference(s)." % differences)
        return 0

    print()
    print(format_block(parameters))
    return 0


if __name__ == "__main__":
    sys.exit(main())
