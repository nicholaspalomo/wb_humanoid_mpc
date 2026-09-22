#!/usr/bin/env python3
"""Derives a robot's `locomotion_heuristics` parameter block from its URDF, its task file and its gait schedule.

The coefficients of Bledt's regularization heuristics (humanoid_nmpc/docs/locomotion_heuristics/README.md) are not
free numbers: most of them follow from the robot's mass and geometry, from the cadence of the gait it is walking
with, and from the command limits the operator can reach. This script does that arithmetic, so that the block in a
robot's `config/mpc/task.yaml` can be regenerated rather than maintained by hand, and so that the derivation of every
number is visible next to the number.

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
  * Sized from the command limits: `orientation_compensation.pitchPerForwardVelocity`,
    `height_compensation.heightPerSpeed`, `translational_stepping`, `in_place_turning`. There is a real choice in
    these - how much lean, how much crouch - so the script takes it from an explicit, printed assumption.
  * Genuinely fitted, and left at zero: the rest of Table C.2. They are what Bledt's extraction framework produces
    from data (chapter 4), and a number invented here would be a guess wearing a derivation's clothes.
"""

import argparse
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

try:
    import pinocchio as pin
except ImportError:  # pragma: no cover - the dev container always has it
    sys.exit(
        "pinocchio is not importable. Run this inside the dev container:\n"
        "  docker compose exec app python3 tools/locomotion_heuristics/derive_parameters.py --robot drc_atlas"
    )

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

#: Where a robot keeps its files, by the name used with --robot. Every centroidal MPC package in robot_models is
#: listed; the whole-body ones are deliberately absent, because WBMpcInterface never builds the heuristic layer.
#:
#: NOT a Bazel target. Pinocchio reaches Python through the ROS install (`/opt/ros/jazzy`, Python 3.12) while Bazel's
#: toolchain is a hermetic 3.11 that cannot see it, so this runs under the dev container's system interpreter like
#: tools/hooks/format_code.py does. `make test-heuristic-parameters` is the entry point.
# LINT.IfChange(derive_parameters_robots)
ROBOTS: Dict[str, Dict[str, str]] = {
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
# LINT.ThenChange(//humanoid_nmpc/docs/locomotion_heuristics/README.md:derive_parameters_usage)

GRAVITY = 9.81

# ---------------------------------------------------------------------------------------------------------------
# Assumptions the sized-from-the-command-limits group rests on. They are constants HERE, and printed with the
# result, rather than buried in the expressions, because they are the only judgement calls in the script and the
# first thing anyone should disagree with.
# ---------------------------------------------------------------------------------------------------------------

#: [rad] How far the base leans into the command at the maximum forward stick. A few degrees carries the centre of
#: mass towards the leading edge of the support and keeps the body inside the small-roll-and-pitch approximation the
#: control model's yaw-only rotation is built on (dissertation section 4.3).
PITCH_AT_MAX_FORWARD_SPEED = math.radians(2.8)

#: [m] How far the base is lowered at the maximum forward stick. Buys the swing leg vertical room inside the same
#: joint range, and costs knee torque headroom.
CROUCH_AT_MAX_FORWARD_SPEED = 0.03

#: [rad] Pelvic obliquity of the periodic roll: the stance-side hip rises, the swing-side hip drops. Human gait runs
#: 4-8 degrees; this is deliberately conservative.
PELVIC_OBLIQUITY = math.radians(1.7)

#: Fraction of the tilt clamp the reference is allowed to use, i.e. how much of the small-angle assumption of the
#: yaw-only rotation matrix the base-pose reference may spend.
MAXIMUM_TILT = math.radians(8.6)

#: Fraction of one step along the heading the capture point may consume.
CAPTURE_POINT_STEP_FRACTION = 0.4

#: Fraction of a single stance foot's friction budget the centripetal force reference may use.
CENTRIPETAL_FRICTION_FRACTION = 0.4

#: Margin below the smallest duty factor of any shipped gait, for the 1/beta clamp.
DUTY_FACTOR_MARGIN = 0.075


@dataclass
class RobotGeometry:
    """What the URDF says, evaluated at the nominal standing posture of the task file's `initialState`."""

    total_mass: float  # [kg]
    com_height: float  # [m] centre of mass above the mean foot height
    hip_offset: List[Tuple[float, float]] = field(
        default_factory=list
    )  # [m] (x, y) per foot, in the base frame
    hip_joint_names: List[str] = field(default_factory=list)
    #: [m] lateral distance between the CONTACT FRAMES at the nominal posture - the stance the robot actually stands
    #: in, which is wider than the hip joints are apart wherever the legs splay.
    nominal_foot_separation: float = 0.0

    @property
    def total_weight(self) -> float:
        return self.total_mass * GRAVITY

    @property
    def hip_half_width(self) -> float:
        return max(abs(y) for _, y in self.hip_offset) if self.hip_offset else 0.0


@dataclass
class GaitCadence:
    """The timing of one named gait of `humanoid_common_mpc/config/command/gait.yaml`."""

    name: str
    stride_duration: float  # [s] one full left-right cycle
    step_duration: float  # [s] between consecutive touch-downs
    duty_factor: float  # fraction of the stride each foot is on the ground
    has_double_support: bool


def load_yaml(path: str) -> dict:
    with open(path, "r") as handle:
        return yaml.safe_load(handle) or {}


def eigen_matrix(node: dict, rows: int) -> np.ndarray:
    """Reads the `"(i,j)": v` matrix layout the task files use into a dense column."""
    values = np.zeros(rows)
    for key, value in (node or {}).items():
        stripped = key.strip('"').strip("()")
        row = int(stripped.split(",")[0])
        if row < rows:
            values[row] = float(value)
    return values


# ---------------------------------------------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------------------------------------------


def derive_geometry(urdf_path: str, task: dict, model_settings: dict) -> RobotGeometry:
    """The C++ deriveLocomotionHeuristicModelParameters(), in Python and against the same nominal state.

    Mirrors it deliberately, including the walk up the kinematic tree to the last joint before the floating base:
    that joint is the hip however the URDF happens to name it, which is why neither this nor the C++ looks for a name.

    The REDUCED model is what has to be built, not the URDF's own. `model_settings.fixedJointNames` locks joints out
    of the MPC model, and `initialState`'s joint block is written in the reduced model's order - so reading it against
    the full model would assign each value to the wrong joint from the first locked one onwards. On the DRC Atlas that
    is the four wrist joints, and the legs would end up carrying the arms' angles.
    """
    full_model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
    locked = [
        full_model.getJointId(name)
        for name in model_settings.get("fixedJointNames", [])
        if full_model.existJointName(name)
    ]
    model = (
        pin.buildReducedModel(full_model, locked, pin.neutral(full_model))
        if locked
        else full_model
    )
    data = model.createData()

    # The nominal posture. `initialState` is [normalised momentum (6), base pose (6), joints], and its joint block is
    # the crouch the robot stands in - not the URDF's neutral, where the legs are straight.
    # names[0] is "universe" and names[1] the floating base, so the MPC joints start at 2.
    mpc_joints: List[str] = list(model.names[2:])
    state = eigen_matrix(task.get("initialState"), 12 + len(mpc_joints))
    base_pose = state[6:12]

    q = pin.neutral(model)
    q[0:3] = base_pose[0:3]
    quaternion = pin.Quaternion(
        pin.rpy.rpyToMatrix(base_pose[5], base_pose[4], base_pose[3])
    )
    q[3:7] = np.array([quaternion.x, quaternion.y, quaternion.z, quaternion.w])
    for index, name in enumerate(mpc_joints):
        joint = model.getJointId(name)
        if model.joints[joint].nq == 1:
            q[model.joints[joint].idx_q] = state[12 + index]

    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    pin.centerOfMass(model, data, q, False)

    # The contact frames are NOT in the URDF: createPinocchioModel() adds them at run time, one per entry of
    # `contactParentJointNames`, offset from that joint by `contacts.contact_frame_translation`. So the foot position
    # is computed here the same way rather than looked up by name.
    translation = task.get("contacts", {}).get("contact_frame_translation", {}) or {}
    contact_offset = np.array(
        [
            float(translation.get("x", 0.0)),
            float(translation.get("y", 0.0)),
            float(translation.get("z", 0.0)),
        ]
    )
    parent_joints = model_settings.get("contactParentJointNames", [])
    foot_heights = [
        float((data.oMi[model.getJointId(name)] * contact_offset)[2])
        for name in parent_joints
        if model.existJointName(name)
    ]
    if not foot_heights:
        raise SystemExit(
            "none of model_settings.contactParentJointNames %s exist in %s"
            % (parent_joints, os.path.basename(urdf_path))
        )
    mean_foot_height = float(np.mean(foot_heights))

    foot_positions = [
        np.array(data.oMi[model.getJointId(name)] * contact_offset)
        for name in parent_joints
        if model.existJointName(name)
    ]
    geometry = RobotGeometry(
        total_mass=float(pin.computeTotalMass(model)),
        com_height=float(data.com[0][2] - mean_foot_height),
        nominal_foot_separation=(
            float(abs(foot_positions[0][1] - foot_positions[1][1]))
            if len(foot_positions) > 1
            else 0.0
        ),
    )

    base_position = np.array(base_pose[0:2])
    yaw = float(base_pose[3])
    rotation = np.array(
        [[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]]
    )
    for name in model_settings.get("contactParentJointNames", []):
        if not model.existJointName(name):
            geometry.hip_offset.append((0.0, 0.0))
            geometry.hip_joint_names.append("(not found: %s)" % name)
            continue
        joint = model.getJointId(name)
        while (
            model.parents[joint] > 1
        ):  # stop at the first child of the floating base: the hip
            joint = model.parents[joint]
        offset = rotation @ (np.array(data.oMi[joint].translation[0:2]) - base_position)
        geometry.hip_offset.append((float(offset[0]), float(offset[1])))
        geometry.hip_joint_names.append(model.names[joint])
    return geometry


# ---------------------------------------------------------------------------------------------------------------
# Gait
# ---------------------------------------------------------------------------------------------------------------


def gait_cadence(gaits: dict, name: str) -> GaitCadence:
    """Duty factor and step period of one gait, read the way the mode schedule defines them.

    The mode names say which foot is IN CONTACT - `LF` is contact flags {true, false} - which is the thing most
    easily got backwards when reading gait.yaml.
    """
    if name not in gaits:
        raise SystemExit(
            "unknown gait '%s'; gait.yaml lists: %s"
            % (name, ", ".join(gaits.get("list", [])))
        )
    modes = gaits[name]["modeSequence"]
    times = gaits[name]["switchingTimes"]
    stride = float(times[-1]) - float(times[0])
    contact = {"left": 0.0, "right": 0.0}
    touchdowns = 0
    previous_left = None
    for index, mode in enumerate(modes):
        duration = float(times[index + 1]) - float(times[index])
        left = mode in ("LF", "STANCE")
        right = mode in ("RF", "STANCE")
        contact["left"] += duration if left else 0.0
        contact["right"] += duration if right else 0.0
        if previous_left is not None and left and not previous_left:
            touchdowns += 1
        previous_left = left
    duty = min(contact["left"], contact["right"]) / stride if stride > 0 else 1.0
    # Two touch-downs per stride in any alternating gait, so the step period is half the stride. Counting them from
    # the sequence would miscount a stride that begins mid-swing.
    return GaitCadence(
        name=name,
        stride_duration=stride,
        step_duration=stride / 2.0,
        duty_factor=duty,
        has_double_support=any(mode == "STANCE" for mode in modes),
    )


def smallest_duty_factor(gaits: dict) -> Tuple[str, float]:
    """The lowest duty factor over every named gait: what the 1/beta clamp has to stay below."""
    worst_name, worst = "", 1.0
    for name in gaits.get("list", []):
        if name not in gaits:
            continue
        try:
            cadence = gait_cadence(gaits, name)
        except (KeyError, IndexError, ZeroDivisionError):
            continue
        if 0.0 < cadence.duty_factor < worst:
            worst_name, worst = name, cadence.duty_factor
    return worst_name, worst


def largest_double_support_ratio(gaits: dict) -> Tuple[str, float]:
    """The largest W/(F beta) that any shipped gait asks for, as a multiple of instantaneous weight compensation.

    The binding case is always a double-support phase: there the baseline is W/2 while the reference stays at
    W/(F beta), so the ratio is 1/beta. Single support gives 2/(F beta) * (F beta / 2) = 1 whenever beta = 1/2.
    """
    worst_name, worst = "", 1.0
    for name in gaits.get("list", []):
        if name not in gaits:
            continue
        try:
            cadence = gait_cadence(gaits, name)
        except (KeyError, IndexError, ZeroDivisionError):
            continue
        if not cadence.has_double_support or cadence.duty_factor <= 0.0:
            continue
        ratio = 1.0 / cadence.duty_factor
        if ratio > worst:
            worst_name, worst = name, ratio
    return worst_name, worst


# ---------------------------------------------------------------------------------------------------------------
# The coefficients
# ---------------------------------------------------------------------------------------------------------------


def derive_parameters(
    geometry: RobotGeometry,
    cadence: GaitCadence,
    limits: dict,
    task: dict,
    planning: dict,
    gaits: dict,
) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    """Every coefficient, with the one-line derivation of each returned alongside it."""
    notes: List[str] = []
    out: Dict[str, Dict[str, float]] = {}

    max_forward = float(limits.get("maxDisplacementVelocityX", 1.0))
    max_lateral = float(limits.get("maxDisplacementVelocityY", 0.3))
    max_yaw_rate = float(limits.get("maxRotationVelocity", 1.0))
    base_height = float(limits.get("defaultBaseHeight", geometry.com_height))
    friction = float(
        task.get("contacts", {})
        .get("contactWrenchConeSoftConstraint", {})
        .get("frictionCoefficient", 0.5)
    )
    step_width = float(task.get("nominal_foothold", {}).get("stepWidth", 0.0))
    foot_separation = planning.get("contact_planning", {}).get("foot_separation", {})
    # foot_separation.maxStepLength, i.e. the REACHABILITY bound, not hlip.maxStepLength which is the
    # closed-form planner's own clip on the step it emits. With no planner running it is the reach that bounds
    # how far a landing target may be pushed.
    max_step_length = float(foot_separation.get("maxStepLength", 0.0))
    max_step_width = float(foot_separation.get("maxStepWidth", 0.0))
    min_step_width = float(foot_separation.get("minStepWidth", 0.0))
    planner_step_width = float(
        planning.get("contact_planning", {}).get("hlip", {}).get("stepWidth", 0.0)
    )

    # THE PENDULUM LENGTH. Every linear-inverted-pendulum expression in this controller - the DCM terminal cost, the
    # contact planner, and the two heuristics below - should use ONE length, or they disagree about where the robot is
    # heading. So the configured one wins over the model's, and a disagreement between them is reported rather than
    # quietly resolved: it means the controller is reasoning about a pendulum the robot does not have.
    configured_lip = float(
        task.get("dcm_terminal_cost", {}).get("comHeight", 0.0)
    ) or float(
        planning.get("contact_planning", {}).get("shared", {}).get("comHeight", 0.0)
    )
    lip_height = configured_lip if configured_lip > 0 else geometry.com_height
    if (
        configured_lip > 0
        and abs(configured_lip - geometry.com_height) > 0.05 * geometry.com_height
    ):
        notes.append(
            "WARNING the configured LIP height (%.4f m, dcm_terminal_cost.comHeight) and the model's centre of mass "
            "above the feet (%.4f m at initialState) differ by %.0f%%. The configured one is used below so that the "
            "capture point, the DCM terminal cost and the contact planner agree, but one of the two is wrong."
            % (
                configured_lip,
                geometry.com_height,
                100.0 * abs(configured_lip - geometry.com_height) / geometry.com_height,
            )
        )

    # ---- base pose ----
    pitch_per_speed = (
        PITCH_AT_MAX_FORWARD_SPEED / max_forward if max_forward > 0 else 0.0
    )
    out["orientation_compensation"] = {
        # 0, and deliberately: at a CONSTANT sidestep velocity there is no net lateral force to lean against, and
        # what a biped's roll really does in single support is oscillate, which is periodic_orientation's job.
        "rollPerLateralVelocity": 0.0,
        "rollOffset": 0.0,
        "pitchPerForwardVelocity": round(pitch_per_speed, 4),
        "pitchOffset": 0.0,
        "maximumTilt": round(MAXIMUM_TILT, 3),
    }
    notes.append(
        "orientation_compensation.pitchPerForwardVelocity = %.1f deg / %.2f m/s = %.4f rad s/m  "
        "(POSITIVE is nose-down in this Euler-ZYX layout)"
        % (math.degrees(PITCH_AT_MAX_FORWARD_SPEED), max_forward, pitch_per_speed)
    )
    notes.append(
        "orientation_compensation.rollPerLateralVelocity left at 0: no steady lateral lean to fit"
    )

    out["periodic_orientation"] = {
        "rollAmplitude": round(PELVIC_OBLIQUITY, 3),
        "rollPhaseRate": round(2.0 * math.pi, 7),
        # Phase [0, 0.5) is LF = LEFT STANCE, so its midpoint is 0.25; a positive roll raises the left (stance) side;
        # and sin(2*pi*0.25 + 0) is its maximum. So offset 0 already drops the hip on the SWING side. Nothing to fit.
        "rollPhaseOffset": 0.0,
        # 0: pitch oscillates once per STEP and where its extremum falls is a property of this robot's mass
        # distribution, not of the gait's geometry. A sinusoid at the wrong phase is worse than none.
        "pitchAmplitude": 0.0,
        "pitchPhaseRate": round(4.0 * math.pi, 7),
        "pitchPhaseOffset": 0.0,
    }
    notes.append(
        "periodic_orientation.rollAmplitude = %.1f deg of pelvic obliquity; phase 0 peaks at mid LEFT stance"
        % math.degrees(PELVIC_OBLIQUITY)
    )
    notes.append(
        "periodic_orientation.pitchAmplitude left at 0: its phase needs data, not arithmetic"
    )

    crouch_per_speed = (
        -CROUCH_AT_MAX_FORWARD_SPEED / max_forward if max_forward > 0 else 0.0
    )
    out["height_compensation"] = {
        "heightPerSpeedSquared": 0.0,
        "heightPerSpeed": round(crouch_per_speed, 4),
        "heightOffset": 0.0,
        "maximumHeightOffset": round(max(0.05, 1.5 * CROUCH_AT_MAX_FORWARD_SPEED), 3),
    }
    notes.append(
        "height_compensation.heightPerSpeed = -%.3f m / %.2f m/s = %.4f  (%.1f%% of the %.4f m base height)"
        % (
            CROUCH_AT_MAX_FORWARD_SPEED,
            max_forward,
            crouch_per_speed,
            (
                100.0 * CROUCH_AT_MAX_FORWARD_SPEED / base_height
                if base_height > 0
                else 0.0
            ),
            base_height,
        )
    )

    # ---- foothold ----
    # This heuristic REPLACES the stance-foot anchor, so the scale that changes nothing about the nominal geometry is
    # the one reproducing the stance the robot already walks with. Listing it at 1.0 would resize the stance the
    # moment it is switched on, which is a different gait rather than a correction.
    hip_width = 2.0 * geometry.hip_half_width
    lateral_scale = (
        (step_width / hip_width) if (hip_width > 0 and step_width > 0) else 1.0
    )
    out["hip_centered_stepping"] = {
        "lateralScale": round(lateral_scale, 3),
        "longitudinalScale": 1.0,
    }
    notes.append(
        (
            "hip_centered_stepping.lateralScale = nominal_foothold.stepWidth %.3f / hip width %.3f = %.3f%s"
            if step_width > 0
            else "hip_centered_stepping.lateralScale = 1.0: nominal_foothold.stepWidth is %.3f, so there is no stance "
            "to reproduce and the hips' own %.3f m is the width%.0s%s"
        )
        % (
            step_width,
            hip_width,
            lateral_scale,
            (
                "  (%.2f matches the %.3f m stance of the nominal posture, %.2f the planner's %.2f m orbit)"
                % (
                    geometry.nominal_foot_separation / hip_width,
                    geometry.nominal_foot_separation,
                    planner_step_width / hip_width,
                    planner_step_width,
                )
                if planner_step_width > 0 and hip_width > 0
                else ""
            ),
        )
    )
    if abs(geometry.hip_offset[0][0] if geometry.hip_offset else 0.0) < 1e-6:
        notes.append(
            "hip_centered_stepping.longitudinalScale is inert here: the hip joints sit at x = 0"
        )

    # The lateral room is asymmetric: the shipped stance may already be AT maxStepWidth, in which case the only room
    # is inward. Take the tighter of the two bounds so the clamp fits whichever way the step is pushed.
    inward_room = max(step_width - min_step_width, 0.0) if step_width > 0 else 0.0
    outward_room = max(max_step_width - step_width, 0.0) if step_width > 0 else 0.0
    capture_clamp = (
        CAPTURE_POINT_STEP_FRACTION * max_step_length if max_step_length > 0 else 0.25
    )
    out["capture_point"] = {
        "gain": 1.0,
        "comHeightOverride": round(lip_height, 4),
        "gravity": GRAVITY,
        "maximumOffset": round(capture_clamp, 3),
    }
    notes.append(
        "capture_point: omega = sqrt(%.2f / %.4f) = %.3f rad/s, so the offset is %.4f s per m/s of velocity error"
        % (
            GRAVITY,
            lip_height,
            math.sqrt(GRAVITY / lip_height),
            math.sqrt(lip_height / GRAVITY),
        )
    )
    notes.append(
        "capture_point.maximumOffset = %.0f%% of foot_separation.maxStepLength %.2f = %.3f m"
        % (100 * CAPTURE_POINT_STEP_FRACTION, max_step_length, capture_clamp)
    )
    if step_width > 0 and outward_room < capture_clamp:
        notes.append(
            "WARNING the lateral room is asymmetric: the %.2f m nominal stance leaves %.3f m outward to maxStepWidth "
            "%.2f but %.3f m inward to minStepWidth %.2f, so a capture step pushed OUTWARD can exceed the planner's "
            "own bound. Narrow nominal_foothold.stepWidth, or accept that this clamp only fits inward."
            % (step_width, outward_room, max_step_width, inward_room, min_step_width)
        )

    # Raibert's half-stance rule, composed with the anchor rather than duplicating it: the anchor is the stance foot
    # carried forward by the commanded velocity over the time remaining until touch-down, so at mid-swing it supplies
    # half a step and this supplies the other half - putting the target one step length ahead of the stance foot.
    half_step = cadence.step_duration / 2.0
    out["translational_stepping"] = {
        "forwardPerForwardVelocity": round(half_step, 3),
        "forwardOffset": 0.0,
        "lateralPerLateralVelocity": round(half_step, 3),
        "lateralOffset": 0.0,
    }
    notes.append(
        "translational_stepping = half the step period of `%s` (stride %.2f s, step %.2f s) = %.3f s  "
        "-- RESCALE WITH THE GAIT"
        % (cadence.name, cadence.stride_duration, cadence.step_duration, half_step)
    )

    # The same half-stance rule applied to the speed the HIP travels at rather than the body: a foot half the stance
    # width from the centre moves at (stance/2) * psidot under a body yaw rate.
    turn_radius = step_width / 2.0 if step_width > 0 else geometry.hip_half_width
    out["in_place_turning"] = {
        "forwardPerYawRate": round(turn_radius * half_step, 4),
        "forwardOffset": 0.0,
        "lateralPerYawRate": 0.0,
        "lateralOffset": 0.0,
    }
    notes.append(
        "in_place_turning.forwardPerYawRate = (stance/2 = %.3f m) * (half step = %.3f s) = %.4f m s/rad, "
        "i.e. %.3f m per foot at the %.2f rad/s stick"
        % (
            turn_radius,
            half_step,
            turn_radius * half_step,
            turn_radius * half_step * max_yaw_rate,
            max_yaw_rate,
        )
    )

    # Closed form, not fitted: Bledt equation 4.31 places the foot along the resultant of gravity and the centripetal
    # acceleration, which displaces it by (z_com / g) * (pdot x omega). The SAME coefficient on both axes, because
    # the two entries are components of one vector.
    lean_coefficient = lip_height / GRAVITY
    out["high_speed_turning"] = {
        "forwardPerCrossTerm": round(lean_coefficient, 4),
        "forwardOffset": 0.0,
        "lateralPerCrossTerm": round(lean_coefficient, 4),
        "lateralOffset": 0.0,
    }
    notes.append(
        "high_speed_turning = z_com / g = %.4f / %.2f = %.4f s^2 (Bledt eq. 4.31); throws the foot %.3f m outwards "
        "at the (%.2f m/s, %.2f rad/s) stick"
        % (
            lip_height,
            GRAVITY,
            lean_coefficient,
            lean_coefficient * max_forward * max_yaw_rate,
            max_forward,
            max_yaw_rate,
        )
    )

    # ---- wrench ----
    worst_gait, worst_duty = smallest_duty_factor(gaits)
    duty_clamp = max(0.05, round(worst_duty - DUTY_FACTOR_MARGIN, 2))
    ratio_gait, ratio = largest_double_support_ratio(gaits)
    out["impulse_scaling"] = {
        "scale": 1.0,
        "minimumDutyFactor": duty_clamp,
        "maximumForceRatio": round(max(2.0, math.ceil(ratio * 10.0) / 10.0), 2),
    }
    notes.append(
        "impulse_scaling.minimumDutyFactor = %.3f, just below the smallest of any shipped gait (`%s`, beta = %.3f)"
        % (duty_clamp, worst_gait, worst_duty)
    )
    notes.append(
        "impulse_scaling.maximumForceRatio covers the binding gait `%s`, whose double support asks for %.2f x "
        "weight compensation" % (ratio_gait, ratio)
    )

    # A single stance foot carries the whole weight, so its friction budget is mu * W; take a fraction of it so the
    # reference stays inside the cone the wrench-cone barrier enforces rather than fighting it.
    needed = geometry.total_mass * max_forward * max_yaw_rate / geometry.total_weight
    out["centripetal_acceleration"] = {
        "scale": 1.0,
        "maximumForce": 0.0,
        "maximumForceRatioOfWeight": round(CENTRIPETAL_FRICTION_FRACTION * friction, 3),
    }
    notes.append(
        "centripetal_acceleration.maximumForceRatioOfWeight = %.0f%% of the mu = %.2f friction budget = %.3f W "
        "(%.0f N); the largest legitimate demand is m*v*psidot = %.0f N = %.3f W"
        % (
            100 * CENTRIPETAL_FRICTION_FRACTION,
            friction,
            CENTRIPETAL_FRICTION_FRACTION * friction,
            CENTRIPETAL_FRICTION_FRACTION * friction * geometry.total_weight,
            needed * geometry.total_weight,
            needed,
        )
    )
    return out, notes


# ---------------------------------------------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------------------------------------------


def format_block(parameters: Dict[str, Dict[str, float]]) -> str:
    """The parameter blocks as YAML, in the order the task files and knownHeuristicNames() list them."""
    lines: List[str] = []
    for name, values in parameters.items():
        lines.append("  %s:" % name)
        width = max(len(key) for key in values)
        for key, value in values.items():
            # Integers are printed with a decimal point so that a YAML reader keeps them as floats, and so that a
            # coefficient of exactly 1 does not read as a count.
            text = ("%g" % value) if value != int(value) else ("%.1f" % value)
            lines.append("    %s %s" % ((key + ":").ljust(width + 2), text))
    return "\n".join(lines)


def shipped_parameters(task: dict) -> Dict[str, Dict[str, float]]:
    block = task.get("locomotion_heuristics", {}) or {}
    return {
        name: {key: float(value) for key, value in (values or {}).items()}
        for name, values in block.items()
        if name not in ("base_pose", "foothold", "wrench") and isinstance(values, dict)
    }


def compare(
    derived: Dict[str, Dict[str, float]], shipped: Dict[str, Dict[str, float]]
) -> int:
    """Reports every coefficient whose shipped value differs from the derivation. Returns the number of differences.

    NOT a test: a coefficient that has been swept in simulation SHOULD differ from its starting point, and that is
    the whole purpose of sweeping it. This exists so that a gait or a geometry moving out from under a number is
    noticed, and so that the differences can be read as a list of what has been tuned.
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
                    "  %-28s %-26s shipped %-10g derived %g"
                    % (name, key, current, value)
                )
                differences += 1
    return differences


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--robot",
        choices=sorted(ROBOTS),
        help="a robot of robot_models/, which supplies every path",
    )
    parser.add_argument("--task-file", help="config/mpc/task.yaml (overrides --robot)")
    parser.add_argument("--urdf", help="the robot's URDF (overrides --robot)")
    parser.add_argument(
        "--reference-file", help="config/command/reference.yaml (overrides --robot)"
    )
    parser.add_argument(
        "--planning-file", help="config/mpc/contact_planning.yaml (overrides --robot)"
    )
    parser.add_argument(
        "--gait",
        default="trot",
        help="the gait of gait.yaml the stepping coefficients are sized for (default: trot). The step period is what "
        "sets them, so a robot walking with another gait needs them re-derived.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare the shipped block against the derivation and report the differences instead of printing YAML",
    )
    args = parser.parse_args()

    if not args.robot and not (args.task_file and args.urdf):
        parser.error("give --robot, or both --task-file and --urdf")

    paths = ROBOTS.get(args.robot, {})
    mpc_dir = os.path.join(REPO_ROOT, paths.get("mpc", "")) if paths else ""
    task_file = args.task_file or os.path.join(mpc_dir, "config/mpc/task.yaml")
    urdf_file = args.urdf or os.path.join(REPO_ROOT, paths.get("urdf", ""))
    reference_file = args.reference_file or os.path.join(
        mpc_dir, "config/command/reference.yaml"
    )
    planning_file = args.planning_file or os.path.join(
        mpc_dir, "config/mpc/contact_planning.yaml"
    )
    gait_file = os.path.join(
        REPO_ROOT, "humanoid_nmpc/humanoid_common_mpc/config/command/gait.yaml"
    )

    for label, path in (
        ("task file", task_file),
        ("URDF", urdf_file),
        ("reference file", reference_file),
    ):
        if not os.path.isfile(path):
            print("%s not found: %s" % (label, path), file=sys.stderr)
            return 1

    task = load_yaml(task_file)
    limits = load_yaml(reference_file)
    planning = load_yaml(planning_file) if os.path.isfile(planning_file) else {}
    gaits = load_yaml(gait_file)
    model_settings = task.get("model_settings", {})

    geometry = derive_geometry(urdf_file, task, model_settings)
    cadence = gait_cadence(gaits, args.gait)
    parameters, notes = derive_parameters(
        geometry, cadence, limits, task, planning, gaits
    )

    print(
        "# Robot:      %s"
        % (args.robot or os.path.basename(os.path.dirname(os.path.dirname(task_file))))
    )
    print("# URDF:       %s" % os.path.relpath(urdf_file, REPO_ROOT))
    print("# Task file:  %s" % os.path.relpath(task_file, REPO_ROOT))
    print("#")
    print("# From the model, at the nominal posture of `initialState`:")
    print(
        "#   total mass            %.3f kg  (weight %.1f N)"
        % (geometry.total_mass, geometry.total_weight)
    )
    print("#   CoM above the feet    %.4f m" % geometry.com_height)
    for index, ((x, y), name) in enumerate(
        zip(geometry.hip_offset, geometry.hip_joint_names)
    ):
        print(
            "#   hip %d (%-12s) (%+.4f, %+.4f) m in the base frame"
            % (index, name, x, y)
        )
    print(
        "#   nominal stance        %.4f m between the contact frames"
        % geometry.nominal_foot_separation
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

    # The two formulation toggles that decide whether any of this reaches the solver. Reported because the failure is
    # silent: the block is read and the layer is built either way.
    if task.get("useComAndAcomTracking"):
        print(
            "# WARNING useComAndAcomTracking is true, so the cost factory zeroes Q's base-pose block and the three"
        )
        print("#         base_pose heuristics are INERT on this robot as shipped.")
    if task.get("useContactPlanning"):
        print(
            "# WARNING useContactPlanning is true, so the foothold heuristics are REJECTED at start-up: the planner"
        )
        print("#         supplies footholds itself.")
    if task.get("useContactBasisVectorInputs"):
        print(
            "# NOTE    useContactBasisVectorInputs is true, so centripetal_acceleration - the one heuristic whose"
        )
        print(
            "#         force is horizontal - puts the contact-force reference on the forward-kinematics path."
        )
    weights = task.get("task_space_foot_cost_weights", {})
    if (
        float(weights.get("pos_x", 0.0)) == 0.0
        and float(weights.get("pos_y", 0.0)) == 0.0
    ):
        print(
            "# NOTE    task_space_foot_cost_weights.pos_x and pos_y are both 0, so a foothold heuristic's landing"
        )
        print(
            "#         target is computed and then multiplied by zero. Raise them before sweeping one."
        )

    if args.check:
        print()
        print(
            "# Differences between the shipped block and this derivation (a swept coefficient SHOULD differ):"
        )
        differences = compare(parameters, shipped_parameters(task))
        print("#   %d difference(s)." % differences)
        return 0

    print()
    print(format_block(parameters))
    return 0


if __name__ == "__main__":
    sys.exit(main())
