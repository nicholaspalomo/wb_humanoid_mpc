"""****************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
****************************************************************************"""

"""Checks that derive_parameters.py still speaks to every shipped robot, and that what it derives means what it says.

Not a check on the VALUES: a coefficient that has been swept in simulation should differ from its starting point, and
freezing the derivation would make tuning a test failure. What this guards is the machinery underneath and the
PROPERTY each closed-form coefficient claims - that lateralScale reproduces the stance, that every pendulum expression
uses the configured LIP height, that the stepping leads follow the stance rather than one gait, that a printed block
reads back as exactly what was derived, and that every key the script prints is one the C++ loader reads.

Three classes, split by what they need:
  * ShippedConfigurationTest reads the shipped task files and gait table with PyYAML alone: the default-off rule, the
    key tables, the friction cone, the gait cadence and the speed ladder.
  * DerivationPropertiesTest runs derive_parameters() and the output helpers against a synthetic geometry, so no URDF
    is parsed.
  * DeriveParametersTest builds the real models with pinocchio, and is skipped where pinocchio is not importable.
The first two run on any interpreter with PyYAML and numpy (the script imports numpy). That is what keeps the
default-off guard for SA01, G1 and R1 alive outside the dev container, where the third class is skipped.
"""

import contextlib
import copy
import glob
import importlib
import importlib.util
import io
import math
import os
import re
import sys
import types
import unittest

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPT = os.path.join(
    REPO_ROOT, "tools", "locomotion_heuristics", "derive_parameters.py"
)
GAIT_FILE = os.path.join(
    REPO_ROOT, "humanoid_nmpc/humanoid_common_mpc/config/command/gait.yaml"
)
LOADER_SOURCE = os.path.join(
    REPO_ROOT,
    "humanoid_nmpc/humanoid_common_mpc/src/locomotion_heuristics/LocomotionHeuristicConfig.cpp",
)
MOTION_MANAGER_HEADER = os.path.join(
    REPO_ROOT,
    "humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/reference_manager/ProceduralMpcMotionManager.h",
)

# The names of Bledt's Appendix C by list, which is also what LocomotionHeuristicFormulation registers.
# LINT.IfChange(expected_heuristic_names)
EXPECTED_HEURISTIC_NAMES = {
    "base_pose": (
        "orientation_compensation",
        "periodic_orientation",
        "height_compensation",
    ),
    "foothold": (
        "hip_centered_stepping",
        "capture_point",
        "translational_stepping",
        "in_place_turning",
        "high_speed_turning",
    ),
    "wrench": ("impulse_scaling", "centripetal_acceleration"),
}
# LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/src/locomotion_heuristics/LocomotionHeuristicFormulation.cpp:known_heuristic_names)

ALL_HEURISTIC_NAMES = tuple(
    name for names in EXPECTED_HEURISTIC_NAMES.values() for name in names
)

#: The two friction cones a task file can list in `soft_constraints`, and the `contacts` block each one reads.
FRICTION_CONES = (
    ("contact_wrench_cone", "contactWrenchConeSoftConstraint"),
    ("friction_force_cone", "frictionForceConeSoftConstraint"),
)

#: The walking gaits of ProceduralMpcMotionManager::gaitModeStates_, slowest first. The motion manager moves along
#: this ladder with the commanded speed, so a stepping coefficient has to be right on every rung, not only on `trot`.
#: A copy of the C++ table, which test_speed_ladder_is_the_motion_managers_and_every_rung_is_a_gait reads back out of
#: the header: a rung added there fails that test until it is added here.
SPEED_LADDER = ("slow_walk", "walk", "slower_trot", "slow_trot", "trot", "run")

#: Which feet each mode name of gait.yaml puts IN CONTACT, as (left, right). The names say which foot is DOWN - `LF`
#: is left stance, right swing - and FLY carries neither. The test's own reading, independent of the script's.
MODE_CONTACTS = {
    "LF": (True, False),
    "RF": (False, True),
    "STANCE": (True, True),
    "FLY": (False, False),
}

#: Half-width of the round-to-4-decimals the script applies to most coefficients, plus float slack.
ROUNDING_4 = 0.5e-4 + 1e-12

#: Half-width of the round-to-3-decimals the script applies to the clamps, plus float slack.
ROUNDING_3 = 0.5e-3 + 1e-12

#: How far below the smallest duty factor of any gait the 1/beta clamp may sit and still be the "just below it" the
#: script's note claims. The script's margin plus its two-decimal rounding fits inside it with room to retune the
#: margin; a margin that has run down to the 0.05 floor, which lets 1/beta reach 20 as a flight phase opens, does not.
DUTY_CLAMP_HEADROOM = 0.1

#: [m] The capture-point clamp the script falls back to for a robot with no contact_planning.yaml, and so no reach
#: bound to take a fraction of (G1 and R1). A fixed constant of the script rather than a derived number.
CAPTURE_POINT_FALLBACK_CLAMP = 0.25

#: The blend gains and scales that multiply a heuristic's whole effect. validate() admits 0 for each, and at 0 the
#: listed heuristic does nothing; the derivation's starting point is the heuristic at full strength.
UNIT_GAINS = (
    "capture_point.gain",
    "impulse_scaling.scale",
    "centripetal_acceleration.scale",
    "hip_centered_stepping.longitudinalScale",
)


def _pinocchio_available():
    """Whether the real pinocchio can be imported here."""
    try:
        importlib.import_module("pinocchio")
    except ImportError:
        return False
    return True


def _load_script(have_pinocchio):
    """Loads derive_parameters.py as a module, with or without pinocchio.

    The script exits at import when pinocchio is missing. Only derive_geometry() uses it, and only DeriveParametersTest
    calls that, so without pinocchio an empty placeholder module stands in while the script executes and is removed
    again afterwards. Everything else the script does needs PyYAML alone.
    """
    spec = importlib.util.spec_from_file_location("derive_parameters", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    if have_pinocchio:
        spec.loader.exec_module(module)
        return module
    missing = object()
    previous = sys.modules.get("pinocchio", missing)
    sys.modules["pinocchio"] = types.ModuleType("pinocchio")
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is missing:
            del sys.modules["pinocchio"]
        else:
            sys.modules["pinocchio"] = previous
    return module


HAVE_PINOCCHIO = _pinocchio_available()
derive = _load_script(HAVE_PINOCCHIO)


def _read_yaml(path):
    """A YAML file, read with PyYAML directly rather than through the script under test."""
    with open(path, "r") as handle:
        return yaml.safe_load(handle) or {}


def _read_text(path):
    with open(path, "r") as handle:
        return handle.read()


def _task_file(robot):
    return os.path.join(REPO_ROOT, derive.ROBOTS[robot]["mpc"], "config/mpc/task.yaml")


def _robot_inputs(robot):
    """(task, limits, planning) of one robot of derive.ROBOTS, as main() assembles them."""
    mpc_dir = os.path.join(REPO_ROOT, derive.ROBOTS[robot]["mpc"])
    task = _read_yaml(os.path.join(mpc_dir, "config/mpc/task.yaml"))
    limits = _read_yaml(os.path.join(mpc_dir, "config/command/reference.yaml"))
    planning_file = os.path.join(mpc_dir, "config/mpc/contact_planning.yaml")
    planning = _read_yaml(planning_file) if os.path.isfile(planning_file) else {}
    return task, limits, planning


def _step_width(task):
    return float((task.get("nominal_foothold") or {}).get("stepWidth", 0.0))


def _configured_lip_height(task, planning):
    """The LIP length the robot's files configure: the DCM terminal cost's, else the contact planner's, else 0."""
    dcm = float((task.get("dcm_terminal_cost") or {}).get("comHeight", 0.0))
    if dcm > 0.0:
        return dcm
    shared = (planning.get("contact_planning") or {}).get("shared") or {}
    return float(shared.get("comHeight", 0.0))


def _hip_width(geometry):
    """Distance between the two hips, from the hip offsets themselves."""
    return 2.0 * max(abs(y) for _, y in geometry.hip_offset)


def _listed_names(block):
    """The names each of the three lists of a `locomotion_heuristics` block turns on; a bare key reads as none."""
    return {kind: list(block.get(kind) or []) for kind in EXPECTED_HEURISTIC_NAMES}


def _commented_list_entries(lines, kind):
    """(line index, name) of every `# - name` line directly beneath `  <kind>:` in the locomotion_heuristics block."""
    start = next(
        index
        for index, line in enumerate(lines)
        if line.rstrip() == "locomotion_heuristics:"
    )
    header = re.compile(r"^  %s:\s*(#.*)?$" % kind)
    entry = re.compile(r"^\s+#\s*-\s+([A-Za-z_]+)")
    for index in range(start + 1, len(lines)):
        if header.match(lines[index]):
            entries = []
            for following in range(index + 1, len(lines)):
                match = entry.match(lines[following])
                if not match:
                    break
                entries.append((following, match.group(1)))
            return entries
    return []


def _loader_table():
    """The C++ loader's key table: every "<name>.<key>" string of coefficientKeys(), and the number of macro calls."""
    source = _read_text(LOADER_SOURCE)
    start = source.index("LINT.IfChange(locomotion_heuristic_keys)")
    table = source[start : source.index("LINT.ThenChange", start)]
    keys = re.findall(r'HEURISTIC_COEFFICIENT\(\s*"([^"]+)"', table)
    calls = len(re.findall(r"HEURISTIC_COEFFICIENT\(", table))
    return keys, calls


def _validate_range_checks():
    """The range checks of LocomotionHeuristicConfig::validate().

    Returns the keys it passes to requirePositive, the keys it passes to requireNonNegative, and the number of calls to
    either. The key patterns only match a plain first argument and a string-literal key, so a call such as
    `requireNonNegative(std::abs(x), "...")` is not parsed; the call count is what notices that, the same guard
    _loader_table() keeps on the key table.
    """
    source = _read_text(LOADER_SOURCE)
    start = source.index("LocomotionHeuristicConfig::validate() const {")
    body = source[start : source.index("\n}\n", start)]
    positive = re.findall(r'requirePositive\(\s*[^,()]+,\s*"([^"]+)"\)', body)
    non_negative = re.findall(r'requireNonNegative\(\s*[^,()]+,\s*"([^"]+)"\)', body)
    calls = len(re.findall(r"\brequire(?:Positive|NonNegative)\(", body))
    return positive, non_negative, calls


def _motion_manager_gait_ladder():
    """The gait names of ProceduralMpcMotionManager::gaitModeStates_, in order, and the number of rows it has."""
    source = _read_text(MOTION_MANAGER_HEADER)
    match = re.search(r"gaitModeStates_\s*\{(.*?)\n\s*\};", source, re.DOTALL)
    if match is None:
        return [], 0
    body = match.group(1)
    return re.findall(r'\{\s*"([^"]+)"', body), len(re.findall(r"\{", body))


def _foot_stance_durations(gait):
    """(left, right, stride) [s] of one gait.yaml entry: how long each foot is down over one stride, and the stride.

    Read with MODE_CONTACTS rather than with derive.gait_cadence(), so that it can check the script. A mode name that
    table does not know raises rather than reading as flight.
    """
    modes = gait["modeSequence"]
    times = [float(time) for time in gait["switchingTimes"]]
    if len(times) != len(modes) + 1:
        raise ValueError(
            "%d modes need %d switching times, not %d"
            % (len(modes), len(modes) + 1, len(times))
        )
    left = right = 0.0
    for mode, start, end in zip(modes, times, times[1:]):
        left_down, right_down = MODE_CONTACTS[mode]
        left += (end - start) if left_down else 0.0
        right += (end - start) if right_down else 0.0
    return left, right, times[-1] - times[0]


def _gait_table_extremes(gaits):
    """(smallest duty factor, largest 1/beta of a gait with double support) over the listed gaits, by MODE_CONTACTS.

    Only a foot that bears load has a duty factor: `left_leg` stands on the left foot throughout, so its beta is 1, not
    the 0 of a right foot that never touches down.
    """
    smallest, largest_ratio = 1.0, 1.0
    for name in gaits.get("list", []):
        if name not in gaits:
            continue
        left, right, stride = _foot_stance_durations(gaits[name])
        duty = min(down for down in (left, right) if down > 0.0) / stride
        smallest = min(smallest, duty)
        if "STANCE" in gaits[name]["modeSequence"]:
            largest_ratio = max(largest_ratio, 1.0 / duty)
    return smallest, largest_ratio


def _derived_keys(parameters):
    return {
        "%s.%s" % (name, key) for name, values in parameters.items() for key in values
    }


def _synthetic_geometry():
    """A stand-in for derive_geometry(), for the tests that do not need a URDF.

    The numbers are chosen to be NOT any robot's: the CoM sits 0.70 m above the feet, unlike Atlas's (1.0805 m) or
    SA01's (0.6124 m) model-derived LIP height, and the hips are 0.1786 m apart, unlike Atlas's 0.45 m stance. A
    derivation that reads the wrong one of two lengths therefore lands on a visibly different number.
    """
    return derive.RobotGeometry(
        total_mass=60.0,
        com_height=0.70,
        hip_offset=[(0.02, 0.0893), (0.02, -0.0893)],
        hip_joint_names=["left_hip", "right_hip"],
        nominal_foot_separation=0.23,
    )


def _derive(geometry, task, limits, planning, gait="trot"):
    gaits = _read_yaml(GAIT_FILE)
    return derive.derive_parameters(
        geometry, derive.gait_cadence(gaits, gait), limits, task, planning, gaits
    )


# The ranges LocomotionHeuristicConfig::validate() enforces with requirePositive / requireNonNegative, spelled out so
# that each admissibility assertion names its key. test_admissibility_table_matches_validate keeps it in step with the
# C++, which is how the checks of lateralScale, maximumForce, maximumForceRatioOfWeight, gain and longitudinalScale
# came to be missing before.
MUST_BE_POSITIVE = (
    "orientation_compensation.maximumTilt",
    "capture_point.gravity",
    "capture_point.maximumOffset",
)
MUST_NOT_BE_NEGATIVE = (
    "height_compensation.maximumHeightOffset",
    "capture_point.comHeightOverride",
    "hip_centered_stepping.lateralScale",
    "hip_centered_stepping.longitudinalScale",
    "capture_point.gain",
    "impulse_scaling.scale",
    "centripetal_acceleration.scale",
    "centripetal_acceleration.maximumForce",
    "centripetal_acceleration.maximumForceRatioOfWeight",
)


class _DerivedBlockAssertions:
    """The properties of a derived block that hold for every robot, whichever geometry it was derived from."""

    def assert_admissible(self, parameters):
        """Everything validate() would reject. A derivation the loader refuses is worse than no derivation."""
        for dotted in MUST_BE_POSITIVE:
            name, key = dotted.split(".")
            self.assertGreater(parameters[name][key], 0.0, dotted)
        for dotted in MUST_NOT_BE_NEGATIVE:
            name, key = dotted.split(".")
            self.assertGreaterEqual(parameters[name][key], 0.0, dotted)
        duty = parameters["impulse_scaling"]["minimumDutyFactor"]
        self.assertTrue(
            0.0 < duty <= 1.0, "minimumDutyFactor %s is outside (0, 1]" % duty
        )
        self.assertGreaterEqual(parameters["impulse_scaling"]["maximumForceRatio"], 1.0)
        centripetal = parameters["centripetal_acceleration"]
        self.assertTrue(
            centripetal["maximumForce"] > 0.0
            or centripetal["maximumForceRatioOfWeight"] > 0.0,
            "both centripetal clamps at zero would remove the clamp",
        )

    def assert_lateral_scale_reproduces_the_stance(self, parameters, geometry, task):
        """hip_centered_stepping REPLACES the stance-foot anchor, so the scale must give back the stance it replaces.

        Catches a hip_width / hip_half_width mix-up, which doubles lateralScale (Atlas's 0.45 m stance becomes 0.9 m).
        The tolerance is exactly what rounding lateralScale to three decimals can cost.
        """
        scale = parameters["hip_centered_stepping"]["lateralScale"]
        step_width = _step_width(task)
        hip_width = _hip_width(geometry)
        if step_width > 0.0:
            self.assertLessEqual(
                abs(hip_width * scale - step_width),
                0.5e-3 * hip_width + 1e-12,
                "lateralScale %s spreads hips %.4f m apart to %.4f m, not the %.4f m nominal_foothold.stepWidth"
                % (scale, hip_width, hip_width * scale, step_width),
            )
        else:
            # No stance to reproduce, so the hips' own width is the width.
            self.assertEqual(scale, 1.0)

    def assert_pendulum_is_the_configured_one(
        self, parameters, geometry, task, planning
    ):
        """capture_point and high_speed_turning use ONE LIP height: the configured one, else the model's CoM.

        Catches either heuristic reading the model's CoM height where the task file configures another (they then
        disagree with the DCM terminal cost and the contact planner about where the robot is heading), g / z in place
        of z / g, and the two high_speed_turning axes drifting apart - they are components of one vector.
        """
        lip = _configured_lip_height(task, planning) or geometry.com_height
        capture = parameters["capture_point"]
        self.assertAlmostEqual(capture["comHeightOverride"], lip, delta=ROUNDING_4)
        turning = parameters["high_speed_turning"]
        self.assertEqual(turning["forwardPerCrossTerm"], turning["lateralPerCrossTerm"])
        # Bledt eq. 4.31: the lean is z / g, with the same g the capture point uses.
        self.assertAlmostEqual(
            turning["forwardPerCrossTerm"], lip / capture["gravity"], delta=ROUNDING_4
        )
        self.assertEqual(turning["forwardOffset"], 0.0)
        self.assertEqual(turning["lateralOffset"], 0.0)

    def assert_stepping_follows_the_stance(self, parameters, geometry, task):
        """Raibert's lead, T_stance / 2, carried by the stance-duration terms rather than by one gait's constant.

        Catches a return of `step_duration / 2` as a velocity gain (HC2: right on `trot` only, and half the lead
        slow_walk needs), and in_place_turning's lever drifting from half the arm r the foot turns about.
        """
        stepping = parameters["translational_stepping"]
        self.assertEqual(stepping["forwardStanceFraction"], 0.5)
        self.assertEqual(stepping["lateralStanceFraction"], 0.5)
        for key in (
            "forwardPerForwardVelocity",
            "forwardOffset",
            "lateralPerLateralVelocity",
            "lateralOffset",
        ):
            self.assertEqual(stepping[key], 0.0, "translational_stepping.%s" % key)

        step_width = _step_width(task)
        lever = step_width / 2.0 if step_width > 0.0 else _hip_width(geometry) / 2.0
        turning = parameters["in_place_turning"]
        self.assertAlmostEqual(
            turning["forwardStanceLever"], lever / 2.0, delta=ROUNDING_4
        )
        for key in (
            "forwardPerYawRate",
            "forwardOffset",
            "lateralPerYawRate",
            "lateralOffset",
        ):
            self.assertEqual(turning[key], 0.0, "in_place_turning.%s" % key)


class FormulationWarningsTest(unittest.TestCase):
    """What the report says about the settings that decide whether a heuristic reaches the solver. Never skipped."""

    MPC_FORMULATION_CONFIG = os.path.join(
        REPO_ROOT,
        "humanoid_nmpc/humanoid_common_mpc/src/common/MpcFormulationConfig.cpp",
    )

    @staticmethod
    def _mentions(lines, *phrases):
        return any(all(phrase in line for phrase in phrases) for line in lines)

    def test_the_cost_name_is_the_one_the_cpp_registry_prints(self):
        # The report recognizes the cost by the name the C++ loader reads; a rename there must not silently stop it.
        source = _read_text(self.MPC_FORMULATION_CONFIG)
        match = re.search(
            r"case MpcCostType::ComAndAcomTrackingCost:\s*return \"([a-z_]+)\";",
            source,
        )
        self.assertIsNotNone(match, "the C++ registry has no ComAndAcomTrackingCost")
        self.assertEqual(derive.COM_AND_ACOM_TRACKING_COST, match.group(1))

    def test_the_acom_warning_follows_the_costs_list(self):
        without = {"costs": ["state_quadratic_cost", "input_quadratic_cost"]}
        self.assertFalse(derive.lists_com_and_acom_tracking(without))
        self.assertFalse(self._mentions(derive.formulation_warnings(without), "INERT"))
        # Positive control, in both spellings the C++ loader accepts.
        for spelling in ("com_and_acom_tracking_cost", "comAndAcomTrackingCost"):
            with self.subTest(spelling=spelling):
                listed = {"costs": ["state_quadratic_cost", spelling]}
                self.assertTrue(derive.lists_com_and_acom_tracking(listed))
                self.assertTrue(
                    self._mentions(
                        derive.formulation_warnings(listed),
                        "com_and_acom_tracking_cost",
                        "base-pose",
                    )
                )

    def test_the_retired_boolean_is_reported_whatever_its_value_and_switches_nothing(
        self,
    ):
        # The MPC refuses a task file that still carries useComAndAcomTracking, so the report says so, and the key no
        # longer counts as the ACoM cost being listed - even when it is true.
        for value in (True, False):
            with self.subTest(value=value):
                task = {"useComAndAcomTracking": value, "costs": []}
                warnings = derive.formulation_warnings(task)
                self.assertTrue(
                    self._mentions(
                        warnings, "useComAndAcomTracking", "retired", "refuses"
                    )
                )
                self.assertFalse(derive.lists_com_and_acom_tracking(task))
                self.assertFalse(self._mentions(warnings, "INERT"))

    def test_the_shipped_files_carry_no_retired_key_and_atlas_alone_lists_the_cost(
        self,
    ):
        # Atlas's ACoM network is the one validated for closed-loop use; the others must not list the cost
        # (testAcomAngularVelocityConsistency guards the same thing on the C++ side).
        for robot in sorted(derive.ROBOTS):
            with self.subTest(robot=robot):
                task = _read_yaml(_task_file(robot))
                self.assertNotIn(derive.RETIRED_ACOM_KEY, task)
                self.assertEqual(
                    derive.lists_com_and_acom_tracking(task), robot == "drc_atlas"
                )


class ContactInputParameterizationReportTest(unittest.TestCase):
    """The report's note on basis-vector contact inputs follows the named parameterization, not the retired boolean."""

    CONTACT_INPUT_PARAMETERIZATION_HEADER = os.path.join(
        REPO_ROOT,
        "humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/common/ContactInputParameterization.h",
    )

    @staticmethod
    def _mentions(lines, *phrases):
        return any(all(phrase in line for phrase in phrases) for line in lines)

    def test_the_names_are_the_ones_the_cpp_registry_reads(self):
        # A rename on the C++ side must not silently stop the report from recognizing the parameterization.
        source = _read_text(self.CONTACT_INPUT_PARAMETERIZATION_HEADER)
        for constant, expected in (
            (
                "kContactInputParameterizationKey",
                derive.CONTACT_INPUT_PARAMETERIZATION_KEY,
            ),
            (
                "kBasisVectorsContactInputParameterization",
                derive.BASIS_VECTOR_CONTACT_INPUTS,
            ),
            ("kRetiredContactBasisVectorInputsKey", derive.RETIRED_BASIS_KEY),
        ):
            with self.subTest(constant=constant):
                match = re.search(
                    r"%s = \"([A-Za-z_]+)\";" % constant,
                    source,
                )
                self.assertIsNotNone(match, "%s is not in the C++ header" % constant)
                self.assertEqual(expected, match.group(1))

    def test_the_forward_kinematics_note_follows_the_named_parameterization(self):
        for name, noted in (("basis_vectors", True), ("wrench", False)):
            with self.subTest(parameterization=name):
                task = {"contactInputParameterization": name, "costs": []}
                self.assertEqual(
                    self._mentions(
                        derive.formulation_warnings(task),
                        "forward-kinematics path",
                    ),
                    noted,
                )
        # No key is the wrench parameterization.
        self.assertFalse(
            self._mentions(
                derive.formulation_warnings({"costs": []}), "forward-kinematics path"
            )
        )

    def test_the_retired_boolean_is_reported_whatever_its_value_and_notes_nothing(
        self,
    ):
        for value in (True, False):
            with self.subTest(value=value):
                warnings = derive.formulation_warnings(
                    {"useContactBasisVectorInputs": value, "costs": []}
                )
                self.assertTrue(
                    self._mentions(
                        warnings, "useContactBasisVectorInputs", "retired", "refuses"
                    )
                )
                self.assertTrue(
                    self._mentions(
                        warnings, "contactInputParameterization: basis_vectors"
                    )
                )
                self.assertFalse(self._mentions(warnings, "forward-kinematics path"))

    def test_the_shipped_files_carry_no_retired_basis_key(self):
        for robot in sorted(derive.ROBOTS):
            with self.subTest(robot=robot):
                self.assertNotIn(
                    derive.RETIRED_BASIS_KEY, _read_yaml(_task_file(robot))
                )


class ContactScheduleSourceReportTest(unittest.TestCase):
    """The report's foothold warning follows the named contact schedule source, not the retired boolean."""

    MPC_FORMULATION_CONFIG_HEADER = os.path.join(
        REPO_ROOT,
        "humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/common/MpcFormulationConfig.h",
    )

    @staticmethod
    def _mentions(lines, *phrases):
        return any(all(phrase in line for phrase in phrases) for line in lines)

    def test_the_names_are_the_ones_the_cpp_registry_reads(self):
        # A rename on the C++ side must not silently stop the report from recognizing the planner.
        source = _read_text(self.MPC_FORMULATION_CONFIG_HEADER)
        for constant, expected in (
            ("kContactScheduleSourceKey", derive.CONTACT_SCHEDULE_SOURCE_KEY),
            (
                "kContactPlannerContactScheduleSource",
                derive.CONTACT_PLANNER_SCHEDULE_SOURCE,
            ),
            ("kGaitScheduleContactScheduleSource", derive.GAIT_SCHEDULE_SOURCE),
            ("kRetiredContactPlanningKey", derive.RETIRED_CONTACT_PLANNING_KEY),
        ):
            with self.subTest(constant=constant):
                match = re.search(r"%s = \"([A-Za-z_]+)\";" % constant, source)
                self.assertIsNotNone(match, "%s is not in the C++ header" % constant)
                self.assertEqual(expected, match.group(1))

    def test_the_foothold_warning_follows_the_named_source(self):
        for name, warned in (("contact_planner", True), ("gait_schedule", False)):
            with self.subTest(source=name):
                task = {"contactScheduleSource": name, "costs": []}
                self.assertEqual(
                    self._mentions(derive.formulation_warnings(task), "REJECTED"),
                    warned,
                )
        # No key is the gait schedule.
        self.assertFalse(
            self._mentions(derive.formulation_warnings({"costs": []}), "REJECTED")
        )

    def test_the_retired_boolean_is_reported_whatever_its_value_and_warns_nothing_else(
        self,
    ):
        # The MPC refuses a task file that still carries useContactPlanning, so the report says so, and the key no
        # longer counts as the planner being selected - even when it is true.
        for value in (True, False):
            with self.subTest(value=value):
                warnings = derive.formulation_warnings(
                    {"useContactPlanning": value, "costs": []}
                )
                self.assertTrue(
                    self._mentions(warnings, "useContactPlanning", "retired", "refuses")
                )
                self.assertTrue(
                    self._mentions(warnings, "contactScheduleSource: contact_planner")
                )
                self.assertFalse(self._mentions(warnings, "REJECTED"))

    def test_every_shipped_robot_names_the_gait_schedule(self):
        # The online contact planner changes the closed loop and ships switched off on every robot.
        for robot in sorted(derive.ROBOTS):
            with self.subTest(robot=robot):
                task = _read_yaml(_task_file(robot))
                self.assertNotIn(derive.RETIRED_CONTACT_PLANNING_KEY, task)
                self.assertEqual(
                    task.get(derive.CONTACT_SCHEDULE_SOURCE_KEY),
                    derive.GAIT_SCHEDULE_SOURCE,
                )


class DcmTerminalCostReportTest(unittest.TestCase):
    """The report recognizes the retired DCM terminal-cost boolean by the name the C++ loader refuses."""

    MPC_FORMULATION_CONFIG = os.path.join(
        REPO_ROOT,
        "humanoid_nmpc/humanoid_common_mpc/src/common/MpcFormulationConfig.cpp",
    )

    @staticmethod
    def _mentions(lines, *phrases):
        return any(all(phrase in line for phrase in phrases) for line in lines)

    def test_the_names_are_the_ones_the_cpp_registry_reads(self):
        source = _read_text(self.MPC_FORMULATION_CONFIG)
        cost = re.search(
            r"case MpcCostType::DcmTerminalCost:\s*return \"([a-z_]+)\";", source
        )
        self.assertIsNotNone(cost, "the C++ registry has no DcmTerminalCost")
        self.assertEqual(derive.DCM_TERMINAL_COST, cost.group(1))
        retired = re.search(
            r'\{"([A-Za-z]+)", "costs", "%s"' % derive.DCM_TERMINAL_COST, source
        )
        self.assertIsNotNone(retired, "the C++ loader no longer refuses a retired key")
        self.assertEqual(derive.RETIRED_DCM_TERMINAL_COST_KEY, retired.group(1))

    def test_the_retired_boolean_is_reported_whatever_its_value(self):
        for value in (True, False):
            with self.subTest(value=value):
                warnings = derive.formulation_warnings(
                    {"useDcmTerminalCost": value, "costs": ["terminal_cost"]}
                )
                self.assertTrue(
                    self._mentions(warnings, "useDcmTerminalCost", "retired", "refuses")
                )
                self.assertTrue(self._mentions(warnings, "dcm_terminal_cost"))
        # Positive control: the named cost itself is not an error.
        self.assertFalse(
            self._mentions(
                derive.formulation_warnings({"costs": ["dcm_terminal_cost"]}),
                "retired",
            )
        )

    def test_the_shipped_files_carry_no_retired_key(self):
        for robot in sorted(derive.ROBOTS):
            with self.subTest(robot=robot):
                self.assertNotIn(
                    derive.RETIRED_DCM_TERMINAL_COST_KEY, _read_yaml(_task_file(robot))
                )


class ShippedConfigurationTest(unittest.TestCase):
    """The shipped files, read with PyYAML alone. Never skipped."""

    def test_every_centroidal_package_is_a_robot_of_the_script(self):
        # The default-off guard below loops over derive.ROBOTS, so a centroidal package missing from it would be a robot
        # whose heuristic lists nobody checks. The whole-body packages are deliberately absent (no layer there).
        on_disk = {
            os.path.relpath(path, REPO_ROOT)
            for path in glob.glob(
                os.path.join(REPO_ROOT, "robot_models/*/*_centroidal_mpc")
            )
        }
        self.assertTrue(on_disk, "no centroidal package found; is REPO_ROOT right?")
        listed = {paths["mpc"] for paths in derive.ROBOTS.values()}
        self.assertEqual(listed, on_disk)
        for robot, paths in sorted(derive.ROBOTS.items()):
            with self.subTest(robot=robot):
                self.assertTrue(os.path.isfile(_task_file(robot)))
                self.assertTrue(os.path.isfile(os.path.join(REPO_ROOT, paths["urdf"])))

    def test_every_shipped_robot_keeps_all_three_lists_empty(self):
        """The default-off rule, checked on the files as they ship.

        A heuristic changes the closed loop, and none has been validated in simulation under hardware-like conditions
        on these robots, so every list stays empty until someone opts a robot in deliberately. This is cheap to get
        wrong in a way that is invisible in review - a block of commented-out names is one stray edit away from being
        a block of live ones - and the consequence is a robot that walks differently the next time it is launched.
        It reads the files with PyYAML alone, so it runs wherever Python does rather than only where pinocchio can be
        imported. The helper that reads the lists is shown to see a live name in
        test_uncommenting_one_name_turns_on_exactly_that_name.

        It is separate from the coefficients: those ARE derived per robot, and the DRC Atlas ships real values. The
        list is the switch; the block is the tuning.
        """
        for robot in sorted(derive.ROBOTS):
            with self.subTest(robot=robot):
                block = _read_yaml(_task_file(robot)).get("locomotion_heuristics")
                self.assertIsNotNone(
                    block, "%s has no locomotion_heuristics block" % robot
                )
                for kind, listed in _listed_names(block).items():
                    self.assertEqual(
                        listed,
                        [],
                        "%s ships locomotion_heuristics.%s = %s. New controller features default OFF until they have "
                        "been validated in simulation; comment the names out again."
                        % (robot, kind, listed),
                    )

    def test_uncommenting_one_name_turns_on_exactly_that_name(self):
        # The documented opt-in is "uncomment one name". With the lists shipped as `base_pose: []` that edit produced a
        # block sequence after a flow sequence, which is not YAML at all (HC1) - so every name offered under every list
        # is uncommented here, one at a time, and the file must still parse and list exactly that name.
        for robot in sorted(derive.ROBOTS):
            lines = _read_text(_task_file(robot)).splitlines(keepends=True)
            for kind, known in EXPECTED_HEURISTIC_NAMES.items():
                entries = _commented_list_entries(lines, kind)
                with self.subTest(robot=robot, kind=kind):
                    # Every registered name is on offer, and nothing else: a misspelt name would parse here and then be
                    # rejected by the loader at start-up.
                    self.assertEqual(sorted(name for _, name in entries), sorted(known))
                for index, name in entries:
                    with self.subTest(robot=robot, kind=kind, name=name):
                        edited = list(lines)
                        edited[index] = re.sub(r"#\s*-", "-", edited[index], count=1)
                        try:
                            task = yaml.safe_load("".join(edited))
                        except yaml.YAMLError as error:
                            self.fail(
                                "uncommenting '%s' breaks the file: %s" % (name, error)
                            )
                        listed = _listed_names(task["locomotion_heuristics"])
                        self.assertEqual(listed[kind], [name])
                        for other in EXPECTED_HEURISTIC_NAMES:
                            if other != kind:
                                self.assertEqual(listed[other], [])

    def test_the_whole_body_task_files_have_no_block_to_be_inert(self):
        # WBMpcInterface never builds the layer, so a block there would be read by nobody. It is absent rather than
        # empty, and the file says why.
        whole_body = glob.glob(
            os.path.join(REPO_ROOT, "robot_models/*/*_wb_mpc/config/mpc/task.yaml")
        )
        self.assertTrue(whole_body, "no whole-body task file found")
        live_key = re.compile(r"^[ \t]*locomotion_heuristics[ \t]*:", re.MULTILINE)
        for task_file in whole_body:
            with self.subTest(task_file=os.path.relpath(task_file, REPO_ROOT)):
                self.assertIsNone(live_key.search(_read_text(task_file)))
                self.assertNotIn("locomotion_heuristics", _read_yaml(task_file))
        # Positive control: the same search finds the block in a centroidal file.
        self.assertIsNotNone(live_key.search(_read_text(_task_file("unitree_g1"))))

    def test_admissibility_table_matches_validate(self):
        # MUST_BE_POSITIVE / MUST_NOT_BE_NEGATIVE are what the derivation tests hold every derived block to. They fell
        # behind validate() once (lateralScale, maximumForce, maximumForceRatioOfWeight), so they are read against it.
        positive, non_negative, calls = _validate_range_checks()
        self.assertTrue(positive, "no requirePositive call found in validate()")
        self.assertTrue(non_negative, "no requireNonNegative call found in validate()")
        # Every call parsed: a check whose argument the key patterns cannot read would otherwise drop out of BOTH sides
        # of the comparisons below, and the tables would fall behind validate() with this test still green.
        self.assertEqual(
            len(positive) + len(non_negative),
            calls,
            "a range check of validate() was not parsed; extend _validate_range_checks()",
        )
        self.assertEqual(sorted(positive), sorted(MUST_BE_POSITIVE))
        self.assertEqual(sorted(non_negative), sorted(MUST_NOT_BE_NEGATIVE))

    def test_every_shipped_block_spells_exactly_the_keys_the_loader_reads(self):
        # The loader reads the keys of its table and ignores anything else, so a misspelt key in a task file is a
        # coefficient that silently stays at its default.
        cpp_keys, calls = _loader_table()
        self.assertTrue(cpp_keys, "the coefficientKeys() table was not found")
        self.assertEqual(len(cpp_keys), calls, "a table entry was not parsed")
        for robot in sorted(derive.ROBOTS):
            with self.subTest(robot=robot):
                block = _read_yaml(_task_file(robot))["locomotion_heuristics"]
                shipped = {
                    "%s.%s" % (name, key)
                    for name, values in block.items()
                    if isinstance(values, dict)
                    for key in values
                }
                self.assertEqual(
                    shipped,
                    set(cpp_keys),
                    "not read by the loader: %s; missing from the file: %s"
                    % (
                        sorted(shipped - set(cpp_keys)),
                        sorted(set(cpp_keys) - shipped),
                    ),
                )

    def test_friction_coefficient_is_the_listed_cones(self):
        # H22: G1 and R1 list friction_force_cone (frictionForceConeSoftConstraint), and reading only
        # contactWrenchConeSoftConstraint handed them a silent 0.5 in place of their own coefficient.
        for robot in sorted(derive.ROBOTS):
            with self.subTest(robot=robot):
                task = _read_yaml(_task_file(robot))
                listed = [
                    block
                    for cone, block in FRICTION_CONES
                    if cone in (task.get("soft_constraints") or [])
                ]
                self.assertTrue(listed, "%s lists no friction cone" % robot)
                friction, source = derive.friction_coefficient(task)
                self.assertIn(
                    source,
                    ["contacts.%s.frictionCoefficient" % block for block in listed],
                )
                block = source.split(".")[1]
                self.assertEqual(
                    friction, float(task["contacts"][block]["frictionCoefficient"])
                )
                if "contact_wrench_cone" not in (task.get("soft_constraints") or []):
                    self.assertEqual(block, "frictionForceConeSoftConstraint")

    def test_gait_cadence_reads_the_mode_names_as_contact_not_swing(self):
        gaits = _read_yaml(GAIT_FILE)
        # `trot` is [LF, RF] at 0.5 s each: alternating single support, so each foot is down half the time and there
        # is no double support at all.
        trot = derive.gait_cadence(gaits, "trot")
        self.assertAlmostEqual(trot.stride_duration, 1.0, places=9)
        self.assertAlmostEqual(trot.step_duration, 0.5, places=9)
        self.assertAlmostEqual(trot.duty_factor, 0.5, places=9)
        self.assertFalse(trot.has_double_support)
        # `walk` is LF 0.6, STANCE 0.1, RF 0.6, STANCE 0.1: each foot is down 0.6 + 0.1 + 0.1 = 0.8 s of a 1.4 s stride.
        walk = derive.gait_cadence(gaits, "walk")
        self.assertTrue(walk.has_double_support)
        self.assertAlmostEqual(walk.stride_duration, 1.4, places=9)
        self.assertAlmostEqual(walk.duty_factor, 0.8 / 1.4, places=9)
        # `run` has flight phases, so it is the gait that sets the 1/beta clamp: 0.3 s down per 0.8 s stride. FLY names
        # no foot in contact; counting it as contact would give run 0.5 and move the clamp to a trot.
        worst_name, worst_duty = derive.smallest_duty_factor(gaits)
        self.assertEqual(worst_name, "run")
        self.assertAlmostEqual(worst_duty, 0.3 / 0.8, places=9)

    def test_duty_factor_is_the_less_loaded_foots(self):
        # Every shipped walking gait is symmetric, so which foot the duty factor is taken from is invisible there. Two
        # mirror-image synthetic gaits: `limp_right` is LF 0.5, STANCE 0.1, RF 0.3, STANCE 0.1, so the left foot is down
        # 0.5 + 0.1 + 0.1 = 0.7 s and the right foot 0.1 + 0.3 + 0.1 = 0.5 s of a 1.0 s stride; `limp_left` the reverse.
        # The 1/beta clamp has to stay below EITHER foot's duty factor, so both must report 0.5, the smaller one.
        gaits = {
            "list": ["limp_right", "limp_left"],
            "limp_right": {
                "modeSequence": ["LF", "STANCE", "RF", "STANCE"],
                "switchingTimes": [0.0, 0.5, 0.6, 0.9, 1.0],
            },
            "limp_left": {
                "modeSequence": ["LF", "STANCE", "RF", "STANCE"],
                "switchingTimes": [0.0, 0.3, 0.4, 0.9, 1.0],
            },
        }
        for name in gaits["list"]:
            with self.subTest(gait=name):
                cadence = derive.gait_cadence(gaits, name)
                self.assertAlmostEqual(cadence.stride_duration, 1.0, places=9)
                self.assertAlmostEqual(cadence.duty_factor, 0.5, places=9)
                self.assertTrue(cadence.has_double_support)

    def test_largest_double_support_ratio_is_walks_one_over_beta(self):
        # Of the gaits with double support, walk has the smallest duty factor (0.8 / 1.4 = 0.571; slow_walk 1.05 / 1.7,
        # fast_walk 0.7, very_slow_walk 1.5 / 2.2, jump 0.3 / 0.5), so it is the one impulse_scaling's force clamp
        # has to cover. run and skip have lower duty factors but no double support, and must not be counted.
        gaits = _read_yaml(GAIT_FILE)
        name, ratio = derive.largest_double_support_ratio(gaits)
        self.assertEqual(name, "walk")
        self.assertAlmostEqual(ratio, 1.4 / 0.8, places=9)

    def test_largest_double_support_ratio_skips_flight_gaits(self):
        # A synthetic table, so the rule is pinned whatever gait.yaml becomes. `hop` has the lowest duty factor
        # (0.2 / 0.8 = 0.25, ratio 4) but no double support; `amble` is down 0.4 + 0.2 + 0.2 = 0.8 s of 1.2 s (ratio
        # 1.5); `stroll` 0.3 + 0.5 + 0.5 = 1.3 s of 1.6 s (ratio 1.23).
        gaits = {
            "list": ["hop", "amble", "stroll"],
            "hop": {
                "modeSequence": ["LF", "FLY", "RF", "FLY"],
                "switchingTimes": [0.0, 0.2, 0.4, 0.6, 0.8],
            },
            "amble": {
                "modeSequence": ["LF", "STANCE", "RF", "STANCE"],
                "switchingTimes": [0.0, 0.4, 0.6, 1.0, 1.2],
            },
            "stroll": {
                "modeSequence": ["LF", "STANCE", "RF", "STANCE"],
                "switchingTimes": [0.0, 0.3, 0.8, 1.1, 1.6],
            },
        }
        name, ratio = derive.largest_double_support_ratio(gaits)
        self.assertEqual(name, "amble")
        self.assertAlmostEqual(ratio, 1.5, places=9)
        # Positive control: the flight gait IS the smallest duty factor, so skipping it above was the rule at work.
        smallest_name, smallest_duty = derive.smallest_duty_factor(gaits)
        self.assertEqual(smallest_name, "hop")
        self.assertAlmostEqual(smallest_duty, 0.25, places=9)

    def test_an_unknown_gait_is_reported_rather_than_guessed(self):
        with self.assertRaises(SystemExit):
            derive.gait_cadence(_read_yaml(GAIT_FILE), "moonwalk")

    def test_speed_ladder_is_the_motion_managers_and_every_rung_is_a_gait(self):
        # SPEED_LADDER is a copy of ProceduralMpcMotionManager::gaitModeStates_, and the stepping tests claim "every
        # rung" on its strength; a rung added to the C++ ladder (a fast_trot, say) has to fail here until it is copied,
        # or the claim goes stale silently.
        names, rows = _motion_manager_gait_ladder()
        self.assertTrue(names, "the gaitModeStates_ initializer was not found")
        self.assertEqual(len(names), rows, "a gaitModeStates_ row was not parsed")
        # The standing state heads the C++ ladder but is not a walking gait: it has no step to place.
        self.assertEqual(names[0], "stance")
        self.assertEqual(tuple(names[1:]), SPEED_LADDER)
        # Every rung is a gait the scheduler can load, and the script reads its stance the way the mode names define
        # it: slow_walk's STANCE phases count for both feet, run's FLY phases for neither.
        gaits = _read_yaml(GAIT_FILE)
        for rung in SPEED_LADDER:
            with self.subTest(rung=rung):
                self.assertIn(rung, gaits.get("list", []))
                left, right, stride = _foot_stance_durations(gaits[rung])
                cadence = derive.gait_cadence(gaits, rung)
                self.assertAlmostEqual(cadence.stride_duration, stride, places=9)
                self.assertAlmostEqual(
                    cadence.duty_factor * cadence.stride_duration,
                    min(left, right),
                    places=9,
                )


class DerivationPropertiesTest(_DerivedBlockAssertions, unittest.TestCase):
    """derive_parameters() and the output helpers against a synthetic geometry. Never skipped."""

    def _derive_robot(self, robot, gait="trot"):
        task, limits, planning = _robot_inputs(robot)
        geometry = _synthetic_geometry()
        parameters, notes = _derive(geometry, task, limits, planning, gait)
        return geometry, task, planning, parameters, notes

    def test_every_robot_derives_a_complete_and_admissible_block(self):
        for robot in sorted(derive.ROBOTS):
            with self.subTest(robot=robot):
                _, _, _, parameters, notes = self._derive_robot(robot)
                self.assertEqual(list(parameters), list(ALL_HEURISTIC_NAMES))
                self.assertTrue(notes)
                self.assert_admissible(parameters)

    def test_derived_keys_are_exactly_the_keys_the_cpp_loader_reads(self):
        # A key the script prints that the loader does not read would be pasted into a task file and silently
        # ignored; a key the loader reads that the script does not print would never be derived.
        cpp_keys, calls = _loader_table()
        self.assertTrue(cpp_keys, "the coefficientKeys() table was not found")
        self.assertEqual(len(cpp_keys), calls, "a table entry was not parsed")
        self.assertEqual(len(set(cpp_keys)), len(cpp_keys), "a key is listed twice")
        for robot in sorted(derive.ROBOTS):
            with self.subTest(robot=robot):
                derived = _derived_keys(self._derive_robot(robot)[3])
                self.assertEqual(
                    derived,
                    set(cpp_keys),
                    "not read by the loader: %s; not derived: %s"
                    % (
                        sorted(derived - set(cpp_keys)),
                        sorted(set(cpp_keys) - derived),
                    ),
                )

    def test_blocks_print_in_the_loader_table_order(self):
        # format_block() promises the task files' order, so a printed block can be compared with a shipped one by eye.
        cpp_keys, _ = _loader_table()
        table_order = []
        for dotted in cpp_keys:
            name = dotted.split(".")[0]
            if name not in table_order:
                table_order.append(name)
        printed = re.findall(
            r"^  ([a-z_]+):$",
            derive.format_block(self._derive_robot("drc_atlas")[3]),
            re.MULTILINE,
        )
        self.assertEqual(printed, table_order)

    def test_lateral_scale_reproduces_the_nominal_stance(self):
        for robot in sorted(derive.ROBOTS):
            with self.subTest(robot=robot):
                geometry, task, _, parameters, _ = self._derive_robot(robot)
                self.assert_lateral_scale_reproduces_the_stance(
                    parameters, geometry, task
                )
        # A robot with a stance is exercised whatever the shipped files say: 0.30 m over the synthetic 0.1786 m hips
        # is 1.680 after rounding, and the property must still hold within that rounding.
        geometry = _synthetic_geometry()
        task = {"nominal_foothold": {"stepWidth": 0.30}}
        parameters, _ = _derive(geometry, task, {}, {})
        self.assertAlmostEqual(
            parameters["hip_centered_stepping"]["lateralScale"], 1.680, places=9
        )
        self.assert_lateral_scale_reproduces_the_stance(parameters, geometry, task)

    def test_capture_point_and_turning_lean_use_the_configured_lip_height(self):
        for robot in sorted(derive.ROBOTS):
            with self.subTest(robot=robot):
                geometry, task, planning, parameters, _ = self._derive_robot(robot)
                self.assert_pendulum_is_the_configured_one(
                    parameters, geometry, task, planning
                )

    def test_lip_height_prefers_the_dcm_cost_then_the_planner_then_the_model(self):
        # Hand-picked lengths, all different from the synthetic model's 0.70 m CoM height.
        geometry = _synthetic_geometry()
        planning = {"contact_planning": {"shared": {"comHeight": 0.80}}}
        cases = (
            ({"dcm_terminal_cost": {"comHeight": 0.90}}, planning, 0.90),
            ({}, planning, 0.80),
            ({}, {}, 0.70),
        )
        for task, planning_file, expected in cases:
            with self.subTest(expected=expected):
                parameters, _ = _derive(geometry, task, {}, planning_file)
                self.assertAlmostEqual(
                    parameters["capture_point"]["comHeightOverride"], expected, places=9
                )
                self.assertAlmostEqual(
                    parameters["high_speed_turning"]["forwardPerCrossTerm"],
                    round(expected / 9.81, 4),
                    places=9,
                )

    def test_stepping_leads_follow_the_stance_duration(self):
        for robot in sorted(derive.ROBOTS):
            with self.subTest(robot=robot):
                geometry, task, _, parameters, _ = self._derive_robot(robot)
                self.assert_stepping_follows_the_stance(parameters, geometry, task)

    def test_stepping_coefficients_do_not_depend_on_the_gait_flag(self):
        # HC2: `--gait` defaulted to trot and sized the stepping gains from its step period, so the printed block was
        # right in one speed band. The stance-duration terms carry the cadence now, so every rung prints the same block.
        gaits = _read_yaml(GAIT_FILE)
        geometry = _synthetic_geometry()
        for robot in sorted(derive.ROBOTS):
            with self.subTest(robot=robot):
                task, limits, planning = _robot_inputs(robot)
                stances = set()
                blocks = {}
                for gait in SPEED_LADDER:
                    cadence = derive.gait_cadence(gaits, gait)
                    stances.add(round(min(_foot_stance_durations(gaits[gait])[:2]), 9))
                    parameters, _ = derive.derive_parameters(
                        geometry, cadence, limits, task, planning, gaits
                    )
                    blocks[gait] = (
                        parameters["translational_stepping"],
                        parameters["in_place_turning"],
                    )
                # Positive control: the rungs genuinely differ in stance, so equal blocks are not a coincidence.
                self.assertGreater(len(stances), 1)
                for gait in SPEED_LADDER:
                    self.assertEqual(blocks[gait], blocks["trot"], gait)

    def assert_duty_clamp_is_just_below(self, clamp, smallest_duty):
        """minimumDutyFactor sits below the smallest duty factor of any gait, and only just below it.

        Above it, the clamp holds a real gait's beta up and scales that gait's force down. Far below it, the clamp lets
        1/beta run up to 1/clamp as a flight phase opens, which is the blow-up it exists to stop. The lower bound is
        the script's margin plus what rounding to two decimals can cost; DUTY_CLAMP_HEADROOM catches the margin itself
        running away, which moves both sides of that bound together.
        """
        self.assertLess(clamp, smallest_duty)
        self.assertGreaterEqual(
            clamp, smallest_duty - derive.DUTY_FACTOR_MARGIN - 0.005 - 1e-12
        )
        self.assertLessEqual(smallest_duty - clamp, DUTY_CLAMP_HEADROOM + 1e-12)

    def test_impulse_clamps_bracket_every_shipped_gait(self):
        # minimumDutyFactor has to sit just below the smallest duty factor of any gait, and maximumForceRatio at or
        # above the largest W / (F beta) of any double support (or it clips a real one). The gait table is read with
        # the test's own mode parse.
        smallest_duty, largest_ratio = _gait_table_extremes(_read_yaml(GAIT_FILE))
        for robot in sorted(derive.ROBOTS):
            with self.subTest(robot=robot):
                block = self._derive_robot(robot)[3]["impulse_scaling"]
                self.assert_duty_clamp_is_just_below(
                    block["minimumDutyFactor"], smallest_duty
                )
                self.assertGreaterEqual(block["maximumForceRatio"], largest_ratio)

    def test_impulse_clamps_follow_a_gait_table_beyond_the_floors(self):
        # The shipped gaits never reach the part of maximumForceRatio that reads the gait table: walk's 1/beta is 1.75
        # and the script floors the clamp at 2.0, so a clamp hard-coded to 2.0, or to 1000, passes the test above.
        # `bound` is LF 0.2, STANCE 0.1, FLY 0.3, RF 0.2, STANCE 0.1, FLY 0.3: each foot is down 0.2 + 0.1 + 0.1 = 0.4 s
        # of a 1.2 s stride, so beta = 1/3 and its double support asks for 3 x weight compensation; the clamp has to
        # cover it, and by no more than the one-decimal ceiling the script rounds up to. `hop` (LF, FLY, RF, FLY at
        # 0.2 s each) has the smaller beta, 1/4, but no double support, so beside `bound` it moves the duty clamp and
        # must leave the force clamp alone.
        bound = {
            "modeSequence": ["LF", "STANCE", "FLY", "RF", "STANCE", "FLY"],
            "switchingTimes": [0.0, 0.2, 0.3, 0.6, 0.8, 0.9, 1.2],
        }
        hop = {
            "modeSequence": ["LF", "FLY", "RF", "FLY"],
            "switchingTimes": [0.0, 0.2, 0.4, 0.6, 0.8],
        }
        cases = (
            ({"list": ["bound"], "bound": bound}, 1.0 / 3.0),
            ({"list": ["bound", "hop"], "bound": bound, "hop": hop}, 0.25),
        )
        geometry = _synthetic_geometry()
        for gaits, smallest_duty in cases:
            with self.subTest(gaits=gaits["list"]):
                parameters, _ = derive.derive_parameters(
                    geometry, derive.gait_cadence(gaits, "bound"), {}, {}, {}, gaits
                )
                block = parameters["impulse_scaling"]
                self.assert_duty_clamp_is_just_below(
                    block["minimumDutyFactor"], smallest_duty
                )
                self.assertGreaterEqual(block["maximumForceRatio"], 3.0)
                self.assertLessEqual(block["maximumForceRatio"], 3.1)

    def test_capture_point_clamp_is_a_fraction_of_the_reach_bound(self):
        # maximumOffset is CAPTURE_POINT_STEP_FRACTION of foot_separation.maxStepLength, the reach bound, and not of
        # hlip.maxStepLength, the closed-form planner's clip on the step it emits: with no planner running it is the
        # reach that bounds how far a landing target may be pushed. The two differ on Atlas (0.28 m against 0.20 m)
        # and SA01 (0.232 m against 0.16 m). The planning file is read here with PyYAML, not through the script.
        fraction = derive.CAPTURE_POINT_STEP_FRACTION
        robots_with_a_reach_bound = []
        for robot in sorted(derive.ROBOTS):
            with self.subTest(robot=robot):
                _, _, planning, parameters, _ = self._derive_robot(robot)
                clamp = parameters["capture_point"]["maximumOffset"]
                separation = (planning.get("contact_planning") or {}).get(
                    "foot_separation"
                ) or {}
                if "maxStepLength" in separation:
                    robots_with_a_reach_bound.append(robot)
                    reach = float(separation["maxStepLength"])
                    self.assertLessEqual(abs(clamp - fraction * reach), ROUNDING_3)
                else:
                    self.assertEqual(clamp, CAPTURE_POINT_FALLBACK_CLAMP)
        # Positive control: the reach branch above was taken, so it is not the fallback that every robot passed on.
        self.assertTrue(robots_with_a_reach_bound)

        # And whatever the shipped planning files come to say: two bounds that differ, with the reach one expected.
        planning = {
            "contact_planning": {
                "foot_separation": {"maxStepLength": 0.9},
                "hlip": {"maxStepLength": 0.3},
            }
        }
        parameters, notes = _derive(_synthetic_geometry(), {}, {}, planning)
        self.assertLessEqual(
            abs(parameters["capture_point"]["maximumOffset"] - fraction * 0.9),
            ROUNDING_3,
        )
        # The note says where the clamp came from: the reach bound here, the fallback with no planning file. It once
        # printed "40% of foot_separation.maxStepLength 0.00 = 0.250 m" for the fallback.
        clamp_notes = [n for n in notes if n.startswith("capture_point.maximumOffset")]
        self.assertEqual(len(clamp_notes), 1)
        self.assertIn("foot_separation.maxStepLength 0.90", clamp_notes[0])
        parameters, notes = _derive(_synthetic_geometry(), {}, {}, {})
        self.assertEqual(
            parameters["capture_point"]["maximumOffset"], CAPTURE_POINT_FALLBACK_CLAMP
        )
        clamp_notes = [n for n in notes if n.startswith("capture_point.maximumOffset")]
        self.assertEqual(len(clamp_notes), 1)
        self.assertIn("fallback", clamp_notes[0])
        self.assertNotIn("maxStepLength 0.00", clamp_notes[0])

    def test_lean_and_crouch_reach_their_assumed_values_at_the_forward_limit(self):
        # pitchPerForwardVelocity and heightPerSpeed are sized so that the maximum FORWARD stick gives the script's
        # stated lean and crouch. Dividing by the lateral or the yaw-rate limit instead lands on a different lean; the
        # synthetic limits keep the three apart whatever the shipped reference files come to say. The tolerance is the
        # four-decimal rounding of the coefficient, magnified by vmax.
        cases = [(robot,) + _robot_inputs(robot) for robot in sorted(derive.ROBOTS)]
        cases.append(
            (
                "synthetic",
                {},
                {
                    "maxDisplacementVelocityX": 1.5,
                    "maxDisplacementVelocityY": 0.4,
                    "maxRotationVelocity": 0.7,
                },
                {},
            )
        )
        for name, task, limits, planning in cases:
            with self.subTest(robot=name):
                parameters, _ = _derive(_synthetic_geometry(), task, limits, planning)
                vmax = float(limits["maxDisplacementVelocityX"])
                orientation = parameters["orientation_compensation"]
                height = parameters["height_compensation"]
                lean = orientation["pitchPerForwardVelocity"] * vmax
                crouch = height["heightPerSpeed"] * vmax
                self.assertLessEqual(
                    abs(lean - derive.PITCH_AT_MAX_FORWARD_SPEED),
                    0.5e-4 * vmax + 1e-12,
                )
                self.assertLessEqual(
                    abs(crouch + derive.CROUCH_AT_MAX_FORWARD_SPEED),
                    0.5e-4 * vmax + 1e-12,
                )
                # And the clamps leave them whole: OrientationCompensationHeuristic clamps the pitch to +/- maximumTilt
                # and HeightCompensationHeuristic the offset to +/- maximumHeightOffset, so a clamp below the designed
                # value would cut the lean or the crouch short of the maximum stick.
                self.assertGreater(orientation["maximumTilt"], lean)
                self.assertGreaterEqual(height["maximumHeightOffset"], abs(crouch))

    def test_every_blend_starts_at_the_unattenuated_heuristic(self):
        # UNIT_GAINS scale a heuristic's whole effect, and validate() admits 0 for each: a derivation printing 0 would
        # hand out a block in which listing the heuristic changes nothing, so the sweep would start from "off". The
        # starting point is the heuristic at full strength - Bledt's value, the textbook capture point, the hip's own
        # x - which is 1.
        for robot in sorted(derive.ROBOTS):
            with self.subTest(robot=robot):
                parameters = self._derive_robot(robot)[3]
                for dotted in UNIT_GAINS:
                    name, key = dotted.split(".")
                    self.assertEqual(parameters[name][key], 1.0, dotted)

    def test_centripetal_clamp_is_sized_from_the_listed_cone(self):
        # Synthetic contacts with DIFFERENT coefficients in the two cones, so reading the wrong block cannot pass.
        contacts = {
            "contactWrenchConeSoftConstraint": {"frictionCoefficient": 0.7},
            "frictionForceConeSoftConstraint": {"frictionCoefficient": 0.3},
        }
        force_cone = {"soft_constraints": ["friction_force_cone"], "contacts": contacts}
        wrench_cone = {
            "soft_constraints": ["contact_wrench_cone"],
            "contacts": contacts,
        }
        neither = {"soft_constraints": ["joint_limits"], "contacts": contacts}
        self.assertEqual(
            derive.friction_coefficient(force_cone),
            (0.3, "contacts.frictionForceConeSoftConstraint.frictionCoefficient"),
        )
        self.assertEqual(
            derive.friction_coefficient(wrench_cone),
            (0.7, "contacts.contactWrenchConeSoftConstraint.frictionCoefficient"),
        )
        self.assertEqual(derive.friction_coefficient(neither), (0.5, None))
        self.assertEqual(
            derive.friction_coefficient({"contacts": contacts}), (0.5, None)
        )

        fraction = derive.CENTRIPETAL_FRICTION_FRACTION
        warning = "lists neither contact_wrench_cone nor friction_force_cone"
        geometry = _synthetic_geometry()
        parameters, notes = _derive(geometry, force_cone, {}, {})
        self.assertAlmostEqual(
            parameters["centripetal_acceleration"]["maximumForceRatioOfWeight"],
            round(fraction * 0.3, 3),
            places=9,
        )
        self.assertFalse([note for note in notes if warning in note])
        # With no cone listed the assumed coefficient is used and SAID to be assumed.
        parameters, notes = _derive(geometry, neither, {}, {})
        self.assertAlmostEqual(
            parameters["centripetal_acceleration"]["maximumForceRatioOfWeight"],
            round(fraction * 0.5, 3),
            places=9,
        )
        self.assertTrue([note for note in notes if warning in note])

    def test_every_robot_sizes_its_centripetal_clamp_from_its_own_cone(self):
        for robot in sorted(derive.ROBOTS):
            with self.subTest(robot=robot):
                _, task, _, parameters, _ = self._derive_robot(robot)
                soft = task.get("soft_constraints") or []
                friction = next(
                    float(task["contacts"][block]["frictionCoefficient"])
                    for cone, block in FRICTION_CONES
                    if cone in soft
                )
                self.assertAlmostEqual(
                    parameters["centripetal_acceleration"]["maximumForceRatioOfWeight"],
                    round(derive.CENTRIPETAL_FRICTION_FRACTION * friction, 3),
                    places=9,
                )

    def test_printed_block_reads_back_as_the_derivation_and_checks_clean(self):
        # HC4: printed at six significant digits, 2*pi came back as 6.28319, so the tool's own output failed its own
        # `--check`. Printed, read back with YAML and compared, a block must be EXACTLY what was derived.
        for robot in sorted(derive.ROBOTS):
            with self.subTest(robot=robot):
                parameters = self._derive_robot(robot)[3]
                task = yaml.safe_load(
                    "locomotion_heuristics:\n" + derive.format_block(parameters)
                )
                read_back = task["locomotion_heuristics"]
                self.assertEqual(read_back, parameters)
                for name, values in read_back.items():
                    for key, value in values.items():
                        # 1.0, never 1: an integer would read as a count, and the C++ reads a double.
                        self.assertIsInstance(value, float, "%s.%s" % (name, key))
                printed = io.StringIO()
                with contextlib.redirect_stdout(printed):
                    differences = derive.compare(
                        parameters, derive.shipped_parameters(task)
                    )
                self.assertEqual(differences, 0, printed.getvalue())

    def test_compare_counts_a_changed_and_a_missing_coefficient(self):
        # Positive control for the zero above: compare() does see a difference, one per coefficient.
        parameters = self._derive_robot("drc_atlas")[3]
        shipped = copy.deepcopy(parameters)
        shipped["capture_point"]["gain"] += 1e-5
        del shipped["impulse_scaling"]["scale"]
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            differences = derive.compare(parameters, shipped)
        self.assertEqual(differences, 2)
        self.assertIn("gain", printed.getvalue())
        self.assertIn("absent from the task file", printed.getvalue())

    def test_shipped_parameters_reads_the_blocks_and_not_the_lists(self):
        task = {
            "locomotion_heuristics": {
                "base_pose": ["orientation_compensation"],
                "foothold": None,
                "wrench": [],
                "capture_point": {"gain": 2, "maximumOffset": 0.3},
            }
        }
        shipped = derive.shipped_parameters(task)
        self.assertEqual(
            shipped, {"capture_point": {"gain": 2.0, "maximumOffset": 0.3}}
        )
        self.assertIsInstance(shipped["capture_point"]["gain"], float)
        self.assertEqual(derive.shipped_parameters({}), {})

    def test_fitted_coefficients_are_left_at_zero(self):
        # The script derives what follows from geometry and cadence, and refuses to invent what does not. Table C.2's
        # roll lean and its pitch limit cycle are the two that need data, and a future edit that quietly filled them
        # in with a plausible-looking number is exactly what this catches.
        for robot in sorted(derive.ROBOTS):
            with self.subTest(robot=robot):
                parameters = self._derive_robot(robot)[3]
                self.assertEqual(
                    parameters["orientation_compensation"]["rollPerLateralVelocity"],
                    0.0,
                )
                self.assertEqual(
                    parameters["periodic_orientation"]["pitchAmplitude"], 0.0
                )
                self.assertEqual(
                    parameters["height_compensation"]["heightPerSpeedSquared"], 0.0
                )
                for name in ALL_HEURISTIC_NAMES:
                    self.assertEqual(parameters[name].get("forwardOffset", 0.0), 0.0)
                    self.assertEqual(parameters[name].get("lateralOffset", 0.0), 0.0)
                # Positive control: the SIZED coefficients beside them are not zero, so a block of zeros fails.
                self.assertGreater(
                    parameters["orientation_compensation"]["pitchPerForwardVelocity"],
                    0.0,
                )
                self.assertLess(
                    parameters["height_compensation"]["heightPerSpeed"], 0.0
                )
                self.assertGreater(
                    parameters["periodic_orientation"]["rollAmplitude"], 0.0
                )

    def test_periodic_roll_peaks_at_mid_left_stance(self):
        # The one phase in the family that is derived rather than fitted, and the derivation is easy to invert by
        # accident: phase [0, 0.5) is the LF mode, which names the foot IN CONTACT, so 0.25 is mid LEFT stance; and a
        # positive roll raises the left side, i.e. drops the hip on the SWING side, which is what a biped does.
        block = self._derive_robot("drc_atlas")[3]["periodic_orientation"]
        self.assertGreater(block["rollAmplitude"], 0.0)
        # Roll once per gait cycle, pitch once per step, i.e. twice per cycle.
        self.assertAlmostEqual(block["rollPhaseRate"], 2.0 * math.pi, delta=1e-7)
        self.assertAlmostEqual(block["pitchPhaseRate"], 4.0 * math.pi, delta=1e-7)
        at_mid_left_stance = math.sin(
            block["rollPhaseRate"] * 0.25 + block["rollPhaseOffset"]
        )
        self.assertAlmostEqual(at_mid_left_stance, 1.0, places=6)
        at_mid_right_stance = math.sin(
            block["rollPhaseRate"] * 0.75 + block["rollPhaseOffset"]
        )
        self.assertAlmostEqual(at_mid_right_stance, -1.0, places=6)


@unittest.skipIf(
    not HAVE_PINOCCHIO,
    "pinocchio is not importable; the URDF-based checks run inside the dev container",
)
class DeriveParametersTest(_DerivedBlockAssertions, unittest.TestCase):
    """The real models: one case per shipped robot, built with pinocchio at the nominal posture."""

    def _derive(self, robot):
        task, limits, planning = _robot_inputs(robot)
        geometry = derive.derive_geometry(
            os.path.join(REPO_ROOT, derive.ROBOTS[robot]["urdf"]),
            task,
            task.get("model_settings", {}),
        )
        parameters, notes = _derive(geometry, task, limits, planning)
        return geometry, task, planning, parameters, notes

    def test_every_robot_derives_a_complete_and_admissible_block(self):
        for robot in sorted(derive.ROBOTS):
            with self.subTest(robot=robot):
                geometry, _, _, parameters, notes = self._derive(robot)

                # The model reconstructed at the nominal posture, not at the URDF's neutral: a humanoid standing on
                # its feet has its center of mass around half a leg above them, and both feet on the ground.
                self.assertGreater(geometry.total_mass, 1.0, "the model has no mass")
                self.assertGreater(
                    geometry.com_height,
                    0.3,
                    "the CoM is not above the feet; check initialState",
                )
                self.assertLess(geometry.com_height, 2.0)
                self.assertGreater(
                    geometry.nominal_foot_separation,
                    0.05,
                    "the feet are on top of each other",
                )
                self.assertEqual(len(geometry.hip_offset), 2)
                for name in geometry.hip_joint_names:
                    self.assertNotIn(
                        "not found", name, "the walk up the kinematic tree failed"
                    )
                # Left hip to the left, right hip to the right, and symmetric about the base.
                self.assertGreater(geometry.hip_offset[0][1], 0.0)
                self.assertLess(geometry.hip_offset[1][1], 0.0)
                self.assertAlmostEqual(
                    geometry.hip_offset[0][1], -geometry.hip_offset[1][1], places=4
                )

                self.assertEqual(list(parameters), list(ALL_HEURISTIC_NAMES))
                self.assertTrue(notes)
                self.assert_admissible(parameters)

    def test_closed_form_coefficients_hold_on_the_real_models(self):
        # The same properties as DerivationPropertiesTest, on the hips and CoM the URDF actually gives: G1 and R1
        # configure no LIP height, so theirs is the model's, and Atlas's lateralScale spreads its real hips.
        for robot in sorted(derive.ROBOTS):
            with self.subTest(robot=robot):
                geometry, task, planning, parameters, _ = self._derive(robot)
                self.assert_lateral_scale_reproduces_the_stance(
                    parameters, geometry, task
                )
                self.assert_pendulum_is_the_configured_one(
                    parameters, geometry, task, planning
                )
                self.assert_stepping_follows_the_stance(parameters, geometry, task)

    def test_every_shipped_pendulum_number_is_the_models(self):
        # The pendulum is the model's: a comHeight of 0 in dcm_terminal_cost or in the planner's shared block is resolved
        # by the C++ computeComHeightAboveFeet(), and the two heuristic numbers the task files write out -
        # capture_point.comHeightOverride and high_speed_turning's z / g - are derived from it. This pins, per robot, that
        # the numbers a task file SHIPS are the ones this script derives from the model within their printed rounding;
        # humanoid_centroidal_mpc:testNominalPendulum pins the same shipped numbers against the C++ derivation, so the two
        # derivations agree wherever a robot ships a derived number. A zero is not a derived number (it is "derive it"),
        # and a positive LIP height is an override, which must still be within a few percent of the model.
        checked = 0
        for robot in sorted(derive.ROBOTS):
            with self.subTest(robot=robot):
                geometry, task, planning, _, _ = self._derive(robot)
                block = task.get("locomotion_heuristics") or {}
                capture = block.get("capture_point") or {}
                override = float(capture.get("comHeightOverride", 0.0))
                if override > 0.0:
                    self.assertAlmostEqual(
                        override,
                        geometry.com_height,
                        delta=ROUNDING_4,
                        msg="capture_point.comHeightOverride is not the model's CoM height above its feet",
                    )
                    checked += 1
                turning = block.get("high_speed_turning") or {}
                gravity = float(capture.get("gravity", derive.GRAVITY))
                for key in ("forwardPerCrossTerm", "lateralPerCrossTerm"):
                    value = float(turning.get(key, 0.0))
                    if value > 0.0:
                        self.assertAlmostEqual(
                            value,
                            geometry.com_height / gravity,
                            delta=ROUNDING_4,
                            msg="high_speed_turning.%s is not z_com / g of the model"
                            % key,
                        )
                        checked += 1
                for label, height in (
                    (
                        "dcm_terminal_cost.comHeight",
                        float(
                            (task.get("dcm_terminal_cost") or {}).get("comHeight", 0.0)
                        ),
                    ),
                    (
                        "shared.comHeight",
                        float(
                            (
                                (planning.get("contact_planning") or {}).get("shared")
                                or {}
                            ).get("comHeight", 0.0)
                        ),
                    ),
                ):
                    self.assertGreaterEqual(height, 0.0, label)
                    if height > 0.0:
                        self.assertLess(
                            abs(height - geometry.com_height),
                            0.05 * geometry.com_height,
                            "%s = %.4f is an override %.0f%% off the model's %.4f m"
                            % (
                                label,
                                height,
                                100.0
                                * abs(height - geometry.com_height)
                                / geometry.com_height,
                                geometry.com_height,
                            ),
                        )
        # Positive control: at least one robot ships a derived number, so the loop above is not vacuous.
        self.assertGreater(checked, 0)
        # And the hand-set 0.85 m Atlas used to ship for its pendulum is exactly what the 5 % check refuses.
        atlas, _, _, _, _ = self._derive("drc_atlas")
        self.assertGreater(abs(0.85 - atlas.com_height), 0.05 * atlas.com_height)


if __name__ == "__main__":
    unittest.main()
