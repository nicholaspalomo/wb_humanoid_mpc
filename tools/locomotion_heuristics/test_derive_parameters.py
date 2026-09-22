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

"""Checks that derive_parameters.py still speaks to every shipped robot.

Not a check on the VALUES: a coefficient that has been swept in simulation should differ from its starting point, and
freezing the derivation would make tuning a test failure. What this guards is the machinery underneath - that each
robot's URDF still parses, that `initialState`'s joint block still lines up with the reduced model, that the contact
frames are still reconstructible from `contacts.contact_frame_translation`, and that the gait table still yields a
cadence. Those are the things that break silently when a model or a config is edited, and that would otherwise be
found only by someone regenerating a block months later.
"""

import importlib.util
import math
import os
import sys
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPT = os.path.join(
    REPO_ROOT, "tools", "locomotion_heuristics", "derive_parameters.py"
)


def _load_script():
    spec = importlib.util.spec_from_file_location("derive_parameters", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:
    derive = _load_script()
except SystemExit as exit_error:  # pinocchio missing outside the dev container
    derive = None
    _SKIP_REASON = str(exit_error)


@unittest.skipIf(
    derive is None,
    "derive_parameters.py could not be imported: %s"
    % globals().get("_SKIP_REASON", ""),
)
class DeriveParametersTest(unittest.TestCase):
    """One case per shipped robot, plus the invariants that hold for all of them."""

    def _derive(self, robot):
        paths = derive.ROBOTS[robot]
        mpc_dir = os.path.join(REPO_ROOT, paths["mpc"])
        task = derive.load_yaml(os.path.join(mpc_dir, "config/mpc/task.yaml"))
        limits = derive.load_yaml(
            os.path.join(mpc_dir, "config/command/reference.yaml")
        )
        planning_file = os.path.join(mpc_dir, "config/mpc/contact_planning.yaml")
        planning = (
            derive.load_yaml(planning_file) if os.path.isfile(planning_file) else {}
        )
        gaits = derive.load_yaml(
            os.path.join(
                REPO_ROOT, "humanoid_nmpc/humanoid_common_mpc/config/command/gait.yaml"
            )
        )
        geometry = derive.derive_geometry(
            os.path.join(REPO_ROOT, paths["urdf"]), task, task.get("model_settings", {})
        )
        cadence = derive.gait_cadence(gaits, "trot")
        parameters, notes = derive.derive_parameters(
            geometry, cadence, limits, task, planning, gaits
        )
        return geometry, parameters, notes

    def test_every_robot_derives_a_complete_and_admissible_block(self):
        # The ten names of Bledt's Appendix C, which is also what LocomotionHeuristicFormulation registers.
        expected = {
            "orientation_compensation",
            "periodic_orientation",
            "height_compensation",
            "hip_centered_stepping",
            "capture_point",
            "translational_stepping",
            "in_place_turning",
            "high_speed_turning",
            "impulse_scaling",
            "centripetal_acceleration",
        }
        for robot in sorted(derive.ROBOTS):
            with self.subTest(robot=robot):
                geometry, parameters, notes = self._derive(robot)

                # The model reconstructed at the nominal posture, not at the URDF's neutral: a humanoid standing on
                # its feet has its centre of mass around half a leg above them, and both feet on the ground.
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

                self.assertEqual(set(parameters), expected)
                self.assertTrue(notes)

                # The ranges LocomotionHeuristicConfig::validate() enforces. A derivation that produced a block the
                # loader rejects would be worse than no derivation at all.
                self.assertGreater(
                    parameters["orientation_compensation"]["maximumTilt"], 0.0
                )
                self.assertGreaterEqual(
                    parameters["height_compensation"]["maximumHeightOffset"], 0.0
                )
                self.assertGreater(parameters["capture_point"]["maximumOffset"], 0.0)
                self.assertGreater(parameters["capture_point"]["gravity"], 0.0)
                self.assertGreaterEqual(
                    parameters["capture_point"]["comHeightOverride"], 0.0
                )
                self.assertGreaterEqual(parameters["impulse_scaling"]["scale"], 0.0)
                duty = parameters["impulse_scaling"]["minimumDutyFactor"]
                self.assertTrue(
                    0.0 < duty <= 1.0, "minimumDutyFactor %s is outside (0, 1]" % duty
                )
                self.assertGreaterEqual(
                    parameters["impulse_scaling"]["maximumForceRatio"], 1.0
                )
                self.assertGreaterEqual(
                    parameters["centripetal_acceleration"]["scale"], 0.0
                )
                self.assertGreater(
                    parameters["centripetal_acceleration"]["maximumForce"]
                    + parameters["centripetal_acceleration"][
                        "maximumForceRatioOfWeight"
                    ],
                    0.0,
                    "both centripetal clamps at zero would remove the clamp",
                )

    def test_fitted_coefficients_are_left_at_zero(self):
        # The script derives what follows from geometry and cadence, and refuses to invent what does not. Table C.2's
        # roll lean and its pitch limit cycle are the two that need data, and a future edit that quietly filled them
        # in with a plausible-looking number is exactly what this catches.
        for robot in sorted(derive.ROBOTS):
            with self.subTest(robot=robot):
                _, parameters, _ = self._derive(robot)
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
                for name in (
                    "orientation_compensation",
                    "periodic_orientation",
                    "height_compensation",
                    "translational_stepping",
                    "in_place_turning",
                    "high_speed_turning",
                ):
                    self.assertEqual(parameters[name].get("forwardOffset", 0.0), 0.0)
                    self.assertEqual(parameters[name].get("lateralOffset", 0.0), 0.0)

    def test_periodic_roll_peaks_at_mid_left_stance(self):
        # The one phase in the family that is derived rather than fitted, and the derivation is easy to invert by
        # accident: phase [0, 0.5) is the LF mode, which names the foot IN CONTACT, so 0.25 is mid LEFT stance; and a
        # positive roll raises the left side, i.e. drops the hip on the SWING side, which is what a biped does.
        _, parameters, _ = self._derive("drc_atlas")
        block = parameters["periodic_orientation"]
        self.assertGreater(block["rollAmplitude"], 0.0)
        at_mid_left_stance = math.sin(
            block["rollPhaseRate"] * 0.25 + block["rollPhaseOffset"]
        )
        self.assertAlmostEqual(at_mid_left_stance, 1.0, places=6)
        at_mid_right_stance = math.sin(
            block["rollPhaseRate"] * 0.75 + block["rollPhaseOffset"]
        )
        self.assertAlmostEqual(at_mid_right_stance, -1.0, places=6)

    def test_gait_cadence_reads_the_mode_names_as_contact_not_swing(self):
        gaits = derive.load_yaml(
            os.path.join(
                REPO_ROOT, "humanoid_nmpc/humanoid_common_mpc/config/command/gait.yaml"
            )
        )
        # `trot` is [LF, RF] at 0.5 s each: alternating single support, so each foot is down half the time and there
        # is no double support at all.
        trot = derive.gait_cadence(gaits, "trot")
        self.assertAlmostEqual(trot.stride_duration, 1.0, places=6)
        self.assertAlmostEqual(trot.step_duration, 0.5, places=6)
        self.assertAlmostEqual(trot.duty_factor, 0.5, places=6)
        self.assertFalse(trot.has_double_support)
        # `walk` interleaves STANCE, so each foot is down longer than half the stride.
        walk = derive.gait_cadence(gaits, "walk")
        self.assertTrue(walk.has_double_support)
        self.assertGreater(walk.duty_factor, 0.5)
        # `run` has flight phases, so it is the gait that sets the 1/beta clamp.
        worst_name, worst_duty = derive.smallest_duty_factor(gaits)
        self.assertEqual(worst_name, "run")
        self.assertLess(worst_duty, 0.5)

    def test_every_shipped_robot_keeps_all_three_lists_empty(self):
        """The default-off rule, checked on the files as they ship.

        A heuristic changes the closed loop, and none has been validated in simulation under hardware-like conditions
        on these robots, so every list stays empty until someone opts a robot in deliberately. This is cheap to get
        wrong in a way that is invisible in review - a block of commented-out names is one stray edit away from being
        a block of live ones - and the consequence is a robot that walks differently the next time it is launched.

        It is separate from the coefficients: those ARE derived per robot, and the DRC Atlas ships real values. The
        list is the switch; the block is the tuning.
        """
        for robot in sorted(derive.ROBOTS):
            with self.subTest(robot=robot):
                task_file = os.path.join(
                    REPO_ROOT, derive.ROBOTS[robot]["mpc"], "config/mpc/task.yaml"
                )
                block = derive.load_yaml(task_file).get("locomotion_heuristics")
                self.assertIsNotNone(
                    block, "%s has no locomotion_heuristics block" % robot
                )
                for kind in ("base_pose", "foothold", "wrench"):
                    listed = block.get(kind) or []
                    self.assertEqual(
                        listed,
                        [],
                        "%s ships locomotion_heuristics.%s = %s. New controller features default OFF until they have "
                        "been validated in simulation; comment the names out again."
                        % (robot, kind, listed),
                    )

    def test_the_whole_body_task_file_has_no_block_to_be_inert(self):
        # WBMpcInterface never builds the layer, so a block there would be read by nobody. It is absent rather than
        # empty, and the file says why.
        task_file = os.path.join(
            REPO_ROOT, "robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.yaml"
        )
        self.assertNotIn("locomotion_heuristics:", open(task_file).read())

    def test_an_unknown_gait_is_reported_rather_than_guessed(self):
        gaits = derive.load_yaml(
            os.path.join(
                REPO_ROOT, "humanoid_nmpc/humanoid_common_mpc/config/command/gait.yaml"
            )
        )
        with self.assertRaises(SystemExit):
            derive.gait_cadence(gaits, "moonwalk")


if __name__ == "__main__":
    unittest.main()
