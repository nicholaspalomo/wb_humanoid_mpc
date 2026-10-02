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

"""The contact sliders that cannot act on the running MPC say so (finding AC9).

Under basis-vector contact inputs the wrench cone's soft constraint is never built, so the barrier sliders of
contacts.contactWrenchConeSoftConstraint moved nothing, and the geometry keys of the same block - which build the
basis generators, one of them the input dimension - only take effect at the next start. The generic renderer labeled
all of them as live controls. The classification is a pure function, tested here headless; one test renders the real
tab and is skipped where there is no display.
"""

import os
import re
import shutil
import tempfile
import unittest

from operator_test_support import requires_display
from remote_control.tk_app import yaml_param_tree
from remote_control.tk_app.mpc_params_tab import (
    BASIS_VECTOR_CONTACT_INPUTS,
    CONTACT_INPUT_PARAMETERIZATION_KEY,
    WRENCH_CONTACT_INPUTS,
    _contact_slider_label,
    contact_slider_annotation,
)
from remote_control.tk_app.yaml_editor_utils import load_yaml_safe

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
ATLAS_TASK_FILE = os.path.join(
    REPO_ROOT, "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml"
)
G1_TASK_FILE = os.path.join(
    REPO_ROOT, "robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.yaml"
)
CONTACT_INPUT_PARAMETERIZATION_HEADER = os.path.join(
    REPO_ROOT,
    "humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/common/ContactInputParameterization.h",
)
UPDATER_SOURCE = os.path.join(
    REPO_ROOT,
    "humanoid_nmpc/humanoid_centroidal_mpc/src/mrt/MpcParameterUpdaterModule.cpp",
)

CONE = "contactWrenchConeSoftConstraint"
CONE_GEOMETRY_KEYS = (
    "frictionCoefficient",
    "torsionalFrictionCoefficient",
    "minNormalForce",
    "gripperForce",
    "numBasisVectors",
)


def _read(path):
    with open(path, "r") as handle:
        return handle.read()


def _contact_paths(task_file):
    """The key paths of every numeric leaf under `contacts` of a task file, as the renderer sees them."""
    return [
        tunable.path
        for tunable in yaml_param_tree.tunables(task_file)
        if tunable.path[0] == "contacts"
    ]


class ContactSliderAnnotationTest(unittest.TestCase):
    def test_the_names_are_the_ones_the_cpp_registry_reads(self):
        source = _read(CONTACT_INPUT_PARAMETERIZATION_HEADER)
        for constant, expected in (
            ("kContactInputParameterizationKey", CONTACT_INPUT_PARAMETERIZATION_KEY),
            ("kWrenchContactInputParameterization", WRENCH_CONTACT_INPUTS),
            ("kBasisVectorsContactInputParameterization", BASIS_VECTOR_CONTACT_INPUTS),
        ):
            with self.subTest(constant=constant):
                match = re.search(r'%s = "([A-Za-z_]+)";' % constant, source)
                self.assertIsNotNone(match, "%s is not in the C++ header" % constant)
                self.assertEqual(expected, match.group(1))

    def test_under_basis_vectors_the_cone_barrier_is_not_applicable_and_its_geometry_restart_only(
        self,
    ):
        # The shipped Atlas runs basis vectors; the classification is checked on the paths its file actually renders.
        self.assertEqual(
            load_yaml_safe(ATLAS_TASK_FILE)[CONTACT_INPUT_PARAMETERIZATION_KEY],
            BASIS_VECTOR_CONTACT_INPUTS,
        )
        paths = _contact_paths(ATLAS_TASK_FILE)
        seen = set()
        for path in paths:
            annotation = contact_slider_annotation(path, BASIS_VECTOR_CONTACT_INPUTS)
            if path[1] == CONE and path[-1] in ("mu", "delta"):
                self.assertIsNotNone(annotation, path)
                self.assertTrue(annotation.startswith("not applicable"), path)
                seen.add(path[-1])
            elif path[1] == CONE:
                self.assertEqual(annotation, "restart", path)
                seen.add(path[-1])
            elif path[1] in ("basisNonNegativityBarrier", "basisScalingRegularization"):
                self.assertIsNone(annotation, "%s is live under basis vectors" % path)
                seen.add(path[1])
        # Positive control: the file renders every one of them, so the loop above checked something.
        self.assertTrue(
            set(CONE_GEOMETRY_KEYS)
            | {"mu", "delta", "basisNonNegativityBarrier", "basisScalingRegularization"}
            <= seen,
            seen,
        )

    def test_under_wrench_inputs_the_cone_barrier_is_live_and_the_basis_blocks_are_not_applicable(
        self,
    ):
        paths = _contact_paths(ATLAS_TASK_FILE)
        for path in paths:
            annotation = contact_slider_annotation(path, WRENCH_CONTACT_INPUTS)
            if path[1] == CONE and path[-1] in ("mu", "delta"):
                self.assertIsNone(annotation, path)
            elif path[1] == CONE:
                self.assertEqual(annotation, "restart", path)
            elif path[1] in ("basisNonNegativityBarrier", "basisScalingRegularization"):
                self.assertIsNotNone(annotation, path)
                self.assertTrue(annotation.startswith("not applicable"), path)

    def test_every_other_slider_is_left_alone(self):
        # Outside `contacts` nothing is annotated, and neither are the live barriers of the other cones.
        for path in (
            ["Q", '"(0,0)"'],
            ["jointLimits", "mu"],
            ["contacts", "frictionForceConeSoftConstraint", "mu"],
            ["contacts", "contactMomentXYSoftConstraint", "delta"],
        ):
            for parameterization in (
                WRENCH_CONTACT_INPUTS,
                BASIS_VECTOR_CONTACT_INPUTS,
            ):
                with self.subTest(path=path, parameterization=parameterization):
                    self.assertIsNone(contact_slider_annotation(path, parameterization))
        # The geometry of the other cone and of the footprint is build-time whatever the parameterization.
        for path in (
            ["contacts", "frictionForceConeSoftConstraint", "frictionCoefficient"],
            ["contacts", "contact_rectangle", "x_max"],
            ["contacts", "contact_frame_translation", "z"],
        ):
            self.assertEqual(
                contact_slider_annotation(path, WRENCH_CONTACT_INPUTS), "restart", path
            )

    def test_a_wrench_robot_without_the_cone_block_gets_no_basis_annotation_it_does_not_render(
        self,
    ):
        # G1 runs wrench inputs with friction_force_cone and carries no basis block: its live barrier stays unmarked.
        self.assertEqual(
            load_yaml_safe(G1_TASK_FILE).get(
                CONTACT_INPUT_PARAMETERIZATION_KEY, WRENCH_CONTACT_INPUTS
            ),
            WRENCH_CONTACT_INPUTS,
        )
        for path in _contact_paths(G1_TASK_FILE):
            annotation = contact_slider_annotation(path, WRENCH_CONTACT_INPUTS)
            if path[-1] in ("mu", "delta"):
                self.assertIsNone(annotation, path)

    def test_the_label_carries_the_annotation(self):
        self.assertEqual(
            _contact_slider_label(
                "mu [relaxed barrier]", ["contacts", CONE, "mu"], WRENCH_CONTACT_INPUTS
            ),
            "mu [relaxed barrier]",
        )
        label = _contact_slider_label(
            "mu [relaxed barrier]",
            ["contacts", CONE, "mu"],
            BASIS_VECTOR_CONTACT_INPUTS,
        )
        self.assertTrue(label.startswith("mu [relaxed barrier] (not applicable"), label)
        self.assertEqual(
            _contact_slider_label(
                "numBasisVectors",
                ["contacts", CONE, "numBasisVectors"],
                WRENCH_CONTACT_INPUTS,
            ),
            "numBasisVectors (restart)",
        )

    def test_the_updater_hot_reloads_every_barrier_the_labels_call_live(self):
        # The labels promise that a barrier's mu and delta act live; the C++ updater is what keeps that promise.
        source = _read(UPDATER_SOURCE)
        reloaded = set(re.findall(r'loadBarrierSection\(pt, "contacts\.(\w+)"', source))
        self.assertIn(CONE, reloaded)
        self.assertIn("basisNonNegativityBarrier", reloaded)
        for block in reloaded:
            with self.subTest(block=block):
                live_somewhere = any(
                    contact_slider_annotation(["contacts", block, "mu"], name) is None
                    for name in (WRENCH_CONTACT_INPUTS, BASIS_VECTOR_CONTACT_INPUTS)
                )
                self.assertTrue(live_somewhere, block)


@requires_display
class ContactSliderRenderTest(unittest.TestCase):
    """The tab renders the annotation into the slider's label. Needs a display."""

    def setUp(self):
        import tkinter as tk

        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest("no display: %s" % error)
        self.root.withdraw()
        self.tmpdir = tempfile.mkdtemp()
        self.task_file = os.path.join(self.tmpdir, "task.yaml")
        shutil.copy2(ATLAS_TASK_FILE, self.task_file)

    def tearDown(self):
        self.root.destroy()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_the_rendered_contact_sliders_say_which_ones_do_not_act_live(self):
        from remote_control.tk_app.mpc_params_tab import MpcParamsTab

        tab = MpcParamsTab(
            self.root, task_file=self.task_file, enable_online_tuning=False
        )
        self.assertEqual(
            tab.contact_input_parameterization(), BASIS_VECTOR_CONTACT_INPUTS
        )
        self.assertTrue(tab.render_category_containing("contacts.%s.mu" % CONE))
        self.assertIn("(not applicable", tab.slider_rows["contacts.%s.mu" % CONE].name)
        self.assertIn(
            "(restart)", tab.slider_rows["contacts.%s.numBasisVectors" % CONE].name
        )
        barrier = tab.slider_rows["contacts.basisNonNegativityBarrier.mu"].name
        self.assertNotIn("(not applicable", barrier)
        self.assertNotIn("(restart)", barrier)


if __name__ == "__main__":
    unittest.main()
