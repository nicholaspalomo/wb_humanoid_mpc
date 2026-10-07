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

"""The contact rows that cannot act on the running MPC say so (finding AC9), from the schema's own annotations.

Under basis-vector contact inputs the wrench cone's soft constraint is never built, so its barrier rows moved nothing,
and the geometry of the same block - which builds the basis generators, one of them the input dimension - only takes
effect at the next start. The schema says both (contacts_config.proto: the barrier's `active_when`, the block's
RELOAD_START_UP), and the labels come from it alone: there is no list of contact keys in the GUI any more. The
classification is checked headless on the DRC Atlas file and on a copy switched to wrench inputs; one test renders the
real tab and needs a display.
"""

import tkinter as tk
import unittest

from humanoid_mpc_config import task_file_pb2

import nproto_textproto
import operator_test_support
from remote_control import config_schema
from remote_control.tk_app import mpc_params_tab

CONE = "contacts.contact_wrench_cone_soft_constraint"
CONE_GEOMETRY = (
    "friction_coefficient",
    "torsional_friction_coefficient",
    "min_normal_force",
    "gripper_force",
    "num_basis_vectors",
)
BASIS_BLOCKS = (
    "contacts.basis_non_negativity_barrier",
    "contacts.basis_scaling_regularization",
)


def _labels(parameterization: str | None = None) -> dict[str, list[str]]:
    """The annotations of every rendered parameter under `contacts` of the Atlas task file, by path.

    Args:
        parameterization: The contact input parameterization to give the file; None: the file's own.

    Returns:
        The annotations (config_schema.annotations()) of each parameter, by its path.
    """
    task = nproto_textproto.load_textproto(
        operator_test_support.ATLAS_TASK_FILE, task_file_pb2.TaskFile
    )
    if parameterization is not None:
        task.contact_input_parameterization = parameterization
    return {
        spec.path: config_schema.annotations(spec, task)
        for spec in config_schema.parameters(task)
        if spec.path.startswith("contacts.") and spec.renders
    }


def _not_applicable(notes: list[str]) -> bool:
    return any(note.startswith("not applicable") for note in notes)


class ContactAnnotationTest(unittest.TestCase):
    def test_the_atlas_file_runs_basis_vectors(self):
        task = nproto_textproto.load_textproto(
            operator_test_support.ATLAS_TASK_FILE, task_file_pb2.TaskFile
        )
        self.assertEqual(task.contact_input_parameterization, "basis_vectors")

    def test_under_basis_vectors_the_cone_barrier_is_not_applicable_and_its_geometry_restart_only(
        self,
    ):
        labels = _labels()
        for key in ("mu", "delta"):
            with self.subTest(key=key):
                self.assertTrue(_not_applicable(labels[f"{CONE}.{key}"]))
        for key in CONE_GEOMETRY:
            with self.subTest(key=key):
                notes = labels[f"{CONE}.{key}"]
                self.assertIn("restart", notes)
                self.assertFalse(_not_applicable(notes))
        for path, notes in labels.items():
            if path.startswith(BASIS_BLOCKS):
                with self.subTest(path=path):
                    self.assertNotIn("restart", notes)
                    self.assertFalse(_not_applicable(notes))

    def test_under_wrench_inputs_the_cone_barrier_is_live_and_the_basis_blocks_are_not_applicable(
        self,
    ):
        labels = _labels("wrench")
        for key in ("mu", "delta"):
            with self.subTest(key=key):
                notes = labels[f"{CONE}.{key}"]
                self.assertNotIn("restart", notes)
                self.assertFalse(_not_applicable(notes))
        basis = [path for path in labels if path.startswith(BASIS_BLOCKS)]
        self.assertTrue(basis)
        for path in basis:
            with self.subTest(path=path):
                self.assertTrue(_not_applicable(labels[path]))

    def test_the_other_barriers_are_live_and_the_geometry_restart_only(self):
        for parameterization in ("wrench", "basis_vectors"):
            labels = _labels(parameterization)
            for block in (
                "contacts.friction_force_cone_soft_constraint",
                "contacts.contact_moment_xy_soft_constraint",
            ):
                for key in ("mu", "delta"):
                    with self.subTest(
                        parameterization=parameterization, block=block, key=key
                    ):
                        notes = labels[f"{block}.{key}"]
                        self.assertNotIn("restart", notes)
                        self.assertFalse(_not_applicable(notes))
            for path in (
                "contacts.friction_force_cone_soft_constraint.friction_coefficient",
                "contacts.contact_rectangle.x_max",
                "contacts.contact_frame_translation.z",
            ):
                with self.subTest(parameterization=parameterization, path=path):
                    self.assertIn("restart", labels[path])

    def test_the_note_names_the_selection(self):
        labels = _labels()
        self.assertIn(
            "not applicable: contact_input_parameterization is basis_vectors",
            labels[f"{CONE}.mu"],
        )


@operator_test_support.requires_display
class ContactRowRenderTest(unittest.TestCase):
    """The tab renders the annotations into the rows' labels. Needs a display."""

    def test_the_rendered_contact_rows_say_which_ones_do_not_act_live(self):
        root = tk.Tk()
        root.withdraw()
        self.addCleanup(root.destroy)
        tab = mpc_params_tab.MpcParamsTab(
            root,
            task_file=operator_test_support.ATLAS_TASK_FILE,
            enable_online_tuning=False,
        )
        self.assertEqual(tab.contact_input_parameterization(), "basis_vectors")
        self.assertTrue(tab.render_category_containing(f"{CONE}.mu"))
        self.assertIn("not applicable", tab.slider_rows[f"{CONE}.mu"].name)
        self.assertIn("restart", tab.slider_rows[f"{CONE}.num_basis_vectors"].name)
        barrier = tab.slider_rows["contacts.basis_non_negativity_barrier.mu"].name
        self.assertNotIn("not applicable", barrier)
        self.assertNotIn("restart", barrier)


if __name__ == "__main__":
    unittest.main()
