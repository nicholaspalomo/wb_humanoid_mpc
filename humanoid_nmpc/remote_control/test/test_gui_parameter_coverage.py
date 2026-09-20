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

"""Every tunable parameter of every robot's configuration reaches the GUI, because the GUI is built from the YAML.

There are no lists of parameter names anywhere in the tabs: `yaml_param_tree` reads a configuration, every numeric leaf
becomes a slider, its label is the key followed by the trailing comment the file carries next to it, and its range
comes from its own
magnitude. This test holds that property from the outside - it walks the shipped YAML itself and asserts the tabs
render a slider for every numeric leaf - so that the day someone reintroduces a hardcoded list and it falls behind the
file, the test fails rather than the parameter quietly disappearing from the GUI.
"""

import os
import unittest
from typing import Set

from remote_control.tk_app.yaml_editor_utils import load_yaml_safe
from remote_control.tk_app.yaml_param_tree import tunables as read_tunables

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def robot_configs():
    """Every shipped robot task file, with its sibling planner and reference files."""
    found = []
    for dirpath, _dirnames, filenames in os.walk(
        os.path.join(REPO_ROOT, "robot_models")
    ):
        if "task.yaml" not in filenames or not dirpath.endswith(
            os.path.join("config", "mpc")
        ):
            continue
        planner = os.path.join(dirpath, "contact_planning.yaml")
        reference = os.path.join(os.path.dirname(dirpath), "command", "reference.yaml")
        found.append(
            (
                os.path.join(dirpath, "task.yaml"),
                planner if os.path.exists(planner) else None,
                reference if os.path.exists(reference) else None,
            )
        )
    return sorted(found)


def normalised(keys) -> Set[str]:
    """Slider keys without the quoting the YAML writer needs, so they compare against the parsed key paths."""
    return {key.replace('"', "") for key in keys}


class TestGuiParameterCoverage(unittest.TestCase):
    def test_every_task_parameter_has_a_slider(self):
        import tkinter as tk

        from remote_control.tk_app.mpc_params_tab import MpcParamsTab

        for task_file, planner_file, _reference in robot_configs():
            with self.subTest(robot=os.path.relpath(task_file, REPO_ROOT)):
                expected = {tunable.dotted for tunable in read_tunables(task_file)}
                if planner_file:
                    planner = load_yaml_safe(planner_file).get("contact_planning")
                    if planner:
                        expected |= {
                            tunable.dotted
                            for tunable in read_tunables(
                                planner_file, root={"contact_planning": planner}
                            )
                        }
                self.assertTrue(
                    expected, "the task file parsed to no numeric parameters at all"
                )

                root = tk.Tk()
                root.withdraw()
                try:
                    tab = MpcParamsTab(
                        root, task_file=task_file, enable_online_tuning=True
                    )
                    rendered: Set[str] = set()
                    for category in tab.categories:
                        tab.active_category.set(category)
                        tab._render_active_category()
                        rendered |= normalised(tab.slider_rows)
                finally:
                    root.destroy()

                missing = sorted(expected - rendered)
                self.assertEqual(
                    missing,
                    [],
                    "these parameters reach no slider:\n  " + "\n  ".join(missing),
                )

    def test_every_reference_parameter_has_a_slider(self):
        import tkinter as tk

        from remote_control.tk_app.command_limits_tab import CommandLimitsTab

        for _task, _planner, reference_file in robot_configs():
            if reference_file is None:
                continue
            with self.subTest(robot=os.path.relpath(reference_file, REPO_ROOT)):
                expected = {tunable.dotted for tunable in read_tunables(reference_file)}
                root = tk.Tk()
                root.withdraw()
                try:
                    tab = CommandLimitsTab(root, reference_file=reference_file)
                    rendered = normalised(tab.slider_rows)
                finally:
                    root.destroy()
                missing = sorted(expected - rendered)
                self.assertEqual(
                    missing,
                    [],
                    "these reference parameters reach no slider:\n  "
                    + "\n  ".join(missing),
                )


class TestGeneratedFromTheFile(unittest.TestCase):
    """The GUI follows the configuration, including parts of it that did not exist when this code was written."""

    def test_an_invented_block_and_key_are_rendered(self):
        import shutil
        import tempfile
        import tkinter as tk

        from remote_control.tk_app.mpc_params_tab import MpcParamsTab

        source = os.path.join(
            REPO_ROOT,
            "robot_models",
            "drc_atlas",
            "drc_atlas_centroidal_mpc",
            "config",
            "mpc",
            "task.yaml",
        )
        with tempfile.TemporaryDirectory() as tmp:
            task_file = os.path.join(tmp, "task.yaml")
            shutil.copy(source, task_file)
            with open(task_file, "a") as out:
                out.write(
                    "\naBlockInventedByThisTest:\n  someWeight: 7.0  # a label only this test knows\n"
                )

            root = tk.Tk()
            root.withdraw()
            try:
                tab = MpcParamsTab(root, task_file=task_file, enable_online_tuning=True)
                self.assertIn(
                    "aBlockInventedByThisTest",
                    tab.categories,
                    "a new block must become a category",
                )
                tab.active_category.set("aBlockInventedByThisTest")
                tab._render_active_category()
                key = "aBlockInventedByThisTest.someWeight"
                self.assertIn(key, tab.slider_rows, "a new key must become a slider")
                # And its label is the key followed by the comment the file carries, not anything written in
                # Python. The key half is what makes a slider findable against the file it edits; the comment half is
                # what the number means, which a key like "(2,2)" never says on its own.
                self.assertEqual(
                    tab.slider_rows[key].name,
                    "someWeight [a label only this test knows]",
                )
            finally:
                root.destroy()


if __name__ == "__main__":
    unittest.main()
