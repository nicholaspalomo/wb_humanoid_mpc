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

"""
Every tunable parameter of every robot's configuration must be reachable from the GUI.

The MPC parameters tab renders most of its groups by hand, so a key added to a task file is exposed only if somebody
remembers to add a slider for it. This test removes the remembering: it walks the shipped YAML, collects every numeric
leaf, renders every category of the tab and asserts that each leaf either has a slider or appears in the exclusion
table below with a reason. Adding a tunable key therefore fails this test until it is either exposed or explicitly
declared not tunable.

The `contact_planning` block is the one part that is already generic (`_render_contact_planning_block` walks the block
and makes a slider for every numeric scalar it finds), which is why its parameters need no per-key work here.
"""

import os
import unittest
from typing import Dict, Set

from remote_control.tk_app.yaml_editor_utils import load_yaml_safe

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# LINT.IfChange(gui_coverage_tables)
# The tables live on the tab itself: the generic renderer consults them to decide what to draw, and this test
# consults them to decide what may be missing. One source, so they cannot disagree.
from remote_control.tk_app.command_limits_tab import CommandLimitsTab  # noqa: E402
from remote_control.tk_app.mpc_params_tab import MpcParamsTab  # noqa: E402

NOT_TUNABLE_TASK_KEYS: Dict[str, str] = MpcParamsTab.NOT_TUNABLE_KEYS
NOT_TUNABLE_TASK_PREFIXES: Dict[str, str] = MpcParamsTab.NOT_TUNABLE_PREFIXES
# LINT.ThenChange(//humanoid_nmpc/remote_control/remote_control/tk_app/mpc_params_tab.py:category_blocks)

#: Blocks of the reference file that another tab owns, so the command limits tab does not render them.
REFERENCE_OWNED_ELSEWHERE = CommandLimitsTab.OWNED_ELSEWHERE


def numeric_leaves(node, prefix: str = "") -> Dict[str, float]:
    """Every numeric scalar of a parsed YAML document, keyed by its dotted path.

    Booleans are excluded: a slider would write a float back into a bool key and break the next reload, which is the
    same reason the renderers skip them.
    """
    leaves: Dict[str, float] = {}
    if isinstance(node, dict):
        for key, value in node.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            leaves.update(numeric_leaves(value, child))
    elif isinstance(node, bool) or node is None:
        pass
    elif isinstance(node, (int, float)):
        leaves[prefix] = node
    return leaves


def robot_task_files():
    """Every shipped robot task file, with its sibling planner and reference files."""
    found = []
    models_root = os.path.join(REPO_ROOT, "robot_models")
    for dirpath, _dirnames, filenames in os.walk(models_root):
        if "task.yaml" not in filenames or not dirpath.endswith(
            os.path.join("config", "mpc")
        ):
            continue
        task = os.path.join(dirpath, "task.yaml")
        planner = os.path.join(dirpath, "contact_planning.yaml")
        reference = os.path.join(os.path.dirname(dirpath), "command", "reference.yaml")
        found.append(
            (
                task,
                planner if os.path.exists(planner) else None,
                reference if os.path.exists(reference) else None,
            )
        )
    return sorted(found)


class TestGuiParameterCoverage(unittest.TestCase):
    """Every tunable key of every robot reaches a slider, or is declared not tunable."""

    @staticmethod
    def _slider_keys(task_file: str) -> Set[str]:
        """Every slider MpcParamsTab renders for this task file, over all of its categories."""
        import tkinter as tk

        from remote_control.tk_app.mpc_params_tab import MpcParamsTab

        root = tk.Tk()
        root.withdraw()
        try:
            tab = MpcParamsTab(root, task_file=task_file, enable_online_tuning=True)
            keys: Set[str] = set()
            for category in tab.categories:
                tab.active_category.set(category)
                tab._render_active_category()
                # Matrix entries are keyed with the quoted YAML spelling, e.g. Q."(0,0)".
                keys |= {key.replace('"', "") for key in tab.slider_rows}
            return keys
        finally:
            root.destroy()

    def _excluded(self, key: str) -> bool:
        if key in NOT_TUNABLE_TASK_KEYS:
            return True
        return any(
            key == prefix or key.startswith(prefix + ".")
            for prefix in NOT_TUNABLE_TASK_PREFIXES
        )

    def test_every_tunable_task_key_has_a_slider(self):
        for task_file, planner_file, _reference_file in robot_task_files():
            with self.subTest(robot=os.path.relpath(task_file, REPO_ROOT)):
                data = load_yaml_safe(task_file)
                if planner_file:
                    planner = load_yaml_safe(planner_file).get("contact_planning")
                    if planner:
                        data["contact_planning"] = planner
                leaves = numeric_leaves(data)
                self.assertTrue(
                    leaves, "the task file parsed to no numeric parameters at all"
                )

                sliders = self._slider_keys(task_file)
                missing = sorted(
                    key
                    for key in leaves
                    if key not in sliders and not self._excluded(key)
                )
                self.assertEqual(
                    missing,
                    [],
                    "these parameters have no slider in the MPC parameters tab. Add one, or add the key to "
                    "NOT_TUNABLE_TASK_KEYS with the reason it must not be tuned live:\n  "
                    + "\n  ".join(missing),
                )

    def test_the_exclusion_tables_do_not_rot(self):
        """An excluded key that no longer exists, or that has since been exposed, must be removed from the table."""
        every_task_key: Set[str] = set()
        exposed: Set[str] = set()
        for task_file, planner_file, _reference_file in robot_task_files():
            data = load_yaml_safe(task_file)
            if planner_file:
                planner = load_yaml_safe(planner_file).get("contact_planning")
                if planner:
                    data["contact_planning"] = planner
            every_task_key |= set(numeric_leaves(data))
            exposed |= self._slider_keys(task_file)

        for key in NOT_TUNABLE_TASK_KEYS:
            self.assertIn(
                key,
                every_task_key,
                f"'{key}' is excluded but no robot's task file has it any more",
            )
            self.assertNotIn(
                key,
                exposed,
                f"'{key}' is excluded but the GUI now renders a slider for it",
            )

    def test_every_reference_key_has_a_slider(self):
        """The command limits tab renders reference.yaml, and it renders it generically, so every scalar in the file
        is a slider unless another tab owns that block."""
        import tkinter as tk

        for _task_file, _planner_file, reference_file in robot_task_files():
            if reference_file is None:
                continue
            with self.subTest(robot=os.path.relpath(reference_file, REPO_ROOT)):
                leaves = numeric_leaves(load_yaml_safe(reference_file))
                root = tk.Tk()
                root.withdraw()
                try:
                    tab = CommandLimitsTab(root, reference_file=reference_file)
                    rendered = set(tab.slider_rows)
                finally:
                    root.destroy()
                missing = sorted(
                    key
                    for key in leaves
                    if key not in rendered
                    and not any(
                        key == owned or key.startswith(owned + ".")
                        for owned in REFERENCE_OWNED_ELSEWHERE
                    )
                )
                self.assertEqual(
                    missing,
                    [],
                    "these reference parameters reach no GUI:\n  "
                    + "\n  ".join(missing),
                )


if __name__ == "__main__":
    unittest.main()
