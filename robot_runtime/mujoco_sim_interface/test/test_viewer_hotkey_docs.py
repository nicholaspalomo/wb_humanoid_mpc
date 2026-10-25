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

"""Tests that the viewer hotkey tables of the READMEs match the visualizations of the shipped task files.

A visualization's key works only when the task file lists it in `sim_visualizations`; the tables once promised keys
that no shipped configuration has and left out keys that every one has.
"""

import re
import unittest

_ROOT_README = "README.md"
_DEV_CONTAINER_README = ".devcontainer/README.md"
_TASK_FILES = (
    "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.textproto",
    "robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/task.textproto",
    "robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.textproto",
    "robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto",
    "robot_models/unitree_r1/unitree_r1_centroidal_mpc/config/mpc/task.textproto",
)
# A `sim_visualizations` line with its `[key]` comment, commented out or not.
_VISUALIZATION_LINE = re.compile(
    r'^(#\s*)?sim_visualizations:\s*"([a-z_]+)"\s*#\s*\[([a-z])\]'
)
# A table row of one key: | **`x`** | action | effect |
_KEY_ROW = re.compile(r"^\|\s*\*\*`([a-z0-9])`\*\*\s*\|")
# The keys the viewers handle themselves: the camera, the geometry groups and the markers panel.
_VIEWER_KEYS = frozenset("kp012345")
_LIST_IT = "list it in `sim_visualizations`"


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as file:
        return file.read()


def _shipped_visualizations() -> dict[str, tuple[str, bool]]:
    """The hotkey of each visualization the shipped task files name, with whether any of them lists it uncommented.

    Returns:
        A map from the key to the visualization's name and whether a shipped task file enables it.
    """
    visualizations: dict[str, tuple[str, bool]] = {}
    for path in _TASK_FILES:
        for line in _read(path).splitlines():
            match = _VISUALIZATION_LINE.match(line.strip())
            if match is None:
                continue
            enabled = match.group(1) is None
            name, key = match.group(2), match.group(3)
            previous = visualizations.get(key, (name, False))
            visualizations[key] = (name, previous[1] or enabled)
    return visualizations


def _key_rows(text: str) -> dict[str, str]:
    """The rows of single keys of a hotkey table.

    Args:
        text: The table's Markdown.

    Returns:
        A map from the key to its row.
    """
    rows = {}
    for line in text.splitlines():
        match = _KEY_ROW.match(line)
        if match is not None:
            rows[match.group(1)] = line
    return rows


def _root_table() -> str:
    text = _read(_ROOT_README)
    start = text.index("<!-- LINT.IfChange(viewer_hotkeys) -->")
    return text[start : text.index("LINT.ThenChange", start)]


class ViewerHotkeyDocsTest(unittest.TestCase):

    def test_the_task_files_name_hotkeys(self) -> None:
        self.assertGreaterEqual(len(_shipped_visualizations()), 10)

    def test_every_visualization_key_has_a_row_naming_it(self) -> None:
        rows = _key_rows(_root_table())
        for key, (name, _) in _shipped_visualizations().items():
            self.assertIn(
                key, rows, f"the README has no row for [{key}] {name}"
            )
            self.assertIn(
                f"`{name}",
                rows[key],
                f"the row of [{key}] does not name {name}",
            )

    def test_every_row_is_a_viewer_key_or_a_shipped_visualization(self) -> None:
        visualizations = _shipped_visualizations()
        for key in _key_rows(_root_table()):
            self.assertTrue(
                key in _VIEWER_KEYS or key in visualizations,
                f"the README's [{key}] belongs to no visualization",
            )

    def test_a_visualization_no_shipped_file_enables_says_to_list_it(
        self,
    ) -> None:
        rows = _key_rows(_root_table())
        for key, (name, enabled) in _shipped_visualizations().items():
            if enabled:
                self.assertNotIn(
                    _LIST_IT, rows[key], f"{name} is on in a shipped task file"
                )
            else:
                self.assertIn(
                    _LIST_IT,
                    rows[key],
                    f"{name} is off in every shipped task file: its row must"
                    " say so",
                )

    def test_p_opens_the_markers_panel_of_the_web_page(self) -> None:
        self.assertIn("Markers panel", _key_rows(_root_table())["p"])
        self.assertIn(
            "Markers panel", _key_rows(_read(_DEV_CONTAINER_README))["p"]
        )

    def test_the_dev_container_readme_leaves_the_visualization_keys_to_the_root_readme(
        self,
    ) -> None:
        text = _read(_DEV_CONTAINER_README)
        for key in _key_rows(text):
            self.assertIn(
                key,
                _VIEWER_KEYS,
                f".devcontainer/README.md documents [{key}], a"
                " visualization's key",
            )
        self.assertIn("../README.md#mujoco-3d-viewer-hotkeys--controls", text)


if __name__ == "__main__":
    unittest.main()
