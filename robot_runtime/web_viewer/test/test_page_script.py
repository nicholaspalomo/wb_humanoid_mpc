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

"""Tests of the viewer page's script: its input, status line, HUD and panels, run under node.

Each test runs one scenario of test/page_script_harness.js, which loads the <script> of page/index.html against a
small DOM, a fetch the scenario answers and a virtual clock. node is a declared dependency (dependencies.txt), so a
missing node fails the test rather than skipping it.
"""

import os
import shutil
import subprocess
import unittest

# The files as the BUILD file's data puts them in the test's runfiles.
_PAGE = "robot_runtime/web_viewer/page/index.html"
_HARNESS = "robot_runtime/web_viewer/test/page_script_harness.js"
# [s] A scenario runs on a virtual clock and takes well under a second.
_SCENARIO_TIMEOUT_SECONDS = 60


def _node() -> str:
    """The node executable: the one on PATH, else the dev image's."""
    return shutil.which("node") or "/usr/bin/node"


class PageScriptTest(unittest.TestCase):

    def _run_scenario(self, scenario: str) -> None:
        """Runs one scenario through the page script harness in Node.

        Args:
            scenario: The name of the test scenario function to execute in the harness.
        """
        node = _node()
        self.assertTrue(
            os.access(node, os.X_OK),
            f"node is not installed at {node}: install the nodejs package (dependencies.txt)",
        )
        result = subprocess.run(
            [node, _HARNESS, _PAGE, scenario],
            capture_output=True,
            text=True,
            timeout=_SCENARIO_TIMEOUT_SECONDS,
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"scenario {scenario} failed:\n{result.stdout}{result.stderr}",
        )
        self.assertIn(f"PASS {scenario}", result.stdout)

    def test_marker_rows_survive_a_poll(self) -> None:
        self._run_scenario("markerRowsSurviveAPoll")

    def test_an_input_refusal_outlives_the_frame_loop(self) -> None:
        self._run_scenario("anInputRefusalOutlivesTheFrameLoop")

    def test_inputs_keep_their_order_after_motion(self) -> None:
        self._run_scenario("inputsKeepTheirOrderAfterMotion")

    def test_the_frame_age_keeps_growing_without_new_frames(self) -> None:
        self._run_scenario("theFrameAgeKeepsGrowingWithoutNewFrames")

    def test_a_page_before_the_first_frame_says_it_waits(self) -> None:
        self._run_scenario("aPageBeforeTheFirstFrameSaysItWaits")

    def test_the_settings_controls_show_the_renderers_settings(self) -> None:
        self._run_scenario("theSettingsControlsShowTheRenderersSettings")

    def test_a_clamped_or_refused_setting_still_follows_the_renderer(self) -> None:
        self._run_scenario("aClampedOrRefusedSettingStillFollowsTheRenderer")

    def test_drags_are_fractions_of_the_displayed_height(self) -> None:
        self._run_scenario("dragsAreFractionsOfTheDisplayedHeight")

    def test_every_frame_request_names_this_page(self) -> None:
        self._run_scenario("everyFrameRequestNamesThisPage")


if __name__ == "__main__":
    unittest.main()
