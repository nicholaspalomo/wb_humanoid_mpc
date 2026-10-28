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

"""The plan playback end to end: the hand-written plan certified by certify_plan, then played into an .rrd file."""

import os
import subprocess
import tempfile
import unittest

import numpy as np

from humanoid_rerun_viewer import scene_contract
from multi_contact_playback import plan_playback
import rrd_contents

_SCENARIO = (
    "humanoid_nmpc/humanoid_multi_contact/scenarios/config/box_climb.textproto"
)
_PLAN = (
    "humanoid_nmpc/humanoid_multi_contact/app/testdata/box_climb_sway.textproto"
)
# The knots of the hand-written plan.
_KNOTS = 5
_ROBOT_ROOT = scene_contract.link_path(plan_playback.ROBOT_INSTANCE, "pelvis")


def _play(plan: str, rrd_path: str) -> rrd_contents.RrdContents:
    """Plays `plan` into `rrd_path` and reads the recording back."""
    status = plan_playback.main(
        [
            "--scenario",
            _SCENARIO,
            "--plan",
            plan,
            "--rerun_sink",
            "save",
            "--rrd_path",
            rrd_path,
        ]
    )
    if status != plan_playback.EXIT_OK:
        raise AssertionError(f"plan_playback exited {status}")
    return rrd_contents.RrdContents(rrd_path)


class PlanPlaybackTest(unittest.TestCase):
    """Plays the hand-written plan, certified and not."""

    def setUp(self) -> None:
        # pylint: disable-next=consider-using-with  # Closed by the cleanup below.
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)

    def _path(self, name: str) -> str:
        return os.path.join(self._directory.name, name)

    def test_certified_plan(self) -> None:
        certified = self._path("sway.binpb")
        result = subprocess.run(
            [
                os.environ["CERTIFY_PLAN_BINARY"],
                f"--scenario={_SCENARIO}",
                f"--plan={_PLAN}",
                f"--output={certified}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASSED", result.stdout)
        self.assertTrue(
            os.path.exists(certified[: -len(".binpb")] + ".summary.textproto")
        )

        contents = _play(certified, self._path("certified.rrd"))
        for primitive in ("ground", "box"):
            self.assertIn(
                f"{plan_playback.TERRAIN_ROOT}/{primitive}", contents.entities
            )
        self.assertEqual(
            len(
                contents.entities[_ROBOT_ROOT].times(
                    plan_playback.PLAN_TIMELINE
                )
            ),
            _KNOTS,
        )
        forces = contents.entities[f"{plan_playback.PLAN_ROOT}/contact_forces"]
        self.assertEqual(len(forces.times(plan_playback.PLAN_TIMELINE)), _KNOTS)
        self.assertIn(
            f"{plan_playback.PLAN_ROOT}/support_polygon", contents.entities
        )
        self.assertIn(f"{plan_playback.PLAN_ROOT}/com", contents.entities)
        margin = contents.entities[
            f"{plan_playback.MARGINS_ROOT}/static_margin"
        ]
        self.assertEqual(len(margin.times(plan_playback.PLAN_TIMELINE)), _KNOTS)
        self.assertNotIn(plan_playback.FAILURES_PATH, contents.entities)

    def test_uncertified_plan(self) -> None:
        contents = _play(_PLAN, self._path("uncertified.rrd"))
        self.assertEqual(
            len(
                contents.entities[_ROBOT_ROOT].times(
                    plan_playback.PLAN_TIMELINE
                )
            ),
            _KNOTS,
        )
        self.assertNotIn(
            f"{plan_playback.PLAN_ROOT}/support_polygon", contents.entities
        )
        self.assertNotIn(
            f"{plan_playback.MARGINS_ROOT}/static_margin", contents.entities
        )

    def test_pads_follow_the_kinematics(self) -> None:
        inputs = plan_playback.load_inputs(_SCENARIO, _PLAN)
        pads = plan_playback.pad_positions(inputs, inputs.plan.knots[0])
        # The soles are on the floor, 0.1115 m to each side.
        self.assertAlmostEqual(pads["foot_l_contact"][2], 0.0, places=3)
        self.assertAlmostEqual(pads["foot_l_contact"][1], 0.1115, places=4)
        self.assertAlmostEqual(pads["foot_r_contact"][1], -0.1115, places=4)

    def test_quaternion_between(self) -> None:
        for target in ([0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [1.0, 2.0, -0.5]):
            quaternion = plan_playback.quaternion_between(
                np.array([0.0, 0.0, 1.0]),
                np.array(target),
            )
            rotated = plan_playback.rotation_from_xyzw(quaternion) @ [
                0.0,
                0.0,
                1.0,
            ]
            expected = np.array(target) / np.linalg.norm(target)
            for axis in range(3):
                self.assertAlmostEqual(rotated[axis], expected[axis], places=9)

    def test_refuses_a_missing_plan(self) -> None:
        self.assertEqual(
            plan_playback.main(
                ["--scenario", _SCENARIO, "--plan", self._path("none.binpb")]
            ),
            plan_playback.EXIT_USAGE,
        )


if __name__ == "__main__":
    unittest.main()
