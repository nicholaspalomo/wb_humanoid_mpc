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

"""Tests for state_interpolation.py: states are interpolated on the manifold, through the robot models."""

import unittest

from tools.hooks import check_test_support
from tools.hooks import state_interpolation

_PATH = "humanoid_nmpc/humanoid_common_mpc/src/reference_manager/SwitchedModelReferenceManager.cpp"


def _found(source: str, path: str = _PATH) -> list[tuple[int, int]]:
    return [
        (f.line, f.column)
        for f in check_test_support.findings(state_interpolation.NAME, source, path)
    ]


class GetDesiredStateTest(unittest.TestCase):
    def test_the_one_argument_member_call_is_flagged(self):
        source = (
            "const vector_t a = targetTrajectories.getDesiredState(time);\n"
            "const vector_t b = targets->getDesiredState(\n"
            "    std::max(time, 0.0));\n"
        )
        self.assertEqual(_found(source), [(1, 39), (2, 29)])

    def test_the_robot_models_and_the_reference_managers_overloads_are_not(self):
        source = (
            "const vector_t a = model.getDesiredState(targetTrajectories, time);\n"
            "const vector_t b = referenceManager->getDesiredState(targets, state, time);\n"
            "vector_t getDesiredState(scalar_t time) const;\n"
            "const vector_t c = targets.getDesiredInput(time);\n"
            "// targets.getDesiredState(time) interpolates linearly.\n"
        )
        self.assertEqual(_found(source), [])


class LinearInterpolationTest(unittest.TestCase):
    def test_an_interpolation_of_states_is_flagged(self):
        source = (
            "const vector_t a = LinearInterpolation::interpolate(time, timeTrajectory, stateTrajectory);\n"
            "x = ocs2::LinearInterpolation::interpolate(t, plan.time, plan.state);\n"
        )
        self.assertEqual(_found(source), [(1, 20), (2, 11)])

    def test_an_interpolation_of_inputs_is_not(self):
        source = (
            "*input = LinearInterpolation::interpolate(time, plan.time, plan.input);\n"
            "u = LinearInterpolation::interpolate(time, target.timeTrajectory, target.inputTrajectory);\n"
            "const scalar_t alpha = LinearInterpolation::timeSegment(time, times).second;\n"
        )
        self.assertEqual(_found(source), [])

    def test_a_using_declaration_or_alias_that_hides_the_qualified_call_is_flagged(
        self,
    ):
        # Unqualified, the call no longer reads as LinearInterpolation's, so the input exemption cannot be read.
        source = (
            "using ocs2::LinearInterpolation::interpolate;\n"
            "namespace li = ocs2::LinearInterpolation;\n"
            "using LinearInterpolation::interpolate;\n"
        )
        self.assertEqual(_found(source), [(1, 1), (2, 1), (3, 1)])

    def test_other_using_declarations_are_not(self):
        source = (
            "using LinearInterpolation::index_alpha_t;\n"
            "using ocs2::LinearInterpolation::timeSegment;\n"
            "namespace oc = ocs2::humanoid;\n"
            "using namespace_alias = LinearInterpolation;\n"
        )
        self.assertEqual(_found(source), [])

    def test_the_robot_model_headers_and_other_directories_are_not_read(self):
        source = "const vector_t a = LinearInterpolation::interpolate(time, times, states);\n"
        for path in (
            "humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/common/MpcRobotModelBase.h",
            "humanoid_nmpc/humanoid_wb_mpc/include/humanoid_wb_mpc/common/WBAccelMpcRobotModel.h",
            "robot_runtime/robot_model/src/RobotState.cpp",
            "lib/ocs2/core/src/misc/LinearInterpolation.cpp",
        ):
            with self.subTest(path=path):
                self.assertEqual(_found(source, path), [])

    def test_behaves_like_every_check(self):
        check_test_support.assert_check_behaves(
            self,
            state_interpolation.NAME,
            "const vector_t a = targets.getDesiredState(time);\n",
            _PATH,
            clean="const vector_t a = model.getDesiredState(targets, time);\n",
        )


if __name__ == "__main__":
    unittest.main()
