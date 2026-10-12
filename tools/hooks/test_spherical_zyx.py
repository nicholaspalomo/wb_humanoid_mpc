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

"""Tests for spherical_zyx.py: no SphericalZYX root joint in the MPC's code."""

import unittest

from tools.hooks import check_test_support
from tools.hooks import spherical_zyx

_PATH = "humanoid_nmpc/humanoid_common_mpc/src/pinocchio_model/createPinocchioModel.cpp"


def _found(source: str, path: str = _PATH) -> list[tuple[int, int]]:
    return [
        (f.line, f.column)
        for f in check_test_support.findings(spherical_zyx.NAME, source, path)
    ]


class SphericalZyxTest(unittest.TestCase):
    def test_the_euler_root_joint_is_flagged_in_every_spelling(self):
        source = (
            "root.addJoint(pinocchio::JointModelSphericalZYX());\n"
            "using Joint = pinocchio::JointModelSphericalZYXTpl<double>;\n"
            "  pinocchio::JointDataSphericalZYX data;\n"
        )
        self.assertEqual(_found(source), [(1, 26), (2, 26), (3, 14)])

    def test_every_directory_of_the_rule_is_read(self):
        source = "pinocchio::JointModelSphericalZYX joint;\n"
        for path in (
            "humanoid_nmpc/humanoid_wb_mpc/test/testX.cpp",
            "robot_runtime/robot_model/src/RobotState.cpp",
            "lib/ocs2/pinocchio/centroidal_model/src/FactoryFunctions.cpp",
        ):
            with self.subTest(path=path):
                self.assertEqual(_found(source, path), [(1, 12)])

    def test_other_directories_comments_strings_and_the_quaternion_root_are_not(self):
        self.assertEqual(
            _found(
                "pinocchio::JointModelSphericalZYX j;\n",
                "lib/ocs2/robotic_tools/src/a.cpp",
            ),
            [],
        )
        self.assertEqual(
            _found("pinocchio::JointModelSphericalZYX j;\n", "tools/config_dump/a.cpp"),
            [],
        )
        source = (
            "// The old SphericalZYX root was singular at a pitch of 90 degrees.\n"
            "/* JointModelSphericalZYX */\n"
            'const char kName[] = "SphericalZYX";\n'
            "root.addJoint(pinocchio::JointModelSpherical());\n"
        )
        self.assertEqual(_found(source), [])

    def test_behaves_like_every_check(self):
        check_test_support.assert_check_behaves(
            self,
            spherical_zyx.NAME,
            "pinocchio::JointModelSphericalZYX joint;\n",
            _PATH,
            clean="pinocchio::JointModelSpherical joint;\n",
        )


class RepositoryTest(unittest.TestCase):
    def test_the_check_lists_the_directories_of_the_rule(self):
        self.assertEqual(
            spherical_zyx.DIRS,
            (
                "humanoid_nmpc/",
                "robot_runtime/",
                "lib/ocs2/pinocchio/centroidal_model/",
            ),
        )


if __name__ == "__main__":
    unittest.main()
