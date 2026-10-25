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

"""Tests for quaternion_scalar_constructor.py: no quaternion from four scalars outside orientation/."""

import unittest

from tools.hooks import check_test_support
from tools.hooks import quaternion_scalar_constructor

_NAME = quaternion_scalar_constructor.NAME
_PATH = "humanoid_nmpc/humanoid_common_mpc_app/visualization/src/RobotStateDecoder.cpp"


def _found(source: str, path: str = _PATH) -> list[tuple[int, int]]:
    return [
        (f.line, f.column)
        for f in check_test_support.findings(_NAME, source, path)
    ]


class QuaternionScalarConstructorTest(unittest.TestCase):
    def test_every_type_and_form_of_a_four_scalar_construction_is_flagged(self):
        source = (
            "quaternion_t a(m.w, m.x, m.y, m.z);\n"
            "return quaternion_t(1.0, 0.0, 0.0, 0.0);\n"
            "const ad_quaternion_t b{w, x, y, z};\n"
            "Eigen::Quaterniond c(w, x,\n"
            "                     y, z);\n"
            "auto d = Eigen::Quaternion<scalar_t>(w, x, y, z);\n"
            "QUATERNION_T<SCALAR_T> e(w, x, y, f(z, 1));\n"
        )
        self.assertEqual(
            _found(source), [(1, 1), (2, 8), (3, 7), (4, 8), (6, 17), (7, 1)]
        )

    def test_other_constructions_and_declarations_are_not(self):
        source = (
            "quaternion_t a(vector4_t(m.x, m.y, m.z, m.w));\nconst quaternion_t"
            " b(rotation);\nquaternion_t c ="
            " quaternion_t::Identity();\nEigen::Quaternion<scalar_t>"
            " d(Eigen::AngleAxis<scalar_t>(angle, axis));\nquaternion_t"
            " make(scalar_t w, scalar_t x, scalar_t y, scalar_t"
            " z);\nquaternion_t make2(const scalar_t& w, scalar_t x, scalar_t"
            " y, scalar_t z) {\nstd::vector<quaternion_t> e(4,"
            " quaternion_t::Identity());\nusing QUATERNION_T ="
            " Eigen::Quaternion<SCALAR_T>;\n// quaternion_t(w, x, y, z) takes"
            ' the scalar part first.\nconst char kText[] = "quaternion_t(w, x,'
            ' y, z)";\n'
        )
        self.assertEqual(_found(source), [])

    def test_a_nested_quaternion_type_and_a_files_own_alias_are_flagged(self):
        source = (
            "pinocchio::SE3::Quaternion a(w, x, y, z);\n"
            "using Quat = Eigen::Quaterniond;\n"
            "Quat b(w, x, y, z);\n"
            "typedef Eigen::Quaternion<double> Q;\n"
            "return Q{w, x, y, z};\n"
            "typename SE3Tpl<SCALAR_T>::Quaternion c(w, x, y, z);\n"
        )
        self.assertEqual(_found(source), [(1, 17), (3, 1), (5, 8), (6, 28)])

    def test_aliases_of_other_types_and_other_uses_are_not(self):
        source = (
            "using Quat = Eigen::Quaterniond;\n"
            "Quat a(vector4_t(x, y, z, w));\n"
            "using Vec = Eigen::Matrix<double, 4, 1>;\n"
            "Vec b(w, x, y, z);\n"
            "pinocchio::SE3::Quaternion c(rotation);\n"
            "Quaternion d(w, x, y, z);\n"
        )
        self.assertEqual(_found(source), [])

    def test_the_orientation_directory_and_other_trees_are_not_read(self):
        source = "quaternion_t a(w, x, y, z);\n"
        for path in (
            "humanoid_nmpc/humanoid_common_mpc/src/orientation/BaseOrientation.cpp",
            "humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/orientation/EulerBoundary.h",
            "robot_runtime/mujoco_sim_interface/src/MujocoSimInterface.cpp",
            "lib/ocs2/core/src/manifold/UnitQuaternionMath.cpp",
        ):
            with self.subTest(path=path):
                self.assertEqual(_found(source, path), [])

    def test_behaves_like_every_check(self):
        check_test_support.assert_check_behaves(
            self,
            _NAME,
            "quaternion_t a(m.w, m.x, m.y, m.z);\n",
            _PATH,
            clean="quaternion_t a(vector4_t(m.x, m.y, m.z, m.w));\n",
        )


if __name__ == "__main__":
    unittest.main()
