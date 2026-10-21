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

"""Tests for euler_boundary.py: the Euler API stays out of the formulation and at the robot models' boundary."""

import unittest

from tools.hooks import check_test_support
from tools.hooks import euler_boundary

_COST = "humanoid_nmpc/humanoid_common_mpc/src/cost/StateInputQuadraticCost.cpp"


def _found(source: str, path: str = _COST) -> list[tuple[int, int]]:
    return [
        (f.line, f.column)
        for f in check_test_support.findings(euler_boundary.NAME, source, path)
    ]


class EulerBoundaryTest(unittest.TestCase):
    def test_the_euler_api_is_flagged_in_every_capitalization(self):
        source = (
            "const vector3_t a = model.getBaseOrientationEulerZyx(state);\n"
            "const vector3_t b = quaternionToEulerZYX(xi);\n"
            "const vector3_t c = eulerZyxFromQuaternion(xi);\n"
            "const matrix3_t d = getRotationMatrixFromZyxEulerAngles(c);\n"
            "const vector3_t e = getEulerAnglesZyxDerivativesFromGlobalAngularVelocity(c, w);\n"
        )
        self.assertEqual(_found(source), [(1, 27), (2, 21), (3, 21), (4, 21), (5, 21)])

    def test_the_euler_headers_are_flagged(self):
        source = (
            '#include "humanoid_common_mpc/orientation/EulerBoundary.h"\n'
            '#include "ocs2_robotic_tools/common/RotationDerivativesTransforms.h"\n'
            '#include "humanoid_common_mpc/orientation/BaseOrientation.h"\n'
            '#include "my/EulerBoundary.h"\n'
        )
        self.assertEqual(_found(source), [(1, 11), (2, 11)])

    def test_every_formulation_directory_and_robot_model_header_is_read(self):
        source = "const vector3_t a = eulerZyxFromQuaternion(xi);\n"
        for path in (
            "humanoid_nmpc/humanoid_centroidal_mpc/src/dynamics/CentroidalDynamicsAD.cpp",
            "humanoid_nmpc/humanoid_wb_mpc/include/humanoid_wb_mpc/cost/X.h",
            "humanoid_nmpc/humanoid_common_mpc/src/constraint/sub/Y.cpp",
            "humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/end_effector/Z.h",
            "humanoid_nmpc/humanoid_common_mpc/src/acom/AngularCenterOfMass.cpp",
            "humanoid_nmpc/humanoid_centroidal_mpc/src/parameter_update/BasisInputsCostApplier.cpp",
            "humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/common/MpcRobotModelBase.h",
            "humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/common/BasisInputsModelDecorator.h",
            "humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/common/BasisInputsMappingDecorator.h",
            "humanoid_nmpc/humanoid_centroidal_mpc/include/humanoid_centroidal_mpc/common/CentroidalMpcRobotModel.h",
            "humanoid_nmpc/humanoid_wb_mpc/include/humanoid_wb_mpc/common/WBAccelMpcRobotModel.h",
        ):
            with self.subTest(path=path):
                self.assertEqual(_found(source, path), [(1, 21)])

    def test_every_directory_the_tapes_are_recorded_through_is_read(self):
        # The helpers of the taped flow maps and terms: an Euler conversion there would reach a CppAD tape.
        flagged = (
            "const vector3_t a = getEulerAnglesZyxDerivativesFromGlobalAngularVelocity<SCALAR_T>(c, w);\n"
            '#include "ocs2_robotic_tools/common/RotationDerivativesTransforms.h"\n'
        )
        clean = "const vector4_t xi = model.getBaseOrientation(state);\n"
        for path in (
            "humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h",
            "humanoid_nmpc/humanoid_common_mpc/src/pinocchio_model/DynamicsHelperFunctions.cpp",
            "humanoid_nmpc/humanoid_wb_mpc/src/pinocchio_model/X.cpp",
            "humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/contact/FootprintCornerHeights.h",
            "humanoid_nmpc/humanoid_common_mpc/src/contact/FootprintCornerHeights.cpp",
            "humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/HumanoidPreComputation.h",
            "humanoid_nmpc/humanoid_common_mpc/src/HumanoidPreComputation.cpp",
            "lib/ocs2/pinocchio/centroidal_model/src/ModelHelperFunctions.cpp",
            "lib/ocs2/pinocchio/centroidal_model/include/ocs2_centroidal_model/implementation/ModelHelperFunctionsImpl.h",
            # The whole-body precomputation the swing-foot constraints read their configs from, the heading-tilt
            # residual, and the end-effector kinematics the centroidal MPC tapes.
            "humanoid_nmpc/humanoid_wb_mpc/include/humanoid_wb_mpc/WBMpcPreComputation.h",
            "humanoid_nmpc/humanoid_wb_mpc/src/WBMpcPreComputation.cpp",
            "humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/orientation/BaseOrientation.h",
            "humanoid_nmpc/humanoid_common_mpc/src/orientation/BaseOrientation.cpp",
            "lib/ocs2/pinocchio/pinocchio_interface/include/ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h",
            "lib/ocs2/pinocchio/pinocchio_interface/src/PinocchioEndEffectorKinematicsCppAd.cpp",
        ):
            with self.subTest(path=path):
                self.assertEqual(_found(flagged, path), [(1, 21), (2, 11)])
                self.assertEqual(_found(clean, path), [])

    def test_neighbors_of_the_taped_directories_are_not_read(self):
        source = "const vector3_t a = eulerZyxFromQuaternion(xi);\n"
        for path in (
            # contact_planning/ is not contact/: the planner turns its heading command into Euler targets.
            "humanoid_nmpc/humanoid_common_mpc/src/contact_planning/ContactPlanner.cpp",
            "humanoid_nmpc/humanoid_common_mpc/src/HumanoidPreComputationTest.cpp",
            "lib/ocs2/pinocchio/pinocchio_interface/src/PinocchioInterface.cpp",
            "lib/ocs2/robotic_tools/src/common/RotationTransforms.cpp",
        ):
            with self.subTest(path=path):
                self.assertEqual(_found(source, path), [])

    def test_the_conversion_points_comments_and_strings_are_not_read(self):
        source = "const vector3_t a = eulerZyxFromQuaternion(xi);\n"
        for path in (
            "humanoid_nmpc/humanoid_common_mpc/src/reference_manager/SwitchedModelReferenceManager.cpp",
            "humanoid_nmpc/humanoid_common_mpc/src/orientation/EulerBoundary.cpp",
            "humanoid_nmpc/humanoid_centroidal_mpc/src/command/CentroidalMpcTargetTrajectoriesCalculator.cpp",
            "humanoid_nmpc/humanoid_common_mpc_app/visualization/src/TelemetryBuilder.cpp",
            "humanoid_nmpc/humanoid_common_mpc/test/cost/testCost.cpp",
            "lib/ocs2/core/src/cost/QuadraticStateCost.cpp",
        ):
            with self.subTest(path=path):
                self.assertEqual(_found(source, path), [])
        comments = (
            "// eulerZyxFromQuaternion() is the display's, not the cost's.\n"
            '/* #include "humanoid_common_mpc/orientation/EulerBoundary.h" */\n'
            'const char kName[] = "getBaseOrientationEulerZyx";\n'
            "const vector4_t xi = model.getBaseOrientation(state);\n"
        )
        self.assertEqual(_found(comments), [])

    def test_behaves_like_every_check(self):
        check_test_support.assert_check_behaves(
            self,
            euler_boundary.NAME,
            "const vector3_t a = eulerZyxFromQuaternion(xi);\n",
            _COST,
            clean="const vector4_t xi = model.getBaseOrientation(state);\n",
        )


class RepositoryTest(unittest.TestCase):
    def test_the_check_lists_the_directories_and_headers_of_the_rule(self):
        self.assertEqual(
            euler_boundary.FORMULATION_DIRS,
            (
                "dynamics",
                "cost",
                "constraint",
                "end_effector",
                "acom",
                "parameter_update",
                "pinocchio_model",
                "contact",
            ),
        )
        self.assertEqual(len(euler_boundary.FORMULATION_FILES), 8)
        self.assertEqual(
            euler_boundary.FORMULATION_TREES, ("lib/ocs2/pinocchio/centroidal_model/",)
        )
        self.assertEqual(len(euler_boundary.ROBOT_MODEL_HEADERS), 5)
        for header in euler_boundary.ROBOT_MODEL_HEADERS:
            with self.subTest(header=header):
                self.assertTrue(header.endswith(".h"))


if __name__ == "__main__":
    unittest.main()
