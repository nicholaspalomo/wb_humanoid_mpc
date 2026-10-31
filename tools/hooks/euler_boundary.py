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

"""Keeps the Euler angles at the boundary of the MPC's formulation.

The formulation reads the base orientation only as a unit quaternion; Euler angles live at the conversion points of
humanoid_nmpc/docs/quaternion_base_orientation/README.md, section 4.2 (the tuning, the references and heuristics, the
display), all of which go through humanoid_common_mpc/orientation/EulerBoundary.h or the robot models' `*EulerZyx`
accessors. This check keeps the Euler API out of the code that must not convert:

  euler-boundary  in the formulation directories of humanoid_nmpc/ - `<package>/src/<dir>/**` and
                  `<package>/include/<package>/<dir>/**` for <dir> one of FORMULATION_DIRS (the dynamics, the costs and
                  constraints, the end-effector kinematics, the ACoM, the parameter updaters, and the helpers the
                  CppAD tapes are recorded through: the Pinocchio model's kinematics and dynamics, the foot contact
                  geometry) - in the formulation files FORMULATION_FILES (the precomputations, the heading-tilt
                  residual BaseOrientation, the taped end-effector kinematics), in the trees FORMULATION_TREES of lib/ocs2
                  (the centroidal model's flow map and mapping, taped by the centroidal MPC), and in the robot-model
                  headers ROBOT_MODEL_HEADERS:
                    - an identifier that names the Euler API: one containing `EulerZyx`, `EulerAnglesZyx` or
                      `ZyxEulerAngles` in any capitalization (`EulerZYX`, `eulerZyxFromQuaternion`,
                      `getEulerzyxBaseRtWorld`, `getEulerAnglesZyxDerivativesFromGlobalAngularVelocity`, ...);
                    - an #include of `RotationDerivativesTransforms.h` or `orientation/EulerBoundary.h`.
                  Comments and string literals are not read.

The robot-model headers are the one place in scope that converts: their boundary lines carry `NOLINT(euler-boundary):
Euler boundary accessor (design section 4.2)` where they name the Euler API (the EulerBoundary.h include and the
`*EulerZyx` accessors). Any other finding is a conversion in the formulation: read the quaternion (`getQuatBaseRtWorldXyzw`), the heading
(`getYawBaseRtWorld`) or the body angular velocity instead.

    python3 -m tools.hooks.lint_code --only euler-boundary
"""

import re

from tools.hooks import check_types
from tools.hooks import cpp_source
from tools.hooks import lint_files

NAME = "euler-boundary"

# LINT.IfChange(euler_boundary)
# The formulation directories, below a package's src/ or include/<package>/.
FORMULATION_DIRS = (
    "dynamics",
    "cost",
    "constraint",
    "end_effector",
    "acom",
    "parameter_update",
    "pinocchio_model",
    "contact",
)
# Formulation files outside those directories: the precomputations the constraints read their kinematics and the
# swing-foot constraint configs from, the heading-tilt residual, and the end-effector kinematics the centroidal MPC
# tapes (only those files of lib/ocs2/pinocchio/pinocchio_interface/, whose other files are not taped).
FORMULATION_FILES = (
    "humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/HumanoidPreComputation.h",
    "humanoid_nmpc/humanoid_common_mpc/src/HumanoidPreComputation.cpp",
    "humanoid_nmpc/humanoid_wb_mpc/include/humanoid_wb_mpc/WBMpcPreComputation.h",
    "humanoid_nmpc/humanoid_wb_mpc/src/WBMpcPreComputation.cpp",
    "humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/orientation/BaseOrientation.h",
    "humanoid_nmpc/humanoid_common_mpc/src/orientation/BaseOrientation.cpp",
    "lib/ocs2/pinocchio/pinocchio_interface/include/ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h",
    "lib/ocs2/pinocchio/pinocchio_interface/src/PinocchioEndEffectorKinematicsCppAd.cpp",
)
# Trees of lib/ocs2 the MPCs tape: the centroidal model's flow map, mapping and kinematics.
FORMULATION_TREES = ("lib/ocs2/pinocchio/centroidal_model/",)
# The robot-model headers: in scope by name; their Euler boundary accessors are marked.
ROBOT_MODEL_HEADERS = (
    "MpcRobotModelBase.h",
    "BasisInputsModelDecorator.h",
    "BasisInputsMappingDecorator.h",
    "CentroidalMpcRobotModel.h",
    "WBAccelMpcRobotModel.h",
)
# LINT.ThenChange(//humanoid_nmpc/docs/quaternion_base_orientation/README.md:euler_boundary, //AGENTS.md:orientation_rules)

_FORMULATION_PATH = re.compile(
    rf"^humanoid_nmpc/(?:.*/)?(?:src|include/[^/]+)/(?:{'|'.join(FORMULATION_DIRS)})/"
)
_EULER_IDENTIFIER = re.compile(
    r"euler_?(?:angles_?)?zyx|zyx_?euler_?angles", re.IGNORECASE
)
_EULER_INCLUDE = re.compile(
    r'^[ \t]*#[ \t]*include(?:_next)?[ \t]*[<"]'
    r'((?:[^>"\n]*/)?RotationDerivativesTransforms\.h|(?:[^>"\n]*/)?orientation/EulerBoundary\.h)[>"]',
    re.MULTILINE,
)


def in_scope(path: str) -> bool:
    """True for a file of a formulation directory, file or tree, or a robot-model header (`path` relative to the root)."""
    path = lint_files.normalize(path)
    if path in FORMULATION_FILES or path.startswith(FORMULATION_TREES):
        return True
    if not path.startswith("humanoid_nmpc/"):
        return False
    return (
        bool(_FORMULATION_PATH.match(path))
        or path.rsplit("/", 1)[-1] in ROBOT_MODEL_HEADERS
    )


def _line_column(source: str, offset: int) -> tuple[int, int]:
    line = source.count("\n", 0, offset) + 1
    return line, offset - (source.rfind("\n", 0, offset) + 1) + 1


def _findings(source: str, path: str) -> list[check_types.Finding]:
    """The Euler identifiers and the Euler headers in the code of `source`, the file at `path`, when it is in scope."""
    if not in_scope(path):
        return []
    findings = []
    for token in cpp_source.code_tokens(source):
        if token.kind == cpp_source.IDENTIFIER and _EULER_IDENTIFIER.search(
            token.text
        ):
            findings.append(
                check_types.Finding(
                    path,
                    token.line,
                    token.col,
                    NAME,
                    f"`{token.text}` is the Euler API, which the formulation"
                    " does not read: the base orientation is a unit"
                    " quaternion.",
                )
            )
    without_comments, _ = cpp_source.mask(source)
    for match in _EULER_INCLUDE.finditer(without_comments):
        line, column = _line_column(source, match.start(1))
        findings.append(
            check_types.Finding(
                path,
                line,
                column,
                NAME,
                f"`{match.group(1)}` is the Euler API, which the formulation"
                " does not include.",
            )
        )
    return sorted(findings)


CHECKS = [
    check_types.Check(
        name=NAME,
        languages=frozenset({check_types.Language.CPP}),
        # lib/ocs2 included: the centroidal model's tree (FORMULATION_TREES) is taped by the centroidal MPC.
        scope=lint_files.Scope.FIRST_PARTY_AND_OCS2,
        check_source=_findings,
        description=(
            "no Euler API (`EulerZyx`, `EulerAnglesZyx`, `ZyxEulerAngles`,"
            " RotationDerivativesTransforms.h, orientation/EulerBoundary.h) in"
            " the formulation directories and files of humanoid_nmpc/, the"
            " taped trees of lib/ocs2 or the robot-model headers but at their"
            " marked boundary accessors"
            " (humanoid_nmpc/docs/quaternion_base_orientation/README.md,"
            " section 4.2)."
        ),
        hint=(
            "Read the quaternion (getQuatBaseRtWorldXyzw), the heading"
            " (getYawBaseRtWorld) or the body angular velocity; convert to Euler"
            " angles only at a conversion point of the design's section 4.2."
        ),
    ),
]
