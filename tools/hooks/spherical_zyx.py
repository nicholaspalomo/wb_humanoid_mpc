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

"""Keeps the Euler root joint out of the MPC: no `SphericalZYX` in the MPC's code.

The MPC's floating base is Pinocchio's Composite(Translation, Spherical) root on a unit-quaternion state
(humanoid_nmpc/docs/quaternion_base_orientation/README.md, sections 2.2 and 4.1). The SphericalZYX root it replaced
parameterizes the attitude by Euler angles, whose rates are singular at a pitch of ±90° and whose yaw wraps; a model
built on it again would bring the singular velocity map, and its gimbal-locked mass matrix, back into the dynamics.

  spherical-zyx  an identifier containing `SphericalZYX` (`JointModelSphericalZYX`, `JointModelSphericalZYXTpl`, ...)
                 in the C++ code of humanoid_nmpc/, robot_runtime/ and lib/ocs2/pinocchio/centroidal_model/. Comments
                 and string literals are not read: they may name the old root.

The equivalence tests build the Euler root on purpose, as the oracle the quaternion root is compared with; each such
line says so with `NOLINT(spherical-zyx): <reason>`.

    python3 -m tools.hooks.lint_code --only spherical-zyx
"""

from tools.hooks import check_types
from tools.hooks import cpp_source
from tools.hooks import lint_files

NAME = "spherical-zyx"

# The directories whose models must use the quaternion root.
DIRS = (
    "humanoid_nmpc/",
    "robot_runtime/",
    "lib/ocs2/pinocchio/centroidal_model/",
)
_BANNED = "SphericalZYX"


def _findings(source: str, path: str) -> list[check_types.Finding]:
    if not lint_files.normalize(path).startswith(DIRS):
        return []
    return [
        check_types.Finding(
            path,
            token.line,
            token.col,
            NAME,
            f"`{token.text}` is the Euler root joint; the MPC's base is"
            " Composite(Translation, Spherical) on the quaternion state.",
        )
        for token in cpp_source.code_tokens(source)
        if token.kind == cpp_source.IDENTIFIER and _BANNED in token.text
    ]


CHECKS = [
    check_types.Check(
        name=NAME,
        languages=frozenset({check_types.Language.CPP}),
        scope=lint_files.Scope.FIRST_PARTY_AND_OCS2,
        check_source=_findings,
        description=(
            "no SphericalZYX root joint in humanoid_nmpc/, robot_runtime/ or"
            " lib/ocs2/pinocchio/centroidal_model/: the base is a unit"
            " quaternion"
            " (humanoid_nmpc/docs/quaternion_base_orientation/README.md)."
        ),
        hint=(
            "Build the root as pinocchio::JointModelComposite of"
            " JointModelTranslation and JointModelSpherical"
            " (humanoid_common_mpc/pinocchio_model/createPinocchioModel.cpp);"
            " read Euler angles through"
            " humanoid_common_mpc/orientation/EulerBoundary.h."
        ),
    ),
]
