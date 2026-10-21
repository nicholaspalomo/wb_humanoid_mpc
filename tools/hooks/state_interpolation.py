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

"""Keeps the state trajectories of the MPC interpolated on the state manifold.

A state stores the base orientation as a unit quaternion, which a linear interpolation between two knots takes off the
unit sphere and, between a quaternion and its negative, through zero. The robot models interpolate states on the
manifold - slerp on the quaternion block, linear elsewhere (`MpcRobotModelBase::getDesiredState(targets, time)`,
`interpolateStateTrajectory`) - and the policy and the rollouts through the problem's `StateManifold`
(humanoid_nmpc/docs/quaternion_base_orientation/README.md, sections 2.7 and 2.9). This check flags the two linear
interpolations of a state under humanoid_nmpc/ that bypass them:

  state-interpolation  - a member call `.getDesiredState(t)` or `->getDesiredState(t)` with one argument: OCS2's
                         TargetTrajectories::getDesiredState(), which interpolates the state trajectory linearly. The
                         robot model's two-argument and the reference manager's three-argument overloads are not
                         flagged;
                       - a call `LinearInterpolation::interpolate(...)` none of whose arguments names an input (an
                         identifier containing `input`, as in `plan.input` or `target.inputTrajectory`): inputs live in
                         a vector space and are interpolated linearly, PolicySnapshot's included;
                       - a using-declaration of `LinearInterpolation::interpolate` and a namespace alias of
                         `LinearInterpolation`, which hide the qualified call and with it the input exemption: write
                         the qualified call.
                       The robot-model headers (euler_boundary.ROBOT_MODEL_HEADERS), which implement the manifold
                       interpolation from its linear Euclidean part, are not read. Comments and string literals are not
                       read either.

A test that compares with the linear interpolation on purpose (the flat-model contract) says so with
`NOLINT(state-interpolation): <reason>`.

    python3 -m tools.hooks.lint_code --only state-interpolation
"""

from tools.hooks import check_types
from tools.hooks import cpp_source
from tools.hooks import euler_boundary
from tools.hooks import lint_files

NAME = "state-interpolation"


def in_scope(path: str) -> bool:
    """True for a file under humanoid_nmpc/ other than a robot-model header (`path` relative to the repository root)."""
    path = lint_files.normalize(path)
    return (
        path.startswith("humanoid_nmpc/")
        and path.rsplit("/", 1)[-1] not in euler_boundary.ROBOT_MODEL_HEADERS
    )


def _names_an_input(argument: tuple[cpp_source.Token, ...]) -> bool:
    return any(
        token.kind == cpp_source.IDENTIFIER and "input" in token.text.lower()
        for token in argument
    )


def _finding(path: str, token: cpp_source.Token, message: str) -> check_types.Finding:
    return check_types.Finding(path, token.line, token.col, NAME, message)


def _statement_end(tokens: tuple[cpp_source.Token, ...], k: int) -> int:
    """The index of the first `;` at or after tokens[k], or len(tokens) when there is none."""
    for end in range(k, len(tokens)):
        if tokens[end].text == ";":
            return end
    return len(tokens)


def _hides_the_qualified_call(tokens: tuple[cpp_source.Token, ...], k: int) -> bool:
    """Tells whether the statement at tokens[k] hides the qualified `LinearInterpolation::interpolate` call.

    Args:
      tokens: The code tokens of the file.
      k: The index of the statement's `using` or `namespace` token.

    Returns:
      True for `using ...LinearInterpolation::interpolate;` and `namespace X = ...LinearInterpolation;`.
    """
    texts = [token.text for token in tokens[k : _statement_end(tokens, k)]]
    if texts[:1] == ["using"]:
        return texts[-3:] == ["LinearInterpolation", "::", "interpolate"]
    return (
        texts[:1] == ["namespace"]
        and len(texts) >= 4
        and texts[2] == "="
        and texts[-1] == "LinearInterpolation"
    )


def _findings(source: str, path: str) -> list[check_types.Finding]:
    """The linear interpolations of a state in the code of `source`, the file at `path`, when it is in scope."""
    if not in_scope(path):
        return []
    tokens = cpp_source.code_tokens(source)
    findings = []
    for k, token in enumerate(tokens):
        if token.text in ("using", "namespace") and _hides_the_qualified_call(
            tokens, k
        ):
            findings.append(
                _finding(
                    path,
                    token,
                    "a using-declaration or alias of LinearInterpolation hides which calls interpolate states; write "
                    "the qualified `LinearInterpolation::interpolate` call.",
                )
            )
            continue
        if k + 1 >= len(tokens) or tokens[k + 1].text != "(":
            continue
        if (
            token.text == "getDesiredState"
            and k > 0
            and tokens[k - 1].text in (".", "->")
        ):
            arguments = cpp_source.call_arguments(tokens, k + 1)
            if arguments is not None and len(arguments) == 1:
                findings.append(
                    _finding(
                        path,
                        token,
                        "`getDesiredState(t)` interpolates the state trajectory linearly, off the unit quaternion; use "
                        "the robot model's getDesiredState(targets, t), which slerps the orientation.",
                    )
                )
        elif (
            token.text == "interpolate"
            and k >= 2
            and tokens[k - 1].text == "::"
            and tokens[k - 2].text == "LinearInterpolation"
        ):
            arguments = cpp_source.call_arguments(tokens, k + 1)
            if arguments is not None and not any(
                _names_an_input(argument) for argument in arguments
            ):
                findings.append(
                    _finding(
                        path,
                        tokens[k - 2],
                        "`LinearInterpolation::interpolate` of a state takes the quaternion off the unit sphere; use "
                        "the robot model's interpolateStateTrajectory() or the problem's StateManifold.",
                    )
                )
    return findings


CHECKS = [
    check_types.Check(
        name=NAME,
        languages=frozenset({check_types.Language.CPP}),
        scope=lint_files.Scope.FIRST_PARTY,
        check_source=_findings,
        description="no linear interpolation of a state trajectory under humanoid_nmpc/ outside the robot models: no "
        "one-argument getDesiredState(t), no LinearInterpolation::interpolate of anything but inputs, and no "
        "using-declaration or alias that hides that call "
        "(humanoid_nmpc/docs/quaternion_base_orientation/README.md, section 2.9).",
        hint="Interpolate states with MpcRobotModelBase::getDesiredState(targets, t) / interpolateStateTrajectory(), or "
        "oc_data/StateTrajectoryInterpolation.h with the problem's StateManifold.",
    ),
]
