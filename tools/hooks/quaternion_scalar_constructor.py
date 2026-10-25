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

"""Keeps the (w, x, y, z) trap of Eigen's quaternion constructor out of the MPC.

Eigen's `Quaternion(w, x, y, z)` takes the scalar part FIRST, while its `coeffs()`, its constructor from a 4-vector and
every quaternion block of the MPC's state store (x, y, z, w), the scalar part LAST
(humanoid_nmpc/docs/quaternion_base_orientation/README.md, section 2.1). A four-scalar construction with the state's
order silently builds another rotation. This check flags it under humanoid_nmpc/:

  quaternion-scalar-constructor  a construction of `quaternion_t`, `ad_quaternion_t`, `QUATERNION_T<...>`,
                                 `Eigen::Quaternion<...>`, `Eigen::Quaterniond` or `Eigen::Quaternionf` from four
                                 arguments, as a temporary, `T(w, x, y, z)` or `T{w, x, y, z}`, or as a variable,
                                 `T q(w, x, y, z)` or `T q{w, x, y, z}`. A nested `Quaternion` without template
                                 arguments (`pinocchio::SE3::Quaternion`) and the file's own aliases of a quaternion
                                 type (`using Q = Eigen::Quaterniond;`, `typedef ... Q;`) count as quaternion types.
                                 Files in an `orientation/` directory, where
                                 the quaternion helpers live, are not read; neither are comments and string literals.

Build it from the coefficient vector instead, in the storage order: `quaternion_t(vector4_t(x, y, z, w))`.

    python3 -m tools.hooks.lint_code --only quaternion-scalar-constructor
"""

from tools.hooks import check_types
from tools.hooks import cpp_source
from tools.hooks import lint_files

NAME = "quaternion-scalar-constructor"

# The quaternion types, written without template arguments, and those written with them.
_PLAIN_TYPES = frozenset(
    {"quaternion_t", "ad_quaternion_t", "Quaterniond", "Quaternionf"}
)
_TEMPLATE_TYPES = frozenset({"QUATERNION_T", "Quaternion"})
_KEYWORDS_OF_A_DECLARATION = frozenset(
    {"const", "volatile", "typename", "class", "struct"}
)


def in_scope(path: str) -> bool:
    """True for a file under humanoid_nmpc/ outside an orientation/ directory (`path` relative to the root)."""
    path = lint_files.normalize(path)
    return (
        path.startswith("humanoid_nmpc/")
        and "orientation" not in path.split("/")[:-1]
    )


def _declares_parameters(argument: tuple[cpp_source.Token, ...]) -> bool:
    """True when `argument` reads as a parameter declaration (`scalar_t w`, `const T& x`), not as an expression."""
    if argument and argument[0].text in _KEYWORDS_OF_A_DECLARATION:
        return True
    for first, second in zip(argument, argument[1:]):
        if (
            first.kind == cpp_source.IDENTIFIER
            and second.kind == cpp_source.IDENTIFIER
        ):
            return True
    return False


def _type_end(
    tokens: tuple[cpp_source.Token, ...],
    k: int,
    aliases: frozenset[str] = frozenset(),
) -> int | None:
    """The index of the last token of the quaternion type that starts at tokens[k], or None when none starts there.

    Args:
      tokens: The code tokens of the file.
      k: The index of the type's first token.
      aliases: The file's own names of a quaternion type.

    Returns:
      The index of the type's last token, or None.
    """
    token = tokens[k]
    if token.kind != cpp_source.IDENTIFIER:
        return None
    if token.text in _PLAIN_TYPES or token.text in aliases:
        return k
    nested = (
        token.text == "Quaternion"
        and k > 0
        and tokens[k - 1].text == "::"
        and (k + 1 >= len(tokens) or tokens[k + 1].text != "<")
    )
    if nested:
        return k
    if (
        token.text in _TEMPLATE_TYPES
        and k + 1 < len(tokens)
        and tokens[k + 1].text == "<"
    ):
        close = cpp_source.matching_angle(tokens, k + 1)
        return close if close < len(tokens) else None
    return None


def _names_a_quaternion_type(
    tokens: tuple[cpp_source.Token, ...], start: int, stop: int
) -> bool:
    """True when tokens[start:stop] is a (namespace-qualified) quaternion type and nothing else."""
    for k in range(start, stop):
        if _type_end(tokens, k) == stop - 1:
            return all(
                token.kind == cpp_source.IDENTIFIER or token.text == "::"
                for token in tokens[start:k]
            )
    return False


def _quaternion_aliases(tokens: tuple[cpp_source.Token, ...]) -> frozenset[str]:
    """The names the file gives a quaternion type: `using X = <type>;` and `typedef <type> X;`."""
    aliases = set()
    for k, token in enumerate(tokens):
        if token.text not in ("using", "typedef"):
            continue
        end = next(
            (j for j in range(k, len(tokens)) if tokens[j].text == ";"),
            len(tokens),
        )
        if (
            token.text == "using"
            and end - k >= 4
            and tokens[k + 2].text == "="
            and _names_a_quaternion_type(tokens, k + 3, end)
        ):
            aliases.add(tokens[k + 1].text)
        elif (
            token.text == "typedef"
            and end - k >= 3
            and _names_a_quaternion_type(tokens, k + 1, end - 1)
        ):
            aliases.add(tokens[end - 1].text)
    return frozenset(aliases)


def _findings(source: str, path: str) -> list[check_types.Finding]:
    """The four-argument quaternion constructions in the code of `source`, the file at `path`, when it is in scope."""
    if not in_scope(path):
        return []
    tokens = cpp_source.code_tokens(source)
    aliases = _quaternion_aliases(tokens)
    findings = []
    for k in range(len(tokens)):
        end = _type_end(tokens, k, aliases)
        if end is None or end + 1 >= len(tokens):
            continue
        opening = end + 1
        declarator = tokens[opening].kind == cpp_source.IDENTIFIER
        if declarator:
            opening += 1
        if opening >= len(tokens) or tokens[opening].text not in ("(", "{"):
            continue
        arguments = cpp_source.call_arguments(tokens, opening)
        if arguments is None or len(arguments) != 4:
            continue
        if declarator and any(
            _declares_parameters(argument) for argument in arguments
        ):
            continue  # A function of four parameters that returns a quaternion.
        findings.append(
            check_types.Finding(
                path,
                tokens[k].line,
                tokens[k].col,
                NAME,
                f"`{tokens[k].text}` from four scalars takes (w, x, y, z), the"
                " scalar part first, while the state stores (x, y, z, w);"
                " build it from the coefficients, quaternion_t(vector4_t(x, y,"
                " z, w)).",
            )
        )
    return findings


CHECKS = [
    check_types.Check(
        name=NAME,
        languages=frozenset({check_types.Language.CPP}),
        scope=lint_files.Scope.FIRST_PARTY,
        check_source=_findings,
        description=(
            "no quaternion built from four scalars under humanoid_nmpc/ outside"
            " orientation/: Eigen's (w, x, y, z) order is not the state's (x,"
            " y, z, w)"
            " (humanoid_nmpc/docs/quaternion_base_orientation/README.md)."
        ),
        hint=(
            "Build the quaternion from its coefficient vector in the storage"
            " order, quaternion_t(vector4_t(x, y, z, w)), or use a helper of"
            " humanoid_common_mpc/orientation/."
        ),
    ),
]
