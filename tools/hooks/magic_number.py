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

"""Magic numbers in production C++ are named constants (AGENTS.md, C++ "Magic numbers").

  magic-number  a numeric literal in the code of a production C++ file, other than the exempt ones below. Name it
                instead: `static constexpr T kName = <literal>;` in its class, `constexpr T kName = <literal>;` in the
                unnamed namespace of a .cpp, `inline constexpr T kName = <literal>;` in a header that shares it, with
                the type the literal had, so that the arithmetic stays the same.

Exempt:
- the values 0 and 1, in any spelling (`0`, `0.0`, `0.`, `0x0`, `1`, `1.0`, `1.0f`, `1u`, `1e0`, ...);
- a floating-point spelling of one half or two (`0.5`, `.5`, `5e-1`, `2.0`, `2.`, `2.0f`) that is an operand of `*` or
  `/`, a pure arithmetic factor such as `0.5 * a * t * t`, `2.0 * kPi` or `x / 2.0`. A unary sign in between is allowed
  (`x * -0.5`), and the compound assignments `*=` and `/=` count (`x *= 0.5`). The integer 2 is never exempt: an index
  or a size such as `(2)` or `segment<2>` is named like any other;
- a literal inside the definition of a named constant: a `const` or `constexpr` declaration of a `kName`, with its
  whole initializer, multi-line braced tables included;
- an enumerator's value;
- comments, string and character literals, and preprocessor directives.

The check reads first-party C++ and the OCS2 fork in lib/ocs2 (not lib/ocs2/thirdparty), and leaves out what is not
production code: tests and their support code (lint_files.is_test_path()), `benchmark` directories, the test-only
libraries that live outside them (TEST_ONLY_SOURCES), the generated ACoM weight headers and the clang-tidy fixtures. A
literal that has to stay, such as a published algorithm's table that is not a constexpr array, carries
`// NOLINT(magic-number): <reason>`.

    python3 -m tools.hooks.lint_code --only magic-number
"""

import fractions
import re
from typing import NamedTuple

from tools.hooks import check_types
from tools.hooks import cpp_source
from tools.hooks import lint_files

NAME = "magic-number"

# The sources of the `testonly = True` C++ targets outside the test directories lint_files.is_test_path() knows: single
# files, or whole directories ending in `/`. test_magic_number.py keeps the list in step with the BUILD files.
TEST_ONLY_SOURCES = (
    "humanoid_nmpc/humanoid_mpc_validation/exe/benchmarkMpcSolveMain.cpp",
    "humanoid_nmpc/humanoid_mpc_validation/include/humanoid_mpc_validation/closed_loop/CentroidalClosedLoopDriver.h",
    "humanoid_nmpc/humanoid_mpc_validation/include/humanoid_mpc_validation/closed_loop/ClosedLoopDriver.h",
    "humanoid_nmpc/humanoid_mpc_validation/include/humanoid_mpc_validation/closed_loop/LockstepClosedLoop.h",
    "humanoid_nmpc/humanoid_mpc_validation/include/humanoid_mpc_validation/closed_loop/LockstepOutputs.h",
    "humanoid_nmpc/humanoid_mpc_validation/include/humanoid_mpc_validation/closed_loop/TimeSeriesLabels.h",
    "humanoid_nmpc/humanoid_mpc_validation/include/humanoid_mpc_validation/closed_loop/WholeBodyClosedLoopDriver.h",
    "humanoid_nmpc/humanoid_mpc_validation/src/closed_loop/CentroidalClosedLoopDriver.cpp",
    "humanoid_nmpc/humanoid_mpc_validation/src/closed_loop/ClosedLoopDriver.cpp",
    "humanoid_nmpc/humanoid_mpc_validation/src/closed_loop/ClosedLoopDriverFactory.cpp",
    "humanoid_nmpc/humanoid_mpc_validation/src/closed_loop/LockstepClosedLoop.cpp",
    "humanoid_nmpc/humanoid_mpc_validation/src/closed_loop/LockstepOutputs.cpp",
    "humanoid_nmpc/humanoid_mpc_validation/src/closed_loop/WholeBodyClosedLoopDriver.cpp",
    "robot_runtime/web_viewer/include/web_viewer/JpegDecoder.h",
    "robot_runtime/web_viewer/src/JpegDecoder.cpp",
)

# Directories of benchmarks, which are not production code either.
_BENCHMARK_DIRECTORIES = frozenset({"benchmark", "benchmarks"})

# The multiplicative operators a factor of one half or two may stand next to (the module docstring).
_MULTIPLICATIVE = frozenset({"*", "/"})
_MULTIPLICATIVE_ASSIGNMENT = frozenset({"*=", "/="})
_UNARY_SIGNS = frozenset({"-", "+"})
_FACTORS = frozenset({fractions.Fraction(1, 2), fractions.Fraction(2)})
_EXEMPT_VALUES = frozenset({fractions.Fraction(0), fractions.Fraction(1)})

# What follows a factor after `*=` or `/=` when it is the whole right operand or a factor of it.
_OPERAND_END = frozenset({";", ")", "]", "}", ","}) | _MULTIPLICATIVE
# The keywords that start a statement which declares nothing (`if constexpr (...) return kX;`), and those that declare a
# type rather than a constant. `case` and `default` are labels, which the `:` of the label ends.
_STATEMENT_KEYWORDS = frozenset(
    {
        "return",
        "if",
        "else",
        "throw",
        "co_return",
        "co_yield",
        "while",
        "do",
        "for",
        "switch",
        "goto",
        "typedef",
        "using",
    }
)

# The name of a constant (Google style: a leading `k` and mixed case).
_CONSTANT_NAME = re.compile(r"^k[A-Z0-9]\w*$")
_CONSTANT_KEYWORDS = frozenset({"const", "constexpr", "constinit"})
# What a declarator's name follows: its type (`int`, `T`, `std::array<T, 3>`), a pointer or a reference, or the `::` of
# an out-of-line definition (`constexpr int Foo::kBar = 3;`).
_BEFORE_DECLARATOR = frozenset({">", ">>", "*", "&", "&&", "::"})
# What a declarator's name precedes: its initializer, an array bound, the next declarator or the end.
_AFTER_DECLARATOR = frozenset({"=", "{", "(", "[", ",", ";"})
_OPENING = frozenset({"(", "[", "{"})
_CLOSING = frozenset({")", "]", "}"})

_HEXADECIMAL = re.compile(
    r"^0[xX](?P<whole>[0-9a-fA-F]*)(?:\.(?P<fraction>[0-9a-fA-F]*))?(?:[pP](?P<exponent>[+-]?\d+))?(?P<suffix>[a-zA-Z_]*)$"
)
_BINARY = re.compile(r"^0[bB](?P<digits>[01]+)(?P<suffix>[a-zA-Z_]*)$")
_DECIMAL = re.compile(
    r"^(?P<whole>\d*)(?:(?P<point>\.)(?P<fraction>\d*))?(?:[eE](?P<exponent>[+-]?\d+))?(?P<suffix>[a-zA-Z_]*)$"
)


class Literal(NamedTuple):
    """The value of a C++ numeric literal and whether it is spelled as a floating-point number.

    Attributes:
      value: The exact value, or None when the token is not a literal this parser reads.
      floating: True for a floating-point spelling: a radix point, an exponent or a hexadecimal `p` exponent.
    """

    value: fractions.Fraction | None
    floating: bool


def parse_literal(text: str) -> Literal:
    """Reads the numeric literal `text` (digit separators, prefixes, exponents and suffixes included).

    Args:
      text: The text of a NUMBER token of cpp_source.tokenize().

    Returns:
      Its value and spelling; the value is None for text that is no numeric literal.
    """
    text = text.replace("'", "")
    hexadecimal = _HEXADECIMAL.match(text)
    if hexadecimal:
        whole = hexadecimal.group("whole")
        fraction = hexadecimal.group("fraction")
        exponent = hexadecimal.group("exponent")
        floating = fraction is not None or exponent is not None
        digits = whole + (fraction or "")
        if not digits:
            return Literal(None, floating)
        value = fractions.Fraction(int(digits, 16), 16 ** len(fraction or ""))
        if exponent is not None:
            value *= fractions.Fraction(2) ** int(exponent)
        return Literal(value, floating)
    binary = _BINARY.match(text)
    if binary:
        return Literal(
            fractions.Fraction(int(binary.group("digits"), 2)), False
        )
    decimal = _DECIMAL.match(text)
    if not decimal or not (decimal.group("whole") or decimal.group("fraction")):
        return Literal(None, False)
    whole = decimal.group("whole")
    fraction = decimal.group("fraction") or ""
    exponent = decimal.group("exponent")
    floating = decimal.group("point") is not None or exponent is not None
    if not floating and len(whole) > 1 and whole.startswith("0"):
        # An octal integer.
        if any(digit in "89" for digit in whole):
            return Literal(None, False)
        return Literal(fractions.Fraction(int(whole, 8)), False)
    value = fractions.Fraction(
        int(whole + fraction or "0"), 10 ** len(fraction)
    )
    if exponent is not None:
        value *= fractions.Fraction(10) ** int(exponent)
    return Literal(value, floating)


def in_scope(path: str) -> bool:
    """True for a production C++ file: not a test, a benchmark or a test-only library (`path` relative to the root)."""
    path = lint_files.normalize(path)
    if lint_files.is_test_path(path):
        return False
    if any(part in _BENCHMARK_DIRECTORIES for part in path.split("/")[:-1]):
        return False
    return not any(
        path == source or (source.endswith("/") and path.startswith(source))
        for source in TEST_ONLY_SOURCES
    )


def _is_factor(tokens: tuple[cpp_source.Token, ...], k: int) -> bool:
    """Tells whether the literal tokens[k] is an operand of `*` or `/`, a unary sign allowed between.

    Args:
      tokens: Code tokens (cpp_source.code_tokens()).
      k: The index of the literal.

    Returns:
      True for a factor; after `*=` or `/=` only when the literal is the whole right-hand side or a factor of it
      (`x *= 0.5;`, `x /= 2.0 * y;`), not an operand of a sum (`x *= 2.0 + y;`).
    """
    after = tokens[k + 1].text if k + 1 < len(tokens) else ";"
    if after in _MULTIPLICATIVE:
        return True
    before = k - 1
    if before >= 0 and tokens[before].text in _UNARY_SIGNS:
        before -= 1
    if before < 0:
        return False
    if tokens[before].text in _MULTIPLICATIVE:
        return True
    return (
        tokens[before].text in _MULTIPLICATIVE_ASSIGNMENT
        and after in _OPERAND_END
    )


def is_exempt_literal(tokens: tuple[cpp_source.Token, ...], k: int) -> bool:
    """True when the NUMBER token tokens[k] may stay unnamed by its value and its neighbors alone."""
    literal = parse_literal(tokens[k].text)
    if literal.value is None:
        return False
    if literal.value in _EXEMPT_VALUES:
        return True
    return (
        literal.floating and literal.value in _FACTORS and _is_factor(tokens, k)
    )


def _template_arguments_end(
    tokens: tuple[cpp_source.Token, ...], k: int, stop: int
) -> int | None:
    """The `>` that closes the template argument list opened by the `<` at tokens[k], or None when it opens none.

    A `<` opens one after a name (`std::array<`, `kSize<`) when it closes before `stop`; otherwise it is a comparison.

    Args:
      tokens: The code tokens (cpp_source.code_tokens()).
      k: The index of the token that may open the list.
      stop: The index the list must close before.

    Returns:
      The index of the closing `>`, or None.
    """
    if (
        tokens[k].text != "<"
        or k == 0
        or tokens[k - 1].kind != cpp_source.IDENTIFIER
    ):
        return None
    close = cpp_source.matching_angle(tokens, k)
    return close if close < stop else None


def _is_declarator_name(
    tokens: tuple[cpp_source.Token, ...], k: int, stop: int
) -> bool:
    """Tells whether tokens[k] is a `kName` standing as a declarator.

    Args:
      tokens: Code tokens (cpp_source.code_tokens()).
      k: The index of the name.
      stop: The end of the statement.

    Returns:
      True for a `kName` after a type, `*`, `&`, `::` or the comma of a declarator list, and before its initializer, an
      array bound, another declarator or the end; a variable template's specialization (`kSize<Foo> = 3`) names its
      template arguments in between.
    """
    token = tokens[k]
    if token.kind != cpp_source.IDENTIFIER or not _CONSTANT_NAME.match(
        token.text
    ):
        return False
    if not (
        tokens[k - 1].kind == cpp_source.IDENTIFIER
        or tokens[k - 1].text in _BEFORE_DECLARATOR
        or tokens[k - 1].text == ","
    ):
        return False
    after = k + 1
    if after < stop:
        close = _template_arguments_end(tokens, after, stop)
        if close is not None:
            after = close + 1
    return after >= stop or tokens[after].text in _AFTER_DECLARATOR


def _named_constant_ranges(
    tokens: tuple[cpp_source.Token, ...], start: int, end: int
) -> list[tuple[int, int]]:
    """Finds the parts of tokens[start:end], a statement, that define `const` or `constexpr` constants named `kName`.

    A declaration's declarators are separated by its top-level commas; each one whose name is a `kName` (after a type,
    a `*`, a `&` or a `::`, before an initializer, an array bound, another declarator or the end) is a named constant,
    the declaration's specifiers and template header included for the first one. What stands in front of the
    declaration is not part of it: a label (`case 2:`), or a macro call or an attribute with arguments
    (`HUMANOID_MACRO(3)`, `alignas(64)`). A statement that a keyword starts (`return`, `if constexpr (...)`, `typedef`)
    declares no constant, and a `const` or a `kName` inside a template argument list is the type's, not the
    declaration's.

    Args:
      tokens: Code tokens (cpp_source.code_tokens()).
      start: The index of the statement's first token.
      end: The index of its terminator: its `;`, or the `{` of the code body that ends its header (a lambda's).

    Returns:
      The half-open token ranges of the definitions, in order; empty when the statement defines no named constant.
    """
    stop = min(end, len(tokens))
    ranges = []
    depth = 0
    constant = False
    first = start
    declarator_start: int | None = None
    named = False
    in_initializer = False
    k = start - 1
    while k + 1 < stop:
        k += 1
        token = tokens[k]
        if (
            token.text == "template"
            and k + 1 < stop
            and tokens[k + 1].text == "<"
        ):
            # A template header, whose default arguments are not the declaration's initializer.
            k = cpp_source.matching_angle(tokens, k + 1)
            continue
        if token.text in _OPENING:
            depth += 1
            continue
        if token.text in _CLOSING:
            depth -= 1
            if depth == 0 and not constant:
                # A macro call or an attribute with arguments in front of the declaration.
                first = k + 1
            continue
        if depth != 0:
            continue
        close = _template_arguments_end(tokens, k, stop)
        if close is not None:
            k = close
            continue
        if token.text in _STATEMENT_KEYWORDS:
            return []
        if token.text == ":" and not in_initializer:
            # A label: the declaration starts after it.
            constant = False
            first = k + 1
            continue
        if token.text == "=":
            if not constant:
                return []
            in_initializer = True
            continue
        if token.text == ",":
            if named and declarator_start is not None:
                ranges.append((declarator_start, k + 1))
            named = False
            in_initializer = False
            declarator_start = k + 1
            continue
        if in_initializer or named:
            continue
        if token.text in _CONSTANT_KEYWORDS:
            constant = True
            continue
        if constant and _is_declarator_name(tokens, k, stop):
            named = True
            if declarator_start is None:
                declarator_start = first
    if named and declarator_start is not None:
        ranges.append((declarator_start, min(end + 1, len(tokens))))
    return ranges


def _exempt_by_context(source: str) -> list[bool]:
    """For every code token of `source`, whether it is part of a named constant's definition or of an enumerator."""
    tokens = cpp_source.code_tokens(source)
    exempt = [
        cpp_source.ENUM_SCOPE in scope for scope in cpp_source.scopes(source)
    ]
    for statement in cpp_source.statements(source):
        if statement.body not in (None, cpp_source.CODE_SCOPE):
            continue
        # A statement with a code body is a function's or a lambda's header: a constant a lambda initializes is defined
        # by its header, and the lambda's body is code.
        for first, last in _named_constant_ranges(
            tokens, statement.start, statement.end
        ):
            for k in range(first, last):
                exempt[k] = True
    # A lambda body inside a constant's braced initializer is code, not part of the constant's definition; only the
    # constants it defines itself are exempt.
    for body_open, body_close in cpp_source.code_bodies_in_initializers(source):
        for k in range(body_open + 1, body_close):
            exempt[k] = False
        for first, last in _body_statement_ranges(
            tokens, body_open, body_close
        ):
            for constant_first, constant_last in _named_constant_ranges(
                tokens, first, last
            ):
                for k in range(constant_first, constant_last):
                    exempt[k] = True
    return exempt


def _body_statement_ranges(
    tokens: tuple[cpp_source.Token, ...], body_open: int, body_close: int
) -> list[tuple[int, int]]:
    """Splits a code body into statements at every `;`, `{` and `}`.

    Args:
      tokens: Code tokens (cpp_source.code_tokens()).
      body_open: The index of the body's `{`.
      body_close: The index of its `}`.

    Returns:
      The half-open ranges (first token, terminator) of the non-empty statements, in order.
    """
    ranges = []
    first = body_open + 1
    for k in range(body_open + 1, body_close + 1):
        if tokens[k].text in (";", "{", "}"):
            if k > first:
                ranges.append((first, k))
            first = k + 1
    return ranges


def _findings(source: str, path: str) -> list[check_types.Finding]:
    """The magic numbers in the code of `source`, the file at `path`, when it is production code."""
    if not in_scope(path):
        return []
    tokens = cpp_source.code_tokens(source)
    exempt = _exempt_by_context(source)
    findings = []
    for k, token in enumerate(tokens):
        if token.kind != cpp_source.NUMBER or exempt[k]:
            continue
        if is_exempt_literal(tokens, k):
            continue
        findings.append(
            check_types.Finding(
                path,
                token.line,
                token.col,
                NAME,
                f"magic number `{token.text}`: declare it as a named constant,"
                f" `static constexpr T kName = {token.text};`, and use the"
                " name.",
            )
        )
    return findings


CHECKS = [
    check_types.Check(
        name=NAME,
        languages=frozenset({check_types.Language.CPP}),
        scope=lint_files.Scope.FIRST_PARTY_AND_OCS2,
        check_source=_findings,
        description=(
            "no magic numbers in production C++: a literal other than 0 and 1"
            " (and 0.5 or 2.0 as a factor of `*` or `/`) is a named constant,"
            " outside constant definitions and enumerators (AGENTS.md)."
        ),
        hint=(
            "Name the literal after what it is (kPositionDim,"
            " kDefaultSwingHeight), with the literal's type: `static constexpr"
            " T kName = ...;` in its class, `constexpr` in the .cpp's unnamed"
            " namespace, `inline constexpr` in a header that shares it (the"
            " generic dimensions are in ocs2_core/Dimensions.h). A literal that"
            " must stay carries `// NOLINT(magic-number): <reason>`."
        ),
    ),
]
