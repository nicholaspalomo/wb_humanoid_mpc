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

"""Tests for magic_number.py: production C++ names its numeric literals."""

import ast
import fractions
import glob
import os
import unittest

from tools.hooks import check_test_support
from tools.hooks import lint_files
from tools.hooks import magic_number

_NAME = magic_number.NAME
_PATH = "humanoid_nmpc/humanoid_common_mpc/src/cost/SomeCost.cpp"
_CC_RULE_SUFFIXES = ("cc_binary", "cc_library", "cc_test")
_SOURCE_ATTRIBUTES = ("srcs", "hdrs", "textual_hdrs")


def _found(source: str, path: str = _PATH) -> list[str]:
    """The literals the check reports in `source`, in order."""
    return [
        f.message.split("`")[1]
        for f in check_test_support.findings(_NAME, source, path)
    ]


def _expression(expression: str) -> list[str]:
    """The literals the check reports in a statement that uses `expression` (no constant is being defined)."""
    return _found(f"value = {expression};\n")


def _rule_name(call: ast.Call) -> str:
    """The name of the rule or macro `call` calls: `cc_library` for `cc_library(...)` and `native.cc_library(...)`."""
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return ""


def _is_cc_rule(name: str) -> bool:
    """True for a C++ rule, or a macro named after one (`humanoid_cc_test`)."""
    return name.endswith(_CC_RULE_SUFFIXES)


def _string_list(node: ast.expr) -> list[str]:
    """The string constants of `node`, a list literal or a single string."""
    return [
        leaf.value
        for leaf in ast.walk(node)
        if isinstance(leaf, ast.Constant) and isinstance(leaf.value, str)
    ]


def _glob(root: str, package: str, call: ast.Call) -> set[str]:
    """The files `glob(include, exclude = ...)` of the BUILD file of `package` matches, relative to `root`."""

    def matches(patterns: list[str]) -> set[str]:
        found = set()
        for pattern in patterns:
            for match in glob.glob(
                os.path.join(root, package, pattern), recursive=True
            ):
                if os.path.isfile(match):
                    found.add(
                        lint_files.normalize(os.path.relpath(match, root))
                    )
        return found

    include = matches(_string_list(call.args[0])) if call.args else set()
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
    if "include" in keywords:
        include |= matches(_string_list(keywords["include"]))
    excluded = (
        matches(_string_list(keywords["exclude"]))
        if "exclude" in keywords
        else set()
    )
    return include - excluded


def _sources(root: str, package: str, node: ast.expr) -> set[str]:
    """The C++ files a `srcs` / `hdrs` value of the BUILD file of `package` names: its strings and its globs."""
    sources = set()
    for leaf in ast.walk(node):
        if isinstance(leaf, ast.Call) and _rule_name(leaf) == "glob":
            sources |= _glob(root, package, leaf)
        elif (
            isinstance(leaf, ast.Constant)
            and isinstance(leaf.value, str)
            and "*" not in leaf.value
            and not leaf.value.startswith((":", "//", "@"))
        ):
            sources.add(lint_files.normalize(os.path.join(package, leaf.value)))
    return {
        path for path in sources if path.endswith(lint_files.CPP_EXTENSIONS)
    }


def _test_only_cpp_sources(root: str) -> set[str]:
    """Finds the C++ sources of every testonly C++ target of the first-party and OCS2 BUILD files.

    Args:
      root: The source tree.

    Returns:
      The sources, relative to `root`, of the targets whose `testonly` is a true constant (`True`, `1`) and whose rule
      is a C++ rule or a macro named after one; their sources are string lists and globs.
    """
    sources = set()
    for path in lint_files.repository_files(root):
        if os.path.basename(path) not in ("BUILD", "BUILD.bazel"):
            continue
        if not (lint_files.is_first_party(path) or lint_files.is_ocs2(path)):
            continue
        package = os.path.dirname(path)
        with open(os.path.join(root, path), encoding="utf-8") as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call) and _is_cc_rule(_rule_name(node))
            ):
                continue
            arguments = {
                keyword.arg: keyword.value for keyword in node.keywords
            }
            testonly = arguments.get("testonly")
            if not (
                isinstance(testonly, ast.Constant) and bool(testonly.value)
            ):
                continue
            for attribute in _SOURCE_ATTRIBUTES:
                if attribute in arguments:
                    sources |= _sources(root, package, arguments[attribute])
    return sources


class ParseLiteralTest(unittest.TestCase):
    def test_values_and_spellings(self):
        half = fractions.Fraction(1, 2)
        cases = {
            "0": (0, False),
            "00": (0, False),
            "0x0": (0, False),
            "0b0": (0, False),
            "0.0": (0, True),
            "0.": (0, True),
            ".0": (0, True),
            "0e5": (0, True),
            "1": (1, False),
            "1u": (1, False),
            "1ULL": (1, False),
            "1.0f": (1, True),
            "1.f": (1, True),
            "1e0": (1, True),
            "10.0e-1": (1, True),
            "0x1": (1, False),
            "0b1": (1, False),
            "2": (2, False),
            "2.0": (2, True),
            "2.": (2, True),
            "2e0": (2, True),
            "0.5": (half, True),
            ".5f": (half, True),
            "5e-1": (half, True),
            "0x1p-1": (half, True),
            "0x1.8p1": (3, True),
            "010": (8, False),
            "0x10": (16, False),
            "0xFFu": (255, False),
            "0b1010": (10, False),
            "1'000'000": (1_000_000, False),
            "0'0": (0, False),
            "1.0e-6": (fractions.Fraction(1, 1_000_000), True),
            "1.0000000000000001": (
                fractions.Fraction(
                    10_000_000_000_000_001, 10_000_000_000_000_000
                ),
                True,
            ),
        }
        for text, (value, floating) in cases.items():
            with self.subTest(literal=text):
                literal = magic_number.parse_literal(text)
                self.assertEqual(literal.value, value)
                self.assertEqual(literal.floating, floating)

    def test_text_that_is_no_literal_has_no_value(self):
        for text in ("089", "0x", "1.2.3"):
            with self.subTest(text=text):
                self.assertIsNone(magic_number.parse_literal(text).value)


class ExemptValuesTest(unittest.TestCase):
    def test_zero_and_one_in_any_spelling_are_exempt(self):
        for literal in (
            "0",
            "0.0",
            "0.",
            ".0",
            "0x0",
            "0b0",
            "00",
            "0e5",
            "0.0f",
            "0u",
            "1",
            "1.0",
            "1.",
            "1.f",
            "1.0f",
            "1u",
            "1ul",
            "1LL",
            "0x1",
            "0b1",
            "1e0",
            "10.0e-1",
            "-1",
            "-1.0",
        ):
            with self.subTest(literal=literal):
                self.assertEqual(_expression(f"f({literal})"), [])

    def test_every_other_value_is_flagged(self):
        for literal in (
            "2",
            "3",
            "2u",
            "0x2",
            "0b10",
            "010",
            "0xFF",
            "1'000",
            "1'000'000",
            "1.0e-6",
            "1e-6",
            "1.5",
            "0.25",
            "9.81",
            "3.0f",
            "1.0000000000000001",
        ):
            with self.subTest(literal=literal):
                self.assertEqual(_expression(f"f({literal})"), [literal])

    def test_a_negative_literal_is_flagged_by_its_magnitude(self):
        self.assertEqual(_expression("f(-2)"), ["2"])
        self.assertEqual(_expression("f(-0.5)"), ["0.5"])


class MultiplicativeFactorTest(unittest.TestCase):
    def test_a_half_or_two_spelled_floating_next_to_a_multiplication_or_division_is_exempt(
        self,
    ):
        for expression in (
            "0.5 * a * t * t",
            "2.0 * kPi",
            "x / 2.0",
            "x * 0.5",
            "a + 0.5 * b",
            "0.5 / x",
            "(0.5 * (a + b))",
            "x * -0.5",
            "-0.5 * x",
            "x / +2.0",
            "0.5f * x",
            "2.f * x",
            ".5 * x",
            "2. * x",
            "5e-1 * x",
            "2e0 * x",
            "0x1p-1 * x",
            "x /\n    2.0",
        ):
            with self.subTest(expression=expression):
                self.assertEqual(_expression(expression), [])

    def test_the_compound_multiplicative_assignments_count(self):
        self.assertEqual(
            _found(
                "x *= 0.5;\ny /= 2.0;\nz *= -0.5;\nw *= 0.5 * v;\nf(x *="
                " 2.0);\n"
            ),
            [],
        )
        self.assertEqual(
            _found("x += 0.5;\ny -= 2.0;\nz %= 2;\n"), ["0.5", "2.0", "2"]
        )

    def test_after_a_compound_assignment_an_operand_of_a_sum_is_flagged(self):
        self.assertEqual(_found("x *= 2.0 + y;\n"), ["2.0"])
        self.assertEqual(_found("x /= 0.5 - y;\n"), ["0.5"])

    def test_elsewhere_a_half_or_two_is_flagged(self):
        cases = {
            "x + 0.5": ["0.5"],
            "x - 2.0": ["2.0"],
            "f(0.5)": ["0.5"],
            "std::pow(x, 2.0)": ["2.0"],
            "x * y - 0.5": ["0.5"],
            "x * y + 2.0": ["2.0"],
            "x < 2.0": ["2.0"],
            "scalar_t(0.5) * x": ["0.5"],
        }
        for expression, expected in cases.items():
            with self.subTest(expression=expression):
                self.assertEqual(_expression(expression), expected)

    def test_the_integer_two_is_never_exempt(self):
        cases = {
            "2 * x": ["2"],
            "x / 2": ["2"],
            "v(2)": ["2"],
            "v.segment<2>(3)": ["2", "3"],
            "std::pow(x, 2)": ["2"],
            "x[2]": ["2"],
            "2u * x": ["2u"],
        }
        for expression, expected in cases.items():
            with self.subTest(expression=expression):
                self.assertEqual(_expression(expression), expected)

    def test_other_factors_are_flagged(self):
        self.assertEqual(_expression("0.25 * x"), ["0.25"])
        self.assertEqual(_expression("x / 3.0"), ["3.0"])
        self.assertEqual(_expression("4.0 * x"), ["4.0"])


class NamedConstantTest(unittest.TestCase):
    def test_literals_in_the_definition_of_a_named_constant_are_exempt(self):
        source = (
            "constexpr int kPositionDim = 3;\nstatic constexpr scalar_t"
            " kDefaultGain = 3.0;\ninline constexpr scalar_t"
            " kGravitationalAcceleration = 9.81;  // [m/s^2]\nconst scalar_t"
            " kTolerance = 1.0e-6;\nconstexpr scalar_t kQuarterTurn = 0.25 *"
            " M_PI;\nconstexpr std::array<int, 3> kAxes = {2, 0, 1};\nconstexpr"
            " Rgba kTick{.r = 0.25f, .g = 0.5f, .b = 0.75f, .a ="
            " 1.0f};\nconstexpr scalar_t Foo::kBar = 3.0;\n[[maybe_unused]]"
            " constexpr int kSpare = 7;\nstatic const std::string kName ="
            " std::string(5, 'x');\ninline constexpr char kText[] ="
            ' "7";\nconstexpr int kFirst = 2, kSecond = 3;\nconst vector_t'
            " kDefault = vector_t::Constant(12, 0.3);\nconstexpr absl::Duration"
            " kPoll = absl::Milliseconds(100);\ntemplate <typename"
            " T>\nconstexpr T kPi = T(3.14159);\ntemplate <typename T = double,"
            " int N = 3>\nconstexpr T kTau = T(6.28318);\nconstexpr const"
            " scalar_t* kPointer = &kTable[2];\nconst Eigen::Matrix<scalar_t,"
            " 3, 1> kUp(0.0, 0.0, 9.81);\nstatic constexpr std::array<const"
            ' char* absl_nonnull, 4> kNames = {"a", "b", "c", "d"};\ntemplate'
            " <>\nconstexpr int kSize<Foo> = 3;\n"
        )
        self.assertEqual(_found(source), [])

    def test_a_multi_line_table_is_exempt(self):
        source = (
            "namespace {\n"
            "constexpr std::array<std::uint32_t, 4> kRoundConstants = {\n"
            "    0x428a2f98, 0x71374491,\n"
            "    0xb5c0fbcf, 0xe9b5dba5,\n"
            "};\n"
            "constexpr Band kBands[] = {\n"
            '    {.path = "a", .relative = 0.20, .absolute = 0.005},\n'
            '    {.path = "b", .relative = 0.30, .absolute = 0.010},\n'
            "};\n"
            "}  // namespace\n"
            "class Foo {\n"
            " public:\n"
            "  static constexpr std::array<scalar_t, 3> kWeights{\n"
            "      0.2, 0.3,\n"
            "      0.5};\n"
            "};\n"
        )
        self.assertEqual(_found(source), [])

    def test_declarations_that_are_no_named_constant_are_flagged(self):
        cases = {
            "constexpr int dim = 3;\n": ["3"],
            "static constexpr size_t STATE_DIM = 12;\n": ["12"],
            "int kCount = 3;\n": ["3"],
            "scalar_t kGain = 3.0;\n": ["3.0"],
            "const scalar_t value = kTable[2];\n": ["2"],
            "const scalar_t value = y * kScale(2);\n": ["2"],
            "const int count = kBase + 2;\n": ["2"],
            "static_assert(kSize == 3);\n": ["3"],
            "constexpr int size() { return 3; }\n": ["3"],
            "for (const int kStep : {2, 3}) {}\n": ["2", "3"],
            # A `const` or a kName in a template argument list is the type's.
            "const Eigen::Matrix<scalar_t, ocs2::kPositionDim, ocs2::kPositionDim> gains = Eigen::Matrix3d::Identity() * 3.0;\n": [
                "3.0"
            ],
            "const Eigen::Matrix<scalar_t, Layout::kStateDim, 1> x = vector_t::Zero(12);\n": [
                "12"
            ],
            "struct S {\n  const Eigen::Matrix<scalar_t, ocs2::kForceDim, 1> force = Eigen::Matrix<scalar_t, 3, 1>::Constant(9);\n};\n": [
                "3",
                "9",
            ],
            'static absl::flat_hash_map<const char*, int> kTable = {{"a", 3}};\n': [
                "3"
            ],
            "std::vector<const Foo*> kList(3);\n": ["3"],
            # A statement that a keyword starts declares no constant.
            "void f() {\n  if constexpr (kDim == 3) return kDefault;\n}\n": [
                "3"
            ],
            "int f() {\n  if constexpr (N == 3) return kTable[4];\n}\n": [
                "3",
                "4",
            ],
            "int f() {\n  if (x) return 1;\n  else return kTable[3];\n}\n": [
                "3"
            ],
            "typedef const int kAlias[3];\n": ["3"],
            # Only the declarators named kName are constants, and a macro call in front of one is not part of it.
            "const int kA = 2, b = 5;\n": ["5"],
            "constexpr int value = 3, kOther = 4;\n": ["3"],
            "HUMANOID_SOMETHING(3)\nconstexpr int kAfter = 4;\n": ["3"],
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(_found(source), expected)

    def test_a_label_in_front_of_a_constant_is_not_part_of_it(self):
        source = "switch (x) {\n  case 2: const int kY = 3;\n}\n"
        self.assertEqual(_found(source), ["2"])

    def test_a_lambda_in_a_constant_is_code(self):
        source = "const std::function<int()> kMake = [] { return 3; };\n"
        self.assertEqual(_found(source), ["3"])

    def test_a_constant_a_lambda_initializes_is_defined_by_the_lambda_s_header(
        self,
    ):
        self.assertEqual(
            _found(
                "constexpr std::array<int, 2> kTable = [] { return makeTable();"
                " }();\n"
            ),
            [],
        )
        self.assertEqual(
            _found("const int kN = compute(5, [](int x) { return x + 3; });\n"),
            ["3"],
        )

    def test_a_lambda_in_a_constant_s_braced_initializer_is_code(self):
        self.assertEqual(
            _found(
                "const std::vector<Fn> kFns = {[](double x) { return 3.0 * x;"
                " }};\n"
            ),
            ["3.0"],
        )
        self.assertEqual(
            _found("const Foo kFoo{.make = [](int x) { return x * 9; }};\n"),
            ["9"],
        )
        self.assertEqual(
            _found(
                "void f() {\n  const Foo kFoo{.make = [](int x) { return x * 9;"
                " }};\n}\n"
            ),
            ["9"],
        )
        self.assertEqual(
            _found(
                'const std::array<Entry, 2> kEntries = {{{"a", [](double x) {'
                " return x * 3.7; }}}};\n"
            ),
            ["3.7"],
        )
        self.assertEqual(
            _found("const Foo kFoo{.make = []() constexpr { return 4; }};\n"),
            ["4"],
        )

    def test_a_constant_inside_a_lambda_of_a_braced_initializer_is_exempt(self):
        self.assertEqual(
            _found(
                "const Foo kFoo{.make = [](int x) { constexpr int kScale = 9;"
                " return x * kScale + 3; }};\n"
            ),
            ["3"],
        )

    def test_a_braced_initializer_of_a_constant_stays_exempt(self):
        self.assertEqual(
            _found("const Foo kFoo{{1, 2}, Bar{3}, std::array<int, 1>{4}};\n"),
            [],
        )

    def test_a_function_returning_an_elaborated_enum_is_code(self):
        self.assertEqual(
            _found(
                "enum Color parse(int x) { return static_cast<Color>(x +"
                " 3); }\n"
            ),
            ["3"],
        )
        self.assertEqual(_found("enum E : decltype(kX) { kA = 2 };\n"), [])


class ContextTest(unittest.TestCase):
    def test_enumerator_values_are_exempt(self):
        source = (
            "enum class Request { kDynamics = 2, kCost = 4, kAll = kDynamics |"
            " 8 };\nenum : int { STATE_DIM = 12 };\nclass Foo {\n  enum class"
            " Mode : std::uint8_t {\n    kA = 0x10,\n    kB = 3,\n  };\n};\n"
        )
        self.assertEqual(_found(source), [])

    def test_comments_strings_characters_and_preprocessor_lines_are_not_read(
        self,
    ):
        source = (
            "#define HUMANOID_SIZE 3\n"
            "#define HUMANOID_SUM(a) \\\n"
            "  ((a) + 4)\n"
            "#if EIGEN_VERSION_AT_LEAST(3, 4, 0)\n"
            "#endif\n"
            "// A comment with 3 and 4.5.\n"
            "/* A block comment with 6\n"
            "   and 7. */\n"
            'const char* absl_nonnull text = "8 and 9.5";\n'
            "const char c = '7';\n"
            'const char* absl_nonnull raw = R"(10 and 11)";\n'
            "f(u8\"12\", L'3');\n"
        )
        self.assertEqual(_found(source), [])

    def test_everything_else_is_flagged(self):
        cases = {
            "struct Settings {\n  scalar_t gain = 3.0;\n};\n": ["3.0"],
            "void f(int n = 3);\n": ["3"],
            "using vector3_t = Eigen::Matrix<scalar_t, 3, 1>;\n": ["3"],
            "std::array<scalar_t, 4> coefficients;\n": ["4"],
            "for (int i = 0; i < 6; ++i) {}\n": ["6"],
            "static_assert(sizeof(T) == 8);\n": ["8"],
            "double buffer[16];\n": ["16"],
            "switch (x) {\n  case 3:\n    break;\n}\n": ["3"],
            "return std::get<2>(tuple);\n": ["2"],
            "const vector_t gravity = (vector_t(3) << 0.0, 0.0, -9.81).finished();\n": [
                "3",
                "9.81",
            ],
            "alignas(64) std::atomic<int> counter;\n": ["64"],
            "x = 1 << 3;\n": ["3"],
            "f(/*tolerance=*/1.0e-6);\n": ["1.0e-6"],
            "void set(enum Mode mode) {\n  gain = 3;\n}\n": ["3"],
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(_found(source), expected)

    def test_the_position_is_the_literal_s(self):
        found = check_test_support.findings(
            _NAME, "x = f(a,\n      2.5);\n", _PATH
        )
        self.assertEqual([(f.line, f.column) for f in found], [(2, 7)])
        self.assertIn("`2.5`", found[0].message)

    def test_a_nolint_block_suppresses_a_table(self):
        source = (
            "// NOLINTBEGIN(magic-number): the published Butcher tableau of"
            " Dormand and Prince.\nconst scalar_t a2 = 1.0 / 5;\nconst scalar_t"
            " a3 = 3.0 / 10;\n// NOLINTEND(magic-number)\nconst scalar_t a4 ="
            " 4.0 / 5;\n"
        )
        self.assertEqual(_found(source), ["4.0", "5"])


class ScopeTest(unittest.TestCase):
    def test_production_code_of_the_repository_and_of_ocs2_is_read(self):
        for path in (
            _PATH,
            "humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/common/Types.h",
            "robot_runtime/robot_core/include/robot_core/TripleBuffer.h",
            "tools/config_dump/ProblemDump.cpp",
            "lib/ocs2/core/include/ocs2_core/Types.h",
            "lib/ocs2/core/src/integration/RungeKuttaDormandPrince5.cpp",
            "humanoid_nmpc/humanoid_mpc_validation/src/closed_loop/ClosedLoopMetrics.cpp",
        ):
            with self.subTest(path=path):
                self.assertEqual(_found("x = 3;\n", path), ["3"])

    def test_tests_benchmarks_test_only_code_and_other_files_are_not_read(self):
        for path in (
            "humanoid_nmpc/humanoid_common_mpc/test/testSomeCost.cpp",
            "humanoid_nmpc/humanoid_common_mpc/test/support/LayoutRobotModel.h",
            "humanoid_nmpc/humanoid_centroidal_mpc_test/include/x/Helpers.h",
            "tools/config_dump/testValueDump.cpp",
            "robot_runtime/robot_ipc/src/codec_test.cc",
            "tools/nproto/testdata/Fixture.h",
            "lib/ocs2/core/test/testTypes.cpp",
            "humanoid_state_estimation/humanoid_state_estimator/benchmark/benchmarkKalmanFilter.cpp",
            "humanoid_nmpc/humanoid_mpc_validation/src/benchmark/MpcSolveBenchmark.cpp",
            "humanoid_nmpc/humanoid_mpc_validation/src/closed_loop/LockstepClosedLoop.cpp",
            "robot_runtime/web_viewer/src/JpegDecoder.cpp",
            "humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/acom/AcomSirenWeightsG1.h",
            "tools/clang_tidy/testdata/nullability/Fixture.cpp",
            "lib/ocs2/thirdparty/include/cppad/cppad.hpp",
            "lib/vendored/Foo.cpp",
            "tools/hooks/magic_number.py",
        ):
            with self.subTest(path=path):
                self.assertEqual(_found("x = 3;\n", path), [])

    def test_the_test_only_sources_are_the_testonly_targets_outside_test_directories(
        self,
    ):
        root = check_test_support.source_tree_root()
        test_only = _test_only_cpp_sources(root)
        self.assertIn("robot_runtime/web_viewer/src/JpegDecoder.cpp", test_only)
        # Every source of a testonly C++ target is out of the check's scope.
        self.assertEqual(
            sorted(path for path in test_only if magic_number.in_scope(path)),
            [],
        )
        # Every entry of the list is the source of a testonly target that the test directories do not cover already.
        for entry in magic_number.TEST_ONLY_SOURCES:
            with self.subTest(entry=entry):
                self.assertTrue(
                    os.path.exists(os.path.join(root, entry)), entry
                )
                self.assertFalse(lint_files.is_test_path(entry))
                self.assertTrue(
                    any(
                        path == entry
                        or (entry.endswith("/") and path.startswith(entry))
                        for path in test_only
                    ),
                    f"{entry} is no source of a testonly target",
                )

    def test_behaves_like_every_check(self):
        check_test_support.assert_check_behaves(
            self,
            _NAME,
            "const scalar_t height = 0.25 * legLength;\n",
            _PATH,
            clean=(
                "constexpr scalar_t kHeightRatio = 0.25;\nconst scalar_t height"
                " = kHeightRatio * legLength;\n"
            ),
        )


if __name__ == "__main__":
    unittest.main()
