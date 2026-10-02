"""Tests for argument_comments.py, the lint check for literal arguments without a Google-style argument comment."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argument_comments  # noqa: E402

from tools.hooks import check_test_support  # noqa: E402


def flagged(source):
    """The (callee, literal, position) of every violation in `source`."""
    return [
        (v.callee, v.literal, v.position)
        for v in argument_comments.check_source(source)
    ]


class FlagsBareLiteralsTest(unittest.TestCase):
    def test_the_call_the_rule_was_written_for(self):
        source = "const ContactRectangle footprint = ContactRectangle::loadContactRectangle(taskFile_, modelSettings_, 0, verbose_);"
        self.assertEqual(flagged(source), [("loadContactRectangle", "0", 3)])

    def test_an_argument_comment_satisfies_it(self):
        self.assertEqual(
            flagged(
                "loadContactRectangle(taskFile_, modelSettings_, /*contactIndex=*/0, verbose_);"
            ),
            [],
        )
        self.assertEqual(flagged("f(a, /* verbose = */ true);"), [])

    def test_a_comment_that_does_not_name_the_parameter_does_not(self):
        self.assertEqual(flagged("f(a, /* index */ 0);"), [("f", "0", 2)])

    def test_every_kind_of_literal(self):
        cases = {
            "true": "true",
            "false": "false",
            "nullptr": "nullptr",
            "-1": "-1",
            "+2": "+2",
            "0.5f": "0.5f",
            "1e-3": "1e-3",
            ".25": ".25",
            "0x1F": "0x1F",
            "1'000'000": "1'000'000",
            "42u": "42u",
        }
        for literal, expected in cases.items():
            with self.subTest(literal=literal):
                self.assertEqual(flagged("f(a, %s);" % literal), [("f", expected, 2)])

    def test_each_literal_of_a_call_is_reported_with_its_position(self):
        self.assertEqual(
            flagged("f(x, 0, y, false);"), [("f", "0", 2), ("f", "false", 4)]
        )

    def test_the_location_is_the_literal(self):
        violations = argument_comments.check_source(
            "int x;\nfoo(a,\n    b,\n      7);\n"
        )
        self.assertEqual([(v.line, v.col) for v in violations], [(4, 7)])

    def test_a_call_inside_an_exempt_macro_is_still_checked(self):
        self.assertEqual(
            flagged("EXPECT_TRUE(isInContact(0.1, foot));"), [("isInContact", "0.1", 1)]
        )

    def test_templated_and_qualified_callees(self):
        self.assertEqual(flagged("ns::foo<int>(a, 0);"), [("foo", "0", 2)])
        self.assertEqual(flagged("foo<Bar<int>>(a, 0);"), [("foo", "0", 2)])
        self.assertEqual(flagged("object->method(a, 0);"), [("method", "0", 2)])
        self.assertEqual(
            flagged("std::make_unique<Term>(model, 0.5);"), [("make_unique", "0.5", 2)]
        )

    def test_a_direct_initialized_object_is_a_constructor_call(self):
        self.assertEqual(
            flagged("ContactRectangle rectangle(file, 0);"), [("rectangle", "0", 2)]
        )


class ExemptionsTest(unittest.TestCase):
    def test_all_numeric_arguments_are_indices_or_coordinates(self):
        self.assertEqual(flagged("x = R(2, 2);"), [])
        self.assertEqual(flagged("v = vector3_t(0.0, 0.0, 1.0);"), [])
        # A boolean among them is not a coordinate.
        self.assertEqual(flagged("f(0, true);"), [("f", "0", 1), ("f", "true", 2)])

    def test_one_argument_calls(self):
        self.assertEqual(flagged("setVerbose(true); v.resize(3); wait(0.5);"), [])

    def test_conventional_callees(self):
        for source in [
            "std::max(a, 0.0);",
            "std::clamp(v, -1.0, 1.0);",
            "std::pow(x, 2);",
            "m.block<3, 3>(0, j);",
            "v.segment(i, 3);",
            'absl::StrCat("x", 1, y);',
            "EXPECT_NEAR(a, 0.0, 1e-9);",
            "ASSERT_EQ(count, 0);",
            "LOG_EVERY_N(WARNING, 100);",
            "a.isApprox(b, 1e-12);",
            "vector_t::Constant(n, 1.0);",
            "Eigen::Quaterniond(w, 0.0, 0.0, 1.0);",
            "vector3_t(x, 0.0, 1.0);",
            # glibc leaves memset's fill byte unnamed, so no argument comment could name it.
            "memset(&state, 0, sizeof(State));",
        ]:
            with self.subTest(source=source):
                self.assertEqual(flagged(source), [])

    def test_conventional_declared_types(self):
        self.assertEqual(flagged("std::vector<bool> flags(n, false);"), [])
        self.assertEqual(flagged("std::string padding(n, 0);"), [])

    def test_later_declarators_of_a_conventional_declaration(self):
        # Every declarator of `Type a(...), b(...)` has the declaration's type, not only the first.
        self.assertEqual(
            flagged(
                "std::vector<double*> A(N + 1, nullptr), B(N + 1, nullptr), C(N, nullptr);"
            ),
            [],
        )
        # Positive controls: a declaration of a type that is not conventional, and calls that are not declarators.
        self.assertEqual(flagged("int a(1), b(n, true);"), [("b", "true", 2)])
        self.assertEqual(flagged("f(g(1), h(n, true));"), [("h", "true", 2)])
        self.assertEqual(flagged("f(x, vector(1), h(n, true));"), [("h", "true", 2)])
        self.assertEqual(flagged("return a(1), b(n, true);"), [("b", "true", 2)])

    def test_template_arguments_of_a_conventional_type_are_not_call_arguments(self):
        # The commas and sizes of `Eigen::Matrix<SCALAR, 3, 3>` are not arguments of the call or declaration around it,
        # neither in a parameter declaration nor in a temporary passed as an argument.
        for source in [
            "SCALAR footYawError(const Eigen::Matrix<SCALAR, 3, 3>& rotation, const vector3_t& axis);",
            "EndEffectorKinematicsCostElement(const Eigen::Matrix<SCALAR_T, 13, 1>& vector);",
            "costElementVector(Eigen::Matrix<SCALAR_T, 13, 1>::Zero());",
            "f(getRotationMatrixFromZyxEulerAngles(Eigen::Matrix<ad_scalar_t, 3, 1>(yaw, 0.0, 0.0)), frame);",
            "g(std::pair<int, bool>{index, flag}, name);",
        ]:
            with self.subTest(source=source):
                self.assertEqual(flagged(source), [])
        # Positive controls: a bare literal beside such a type is still reported, at its real position.
        self.assertEqual(
            flagged("f(Eigen::Matrix<ad_scalar_t, 3, 1>(x, y, z), 0);"), [("f", "0", 2)]
        )
        self.assertEqual(
            flagged("h(std::vector<std::pair<int, int>>{}, false);"),
            [("h", "false", 2)],
        )

    def test_a_less_than_after_a_type_like_name_does_not_swallow_the_file(self):
        # `array < 3` is a comparison: the closing parenthesis still ends the call, and the next call is checked.
        self.assertEqual(flagged("f(array < 3, x); g(a, 0);"), [("g", "0", 2)])

    def test_strings_characters_and_user_defined_literals(self):
        self.assertEqual(flagged("f(a, \"text\"); g(a, 'c'); sleep(clock, 100ms);"), [])

    def test_not_calls(self):
        for source in [
            "if (a == 0) {}",
            "while (flag && x > 0.5) {}",
            "for (int i = 0; i < 3; ++i) {}",
            "int y = static_cast<int>(0);",
            "return (a, 0);",
            "void f(int a = 0, bool b = true);",
            "bool operator()(int a, int b = 0) const;",
            'static_assert(sizeof(T) == 8, "size");',
        ]:
            with self.subTest(source=source):
                self.assertEqual(flagged(source), [])

    def test_comments_strings_and_the_preprocessor_are_not_code(self):
        source = "\n".join(
            [
                "// f(a, 0)",
                "/* f(a, 0) */",
                'const char* s = "f(a, 0)";',
                'const char* r = R"(f(a, 0))";',
                "#define CALL f(a, 0)",
                "#define LONG f(a, \\",
                "  0)",
            ]
        )
        self.assertEqual(flagged(source), [])

    def test_nolint(self):
        self.assertEqual(flagged("f(a, 0);  // NOLINT(argument-comment)"), [])
        self.assertEqual(flagged("// NOLINTNEXTLINE(argument-comment)\nf(a, 0);"), [])
        # Only the named check is suppressed.
        self.assertEqual(
            flagged("f(a, 0);  // NOLINT(some-other-check)"), [("f", "0", 2)]
        )


class StagedModeTest(unittest.TestCase):
    """The pre-commit hook's mode: the index of a real git repository, not its working tree."""

    def setUp(self):
        import subprocess
        import tempfile

        self.directory = tempfile.TemporaryDirectory()
        self.root = self.directory.name
        self.run_git = lambda *args: subprocess.run(
            ["git", "-C", self.root, *args], check=True, capture_output=True, text=True
        )
        self.run_git("init", "-q")
        self.run_git("config", "user.email", "test@example.com")
        self.run_git("config", "user.name", "test")

    def tearDown(self):
        self.directory.cleanup()

    def write(self, path, text):
        full = os.path.join(self.root, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as f:
            f.write(text)

    def staged(self):
        return [(v.path, v.literal) for v in argument_comments.check_staged(self.root)]

    def test_a_staged_violation_is_reported(self):
        self.write("src/a.cpp", "void g() { f(a, 0); }\n")
        self.run_git("add", "src/a.cpp")
        self.assertEqual(self.staged(), [("src/a.cpp", "0")])

    def test_the_index_is_checked_not_the_working_tree(self):
        # Staged fixed, then broken again in the working tree only: the commit is fine.
        self.write("a.cpp", "void g() { f(a, /*index=*/0); }\n")
        self.run_git("add", "a.cpp")
        self.write("a.cpp", "void g() { f(a, 0); }\n")
        self.assertEqual(self.staged(), [])
        # And the reverse: a violation staged, fixed only in the working tree, still blocks.
        self.run_git("add", "a.cpp")
        self.write("a.cpp", "void g() { f(a, /*index=*/0); }\n")
        self.assertEqual(self.staged(), [("a.cpp", "0")])

    def test_only_files_the_commit_touches(self):
        self.write("old.cpp", "void g() { f(a, 0); }\n")
        self.run_git("add", "old.cpp")
        self.run_git("commit", "-q", "-m", "legacy")
        self.write("new.cpp", "void g() { f(a, /*index=*/0); }\n")
        self.run_git("add", "new.cpp")
        self.assertEqual(
            self.staged(), [], "a committed file the commit does not touch blocked it"
        )

    def test_vendored_code_other_languages_and_deletions_are_ignored(self):
        self.write("lib/ocs2/x.cpp", "void g() { f(a, 0); }\n")
        self.write("tool.py", "f(a, 0)\n")
        self.write("gone.h", "void g() { f(a, 0); }\n")
        self.run_git("add", ".")
        self.run_git("commit", "-q", "-m", "base")
        self.run_git("rm", "-q", "gone.h")
        self.write("lib/ocs2/x.cpp", "void g() { f(b, 0); }\n")
        self.run_git("add", ".")
        self.assertEqual(self.staged(), [])


class PreCommitHookTest(unittest.TestCase):
    def test_the_hook_runs_the_staged_check_after_formatting_and_fails_on_it(self):
        # The hook runs the registry's checks on the staged files, after the formatter's changes are staged.
        check_test_support.assert_hook_runs_the_linter(self)
        check_test_support.assert_check_behaves(
            self,
            "argument-comment",
            "void g() {\n  f(a, 0);\n}\n",
            "src/a.cpp",
            clean="void g() {\n  f(a, /*index=*/0);\n}\n",
        )


class CommandLineTest(unittest.TestCase):
    def test_exit_status_and_report(self):
        import io
        import tempfile
        from contextlib import redirect_stdout

        with tempfile.TemporaryDirectory() as directory:
            bad = os.path.join(directory, "bad.cpp")
            good = os.path.join(directory, "good.cpp")
            with open(bad, "w") as f:
                f.write("void g() { f(a, 0); }\n")
            with open(good, "w") as f:
                f.write("void g() { f(a, /*index=*/0); }\n")
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(argument_comments.main([good]), 0)
                self.assertEqual(argument_comments.main([bad]), 1)
            self.assertIn(
                "bad.cpp:1:17: bare literal `0` as argument 2 of `f(...)`",
                output.getvalue(),
            )


if __name__ == "__main__":
    unittest.main()
