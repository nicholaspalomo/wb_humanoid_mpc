"""Tests for boost_usage.py, the lint check that keeps Boost out of the project's C++ and Bazel files."""

import io
import os
import re
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import boost_usage  # noqa: E402

from tools.hooks import check_test_support  # noqa: E402

_HOOKS_DIR = os.path.dirname(os.path.abspath(__file__))


def _repository_file(relative_path):
    """A repository file, from the Bazel runfiles when run by Bazel and from the source tree otherwise."""
    roots = []
    if "TEST_SRCDIR" in os.environ:
        roots.append(os.path.join(os.environ["TEST_SRCDIR"], "_main"))
    roots.append(os.path.join(_HOOKS_DIR, "..", ".."))
    for root in roots:
        candidate = os.path.join(root, relative_path)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(relative_path)


def cpp(source):
    """The (kind, text) of every finding in C++ `source`."""
    return [(v.kind, v.text) for v in boost_usage.check_source(source, boost_usage.CPP)]


def starlark(source):
    """The (kind, text) of every finding in a BUILD or .bzl `source`."""
    return [
        (v.kind, v.text) for v in boost_usage.check_source(source, boost_usage.STARLARK)
    ]


def bazelrc(source):
    """The (kind, text) of every finding in a .bazelrc `source`."""
    return [
        (v.kind, v.text) for v in boost_usage.check_source(source, boost_usage.BAZELRC)
    ]


class CppIncludesTest(unittest.TestCase):
    def test_a_boost_header_is_flagged_in_either_form(self):
        for source, expected in [
            ("#include <boost/variant.hpp>\n", "#include <boost/variant.hpp>"),
            ('#include "boost/optional.hpp"\n', '#include "boost/optional.hpp"'),
            (
                "  #  include <boost/fusion/invoke.hpp>\n",
                "#include <boost/fusion/invoke.hpp>",
            ),
            ("#include_next <boost/any.hpp>\n", "#include_next <boost/any.hpp>"),
        ]:
            with self.subTest(source=source):
                self.assertEqual(cpp(source), [("include", expected)])

    def test_other_headers_are_fine(self):
        source = (
            "#include <pinocchio/multibody/model.hpp>\n"
            '#include "ocs2_core/misc/PropertyTree.h"\n'
            "#include <myboost/x.h>\n"
            '#include "third_party/boost/x.h"\n'
        )
        self.assertEqual(cpp(source), [])

    def test_a_boost_header_named_in_another_directive_is_flagged(self):
        # A macro an #include expands, and __has_include, whose header an #include would otherwise read unchecked.
        for source, expected in [
            ("#define INC <boost/variant.hpp>\n#include INC\n", "<boost/variant.hpp>"),
            ('#define INC "boost/variant.hpp"\n#include INC\n', '"boost/variant.hpp"'),
            (
                "#if __has_include(<boost/optional.hpp>)\n#endif\n",
                "<boost/optional.hpp>",
            ),
            ("#elif __has_include_next(<boost/any.hpp>)\n", "<boost/any.hpp>"),
        ]:
            with self.subTest(source=source):
                self.assertEqual(cpp(source), [("header", expected)])

    def test_a_continued_directive_is_read_to_its_end(self):
        violations = boost_usage.check_source(
            "#if defined(X) && \\\n    __has_include(<boost/optional.hpp>)\n#endif\nint y = a < b;\n",
            boost_usage.CPP,
        )
        self.assertEqual(
            [(v.line, v.col, v.kind) for v in violations], [(2, 19, "header")]
        )

    def test_other_directives_and_comments_naming_a_header_are_fine(self):
        source = (
            "#define INC <ocs2_core/misc/PropertyTree.h>\n"
            "#if __has_include(<optional>)\n#endif\n"
            "// #define INC <boost/variant.hpp>\n"
            "#define NOTE 1  /* was <boost/variant.hpp> */\n"
            '#define DOC "see the boost/ replacement"\n'
        )
        self.assertEqual(cpp(source), [])

    def test_the_location_is_the_header_name(self):
        violations = boost_usage.check_source(
            "#include <vector>\n#include <boost/variant.hpp>\n", boost_usage.CPP, "a.h"
        )
        self.assertEqual(
            [(v.path, v.line, v.col) for v in violations], [("a.h", 2, 10)]
        )


class CppCodeTest(unittest.TestCase):
    def test_the_boost_namespace_is_flagged(self):
        for source, expected in [
            ("boost::optional<int> value;", "boost::"),
            (
                "const auto& joint = ::boost::get<JointModelFreeFlyer>(model.joints[1]);",
                "boost::",
            ),
            ("namespace boost {\n}", "namespace boost"),
            ("using namespace boost;", "namespace boost"),
            ("namespace bp = boost::python;", "boost::"),
            ("boost  ::  variant<int> v;", "boost::"),
        ]:
            with self.subTest(source=source):
                self.assertEqual(cpp(source), [("namespace", expected)])

    def test_an_alias_of_the_namespace_is_flagged(self):
        # Every use through the alias would otherwise pass: Pinocchio's headers bring Boost in without an #include.
        for source, expected in [
            ("namespace bst = boost;\nbst::optional<int> x;", "namespace bst = boost"),
            ("namespace bst = ::boost;", "namespace bst = ::boost"),
            ("namespace  bst=boost ;", "namespace bst=boost"),
        ]:
            with self.subTest(source=source):
                self.assertEqual(cpp(source), [("namespace", expected)])

    def test_other_namespace_aliases_are_fine(self):
        self.assertEqual(
            cpp(
                "namespace fs = std::filesystem;\nnamespace boosted = mylib;\nnamespace b = boost_like;\n"
            ),
            [],
        )

    def test_boost_macros_are_flagged(self):
        for source, expected in [
            ("BOOST_FOREACH(int x, values) {}", "BOOST_FOREACH"),
            ("#define BOOST_VARIANT_LIMIT_TYPES 50\n", "BOOST_VARIANT_LIMIT_TYPES"),
            ("#ifdef BOOST_VERSION\n#endif\n", "BOOST_VERSION"),
        ]:
            with self.subTest(source=source):
                self.assertEqual(cpp(source), [("macro", expected)])

    def test_identifiers_that_only_contain_the_word_are_fine(self):
        # The MuJoCo simulator boosts joint damping in ragdoll mode; that is not Boost.
        source = (
            "double boost = 2.0;\n"
            "dampingBoost_ = boostFactor * boost_gain;\n"
            "constexpr int kRAGDOLL_BOOST_FACTOR = 3;\n"
            "struct Boost {};\nBoost::Boost() {}\n"
        )
        self.assertEqual(cpp(source), [])

    def test_every_use_is_reported_where_it_is(self):
        violations = boost_usage.check_source(
            "int a;\n  x = boost::get<int>(v);\nBOOST_FOREACH(a, b);\n",
            boost_usage.CPP,
        )
        self.assertEqual(
            [(v.line, v.col, v.kind) for v in violations],
            [(2, 7, "namespace"), (3, 1, "macro")],
        )


class CppCommentsAndLiteralsTest(unittest.TestCase):
    def test_comments_are_prose(self):
        source = (
            "// Replaces boost::property_tree::ptree; see #include <boost/property_tree/ptree.hpp>.\n"
            "/* Distributed under the Boost Software License, Version 1.0. BOOST_FOREACH */\n"
            "/**\n"
            " * Splits off the first key, as boost::property_tree::string_path::reduce does.\n"
            "#include <boost/variant.hpp>\n"
            " */\n"
            "int x;  // namespace boost\n"
        )
        self.assertEqual(cpp(source), [])

    def test_a_commented_out_include_is_not_an_include(self):
        self.assertEqual(cpp("// #include <boost/variant.hpp>\n"), [])
        self.assertEqual(cpp("/* #include <boost/variant.hpp> */\n"), [])

    def test_a_backslash_continues_a_line_comment(self):
        self.assertEqual(cpp("// a note \\\nboost::get<int>(v);\n"), [])
        # Positive control: without the backslash the next line is code.
        self.assertEqual(
            cpp("// a note\nboost::get<int>(v);\n"), [("namespace", "boost::")]
        )

    def test_string_and_character_literals_are_not_code(self):
        source = (
            'const char* note = "boost::optional was replaced";\n'
            "LOG(INFO) << \"BOOST_FOREACH\" << 'b';\n"
            'const char* text = u8"namespace boost";\n'
            'const char* escaped = "a \\" boost:: \\" b";\n'
            '#define NAME "boost::x"\n'
        )
        self.assertEqual(cpp(source), [])

    def test_raw_strings_are_not_code(self):
        source = (
            'const char* generated = R"(\n#include <boost/variant.hpp>\nboost::get<int>(v);\n)";\n'
            'const char* delimited = R"cpp( )" boost:: )cpp";\n'
            'const char* wide = LR"(BOOST_FOREACH)";\n'
        )
        self.assertEqual(cpp(source), [])
        # Positive control: code after the raw string is code again.
        self.assertEqual(
            cpp('const char* s = R"x( )" )x"; boost::get<int>(v);'),
            [("namespace", "boost::")],
        )

    def test_a_digit_separator_does_not_open_a_character_literal(self):
        # Read as a character literal, `'000; y = boost::get<int>(v); z = '` would hide the use between them.
        self.assertEqual(
            cpp("x = 1'000; y = boost::get<int>(v); z = 'a';"),
            [("namespace", "boost::")],
        )

    def test_an_unterminated_quote_ends_with_its_line(self):
        self.assertEqual(
            cpp("#error don't build this\nboost::get<int>(v);\n"),
            [("namespace", "boost::")],
        )


class StarlarkTest(unittest.TestCase):
    def test_boost_labels_are_flagged(self):
        for source, expected in [
            ('deps = ["@boost"]', "@boost"),
            ('deps = ["@boost//:boost"]', "@boost//:boost"),
            ('deps = ["@@boost//:boost"]', "@@boost//:boost"),
            ('deps = ["@boost.optional"]', "@boost.optional"),
            ("deps = ['@boost//:headers']", "@boost//:headers"),
        ]:
            with self.subTest(source=source):
                self.assertEqual(starlark(source), [("label", expected)])

    def test_other_labels_and_paths_are_fine(self):
        source = (
            'deps = ["@pinocchio", "@boosted", "@boost_like//:x"]\n'
            'hdrs = glob(["include/boost/**"], allow_empty = True)\n'
            'repo_ctx.symlink("/usr/include/boost", "include/boost")\n'
            'boost_repository(name = "boost")\n'
        )
        self.assertEqual(starlark(source), [])

    def test_boost_defines_are_flagged(self):
        for source, expected in [
            (
                'copts = ["-DBOOST_MPL_LIMIT_LIST_SIZE=50"]',
                "-DBOOST_MPL_LIMIT_LIST_SIZE=50",
            ),
            ('copts = ["-D BOOST_X"]', "-D BOOST_X"),
            ('cxxopts = ["--copt=-DBOOST_X"]', "-DBOOST_X"),
            (
                'local_defines = ["BOOST_VARIANT_LIMIT_TYPES=50"]',
                "BOOST_VARIANT_LIMIT_TYPES=50",
            ),
            (
                "defines = ['BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS']",
                "BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS",
            ),
        ]:
            with self.subTest(source=source):
                self.assertEqual(starlark(source), [("define", expected)])

    def test_boost_libraries_linked_are_flagged(self):
        for source, expected in [
            (
                'linkopts = ["-lboost_system", "-lboost_filesystem"]',
                [("link", "-lboost_system"), ("link", "-lboost_filesystem")],
            ),
            (
                'linkopts = ["-l:libboost_system.so.1.83.0"]',
                [("link", "-l:libboost_system.so.1.83.0")],
            ),
            (
                'srcs = ["/usr/lib/x86_64-linux-gnu/libboost_serialization.so"]',
                [("link", "libboost_serialization.so")],
            ),
        ]:
            with self.subTest(source=source):
                self.assertEqual(starlark(source), expected)

    def test_other_libraries_are_fine(self):
        source = (
            'linkopts = ["-lpinocchio_parsers", "-lmyboost_x", "-lboosted", "-L/usr/lib/boost"]\n'
            'srcs = ["lib/libmyboost_x.so", "include/boost/**"]\n'
        )
        self.assertEqual(starlark(source), [])

    def test_other_defines_are_fine(self):
        source = (
            'defines = ["PINOCCHIO_URDFDOM_USE_STD_SHARED_PTR", "PINOCCHIO_WITH_BOOST"]\n'
            'copts = ["-DMY_BOOST_X", "-DPINOCCHIO_URDFDOM_TYPEDEF_SHARED_PTR"]\n'
        )
        self.assertEqual(starlark(source), [])

    def test_comments_are_prose_but_strings_are_code(self):
        self.assertEqual(
            starlark(
                "# Only @pinocchio depends on @boost; it carries -DBOOST_X.\nx = 1  # @boost\n"
            ),
            [],
        )
        # A `#` inside a string does not start a comment.
        self.assertEqual(starlark('x = "#" + "@boost"'), [("label", "@boost")])

    def test_a_build_file_written_from_a_string_is_checked_as_one(self):
        # The comment lines of the embedded BUILD file are its comments; its labels are its labels.
        source = (
            'repo_ctx.file("BUILD.bazel", content = """\n'
            "cc_library(\n"
            '    name = "x",\n'
            "    # Only Pinocchio may depend on @boost.\n"
            '    deps = ["@boost"],\n'
            ")\n"
            '""")\n'
        )
        violations = boost_usage.check_source(source, boost_usage.STARLARK)
        self.assertEqual([(v.line, v.kind) for v in violations], [(5, "label")])


class BazelrcTest(unittest.TestCase):
    def test_boost_defines_and_labels_are_flagged(self):
        for line, expected in [
            ("build --copt=-DBOOST_X", ("define", "-DBOOST_X")),
            (
                "build --cxxopt=-DBOOST_VARIANT_LIMIT_TYPES=50",
                ("define", "-DBOOST_VARIANT_LIMIT_TYPES=50"),
            ),
            (
                "build --per_file_copt=//humanoid_nmpc/.*@-DBOOST_X",
                ("define", "-DBOOST_X"),
            ),
            ("build --host_copt=-D BOOST_X", ("define", "-D BOOST_X")),
            ("build --@boost//:flag=1", ("label", "@boost//:flag")),
            ("build --linkopt=-lboost_system", ("link", "-lboost_system")),
        ]:
            with self.subTest(line=line):
                self.assertEqual(bazelrc(line + "\n"), [expected])

    def test_comments_and_other_flags_are_fine(self):
        source = (
            "# Pinocchio's Boost defines are not set here (no --copt=-DBOOST_X): @pinocchio carries them.\n"
            "build --copt=-DPINOCCHIO_X\n"
            "build --cxxopt=-std=c++20\n"
        )
        self.assertEqual(bazelrc(source), [])


class NolintTest(unittest.TestCase):
    def test_a_marker_with_a_reason_exempts_its_line(self):
        self.assertEqual(
            cpp(
                "x = boost::get<J>(joint);  // NOLINT(boost): Pinocchio's joint model is a boost::variant.\n"
            ),
            [],
        )
        self.assertEqual(
            starlark(
                'deps = ["@boost"],  # NOLINT(boost): Pinocchio\'s headers include it.\n'
            ),
            [],
        )
        # Only its own line.
        self.assertEqual(
            cpp("a = boost::x;  // NOLINT(boost): reason\nb = boost::y;\n"),
            [("namespace", "boost::")],
        )

    def test_a_marker_in_a_string_literal_exempts_nothing(self):
        # The text of a literal is not a comment, so it is not a justification either.
        for source in [
            'const char* s = "NOLINT(boost): reason"; boost::get<int>(v);\n',
            'const char* s = R"(// NOLINT(boost): reason)"; boost::get<int>(v);\n',
        ]:
            with self.subTest(source=source):
                self.assertEqual(cpp(source), [("namespace", "boost::")])
        self.assertEqual(
            starlark('deps = ["@boost", "NOLINT(boost): reason"]\n'),
            [("label", "@boost")],
        )
        # A block marker in a literal neither opens nor closes a block.
        self.assertEqual(
            cpp('const char* s = "NOLINTEND(boost)";\n'),
            [],
        )

    def test_a_marker_without_a_reason_exempts_nothing(self):
        for marker in ["// NOLINT(boost)", "/* NOLINT(boost) */", "// NOLINT(boost):"]:
            with self.subTest(marker=marker):
                self.assertEqual(
                    cpp(f"x = boost::get<J>(joint);  {marker}\n"),
                    [("namespace", "boost::")],
                )

    def test_another_checks_marker_exempts_nothing(self):
        for marker in [
            "// NOLINT(argument-comment)",
            "// NOLINT",
            "// NOLINT(boost-ish): reason",
        ]:
            with self.subTest(marker=marker):
                self.assertEqual(
                    cpp(f"x = boost::get<J>(joint);  {marker}\n"),
                    [("namespace", "boost::")],
                )

    def test_nextline_exempts_the_line_after_it_only(self):
        source = (
            "// NOLINTNEXTLINE(boost): the URDF parser takes a boost::optional.\n"
            "a = boost::x;\n"
            "b = boost::y;\n"
        )
        violations = boost_usage.check_source(source, boost_usage.CPP)
        self.assertEqual([v.line for v in violations], [3])

    def test_a_block_exempts_the_lines_from_begin_to_end(self):
        source = (
            "defines = [\n"
            "    # NOLINTBEGIN(boost): Pinocchio's Boost configuration.\n"
            '    "BOOST_MPL_LIMIT_LIST_SIZE=50",\n'
            '    "BOOST_VARIANT_LIMIT_TYPES=50",\n'
            "    # NOLINTEND(boost)\n"
            '    "BOOST_X",\n'
            "]\n"
        )
        violations = boost_usage.check_source(source, boost_usage.STARLARK)
        self.assertEqual([(v.line, v.text) for v in violations], [(6, "BOOST_X")])

    def test_a_block_without_a_reason_exempts_nothing(self):
        source = '# NOLINTBEGIN(boost)\n"BOOST_X",\n# NOLINTEND(boost)\n'
        self.assertEqual(starlark(source), [("define", "BOOST_X")])

    def test_unmatched_block_markers_are_reported_and_exempt_nothing(self):
        unclosed = boost_usage.check_source(
            '# NOLINTBEGIN(boost): reason\n"BOOST_X",\n', boost_usage.STARLARK, "BUILD"
        )
        self.assertEqual(
            [(v.line, v.kind) for v in unclosed], [(1, "unclosed"), (2, "define")]
        )
        self.assertIn("never closed by a NOLINTEND(boost)", str(unclosed[0]))
        unopened = boost_usage.check_source(
            "int x;\n// NOLINTEND(boost)\n", boost_usage.CPP
        )
        self.assertEqual([(v.line, v.kind) for v in unopened], [(2, "unopened")])


class MessageTest(unittest.TestCase):
    def test_a_cpp_finding_says_what_to_use_and_how_to_justify_an_exception(self):
        message = str(
            boost_usage.check_source("boost::any x;", boost_usage.CPP, "a.cpp")[0]
        )
        self.assertTrue(
            message.startswith("a.cpp:1:1: `boost::` uses the boost namespace.")
        )
        self.assertIn("Abseil", message)
        self.assertIn("// NOLINT(boost): <reason>", message)

    def test_a_bazel_finding_points_at_the_pinocchio_target(self):
        message = str(
            boost_usage.check_source(
                'deps = ["@boost"]', boost_usage.STARLARK, "BUILD"
            )[0]
        )
        self.assertIn("@pinocchio", message)
        self.assertIn("bazel/system_libs.bzl", message)
        self.assertIn("# NOLINT(boost): <reason>", message)


class ScopeTest(unittest.TestCase):
    def test_kinds_of_file(self):
        for path, kind in [
            ("humanoid_nmpc/x/src/A.cpp", boost_usage.CPP),
            ("lib/ocs2/core/include/ocs2_core/integration/steppers.h", boost_usage.CPP),
            ("lib/ocs2/core/include/ocs2_core/x/implementation/A.hpp", boost_usage.CPP),
            ("lib/ocs2/thirdparty/include/cppad/core/testvector.hpp", None),
            ("BUILD.bazel", boost_usage.STARLARK),
            ("lib/ocs2/BUILD", boost_usage.STARLARK),
            ("bazel/system_libs.bzl", boost_usage.STARLARK),
            ("MODULE.bazel", boost_usage.STARLARK),
            ("WORKSPACE", boost_usage.STARLARK),
            ("third_party/foo.BUILD", boost_usage.STARLARK),
            (".bazelrc", boost_usage.BAZELRC),
            ("tools/ci.bazelrc", boost_usage.BAZELRC),
            ("MODULE.bazel.lock", None),
            ("tools/hooks/boost_usage.py", None),
            ("README.md", None),
        ]:
            with self.subTest(path=path):
                self.assertEqual(boost_usage.kind_of(path), kind)

    def test_check_files_skips_the_vendored_cppad_and_other_files(self):
        with tempfile.TemporaryDirectory() as root:
            for path in [
                "lib/ocs2/thirdparty/include/cppad/core/testvector.hpp",
                "lib/ocs2/core/src/a.cpp",
                "notes.md",
            ]:
                os.makedirs(os.path.join(root, os.path.dirname(path)), exist_ok=True)
                with open(os.path.join(root, path), "w") as f:
                    f.write("# include <boost/numeric/ublas/vector.hpp>\n")
            found = boost_usage.check_files(
                [
                    os.path.join(
                        root, "lib/ocs2/thirdparty/include/cppad/core/testvector.hpp"
                    ),
                    os.path.join(root, "lib/ocs2/core/src/a.cpp"),
                    os.path.join(root, "notes.md"),
                ],
                root,
            )
            self.assertEqual(
                [(v.path, v.kind) for v in found],
                [("lib/ocs2/core/src/a.cpp", "include")],
            )


class RepositoryFilesTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = self.directory.name
        for path in [
            "src/a.cpp",
            "src/untracked.h",
            "src/ignored.cpp",
            "src/notes.md",
            "BUILD.bazel",
            "bazel-out/k8-opt/bin/generated.cpp",
            "build/generated.h",
        ]:
            os.makedirs(os.path.join(self.root, os.path.dirname(path)), exist_ok=True)
            with open(os.path.join(self.root, path), "w") as f:
                f.write("int x;\n")

    def tearDown(self):
        self.directory.cleanup()

    def relative(self, paths):
        return sorted(os.path.relpath(p, self.root) for p in paths)

    def test_the_files_git_tracks_or_would_track(self):
        def git(*args):
            subprocess.run(
                ["git", "-C", self.root, *args], check=True, capture_output=True
            )

        git("init", "-q")
        with open(os.path.join(self.root, ".gitignore"), "w") as f:
            f.write("src/ignored.cpp\nbazel-out/\nbuild/\n")
        git("add", "src/a.cpp", "BUILD.bazel")
        self.assertEqual(
            self.relative(boost_usage.repository_files(self.root)),
            ["BUILD.bazel", "src/a.cpp", "src/untracked.h"],
        )

    def test_without_git_the_tree_is_walked_without_build_output(self):
        with mock.patch.object(
            boost_usage.subprocess, "run", side_effect=OSError("no git")
        ):
            files = boost_usage.repository_files(self.root)
        self.assertEqual(
            self.relative(files),
            ["BUILD.bazel", "src/a.cpp", "src/ignored.cpp", "src/untracked.h"],
        )


class ShippedConfigurationTest(unittest.TestCase):
    """The Bazel configuration as shipped: Boost is reached through the @pinocchio target alone."""

    def test_the_pinocchio_target_is_the_only_exemption(self):
        with open(_repository_file("bazel/system_libs.bzl")) as f:
            source = f.read()
        self.assertEqual(
            boost_usage.check_source(
                source, boost_usage.STARLARK, "bazel/system_libs.bzl"
            ),
            [],
        )
        # Without its markers, everything the file exempts must lie in the Pinocchio repository rule, and be its
        # dependency on @boost and its Boost defines.
        unmarked = re.sub(r"NOLINT(NEXTLINE|BEGIN|END)?\(boost\)", "", source)
        exempted = boost_usage.check_source(unmarked, boost_usage.STARLARK)
        lines = source.split("\n")
        begin = next(
            i
            for i, line in enumerate(lines, 1)
            if line.startswith("def _pinocchio_repository(")
        )
        end = next(
            i
            for i, line in enumerate(lines, 1)
            if line.startswith("pinocchio_repository = ")
        )
        self.assertTrue(
            exempted,
            "the @pinocchio target no longer depends on Boost: drop its exemptions",
        )
        self.assertEqual({v.kind for v in exempted}, {"label", "define"})
        self.assertEqual({v.text for v in exempted if v.kind == "label"}, {"@boost"})
        self.assertEqual([v.line for v in exempted if not begin < v.line < end], [])


class StagedModeTest(unittest.TestCase):
    """The pre-commit hook's mode: the index of a real git repository, not its working tree."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = self.directory.name
        self.run_git("init", "-q")
        self.run_git("config", "user.email", "test@example.com")
        self.run_git("config", "user.name", "test")

    def tearDown(self):
        self.directory.cleanup()

    def run_git(self, *args):
        subprocess.run(
            ["git", "-C", self.root, *args], check=True, capture_output=True, text=True
        )

    def write(self, path, text):
        full = os.path.join(self.root, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as f:
            f.write(text)

    def staged(self):
        return [(v.path, v.kind) for v in boost_usage.check_staged(self.root)]

    def test_staged_cpp_and_bazel_violations_are_reported(self):
        self.write("src/a.cpp", "#include <boost/variant.hpp>\n")
        self.write("src/BUILD.bazel", 'cc_library(name = "a", deps = ["@boost"])\n')
        self.write(".bazelrc", "build --copt=-DBOOST_X\n")
        self.run_git("add", ".")
        self.assertEqual(
            sorted(self.staged()),
            [
                (".bazelrc", "define"),
                ("src/BUILD.bazel", "label"),
                ("src/a.cpp", "include"),
            ],
        )

    def test_the_index_is_checked_not_the_working_tree(self):
        self.write("a.cpp", "std::optional<int> x;\n")
        self.run_git("add", "a.cpp")
        self.write("a.cpp", "boost::optional<int> x;\n")
        self.assertEqual(self.staged(), [])
        # And the reverse: a violation staged, fixed only in the working tree, still blocks.
        self.run_git("add", "a.cpp")
        self.write("a.cpp", "std::optional<int> x;\n")
        self.assertEqual(self.staged(), [("a.cpp", "namespace")])

    def test_only_files_the_commit_touches(self):
        self.write("old.cpp", "boost::optional<int> x;\n")
        self.run_git("add", "old.cpp")
        self.run_git("commit", "-q", "-m", "legacy")
        self.write("new.cpp", "std::optional<int> x;\n")
        self.run_git("add", "new.cpp")
        self.assertEqual(
            self.staged(), [], "a committed file the commit does not touch blocked it"
        )

    def test_vendored_cppad_other_languages_and_deletions_are_ignored(self):
        self.write(
            "lib/ocs2/thirdparty/include/cppad/x.hpp",
            "# include <boost/numeric/ublas/vector.hpp>\n",
        )
        self.write("tool.py", "import boost\n")
        self.write("gone.h", "boost::optional<int> x;\n")
        self.run_git("add", ".")
        self.run_git("commit", "-q", "-m", "base")
        self.run_git("rm", "-q", "gone.h")
        self.write(
            "lib/ocs2/thirdparty/include/cppad/x.hpp",
            "# include <boost/numeric/ublas/matrix.hpp>\n",
        )
        self.run_git("add", ".")
        self.assertEqual(self.staged(), [])


class WiringTest(unittest.TestCase):
    def test_the_linter_runs_it_over_the_repository(self):
        check = check_test_support.assert_registered(self, "boost")
        # lib/ocs2 is Boost-free and checked; only the vendored CppAD copies are not.
        self.assertTrue(check.applies_to("lib/ocs2/core/src/a.cpp"))
        self.assertTrue(check.applies_to("lib/ocs2/BUILD.bazel"))
        self.assertFalse(check.applies_to("lib/ocs2/thirdparty/include/cppad/x.hpp"))
        check_test_support.assert_check_behaves(
            self, "boost", "#include <boost/variant.hpp>\n", "src/a.cpp"
        )
        check_test_support.assert_check_behaves(
            self,
            "boost",
            'cc_library(name = "a", deps = ["@boost"])\n',
            "src/BUILD.bazel",
        )

    def test_the_hook_runs_the_staged_check_after_formatting_and_fails_on_it(self):
        check_test_support.assert_hook_runs_the_linter(self)


class CommandLineTest(unittest.TestCase):
    def test_exit_status_and_report(self):
        with tempfile.TemporaryDirectory() as directory:
            bad = os.path.join(directory, "bad.cpp")
            good = os.path.join(directory, "good.cpp")
            with open(bad, "w") as f:
                f.write("int x;\n#include <boost/variant.hpp>\n")
            with open(good, "w") as f:
                f.write("// Replaces boost::variant.\n#include <variant>\n")
            output = io.StringIO()
            with redirect_stdout(output), mock.patch.object(
                os, "getcwd", return_value=directory
            ):
                self.assertEqual(boost_usage.main([good]), 0)
                self.assertEqual(boost_usage.main([good, bad]), 1)
            self.assertIn(
                "bad.cpp:2:10: `#include <boost/variant.hpp>` includes a Boost header.",
                output.getvalue(),
            )

    def test_paths_and_staged_mode_are_exclusive(self):
        with redirect_stdout(io.StringIO()), mock.patch("sys.stderr", io.StringIO()):
            with self.assertRaises(SystemExit):
                boost_usage.main([])
            with self.assertRaises(SystemExit):
                boost_usage.main(["--git-staged", "a.cpp"])


if __name__ == "__main__":
    unittest.main()
