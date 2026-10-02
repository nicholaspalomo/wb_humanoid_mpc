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

"""What the clang-tidy aspect runs, from the arguments it writes for each file, without running clang-tidy.

//tools/clang_tidy:fixture_arguments and its siblings apply the aspect (tools/clang_tidy/clang_tidy.bzl) and write
each action's clang-tidy arguments to a file. This test pins the command line: the build's GCC flags that clang must
not see are gone, the language standard and libstdc++ are the build's, every first-party include directory is -I (so
that header diagnostics survive), the two header filters, and the files that are never linted.
"""

import json
import os
import re
import unittest

from tools.clang_tidy import clang_tidy_report

_GCC_ONLY_FLAGS = (
    "-fno-canonical-system-headers",
    "-fopenmp",
    "-fstack-usage",
    "-pass-exit-codes",
)


def _runfile(path: str) -> str:
    return os.path.join(os.environ.get("TEST_SRCDIR", ""), "_main", path)


def _read(path: str) -> str:
    with open(_runfile(path), encoding="utf-8") as f:
        return f.read()


def _bzl_constant(name: str) -> str:
    match = re.search(
        rf'^{name} = "((?:[^"\\]|\\.)*)"$',
        _read("tools/clang_tidy/clang_tidy.bzl"),
        re.MULTILINE,
    )
    assert match is not None, name
    return match.group(1).encode().decode("unicode_escape")


def _bzl_list(name: str) -> list[str]:
    match = re.search(
        rf"^{name} = \[(.*?)\]$",
        _read("tools/clang_tidy/clang_tidy.bzl"),
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, name
    return re.findall(r'"([^"]*)"', match.group(1))


def _actions(manifest: str) -> dict[str, list[str]]:
    """The clang-tidy arguments of each linted file of a clang_tidy_args target, by the file's path."""
    actions = {}
    for entry in json.loads(_read(f"tools/clang_tidy/{manifest}.json")):
        actions[entry["source"]] = _read(entry["arguments"]).splitlines()
    return actions


def _compiler_flags(arguments: list[str]) -> list[str]:
    return arguments[arguments.index("--") + 1 :]


def _pairs(flags: list[str], option: str) -> list[str]:
    return [flags[i + 1] for i, flag in enumerate(flags[:-1]) if flag == option]


class FixtureArgumentsTest(unittest.TestCase):
    def setUp(self):
        self.actions = _actions("fixture_arguments")
        self.assertTrue(self.actions)

    def test_the_gcc_flags_clang_must_not_see_are_dropped(self):
        allowed_warnings = set(_bzl_list("CLANG_DIAGNOSTIC_FLAGS"))
        for source, arguments in self.actions.items():
            with self.subTest(source=source):
                for flag in _compiler_flags(arguments):
                    self.assertNotIn(flag, _GCC_ONLY_FLAGS)
                    self.assertFalse(flag.startswith("-pedantic"), flag)
                    if flag.startswith("-W"):
                        self.assertIn(flag, allowed_warnings)

    def test_every_clang_diagnostic_check_has_its_warning_flag(self):
        for arguments in self.actions.values():
            flags = _compiler_flags(arguments)
            for warning in _bzl_list("CLANG_DIAGNOSTIC_FLAGS"):
                self.assertIn(warning, flags)

    def test_the_language_standard_and_libstdcxx_are_the_builds(self):
        for source, arguments in self.actions.items():
            with self.subTest(source=source):
                flags = _compiler_flags(arguments)
                standards = [flag for flag in flags if flag.startswith("-std=")]
                # The toolchain's -std=c++17 comes first and .bazelrc's --cxxopt=-std=c++20 wins.
                self.assertEqual(standards[-1], "-std=c++20")
                self.assertTrue(
                    any(
                        re.fullmatch(r"--gcc-install-dir=/.*/gcc/[^/]+/\d+", flag)
                        for flag in flags
                    )
                )

    def test_external_includes_are_kept_and_first_party_ones_are_not_system(self):
        flags = _compiler_flags(
            self.actions["tools/clang_tidy/testdata/DiscardedStatus.cpp"]
        )
        self.assertIn(
            "external/abseil-cpp+", _pairs(flags, "-iquote") + _pairs(flags, "-isystem")
        )
        for arguments in self.actions.values():
            for directory in _pairs(_compiler_flags(arguments), "-isystem"):
                self.assertTrue(
                    re.match(r"^(bazel-out/[^/]+/bin/)?(external|lib)/", directory),
                    directory,
                )

    def test_a_first_party_includes_directory_is_minus_i(self):
        # Bazel passes `includes` as -isystem, which would make clang-tidy drop the header's diagnostics.
        flags = _compiler_flags(
            self.actions["tools/clang_tidy/testdata/rename/First.cpp"]
        )
        self.assertIn("tools/clang_tidy/testdata/rename/include", _pairs(flags, "-I"))
        self.assertNotIn(
            "tools/clang_tidy/testdata/rename/include", _pairs(flags, "-isystem")
        )

    def test_translation_units_show_first_party_headers_and_headers_only_themselves(
        self,
    ):
        header_filter = "--header-filter=" + _bzl_constant("FIRST_PARTY_HEADER_FILTER")
        for source, arguments in self.actions.items():
            with self.subTest(source=source):
                self.assertEqual(arguments[arguments.index("--") - 1], source)
                if source.endswith(".h"):
                    self.assertIn("--header-filter=^$", arguments)
                    self.assertEqual(
                        _pairs(_compiler_flags(arguments), "-x"), ["c++-header"]
                    )
                    self.assertFalse(
                        any(a.startswith("--exclude-header-filter=") for a in arguments)
                    )
                else:
                    self.assertIn(header_filter, arguments)
                    self.assertNotIn("c++-header", arguments)
                    excluded = [
                        a for a in arguments if a.startswith("--exclude-header-filter=")
                    ]
                    self.assertEqual(len(excluded), 1)
                    for generated in _bzl_list("GENERATED_FILES"):
                        self.assertRegex(
                            generated, excluded[0][len("--exclude-header-filter=") :]
                        )

    def test_the_enforced_configuration_is_read_without_searching_the_sandbox(self):
        for arguments in self.actions.values():
            self.assertIn("--config-file=.clang-tidy", arguments)

    def test_generated_sources_are_not_linted(self):
        linted = set(self.actions)
        self.assertIn("tools/clang_tidy/testdata/Clean.cpp", linted)
        self.assertFalse(any(path.endswith("Generated.cpp") for path in linted), linted)


class ScopeTest(unittest.TestCase):
    def test_the_fixtures_are_linted_only_by_the_self_test(self):
        # `make lint-tidy` over //... uses the default scope, in which the deliberately bad fixtures do not exist.
        self.assertEqual(_actions("fixture_arguments_in_repository_scope"), {})

    def test_generated_headers_in_the_tree_are_not_linted(self):
        linted = set(_actions("generated_header_arguments"))
        self.assertTrue(
            any(
                path.startswith("humanoid_nmpc/humanoid_common_mpc/include/")
                for path in linted
            )
        )
        for generated in _bzl_list("GENERATED_FILES"):
            self.assertNotIn(generated, linted)


class ReportPathsTest(unittest.TestCase):
    def test_sandbox_paths_become_repository_paths(self):
        # A header action prints its own file with the sandbox's absolute path.
        self.assertEqual(
            clang_tidy_report.repository_path(
                "/home/u/.cache/bazel/_bazel_u/1/sandbox/processwrapper-sandbox/7/execroot/_main/tools/x/Foo.h"
            ),
            "tools/x/Foo.h",
        )
        self.assertEqual(
            clang_tidy_report.repository_path("./tools/x/Foo.h"), "tools/x/Foo.h"
        )


if __name__ == "__main__":
    unittest.main()
