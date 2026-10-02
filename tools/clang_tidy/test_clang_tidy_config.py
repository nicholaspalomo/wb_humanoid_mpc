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

"""The clang-tidy configurations: //:.clang-tidy (the enforced checks) and tools/clang_tidy/sweep.clang-tidy (all).

Each check is listed once with the guide section or tip it enforces, the checks that contradict the repository's rules
stay out, the enforced checks are a subset of the swept ones, and the aspect's header filter covers exactly the
first-party C++ (tools/clang_tidy/clang_tidy.bzl). clang-tidy itself checks the names and options (`--verify-config`,
in //tools/clang_tidy:selftest).
"""

import os
import re
import unittest

import yaml

from tools.hooks import check_test_support
from tools.hooks import lint_files

ENFORCED = ".clang-tidy"
SWEEP = "tools/clang_tidy/sweep.clang-tidy"
BZL = "tools/clang_tidy/clang_tidy.bzl"

# Checks that contradict a rule of this repository (the comment block of the configurations says which).
NEVER_ENABLED = (
    "bugprone-easily-swappable-parameters",
    "cert-err58-cpp",
    "cppcoreguidelines-avoid-const-or-ref-data-members",
    "cppcoreguidelines-macro-usage",
    "google-readability-todo",
    "hicpp-multiway-paths-covered",
    "hicpp-use-auto",
    "misc-const-correctness",
    "misc-non-private-member-variables-in-classes",
    "modernize-avoid-c-arrays",
    "modernize-loop-convert",
    "modernize-pass-by-value",
    "modernize-use-auto",
    "modernize-use-trailing-return-type",
    "readability-function-size",
    "readability-identifier-length",
    "readability-implicit-bool-conversion",
    "readability-magic-numbers",
    "readability-qualified-auto",
)

# Paths a .cpp action's header filter must and must not show diagnostics of.
# LINT.IfChange(header_filter)
FIRST_PARTY_HEADERS = (
    "humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/Types.h",
    "./tools/clang_tidy/testdata/MemberInit.h",
    "robot_runtime/robot_core/include/robot_core/ResourcePaths.h",
)
OTHER_HEADERS = (
    "lib/ocs2/core/include/ocs2_core/Types.h",
    "external/abseil-cpp+/absl/status/status.h",
    "bazel-out/k8-opt/bin/humanoid_nmpc/humanoid_mpc_msgs/humanoid_mpc_msgs/mpc_policy.pb.h",
    "bazel-out/k8-opt/bin/humanoid_nmpc/x/_virtual_includes/x/Foo.h",
    "/usr/include/eigen3/Eigen/Core",
    "xhumanoid_nmpc/Foo.h",
)
# LINT.ThenChange(//tools/clang_tidy/clang_tidy.bzl:header_filter)


def _load(path: str) -> dict:
    document = yaml.safe_load(check_test_support.read_repository_file(path))
    assert isinstance(document, dict), path
    return document


def _check_lines(path: str) -> list[str]:
    """The lines of the configuration's `Checks:` list, comments included."""
    lines = check_test_support.read_repository_file(path).splitlines()
    start = lines.index("Checks:") + 1
    end = start
    while end < len(lines) and lines[end].startswith("  "):
        end += 1
    return lines[start:end]


def _bzl_list(name: str) -> list[str]:
    source = check_test_support.read_repository_file(BZL)
    match = re.search(rf"^{name} = \[(.*?)\]$", source, re.MULTILINE | re.DOTALL)
    assert match is not None, name
    return re.findall(r'"([^"]*)"', match.group(1))


def _header_filter() -> re.Pattern[str]:
    source = check_test_support.read_repository_file(BZL)
    match = re.search(
        r'^FIRST_PARTY_HEADER_FILTER = "((?:[^"\\]|\\.)*)"$', source, re.MULTILINE
    )
    assert match is not None
    return re.compile(match.group(1).encode().decode("unicode_escape"))


class ConfigurationTest(unittest.TestCase):
    def _each(self):
        for path in (ENFORCED, SWEEP):
            with self.subTest(config=path):
                yield path

    def test_checks_start_from_nothing_and_are_sorted_within_each_group(self):
        for path in self._each():
            checks = _load(path)["Checks"]
            self.assertEqual(checks[0], "-*")
            groups: list[list[str]] = [[]]
            for line in _check_lines(path)[1:]:
                if line.startswith("  # "):
                    if groups[-1]:
                        groups.append([])
                elif line.startswith("  - "):
                    groups[-1].append(line.split()[1])
            for group in groups:
                self.assertEqual(group, sorted(group))
            self.assertEqual(len(checks), len(set(checks)))

    def test_every_check_names_what_it_enforces(self):
        for path in self._each():
            for line in _check_lines(path)[1:]:
                if line.startswith("  - "):
                    self.assertRegex(line, r"^  - [a-z0-9.-]+ +# \S", line)

    def test_findings_are_errors_and_the_aspect_sets_the_header_filter(self):
        for path in self._each():
            config = _load(path)
            self.assertEqual(config["WarningsAsErrors"], "*")
            self.assertIs(config["SystemHeaders"], False)
            # tools/clang_tidy/clang_tidy.bzl passes --header-filter per action; a value here would be dead.
            self.assertNotIn("HeaderFilterRegex", config)
            self.assertNotIn("ExcludeHeaderFilterRegex", config)

    def test_the_checks_against_the_repositorys_rules_stay_out(self):
        for path in self._each():
            checks = _load(path)["Checks"]
            # A glob would enable checks nobody reviewed, the never-enabled ones among them.
            self.assertEqual([check for check in checks if "*" in check], ["-*"])
            for check in NEVER_ENABLED:
                self.assertNotIn(check, checks)
            self.assertFalse([check for check in checks if check.startswith("llvm-")])

    def test_the_enforced_checks_are_swept_with_the_same_options(self):
        enforced = _load(ENFORCED)
        sweep = _load(SWEEP)
        self.assertLessEqual(set(enforced["Checks"]), set(sweep["Checks"]))
        for key in ("CheckOptions", "WarningsAsErrors", "SystemHeaders", "FormatStyle"):
            self.assertEqual(enforced[key], sweep[key], key)

    def test_each_clang_diagnostic_check_has_its_warning_flag(self):
        # `clang-diagnostic-<name>` reports clang's -W<name>, and the aspect drops every warning flag of the build.
        flags = set(_bzl_list("CLANG_DIAGNOSTIC_FLAGS"))
        for path in self._each():
            for check in _load(path)["Checks"]:
                if check.startswith("clang-diagnostic-"):
                    self.assertIn("-W" + check[len("clang-diagnostic-") :], flags)


class AspectFileListsTest(unittest.TestCase):
    def test_the_aspect_skips_what_the_other_linters_skip(self):
        self.assertEqual(
            tuple(_bzl_list("VENDORED_PACKAGES")), lint_files.VENDORED_DIRS
        )
        self.assertEqual(tuple(_bzl_list("FIXTURE_PACKAGES")), lint_files.FIXTURE_DIRS)
        self.assertEqual(
            frozenset(_bzl_list("GENERATED_FILES")), lint_files.GENERATED_FILES
        )

    def test_the_header_filter_shows_first_party_headers_only(self):
        header_filter = _header_filter()
        self.assertTrue(header_filter.pattern.startswith("^"))
        for path in FIRST_PARTY_HEADERS:
            self.assertRegex(path, header_filter)
        for path in OTHER_HEADERS:
            self.assertNotRegex(path, header_filter)

    def test_the_header_filter_covers_every_directory_with_first_party_cpp(self):
        root = check_test_support.source_tree_root()
        directories = lint_files.cpp_top_level_dirs(root)
        self.assertIn("humanoid_nmpc", directories)
        for directory in directories:
            self.assertRegex(os.path.join(directory, "Foo.h"), _header_filter())


if __name__ == "__main__":
    unittest.main()
