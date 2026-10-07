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


"""Tests for yaml_usage.py: no YAML file where the MPC's configuration lives, and no yaml-cpp."""

import unittest

from tools.hooks import check_test_support
from tools.hooks import yaml_usage


def _found(name, source, path):
    return [(f.line, f.column) for f in check_test_support.findings(name, source, path)]


class ConfigYamlTest(unittest.TestCase):
    def test_a_yaml_file_in_a_configuration_directory_is_flagged(self):
        for path in (
            "robot_models/my_robot/my_robot_mpc/config/mpc/task.yaml",
            "humanoid_nmpc/humanoid_common_mpc/config/command/gait.yml",
            "humanoid_nmpc/x/TASK.YAML",
        ):
            with self.subTest(path=path):
                self.assertEqual(
                    _found(yaml_usage.CONFIG_YAML, "a: 1\n", path), [(1, 1)]
                )

    def test_other_files_and_directories_are_not(self):
        for path in (
            "robot_models/my_robot/my_robot_mpc/config/mpc/task.textproto",
            "docker-compose.yaml",
            ".github/workflows/build_test.yml",
            "tools/deploy/compose.yaml",
        ):
            with self.subTest(path=path):
                self.assertEqual(_found(yaml_usage.CONFIG_YAML, "a: 1\n", path), [])

    def test_behaves_like_every_check(self):
        check_test_support.assert_check_behaves(
            self,
            yaml_usage.CONFIG_YAML,
            "a: 1\n",
            "robot_models/my_robot/config/mpc/task.yaml",
        )


class YamlCppTest(unittest.TestCase):
    def test_an_include_of_a_yaml_cpp_header_is_flagged(self):
        source = '#include <vector>\n#include "yaml-cpp/yaml.h"\n  #  include <yaml-cpp/node/node.h>\n#include_next "yaml-cpp/x.h"\n'
        self.assertEqual(
            _found(yaml_usage.YAML_CPP, source, "src/a.cpp"),
            [(2, 10), (3, 14), (4, 15)],
        )

    def test_comments_strings_and_other_headers_are_not(self):
        source = (
            '// #include "yaml-cpp/yaml.h"\n'
            "/* #include <yaml-cpp/yaml.h> */\n"
            'const char* kText = "#include <yaml-cpp/yaml.h>";\n'
            '#include "yaml_cpp_like.h"\n'
            '#include "my/yaml-cpp/compat.h"\n'
        )
        self.assertEqual(_found(yaml_usage.YAML_CPP, source, "src/a.cpp"), [])

    def test_a_bazel_label_of_the_repository_is_flagged(self):
        source = 'cc_library(\n    deps = [\n        "@yaml_cpp",\n        "@@yaml_cpp//:yaml_cpp",\n    ],\n)\n'
        self.assertEqual(
            _found(yaml_usage.YAML_CPP, source, "pkg/BUILD.bazel"), [(3, 9), (4, 9)]
        )

    def test_a_bazel_comment_or_another_repository_is_not(self):
        source = '# "@yaml_cpp" was removed.\ndeps = ["@yaml_cpp_tools", "//yaml_cpp:x"]  # "@yaml_cpp"\n'
        self.assertEqual(_found(yaml_usage.YAML_CPP, source, "pkg/BUILD.bazel"), [])

    def test_other_languages_are_not_read(self):
        self.assertEqual(
            _found(
                yaml_usage.YAML_CPP, '#include "yaml-cpp/yaml.h"\n', "docs/README.md"
            ),
            [],
        )

    def test_behaves_like_every_check(self):
        check_test_support.assert_check_behaves(
            self,
            yaml_usage.YAML_CPP,
            '#include "yaml-cpp/yaml.h"\n',
            "src/a.cpp",
            clean="#include <vector>\n",
        )


class RepositoryTest(unittest.TestCase):
    def test_the_check_lists_the_configuration_directories(self):
        self.assertEqual(yaml_usage.CONFIG_DIRS, ("robot_models/", "humanoid_nmpc/"))


if __name__ == "__main__":
    unittest.main()
