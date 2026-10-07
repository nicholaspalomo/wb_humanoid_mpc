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

"""Tests for lint_files.py: the one list of vendored, generated and fixture files, the scopes, and the Python roots."""

import os
import tempfile
import unittest
from unittest import mock

from tools.hooks import check_test_support
from tools.hooks import lint_files


class ScopeTest(unittest.TestCase):
    def test_first_party(self):
        for path, expected in [
            ("humanoid_nmpc/x/src/A.cpp", True),
            ("tools/hooks/lint_code.py", True),
            ("lib/ocs2/core/src/A.cpp", False),
            ("lib/mujoco_vendor/BUILD.bazel", False),
            ("tools/ifttt-lint/src/main.rs", False),
            ("tools/clang_tidy/testdata/Bad.cpp", False),
            (sorted(lint_files.GENERATED_FILES)[0], False),
        ]:
            with self.subTest(path=path):
                self.assertEqual(
                    lint_files.in_scope(path, lint_files.Scope.FIRST_PARTY), expected
                )

    def test_first_party_and_ocs2(self):
        scope = lint_files.Scope.FIRST_PARTY_AND_OCS2
        self.assertTrue(lint_files.in_scope("lib/ocs2/core/src/A.cpp", scope))
        self.assertTrue(lint_files.in_scope("humanoid_nmpc/x/A.h", scope))
        self.assertFalse(
            lint_files.in_scope("lib/ocs2/thirdparty/include/cppad/a.hpp", scope)
        )
        self.assertFalse(lint_files.in_scope("lib/mujoco_vendor/x.h", scope))
        self.assertFalse(
            lint_files.in_scope("tools/clang_tidy/testdata/Bad.cpp", scope)
        )

    def test_text_reads_fixtures_and_generated_files_but_not_vendored_code(self):
        scope = lint_files.Scope.TEXT
        self.assertTrue(lint_files.in_scope("tools/clang_tidy/testdata/Bad.cpp", scope))
        self.assertTrue(
            lint_files.in_scope(sorted(lint_files.GENERATED_FILES)[0], scope)
        )
        self.assertFalse(lint_files.in_scope("lib/ocs2/core/src/A.cpp", scope))

    def test_not_thirdparty(self):
        scope = lint_files.Scope.NOT_THIRDPARTY
        self.assertTrue(lint_files.in_scope("lib/ocs2/core/src/A.cpp", scope))
        self.assertTrue(lint_files.in_scope("lib/mujoco_vendor/BUILD.bazel", scope))
        self.assertFalse(
            lint_files.in_scope("lib/ocs2/thirdparty/include/cppad/a.hpp", scope)
        )

    def test_non_test(self):
        scope = lint_files.Scope.FIRST_PARTY_NON_TEST
        self.assertTrue(
            lint_files.in_scope("robot_runtime/robot_core/src/A.cpp", scope)
        )
        self.assertFalse(
            lint_files.in_scope(
                "robot_runtime/robot_core/test/TripleBufferTest.cpp", scope
            )
        )

    def test_test_paths(self):
        for path in [
            "humanoid_nmpc/humanoid_common_mpc/test/testContactPlan.cpp",
            "humanoid_nmpc/humanoid_centroidal_mpc_test/src/Support.cpp",
            "humanoid_nmpc/humanoid_centroidal_mpc_test/include/x/CentroidalTestingModelInterface.h",
            "robot_runtime/robot_core/src/TripleBufferTest.cpp",
            "tools/hooks/test_lint_code.py",
            "humanoid_learning/tests/x.py",
            "a/conftest.py",
            "a/foo_test.py",
            "tools/clang_tidy/testdata/Bad.cpp",
        ]:
            with self.subTest(path=path):
                self.assertTrue(lint_files.is_test_path(path))
        for path in [
            "tools/ipc/bus_test_publisher.py",
            "robot_runtime/robot_model/src/ContactEstimatorRegistry.cpp",
            "humanoid_nmpc/x/src/Latest.cpp",
            "tools/hooks/lint_code.py",
        ]:
            with self.subTest(path=path):
                self.assertFalse(lint_files.is_test_path(path))

    def test_paths_are_normalized(self):
        self.assertTrue(lint_files.is_vendored("./lib/ocs2/x.h"))
        self.assertEqual(lint_files.normalize("./a/b"), "a/b")


class RepositoryFilesTest(unittest.TestCase):
    def setUp(self):
        self.repository = check_test_support.StagedRepository()
        self.root = self.repository.root
        for path in [
            "src/a.cpp",
            "src/untracked.h",
            "src/ignored.cpp",
            "bazel-out/k8-opt/bin/generated.cpp",
            ".bazel/bin/generated.h",
        ]:
            self.repository.write(path, "int x;\n")
        self.repository.write(".gitignore", "src/ignored.cpp\nbazel-out/\n.bazel/\n")

    def tearDown(self):
        self.repository.cleanup()

    def test_the_files_git_tracks_or_would_track(self):
        self.repository.git("add", "src/a.cpp")
        self.assertEqual(
            lint_files.repository_files(self.root),
            [".gitignore", "src/a.cpp", "src/untracked.h"],
        )

    def test_deleted_files_are_left_out(self):
        self.repository.git("add", "src/a.cpp")
        os.remove(os.path.join(self.root, "src/a.cpp"))
        self.assertNotIn("src/a.cpp", lint_files.repository_files(self.root))

    def test_without_git_the_tree_is_walked_without_build_output(self):
        with mock.patch.object(
            lint_files.subprocess, "run", side_effect=OSError("no git")
        ):
            files = lint_files.repository_files(self.root)
        self.assertNotIn(".bazel/bin/generated.h", files)
        self.assertNotIn("bazel-out/k8-opt/bin/generated.cpp", files)
        self.assertIn("src/ignored.cpp", files)

    def test_staged_files_and_their_source(self):
        self.repository.git("add", "src/a.cpp")
        self.repository.write("src/a.cpp", "int y;\n")
        self.assertEqual(lint_files.staged_files(self.root), ["src/a.cpp"])
        self.assertEqual(lint_files.staged_source(self.root, "src/a.cpp"), "int x;\n")


class BuildFilesTest(unittest.TestCase):
    def setUp(self):
        # pylint: disable-next=consider-using-with  # tearDown() deletes it.
        self.directory = tempfile.TemporaryDirectory()
        self.root = self.directory.name
        files = {
            "pkg/BUILD.bazel": 'py_library(name = "a", imports = ["."])\npy_test(name = "t", imports = ["test"])\n',
            "pkg/mod_a.py": "",
            "pkg/subpackage/x.py": "",
            "pkg/test/test_a.py": "",
            "other/BUILD": 'py_library(name = "b", imports = ["python"])\n',
            "other/python/other_pkg/__init__.py": "",
            "lib/v/BUILD.bazel": 'py_library(name = "c", imports = ["."])\n',
            "robot/src/A.cpp": "",
            "lib/ocs2/B.cpp": "",
        }
        for path, text in files.items():
            full = os.path.join(self.root, path)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w", encoding="utf-8") as f:
                f.write(text)

    def tearDown(self):
        self.directory.cleanup()

    def test_python_import_roots(self):
        with mock.patch.object(
            lint_files.subprocess, "run", side_effect=OSError("no git")
        ):
            self.assertEqual(
                lint_files.python_import_roots(self.root),
                ["other/python", "pkg", "pkg/test"],
            )

    def test_python_first_party_packages(self):
        with mock.patch.object(
            lint_files.subprocess, "run", side_effect=OSError("no git")
        ):
            packages = lint_files.python_first_party_packages(self.root)
        for name in [
            "humanoid_learning",
            "tools",
            "mod_a",
            "subpackage",
            "test_a",
            "other_pkg",
        ]:
            self.assertIn(name, packages)

    def test_cpp_top_level_dirs(self):
        with mock.patch.object(
            lint_files.subprocess, "run", side_effect=OSError("no git")
        ):
            self.assertEqual(lint_files.cpp_top_level_dirs(self.root), ["robot"])

    def test_bazel_package(self):
        self.assertEqual(
            lint_files.bazel_package("pkg/test/test_a.py", self.root), "//pkg"
        )
        self.assertEqual(lint_files.bazel_package("robot/src/A.cpp", self.root), "//")


if __name__ == "__main__":
    unittest.main()
