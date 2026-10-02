"""****************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
****************************************************************************"""

"""Which configuration files the GUI edits: found in the source checkout, never in Bazel's runfiles.

Validates remote_control/config_files.py:
  - resolve_source_path re-roots a path through robot_models/ (a runfiles link, a copied tree) into the checkout;
  - find_repo_root and resolve_input_path find the checkout and the files named on the command line, under `bazel run`
    (BUILD_WORKSPACE_DIRECTORY, BUILD_WORKING_DIRECTORY) and without it;
  - resolve_config_files finds the files the GUI was not given next to the ones it was;
  - the defaults read from reference.yaml and task.yaml.
Every case builds a fake checkout in a temporary directory, so nothing here depends on the caller's environment.
"""

import os
import shutil
import tempfile
import unittest
from unittest import mock

from operator_test_support import REPO_ROOT, repo_path
from remote_control import config_files

_ATLAS = os.path.join("robot_models", "drc_atlas", "drc_atlas_centroidal_mpc", "config")


def _write(path, text=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write(text)
    return path


class FakeCheckoutTestCase(unittest.TestCase):
    """A checkout with the DRC Atlas config layout, and a copy of its robot_models tree elsewhere."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        self.repo_root = os.path.join(self.tmpdir, "checkout")
        _write(os.path.join(self.repo_root, "MODULE.bazel"))
        self.config = os.path.join(self.repo_root, _ATLAS)
        self.task = _write(
            os.path.join(self.config, "mpc", "task.yaml"), "enableOnlineTuning: false\n"
        )
        self.reference = _write(
            os.path.join(self.config, "command", "reference.yaml"),
            "defaultBaseHeight: 0.91\n",
        )
        self.pd_gains = _write(
            os.path.join(self.config, "controller", "joint_pd_gains.yaml"), "kp: 1\n"
        )
        # A copy such as a runfiles tree holds.
        self.copy_root = os.path.join(self.tmpdir, "runfiles", "_main")
        shutil.copytree(
            os.path.join(self.repo_root, "robot_models"),
            os.path.join(self.copy_root, "robot_models"),
        )
        self.copied_task = os.path.join(self.copy_root, _ATLAS, "mpc", "task.yaml")


class TestResolveSourcePath(FakeCheckoutTestCase):
    def test_a_copy_through_robot_models_resolves_to_the_checkout(self):
        self.assertEqual(
            config_files.resolve_source_path(self.copied_task, self.repo_root),
            self.task,
        )

    def test_a_path_in_the_checkout_stays(self):
        self.assertEqual(
            config_files.resolve_source_path(self.task, self.repo_root), self.task
        )

    def test_a_copy_without_a_source_file_stays(self):
        os.remove(self.task)
        self.assertEqual(
            config_files.resolve_source_path(self.copied_task, self.repo_root),
            self.copied_task,
        )

    def test_a_path_outside_robot_models_empty_or_missing_stays(self):
        elsewhere = _write(os.path.join(self.tmpdir, "some", "random", "file.yaml"))
        self.assertEqual(
            config_files.resolve_source_path(elsewhere, self.repo_root), elsewhere
        )
        self.assertEqual(config_files.resolve_source_path("", self.repo_root), "")
        missing = "/does/not/exist/robot_models/task.yaml"
        self.assertEqual(
            config_files.resolve_source_path(missing, self.repo_root), missing
        )

    def test_without_a_checkout_every_path_stays(self):
        self.assertEqual(
            config_files.resolve_source_path(self.copied_task, None), self.copied_task
        )


class TestFindRepoRoot(FakeCheckoutTestCase):
    def test_bazel_run_names_the_checkout(self):
        with mock.patch.dict(os.environ, {"BUILD_WORKSPACE_DIRECTORY": self.repo_root}):
            self.assertEqual(config_files.find_repo_root(), self.repo_root)

    def test_a_workspace_directory_that_is_no_checkout_is_ignored(self):
        with mock.patch.dict(os.environ, {"BUILD_WORKSPACE_DIRECTORY": self.tmpdir}):
            self.assertNotEqual(config_files.find_repo_root(), self.tmpdir)

    def test_without_bazel_run_it_is_the_checkout_of_the_sources(self):
        # The real path of this package's sources lies in the checkout, whatever the working directory.
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BUILD_WORKSPACE_DIRECTORY", None)
            root = config_files.find_repo_root(start=self.tmpdir)
        self.assertIsNotNone(root)
        self.assertTrue(os.path.isfile(os.path.join(root, "MODULE.bazel")))
        self.assertTrue(os.path.isdir(os.path.join(root, "robot_models")))
        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    root,
                    "humanoid_nmpc",
                    "remote_control",
                    "remote_control",
                    "config_files.py",
                )
            )
        )


class TestResolveInputPath(FakeCheckoutTestCase):
    def test_an_absolute_path_stays(self):
        self.assertEqual(
            config_files.resolve_input_path("/an/absolute/path.yaml", self.repo_root),
            "/an/absolute/path.yaml",
        )

    def test_relative_to_where_bazel_run_was_started(self):
        start = os.path.join(self.repo_root, "robot_models")
        with mock.patch.dict(os.environ, {"BUILD_WORKING_DIRECTORY": start}):
            resolved = config_files.resolve_input_path(
                os.path.join(
                    "drc_atlas",
                    "drc_atlas_centroidal_mpc",
                    "config",
                    "mpc",
                    "task.yaml",
                ),
                self.repo_root,
            )
        self.assertEqual(resolved, self.task)

    def test_then_relative_to_the_checkout(self):
        with mock.patch.dict(os.environ, {"BUILD_WORKING_DIRECTORY": self.tmpdir}):
            resolved = config_files.resolve_input_path(
                os.path.join(_ATLAS, "mpc", "task.yaml"), self.repo_root
            )
        self.assertEqual(resolved, self.task)

    def test_a_file_that_exists_nowhere_names_the_first_candidate(self):
        with mock.patch.dict(os.environ, {"BUILD_WORKING_DIRECTORY": self.tmpdir}):
            resolved = config_files.resolve_input_path("missing.yaml", self.repo_root)
        self.assertEqual(resolved, os.path.join(self.tmpdir, "missing.yaml"))

    def test_the_default_network_file_is_found(self):
        # The runfiles hold config/ipc/network.textproto for the binaries; here it is found from the checkout.
        resolved = config_files.resolve_input_path("config/ipc/network.textproto")
        self.assertTrue(
            resolved.endswith(os.path.join("config", "ipc", "network.textproto")),
            resolved,
        )


class TestResolveConfigFiles(FakeCheckoutTestCase):
    def test_given_files_are_used_from_the_checkout(self):
        files = config_files.resolve_config_files(
            task_file=self.copied_task,
            reference_file=os.path.join(
                self.copy_root, _ATLAS, "command", "reference.yaml"
            ),
            pd_gains_file=os.path.join(
                self.copy_root, _ATLAS, "controller", "joint_pd_gains.yaml"
            ),
            repo_root=self.repo_root,
        )
        self.assertEqual(
            files, config_files.GuiConfigFiles(self.task, self.reference, self.pd_gains)
        )

    def test_the_task_file_alone_finds_the_others_next_to_it(self):
        files = config_files.resolve_config_files(
            task_file=self.task, repo_root=self.repo_root
        )
        self.assertEqual(
            files, config_files.GuiConfigFiles(self.task, self.reference, self.pd_gains)
        )

    def test_the_reference_file_alone_finds_the_task_file_next_to_it(self):
        files = config_files.resolve_config_files(
            reference_file=self.reference, repo_root=self.repo_root
        )
        self.assertEqual(
            files, config_files.GuiConfigFiles(self.task, self.reference, self.pd_gains)
        )

    def test_nothing_given_falls_back_to_a_shipped_robot(self):
        # The fake checkout holds the first default task file.
        files = config_files.resolve_config_files(repo_root=self.repo_root)
        self.assertEqual(
            files.task_file,
            os.path.join(self.repo_root, config_files.DEFAULT_TASK_FILES[0]),
        )
        self.assertEqual(files.reference_file, self.reference)

    def test_without_a_checkout_and_files_nothing_is_found(self):
        self.assertEqual(
            config_files.resolve_config_files(), config_files.GuiConfigFiles("", "", "")
        )

    def test_the_shipped_defaults_exist(self):
        # Every robot the GUI falls back to is still in the repository.
        for path in tuple(config_files.DEFAULT_TASK_FILES) + tuple(
            config_files.DEFAULT_PD_GAINS_FILES
        ):
            with self.subTest(path=path):
                self.assertTrue(os.path.isfile(repo_path(path)), path)


class TestDefaults(FakeCheckoutTestCase):
    def test_the_pelvis_height_is_reference_yamls_default_base_height(self):
        self.assertAlmostEqual(
            config_files.read_default_pelvis_height(self.reference), 0.91
        )

    def test_a_missing_or_unreadable_reference_falls_back(self):
        self.assertEqual(config_files.read_default_pelvis_height("", fallback=0.5), 0.5)
        broken = _write(
            os.path.join(self.tmpdir, "broken.yaml"), "defaultBaseHeight: [unclosed\n"
        )
        self.assertEqual(
            config_files.read_default_pelvis_height(broken, fallback=0.5), 0.5
        )
        no_key = _write(os.path.join(self.tmpdir, "no_key.yaml"), "other: 1\n")
        self.assertEqual(
            config_files.read_default_pelvis_height(no_key, fallback=0.5), 0.5
        )

    def test_online_tuning_follows_the_task_file_and_defaults_to_on(self):
        self.assertFalse(config_files.read_online_tuning(self.task))
        self.assertTrue(config_files.read_online_tuning(""))
        on = _write(
            os.path.join(self.tmpdir, "on.yaml"), "enable_online_tuning: true\n"
        )
        self.assertTrue(config_files.read_online_tuning(on))
        silent = _write(os.path.join(self.tmpdir, "silent.yaml"), "other: 1\n")
        self.assertTrue(config_files.read_online_tuning(silent))

    def test_every_shipped_task_file_reads(self):
        # The flag of a real robot is a bool, whatever its value.
        task = repo_path(_ATLAS, "mpc", "task.yaml")
        self.assertIsInstance(config_files.read_online_tuning(task), bool)
        self.assertGreater(
            config_files.read_default_pelvis_height(
                repo_path(_ATLAS, "command", "reference.yaml"), fallback=-1.0
            ),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
