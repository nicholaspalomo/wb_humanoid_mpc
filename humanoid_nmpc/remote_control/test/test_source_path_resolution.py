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

"""Which configuration files the GUI edits: found in the source checkout, never in Bazel's runfiles.

Validates remote_control/config_files.py:
  - resolve_source_path re-roots a path through robot_models/ (a runfiles link, a copied tree) into the checkout;
  - find_repo_root and resolve_input_path find the checkout and the files named on the command line, under `bazel run`
    (BUILD_WORKSPACE_DIRECTORY, BUILD_WORKING_DIRECTORY) and without it;
  - resolve_config_files finds the files the GUI was not given next to the ones it was;
  - the defaults read from the reference and task files.
Every case builds a fake checkout in a temporary directory, so nothing here depends on the caller's environment.
"""

import os
import shutil
import tempfile
import unittest
from unittest import mock

import operator_test_support
from remote_control import config_files

_ATLAS = os.path.join("robot_models", "drc_atlas", "drc_atlas_centroidal_mpc", "config")

_TASK_HEADER = (
    "# proto-file: humanoid_nmpc/humanoid_mpc_config/task_file.proto\n"
    "# proto-message: humanoid_mpc_config.TaskFile\n"
)
_REFERENCE_HEADER = (
    "# proto-file: humanoid_nmpc/humanoid_mpc_config/reference_file.proto\n"
    "# proto-message: humanoid_mpc_config.ReferenceFile\n"
)
_PD_GAINS_HEADER = (
    "# proto-file: humanoid_nmpc/humanoid_mpc_config/joint_pd_gains_file.proto\n"
    "# proto-message: humanoid_mpc_config.JointPdGainsFile\n"
)


def _write(path, text=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
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
            os.path.join(self.config, "mpc", "task.textproto"),
            _TASK_HEADER + "enable_online_tuning: false\n",
        )
        self.reference = _write(
            os.path.join(self.config, "command", "reference.textproto"),
            _REFERENCE_HEADER + "default_base_height: 0.91\n",
        )
        self.pd_gains = _write(
            os.path.join(self.config, "controller", "joint_pd_gains.textproto"),
            _PD_GAINS_HEADER + "default_gains { kp: 1 }\n",
        )
        # A copy such as a runfiles tree holds.
        self.copy_root = os.path.join(self.tmpdir, "runfiles", "_main")
        shutil.copytree(
            os.path.join(self.repo_root, "robot_models"),
            os.path.join(self.copy_root, "robot_models"),
        )
        self.copied_task = os.path.join(self.copy_root, _ATLAS, "mpc", "task.textproto")


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
        elsewhere = _write(
            os.path.join(self.tmpdir, "some", "random", "file.textproto")
        )
        self.assertEqual(
            config_files.resolve_source_path(elsewhere, self.repo_root), elsewhere
        )
        self.assertEqual(config_files.resolve_source_path("", self.repo_root), "")
        missing = "/does/not/exist/robot_models/task.textproto"
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
        assert root is not None
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
            config_files.resolve_input_path(
                "/an/absolute/path.textproto", self.repo_root
            ),
            "/an/absolute/path.textproto",
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
                    "task.textproto",
                ),
                self.repo_root,
            )
        self.assertEqual(resolved, self.task)

    def test_then_relative_to_the_checkout(self):
        with mock.patch.dict(os.environ, {"BUILD_WORKING_DIRECTORY": self.tmpdir}):
            resolved = config_files.resolve_input_path(
                os.path.join(_ATLAS, "mpc", "task.textproto"), self.repo_root
            )
        self.assertEqual(resolved, self.task)

    def test_a_file_that_exists_nowhere_names_the_first_candidate(self):
        with mock.patch.dict(os.environ, {"BUILD_WORKING_DIRECTORY": self.tmpdir}):
            resolved = config_files.resolve_input_path(
                "missing.textproto", self.repo_root
            )
        self.assertEqual(resolved, os.path.join(self.tmpdir, "missing.textproto"))

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
                self.copy_root, _ATLAS, "command", "reference.textproto"
            ),
            pd_gains_file=os.path.join(
                self.copy_root, _ATLAS, "controller", "joint_pd_gains.textproto"
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

    def test_the_contact_planning_file_is_found_next_to_the_task_file(self):
        self.assertIsNone(config_files.contact_planning_file(self.task))
        planner = _write(os.path.join(self.config, "mpc", "contact_planning.textproto"))
        self.assertEqual(config_files.contact_planning_file(self.task), planner)
        self.assertIsNone(config_files.contact_planning_file(""))

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

    def test_a_given_file_that_does_not_exist_is_refused_not_replaced(self):
        # Never another robot's files in its place: the GUI would publish them to the robot it is connected to.
        missing = os.path.join(self.tmpdir, "config", "mpc", "missing.textproto")
        with self.assertRaisesRegex(config_files.ConfigFilesError, "missing.textproto"):
            config_files.resolve_config_files(
                task_file=missing, repo_root=self.repo_root
            )
        yaml = os.path.join(self.config, "mpc", "task.yaml")
        with self.assertRaisesRegex(
            config_files.ConfigFilesError, r"task\.yaml.*\.textproto files"
        ):
            config_files.resolve_config_files(task_file=yaml, repo_root=self.repo_root)
        for keyword in ("reference_file", "pd_gains_file"):
            with self.subTest(file=keyword):
                with self.assertRaises(config_files.ConfigFilesError):
                    config_files.resolve_config_files(
                        **{keyword: missing}, repo_root=self.repo_root
                    )

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
                self.assertTrue(
                    os.path.isfile(operator_test_support.repo_path(path)), path
                )


class TestDefaults(FakeCheckoutTestCase):
    def test_the_pelvis_height_is_the_reference_files_default_base_height(self):
        self.assertAlmostEqual(
            config_files.read_default_pelvis_height(self.reference), 0.91
        )

    def test_a_missing_or_unreadable_reference_falls_back(self):
        self.assertEqual(config_files.read_default_pelvis_height("", fallback=0.5), 0.5)
        broken = _write(
            os.path.join(self.tmpdir, "broken.textproto"),
            _REFERENCE_HEADER + "default_base_height: [unclosed\n",
        )
        self.assertEqual(
            config_files.read_default_pelvis_height(broken, fallback=0.5), 0.5
        )
        unknown = _write(
            os.path.join(self.tmpdir, "unknown.textproto"),
            _REFERENCE_HEADER + "defaultBaseHeight: 0.9\n",
        )
        self.assertEqual(
            config_files.read_default_pelvis_height(unknown, fallback=0.5), 0.5
        )
        no_key = _write(
            os.path.join(self.tmpdir, "no_key.textproto"),
            _REFERENCE_HEADER + "max_rotation_velocity: 1\n",
        )
        self.assertEqual(
            config_files.read_default_pelvis_height(no_key, fallback=0.5), 0.5
        )

    def test_online_tuning_follows_the_task_file_and_defaults_to_on(self):
        self.assertFalse(config_files.read_online_tuning_state(self.task).enabled)
        self.assertTrue(config_files.read_online_tuning_state("").enabled)
        on = _write(
            os.path.join(self.tmpdir, "on.textproto"),
            _TASK_HEADER + "enable_online_tuning: true\n",
        )
        self.assertTrue(config_files.read_online_tuning_state(on).enabled)
        silent = _write(
            os.path.join(self.tmpdir, "silent.textproto"),
            _TASK_HEADER + "terrain_height: 0.1\n",
        )
        self.assertTrue(config_files.read_online_tuning_state(silent).enabled)
        # A file that does not parse into the schema (the retired YAML spelling) turns tuning off, saying where.
        retired = _write(
            os.path.join(self.tmpdir, "retired.textproto"),
            _TASK_HEADER + "enableOnlineTuning: false\n",
        )
        state = config_files.read_online_tuning_state(retired)
        self.assertFalse(state.enabled)
        self.assertIn(f"{retired}:3:", state.reason)
        self.assertEqual(config_files.read_online_tuning_state(on).reason, "")
        self.assertIn(
            "enable_online_tuning: false",
            config_files.read_online_tuning_state(self.task).reason,
        )

    def test_every_shipped_task_file_reads(self):
        # The flag of a real robot is a bool, whatever its value.
        task = operator_test_support.repo_path(_ATLAS, "mpc", "task.textproto")
        state = config_files.read_online_tuning_state(task)
        self.assertIsInstance(state.enabled, bool)
        self.assertNotIn("does not parse", state.reason)
        self.assertGreater(
            config_files.read_default_pelvis_height(
                operator_test_support.repo_path(
                    _ATLAS, "command", "reference.textproto"
                ),
                fallback=-1.0,
            ),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
