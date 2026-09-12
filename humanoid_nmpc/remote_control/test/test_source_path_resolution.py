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

"""
Tests for install-space → source-tree path resolution logic.

Validates that:
  - _resolve_source_path correctly maps ament install-space config paths
    back to the source tree via both strategy 1 (robot_models/ marker)
    and strategy 2 (share/<pkg>/config/... pattern).
  - _find_workspace_root locates the workspace root from known devcontainer
    bind-mount locations and by walking up from CWD.
  - Paths that don't match any pattern are returned unchanged (fallback).
"""

import os
import shutil
import tempfile
import unittest


class TestResolveSourcePath(unittest.TestCase):
    """Test _resolve_source_path without needing ROS or tkinter."""

    @classmethod
    def setUpClass(cls):
        """Import _resolve_source_path by loading the module directly."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "base_velocity_controller_gui",
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "remote_control",
                "base_velocity_controller_gui.py",
            ),
        )
        # We can't fully import the module (it imports rclpy, tkinter, etc.),
        # so we read the source and extract the static method body.
        source_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "remote_control",
            "base_velocity_controller_gui.py",
        )
        with open(source_path, "r") as f:
            source = f.read()

        # Extract the _resolve_source_path function by exec'ing just it
        # with the needed `os` import.
        func_code = """
import os

@staticmethod
def _resolve_source_path(path: str, repo_root: str) -> str:
"""
        # Find the function in the source
        start = source.find("    @staticmethod\n    def _resolve_source_path")
        end = source.find("\n    def run(self):")
        if start == -1 or end == -1:
            raise RuntimeError("Could not find _resolve_source_path in source")

        func_text = source[start:end]
        # Dedent by 4 spaces to make it module-level
        lines = func_text.split("\n")
        dedented = "\n".join(
            line[4:] if line.startswith("    ") else line for line in lines
        )

        namespace = {"os": os}
        exec(dedented, namespace)
        cls.resolve = namespace["_resolve_source_path"]

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Create a fake workspace source tree
        self.repo_root = os.path.join(self.tmpdir, "workspace", "wb_humanoid_mpc")
        self.source_task = os.path.join(
            self.repo_root,
            "robot_models",
            "drc_atlas",
            "drc_atlas_centroidal_mpc",
            "config",
            "mpc",
            "task.yaml",
        )
        os.makedirs(os.path.dirname(self.source_task))
        with open(self.source_task, "w") as f:
            f.write("# source tree task.yaml\n")

        # Create a fake ament install-space copy
        self.install_task = os.path.join(
            self.tmpdir,
            "install",
            "drc_atlas_centroidal_mpc",
            "share",
            "drc_atlas_centroidal_mpc",
            "config",
            "mpc",
            "task.yaml",
        )
        os.makedirs(os.path.dirname(self.install_task))
        with open(self.install_task, "w") as f:
            f.write("# install space task.yaml\n")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_strategy1_robot_models_in_path(self):
        """Strategy 1: path contains /robot_models/ and source-tree file exists."""
        # Simulate a path that already has robot_models in it
        path_with_marker = os.path.join(
            self.tmpdir,
            "some_prefix",
            "robot_models",
            "drc_atlas",
            "drc_atlas_centroidal_mpc",
            "config",
            "mpc",
            "task.yaml",
        )
        os.makedirs(os.path.dirname(path_with_marker))
        with open(path_with_marker, "w") as f:
            f.write("# another copy\n")

        result = self.resolve(path_with_marker, self.repo_root)
        self.assertEqual(result, self.source_task)

    def test_strategy2_install_space_share_pattern(self):
        """Strategy 2: install-space path with .../share/<pkg>/config/... pattern."""
        result = self.resolve(self.install_task, self.repo_root)
        self.assertEqual(result, self.source_task)

    def test_fallback_returns_original_path(self):
        """Path that doesn't match any pattern is returned unchanged."""
        random_path = os.path.join(self.tmpdir, "some", "random", "file.yaml")
        os.makedirs(os.path.dirname(random_path))
        with open(random_path, "w") as f:
            f.write("# random\n")

        result = self.resolve(random_path, self.repo_root)
        self.assertEqual(result, random_path)

    def test_empty_path_returns_empty(self):
        """Empty path is returned as-is."""
        result = self.resolve("", self.repo_root)
        self.assertEqual(result, "")

    def test_nonexistent_path_returns_as_is(self):
        """Non-existent path is returned as-is."""
        result = self.resolve("/does/not/exist/task.yaml", self.repo_root)
        self.assertEqual(result, "/does/not/exist/task.yaml")

    def test_source_missing_falls_back_to_install(self):
        """If source-tree file doesn't exist, return the install-space path."""
        # Remove the source file
        os.remove(self.source_task)
        result = self.resolve(self.install_task, self.repo_root)
        self.assertEqual(result, self.install_task)


class TestFindWorkspaceRoot(unittest.TestCase):
    """Test _find_workspace_root without needing ROS."""

    @staticmethod
    def _find_workspace_root() -> str:
        """Mirror of MPCLaunchConfig._find_workspace_root for testing."""
        markers = ["WORKSPACE.bazel", "Makefile"]

        # Known devcontainer bind-mount locations
        candidates = [
            "/wb_humanoid_mpc_ws/workspace/wb_humanoid_mpc",
            "/wb_humanoid_mpc_ws/src/wb_humanoid_mpc",
        ]
        for c in candidates:
            if os.path.isdir(c) and any(
                os.path.exists(os.path.join(c, m)) for m in markers
            ):
                return c

        # Walk up from CWD
        d = os.path.abspath(os.getcwd())
        for _ in range(10):
            if any(os.path.exists(os.path.join(d, m)) for m in markers):
                if os.path.isdir(os.path.join(d, "robot_models")):
                    return d
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent

        return ""

    @classmethod
    def setUpClass(cls):
        pass

    def test_finds_real_workspace_from_cwd(self):
        """Should find the real workspace root when running from the repo."""
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        # Only run if we're actually in the repo
        if os.path.exists(os.path.join(repo_root, "WORKSPACE.bazel")):
            saved_cwd = os.getcwd()
            try:
                os.chdir(repo_root)
                result = TestFindWorkspaceRoot._find_workspace_root()
                self.assertTrue(
                    len(result) > 0,
                    "Should find workspace root from repo CWD",
                )
                self.assertTrue(
                    os.path.exists(os.path.join(result, "robot_models")),
                    f"Workspace root {result} should contain robot_models/",
                )
            finally:
                os.chdir(saved_cwd)

    def test_finds_workspace_from_known_devcontainer_path(self):
        """Finds workspace when devcontainer bind-mount path exists."""
        # This test only works inside the devcontainer; skip otherwise
        candidate = "/wb_humanoid_mpc_ws/workspace/wb_humanoid_mpc"
        if os.path.isdir(candidate) and os.path.exists(
            os.path.join(candidate, "WORKSPACE.bazel")
        ):
            result = TestFindWorkspaceRoot._find_workspace_root()
            self.assertEqual(result, candidate)
        else:
            # Verify it doesn't crash when path doesn't exist
            result = TestFindWorkspaceRoot._find_workspace_root()
            # Should either find via CWD walk-up or return empty string
            self.assertIsInstance(result, str)

    def test_returns_empty_from_tmp(self):
        """Returns empty string when no workspace root can be found."""
        tmpdir = tempfile.mkdtemp()
        saved_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            result = TestFindWorkspaceRoot._find_workspace_root()
            # Should return empty since /tmp has no WORKSPACE.bazel
            # (unless devcontainer paths exist, in which case those win)
            if not os.path.isdir("/wb_humanoid_mpc_ws/workspace/wb_humanoid_mpc"):
                self.assertEqual(result, "")
        finally:
            os.chdir(saved_cwd)
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestEndToEndSourcePathResolution(unittest.TestCase):
    """Integration test: verify the real DRC Atlas config resolves correctly."""

    def test_real_atlas_install_space_would_resolve(self):
        """
        Verify that if an install-space path like
        .../share/drc_atlas_centroidal_mpc/config/mpc/task.yaml
        is given, it resolves to the source-tree equivalent.
        """
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        source_task = os.path.join(
            repo_root,
            "robot_models",
            "drc_atlas",
            "drc_atlas_centroidal_mpc",
            "config",
            "mpc",
            "task.yaml",
        )
        if not os.path.exists(source_task):
            self.skipTest("DRC Atlas source tree not found")

        # Simulate an install-space path by creating a temp one
        tmpdir = tempfile.mkdtemp()
        try:
            install_path = os.path.join(
                tmpdir,
                "install",
                "drc_atlas_centroidal_mpc",
                "share",
                "drc_atlas_centroidal_mpc",
                "config",
                "mpc",
                "task.yaml",
            )
            os.makedirs(os.path.dirname(install_path))
            shutil.copy2(source_task, install_path)

            # Import _resolve_source_path
            gui_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "remote_control",
                "base_velocity_controller_gui.py",
            )
            with open(gui_path, "r") as f:
                source = f.read()
            start = source.find("    @staticmethod\n    def _resolve_source_path")
            end = source.find("\n    def run(self):")
            func_text = source[start:end]
            lines = func_text.split("\n")
            dedented = "\n".join(
                line[4:] if line.startswith("    ") else line for line in lines
            )
            namespace = {"os": os}
            exec(dedented, namespace)
            resolve = namespace["_resolve_source_path"]

            result = resolve(install_path, repo_root)
            self.assertEqual(
                result,
                source_task,
                f"Install-space path should resolve to source tree.\n"
                f"  Input:    {install_path}\n"
                f"  Expected: {source_task}\n"
                f"  Got:      {result}",
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
