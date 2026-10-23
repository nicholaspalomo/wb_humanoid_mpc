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

"""Tests tools/ide/clangd_in_dev_container.sh with a stand-in `docker` that runs the container's command here.

The script is copied into a scratch checkout at <home>/workspace/wb_humanoid_mpc, the layout docker-compose.yaml mounts
at /wb_humanoid_mpc_ws, so the tests see the paths the editor and clangd exchange.
"""

import os
import shutil
import subprocess
import tempfile
import unittest

# Records its arguments; answers `inspect` with $FAKE_RUNNING and runs the command of `exec` after the container name.
_FAKE_DOCKER = """#!/bin/sh
printf '%s\\n' "$@" > "$FAKE_LOG/docker_$1"
case "$1" in
  inspect) echo "${FAKE_RUNNING:-true}" ;;
  exec)
    shift
    while [ "$#" -gt 0 ]; do
      case "$1" in
        --interactive) shift ;;
        --workdir) shift 2 ;;
        *) shift; break ;;
      esac
    done
    exec "$@"
    ;;
esac
"""

# A clangd that records its arguments.
_FAKE_CLANGD = """#!/bin/sh
printf '%s\\n' "$0" "$@" > "$FAKE_LOG/clangd"
"""


def _runfile(relative_path: str) -> str:
    """Returns a data file of this test, found in the runfiles when run by Bazel and in the source tree otherwise."""
    roots = []
    if "TEST_SRCDIR" in os.environ:
        roots.append(os.path.join(os.environ["TEST_SRCDIR"], "_main"))
    roots.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    for root in roots:
        candidate = os.path.join(root, relative_path)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(relative_path)


def _write_executable(path: str, content: str) -> None:
    """Writes an executable file, creating its directory."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    os.chmod(path, 0o755)


class ClangdInDevContainerTest(unittest.TestCase):
    """The wrapper reaches the container, maps the paths and picks the container's clangd."""

    def setUp(self) -> None:
        # pylint: disable-next=consider-using-with  # tearDown() deletes it.
        self._directory = tempfile.TemporaryDirectory()
        self.home = os.path.realpath(self._directory.name)
        self.repo = os.path.join(self.home, "workspace", "wb_humanoid_mpc")
        self.script = os.path.join(
            self.repo, "tools", "ide", "clangd_in_dev_container.sh"
        )
        os.makedirs(os.path.dirname(self.script))
        shutil.copy(_runfile("tools/ide/clangd_in_dev_container.sh"), self.script)
        os.chmod(self.script, 0o755)
        self.bin = os.path.join(self.home, "bin")
        _write_executable(os.path.join(self.bin, "docker"), _FAKE_DOCKER)
        # Only the tools the scripts use, so that a clangd installed on this machine is not found.
        self.tools = os.path.join(self.home, "tools")
        os.mkdir(self.tools)
        for tool in ("sh", "dirname", "basename"):
            found = shutil.which(tool)
            assert found is not None, tool
            os.symlink(found, os.path.join(self.tools, tool))
        self.log = os.path.join(self.home, "log")
        os.mkdir(self.log)

    def tearDown(self) -> None:
        self._directory.cleanup()

    def _run(
        self, *arguments: str, **environment: str
    ) -> subprocess.CompletedProcess[str]:
        """Runs the wrapper on the host, with the stand-in docker, and returns it, finished."""
        return subprocess.run(
            [self.script, *arguments],
            env={
                "PATH": f"{self.bin}{os.pathsep}{self.tools}",
                "HOME": self.home,
                "FAKE_LOG": self.log,
                "WB_DOCKERENV": os.path.join(self.home, "no_dockerenv"),
                **environment,
            },
            capture_output=True,
            text=True,
            check=False,
        )

    def _logged(self, name: str) -> list[str]:
        """Returns the arguments the stand-in `name` recorded, one per line."""
        with open(os.path.join(self.log, name), encoding="utf-8") as f:
            return f.read().splitlines()

    def test_runs_the_containers_clangd_with_the_mount_mapped(self) -> None:
        _write_executable(os.path.join(self.bin, "clangd-21"), _FAKE_CLANGD)
        completed = self._run("--log=verbose")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        docker = self._logged("docker_exec")
        self.assertEqual(
            docker[:4],
            [
                "exec",
                "--interactive",
                "--workdir",
                "/wb_humanoid_mpc_ws/workspace/wb_humanoid_mpc",
            ],
        )
        self.assertEqual(docker[4], "devcontainer-app-1")
        clangd = self._logged("clangd")
        self.assertEqual(clangd[0], os.path.join(self.bin, "clangd-21"))
        self.assertIn(
            f"--path-mappings={self.repo}=/wb_humanoid_mpc_ws/workspace/wb_humanoid_mpc",
            clangd,
        )
        self.assertIn(
            "--compile-commands-dir=/wb_humanoid_mpc_ws/workspace/wb_humanoid_mpc",
            clangd,
        )
        self.assertIn("-j=2", clangd)
        self.assertEqual(clangd[-1], "--log=verbose")

    def test_maps_the_checkout_to_the_path_named_inside_the_container(self) -> None:
        _write_executable(os.path.join(self.bin, "clangd-21"), _FAKE_CLANGD)
        completed = self._run(WB_DEV_CONTAINER_REPO="/srv/checkout")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(self._logged("docker_exec")[3], "/srv/checkout")
        clangd = self._logged("clangd")
        self.assertIn(f"--path-mappings={self.repo}=/srv/checkout", clangd)
        self.assertIn("--compile-commands-dir=/srv/checkout", clangd)

    def test_names_another_container(self) -> None:
        _write_executable(os.path.join(self.bin, "clangd-21"), _FAKE_CLANGD)
        completed = self._run(WB_DEV_CONTAINER="other-dev")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(self._logged("docker_inspect")[-1], "other-dev")
        self.assertEqual(self._logged("docker_exec")[4], "other-dev")

    def test_falls_back_to_the_editors_clangd_through_the_mount(self) -> None:
        extension_clangd = os.path.join(
            self.home,
            ".antigravity-ide-server/data/User/globalStorage/llvm-vs-code-extensions.vscode-clangd",
            "install/22.1.6/clangd_22.1.6/bin/clangd",
        )
        _write_executable(extension_clangd, _FAKE_CLANGD)
        self._run()
        self.assertIn(
            "/wb_humanoid_mpc_ws/.antigravity-ide-server/data/User/globalStorage/llvm-vs-code-extensions.vscode-clangd/"
            "install/22.1.6/clangd_22.1.6/bin/clangd",
            self._logged("docker_exec"),
        )

    def test_fails_without_any_clangd(self) -> None:
        completed = self._run()
        self.assertEqual(completed.returncode, 1)
        self.assertIn("no clangd-21", completed.stderr)

    def test_in_the_container_runs_clangd_directly(self) -> None:
        _write_executable(os.path.join(self.bin, "clangd-21"), _FAKE_CLANGD)
        dockerenv = os.path.join(self.home, "dockerenv")
        with open(dockerenv, "w", encoding="utf-8"):
            pass
        completed = self._run("--log=verbose", WB_DOCKERENV=dockerenv)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertFalse(os.path.exists(os.path.join(self.log, "docker_exec")))
        clangd = self._logged("clangd")
        self.assertIn(f"--compile-commands-dir={self.repo}", clangd)
        self.assertFalse(
            [argument for argument in clangd if argument.startswith("--path-mappings")]
        )
        self.assertEqual(clangd[-1], "--log=verbose")

    def test_fails_when_the_container_is_not_running(self) -> None:
        completed = self._run(FAKE_RUNNING="false")
        self.assertEqual(completed.returncode, 1)
        self.assertIn("'devcontainer-app-1' is not running", completed.stderr)


if __name__ == "__main__":
    unittest.main()
