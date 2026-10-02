"""setup_env.sh is sourced by every shell of the dev container (through BASH_ENV), every CI build (`make build-all` runs
`source setup_env.sh && bazel build //...`) and every launch, so it must only export variables, and export the same
ones however often and whenever it is sourced.

It once opened a lock with `exec {lock_fd}>file 2>/dev/null`. `exec` with redirections and no command applies all of
them to the shell for good, so that line also sent the shell's stderr to /dev/null, and every error of CI's Bazel
build vanished. It also set the environment up differently before a build and after it: .bazelrc passes PATH into
every action's cache key, and the script used to add .bazel/bin only once a build had created it, so CI's `bazel test`
rebuilt everything its `bazel build` had just built.

It leaves LD_LIBRARY_PATH alone: robotpkg's libraries find each other through their RUNPATH, and robotpkg's lib
directory holds a libblasfeo.so of another version than the solver's. (//tools/dev_container:test_dev_container keeps
ROS out of it.)
"""

import os
import shutil
import subprocess
import tempfile
import unittest
from typing import Dict, Optional

_NAMES = ("PATH", "PYTHONPATH", "LD_LIBRARY_PATH", "DISPLAY")
_ROBOTPKG_PREFIX = "/opt/openrobots"


def _runfile(relative_path: str) -> str:
    roots = []
    if "TEST_SRCDIR" in os.environ:
        roots += [
            os.path.join(os.environ["TEST_SRCDIR"], "_main"),
            os.environ["TEST_SRCDIR"],
        ]
    roots.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    for root in roots:
        candidate = os.path.join(root, relative_path)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(relative_path)


class SetupEnvTest(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = directory.name
        # A checkout of its own, so that the test controls what a build has and has not created in it.
        self.workspace = os.path.join(self.directory, "workspace")
        os.makedirs(self.workspace)
        self.script = os.path.join(self.workspace, "setup_env.sh")
        shutil.copy(_runfile("setup_env.sh"), self.script)

    def source_then(
        self, commands: str, extra_environment: Optional[Dict[str, str]] = None
    ) -> subprocess.CompletedProcess:
        """Sources setup_env.sh into a fresh bash with a minimal environment, then runs `commands` in that shell."""
        environment = {
            "PATH": "/usr/bin:/bin",
            "HOME": self.directory,
            "USER": "tester",
            "TMPDIR": self.directory,
        }
        environment.update(extra_environment or {})
        return subprocess.run(
            ["bash", "-c", 'source "$0" >/dev/null; ' + commands, self.script],
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def environment(self, commands: str = "", **extra: str) -> Dict[str, str]:
        """The variables of _NAMES after sourcing (and running `commands`); an unset one is absent."""
        print_variables = " ".join(
            f'[ -n "${{{name}+set}}" ] && printf "{name}=%s\\n" "${name}";'
            for name in _NAMES
        )
        result = self.source_then(commands + print_variables, extra)
        self.assertEqual(result.returncode, 0, result.stderr)
        return dict(line.split("=", 1) for line in result.stdout.splitlines())

    def test_stderr_still_reaches_the_caller_after_sourcing(self) -> None:
        result = self.source_then("echo stderr-is-visible >&2; exit 3")
        self.assertIn("stderr-is-visible", result.stderr)
        # And the shell is otherwise usable: its own exit status comes through.
        self.assertEqual(result.returncode, 3)

    def test_a_non_interactive_shell_prints_nothing(self) -> None:
        # Every bash script in the container sources it (BASH_ENV): its output would end up in theirs.
        result = subprocess.run(
            ["bash", "-c", 'source "$0"', self.script],
            env={"PATH": "/usr/bin:/bin", "HOME": self.directory},
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual((result.returncode, result.stdout, result.stderr), (0, "", ""))

    def test_it_writes_no_file(self) -> None:
        self.source_then("true")
        self.assertEqual(sorted(os.listdir(self.directory)), ["workspace"])
        self.assertEqual(os.listdir(self.workspace), ["setup_env.sh"])

    def test_the_bazel_binaries_and_robotpkg_are_on_the_path(self) -> None:
        path = self.environment()["PATH"].split(":")
        self.assertEqual(path[0], os.path.join(self.workspace, ".bazel", "bin"))
        self.assertIn(os.path.join(_ROBOTPKG_PREFIX, "bin"), path)
        self.assertIn("/usr/bin", path)

    def test_the_python_tools_of_the_checkout_are_on_the_python_path(self) -> None:
        python_path = self.environment()["PYTHONPATH"].split(":")
        self.assertIn(os.path.join(self.workspace, "humanoid_learning"), python_path)
        self.assertIn(
            os.path.join(
                self.workspace, "humanoid_nmpc", "humanoid_common_mpc_pyutils"
            ),
            python_path,
        )
        for entry in python_path:
            if entry.startswith(_ROBOTPKG_PREFIX):
                self.assertRegex(
                    entry, r"^/opt/openrobots/lib/python3\.\d+/site-packages$"
                )
                self.assertTrue(os.path.isdir(entry), entry)

    def test_robotpkg_s_python_bindings_are_on_the_python_path_where_installed(
        self,
    ) -> None:
        site_packages = [
            os.path.join(_ROBOTPKG_PREFIX, "lib", name, "site-packages")
            for name in (
                os.listdir(os.path.join(_ROBOTPKG_PREFIX, "lib"))
                if os.path.isdir(os.path.join(_ROBOTPKG_PREFIX, "lib"))
                else []
            )
            if name.startswith("python3.")
        ]
        site_packages = [path for path in site_packages if os.path.isdir(path)]
        if not site_packages:
            self.skipTest("robotpkg's Python bindings are not installed here")
        python_path = self.environment()["PYTHONPATH"].split(":")
        for path in site_packages:
            self.assertIn(path, python_path)

    def test_the_library_path_is_left_alone(self) -> None:
        self.assertNotIn("LD_LIBRARY_PATH", self.environment())
        self.assertEqual(
            self.environment(LD_LIBRARY_PATH="/some/lib")["LD_LIBRARY_PATH"],
            "/some/lib",
        )

    def test_sourcing_twice_changes_nothing(self) -> None:
        once = self.environment()
        twice = self.environment(f'source "{self.script}" >/dev/null; ')
        self.assertEqual(once, twice)

    def test_the_environment_does_not_depend_on_the_build(self) -> None:
        before = self.environment()
        # What a build leaves behind: the .bazel/ symlinks, here pointing into an output base elsewhere.
        output_base = os.path.join(
            self.directory, "output_base", "execroot", "_main", "bazel-out"
        )
        os.makedirs(os.path.join(output_base, "k8-opt", "bin"))
        os.makedirs(os.path.join(self.workspace, ".bazel"))
        for name, target in (
            ("out", output_base),
            ("bin", os.path.join(output_base, "k8-opt", "bin")),
        ):
            os.symlink(target, os.path.join(self.workspace, ".bazel", name))
        self.assertEqual(before, self.environment())

    def test_the_display_defaults_to_the_vnc_desktop_but_keeps_one_of_its_own(
        self,
    ) -> None:
        self.assertEqual(self.environment()["DISPLAY"], ":99")
        self.assertEqual(self.environment(DISPLAY=":1")["DISPLAY"], ":99")
        self.assertEqual(self.environment(DISPLAY="host:0")["DISPLAY"], "host:0")


if __name__ == "__main__":
    unittest.main()
