"""setup_env.sh is sourced by every shell, launch and CI build (`make build-all` runs `source setup_env.sh && bazel
build //...`), so it must not change the shell it runs in beyond the environment it exports.

It once opened a lock with `exec {lock_fd}>file 2>/dev/null`. `exec` with redirections and no command applies all of
them to the shell for good, so that line also sent the shell's stderr to /dev/null. It runs only when a package's
assets have to be copied - always in a fresh container, never in a warm dev container - so every error of CI's Bazel
build vanished there while the same build printed its errors at a developer's desk.

It also must not set the environment up differently before a build and after it. .bazelrc passes PATH, LD_LIBRARY_PATH,
PYTHONPATH and AMENT_PREFIX_PATH into the actions' and tests' cache keys, and the script used to add .bazel/bin and the
message packages' directories only once they existed. CI's `bazel test` then rebuilt everything its `bazel build` had
just built, and a shell set up before the message repositories were fetched ran the ROS tests without the typesupport
libraries rosidl loads at run time.
"""

import glob
import hashlib
import os
import shutil
import subprocess
import tempfile
import unittest


def _runfile(relative_path):
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


def _simulate_build(output_base, workspace):
    """Creates what a build leaves behind: the message repositories in `output_base`, and `workspace`/.bazel/ pointing
    into it."""
    python = [os.path.basename(d) for d in glob.glob("/opt/ros/*/lib/python3.*")] or [
        "python3.12"
    ]
    for package in ("humanoid_mpc_msgs", "ocs2_ros2_msgs"):
        prefix = os.path.join(
            output_base, "external", f"+system_libs+{package}_repo", "install", package
        )
        os.makedirs(os.path.join(prefix, "share"))
        os.makedirs(os.path.join(prefix, "lib", python[0], "site-packages"))
    bazel_out = os.path.join(output_base, "execroot", "_main", "bazel-out")
    os.makedirs(os.path.join(bazel_out, "k8-opt", "bin"))
    symlinks = os.path.join(workspace, ".bazel")
    os.makedirs(symlinks, exist_ok=True)
    for name, target in (
        ("out", bazel_out),
        ("bin", os.path.join(bazel_out, "k8-opt", "bin")),
    ):
        if os.path.lexists(os.path.join(symlinks, name)):
            os.remove(os.path.join(symlinks, name))
        os.symlink(target, os.path.join(symlinks, name))


class SetupEnvTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.script = _runfile("setup_env.sh")

    def tearDown(self):
        self.directory.cleanup()

    def source_then(self, commands, script=None, home=None):
        """Sources setup_env.sh into a fresh bash with an empty install directory, then runs `commands` in that shell."""
        environment = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": home or self.directory.name,
            "USER": "tester",
            # An empty install directory: every package has to be installed, which is the path that takes the lock.
            "TMPDIR": self.directory.name,
        }
        return subprocess.run(
            [
                "bash",
                "-c",
                'source "$0" >/dev/null; ' + commands,
                script or self.script,
            ],
            env=environment,
            capture_output=True,
            text=True,
            timeout=300,
        )

    def test_the_lock_path_is_exercised(self):
        # Without flock the script skips the lock, and the test below could not fail.
        self.assertIsNotNone(
            shutil.which("flock"), "flock (util-linux) is needed to exercise the lock"
        )
        result = self.source_then('ls -a "$TMPDIR/.bazel_ros_install"')
        self.assertIn(".remote_control.lock", result.stdout)

    def test_stderr_still_reaches_the_caller_after_sourcing(self):
        result = self.source_then("echo stderr-is-visible >&2; exit 3")
        self.assertIn("stderr-is-visible", result.stderr)
        # And the shell is otherwise usable: its own exit status comes through.
        self.assertEqual(result.returncode, 3)

    def test_the_environment_does_not_depend_on_the_build(self):
        # A workspace of its own, so that the test controls what a build has and has not created in it.
        workspace = os.path.join(self.directory.name, "workspace")
        home = os.path.join(self.directory.name, "home")
        os.makedirs(workspace)
        os.makedirs(home)
        script = os.path.join(workspace, "setup_env.sh")
        shutil.copy(self.script, script)
        names = ("PATH", "LD_LIBRARY_PATH", "PYTHONPATH", "AMENT_PREFIX_PATH")
        print_variables = 'printf "%s\\n" ' + " ".join(f'"${name}"' for name in names)

        def environment():
            result = self.source_then(print_variables, script=script, home=home)
            self.assertEqual(result.returncode, 0, result.stderr)
            return dict(zip(names, result.stdout.splitlines()))

        before = environment()

        # What this environment's build leaves behind: Bazel's default output base for the workspace, the message
        # repositories in it and the .bazel/ symlinks into it. Every directory the script used to probe for is created.
        workspace_hash = hashlib.md5(os.path.realpath(workspace).encode()).hexdigest()
        output_base = os.path.join(
            home, ".cache", "bazel", "_bazel_tester", workspace_hash
        )
        _simulate_build(output_base, workspace)
        after_a_build = environment()

        # And what a build from elsewhere leaves: the checkout is shared with the host and with `make ci-local`'s
        # container, whose builds point the .bazel/ symlinks into output bases of their own.
        _simulate_build(
            os.path.join(self.directory.name, "elsewhere", "_bazel_root", "0" * 32),
            workspace,
        )
        after_a_build_elsewhere = environment()

        for name in names:
            self.assertEqual(
                before[name],
                after_a_build[name],
                f"{name} changed once a build had run",
            )
            self.assertEqual(
                before[name],
                after_a_build_elsewhere[name],
                f"{name} followed the .bazel/ symlinks of another environment's build",
            )
        # And the directories are there all along, not missing throughout.
        self.assertIn(
            os.path.join(
                output_base,
                "external",
                "+system_libs+ocs2_ros2_msgs_repo",
                "install",
                "ocs2_ros2_msgs",
                "lib",
            ),
            before["LD_LIBRARY_PATH"].split(":"),
        )
        self.assertIn(
            os.path.join(workspace, ".bazel", "bin"), before["PATH"].split(":")
        )

    def test_the_computed_output_base_is_the_one_bazel_uses(self):
        # The test above takes md5(workspace path) under _bazel_<user> to be Bazel's output base, as setup_env.sh does. Check that against the Bazel running this test: its runfiles live in that output base.
        runfiles = os.path.realpath(os.environ.get("TEST_SRCDIR", ""))
        if "/execroot/" not in runfiles:
            self.skipTest("not run by Bazel")
        output_base = runfiles.split("/execroot/")[0].split("/sandbox/")[0]
        workspace = os.path.dirname(os.path.realpath(_runfile("setup_env.sh")))
        if workspace.startswith(output_base + os.sep):
            self.skipTest(
                "setup_env.sh is a copy in the output base, not the workspace's own file"
            )
        self.assertEqual(
            os.path.basename(output_base),
            hashlib.md5(workspace.encode()).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
