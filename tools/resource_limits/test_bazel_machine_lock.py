"""Tests for tools/bazel, the Bazelisk wrapper that makes the compiling commands of every Bazel on the machine take
turns. Two builds in two containers, each sized to the whole machine by .bazelrc, once crashed a 30 GB workstation.
"""

import os
import stat
import subprocess
import tempfile
import time
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


WRAPPER = _runfile("tools/bazel")
SHELL_INIT = _runfile(".devcontainer/shell_init.sh")

# A stand-in for Bazel: records when it ran, sleeps, optionally leaves a daemon behind (as the Bazel client leaves its
# server), and exits with a chosen status.
FAKE_BAZEL = """#!/bin/sh
echo "start $(date +%s.%N) $*" >> "$FAKE_LOG"
sleep "${FAKE_SLEEP:-0}"
if [ -n "${FAKE_DAEMON:-}" ]; then (sleep 30 &) ; fi
echo "end $(date +%s.%N) $*" >> "$FAKE_LOG"
exit "${FAKE_STATUS:-0}"
"""


class BazelMachineLockTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.real = os.path.join(self.directory.name, "real_bazel")
        with open(self.real, "w") as f:
            f.write(FAKE_BAZEL)
        os.chmod(self.real, os.stat(self.real).st_mode | stat.S_IXUSR)
        self.log = os.path.join(self.directory.name, "log")
        self.lock = os.path.join(self.directory.name, "machine.lock")

    def tearDown(self):
        self.directory.cleanup()

    def environment(self, **extra):
        environment = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "BAZEL_REAL": self.real,
            "FAKE_LOG": self.log,
            "WB_BAZEL_MACHINE_LOCK": self.lock,
        }
        environment.update(extra)
        return environment

    def start(self, *args, **extra):
        return subprocess.Popen(
            [WRAPPER, *args],
            env=self.environment(**extra),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def intervals(self):
        events = {}
        with open(self.log) as f:
            for line in f:
                kind, stamp, *command = line.split()
                events.setdefault(" ".join(command), {})[kind] = float(stamp)
        return events

    def test_two_builds_take_turns_and_the_second_says_why_it_waits(self):
        first = self.start("build", "//a", FAKE_SLEEP="1.5")
        time.sleep(0.3)
        second = self.start("--output_base=/x", "test", "//b", FAKE_SLEEP="0.2")
        _, first_err = first.communicate(timeout=30)
        _, second_err = second.communicate(timeout=30)
        self.assertEqual((first.returncode, second.returncode), (0, 0))
        runs = self.intervals()
        a, b = runs["build //a"], runs["--output_base=/x test //b"]
        self.assertGreaterEqual(b["start"], a["end"], "the two builds overlapped")
        self.assertIn("another Bazel build is running on this machine", second_err)
        self.assertNotIn("another Bazel build", first_err)

    def test_commands_that_do_not_compile_do_not_wait(self):
        build = self.start("build", "//a", FAKE_SLEEP="2")
        time.sleep(0.3)
        began = time.monotonic()
        for command in (
            ["info"],
            ["version"],
            ["shutdown"],
            ["query", "//..."],
            ["run", "//sim"],
        ):
            with self.subTest(command=command):
                result = subprocess.run(
                    [WRAPPER, *command],
                    env=self.environment(),
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(result.returncode, 0)
                self.assertNotIn("another Bazel build", result.stderr)
        self.assertLess(
            time.monotonic() - began,
            1.5,
            "a non-compiling command waited for the build",
        )
        build.communicate(timeout=30)

    def test_the_bazel_server_does_not_inherit_the_lock(self):
        # The Bazel client leaves a server running after it exits. If that server held the lock, every later build
        # would wait for hours.
        subprocess.run(
            [WRAPPER, "build", "//a"],
            env=self.environment(FAKE_DAEMON="1"),
            check=True,
            timeout=30,
        )
        began = time.monotonic()
        result = subprocess.run(
            [WRAPPER, "build", "//b"],
            env=self.environment(),
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertLess(time.monotonic() - began, 5.0)
        self.assertNotIn("another Bazel build", result.stderr)

    def test_the_exit_status_of_bazel_is_returned(self):
        for command in (["build", "//a"], ["info"]):
            with self.subTest(command=command):
                result = subprocess.run(
                    [WRAPPER, *command],
                    env=self.environment(FAKE_STATUS="3"),
                    capture_output=True,
                    timeout=30,
                )
                self.assertEqual(result.returncode, 3)

    def test_the_wrapper_never_reads_bash_env(self):
        # The dev container sets BASH_ENV to its shell setup, which (before it was fixed) called `bazel completion bash`.
        # A bash wrapper sourced that setup from inside every `bazel` call and recursed until the container ran out of
        # memory. Here BASH_ENV points at a script that would re-enter the wrapper; a depth counter bounds the damage if
        # the wrapper ever reads it again, and the counter must stay untouched.
        depth_file = os.path.join(self.directory.name, "depth")
        bash_env = os.path.join(self.directory.name, "bash_env.sh")
        with open(bash_env, "w") as f:
            f.write(
                'depth=$(cat "%s" 2>/dev/null || echo 0)\n'
                'echo $((depth + 1)) > "%s"\n'
                'if [ "$depth" -lt 3 ]; then "%s" info >/dev/null 2>&1; fi\n'
                % (depth_file, depth_file, WRAPPER)
            )
        for command in (["info"], ["build", "//a"]):
            with self.subTest(command=command):
                subprocess.run(
                    [WRAPPER, *command],
                    env=self.environment(BASH_ENV=bash_env),
                    check=True,
                    timeout=30,
                )
        self.assertFalse(os.path.exists(depth_file), "the wrapper sourced BASH_ENV")

    def test_it_refuses_to_run_without_bazelisk(self):
        environment = self.environment()
        del environment["BAZEL_REAL"]
        result = subprocess.run(
            [WRAPPER, "build", "//a"], env=environment, capture_output=True, text=True
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("BAZEL_REAL", result.stderr)


class ShellInitTest(unittest.TestCase):
    """.devcontainer/shell_init.sh is BASH_ENV in the dev container: every bash there sources it, scripts included."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        bin_directory = os.path.join(self.directory.name, "bin")
        os.mkdir(bin_directory)
        self.calls = os.path.join(self.directory.name, "bazel_calls")
        fake = os.path.join(bin_directory, "bazel")
        with open(fake, "w") as f:
            f.write('#!/bin/sh\necho "$*" >> "%s"\n' % self.calls)
        os.chmod(fake, os.stat(fake).st_mode | stat.S_IXUSR)
        self.path = bin_directory + ":" + os.environ.get("PATH", "/usr/bin:/bin")

    def tearDown(self):
        self.directory.cleanup()

    def bazel_calls(self, bash_flags, **extra):
        environment = {"PATH": self.path, "HOME": self.directory.name}
        environment.update(extra)
        subprocess.run(
            ["bash", *bash_flags, "-c", 'source "$0"', SHELL_INIT],
            env=environment,
            capture_output=True,
            timeout=60,
        )
        if not os.path.exists(self.calls):
            return []
        with open(self.calls) as f:
            calls = f.read().splitlines()
        os.remove(self.calls)
        return calls

    def test_a_script_does_not_call_bazel(self):
        self.assertEqual(self.bazel_calls([]), [])

    def test_an_interactive_shell_loads_bazel_completion(self):
        # Positive control: the fake is found, so the empty list above means the setup did not call it.
        self.assertEqual(self.bazel_calls(["-i"]), ["completion bash"])

    def test_nothing_calls_bazel_while_bazelisk_runs_the_wrapper(self):
        self.assertEqual(self.bazel_calls(["-i"], BAZEL_REAL="/bin/true"), [])


class ShellInitCheckoutTest(unittest.TestCase):
    """shell_init.sh sources the setup_env.sh of the checkout the shell starts in, so that every worktree of the
    repository is set up by its own script."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = directory.name

    def make_checkout(self, name):
        checkout = os.path.join(self.directory, name)
        os.makedirs(os.path.join(checkout, "some", "package"))
        open(os.path.join(checkout, "MODULE.bazel"), "w").close()
        with open(os.path.join(checkout, "setup_env.sh"), "w") as f:
            f.write('export WB_SOURCED_FROM="%s"\n' % name)
        return checkout

    def sourced_from(self, working_directory):
        result = subprocess.run(
            [
                "bash",
                "-c",
                'source "$0"; printf "%s|%s" "${WB_SOURCED_FROM-}" "$PWD"',
                SHELL_INIT,
            ],
            env={"PATH": "/usr/bin:/bin", "HOME": self.directory},
            cwd=working_directory,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_the_checkout_of_the_working_directory_is_set_up(self):
        first = self.make_checkout("first")
        second = self.make_checkout("second")
        self.assertEqual(self.sourced_from(first), "first|" + first)
        deep = os.path.join(second, "some", "package")
        # From anywhere inside it, and the shell is left where it started.
        self.assertEqual(self.sourced_from(deep), "second|" + deep)

    def test_a_directory_with_a_setup_script_but_no_module_is_not_a_checkout(self):
        stray = os.path.join(self.directory, "stray")
        os.makedirs(stray)
        with open(os.path.join(stray, "setup_env.sh"), "w") as f:
            f.write('export WB_SOURCED_FROM="stray"\n')
        self.assertNotIn("stray|", self.sourced_from(stray))


if __name__ == "__main__":
    unittest.main()
