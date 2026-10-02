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

"""Pins the guards against a build exhausting the machine's memory, and the local CI emulator's runner resources.

.bazelrc bounds Bazel's parallelism by RAM rather than by cores, and the dev container is capped with no swap beyond
the cap, whose value tools/resource_limits/set_container_memory_limit.sh writes to .env. Each property is asserted
rather than a number, so that tuning the ratio is not a test failure while going back to a fixed job count is.
"""

import os
import re
import subprocess
import tempfile
import unittest


def _runfile(relative_path):
    """A data file of this test, found in the runfiles when run by Bazel and in the source tree otherwise."""
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


SCRIPT = _runfile("tools/resource_limits/set_container_memory_limit.sh")


def _host_ram_factor(bazelrc, command, flag):
    """The factor in `<command> --<flag>=HOST_RAM*<factor>`, or None when the flag is not written that way."""
    # LINT.IfChange(bazel_memory_bound)
    match = re.search(
        r"^%s --%s=HOST_RAM\*([0-9.]+)\s*$" % (command, flag), bazelrc, re.MULTILINE
    )
    # LINT.ThenChange(//.bazelrc:bazel_memory_bound)
    return float(match.group(1)) if match else None


class BazelParallelismTest(unittest.TestCase):
    def setUp(self):
        with open(_runfile(".bazelrc")) as f:
            self.bazelrc = f.read()

    def test_jobs_are_bounded_by_memory_not_a_fixed_count(self):
        # A fixed count is what overcommitted the workstation; any `--jobs=<integer>` in .bazelrc brings it back.
        self.assertIsNone(
            re.search(r"^\w+ --jobs=\d+\s*$", self.bazelrc, re.MULTILINE),
            "a fixed --jobs count in .bazelrc",
        )
        factor = _host_ram_factor(self.bazelrc, "build", "jobs")
        self.assertIsNotNone(factor, "build --jobs is not bounded by HOST_RAM")
        # HOST_RAM is in MB. At least 3 GB per action: a Pinocchio / CppAD translation unit peaks at 2-3 GB.
        self.assertLessEqual(factor, 1.0 / 3072.0)
        # And a 16 GB CI runner still gets more than one action at a time.
        self.assertGreaterEqual(16384 * factor, 2.0)

    def test_tests_are_bounded_by_memory_and_at_most_the_build_share(self):
        build = _host_ram_factor(self.bazelrc, "build", "jobs")
        test = _host_ram_factor(self.bazelrc, "test", "local_test_jobs")
        self.assertIsNotNone(test, "test --local_test_jobs is not bounded by HOST_RAM")
        self.assertLessEqual(test, build)
        self.assertGreaterEqual(16384 * test, 1.0)

    def test_a_local_override_file_is_imported(self):
        self.assertIn("try-import %workspace%/user.bazelrc", self.bazelrc)

    def test_no_command_or_configuration_sets_a_fixed_count(self):
        # `build:<config> --jobs=8` brings a fixed count back as surely as `build --jobs=8`; the RAM-bounded
        # `build --jobs=HOST_RAM*...` is not a count.
        self.assertIsNone(
            re.search(
                r"^[\w:-]+ --(jobs|local_test_jobs)=\d+", self.bazelrc, re.MULTILINE
            ),
            "a fixed --jobs or --local_test_jobs count in .bazelrc",
        )

    def test_no_configuration_overrides_the_memory_bound(self):
        # The `<command>:<config>` lines, the clang-tidy aspect's among them (tools/clang_tidy), inherit the bound.
        configurations = re.findall(r"^\w+:[\w-]+ .*$", self.bazelrc, re.MULTILINE)
        self.assertTrue(
            any(line.startswith("build:clang-tidy ") for line in configurations)
        )
        for line in configurations:
            self.assertNotRegex(line, r"--(jobs|local_test_jobs)\b")


def _apt_packages(script):
    """The packages of the first `apt-get install -y --no-install-recommends` whose list continues over lines."""
    match = re.search(
        r"apt-get install -y --no-install-recommends \\\n((?:.*\\\n)*.*)", script
    )
    return match.group(1).replace("\\", " ").split() if match else None


class LocalCiEmulatorTest(unittest.TestCase):
    """tools/ci_local.sh (`make ci-local`) must fail where .github/workflows/build_test.yml fails.

    It passed on a workstation for weeks while every run on GitHub's runner failed, so it runs the workflow's steps at
    the runner's resources.
    """

    def setUp(self):
        with open(_runfile("tools/ci_local.sh")) as f:
            self.emulator = f.read()
        with open(_runfile(".github/workflows/build_test.yml")) as f:
            self.workflow = f.read()
        with open(_runfile(".bazelrc")) as f:
            self.bazelrc = f.read()

    def test_it_installs_what_the_workflow_installs(self):
        workflow_packages = _apt_packages(self.workflow)
        self.assertIsNotNone(workflow_packages)
        self.assertEqual(_apt_packages(self.emulator), workflow_packages)

    def test_it_runs_the_workflows_container_and_steps(self):
        container = re.search(r"^    container: (\S+)\s*$", self.workflow, re.MULTILINE)
        self.assertIsNotNone(container, "the workflow names no container")
        self.assertIn('CI_IMAGE="%s"' % container.group(1), self.emulator)
        self.assertIn('  "${CI_IMAGE}" \\', self.emulator)
        for step in (
            "sh docker/install_robotpkg.sh",
            "grep -v '^\\s*#' dependencies.txt | envsubst | xargs apt-get install",
            "make build-all",
            "make test-all",
            "sh docker/install_llvm_tools.sh tidy",
            "make lint-tidy",
        ):
            self.assertIn(step, self.workflow)
            self.assertIn(step, self.emulator)

    def test_it_sets_the_environment_the_workflow_sets(self):
        # The workflow writes `NAME=value` to $GITHUB_ENV and directories to $GITHUB_PATH; the emulator exports the same.
        workflow = dict(
            re.findall(r'echo "(\w+)=(\S+)" >> "\$GITHUB_ENV"', self.workflow)
        )
        workflow_path = re.findall(r'echo "(\S+)" >> "\$GITHUB_PATH"', self.workflow)
        emulator = dict(re.findall(r"^\s*export (\w+)=(\S+)$", self.emulator, re.M))
        self.assertTrue(workflow, "the workflow sets no environment")
        self.assertEqual(
            emulator.pop("PATH", None), ":".join(workflow_path + ["\\${PATH}"])
        )
        self.assertEqual(emulator, workflow)

    def test_it_has_the_resources_of_the_runner(self):
        self.assertIn("runs-on: ubuntu-latest", self.workflow)
        # ubuntu-latest: 4 vCPUs, 16 GB of RAM and a 4 GB swap file. --memory-swap is memory plus swap.
        self.assertIn(
            "RUNNER_RESOURCES=(--cpus=4 --memory=16g --memory-swap=20g)",
            self.emulator,
        )
        self.assertIn('docker run --rm "${RUNNER_RESOURCES[@]}"', self.emulator)

    def test_its_job_counts_are_no_more_than_bazelrc_gives_the_runner(self):
        jobs = re.search(r"build --jobs=(\d+)", self.emulator)
        test_jobs = re.search(r"test --local_test_jobs=(\d+)", self.emulator)
        self.assertIsNotNone(jobs)
        self.assertIsNotNone(test_jobs)
        self.assertLessEqual(
            int(jobs.group(1)), 16384 * _host_ram_factor(self.bazelrc, "build", "jobs")
        )
        self.assertLessEqual(
            int(test_jobs.group(1)),
            16384 * _host_ram_factor(self.bazelrc, "test", "local_test_jobs"),
        )
        # Written where Bazel reads it after the workspace's .bazelrc, never into the checkout's own user.bazelrc.
        self.assertIn("> ~/.bazelrc", self.emulator)
        self.assertNotRegex(self.emulator, r">\s*\S*user\.bazelrc")


class ContainerMemoryCapTest(unittest.TestCase):
    def test_compose_caps_memory_and_forbids_swap_beyond_the_cap(self):
        with open(_runfile("docker-compose.yaml")) as f:
            compose = f.read()
        # LINT.IfChange(container_memory_limit)
        self.assertIn("mem_limit: ${WB_CONTAINER_MEMORY_LIMIT:-0}", compose)
        # memswap_limit is memory PLUS swap: equal to mem_limit means no swap at all, which is what makes an overcommit
        # end in seconds rather than in a machine thrashing its disk.
        self.assertIn("memswap_limit: ${WB_CONTAINER_MEMORY_LIMIT:-0}", compose)
        # LINT.ThenChange(//docker-compose.yaml:container_memory_limit)

    def test_the_dev_container_computes_the_cap_before_it_starts(self):
        with open(_runfile(".devcontainer/devcontainer.json")) as f:
            devcontainer = f.read()
        self.assertRegex(
            devcontainer,
            r'"initializeCommand":\s*"bash \$\{localWorkspaceFolder\}/tools/resource_limits/set_container_memory_limit.sh"',
        )


class SetContainerMemoryLimitScriptTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.env_file = os.path.join(self.dir.name, ".env")
        self.meminfo = os.path.join(self.dir.name, "meminfo")

    def tearDown(self):
        self.dir.cleanup()

    def _run(self, total_kb=None, **env):
        if total_kb is not None:
            with open(self.meminfo, "w") as f:
                f.write("MemTotal:       %d kB\nMemFree:        1000 kB\n" % total_kb)
        environment = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "WB_ENV_FILE": self.env_file,
            "WB_MEMINFO": self.meminfo,
        }
        environment.update(env)
        return subprocess.run(
            ["bash", SCRIPT],
            env=environment,
            capture_output=True,
            text=True,
            check=True,
        )

    def _env_lines(self):
        with open(self.env_file) as f:
            return f.read().splitlines()

    def test_the_cap_is_a_share_of_the_hosts_ram(self):
        self._run(total_kb=32 * 1024 * 1024)
        # 85% of 32 GiB, in MiB.
        self.assertIn("WB_CONTAINER_MEMORY_LIMIT=27852m", self._env_lines())

    def test_the_share_can_be_changed(self):
        self._run(total_kb=32 * 1024 * 1024, WB_CONTAINER_MEMORY_PERCENT="50")
        self.assertIn("WB_CONTAINER_MEMORY_LIMIT=16384m", self._env_lines())

    def test_an_explicit_limit_wins(self):
        self._run(total_kb=32 * 1024 * 1024, WB_CONTAINER_MEMORY_LIMIT="20g")
        self.assertIn("WB_CONTAINER_MEMORY_LIMIT=20g", self._env_lines())

    def test_other_variables_are_kept_and_the_line_is_replaced_not_repeated(self):
        with open(self.env_file, "w") as f:
            f.write("GIT_USER_NAME=someone\nWB_CONTAINER_MEMORY_LIMIT=1m\n")
        self._run(total_kb=16 * 1024 * 1024)
        self._run(total_kb=16 * 1024 * 1024)
        lines = self._env_lines()
        self.assertIn("GIT_USER_NAME=someone", lines)
        self.assertEqual(
            [line for line in lines if line.startswith("WB_CONTAINER_MEMORY_LIMIT=")],
            ["WB_CONTAINER_MEMORY_LIMIT=13926m"],
        )

    def test_a_host_without_proc_meminfo_is_left_unlimited(self):
        # macOS and Windows: Docker Desktop's VM bounds the container already. Nothing is written, and it is not an error.
        result = self._run()
        self.assertFalse(os.path.exists(self.env_file))
        self.assertIn("unlimited", result.stderr)


if __name__ == "__main__":
    unittest.main()
