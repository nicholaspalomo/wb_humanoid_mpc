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

"""The lint tools are the same everywhere: the dev image, CI's format job and Bazel all install one hashed lock.

A different cpplint, pylint, mypy or isort finds different things, so a commit that is clean in the container would
fail in CI, or the other way round. tools/hooks/lint_requirements_lock.txt pins every package with its hashes, and this
test keeps every installer on it.
"""

import re
import unittest

from tools.hooks import check_test_support
from tools.hooks import lint_code

LOCK = "tools/hooks/lint_requirements_lock.txt"
_PIN = re.compile(r"^([A-Za-z0-9_.\-]+)==(\S+)")


def _lock_pins() -> dict[str, str]:
    pins = {}
    for line in check_test_support.read_repository_file(LOCK).splitlines():
        match = _PIN.match(line)
        if match:
            pins[match.group(1).lower()] = match.group(2)
    return pins


class LockTest(unittest.TestCase):
    def test_every_requirement_is_pinned_with_hashes(self):
        lock = check_test_support.read_repository_file(LOCK)
        requirements = [
            line
            for line in lock.splitlines()
            if line and not line.startswith((" ", "#"))
        ]
        self.assertTrue(requirements)
        for line in requirements:
            with self.subTest(requirement=line):
                self.assertRegex(line, r"^[A-Za-z0-9_.\-]+==[^\s]+ \\$")
        self.assertGreaterEqual(lock.count("--hash=sha256:"), len(requirements))

    def test_the_lock_holds_the_requested_versions(self):
        pins = _lock_pins()
        for line in check_test_support.read_repository_file(
            "tools/hooks/lint_requirements.txt"
        ).splitlines():
            match = _PIN.match(line)
            if match:
                with self.subTest(package=match.group(1)):
                    self.assertEqual(pins.get(match.group(1).lower()), match.group(2))

    def test_the_linter_runs_the_tools_of_the_lock(self):
        pins = _lock_pins()
        for tool in ("cpplint", "pylint", "mypy", "isort"):
            with self.subTest(tool=tool):
                self.assertIn(tool, pins)
                self.assertIn(tool, lint_code._VERSION_PATTERNS)


class InstallersTest(unittest.TestCase):
    def test_the_dev_image_installs_the_lock(self):
        dockerfile = check_test_support.read_repository_file("docker/Dockerfile")
        self.assertIn(f"COPY {LOCK} ", dockerfile)
        self.assertRegex(
            dockerfile,
            r"pip install [^\n]*--require-hashes -r /tmp/lint_requirements_lock\.txt",
        )
        for tool in ("cpplint", "pylint", "mypy", "isort"):
            self.assertIn(f"/opt/wb-lint/bin/{tool}", dockerfile)

    def test_ci_installs_the_lock(self):
        workflow = check_test_support.read_repository_file(
            ".github/workflows/format_test.yml"
        )
        self.assertIn(f"pip install --require-hashes -r {LOCK}", workflow)
        # Inside the pinned-versions block, so that the formatter test watches it.
        block = workflow[
            workflow.index("LINT.IfChange(formatter_versions)") : workflow.index(
                "LINT.ThenChange"
            )
        ]
        self.assertIn(LOCK, block)

    def test_black_is_the_same_everywhere(self):
        workflow = check_test_support.read_repository_file(
            ".github/workflows/format_test.yml"
        )
        ci = re.search(r"pip install black==(\S+)", workflow)
        assert ci is not None
        self.assertEqual(_lock_pins().get("black"), ci.group(1))

    def test_bazel_reads_the_lock(self):
        module = check_test_support.read_repository_file("MODULE.bazel")
        hub = re.search(
            r'pip\.parse\(\s*hub_name = "lint_deps",(.*?)\)', module, re.DOTALL
        )
        assert hub is not None, "MODULE.bazel has no lint_deps hub"
        self.assertIn(
            'requirements_lock = "//tools/hooks:lint_requirements_lock.txt"',
            hub.group(1),
        )
        self.assertRegex(module, r'use_repo\(pip,[^)]*"lint_deps"')


def _version(path: str, pattern: str) -> str:
    match = re.search(
        pattern, check_test_support.read_repository_file(path), re.MULTILINE
    )
    assert match is not None, f"{path} does not match {pattern}"
    return match.group(1)


class LlvmToolsTest(unittest.TestCase):
    """docker/install_llvm_tools.sh installs clang-format and clang-tidy for the image and for CI's build job."""

    def test_clang_tidy_is_one_major_version_everywhere(self):
        image = _version("docker/Dockerfile", r"^ARG CLANG_TIDY_VERSION=(\d+)\s*$")
        script = _version(
            "docker/install_llvm_tools.sh",
            r'^CLANG_TIDY_VERSION="\$\{CLANG_TIDY_VERSION:-(\d+)\}"$',
        )
        # The aspect's wrapper refuses any other clang-tidy, and a new pin re-runs every action.
        wrapper = _version(
            "tools/clang_tidy/run_clang_tidy.sh", r"^CLANG_TIDY_VERSION=(\d+)$"
        )
        self.assertEqual({image, script, wrapper}, {image})

    def test_clang_tidy_is_one_exact_build_everywhere(self):
        # A point release can change what a check reports, and the wrapper's pin is part of every action's key.
        image = _version(
            "docker/Dockerfile", r"^ARG CLANG_TIDY_PACKAGE_VERSION=(\S+)\s*$"
        )
        script = _version(
            "docker/install_llvm_tools.sh",
            r'^CLANG_TIDY_PACKAGE_VERSION="\$\{CLANG_TIDY_PACKAGE_VERSION:-(\S+)\}"$',
        )
        self.assertEqual(script, image)
        upstream = re.match(r"^\d+:(\d+\.\d+\.\d+)~", image)
        assert upstream is not None, f"{image} is not an apt.llvm.org package version"
        wrapper = _version(
            "tools/clang_tidy/run_clang_tidy.sh", r"^CLANG_TIDY_FULL_VERSION=(\S+)$"
        )
        self.assertEqual(wrapper, upstream.group(1))
        major = _version("docker/Dockerfile", r"^ARG CLANG_TIDY_VERSION=(\d+)\s*$")
        self.assertTrue(
            wrapper.startswith(major + "."), f"{wrapper} is not clang-tidy {major}"
        )
        installer = check_test_support.read_repository_file(
            "docker/install_llvm_tools.sh"
        )
        self.assertIn(
            '"clang-tidy-${CLANG_TIDY_VERSION}=${CLANG_TIDY_PACKAGE_VERSION}"',
            installer,
        )

    def test_clang_format_is_ubuntus_everywhere(self):
        image = _version("docker/Dockerfile", r"^ARG CLANG_FORMAT_VERSION=(\d+)\s*$")
        script = _version(
            "docker/install_llvm_tools.sh",
            r'^CLANG_FORMAT_VERSION="\$\{CLANG_FORMAT_VERSION:-(\d+)\}"$',
        )
        self.assertEqual(script, image)
        installer = check_test_support.read_repository_file(
            "docker/install_llvm_tools.sh"
        )
        # apt.llvm.org is added for clang-tidy's major version only, so clang-format comes from Ubuntu's archive, as
        # in CI's format job (test_ci_formatter_versions.py).
        self.assertEqual(
            re.findall(r"llvm-toolchain-\$\{codename\}-\$\{(\w+)\}", installer),
            ["CLANG_TIDY_VERSION"],
        )
        self.assertIn(
            'apt-get install -y --no-install-recommends "clang-format-${CLANG_FORMAT_VERSION}"',
            installer,
        )

    def test_the_image_and_ci_run_the_script(self):
        dockerfile = check_test_support.read_repository_file("docker/Dockerfile")
        self.assertIn(
            "COPY docker/install_llvm_tools.sh /tmp/install_llvm_tools.sh", dockerfile
        )
        self.assertIn("sh /tmp/install_llvm_tools.sh all", dockerfile)
        self.assertNotIn("llvm-toolchain", dockerfile)
        workflow = check_test_support.read_repository_file(
            ".github/workflows/build_test.yml"
        )
        self.assertIn("sh docker/install_llvm_tools.sh tidy", workflow)
        self.assertIn("make lint-tidy", workflow)


if __name__ == "__main__":
    unittest.main()
