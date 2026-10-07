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

"""Tests for include_style.py: angle brackets only for system headers, and Abseil with quotes, as Bazel exposes it.

Also that no system copy of a library Bazel builds from the BCR is installed, which is what made angle Abseil includes
compile in the dev container and fail in CI.
"""

import os
import re
import unittest

from tools.hooks import check_test_support
from tools.hooks import include_style


def _repository_file(relative_path):
    """A repository file, from the Bazel runfiles when run by Bazel and from the source tree otherwise."""
    roots = []
    if "TEST_SRCDIR" in os.environ:
        roots.append(os.path.join(os.environ["TEST_SRCDIR"], "_main"))
    roots.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    for root in roots:
        candidate = os.path.join(root, relative_path)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(relative_path)


# The Ubuntu development packages of protobuf, ZeroMQ (libzmq and cppzmq) and googletest.
_BCR_BUILT_DEV_PACKAGES = re.compile(
    r"^(libprotobuf\S*-dev|libprotoc\S*-dev|protobuf-compiler\S*|libzmq\S*-dev|cppzmq-dev|libgtest-dev|libgmock-dev"
    r"|google-mock|googletest)$"
)

# The arguments of an `apt-get install`, once continued lines are joined.
_APT_INSTALL = re.compile(r"apt-get install([^\n]*)")


def _installed_packages(relative_path: str) -> list[str]:
    """The package names a file installs: every line of dependencies.txt, or the arguments of each `apt-get install`.

    Comment lines are dropped first, so that a comment may name a package it explains is NOT installed.

    Args:
        relative_path: dependencies.txt, or a Dockerfile or script that runs `apt-get install`.

    Returns:
        The package names, in the order of the file.
    """
    with open(_repository_file(relative_path), encoding="utf-8") as f:
        lines = [
            line for line in f.read().splitlines() if not line.lstrip().startswith("#")
        ]
    if relative_path.endswith(".txt"):
        return [line.strip() for line in lines if line.strip()]
    packages = []
    for match in _APT_INSTALL.finditer("\n".join(lines).replace("\\\n", " ")):
        for word in match.group(1).split():
            if word in ("&&", "||", ";"):
                break
            if not word.startswith("-") and not word.startswith("$"):
                packages.append(word.split("=")[0])
    return packages


def _findings(source: str) -> list[tuple[int, str]]:
    return [
        (f.line, f.message) for f in include_style._style_findings(source, "src/a.cpp")
    ]


class AbseilIncludeTest(unittest.TestCase):
    """An angle Abseil include finds a system Abseil or none, so its finding says why, not only what to write."""

    def test_an_angle_absl_include_is_flagged_with_its_line_and_the_reason(self):
        source = "#include <vector>\n#include <absl/log/log.h>\n  #  include <absl/status/status.h>\n"
        found = _findings(source)
        self.assertEqual([line for line, _ in found], [2, 3])
        self.assertIn('write `#include "absl/log/log.h"`', found[0][1])
        self.assertIn('write `#include "absl/status/status.h"`', found[1][1])
        for _, message in found:
            self.assertIn("@abseil-cpp", message)

    def test_quoted_and_commented_absl_includes_are_fine(self):
        self.assertEqual(
            _findings('#include "absl/log/log.h"\n// #include <absl/x.h>\n'), []
        )

    def test_only_absl_itself_gets_the_abseil_reason(self):
        found = _findings("#include <abslish/x.h>\n#include <Eigen/Dense>\n")
        self.assertEqual(len(found), 2)
        for _, message in found:
            self.assertNotIn("@abseil-cpp", message)
            self.assertIn("system headers only", message)


class InstalledLibrariesTest(unittest.TestCase):
    """No second copy of a library Bazel builds from the BCR is installed where the compiler would find it."""

    def test_no_system_abseil_is_installed(self):
        # A second Abseil on the compiler's default include path is what let the angle includes compile in the dev
        # container and fail in CI, and it still lets a quoted include whose Bazel dependency is missing compile against
        # the wrong release. Bazel's @abseil-cpp is the only one the build may see: neither the dev image builds one
        # into /usr/local nor the dependency list installs Ubuntu's into /usr/include.
        with open(_repository_file("docker/Dockerfile"), encoding="utf-8") as f:
            dockerfile = f.read()
        self.assertNotIn("abseil-cpp.git", dockerfile)
        self.assertNotIn("libabsl_base", dockerfile)
        with open(_repository_file("dependencies.txt"), encoding="utf-8") as f:
            packages = [line.split("#")[0].strip() for line in f]
        self.assertFalse(
            [p for p in packages if p.startswith("libabsl")],
            "dependencies.txt installs a system Abseil",
        )

    def test_no_system_protobuf_zeromq_or_googletest_is_installed(self):
        # The same hazard for the other libraries Bazel builds from the BCR: protobuf, libzmq and cppzmq, and
        # googletest. Their headers on the default include path would let a target that misses its Bazel dependency
        # compile against another release, and a system protoc would generate code for another runtime. So neither the
        # image, nor the dependency list, nor the robotpkg script names one of their development packages. (robotpkg's
        # fatrop still pulls in libgtest-dev; Bazel's -isystem googletest include wins over /usr/include for every
        # target that depends on it.)
        installed = _installed_packages("dependencies.txt")
        for path in ("docker/Dockerfile", "docker/install_robotpkg.sh"):
            installed += _installed_packages(path)
        self.assertIn("libeigen3-dev", installed, "the package lists were not read")
        self.assertIn(
            "xvfb", installed, "the Dockerfile's apt-get installs were not read"
        )
        self.assertEqual(
            [p for p in installed if _BCR_BUILT_DEV_PACKAGES.match(p)],
            [],
            "a system package of a library Bazel builds from the BCR",
        )

    def test_the_bcr_package_pattern_catches_the_ubuntu_names(self):
        for package in (
            "libprotobuf-dev",
            "protobuf-compiler",
            "libprotoc-dev",
            "libzmq3-dev",
            "cppzmq-dev",
            "libgtest-dev",
            "libgmock-dev",
            "google-mock",
            "googletest",
        ):
            self.assertTrue(_BCR_BUILT_DEV_PACKAGES.match(package), package)
        for package in ("libeigen3-dev", "liburdfdom-dev", "python3-pip"):
            self.assertFalse(_BCR_BUILT_DEV_PACKAGES.match(package), package)


class GeneralIncludeStyleTest(unittest.TestCase):
    """The include-style check: angle brackets only for system headers, and full paths."""

    def findings(self, source):
        return [f.message for f in include_style._style_findings(source, "src/a.cpp")]

    def test_system_headers_keep_their_brackets(self):
        source = "#include <vector>\n#include <cmath>\n#include <unistd.h>\n#include <sys/mman.h>\n#include <cxxabi.h>\n"
        self.assertEqual(self.findings(source), [])

    def test_every_other_angle_include_is_quoted(self):
        source = "#include <Eigen/Dense>\n#include <pinocchio/fwd.hpp>\n#include <zmq.hpp>\n#include <absl/log/log.h>\n"
        self.assertEqual(len(self.findings(source)), 4)
        fixed = include_style.fix_include_style(source, "src/a.cpp")
        self.assertEqual(
            fixed,
            '#include "Eigen/Dense"\n#include "pinocchio/fwd.hpp"\n#include "zmq.hpp"\n#include "absl/log/log.h"\n',
        )
        self.assertEqual(self.findings(fixed), [])
        self.assertEqual(include_style.fix_include_style(fixed, "src/a.cpp"), fixed)

    def test_a_project_header_needs_its_directory(self):
        self.assertEqual(len(self.findings('#include "VisualizationTestRobot.h"\n')), 1)
        self.assertEqual(
            self.findings('#include "humanoid_common_mpc/common/Types.h"\n'), []
        )

    def test_flat_third_party_headers_need_no_directory(self):
        # cppzmq, libzmq, HPIPM and BLASFEO install their headers without a directory of their own.
        source = '#include "zmq.hpp"\n#include "hpipm_d_ocp_qp_ipm.h"\n#include "blasfeo_d_aux.h"\n'
        self.assertEqual(self.findings(source), [])
        self.assertEqual(len(self.findings('#include "hpipm.h"\n')), 1)

    def test_comments_are_not_includes(self):
        self.assertEqual(
            self.findings("// #include <Eigen/Dense>\n/* #include <x/y.h> */\n"), []
        )

    def test_the_registry_runs_it(self):
        check_test_support.assert_check_behaves(
            self,
            "include-style",
            "#include <Eigen/Dense>\n",
            "src/a.cpp",
            clean='#include "Eigen/Dense"\n',
        )


if __name__ == "__main__":
    unittest.main()
