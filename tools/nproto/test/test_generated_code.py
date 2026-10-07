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

"""The code nproto generates follows the repository's C++ rules (tools/nproto/README.md).

Over every file generated for nproto's test protos, for humanoid_mpc_msgs and for humanoid_mpc_config:
- it is clang-format clean under the repository's .clang-format;
- every lint check of the registry (tools/hooks/checks.py) that reads C++ finds nothing in it, as in the hand-written
  code: `no-auto`, `pointer-nullability` (every raw pointer, the output parameters of ToProto() / FromProto(), carries
  `absl_nonnull` or `absl_nullable`, and the file includes absl/base/nullability.h), `std-integer-type`, the Google
  style and Tips of the Week checks, ...;
- a struct header (<file>.nproto.h) includes standard headers, Eigen and other struct headers only: no protobuf, no
  Abseil.

`make lint` cannot check the generated code, which lives in bazel-out; this test runs the same checks over it.
"""

import glob
import os
import re
import shutil
import subprocess
import unittest

from tools.hooks import checks

RUNFILES = os.path.join(os.environ.get("TEST_SRCDIR", ""), "_main")
CLANG_FORMAT_CONFIG = os.path.join(RUNFILES, ".clang-format")
STRUCT_HEADER = ".nproto.h"
ALLOWED_STRUCT_HEADER_INCLUDES = re.compile(
    r'#include (<[a-z_]+>|"Eigen/Core"|"[^"]+\.nproto\.h")'
)
# The parameters of a ToProto() / FromProto() declaration or definition, which starts a line of the generated code.
SIGNATURE = re.compile(
    r"^(?:void ToProto|absl::Status FromProto)\(([^)]*)\)", re.MULTILINE
)


def generated_files() -> list[str]:
    files = []
    for suffix in (".nproto.h", ".nproto.pb.h", ".nproto.pb.cc"):
        files += glob.glob(os.path.join(RUNFILES, "**", "*" + suffix), recursive=True)
    return sorted(set(files))


def code_lines(path: str) -> list[str]:
    """The lines of `path` without their // comments."""
    with open(path, encoding="utf-8") as file:
        return [line.split("//")[0] for line in file.read().splitlines()]


class GeneratedCodeTest(unittest.TestCase):
    def test_there_is_code_of_every_library(self):
        names = [os.path.relpath(path, RUNFILES) for path in generated_files()]
        self.assertTrue(
            any("humanoid_mpc_msgs/mpc_policy.nproto.h" in name for name in names),
            names,
        )
        self.assertTrue(
            any("tools/nproto/test/maps.nproto.pb.cc" in name for name in names), names
        )
        self.assertTrue(
            any("nproto_test/other/point.nproto.h" in name for name in names), names
        )
        self.assertTrue(
            any("humanoid_mpc_config/task_file.nproto.h" in name for name in names),
            names,
        )

    @unittest.skipIf(
        shutil.which("clang-format") is None, "clang-format is not installed"
    )
    def test_clang_format_leaves_it_unchanged(self):
        unformatted = []
        for path in generated_files():
            completed = subprocess.run(
                [
                    "clang-format",
                    f"--style=file:{CLANG_FORMAT_CONFIG}",
                    "--dry-run",
                    "--Werror",
                    path,
                ],
                capture_output=True,
                text=True,
                check=False,  # The return code is the finding.
            )
            if completed.returncode != 0:
                unformatted.append(
                    f"{os.path.relpath(path, RUNFILES)}:\n{completed.stderr}"
                )
        self.assertEqual(unformatted, [], "\n".join(unformatted))

    def test_the_lint_checks_of_the_cpp_rules_find_nothing(self):
        findings = []
        for path in generated_files():
            with open(path, encoding="utf-8") as file:
                source = file.read()
            relative = os.path.relpath(path, RUNFILES)
            findings += [
                str(found)
                for found in checks.run_file(source, relative, checks.names(), RUNFILES)
            ]
        self.assertEqual(findings, [], "\n".join(findings))

    def test_the_cpp_checks_read_the_generated_files(self):
        # run_file() skips a check that does not apply to a path, so a path the registry did not take for first-party
        # C++ would pass the test above without being checked.
        expected = {
            "no-auto",
            "pointer-nullability",
            "std-integer-type",
            "totw-enum-switch-default",
        }
        for path in generated_files():
            relative = os.path.relpath(path, RUNFILES)
            applicable = {
                check.name for check in checks.REGISTRY if check.applies_to(relative)
            }
            self.assertLessEqual(expected, applicable, relative)

    def test_the_output_parameters_are_nonnull(self):
        # The lint checks pass a file without pointers too; the conversions have one each, and it is nonnull.
        outputs = []
        for path in generated_files():
            if path.endswith(STRUCT_HEADER):
                continue
            with open(path, encoding="utf-8") as file:
                outputs += [
                    parameters.split(",")[-1].strip()
                    for parameters in SIGNATURE.findall(file.read())
                ]
        self.assertTrue(outputs)
        self.assertEqual(
            [output for output in outputs if "* absl_nonnull " not in output], []
        )

    def test_struct_headers_include_no_protobuf_and_no_abseil(self):
        includes = []
        for path in generated_files():
            if not path.endswith(STRUCT_HEADER):
                continue
            for line in code_lines(path):
                if line.startswith(
                    "#include"
                ) and not ALLOWED_STRUCT_HEADER_INCLUDES.fullmatch(line.strip()):
                    includes.append(
                        f"{os.path.relpath(path, RUNFILES)}: {line.strip()}"
                    )
        self.assertEqual(includes, [])


if __name__ == "__main__":
    unittest.main()
