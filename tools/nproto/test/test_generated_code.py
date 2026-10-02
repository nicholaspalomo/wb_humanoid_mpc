"""The code nproto generates follows the repository's C++ rules (tools/nproto/README.md).

Over every file generated for nproto's test protos and for humanoid_mpc_msgs:
- it is clang-format clean under the repository's .clang-format;
- it never uses `auto`;
- a struct header (<file>.nproto.h) includes standard headers, Eigen and other struct headers only: no protobuf, no
  Abseil.
"""

import glob
import os
import re
import shutil
import subprocess
import unittest
from typing import List

RUNFILES = os.path.join(os.environ.get("TEST_SRCDIR", ""), "_main")
CLANG_FORMAT_CONFIG = os.path.join(RUNFILES, ".clang-format")
STRUCT_HEADER = ".nproto.h"
ALLOWED_STRUCT_HEADER_INCLUDES = re.compile(
    r'#include (<[a-z_]+>|<Eigen/Core>|"[^"]+\.nproto\.h")'
)


def generated_files() -> List[str]:
    files = []
    for suffix in (".nproto.h", ".nproto.pb.h", ".nproto.pb.cc"):
        files += glob.glob(os.path.join(RUNFILES, "**", "*" + suffix), recursive=True)
    return sorted(set(files))


def code_lines(path: str) -> List[str]:
    """The lines of `path` without their // comments."""
    with open(path, encoding="utf-8") as file:
        return [line.split("//")[0] for line in file.read().splitlines()]


class GeneratedCodeTest(unittest.TestCase):
    def test_there_is_code_of_both_libraries(self):
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
            )
            if completed.returncode != 0:
                unformatted.append(
                    f"{os.path.relpath(path, RUNFILES)}:\n{completed.stderr}"
                )
        self.assertEqual(unformatted, [], "\n".join(unformatted))

    def test_no_auto(self):
        uses = []
        for path in generated_files():
            for number, line in enumerate(code_lines(path), start=1):
                if re.search(r"\bauto\b", line):
                    uses.append(
                        f"{os.path.relpath(path, RUNFILES)}:{number}: {line.strip()}"
                    )
        self.assertEqual(uses, [])

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
