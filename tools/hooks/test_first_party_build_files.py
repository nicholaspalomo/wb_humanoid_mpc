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

"""First-party BUILD files keep every header at its source path: no `strip_include_prefix`, no `include_prefix`.

A .cpp action of the clang-tidy aspect shows the diagnostics of the first-party headers it includes by their path
(tools/clang_tidy/clang_tidy.bzl:header_filter), among them a rename's declaration and a member a constructor in the
.cpp initializes. A header that a target re-roots with `strip_include_prefix` or `include_prefix` is reached under
bazel-out/.../_virtual_includes/ instead, which the filter does not match, so those diagnostics would be lost. List the
directory in `includes = [...]` instead, which keeps the source path.
"""

import os
import re
import unittest

from tools.hooks import check_test_support
from tools.hooks import lint_files

_ATTRIBUTE = re.compile(r"^\s*(strip_include_prefix|include_prefix)\s*=")
_BUILD_FILES = ("BUILD", "BUILD.bazel")

# The BUILD files that still re-root headers. Each is converted by the work package that owns its sources, together
# with the include lines that name the re-rooted paths, and leaves this set in that change.
PENDING = frozenset(
    {
        "humanoid_nmpc/humanoid_common_mpc_app/node/BUILD.bazel",
        "humanoid_nmpc/humanoid_common_mpc_app/robot/BUILD.bazel",
        "humanoid_nmpc/humanoid_common_mpc_app/visualization/BUILD.bazel",
    }
)


def prefix_attributes(source: str) -> list[tuple[int, str]]:
    """The `strip_include_prefix` and `include_prefix` attributes of a BUILD file, outside comments.

    Args:
      source: The BUILD file.

    Returns:
      (line number, attribute name) for each.
    """
    found = []
    for number, line in enumerate(source.splitlines(), start=1):
        match = _ATTRIBUTE.match(line.split("#", 1)[0])
        if match:
            found.append((number, match.group(1)))
    return found


def first_party_build_files(root: str) -> list[str]:
    """The first-party BUILD files of the checkout at `root` (lint_files.is_first_party), relative to it."""
    return [
        path
        for path in lint_files.repository_files(root)
        if os.path.basename(path) in _BUILD_FILES and lint_files.is_first_party(path)
    ]


class PrefixAttributesTest(unittest.TestCase):
    def test_both_attributes_are_found_and_comments_are_not(self):
        source = 'cc_library(\n    strip_include_prefix = "test",\n    include_prefix = "x",\n    # include_prefix = "y",\n)\n'
        self.assertEqual(
            prefix_attributes(source),
            [(2, "strip_include_prefix"), (3, "include_prefix")],
        )
        self.assertEqual(
            prefix_attributes('cc_library(\n    includes = ["test"],\n)\n'), []
        )


class FirstPartyBuildFilesTest(unittest.TestCase):
    def setUp(self):
        self.root = check_test_support.source_tree_root()
        self.files = first_party_build_files(self.root)

    def _read(self, path: str) -> str:
        with open(os.path.join(self.root, path), encoding="utf-8") as f:
            return f.read()

    def test_the_tree_is_found(self):
        self.assertIn("tools/hooks/BUILD.bazel", self.files)
        self.assertNotIn("lib/ocs2/BUILD.bazel", self.files)

    def test_no_first_party_target_reroots_its_headers(self):
        for path in self.files:
            if path in PENDING:
                continue
            with self.subTest(path=path):
                found = prefix_attributes(self._read(path))
                self.assertEqual(
                    found,
                    [],
                    f"{path}: use `includes = [...]` (this test's docstring says why)",
                )

    def test_each_pending_file_still_needs_its_entry(self):
        # A converted file leaves PENDING, so that it cannot regress.
        for path in sorted(PENDING):
            with self.subTest(path=path):
                self.assertIn(path, self.files)
                self.assertTrue(
                    prefix_attributes(self._read(path)),
                    f"{path} no longer re-roots headers",
                )


if __name__ == "__main__":
    unittest.main()
