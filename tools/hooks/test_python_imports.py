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

"""Tests for python_imports.py: import modules, absolutely, without aliases or sys.path changes."""

import os
import tempfile
import unittest
from unittest import mock

from tools.hooks import check_test_support
from tools.hooks import python_imports


class FirstPartyTreeTest(unittest.TestCase):
    """Runs against a small repository: a Bazel import root `pkg` and a package below the repository root."""

    def setUp(self):
        # pylint: disable-next=consider-using-with  # tearDown() deletes it.
        self.directory = tempfile.TemporaryDirectory()
        root = self.directory.name
        for path, text in {
            "pkg/BUILD.bazel": 'py_library(name = "a", imports = ["."])\n',
            "pkg/remote_control/__init__.py": "",
            "pkg/remote_control/operator_bus.py": "class TopicPublisher: pass\n",
            "tools/hooks/checks.py": "",
        }.items():
            os.makedirs(os.path.dirname(os.path.join(root, path)), exist_ok=True)
            with open(os.path.join(root, path), "w", encoding="utf-8") as f:
                f.write(text)
        python_imports._roots.cache_clear()
        self.patch = mock.patch.object(python_imports, "ROOT", root)
        self.patch.start()
        self.git = mock.patch.object(
            python_imports.lint_files.subprocess, "run", side_effect=OSError("no git")
        )
        self.git.start()

    def tearDown(self):
        self.git.stop()
        self.patch.stop()
        python_imports._roots.cache_clear()
        self.directory.cleanup()

    def names(self, source: str) -> list[str]:
        return [
            f.message.split("`")[1]
            for f in python_imports.check_import_modules(source, "app/main.py")
        ]

    def test_first_party_modules_and_names(self):
        self.assertEqual(self.names("from remote_control import operator_bus\n"), [])
        self.assertEqual(self.names("from tools.hooks import checks\n"), [])
        self.assertEqual(
            self.names("from remote_control.operator_bus import TopicPublisher\n"),
            ["from remote_control.operator_bus import TopicPublisher"],
        )

    def test_the_standard_library(self):
        self.assertEqual(
            self.names(
                "from os import path\nfrom unittest import mock\nfrom collections import abc\n"
            ),
            [],
        )
        self.assertEqual(
            self.names("from dataclasses import dataclass\nfrom os import getcwd\n"),
            ["from dataclasses import dataclass", "from os import getcwd"],
        )

    def test_exempt_and_generated_modules(self):
        source = (
            "from __future__ import annotations\nfrom typing import List\nfrom collections.abc import Callable\n"
            "from typing_extensions import override\nfrom humanoid_mpc_msgs import mpc_policy_pb2\n"
            "from google.protobuf import text_format\n"
        )
        self.assertEqual(self.names(source), [])

    def test_unknown_third_party_names(self):
        self.assertEqual(
            self.names("from scipy.spatial.transform import Rotation\n"),
            ["from scipy.spatial.transform import Rotation"],
        )


class ImportModulesRegistryTest(unittest.TestCase):
    def test_the_registry_runs_it(self):
        check_test_support.assert_check_behaves(
            self,
            "py-import-modules",
            "from dataclasses import dataclass\n",
            "src/a.py",
            clean="import dataclasses\n",
            pending=True,
        )


class RelativeAliasSysPathTest(unittest.TestCase):
    def test_relative_imports(self):
        self.assertEqual(
            len(
                python_imports.check_relative_import(
                    "from . import a\nfrom .b import c\nimport d\n", "x.py"
                )
            ),
            2,
        )

    def test_aliases(self):
        self.assertEqual(
            python_imports.check_import_alias(
                "import numpy as np\nimport tkinter as tk\n", "x.py"
            ),
            [],
        )
        self.assertEqual(
            len(
                python_imports.check_import_alias(
                    "import xml.etree.ElementTree as ET\n", "x.py"
                )
            ),
            1,
        )
        self.assertEqual(
            python_imports.check_import_alias(
                "from brax import train as ppo\n", "x.py"
            ),
            [],
        )

    def test_sys_path(self):
        self.assertEqual(
            len(
                python_imports.check_sys_path(
                    "import sys\nsys.path.insert(0, 'x')\nsys.path.append('y')\n",
                    "x.py",
                )
            ),
            2,
        )
        self.assertEqual(
            python_imports.check_sys_path(
                "import sys\nprint(sys.path)\nlst.insert(0, 1)\n", "x.py"
            ),
            [],
        )
        check = check_test_support.assert_registered(self, "py-sys-path")
        self.assertFalse(
            check.applies_to("tools/hooks/test_x.py"), "tests may change sys.path"
        )

    def test_the_registry_runs_them(self):
        check_test_support.assert_check_behaves(
            self,
            "py-relative-import",
            "from . import a\n",
            "src/a.py",
            clean="from src import a\n",
            pending=True,
        )
        check_test_support.assert_check_behaves(
            self,
            "py-import-alias",
            "import xml.etree.ElementTree as ET\n",
            "src/a.py",
            pending=True,
        )
        check_test_support.assert_check_behaves(
            self,
            "py-sys-path",
            "import sys\nsys.path.insert(0, 'x')\n",
            "src/a.py",
            clean="import sys\n",
        )


if __name__ == "__main__":
    unittest.main()
