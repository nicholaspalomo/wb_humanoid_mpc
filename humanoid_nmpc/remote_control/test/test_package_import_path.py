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

"""The remote_control tests exercise the sources next to them, and the package needs no ROS.

Bazel puts humanoid_nmpc/remote_control on the import path (`imports = ["."]` in BUILD.bazel), so `import remote_control`
must load the package of this source tree, with its subpackage, rather than any other copy. And with ROS 2 gone from
the operator tools, importing every module of the package must not bring in rclpy, a ROS message package or the ROS
launch system.
"""

import importlib
import os
import pkgutil
import sys
import unittest

import remote_control
from remote_control.tk_app import dodgeball

# The package next to this test, as this test's tree (the runfiles under Bazel) holds it.
SOURCE_PACKAGE = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "remote_control")
)

# Top-level modules of ROS 2 that the operator tools used to import.
ROS_MODULES = (
    "rclpy",
    "std_msgs",
    "launch",
    "launch_ros",
    "ament_index_python",
    "rosidl_runtime_py",
)


class TestPackageImportPath(unittest.TestCase):
    """The modules imported are the files next to this test, wherever the tree links them to.

    Compared as files (os.path.samefile), not as paths: Bazel may hand the interpreter the sources' real paths and this
    test their runfiles links. A package found through a shell's PYTHONPATH (another checkout's
    humanoid_nmpc/remote_control, say) is another file, and fails.
    """

    def test_the_package_is_imported_from_this_source_tree(self):
        self.assertTrue(
            os.path.samefile(
                remote_control.__file__, os.path.join(SOURCE_PACKAGE, "__init__.py")
            ),
            remote_control.__file__,
        )

    def test_a_subpackage_is_imported_from_this_source_tree(self):
        self.assertTrue(
            os.path.samefile(
                dodgeball.__file__,
                os.path.join(SOURCE_PACKAGE, "tk_app", "dodgeball.py"),
            ),
            dodgeball.__file__,
        )

    def test_positive_control_a_module_from_elsewhere_is_told_apart(self):
        # The comparison above can fail: a module loaded from anywhere else is not that file.
        self.assertFalse(
            os.path.samefile(
                unittest.__file__, os.path.join(SOURCE_PACKAGE, "__init__.py")
            )
        )


class TestNoRos(unittest.TestCase):
    def test_every_module_imports_without_ros(self):
        modules = [
            info.name
            for info in pkgutil.walk_packages(
                remote_control.__path__, prefix="remote_control."
            )
        ]
        # Positive control: the walk found the GUI, the bus side and the tabs.
        for expected in (
            "remote_control.base_velocity_controller_gui",
            "remote_control.operator_bus",
            "remote_control.tk_app.mpc_params_tab",
        ):
            self.assertIn(expected, modules)
        for name in modules:
            with self.subTest(module=name):
                importlib.import_module(name)
        loaded = sorted(
            name for name in sys.modules if name.split(".")[0] in ROS_MODULES
        )
        self.assertEqual(loaded, [], "a module of the package imported ROS")

    def test_no_source_mentions_a_ros_import(self):
        # Also the imports a test never reaches, such as one inside a function.
        offending = []
        for directory, _, files in os.walk(SOURCE_PACKAGE):
            for file_name in files:
                if not file_name.endswith(".py"):
                    continue
                path = os.path.join(directory, file_name)
                with open(path, "r", encoding="utf-8") as handle:
                    for number, line in enumerate(handle, start=1):
                        words = line.split()
                        if len(words) >= 2 and words[0] in ("import", "from"):
                            if words[1].split(".")[0] in ROS_MODULES:
                                offending.append(f"{path}:{number}: {line.strip()}")
        self.assertEqual(offending, [])


if __name__ == "__main__":
    unittest.main()
