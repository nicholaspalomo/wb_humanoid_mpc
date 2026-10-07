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

"""The robot model filegroups ship their configuration and nothing that merely sits beside it.

Every robot package globs its directories (`config/**`, `urdf/**`, ...), and git-ignored side files sit next to the
configuration: the `<file>.bak` backup the tuning GUI's Save keeps, and the `<file>.live.yaml` copies that the GUI and the
MPC parameter updater of the YAML configuration wrote while tuning, which a working tree may still hold. The globs used
to take them along, so every test that lists a robot package as data ran against whatever stale backups the working tree
held. The filegroups exclude them now (bazel/robot_files.bzl, ROBOT_FILE_EXCLUDES); this test walks what they actually
put in the runfiles.
"""

import fnmatch
import os
import unittest

# The side files, as .gitignore names them.
# LINT.IfChange(robot_file_excludes)
SIDE_FILE_PATTERNS = ("*.bak", "*.live*")
# LINT.ThenChange(//bazel/robot_files.bzl:robot_file_excludes)

# Every robot package the BUILD target lists as data, and one file each must ship: the walk below is not vacuous.
EXPECTED_FILES = (
    "drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.textproto",
    "drc_atlas/drc_atlas_description/urdf/atlas.urdf",
    "engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/task.textproto",
    "engineai_sa01/engineai_sa01_description/urdf/zq_sa01.urdf",
    "unitree_g1/g1_centroidal_mpc/config/mpc/task.textproto",
    "unitree_g1/g1_wb_mpc/config/mpc/task.textproto",
    "unitree_g1/g1_description/urdf/g1_29dof.urdf",
    "unitree_r1/unitree_r1_centroidal_mpc/config/mpc/task.textproto",
    "unitree_r1/unitree_r1_description/urdf/R1.urdf",
)


def is_side_file(name: str) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in SIDE_FILE_PATTERNS)


class TestRobotPackagesShipNoSideFiles(unittest.TestCase):
    root: str

    @classmethod
    def setUpClass(cls) -> None:
        # This file's directory in the runfiles is robot_models/tests; its parent holds every robot package listed as data.
        cls.root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

    def test_the_patterns_recognize_the_side_files(self):
        # Positive control of the matcher on the names the GUI's Save writes and on the stale YAML-era copies.
        self.assertTrue(is_side_file("task.textproto.bak"))
        self.assertTrue(is_side_file("joint_pd_gains.textproto.bak"))
        self.assertTrue(is_side_file("task.yaml.bak"))
        self.assertTrue(is_side_file("task.yaml.live.yaml"))
        self.assertFalse(is_side_file("task.textproto"))
        self.assertFalse(is_side_file("contact_planning.textproto"))

    def test_every_listed_package_ships_its_files(self):
        for relative in EXPECTED_FILES:
            with self.subTest(file=relative):
                self.assertTrue(
                    os.path.isfile(os.path.join(self.root, relative)),
                    f"{relative} is not in the runfiles",
                )

    def test_no_side_file_reaches_the_runfiles(self):
        shipped = []
        side_files = []
        for dirpath, _, filenames in os.walk(self.root):
            for filename in filenames:
                relative = os.path.relpath(os.path.join(dirpath, filename), self.root)
                shipped.append(relative)
                if is_side_file(filename):
                    side_files.append(relative)
        self.assertGreater(len(shipped), len(EXPECTED_FILES))
        self.assertEqual(side_files, [], "git-ignored side files were shipped as data")


if __name__ == "__main__":
    unittest.main()
