"""The robot model filegroups ship their configuration and nothing that merely sits beside it.

Every robot package globs its directories (`config/**`, `urdf/**`, ...), and the tuning GUI and the MPC parameter updater
write git-ignored side files next to the configuration: `<file>.bak` backups on save and `<file>.live.yaml` copies while
tuning. The globs used to take them along, so every test that lists a robot package as data ran against whatever stale
backups the working tree held. The filegroups exclude them now (bazel/robot_files.bzl, ROBOT_FILE_EXCLUDES); this test
walks what they actually put in the runfiles.
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
    "drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml",
    "drc_atlas/drc_atlas_description/urdf/atlas.urdf",
    "engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/task.yaml",
    "engineai_sa01/engineai_sa01_description/urdf/zq_sa01.urdf",
    "unitree_g1/g1_centroidal_mpc/config/mpc/task.yaml",
    "unitree_g1/g1_wb_mpc/config/mpc/task.yaml",
    "unitree_g1/g1_description/urdf/g1_29dof.urdf",
    "unitree_r1/unitree_r1_centroidal_mpc/config/mpc/task.yaml",
    "unitree_r1/unitree_r1_description/urdf/R1.urdf",
)


def is_side_file(name):
    return any(fnmatch.fnmatch(name, pattern) for pattern in SIDE_FILE_PATTERNS)


class TestRobotPackagesShipNoSideFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This file's directory in the runfiles is robot_models/tests; its parent holds every robot package listed as data.
        cls.root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

    def test_the_patterns_recognize_what_the_gui_and_the_updater_write(self):
        # Positive control of the matcher on the names those writers produce.
        self.assertTrue(is_side_file("task.yaml.bak"))
        self.assertTrue(is_side_file("joint_pd_gains.yaml.bak"))
        self.assertTrue(is_side_file("task.yaml.live.yaml"))
        self.assertFalse(is_side_file("task.yaml"))
        self.assertFalse(is_side_file("contact_planning.yaml"))

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
