"""The robot bundle (//tools/deploy:robot_bundle): what the robot images are built from. Its layout, a tar of plain
files, the robot binaries' libraries found from inside it (libmujoco in their runfiles, the rest on the system), and
every robot configuration's script starting its binary with files the bundle holds."""

import os
import shutil
import stat
import subprocess
import tarfile
import tempfile
import unittest
from typing import Dict, List

BUNDLE = os.path.join(
    os.environ["TEST_SRCDIR"], "_main", "tools", "deploy", "robot_bundle.tar"
)
BINARIES = (
    "bin/humanoid_nmpc/humanoid_centroidal_mpc_app/humanoid_centroidal_mpc_robot",
    "bin/humanoid_nmpc/humanoid_wb_mpc_app/humanoid_wb_mpc_robot",
)
# LINT.IfChange(robot_configurations)
ROBOTS = ("drc_atlas", "engineai_sa01", "unitree_g1", "unitree_g1_wb", "unitree_r1")
# LINT.ThenChange(//tools/deploy/BUILD.bazel:robot_configurations)
# The flags whose values are files the robot process reads.
FILE_FLAGS = (
    "--task_file",
    "--reference_file",
    "--urdf_file",
    "--mjcf_file",
    "--network_config",
)


class RobotBundleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        cls.root = cls._directory.name
        with tarfile.open(BUNDLE) as archive:
            cls.members = archive.getmembers()
            archive.extractall(cls.root, filter="data")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def path(self, relative: str) -> str:
        return os.path.join(self.root, relative)

    def test_the_tar_holds_plain_files_owned_by_root(self) -> None:
        for member in self.members:
            with self.subTest(member=member.name):
                self.assertTrue(member.isfile() or member.isdir())
                self.assertEqual((member.uid, member.gid, member.mtime), (0, 0, 0))
                self.assertIn(member.mode, (0o644, 0o755))

    def test_the_layout(self) -> None:
        for binary in BINARIES:
            with self.subTest(binary=binary):
                self.assertTrue(os.access(self.path(binary), os.X_OK))
                self.assertTrue(os.path.isdir(self.path(binary + ".runfiles/_main")))
        for script in (
            "robot_entrypoint.sh",
            "netem.sh",
            "collect_runtime_libraries.sh",
        ):
            self.assertTrue(os.access(self.path(f"bin/{script}"), os.X_OK), script)
        self.assertEqual(
            sorted(os.listdir(self.path("launch"))),
            sorted(f"{robot}.sh" for robot in ROBOTS),
        )
        for network in ("network.textproto", "two_machine.example.textproto"):
            self.assertTrue(os.path.isfile(self.path(f"config/ipc/{network}")))

    def test_no_side_file_of_the_tuning_gui_is_shipped(self) -> None:
        names = [member.name for member in self.members]
        self.assertEqual(
            [name for name in names if name.endswith(".bak") or ".live" in name], []
        )

    def test_the_binaries_find_their_libraries_from_the_bundle(self) -> None:
        result = subprocess.run(
            ["sh", self.path("bin/collect_runtime_libraries.sh"), "check", self.root],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for binary in BINARIES:
            output = subprocess.run(
                ["ldd", self.path(binary)], capture_output=True, text=True, check=True
            ).stdout
            mujoco = [line for line in output.splitlines() if "libmujoco" in line]
            with self.subTest(binary=binary):
                self.assertEqual(len(mujoco), 1, output)
                # From the bundle's own runfiles, never from Bazel's output tree.
                self.assertIn(
                    os.path.realpath(self.root),
                    os.path.realpath(mujoco[0].split("=>")[1].split()[0]),
                )

    def test_every_robot_script_starts_its_binary_with_files_of_the_bundle(
        self,
    ) -> None:
        # The binaries replaced by a stand-in that prints its arguments, in a copy of the bundle.
        with tempfile.TemporaryDirectory() as directory:
            copy = os.path.join(directory, "bundle")
            shutil.copytree(self.root, copy, symlinks=True)
            for binary in BINARIES:
                with open(os.path.join(copy, binary), "w") as stream:
                    stream.write(
                        '#!/bin/sh\necho "$0"\nfor a in "$@"; do echo "$a"; done\n'
                    )
            for robot in ROBOTS:
                with self.subTest(robot=robot):
                    lines = self.run_script(os.path.join(copy, "launch", f"{robot}.sh"))
                    program, arguments = lines[0], lines[1:]
                    self.assertIn(
                        (
                            os.path.relpath(program, copy)
                            if os.path.isabs(program)
                            else program
                        ),
                        BINARIES,
                    )
                    flags = dict(argument.split("=", 1) for argument in arguments)
                    for flag in FILE_FLAGS:
                        self.assertTrue(
                            os.path.isfile(os.path.join(copy, flags[flag])),
                            f"{flag}={flags[flag]}",
                        )
                    self.assertEqual(flags["--backend"], "mujoco")
                    formulation = (
                        "humanoid_wb_mpc"
                        if robot.endswith("_wb")
                        else "humanoid_centroidal_mpc"
                    )
                    self.assertTrue(program.endswith(f"/{formulation}_robot"), program)

            # The environment chooses: the network file, the headless viewer, the priority.
            lines = self.run_script(
                os.path.join(copy, "launch", "drc_atlas.sh"),
                {
                    "WB_ROBOT_NETWORK_FILE": "config/ipc/two_machine.example.textproto",
                    "WB_ROBOT_HEADLESS": "true",
                    "WB_ROBOT_REALTIME_PRIORITY": "0",
                },
            )
            self.assertIn(
                "--network_config=config/ipc/two_machine.example.textproto", lines
            )
            self.assertIn("--headless=true", lines)
            self.assertIn("--realtime_priority=0", lines)

    def run_script(self, script: str, environment: Dict[str, str] = None) -> List[str]:
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
        env.update(environment or {})
        result = subprocess.run(
            [script], capture_output=True, text=True, env=env, timeout=30
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout.splitlines()


if __name__ == "__main__":
    unittest.main()
