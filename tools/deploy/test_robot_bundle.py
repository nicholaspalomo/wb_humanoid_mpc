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

"""The robot bundle (//tools/deploy:robot_bundle), which the robot images are built from.

Its layout, a tar of plain files; the robot binaries' libraries, found from inside it (libmujoco in their runfiles, the
rest on the system); and every robot configuration's script, which starts its binary with files the bundle holds.
"""

import os
import shutil
import subprocess
import tarfile
import tempfile
import unittest

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
    _directory: tempfile.TemporaryDirectory[str]
    root: str
    members: list[tarfile.TarInfo]

    @classmethod
    def setUpClass(cls) -> None:
        # pylint: disable-next=consider-using-with  # tearDownClass() deletes it.
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

    def test_no_yaml_configuration_is_shipped(self) -> None:
        # The configuration files are textprotos; a YAML file in a robot's config/ would be one the robot ignores.
        names = [member.name for member in self.members]
        self.assertEqual(
            [name for name in names if name.endswith((".yaml", ".yml"))], []
        )

    def test_the_binaries_find_their_libraries_from_the_bundle(self) -> None:
        result = subprocess.run(
            ["sh", self.path("bin/collect_runtime_libraries.sh"), "check", self.root],
            capture_output=True,
            text=True,
            check=False,  # The assertion below shows the output.
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
                with open(os.path.join(copy, binary), "w", encoding="utf-8") as stream:
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
                    # The PD gains the binary derives from the task file (ConfigFiles.h, jointPdGainsFileBeside()).
                    # LINT.IfChange(config_layout)
                    gains = os.path.join(
                        os.path.dirname(os.path.dirname(flags["--task_file"])),
                        "controller",
                        "joint_pd_gains.textproto",
                    )
                    # LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/config/ConfigFiles.h:config_layout)
                    self.assertTrue(os.path.isfile(os.path.join(copy, gains)), gains)
                    self.assertEqual(flags["--backend"], "mujoco")
                    # Without a store directory the robot reads the bundle's files in place; the images set one.
                    self.assertEqual(flags["--config_store_dir"], "")
                    self.assertEqual(flags["--config_seed"], "when_bundle_changes")
                    formulation = (
                        "humanoid_wb_mpc"
                        if robot.endswith("_wb")
                        else "humanoid_centroidal_mpc"
                    )
                    self.assertTrue(program.endswith(f"/{formulation}_robot"), program)

            # The environment chooses: the network file, the headless viewer, the priority, the stored configuration.
            lines = self.run_script(
                os.path.join(copy, "launch", "drc_atlas.sh"),
                {
                    "WB_ROBOT_NETWORK_FILE": "config/ipc/two_machine.example.textproto",
                    "WB_ROBOT_HEADLESS": "true",
                    "WB_ROBOT_REALTIME_PRIORITY": "0",
                    "WB_ROBOT_CONFIG_STORE_DIR": "/var/lib/wb-humanoid-robot/config/drc_atlas",
                    "WB_ROBOT_CONFIG_SEED": "every_start",
                },
            )
            self.assertIn(
                "--config_store_dir=/var/lib/wb-humanoid-robot/config/drc_atlas", lines
            )
            self.assertIn("--config_seed=every_start", lines)
            self.assertIn(
                "--network_config=config/ipc/two_machine.example.textproto", lines
            )
            self.assertIn("--headless=true", lines)
            self.assertIn("--realtime_priority=0", lines)

    def run_script(
        self, script: str, environment: dict[str, str] | None = None
    ) -> list[str]:
        """The output lines of `script`, run with PATH and `environment` only; fails the test when it fails."""
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
        env.update(environment or {})
        result = subprocess.run(
            [script], capture_output=True, text=True, env=env, timeout=30, check=False
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout.splitlines()


if __name__ == "__main__":
    unittest.main()
