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

"""The robots' launch files (tools/launch) start, on the machines of the hardware split, binaries and files that exist.

Every robot configuration has its robot, MPC and dummy-simulator launch files and every robot description its model
sandbox; each parses strictly, runs its processes on the machines of the hardware split, starts binaries that exist with
flags those binaries define and files that exist, and the launch files of one robot agree on its files. The robot side
exports to the script the robot-runtime image runs.
"""

import os
import re
import subprocess
import tempfile
import unittest

import launch_file
import launch_script

RUNFILES = os.path.join(os.environ["TEST_SRCDIR"], "_main")

# The robot configurations: (deployment name, MPC package, formulation).
# LINT.IfChange(robot_configurations)
ROBOT_CONFIGURATIONS = (
    ("drc_atlas", "robot_models/drc_atlas/drc_atlas_centroidal_mpc", "centroidal"),
    (
        "engineai_sa01",
        "robot_models/engineai_sa01/engineai_sa01_centroidal_mpc",
        "centroidal",
    ),
    ("unitree_g1", "robot_models/unitree_g1/g1_centroidal_mpc", "centroidal"),
    ("unitree_g1_wb", "robot_models/unitree_g1/g1_wb_mpc", "wb"),
    ("unitree_r1", "robot_models/unitree_r1/unitree_r1_centroidal_mpc", "centroidal"),
)
ROBOT_DESCRIPTIONS = (
    "robot_models/drc_atlas/drc_atlas_description",
    "robot_models/engineai_sa01/engineai_sa01_description",
    "robot_models/unitree_g1/g1_description",
    "robot_models/unitree_r1/unitree_r1_description",
)
# LINT.ThenChange(//Makefile:robot_configurations, //tools/deploy/BUILD.bazel:robot_configurations)

MPC_LAUNCH_FILES = ("robot.textproto", "mpc.textproto", "dummy_sim.textproto")
# The variables every launch file of a robot configuration gives the same value. The task and reference files are also
# the configuration's identity on the bus (ConfigFiles.h, configFileIdentity()): the robot and the MPC refuse an update
# or a save that names another one, so their launch files must name the same files.
SHARED_VARIABLES = ("task_file", "reference_file", "urdf_file", "network_file")
# Flags whose value is a file the process reads.
_FILE_FLAG = re.compile(r"^--(\w+_file|network_config|urdf)=(.+)$")


def launch_paths() -> list[str]:
    paths = []
    for _, package, _ in ROBOT_CONFIGURATIONS:
        paths += [os.path.join(package, "launch", name) for name in MPC_LAUNCH_FILES]
    paths += [
        os.path.join(description, "launch", "sandbox.textproto")
        for description in ROBOT_DESCRIPTIONS
    ]
    return paths


def load(path: str) -> launch_file.LaunchFile:
    """A launch file as the launcher reads it, its binaries looked up among this test's runfiles."""
    return launch_file.load_launch_file(
        os.path.join(RUNFILES, path),
        overrides={"bin_dir": RUNFILES},
        builtins={"repo_root": RUNFILES},
    )


# The flags of each binary, by path: the launch files of every robot start the same few binaries, and asking one for
# its flags starts a process.
_flag_cache: dict[str, set[str]] = {}


def flags_of(program: str) -> set[str]:
    """The flags a binary's help lists: `--helpfull` of an Abseil binary, `--help` of a Python one."""
    if program not in _flag_cache:
        with open(program, "rb") as stream:
            native = stream.read(4) == b"\x7fELF"
        result = subprocess.run(
            [program, "--helpfull" if native else "--help"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=RUNFILES,
            check=False,  # The flags are read from the help text, whatever the exit status.
        )
        _flag_cache[program] = set(
            re.findall(r"--([A-Za-z0-9_]+)", result.stdout + result.stderr)
        )
    return _flag_cache[program]


class LaunchFilesTest(unittest.TestCase):
    def test_every_robot_has_its_launch_files(self) -> None:
        for path in launch_paths():
            with self.subTest(path=path):
                self.assertTrue(os.path.isfile(os.path.join(RUNFILES, path)), path)

    def test_every_process_starts_a_binary_with_files_that_exist(self) -> None:
        for path in launch_paths():
            launch = load(path)
            for process in launch.processes:
                with self.subTest(path=path, process=process.name):
                    program = process.command[0]
                    self.assertTrue(
                        os.access(program, os.X_OK), f"{program} is not built"
                    )
                    for argument in process.command[1:]:
                        match = _FILE_FLAG.match(argument)
                        if match is not None:
                            self.assertTrue(
                                os.path.isfile(os.path.join(RUNFILES, match.group(2))),
                                f"{argument}: no such file",
                            )

    def test_every_flag_is_one_its_binary_defines(self) -> None:
        for path in launch_paths():
            for process in load(path).processes:
                defined = flags_of(process.command[0])
                with self.subTest(path=path, process=process.name):
                    self.assertGreater(len(defined), 1, "the binary listed no flags")
                    for argument in process.command[1:]:
                        name = re.match(r"^--([A-Za-z0-9_]+)", argument)
                        if name is None:
                            self.fail(f"{argument} is not a --flag")
                        self.assertIn(name.group(1), defined, argument)

    def test_the_processes_run_on_the_machines_of_the_hardware_split(self) -> None:
        for _, package, formulation in ROBOT_CONFIGURATIONS:
            prefix = (
                "humanoid_centroidal_mpc"
                if formulation == "centroidal"
                else "humanoid_wb_mpc"
            )
            programs: dict[str, tuple[tuple[str, str], ...]] = {}
            for name in MPC_LAUNCH_FILES:
                launch = load(os.path.join(package, "launch", name))
                programs[name] = tuple(
                    (process.machine, os.path.basename(process.command[0]))
                    for process in launch.processes
                )
            with self.subTest(package=package):
                # The robot process alone on the robot; everything else on the laptop.
                self.assertEqual(
                    programs["robot.textproto"], (("robot", f"{prefix}_robot"),)
                )
                self.assertEqual(
                    programs["mpc.textproto"],
                    (
                        ("laptop", f"{prefix}_node"),
                        ("laptop", "base_velocity_controller_gui"),
                        ("laptop", "humanoid_rerun_viewer"),
                    ),
                )
                self.assertEqual(
                    programs["dummy_sim.textproto"],
                    (
                        ("laptop", f"{prefix}_node"),
                        ("laptop", f"{prefix}_dummy_sim"),
                        ("laptop", "base_velocity_controller_gui"),
                        ("laptop", "humanoid_rerun_viewer"),
                    ),
                )
        for description in ROBOT_DESCRIPTIONS:
            launch = load(os.path.join(description, "launch", "sandbox.textproto"))
            with self.subTest(description=description):
                self.assertEqual(
                    [
                        (process.machine, os.path.basename(process.command[0]))
                        for process in launch.processes
                    ],
                    [("laptop", "model_sandbox"), ("laptop", "humanoid_rerun_viewer")],
                )

    def test_the_launch_files_of_a_robot_agree_on_its_files(self) -> None:
        for _, package, _ in ROBOT_CONFIGURATIONS:
            launches = {
                name: load(os.path.join(package, "launch", name))
                for name in MPC_LAUNCH_FILES
            }
            with self.subTest(package=package):
                for variable in SHARED_VARIABLES:
                    values = {
                        name: launch.variables[variable]
                        for name, launch in launches.items()
                    }
                    self.assertEqual(len(set(values.values())), 1, values)
                self.assertTrue(
                    launches["robot.textproto"]
                    .variables["task_file"]
                    .startswith(package + "/")
                )
                names: set[str] = set()
                for launch in launches.values():
                    for process in launch.processes:
                        names.update(
                            argument
                            for argument in process.command
                            if argument.startswith("--robot_name=")
                        )
                self.assertEqual(len(names), 1, names)

    def test_the_robot_side_passes_its_configuration_store(self) -> None:
        # The robot images name the store directory and the seed policy through the launch file's variables
        # (docker-compose.robot.yaml); the launch file itself reads its files in place.
        for _, package, _ in ROBOT_CONFIGURATIONS:
            launch = load(os.path.join(package, "launch", "robot.textproto"))
            with self.subTest(package=package):
                self.assertEqual(launch.variables["config_store_dir"], "")
                self.assertEqual(launch.variables["config_seed"], "when_bundle_changes")
                (process,) = launch.processes
                self.assertIn("--config_store_dir=", process.command)
                self.assertIn("--config_seed=when_bundle_changes", process.command)

    def test_the_mpc_and_the_robot_are_never_in_one_launch_file(self) -> None:
        # Simulation runs the hardware topology: no launch file starts the robot process next to the MPC, and none
        # passes the retired --mpc_link (the robot binaries refuse it at start-up).
        for path in launch_paths():
            launch = load(path)
            programs = [os.path.basename(p.command[0]) for p in launch.processes]
            arguments: list[str] = []
            for process in launch.processes:
                arguments.extend(process.command)
            with self.subTest(path=path):
                self.assertFalse(
                    any(p.endswith("_robot") for p in programs)
                    and any(p.endswith("_node") for p in programs)
                )
                self.assertEqual(
                    [a for a in arguments if a.startswith("--mpc_link")], []
                )

    def test_the_robot_side_exports_to_a_script(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for deployment, package, _ in ROBOT_CONFIGURATIONS:
                with self.subTest(robot=deployment):
                    script = os.path.join(directory, f"{deployment}.sh")
                    with open(script, "w", encoding="utf-8") as stream:
                        stream.write(
                            launch_script.export_script(
                                os.path.join(
                                    RUNFILES, package, "launch", "robot.textproto"
                                ),
                                "robot",
                                env_prefix="WB_ROBOT_",
                                root="..",
                            )
                        )
                    subprocess.run(["sh", "-n", script], check=True)


if __name__ == "__main__":
    unittest.main()
