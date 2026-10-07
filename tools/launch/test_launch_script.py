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

"""Tests for launch_script.py: a machine's process of a launch file exported as a POSIX sh script.

The script runs the command the launcher would run, takes its variables from the environment, and refuses what a script
cannot do.
"""

import io
import os
import stat
import subprocess
import tempfile
import unittest

import launch_file
import launch_script

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLE = os.path.join(HERE, "examples", "example.launch.textproto")

# Prints every argument on a line of its own, so that a test sees the argument vector exactly.
PRINT_ARGUMENTS = (
    '["sh", "-c", "for a in \\"$@\\"; do printf \\"%s\\\\n\\" \\"$a\\"; done", "sh"'
)


class ScriptTestCase(unittest.TestCase):
    """A temporary directory for the launch file and the exported script, and the runs of the script."""

    def setUp(self) -> None:
        # pylint: disable-next=consider-using-with  # tearDown() deletes it.
        self._directory = tempfile.TemporaryDirectory()
        self.directory = self._directory.name

    def tearDown(self) -> None:
        self._directory.cleanup()

    def write_launch_file(self, text: str) -> str:
        path = os.path.join(self.directory, "test.launch.textproto")
        with open(path, "w", encoding="utf-8") as stream:
            stream.write(text)
        return path

    def export(self, path: str, machine: str = "robot", **kwargs) -> str:
        script = os.path.join(self.directory, "exported", "run.sh")
        os.makedirs(os.path.dirname(script), exist_ok=True)
        with open(script, "w", encoding="utf-8") as stream:
            stream.write(launch_script.export_script(path, machine, **kwargs))
        os.chmod(script, 0o755)
        return script

    def run_script(
        self, script: str, environment: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess:
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
        env.update(environment or {})
        return subprocess.run(
            [script],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
            check=False,  # The tests check the return code.
        )

    def arguments(
        self, script: str, environment: dict[str, str] | None = None
    ) -> list[str]:
        result = self.run_script(script, environment)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout.splitlines()


class ExampleTest(ScriptTestCase):
    def test_the_robot_process_of_the_example_runs_as_the_launcher_runs_it(
        self,
    ) -> None:
        script = self.export(EXAMPLE, "robot")
        subprocess.run(["sh", "-n", script], check=True)
        self.assertEqual(
            self.arguments(script),
            [f"hello from the robot, cycle {i}" for i in (1, 2, 3)],
        )

    def test_the_environment_and_set_change_its_variables(self) -> None:
        self.assertEqual(
            len(self.arguments(self.export(EXAMPLE), {"LAUNCH_CYCLES": "2"})), 2
        )
        script = self.export(EXAMPLE, overrides={"cycles": "4"})
        self.assertEqual(len(self.arguments(script)), 4)
        # Empty means the default, as for a compose file that passes an unset variable on as "".
        self.assertEqual(len(self.arguments(script, {"LAUNCH_CYCLES": ""})), 4)

    def test_a_machine_with_several_processes_is_refused(self) -> None:
        with self.assertRaisesRegex(
            launch_file.LaunchFileError, r"runs 2 processes \(mpc, viewer\)"
        ):
            launch_script.export_script(EXAMPLE, "laptop")

    def test_the_header_lists_every_variable_with_its_value(self) -> None:
        text = launch_script.export_script(EXAMPLE, "robot")
        self.assertIn("LAUNCH_CYCLES", text)
        self.assertRegex(text, r"#   LAUNCH_GREETING +greeting = hello")
        self.assertIn("Do not edit it", text)


class VariableTest(ScriptTestCase):
    def launch(self, variables: str, command: str, process: str = "") -> str:
        return self.write_launch_file(
            f"{variables}\n"
            'processes { name: "robot" machine: MACHINE_ROBOT '
            f"command: {command}] {process} }}\n"
        )

    def test_a_variable_follows_the_one_it_refers_to(self) -> None:
        path = self.launch(
            'variables { name: "config_dir" value: "robots/a" }\n'
            'variables { name: "task_file" value: "{config_dir}/task.textproto" }',
            PRINT_ARGUMENTS + ', "--task_file={task_file}"',
        )
        script = self.export(path, env_prefix="WB_ROBOT_")
        self.assertEqual(
            self.arguments(script), ["--task_file=robots/a/task.textproto"]
        )
        self.assertEqual(
            self.arguments(script, {"WB_ROBOT_CONFIG_DIR": "/elsewhere"}),
            ["--task_file=/elsewhere/task.textproto"],
        )
        self.assertEqual(
            self.arguments(script, {"WB_ROBOT_TASK_FILE": "mine.textproto"}),
            ["--task_file=mine.textproto"],
        )

    def test_the_dependency_order_does_not_depend_on_the_file_order(self) -> None:
        path = self.launch(
            'variables { name: "b" value: "{a}-b" }\nvariables { name: "a" value: "a" }',
            PRINT_ARGUMENTS + ', "{b}"',
        )
        self.assertEqual(self.arguments(self.export(path)), ["a-b"])

    def test_special_characters_reach_the_process_unchanged(self) -> None:
        tricky = "a b 'c' \\\"d\\\" $HOME `x` \\\\ {{literal}}"
        path = self.launch(
            f'variables {{ name: "v" value: "{tricky}" }}',
            PRINT_ARGUMENTS + f', "{tricky}", "=={{v}}=="',
        )
        expected = "a b 'c' \"d\" $HOME `x` \\ {literal}"
        self.assertEqual(
            self.arguments(self.export(path)), [expected, f"=={expected}=="]
        )
        # A value from the environment is not expanded again either.
        self.assertEqual(
            self.arguments(self.export(path), {"LAUNCH_V": "$(echo no) `no`"})[1],
            "==$(echo no) `no`==",
        )

    def test_the_repository_root_is_the_root_of_the_script(self) -> None:
        path = self.launch("", PRINT_ARGUMENTS + ', "{repo_root}"')
        relative = self.export(path, root="..")
        self.assertEqual(
            [os.path.realpath(argument) for argument in self.arguments(relative)],
            [os.path.realpath(self.directory)],
        )
        absolute = self.export(path, root="/")
        self.assertEqual(self.arguments(absolute), ["/"])

    def test_the_process_runs_in_the_root(self) -> None:
        os.makedirs(os.path.join(self.directory, "data"))
        with open(
            os.path.join(self.directory, "data", "file.txt"), "w", encoding="utf-8"
        ) as stream:
            stream.write("found\n")
        path = self.launch("", '["cat", "data/file.txt"')
        self.assertEqual(self.arguments(self.export(path, root="..")), ["found"])

    def test_the_process_environment_and_delay_are_kept(self) -> None:
        path = self.launch(
            'variables { name: "greeting" value: "hi" }',
            '["sh", "-c", "echo $GREETING"',
            'env { name: "GREETING" value: "{greeting} there" } delay: 0.1',
        )
        script = self.export(path)
        self.assertIn("sleep 0.1", launch_script.export_script(path, "robot"))
        self.assertEqual(self.arguments(script), ["hi there"])
        self.assertEqual(
            self.arguments(script, {"LAUNCH_GREETING": "ho"}), ["ho there"]
        )

    def test_the_process_replaces_the_script(self) -> None:
        # exec: the process has the script's PID, so it receives the signals sent to the script (docker stop).
        path = self.launch("", '["sh", "-c", "echo $$"')
        script = self.export(path)
        with subprocess.Popen([script], stdout=subprocess.PIPE, text=True) as process:
            output, _ = process.communicate(timeout=30)
        self.assertEqual(process.returncode, 0)
        self.assertEqual(output.strip(), str(process.pid))

    def test_variables_that_share_a_shell_variable_are_refused(self) -> None:
        path = self.launch(
            'variables { name: "a_b" value: "1" }\nvariables { name: "A_B" value: "2" }',
            '["true"',
        )
        with self.assertRaisesRegex(launch_file.LaunchFileError, r"LAUNCH_A_B"):
            launch_script.export_script(path, "robot")

    def test_a_terminal_process_is_refused(self) -> None:
        path = self.launch("", '["true"', "terminal: true")
        with self.assertRaisesRegex(launch_file.LaunchFileError, "terminal"):
            launch_script.export_script(path, "robot")

    def test_a_machine_without_a_process_is_refused(self) -> None:
        path = self.launch("", '["true"')
        with self.assertRaisesRegex(launch_file.LaunchFileError, "runs 0 processes"):
            launch_script.export_script(path, "laptop")

    def test_an_invalid_prefix_is_refused(self) -> None:
        path = self.launch("", '["true"')
        for prefix in ("lower_", "WITH-DASH", ""):
            with self.subTest(prefix=prefix):
                with self.assertRaisesRegex(launch_file.LaunchFileError, "env_prefix"):
                    launch_script.export_script(path, "robot", env_prefix=prefix)


class CommandLineTest(ScriptTestCase):
    def test_main_writes_an_executable_script(self) -> None:
        output = os.path.join(self.directory, "robot.sh")
        status = launch_script.main(
            [EXAMPLE, "--machine", "robot", "--output", output, "--set", "cycles=1"]
        )
        self.assertEqual(status, 0)
        self.assertTrue(os.stat(output).st_mode & stat.S_IXUSR)
        self.assertEqual(len(self.arguments(output)), 1)

    def test_an_invalid_launch_file_is_a_usage_error_naming_its_line(self) -> None:
        path = self.write_launch_file('processes { name: "robot" requried: true }\n')
        errors = io.StringIO()
        status = launch_script.main(
            [path, "--machine", "robot", "--output", os.path.join(self.directory, "x")],
            err=errors,
        )
        self.assertEqual(status, launch_script.EXIT_USAGE)
        self.assertRegex(
            errors.getvalue(), r"test\.launch\.textproto:1:\d+: .*requried"
        )


if __name__ == "__main__":
    unittest.main()
