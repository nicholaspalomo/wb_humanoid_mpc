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

"""Tests for launch_file.py: the schema, the strict textproto parsing, the validation and the variable substitution."""

from collections.abc import Iterator
import dataclasses
import os
import re
import tempfile
import unittest

from google.protobuf import descriptor
from launch_proto import launch_file_pb2

import launch_file

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
EXAMPLE = os.path.join(HERE, "examples", "example.launch.textproto")
SOURCE = "test.launch.textproto"
BUILTINS = {"repo_root": "/repo"}


def minimal(process: str = "", extra: str = "") -> str:
    """A launch file with one process; `process` adds fields to it and `extra` adds top-level fields."""
    return (
        f"{extra}\n"
        'processes { name: "mpc" machine: MACHINE_LAPTOP command: "prog" '
        f"{process} }}\n"
    )


def parse(text: str, **overrides: str) -> launch_file.LaunchFile:
    return launch_file.parse_launch_text(text, SOURCE, overrides, BUILTINS)


def schema_messages(
    message: descriptor.Descriptor,
) -> Iterator[descriptor.Descriptor]:
    """LaunchFile and every message of the schema nested in it."""
    yield message
    for nested in message.nested_types:
        yield from schema_messages(nested)


class SchemaTest(unittest.TestCase):

    def test_the_resolved_types_have_the_fields_of_the_schema(self) -> None:
        # The reader turns every field of the schema into a field of its own type, so a field added to the schema
        # and not read fails here.
        cases = {
            "Process": (launch_file.ProcessSpec, launch_file_pb2.LaunchFile.Process),
            "Shutdown": (
                launch_file.ShutdownPolicy,
                launch_file_pb2.LaunchFile.Shutdown,
            ),
        }
        for case, (resolved, message) in cases.items():
            with self.subTest(message=case):
                self.assertEqual(
                    [field.name for field in dataclasses.fields(resolved)],
                    [field.name for field in message.DESCRIPTOR.fields],
                )
        self.assertEqual(
            {field.name for field in launch_file_pb2.LaunchFile.DESCRIPTOR.fields},
            {"variables", "shutdown", "processes"},
        )

    def test_the_machines_are_the_values_of_the_machine_enum(self) -> None:
        values = launch_file_pb2.LaunchFile.Machine.DESCRIPTOR.values
        self.assertEqual(values[0].name, "MACHINE_UNSPECIFIED")
        self.assertEqual(
            list(launch_file.MACHINES),
            [value.name[len("MACHINE_") :].lower() for value in values[1:]],
        )
        self.assertIn("robot", launch_file.MACHINES)
        self.assertIn("laptop", launch_file.MACHINES)


class ExampleTest(unittest.TestCase):

    def setUp(self) -> None:
        with open(EXAMPLE, "r", encoding="utf-8") as stream:
            self.text = stream.read()

    def test_the_example_parses_with_every_field(self) -> None:
        launch = launch_file.load_launch_file(EXAMPLE, builtins=BUILTINS)
        self.assertEqual(
            [p.name for p in launch.processes], ["robot_loop", "mpc", "viewer"]
        )
        self.assertEqual(
            {p.machine for p in launch.processes}, set(launch_file.MACHINES)
        )
        self.assertTrue(launch.processes[0].required)
        self.assertEqual(launch.processes[1].env, (("GREETING", "hello"),))
        self.assertEqual(
            launch.processes[2].command[-1], 'echo "hello from /repo"; exec sleep 60'
        )
        self.assertEqual(launch.shutdown, launch_file.ShutdownPolicy(2.0, 1.0))
        # Every field of the schema appears in the example, so the example documents all of them.
        without_comments = re.sub(r"(?m)#.*$", "", self.text)
        for message in schema_messages(launch_file_pb2.LaunchFile.DESCRIPTOR):
            for field in message.fields:
                with self.subTest(field=field.full_name):
                    self.assertRegex(without_comments, rf"\b{field.name}\s*[:\{{]")

    def test_the_header_names_the_schema(self) -> None:
        header = dict(
            re.findall(r"(?m)^# (proto-file|proto-message): (\S+)$", self.text)
        )
        self.assertEqual(
            header["proto-message"], launch_file_pb2.LaunchFile.DESCRIPTOR.full_name
        )
        self.assertTrue(
            os.path.isfile(os.path.join(REPO_ROOT, header["proto-file"])),
            header["proto-file"],
        )
        self.assertEqual(
            os.path.basename(header["proto-file"]),
            os.path.basename(launch_file_pb2.DESCRIPTOR.name),
        )

    def test_overrides_replace_declared_variables(self) -> None:
        launch = launch_file.load_launch_file(
            EXAMPLE, overrides={"cycles": "7"}, builtins=BUILTINS
        )
        self.assertIn("-lt 7 ]", launch.processes[0].command[-1])
        self.assertEqual(launch.variables["cycles"], "7")


class DefaultsTest(unittest.TestCase):

    def test_optional_fields_default_to_a_plain_process(self) -> None:
        launch = parse(minimal())
        self.assertEqual(
            launch.processes[0],
            launch_file.ProcessSpec(name="mpc", machine="laptop", command=("prog",)),
        )
        self.assertEqual(launch.shutdown, launch_file.ShutdownPolicy())
        self.assertFalse(launch.processes[0].required)
        self.assertFalse(launch.processes[0].terminal)
        self.assertEqual(launch.processes[0].delay, 0.0)

    def test_an_explicit_zero_grace_period_is_kept_and_an_unset_one_defaults(
        self,
    ) -> None:
        launch = parse(minimal(extra="shutdown { sigint_grace_period: 0 }"))
        self.assertEqual(launch.shutdown.sigint_grace_period, 0.0)
        self.assertEqual(
            launch.shutdown.sigterm_grace_period,
            launch_file.DEFAULT_SIGTERM_GRACE_PERIOD_S,
        )

    def test_the_command_may_be_a_list_or_repeated_entries(self) -> None:
        listed = parse(minimal(process='command: ["--port", "5600"]'))
        repeated = parse(minimal(process='command: "--port" command: "5600"'))
        self.assertEqual(listed.processes[0].command, ("prog", "--port", "5600"))
        self.assertEqual(listed.processes, repeated.processes)

    def test_comments_and_single_quoted_strings_are_text_format(self) -> None:
        launch = parse(
            "# a comment\n"
            "processes {  # another\n"
            '  name: "sh" machine: MACHINE_ROBOT\n'
            "  command: ['sh', '-c', 'echo \"$HOME\"']\n"
            "}\n"
        )
        self.assertEqual(launch.processes[0].command, ("sh", "-c", 'echo "$HOME"'))


class SubstitutionTest(unittest.TestCase):

    def test_variables_may_refer_to_variables_and_builtins(self) -> None:
        launch = parse(
            minimal(
                process='command: "--task_file={config}/task.textproto" '
                'env { name: "CONFIG" value: "{config}" }',
                extra='variables { name: "robot" value: "g1" } '
                'variables { name: "config" value: "{repo_root}/robots/{robot}" }',
            ).replace('command: "prog"', 'command: "bin/{robot}_mpc"')
        )
        self.assertEqual(
            launch.processes[0].command,
            ("bin/g1_mpc", "--task_file=/repo/robots/g1/task.textproto"),
        )
        self.assertEqual(launch.processes[0].env, (("CONFIG", "/repo/robots/g1"),))

    def test_overrides_apply_before_references_are_resolved(self) -> None:
        text = minimal(
            process='command: "{config}"',
            extra='variables { name: "robot" value: "g1" } '
            'variables { name: "config" value: "robots/{robot}" }',
        )
        self.assertEqual(
            parse(text, robot="sa01").processes[0].command, ("prog", "robots/sa01")
        )

    def test_doubled_braces_are_literal(self) -> None:
        launch = parse(minimal(process="command: ['-c', 'echo ${{HOME}} {{}}']"))
        self.assertEqual(launch.processes[0].command[-1], "echo ${HOME} {}")

    def test_a_repo_root_with_spaces_is_a_valid_program(self) -> None:
        launch = launch_file.parse_launch_text(
            minimal().replace('"prog"', '"{repo_root}/bin/mpc"'),
            SOURCE,
            builtins={"repo_root": "/home/me/my checkout"},
        )
        self.assertEqual(launch.processes[0].command, ("/home/me/my checkout/bin/mpc",))

    def test_errors_name_the_problem_and_the_known_variables(self) -> None:
        cases = {
            "undefined": ('command: "{nope}"', "undefined variable '{nope}'"),
            "positional": ('command: "{}"', "is not a variable reference"),
            "attribute": (
                'command: "{repo_root.upper}"',
                "is not a variable reference",
            ),
            "format spec": (
                'command: "{repo_root:>10}"',
                "is not a variable reference",
            ),
            "unbalanced": ('command: "{repo_root"', "literal braces"),
        }
        for case, (process, expected) in cases.items():
            with self.subTest(case=case):
                with self.assertRaises(launch_file.LaunchFileError) as raised:
                    parse(minimal(process=process))
                self.assertIn(expected, str(raised.exception))
                self.assertIn(f"{SOURCE}: process 'mpc'", str(raised.exception))
        with self.assertRaises(launch_file.LaunchFileError) as raised:
            parse(minimal(process='command: "{nope}"'))
        self.assertIn("the variables are: repo_root", str(raised.exception))

    def test_a_cycle_of_references_is_reported(self) -> None:
        text = minimal(
            extra='variables { name: "a" value: "{b}" } '
            'variables { name: "b" value: "{c}" } '
            'variables { name: "c" value: "{a}" }'
        )
        with self.assertRaises(launch_file.LaunchFileError) as raised:
            parse(text)
        self.assertIn("cycle: a -> b -> c -> a", str(raised.exception))

    def test_overrides_must_name_declared_variables(self) -> None:
        text = minimal(extra='variables { name: "robot" value: "g1" }')
        with self.assertRaises(launch_file.LaunchFileError) as raised:
            parse(text, robto="sa01")
        self.assertIn("undeclared variable(s) robto", str(raised.exception))
        self.assertIn("declares: robot", str(raised.exception))

    def test_builtins_cannot_be_redefined(self) -> None:
        text = minimal(extra='variables { name: "repo_root" value: "/elsewhere" }')
        with self.assertRaises(launch_file.LaunchFileError) as raised:
            parse(text)
        self.assertIn("built-in", str(raised.exception))


class ParseOverridesTest(unittest.TestCase):

    def test_name_value_pairs(self) -> None:
        self.assertEqual(
            launch_file.parse_overrides(["a=1", "b=", "c=x=y"]),
            {"a": "1", "b": "", "c": "x=y"},
        )

    def test_malformed_assignments_are_rejected(self) -> None:
        for assignment in ("novalue", "=value", "not-an-identifier=1"):
            with self.subTest(assignment=assignment):
                with self.assertRaises(launch_file.LaunchFileError):
                    launch_file.parse_overrides([assignment])


class StrictParsingTest(unittest.TestCase):
    """What the text format parser rejects: the error names the file, the line and the column."""

    def test_an_unknown_field_names_its_line_and_column(self) -> None:
        text = 'processes {\n  name: "mpc"\n  requried: true\n}\n'
        with self.assertRaises(launch_file.LaunchFileError) as raised:
            parse(text)
        self.assertRegex(
            str(raised.exception), rf"^{re.escape(SOURCE)}:3:3: .*requried"
        )

    def test_malformed_documents_are_rejected_with_their_location(self) -> None:
        cases = {
            "unknown top-level field": minimal(extra="nodes {}"),
            "unknown machine": minimal().replace("MACHINE_LAPTOP", "MACHINE_DESKTOP"),
            "machine as a string": minimal().replace("MACHINE_LAPTOP", '"laptop"'),
            "number as an argument": minimal(process="command: 5600"),
            "boolean as an argument": minimal(process="command: true"),
            "required not a bool": minimal(process='required: "yes"'),
            "terminal not a bool": minimal(process="terminal: yes"),
            "delay as text": minimal(process='delay: "1"'),
            "a field given twice": minimal(process="delay: 1 delay: 2"),
            "unknown shutdown field": minimal(extra="shutdown { grace: 1 }"),
            "variable value not a string": minimal(
                extra='variables { name: "a" value: 1 }'
            ),
            "env as a map": minimal(process='env { key: "A" value: "1" }'),
            "unclosed block": "processes {\n",
            "yaml": "processes:\n  - name: mpc\n",
        }
        for case, text in cases.items():
            with self.subTest(case=case):
                with self.assertRaises(launch_file.LaunchFileError) as raised:
                    parse(text)
                self.assertRegex(
                    str(raised.exception), rf"^{re.escape(SOURCE)}:\d+(:\d+)?: "
                )

    def test_a_yaml_file_is_told_that_launch_files_are_textprotos(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "old.launch.yaml")
            with open(path, "w", encoding="utf-8") as stream:
                stream.write("processes:\n  - name: mpc\n")
            with self.assertRaises(launch_file.LaunchFileError) as raised:
                launch_file.load_launch_file(path)
        self.assertIn("textprotos of launch_proto.LaunchFile", str(raised.exception))


class ValidationTest(unittest.TestCase):
    """What the launcher rejects after parsing: the error names the process or the variable."""

    def test_invalid_documents_are_rejected_with_the_offending_entry(self) -> None:
        two_processes = minimal() + minimal()
        cases = {
            "no processes": ("", "has no processes"),
            "only variables": (
                'variables { name: "a" value: "1" }',
                "has no processes",
            ),
            "bad name": (
                minimal().replace('"mpc"', '"my mpc"'),
                "processes[0]: 'name' must be",
            ),
            "missing name": (
                'processes { machine: MACHINE_ROBOT command: "p" }',
                "processes[0]: 'name' must be",
            ),
            "missing machine": (
                'processes { name: "a" command: "p" }',
                "process 'a': 'machine' must be one of MACHINE_ROBOT, MACHINE_LAPTOP, got MACHINE_UNSPECIFIED",
            ),
            "machine by an unknown number": (
                minimal().replace("MACHINE_LAPTOP", "7"),
                "'machine' must be one of MACHINE_ROBOT, MACHINE_LAPTOP, got 7",
            ),
            "no command": (
                'processes { name: "mpc" machine: MACHINE_LAPTOP }',
                "'command' must list the program",
            ),
            "command as one string": (
                minimal().replace('"prog"', '"prog --flag"'),
                "give the program and each argument an entry of their own",
            ),
            "empty program": (
                minimal().replace('"prog"', '""'),
                "command[0], is empty",
            ),
            "bad env name": (
                minimal(process='env { name: "MY-VAR" value: "1" }'),
                "not an environment variable name",
            ),
            "env set twice": (
                minimal(
                    process='env { name: "A" value: "1" } env { name: "A" value: "2" }'
                ),
                "environment variable A is set twice",
            ),
            "negative delay": (
                minimal(process="delay: -1.0"),
                "'delay' must be a non-negative number of seconds",
            ),
            "nan delay": (
                minimal(process="delay: nan"),
                "'delay' must be a non-negative number of seconds",
            ),
            "duplicate names": (two_processes, "must be unique: mpc"),
            "negative grace": (
                minimal(extra="shutdown { sigint_grace_period: -1 }"),
                "shutdown: 'sigint_grace_period' must be a non-negative",
            ),
            "variable name": (
                minimal(extra='variables { name: "a-b" value: "1" }'),
                "variables[0]: name 'a-b' is not an identifier",
            ),
            "variable declared twice": (
                minimal(
                    extra='variables { name: "a" value: "1" } variables { name: "a" value: "2" }'
                ),
                "variable 'a' is declared twice",
            ),
        }
        for case, (text, expected) in cases.items():
            with self.subTest(case=case):
                with self.assertRaises(launch_file.LaunchFileError) as raised:
                    parse(text)
                self.assertIn(expected, str(raised.exception))
                self.assertTrue(
                    str(raised.exception).startswith(f"{SOURCE}: "),
                    str(raised.exception),
                )

    def test_missing_and_undecodable_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(launch_file.LaunchFileError) as raised:
                launch_file.load_launch_file(
                    os.path.join(directory, "absent.textproto")
                )
            self.assertIn("cannot read the launch file", str(raised.exception))
            binary = os.path.join(directory, "binary.textproto")
            with open(binary, "wb") as stream:
                stream.write(b"\xff\xfe\x00")
            with self.assertRaises(launch_file.LaunchFileError) as raised:
                launch_file.load_launch_file(binary)
            self.assertIn("not UTF-8", str(raised.exception))


class SelectMachineTest(unittest.TestCase):

    def test_filters_by_machine_in_file_order(self) -> None:
        launch = launch_file.load_launch_file(EXAMPLE, builtins=BUILTINS)
        selected: list[str] = [
            p.name for p in launch_file.select_machine(launch.processes, "laptop")
        ]
        self.assertEqual(selected, ["mpc", "viewer"])
        self.assertEqual(
            [p.name for p in launch_file.select_machine(launch.processes, "robot")],
            ["robot_loop"],
        )
        self.assertEqual(
            launch_file.select_machine(launch.processes, None), launch.processes
        )

    def test_an_unknown_machine_is_rejected(self) -> None:
        with self.assertRaises(launch_file.LaunchFileError):
            launch_file.select_machine((), "desktop")


if __name__ == "__main__":
    unittest.main()
