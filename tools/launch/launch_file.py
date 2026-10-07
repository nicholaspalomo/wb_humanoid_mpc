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

"""Reads and validates a launch file: the processes of a deployment, their machines and the variables they use.

The variables are those the commands and environments of the processes are written with.

A launch file is a textproto of launch_proto.LaunchFile (proto/launch_file.proto, which documents every field):

    variables { name: "config_dir" value: "robot_models/unitree_g1/g1_centroidal_mpc/config" }
    shutdown { sigint_grace_period: 5.0 sigterm_grace_period: 3.0 }   # optional; the command line overrides it
    processes {
      name: "mpc"
      machine: MACHINE_LAPTOP                  # MACHINE_ROBOT | MACHINE_LAPTOP
      command: [".bazel/bin/humanoid_nmpc/...", "--task_file={config_dir}/mpc/task.textproto"]
      env { name: "DISPLAY" value: ":99" }
      required: true                           # when it exits, everything stops
      terminal: false                          # run inside a terminal emulator (keyboard teleoperation)
      delay: 0.0                               # seconds after start-up
    }

The file is parsed strictly with protobuf's text format: an unknown field, a value of the wrong type or a syntax
error is a LaunchFileError naming the file, the line and the column. The launcher's own checks (names, machines,
commands, durations, variables) name the process or the variable instead.

`{name}` in a command argument or an environment value is replaced by the variable `name` (str.format syntax, `{{`
and `}}` for literal braces). A variable may refer to other variables. `{repo_root}` is the repository root; command
paths are relative to it, because the launcher runs every process there. `--set name=value` overrides a variable the
file declares.
"""

from collections.abc import Iterable, Mapping, Sequence
import dataclasses
import re
import string

from google.protobuf import text_format
from launch_proto import launch_file_pb2

_MACHINE_ENUM = launch_file_pb2.LaunchFile.Machine
_MACHINE_VALUE_PREFIX = "MACHINE_"
# The machine of every value of LaunchFile.Machine but MACHINE_UNSPECIFIED, by number: MACHINE_ROBOT is "robot".
_MACHINE_BY_NUMBER: dict[int, str] = {
    value.number: value.name[len(_MACHINE_VALUE_PREFIX) :].lower()
    for value in _MACHINE_ENUM.DESCRIPTOR.values
    if value.number != launch_file_pb2.LaunchFile.MACHINE_UNSPECIFIED
}
# The names --machine takes, in the order of the schema.
MACHINES: tuple[str, ...] = tuple(_MACHINE_BY_NUMBER.values())

# Variables every launch file has; a file or the command line may not redefine them.
BUILTIN_VARIABLES = ("repo_root",)

DEFAULT_SIGINT_GRACE_PERIOD_S = 5.0
DEFAULT_SIGTERM_GRACE_PERIOD_S = 3.0

_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
_ENV_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_YAML_SUFFIXES = (".yaml", ".yml")


class LaunchFileError(ValueError):
    """The launch file, or a --set override, is not valid. The message names the file and the entry."""


@dataclasses.dataclass(frozen=True)
class ProcessSpec:
    """One process of a launch file, with every variable substituted. Its fields are those of LaunchFile.Process."""

    name: str
    machine: str
    command: tuple[str, ...]
    env: tuple[tuple[str, str], ...] = ()
    required: bool = False
    terminal: bool = False
    delay: float = 0.0


@dataclasses.dataclass(frozen=True)
class ShutdownPolicy:
    """How long each teardown stage waits before the next, stronger signal.

    Its fields are those of LaunchFile.Shutdown.
    """

    sigint_grace_period: float = DEFAULT_SIGINT_GRACE_PERIOD_S
    sigterm_grace_period: float = DEFAULT_SIGTERM_GRACE_PERIOD_S


@dataclasses.dataclass(frozen=True)
class LaunchFile:
    """A launch file after validation and substitution."""

    source: str
    processes: tuple[ProcessSpec, ...]
    variables: Mapping[str, str]
    shutdown: ShutdownPolicy


def parse_overrides(assignments: Iterable[str]) -> dict[str, str]:
    """`--set name=value` arguments as a mapping; the value may be empty and may contain `=`."""
    overrides: dict[str, str] = {}
    for assignment in assignments:
        name, separator, value = assignment.partition("=")
        if not separator or not name.isidentifier():
            raise LaunchFileError(
                f"--set {assignment!r}: expected name=value with an identifier as name"
            )
        overrides[name] = value
    return overrides


def _template_fields(template: str, context: str) -> list[str]:
    """The variable names a template refers to, in order; rejects what str.format would do beyond substitution."""
    try:
        parsed = list(string.Formatter().parse(template))
    except ValueError as error:
        raise LaunchFileError(
            f"{context}: {error} in {template!r} (write {{{{ and }}}} for literal braces)"
        ) from error
    names = []
    for _, field_name, format_spec, conversion in parsed:
        if field_name is None:
            continue
        if not field_name.isidentifier() or format_spec or conversion:
            raise LaunchFileError(
                f"{context}: '{{{field_name}{'!' + conversion if conversion else ''}"
                f"{':' + format_spec if format_spec else ''}}}' in {template!r} is not a variable reference; "
                "write {name}, or {{ and }} for literal braces"
            )
        names.append(field_name)
    return names


def substitute(template: str, variables: Mapping[str, str], context: str) -> str:
    """Replaces every `{name}` of a template by its variable."""
    for name in _template_fields(template, context):
        if name not in variables:
            known = ", ".join(sorted(variables)) or "none"
            raise LaunchFileError(
                f"{context}: undefined variable '{{{name}}}' in {template!r}; the variables are: {known}"
            )
    return template.format_map(variables)


def resolve_variables(
    declared: Mapping[str, str],
    overrides: Mapping[str, str],
    builtins: Mapping[str, str],
    source: str,
) -> dict[str, str]:
    """The value of every variable, after the overrides, with references to other variables substituted."""
    for name in builtins:
        if name in declared or name in overrides:
            raise LaunchFileError(
                f"{source}: '{name}' is a built-in variable and cannot be redefined"
            )
    unknown = sorted(set(overrides) - set(declared))
    if unknown:
        known = ", ".join(sorted(declared)) or "none"
        raise LaunchFileError(
            f"{source}: --set of undeclared variable(s) {', '.join(unknown)}; the launch file declares: {known}"
        )
    raw = {**declared, **overrides}
    resolved: dict[str, str] = dict(builtins)

    def resolve(name: str, chain: tuple[str, ...]) -> str:
        if name in resolved:
            return resolved[name]
        if name in chain:
            cycle = " -> ".join(chain[chain.index(name) :] + (name,))
            raise LaunchFileError(
                f"{source}: variables refer to each other in a cycle: {cycle}"
            )
        if name not in raw:
            known = ", ".join(sorted(set(raw) | set(builtins)))
            raise LaunchFileError(
                f"{source}: variable '{chain[-1]}' refers to undefined variable '{{{name}}}'; the variables are: "
                f"{known}"
            )
        context = f"{source}: variable '{name}'"
        values = {
            reference: resolve(reference, chain + (name,))
            for reference in _template_fields(raw[name], context)
        }
        resolved[name] = substitute(raw[name], values, context)
        return resolved[name]

    for name in raw:
        resolve(name, ())
    return resolved


def _seconds(value: float, context: str) -> float:
    # `not >=` also rejects nan, which the text format accepts for a double.
    if not value >= 0.0:
        raise LaunchFileError(
            f"{context} must be a non-negative number of seconds, got {value!r}"
        )
    return float(value)


def _machine(entry: launch_file_pb2.LaunchFile.Process, context: str) -> str:
    """The machine of a process, as --machine names it.

    The text format rejects an unknown value name, but takes any number for this open enum, and an unset field reads
    as MACHINE_UNSPECIFIED: both are errors here.

    Args:
        entry: The process.
        context: Names the process in the error message.

    Returns:
        A name of MACHINES.

    Raises:
        LaunchFileError: The machine is unset or not a value of LaunchFile.Machine.
    """
    machine = _MACHINE_BY_NUMBER.get(entry.machine)
    if machine is None:
        expected = ", ".join(
            _MACHINE_ENUM.Name(number) for number in _MACHINE_BY_NUMBER
        )
        got = (
            _MACHINE_ENUM.Name(entry.machine)
            if entry.machine in _MACHINE_ENUM.values()
            else str(entry.machine)
        )
        raise LaunchFileError(
            f"{context}: 'machine' must be one of {expected}, got {got}"
        )
    return machine


def _parse_process(
    entry: launch_file_pb2.LaunchFile.Process,
    index: int,
    variables: Mapping[str, str],
    source: str,
) -> ProcessSpec:
    """The process `entry`, the `index`th of the file, validated and with `variables` substituted."""
    context = f"{source}: processes[{index}]"
    if not _NAME_PATTERN.match(entry.name):
        raise LaunchFileError(
            f"{context}: 'name' must be letters, digits, '_', '.' or '-', got {entry.name!r}"
        )
    context = f"{source}: process '{entry.name}'"
    machine = _machine(entry, context)

    if not entry.command:
        raise LaunchFileError(
            f"{context}: 'command' must list the program and its arguments, one entry each (no shell parsing is "
            "done)"
        )
    # Checked before substitution, so that a {repo_root} with a space in it is still a valid program.
    if any(character.isspace() for character in entry.command[0]):
        raise LaunchFileError(
            f"{context}: the program, command[0], is {entry.command[0]!r}; give the program and each argument an "
            'entry of their own, command: ["prog", "--flag"] (no shell parsing is done)'
        )
    arguments = tuple(
        substitute(argument, variables, f"{context}: command[{i}]")
        for i, argument in enumerate(entry.command)
    )
    if not arguments[0]:
        raise LaunchFileError(f"{context}: the program, command[0], is empty")

    env: list[tuple[str, str]] = []
    for variable in entry.env:
        if not _ENV_NAME_PATTERN.match(variable.name):
            raise LaunchFileError(
                f"{context}: {variable.name!r} is not an environment variable name"
            )
        if any(name == variable.name for name, _ in env):
            raise LaunchFileError(
                f"{context}: environment variable {variable.name} is set twice"
            )
        env_context = f"{context}: env {variable.name}"
        env.append((variable.name, substitute(variable.value, variables, env_context)))

    return ProcessSpec(
        name=entry.name,
        machine=machine,
        command=arguments,
        env=tuple(env),
        required=entry.required,
        terminal=entry.terminal,
        delay=_seconds(entry.delay, f"{context}: 'delay'"),
    )


def _parse_shutdown(
    entry: launch_file_pb2.LaunchFile.Shutdown, source: str
) -> ShutdownPolicy:
    """The grace periods the file sets, and the defaults for those it leaves unset.

    An explicit 0 is kept: the fields track presence.

    Args:
        entry: The shutdown entry of the file (unset: every default).
        source: Names the file in the error messages.

    Returns:
        The policy.

    Raises:
        LaunchFileError: A grace period is negative or NaN.
    """
    defaults = ShutdownPolicy()
    periods: dict[str, float] = {}
    for policy_field in dataclasses.fields(ShutdownPolicy):
        name = policy_field.name
        periods[name] = (
            _seconds(getattr(entry, name), f"{source}: shutdown: '{name}'")
            if entry.HasField(name)
            else getattr(defaults, name)
        )
    return ShutdownPolicy(**periods)


def parse_launch_file(
    document: launch_file_pb2.LaunchFile,
    source: str,
    overrides: Mapping[str, str] | None = None,
    builtins: Mapping[str, str] | None = None,
) -> LaunchFile:
    """A parsed launch file, validated and with every variable substituted."""
    builtins = dict(builtins or {})
    for name in BUILTIN_VARIABLES:
        builtins.setdefault(name, "")

    declared: dict[str, str] = {}
    for index, variable in enumerate(document.variables):
        if not variable.name.isidentifier():
            raise LaunchFileError(
                f"{source}: variables[{index}]: name {variable.name!r} is not an identifier"
            )
        if variable.name in declared:
            raise LaunchFileError(
                f"{source}: variable '{variable.name}' is declared twice"
            )
        declared[variable.name] = variable.value
    variables = resolve_variables(declared, overrides or {}, builtins, source)

    if not document.processes:
        raise LaunchFileError(
            f"{source}: the launch file has no processes; add at least one 'processes {{ ... }}' entry"
        )
    processes = tuple(
        _parse_process(entry, index, variables, source)
        for index, entry in enumerate(document.processes)
    )
    names = [process.name for process in processes]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise LaunchFileError(
            f"{source}: process names must be unique: {', '.join(duplicates)}"
        )

    return LaunchFile(
        source=source,
        processes=processes,
        variables=variables,
        shutdown=_parse_shutdown(document.shutdown, source),
    )


def _parse_error_text(error: text_format.ParseError, source: str) -> str:
    """`<source>:<line>:<column>: <what is wrong>` for an error of the text format parser."""
    text = str(error)
    line = error.GetLine()
    if line is None:
        message = f"{source}: {text}"
    else:
        column = error.GetColumn()
        location = f"{line}" if column is None else f"{line}:{column}"
        # The parser puts "<line>:<column> : " in front of its message.
        prefix = f"{location} : "
        if text.startswith(prefix):
            text = text[len(prefix) :]
        message = f"{source}:{location}: {text}"
    if source.endswith(_YAML_SUFFIXES):
        message += " (launch files are textprotos of launch_proto.LaunchFile, not YAML: see tools/launch/README.md)"
    return message


def read_launch_text(text: str, source: str) -> launch_file_pb2.LaunchFile:
    """A launch file's text parsed strictly as a textproto of launch_proto.LaunchFile, not yet validated."""
    document = launch_file_pb2.LaunchFile()
    try:
        text_format.Parse(text, document)
    except text_format.ParseError as error:
        raise LaunchFileError(_parse_error_text(error, source)) from error
    return document


def parse_launch_text(
    text: str,
    source: str,
    overrides: Mapping[str, str] | None = None,
    builtins: Mapping[str, str] | None = None,
) -> LaunchFile:
    """A launch file's text, parsed, validated and with every variable substituted. `source` names it in errors."""
    return parse_launch_file(
        read_launch_text(text, source), source, overrides, builtins
    )


def load_launch_file(
    path: str,
    overrides: Mapping[str, str] | None = None,
    builtins: Mapping[str, str] | None = None,
) -> LaunchFile:
    """The launch file at `path`, validated and with every variable substituted."""
    try:
        with open(path, "r", encoding="utf-8") as stream:
            text = stream.read()
    except OSError as error:
        raise LaunchFileError(
            f"cannot read the launch file {path}: {error.strerror}"
        ) from error
    except UnicodeDecodeError as error:
        raise LaunchFileError(f"{path} is not UTF-8 text: {error}") from error
    return parse_launch_text(text, path, overrides, builtins)


def select_machine(
    processes: Sequence[ProcessSpec], machine: str | None
) -> tuple[ProcessSpec, ...]:
    """The processes of one machine, in file order; all of them when `machine` is None."""
    if machine is not None and machine not in MACHINES:
        raise LaunchFileError(
            f"unknown machine {machine!r}; the machines are: {', '.join(MACHINES)}"
        )
    return tuple(
        process
        for process in processes
        if machine is None or process.machine == machine
    )
