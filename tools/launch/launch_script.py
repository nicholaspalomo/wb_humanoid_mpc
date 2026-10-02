"""Exports the process of one machine of a launch file as a POSIX sh script, for a machine that cannot run the
launcher.

    bazel run //tools/launch:export_script -- <launch file> --machine robot --output robot.sh \
        [--set name=value ...] [--env_prefix WB_ROBOT_] [--root ..]

The robot-runtime image (docker/Dockerfile) carries no Python, so it cannot run the launcher. Its entry point runs the
script exported from the robot's `launch/robot.textproto` instead, so the launch file stays the one place the robot's
command line is written.

The script

- changes into the root the launch file's paths are relative to: `--root`, an absolute directory or one relative to
  the script's own directory (the launcher's repository root);
- gives every variable of the launch file a shell variable `<env_prefix><NAME>` (`task_file` is `WB_ROBOT_TASK_FILE`)
  that the environment may set (unset or empty: the file's value, after `--set`), assigned in dependency order, so a
  variable that refers to another follows that one's value. `{repo_root}` is the root;
- exports the process's environment variables, sleeps its delay, and `exec`s its command, so the process takes the
  script's place and receives its signals.

Only a machine with exactly one process can be exported: several need a supervisor, which is the launcher. A process
with `terminal: true` is refused, because there is no terminal to open.
"""

import argparse
import os
import re
import shlex
import string
import sys
from typing import Dict, List, Mapping, Optional, Sequence, TextIO, Tuple

import launch_file
from launch_proto import launch_file_pb2

EXIT_USAGE = 2
DEFAULT_ENV_PREFIX = "LAUNCH_"
# The shell variable holding the root; {repo_root} refers to it. Lower case, so that no <prefix><NAME> collides.
ROOT_SHELL_VARIABLE = "launch_root"
_ENV_PREFIX_PATTERN = re.compile(r"^[A-Z_][A-Z0-9_]*$")


def shell_variable(env_prefix: str, name: str) -> str:
    """The environment variable of launch-file variable `name`: `WB_ROBOT_` and `task_file` give WB_ROBOT_TASK_FILE."""
    if name in launch_file.BUILTIN_VARIABLES:
        return ROOT_SHELL_VARIABLE
    return env_prefix + name.upper()


def _pieces(template: str) -> List[Tuple[str, Optional[str]]]:
    """A template as (literal text, variable name or None) pairs; `{{` and `}}` are already single braces."""
    return [
        (literal, field_name)
        for literal, field_name, _, _ in string.Formatter().parse(template)
    ]


def _double_quoted(text: str) -> str:
    """`text` escaped for the inside of a double-quoted shell word."""
    return re.sub(r'([\\"$`])', r"\\\1", text)


def template_as_double_quoted(template: str, env_prefix: str) -> str:
    """A template as the inside of a double-quoted shell word, `{name}` becoming `${<PREFIX>NAME}`."""
    parts = []
    for literal, name in _pieces(template):
        parts.append(_double_quoted(literal))
        if name is not None:
            parts.append("${%s}" % shell_variable(env_prefix, name))
    return "".join(parts)


def template_as_word(template: str, env_prefix: str) -> str:
    """A template as one shell word: literal text single-quoted, `{name}` as `"${<PREFIX>NAME}"`."""
    parts = []
    for literal, name in _pieces(template):
        if literal:
            parts.append(shlex.quote(literal))
        if name is not None:
            parts.append('"${%s}"' % shell_variable(env_prefix, name))
    return "".join(parts) or "''"


def _references(template: str) -> List[str]:
    return [name for _, name in _pieces(template) if name is not None]


def _dependency_order(templates: Mapping[str, str]) -> List[str]:
    """The variables, every one after those it refers to; file order otherwise. load_launch_file() has already
    rejected cycles and undefined references."""
    order: List[str] = []
    placed = set()

    def place(name: str) -> None:
        if name in placed or name not in templates:
            return
        placed.add(name)
        for reference in _references(templates[name]):
            place(reference)
        order.append(name)

    for name in templates:
        place(name)
    return order


def _root_assignment(root: str) -> str:
    if os.path.isabs(root):
        return f"{ROOT_SHELL_VARIABLE}={shlex.quote(os.path.normpath(root))}"
    return f'{ROOT_SHELL_VARIABLE}="$(cd "$(dirname "$0")/"{shlex.quote(root)} && pwd)"'


def export_script(
    path: str,
    machine: str,
    overrides: Optional[Mapping[str, str]] = None,
    env_prefix: str = DEFAULT_ENV_PREFIX,
    root: str = ".",
    source_name: Optional[str] = None,
) -> str:
    """The POSIX sh script of the one process `machine` runs in the launch file at `path`.

    Args:
        path: the launch file.
        machine: one of launch_file.MACHINES.
        overrides: `--set` values, which become the script's defaults.
        env_prefix: the prefix of the shell variables, upper case.
        root: the directory the script changes into: absolute, or relative to the script's directory.
        source_name: how the script's header names the launch file (default: `path`).

    Raises:
        launch_file.LaunchFileError: the launch file is invalid (as the launcher reports it), the machine does not run
            exactly one process, the process wants a terminal, or two variables share a shell variable.
    """
    overrides = dict(overrides or {})
    if not _ENV_PREFIX_PATTERN.match(env_prefix):
        raise launch_file.LaunchFileError(
            f"--env_prefix {env_prefix!r} must be upper-case letters, digits and '_'"
        )
    source = source_name or path
    # The launcher's own validation first, so that the script exists only for a file the launcher would run.
    launch = launch_file.load_launch_file(
        path, overrides=overrides, builtins={"repo_root": root}
    )
    selected = launch_file.select_machine(launch.processes, machine)
    if len(selected) != 1:
        names = ", ".join(process.name for process in selected) or "none"
        raise launch_file.LaunchFileError(
            f"{source}: machine '{machine}' runs {len(selected)} processes ({names}); a script runs exactly one, "
            "several need the launcher"
        )
    process = selected[0]
    if process.terminal:
        raise launch_file.LaunchFileError(
            f"{source}: process '{process.name}' runs in a terminal (terminal: true), which a script cannot open"
        )

    with open(path, "r", encoding="utf-8") as stream:
        document = launch_file.read_launch_text(stream.read(), source)
    entry: launch_file_pb2.LaunchFile.Process = next(
        candidate for candidate in document.processes if candidate.name == process.name
    )
    templates: Dict[str, str] = {
        variable.name: variable.value for variable in document.variables
    }
    templates.update(overrides)

    shell_names: Dict[str, str] = {}
    for name in templates:
        shell_name = shell_variable(env_prefix, name)
        if shell_name in shell_names:
            raise launch_file.LaunchFileError(
                f"{source}: variables '{shell_names[shell_name]}' and '{name}' would both be {shell_name}"
            )
        shell_names[shell_name] = name

    lines = [
        "#!/bin/sh",
        f"# Generated by //tools/launch:export_script from {source} (--machine {machine}, process"
        f" '{process.name}'). Do not edit it: change the launch file.",
        "#",
        "# Every variable of the launch file is a shell variable here, which the environment may set (unset or empty:",
        "# the value below):",
    ]
    width = max(
        (len(shell_variable(env_prefix, name)) for name in templates), default=0
    )
    for name in templates:
        lines.append(
            f"#   {shell_variable(env_prefix, name):<{width}}  {name} = {launch.variables[name]}"
        )
    lines += [
        "set -eu",
        "",
        "# The root the paths of the launch file are relative to ({repo_root}).",
        _root_assignment(root),
        f'cd "${{{ROOT_SHELL_VARIABLE}}}"',
        "",
    ]
    for name in _dependency_order(templates):
        variable = shell_variable(env_prefix, name)
        lines.append(
            f'[ -n "${{{variable}:-}}" ] || '
            f'{variable}="{template_as_double_quoted(templates[name], env_prefix)}"'
        )
    if entry.env:
        lines.append("")
        for variable in entry.env:
            lines.append(
                f'export {variable.name}="{template_as_double_quoted(variable.value, env_prefix)}"'
            )
    if process.delay > 0.0:
        lines += ["", f"sleep {process.delay:g}"]
    words = [template_as_word(argument, env_prefix) for argument in entry.command]
    lines += ["", "exec " + " ".join(words), ""]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="export_script",
        description="Export the one process of a machine of a launch file as a POSIX sh script.",
    )
    parser.add_argument(
        "launch_file", help="the launch file, a textproto of launch_proto.LaunchFile"
    )
    parser.add_argument(
        "--machine",
        required=True,
        choices=launch_file.MACHINES,
        help="the machine whose process the script runs",
    )
    parser.add_argument(
        "--output", required=True, help="the script to write (made executable)"
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="change a variable's default in the script; repeatable",
    )
    parser.add_argument(
        "--env_prefix",
        default=DEFAULT_ENV_PREFIX,
        help="prefix of the shell variables (default: %(default)s)",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="the directory the script runs the process in: absolute, or relative to the script's directory "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--source_name",
        default=None,
        help="how the script's header names the launch file (default: its path)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None, err: TextIO = sys.stderr) -> int:
    args = build_parser().parse_args(argv)
    try:
        script = export_script(
            args.launch_file,
            args.machine,
            overrides=launch_file.parse_overrides(args.overrides),
            env_prefix=args.env_prefix,
            root=args.root,
            source_name=args.source_name,
        )
    except launch_file.LaunchFileError as error:
        err.write(f"export_script: {error}\n")
        return EXIT_USAGE
    with open(args.output, "w", encoding="utf-8") as stream:
        stream.write(script)
    os.chmod(args.output, 0o755)
    return 0


if __name__ == "__main__":
    sys.exit(main())
