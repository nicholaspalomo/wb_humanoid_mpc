# tools/launch

Starts the processes of a launch file on one machine and stops them together: the launcher of the distributed runtime
([`humanoid_nmpc/docs/distributed_runtime/README.md`](../../humanoid_nmpc/docs/distributed_runtime/README.md)).

```bash
bazel run //tools/launch -- tools/launch/examples/example.launch.textproto                  # every process
bazel run //tools/launch -- tools/launch/examples/example.launch.textproto --machine laptop # one machine of the split
bazel run //tools/launch -- tools/launch/examples/example.launch.textproto --dry_run --set cycles=5
.bazel/bin/tools/launch/launch <launch file>     # the built binary (bazel build //tools/launch), e.g. on the robot
```

Launch files are textprotos, read with the Python module Bazel generates from the schema, so the launcher runs
through Bazel: `bazel run //tools/launch -- ...`, or the binary `bazel build //tools/launch` leaves in
`.bazel/bin/tools/launch/launch`. `python3 tools/launch/launch.py` does not work on its own.

## What it does

- **Repository root.** The launcher changes into the repository root and runs every process there, so the paths in
  commands are relative to it (`.bazel/bin/...`, `robot_models/...`). The root is `--repo_root`, else the workspace of
  `bazel run`, else the nearest directory above the launch file (or the working directory) that holds `MODULE.bazel`.
- **Process groups.** Every process starts in its own session and process group, with stdin from `/dev/null`. A
  Ctrl-C in the terminal therefore reaches only the launcher, which stops the processes in order, and a signal to a
  group also reaches what the process started itself.
- **Output.** Every line a process writes to stdout or stderr is printed behind its `[name]`, colored per process
  (`--color auto|always|never`; `auto` colors on a terminal unless `NO_COLOR` is set). The launcher's own messages
  carry `[launch]`. Python children get `PYTHONUNBUFFERED=1` unless their environment sets it.
- **Teardown.** On SIGINT (Ctrl-C), SIGTERM or SIGHUP, or when a `required` process exits, the launcher sends SIGINT
  to every process group, waits up to the SIGINT grace period for the groups to empty, then sends SIGTERM and waits
  the SIGTERM grace period, then SIGKILL. A SIGTERM or SIGHUP to the launcher starts at the SIGTERM stage. A second
  Ctrl-C skips to the next stage. A group counts as empty once its leader and every other member have exited, so a
  grandchild that ignores SIGINT still gets SIGTERM after its parent is gone. When every process exits on its own,
  whatever they left running in their groups is stopped the same way before the launcher exits.
- **Death signal.** Each child is started with `PR_SET_PDEATHSIG` = SIGKILL, so it dies with the launcher even when
  the launcher itself is killed with SIGKILL and cannot tear down. The kernel delivers it only to the direct child: its
  own children are left to the group signals of a normal teardown.
- **Bazel runfiles.** Under `bazel run`, the launcher's runfiles variables (`RUNFILES_DIR`, `RUNFILES_MANIFEST_FILE`,
  ...) and the runfiles entries of `PYTHONPATH` are not passed on, so a child that is a Bazel binary finds its own.
- **Exit status.** That of the first `required` process that failed (exited non-zero, `128 + N` when killed by signal
  N, 127 when its program does not exist) before the teardown began; 0 otherwise, including after Ctrl-C. A process
  that is not required may fail without changing it. 2 for an invalid launch file or command line.

## The launch file

A launch file is a textproto of `launch_proto.LaunchFile`, whose schema,
[`proto/launch_file.proto`](proto/launch_file.proto), documents every field. Start it with the header that tells
editors and formatters the schema:

<!-- LINT.IfChange(launch_file_schema) -->
```textproto
# proto-file: tools/launch/proto/launch_file.proto
# proto-message: launch_proto.LaunchFile

# Optional, repeated: a name and a string value.
variables { name: "config_dir" value: "robot_models/unitree_g1/g1_centroidal_mpc/config" }
variables { name: "task_file" value: "{config_dir}/mpc/task.textproto" }   # a variable may refer to other variables

shutdown {                     # optional; --sigint_grace_period / --sigterm_grace_period override it
  sigint_grace_period: 5.0     # seconds after SIGINT before SIGTERM (default 5)
  sigterm_grace_period: 3.0    # seconds after SIGTERM before SIGKILL (default 3)
}

processes {                    # at least one; started in this order, each after its delay
  name: "mpc"                  # unique; letters, digits, '_', '.', '-'
  machine: MACHINE_LAPTOP      # MACHINE_ROBOT | MACHINE_LAPTOP
  command: [".bazel/bin/humanoid_nmpc/...", "--task_file={task_file}"]   # program and arguments: no shell parsing
  env { name: "DISPLAY" value: ":99" }   # optional, repeated: added to the launcher's environment
  required: true               # optional (false): when it exits, everything stops
  terminal: false              # optional (false): run inside --terminal_command (keyboard teleoperation)
  delay: 0.0                   # optional (0): seconds after start-up
}
```
<!-- LINT.ThenChange(//tools/launch/proto/launch_file.proto:launch_file_schema) -->

- **Strictness.** The file is parsed with protobuf's text format (`google.protobuf.text_format.Parse`): an unknown
  field, a value of the wrong type, a field given twice or a syntax error is an error that names the file, the line
  and the column, `my.launch.textproto:5:3: Message type "launch_proto.LaunchFile.Process" has no field named
  "requried".` A typo therefore fails at start-up instead of silently launching a process that is not required. The
  launcher's own checks then name the process (`process 'mpc': 'machine' must be one of ...`) or the variable.
- **Text format.** Strings take double or single quotes, and a backslash starts an escape (`\n`, `\"`); single quotes
  save escaping the double quotes of a shell command. A repeated field is either one entry per line (`command: "sh"
  command: "-c"`) or a list (`command: ["sh", "-c"]`). `#` starts a comment. Values are typed: a variable's value and a
  command argument are strings (`value: "3"`, not `value: 3`), `required` is `true` or `false`.
- **Variables.** `{name}` in a command argument or an environment value is replaced by the variable (Python
  `str.format` syntax; write `{{` and `}}` for literal braces, e.g. `${{HOME}}` in a shell command). `{repo_root}`, the
  absolute repository root, is built in. `--set name=value` overrides a variable the file declares; a misspelled name
  is an error, as are a variable declared twice, undefined references, cycles, and anything beyond a plain `{name}`
  (`{a.b}`, `{a:>10}`).
- **Commands.** The program and each argument are entries of their own: `command: "prog --flag"` is rejected, since
  no shell splits it. Use `["sh", "-c", "..."]` for shell syntax.
<!-- LINT.IfChange(machines) -->
- **Machines.** `MACHINE_ROBOT` (the robot's realtime computer), selected with `--machine robot`, and
  `MACHINE_LAPTOP` (MPC, GUI, viewer), selected with `--machine laptop`; the `--machine` names are the enum values
  without `MACHINE_`, in lower case. Without `--machine` every process starts, for simulation on one machine.
<!-- LINT.ThenChange(//tools/launch/proto/launch_file.proto:launch_file_schema) -->
- **Terminal.** A process with `terminal: true` runs as `<--terminal_command> <command...>`, by default
  `x-terminal-emulator -e`, as the ROS 2 launch files did. Its output stays in that terminal.
<!-- LINT.IfChange(color_modes) -->
- **Colors.** `--color auto` (default), `always` or `never`.
<!-- LINT.ThenChange(//tools/launch/process_supervisor.py:color_modes) -->

[`examples/example.launch.textproto`](examples/example.launch.textproto) uses every field with placeholder shell
commands; the tests run it. The robots' own launch files are in `robot_models/<robot>/<package>/launch/`
(`robot.textproto`, `mpc.textproto`, `dummy_sim.textproto`, and `sandbox.textproto` of each description;
[`humanoid_nmpc/docs/distributed_runtime/README.md`](../../humanoid_nmpc/docs/distributed_runtime/README.md),
"Launching"), and the Makefile's `launch-<robot>-*` targets run them.

## Exporting a machine as a script

A machine without Python cannot run the launcher: the robot-runtime image ([`tools/deploy`](../deploy/README.md)) has
none. `//tools/launch:export_script` (`launch_script.py`) writes the one process of a machine as a POSIX sh script
that runs the same command, so that the launch file stays the only place the command line is written:

```bash
bazel run //tools/launch:export_script -- robot_models/drc_atlas/drc_atlas_centroidal_mpc/launch/robot.textproto \
    --machine robot --env_prefix WB_ROBOT_ --root .. --set bin_dir=bin --output drc_atlas.sh
```

- The launch file is validated as the launcher validates it, and the script runs the process in `--root` (absolute, or
  relative to the script's directory), which `{repo_root}` names.
- Every variable becomes a shell variable `<env_prefix><NAME>` that the environment may set (unset or empty: the
  file's value, after `--set`), assigned in dependency order: overriding `config_dir` moves the `task_file` written
  as `{config_dir}/mpc/task.textproto` along.
- The process's environment is exported, its delay slept, and its command `exec`ed, so the process takes the script's
  PID and receives its signals.
- A machine with more or fewer than one process, or a `terminal: true` process, is refused: several processes need a
  supervisor, which is the launcher.

`//tools/deploy` exports every robot's `robot.textproto` this way into its bundle (`launch/<robot>.sh`).

## Code

| File | Contents |
|---|---|
| `proto/launch_file.proto` | the schema, `launch_proto.LaunchFile` (`from launch_proto import launch_file_pb2`) |
| `launch.py` | the entry point; the command line itself is `launch_cli.py`, because a module named `launch` collides with ROS 2's `launch` package |
| `launch_cli.py` | options, repository root, `--dry_run`, signal forwarding |
| `launch_file.py` | parsing, validating and substituting a launch file |
| `process_supervisor.py` | `Supervisor`: starting, output, exits and the teardown |
| `launch_script.py` | `//tools/launch:export_script`: one machine's process as a POSIX sh script |

The schema is imported as `launch_proto/launch_file.proto` rather than by its path, so that its Python package is
`launch_proto` and not `launch`, which is ROS 2's.

`bazel test //tools/launch/...` runs the tests with small `sh` and Python children (`:test_launch_script`: the
exported script runs the launcher's command, takes its variables and their dependents from the environment, keeps
special characters literal, runs in its root, keeps the PID, and refuses what a script cannot do), and: the schema and the strict parsing,
substitution and validation, `--machine`, `--dry_run`, the teardown of a process tree (a child that ignores SIGINT gets
SIGTERM, one that ignores both gets SIGKILL, grandchildren die), required exits, exit codes, signal forwarding and the
death signal.
