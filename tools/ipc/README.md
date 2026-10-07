# tools/ipc

Inspects the ZeroMQ bus of the distributed runtime from the command line: the topics it carries, their messages and
their rates.
The bus itself is described in [`humanoid_nmpc/docs/distributed_runtime/README.md`](../../humanoid_nmpc/docs/distributed_runtime/README.md).

```bash
bazel run //tools/ipc:ipc_tool -- list                                  # topics seen within 2 s: type, count, rate
bazel run //tools/ipc:ipc_tool -- echo mpc/status --count 1             # decode and print messages
bazel run //tools/ipc:ipc_tool -- echo mpc/policy --fields solver_status.healthy,state_trajectory.0.data
bazel run //tools/ipc:ipc_tool -- echo robot/fsm_state --format json
bazel run //tools/ipc:ipc_tool -- hz robot/mpc_observation              # rate, period jitter, payload size
bazel run //tools/ipc:ipc_tool -- echo robot/config_save_status         # the robot's answers to the GUI's saves
```

Or run the built binary directly: `.bazel/bin/tools/ipc/ipc_tool list`.

## How it listens

The tool only receives. It reads the network file (`--network_config`, default `config/ipc/network.textproto`, looked
up from the directory the tool was started in and then from the repository root) and connects one SUB socket to the
endpoint of **every** node in it, so it sees a topic whichever process publishes it, and processes may start before
or after it. The file is read with the bus library's own loader (`robot_ipc.load_network_config`, see
[`robot_runtime/robot_ipc`](../../robot_runtime/robot_ipc/README.md)), so the tool accepts exactly the files the
processes accept and reports a malformed one alike, with its line and column. The receiving half of the bus contract
is implemented on pyzmq directly, so the tool checks the documented framing rather than reusing `robot_ipc.Bus`.

Every message is three frames: the topic, the full protobuf type name and the payload. The tool imports every
generated module of `humanoid_mpc_msgs` and of `humanoid_mpc_config` (the configuration files the tuning GUI publishes
whole on `operator/mpc_parameters` and `operator/pd_gains`), so the protobuf default descriptor pool knows every
message, and decodes a payload by the type name of its second frame. A message of a type outside those packages is
described (`# type: unknown message type, N bytes`) rather than decoded.

ZeroMQ filters subscriptions by prefix; `echo` and `hz` drop messages whose topic only starts with the requested one.

## Subcommands

| Subcommand | Options | Prints |
|---|---|---|
| `list` | `--duration S` (2) | one line per topic: topic, type name(s), messages received, rate. A topic slower than `1/S` may be missed. |
| `echo <topic>` | `--count N` (0: until Ctrl-C), `--fields a.b,c`, `--format`, `--timeout S` (0: none) | each message, followed by `---` |
| `hz <topic>` | `--window N` (1000), `--report_period S` (1), `--duration S` (0: until Ctrl-C) | average rate, minimum and maximum period, jitter (standard deviation of the period), payload size and bandwidth over the last `N` messages |

<!-- LINT.IfChange(output_formats) -->
`--format text` (the default) prints each message in protobuf text format, a textproto that
`google.protobuf.text_format.Parse` reads back into the message type. Unlike `text_format.MessageToString`, it prints
every field, fields at their default value included (`healthy: false`, `time_trajectory: []`, an unset submessage as
its defaults), so that a zero reads as a zero rather than as a missing line; repeated scalars are one list
(`data: [0.0, 0.5]`), enums print by name, and a oneof member that is not set is a `# name: not set` comment.
`--format json` prints the same values as JSON, which `google.protobuf.json_format.ParseDict` reads back.

```textproto
solver_status {
  healthy: true
  consecutive_failures: 0
  ...
}
observation_time: 2.5
resets_served: 4
...
```

`--fields` takes dotted field paths; a component after a repeated field is an index, negative from the end:
`state_trajectory.-1.data`. With `--format text` the message is printed pruned to the selected fields, still a textproto
of its type, and a repeated field of which only some elements are selected carries their indices as a comment
(`state_trajectory {  # [2] of 3`). With `--format json` the selected values are printed keyed by their path.
<!-- LINT.ThenChange(//tools/ipc/message_decoding.py:output_formats) -->

Times are local receive times, so `hz` measures the stream as it arrives on this machine, link included.

Exit status: 0 on success, 1 when the expected messages did not arrive (`echo` received fewer than `--count`, or none;
`hz` fewer than two; `list` none), 2 for a usage error, an unreadable network file or a bad field path.

## Code

| File | Contents |
|---|---|
| `ipc_tool.py` | the command line (`main`, `run_list`, `run_echo`, `run_hz`) |
| `bus_listener.py` | the network file (through `robot_ipc`) and `BusListener`, the SUB socket |
| `message_decoding.py` | decoding by type name, field paths, the output formats |
| `topic_statistics.py` | `RateStatistics` for `hz`, `TopicCensus` for `list` |
| `bus_test_publisher.py` | test-only: a raw PUB socket on an ephemeral loopback port, and a network file naming it |

`bazel test //tools/ipc/...` runs the tests; the end-to-end ones publish with a raw pyzmq PUB socket in the bus's
framing on loopback.
