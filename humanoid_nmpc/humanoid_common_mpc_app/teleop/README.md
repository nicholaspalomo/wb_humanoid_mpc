# humanoid_common_mpc_app/teleop

The keyboard velocity teleoperation in C++, on the IPC bus
([`humanoid_nmpc/docs/distributed_runtime/README.md`](../../docs/distributed_runtime/README.md)): the walking command
of the pelvis typed as a line and published on `operator/walking_velocity_command` as the bus node `teleop`. For a
command held with the arrow keys, use `remote_control:keyboard_velocity_publisher` (Python).

<!-- LINT.IfChange(teleop_binary) -->
```bash
bazel run //humanoid_nmpc/humanoid_common_mpc_app/teleop:velocity_keyboard_command -- \
    --reference_file=$PWD/robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/command/reference.textproto
```
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc_app/teleop/BUILD.bazel:teleop_binary) -->

Type `v_x v_y delta_height yaw_rate` (m/s, m/s, m, rad/s; words left out are 0, words past the fourth are ignored) and
Enter. The command is clamped to the command limits of the reference file, normalized by them (the MPC scales it back
by its own copy), and the pelvis height is `nom_pos_base_rt_world_ewrt_world_z` plus the change. The limits are read again before every
command, so that a limit changed in the reference file (by hand, or saved from the tuning GUI's Command Limits tab)
applies to the next command; a file that no longer reads keeps the previous limits, with a warning. A word that is not
a number sends nothing.

The latest command is republished at the topic's 25 Hz (`VelocityCommandRepeater`), so that an MPC node that connects
later still gets it; nothing is published before the first line. The repeater holds the command in m/s, not as the
normalized message: the MPC reloads its copy of the limits when the reference file changes, so the repeater looks at the
file about once a second too and renormalizes the held command by the new limits. The robot keeps walking at the speed
the terminal showed (clamped to a lowered limit) instead of a speed nobody typed.

Do not run it next to the velocity controls of the GUI or the Xbox publisher: they publish the same topic (start the
GUI with `--walking_commands=off`). Ctrl-C, or the end of the input, ends it. Because the MPC keeps the last command it
received, it first publishes the stop command (zero velocities, the pelvis height change kept) and waits until the bus
has written it to DDS, plus 100 ms for DDS to put it on the wire before the bus deletes its writer.

<!-- LINT.IfChange(teleop_flags) -->
| Flag | Default | Meaning |
|---|---|---|
| `--reference_file` | required | the robot's `config/command/reference.textproto` (`humanoid_mpc_config.ReferenceFile`): the command limits |
| `--network_config` | empty: the shipped localhost network | the network file of the bus |
| `--ipc_node` | `teleop` | the bus node it publishes as |
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc_app/teleop/src/VelocityKeyboardCommandMain.cpp:teleop_flags) -->

The fields it reads (`keyboardCommandLimitsFromConfig()`) are the command limits the MPC scales the command back by
(`cmd_max_linvel_base_rt_world_ewrt_heading_x`, `cmd_max_linvel_base_rt_world_ewrt_heading_y`, `cmd_max_err_pos_base_rt_world_ewrt_world_z`, `cmd_max_angvel_base_rt_world_ewrt_world_z`) and
`nom_pos_base_rt_world_ewrt_world_z`; every one is required and the four limits must be positive. The file is parsed strictly, as every
configuration file ([`humanoid_mpc_config`](../../humanoid_mpc_config/README.md)): an unknown field or a value of the
wrong type is refused with its file, line and column.

`:test_keyboard_velocity_command` checks the limits of every shipped reference file, the refusals, the parsing, the
normalization, the line reader (pipes, and the stop predicate while it waits), the 25 Hz republication on a loopback
bus, the renormalization after a saved change of the limits and the stop command at the end.
