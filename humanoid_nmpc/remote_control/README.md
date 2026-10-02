# remote_control

The operator tools of the humanoid on the IPC bus (`robot_runtime/robot_ipc`, see
[`humanoid_nmpc/docs/distributed_runtime/README.md`](../docs/distributed_runtime/README.md)): the Tk GUI and the
keyboard and Xbox teleoperation publishers. They run on the laptop, under Bazel's hermetic Python 3.11 with the
`@operator_deps` packages; nothing here imports ROS.

```bash
bazel run //humanoid_nmpc/remote_control:base_velocity_controller_gui -- \
    --task_file=robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml
bazel run //humanoid_nmpc/remote_control:xbox_velocity_publisher
bazel run //humanoid_nmpc/remote_control:keyboard_velocity_publisher
```

The launch files of every robot start the GUI with that robot's files, next to the MPC node and the Rerun bridge
(`robot_models/<robot>/<package>/launch/mpc.textproto` and `dummy_sim.textproto`; `make launch-<robot>-sim`,
`-dummy-sim` and `-mpc`). It publishes the walking command at 25 Hz whenever it runs, so a second publisher of
`operator/walking_velocity_command` (a script, the keyboard teleop) competes with its sticks: close the GUI, or use its
sticks.

| Binary | Node | Flags |
|---|---|---|
| `base_velocity_controller_gui` | `operator` | `--task_file`, `--reference_file`, `--pd_gains_file`, `--urdf_file`, `--default_pelvis_height`, `--network_config`, `--ipc_node` |
| `xbox_velocity_publisher` | `teleop` | `--network_config`, `--ipc_node` |
| `keyboard_velocity_publisher` | `teleop` | `--network_config`, `--ipc_node` |

`--network_config` defaults to `config/ipc/network.textproto`. A relative path is looked up in the directory the
binary was started from and then in the checkout. The GUI edits the configuration files of the checkout, not Bazel's
runfiles copies, and it changes into the checkout so that the tabs' robot presets resolve there. A file it is not
given is looked for next to the others: `config/mpc/task.yaml`, `config/command/reference.yaml` and
`config/controller/joint_pd_gains.yaml` of one robot.

## What goes on the bus

The topics are the constants of `humanoid_nmpc/humanoid_mpc_ipc/python/humanoid_mpc_ipc/topics.py`, with the messages
of the README's topic table. `remote_control/operator_bus.py` lists the rows the GUI uses (`OPERATOR_TOPICS`), and
`test/test_operator_topics.py` checks them against the README.

| Topic | Message | What |
|---|---|---|
| `operator/walking_velocity_command` | `WalkingVelocityCommand` | the sticks and the height slider (or the Xbox controller), at 25 Hz, always: the height slider moves the gantry in simulation |
| `operator/fsm_command` | `FsmCommand` | the FSM mode selector and the gantry checkbox; `sequence` starts at the wall clock in nanoseconds and rises by one per command, so a restarted GUI still counts up |
| `operator/mpc_parameters` | `YamlDocument` | the MPC Parameters tab: the whole task.yaml with the slider values edited in, followed by contact_planning.yaml |
| `operator/pd_gains` | `YamlDocument` | the Joint PD Gains tab: the whole joint_pd_gains.yaml with the slider values edited in |
| `operator/joint_targets` | `JointTargets` | the Joint Targets tab, by joint name, in JOINT_PD only |
| `operator/dodgeball_throw` | `YamlDocument` | the Dodgeball tab's throw (`tk_app/dodgeball.py` `throw_payload`, dumped in block style) |
| `robot/fsm_state` (subscribed) | `FsmState` | the mode selector and the gantry checkbox follow it; the joysticks re-center on it (below) |

The YAML payloads are the text the tabs built for the ROS topics before; the receivers parse them with the loaders of
the files on disk. A later stage replaces them with typed messages, and the tuning tabs with ones rendered from the
proto descriptors.

The tabs know nothing of the bus: each is given a publisher (`operator_bus.TopicPublisher`, anything with
`publish(message)`) and builds its message. `OperatorGui` in `base_velocity_controller_gui.py` runs the window against
the bus: every tick, on Tk's thread, it applies the FSM states the bus's receive thread left in a mailbox, and publishes
one walking command. Tk is never called from another thread, and `Bus.publish()` never waits for the network.

The **Open Rerun viewer** button starts the Rerun bridge,
`.bazel/bin/humanoid_nmpc/humanoid_rerun_viewer/humanoid_rerun_viewer --network_config=<the GUI's> [--urdf=<--urdf_file>]`,
which opens the native viewer (without `--urdf_file` it draws no robot); the GUI stops it when it closes. It does not build it: when the binary is missing, the button says to
run `bazel build //humanoid_nmpc/humanoid_rerun_viewer`.

## The FSM state and the joysticks

The Base Controller tab follows `robot/fsm_state` (`FsmState`: `mode`, `gantry_locked`, `controller_resets`), which
the robot publishes on every change and periodically, so a GUI that starts late learns the current state.
`remote_control/fsm_state.py` reads it; the mode selector and the gantry checkbox mirror it, and a message whose mode
is not a control mode is ignored.

The virtual joysticks are re-centered (`fsm_state.should_recenter`) whenever the robot stops being walked by the MPC
without the stick having moved:

- on every transition into a passive mode (`ZERO_TORQUE`, `JOINT_PD`, `GRAVITY_COMP`, `SAFETY`);
- on every new gantry lock;
- on every controller reset the robot counts: a fall caught on the gantry, a `LOCK_GANTRY`, or the simulator
  putting the robot back in its initial state - also while the gantry was already locked in `JOINT_PD`, which changes
  neither the mode nor the gantry and is covered only by the count.

A stick left forward would otherwise keep commanding a walk, and re-entering `WB_MPC` would execute it. The height
slider is not touched: it is the gantry height while the gantry is locked. A connected Xbox controller writes its own
stick positions every cycle, so the release applies to the on-screen sticks only.

## Teleoperation

The Xbox controller (`xbox_controller_interface.py`) drives the GUI's sticks when it is connected, or publishes on its
own with `xbox_velocity_publisher`: the left stick walks, the right stick turns, RT and LT raise and lower the pelvis
height target. Nothing is published without a controller. On Linux, install the xpad driver:
https://github.com/paroj/xpad

`keyboard_velocity_publisher` reads the terminal it runs in: the arrow keys walk, `a` and `d` turn, `w` and `s` raise
and lower the pelvis height target by 1 cm per tick, and releasing every key stops the robot.

`dashboard_backend.py` holds `SimProcessManager`, which runs one simulation target (the Makefile's `launch-*-vnc`
targets, listed in its `TARGETS` table only) in a process group of its own, and `VirtualJoystick`, walking commands from
a notebook or a script, published as the `teleop` node.

## Tests

```bash
bazel test //humanoid_nmpc/remote_control/...
```

Every test file is a `py_test`. The shared helpers are in `test/operator_test_support.py`: `RecordingPublisher` is the
GUI's own `TopicPublisher` over a bus that records, so the publishing tests assert on the protobuf message that would
have gone out. `test/test_operator_bus.py` also runs the operator bus against real buses on loopback TCP, on ephemeral
ports. The tests of Tk widgets open withdrawn windows on the caller's display (`env_inherit = ["CI", "DISPLAY"]`; the
dev container's VNC session provides one). Where there is none, as in CI, they start an Xvfb of their own (the `xvfb`
package of `dependencies.txt`), which picks a free display number itself. They are skipped only where neither works,
and on CI (the `CI` variable set) they fail instead, since a skipped test passes without having run.

The GUI ends as closing its window does - the bus closed, the Rerun bridge it started stopped - on Ctrl-C, SIGTERM and
SIGHUP (`tools/launch` stops a process with SIGTERM, and with SIGHUP when its terminal closes). The bridge also gets
SIGTERM from the kernel when the GUI dies without that cleanup (`PR_SET_PDEATHSIG`), since it runs in a session of its
own that no teardown of the GUI's process group reaches.
