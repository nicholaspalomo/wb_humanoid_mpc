# remote_control

The operator tools of the humanoid on the IPC bus (`robot_runtime/robot_ipc`, see
[`humanoid_nmpc/docs/distributed_runtime/README.md`](../docs/distributed_runtime/README.md)): the Tk GUI and the
keyboard and Xbox teleoperation publishers. They run on the laptop, under Bazel's hermetic Python 3.11 with the
`@operator_deps` packages; nothing here imports ROS.

```bash
bazel run //humanoid_nmpc/remote_control:base_velocity_controller_gui -- \
    --task_file=robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.textproto
bazel run //humanoid_nmpc/remote_control:xbox_velocity_publisher
bazel run //humanoid_nmpc/remote_control:keyboard_velocity_publisher
bazel run //humanoid_nmpc/remote_control:push_robot_config -- \
    robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto
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
| `push_robot_config` | `config_push` | the file; `--network_config`, `--ipc_node` |

`--network_config` defaults to `config/ipc/network.textproto`. A relative path is looked up in the directory the
binary was started from and then in the checkout. The GUI edits the configuration files of the checkout, not Bazel's
runfiles copies, and it changes into the checkout so that the tabs' robot presets resolve there. A file it is not
given is looked for next to the others: `config/mpc/task.textproto`, `config/command/reference.textproto` and
`config/controller/joint_pd_gains.textproto` of one robot, and the contact planner's `config/mpc/contact_planning.textproto`
next to the task file.

## What goes on the bus

The topics are the constants of `humanoid_nmpc/humanoid_mpc_ipc/python/humanoid_mpc_ipc/topics.py`, with the messages
of the README's topic table. `remote_control/operator_bus.py` lists the rows the GUI uses (`OPERATOR_TOPICS`), and
`test/test_operator_topics.py` checks them against the README.

| Topic | Message | What |
|---|---|---|
| `operator/walking_velocity_command` | `WalkingVelocityCommand` | the sticks and the height slider (or the Xbox controller), at 25 Hz, always: the height slider moves the gantry in simulation |
| `operator/fsm_command` | `FsmCommand` | the FSM mode selector and the gantry checkbox; `sequence` starts at the wall clock in nanoseconds and rises by one per command, so a restarted GUI still counts up |
| `operator/mpc_parameters` | `humanoid_mpc_config.MpcParameterUpdate` | the MPC Parameters tab: the whole task file and, for a robot with a contact planner, the whole contact-planning file, with the operator's values, and the task file's identity (`config_path`) |
| `operator/pd_gains` | `humanoid_mpc_config.JointPdGainsFile` | the Joint PD Gains tab: the whole PD gains file with the operator's values |
| `operator/joint_targets` | `JointTargets` | the Joint Targets tab, by joint name, in JOINT_PD only |
| `operator/dodgeball_throw` | `DodgeballThrow` | the Dodgeball tab's throw (`tk_app/dodgeball.py` `throw_message`) |
| `operator/config_save` | `ConfigFileSave` | a tuning tab's Save: the exact text of the saved task, reference or PD gains file, for the robot's store (below); also `push_robot_config` |
| `robot/fsm_state` (subscribed) | `FsmState` | the mode selector and the gantry checkbox follow it; the joysticks re-center on it (below) |
| `robot/config_save_status` (subscribed) | `ConfigFileSaveStatus` | the robot's answer to each save, which the tab's status line shows |

The tuning payloads are the files, typed (humanoid_nmpc/humanoid_mpc_config/README.md, "Live updates"): a block the
update does not carry means its defaults, as at start-up, so the tabs publish the whole of each file. A receiver built
from another commit that still subscribes with another type drops the message, and the bus logs the mismatch; one with
the same type refuses a payload of another schema version (a field it does not know, or, for the MPC parameters,
another `schema_fingerprint`, which `operator_bus.mpc_parameter_update()` stamps) and an MPC parameter update of
another robot or of another configuration (its `config_path`, the task file's path from `robot_models/` on: the
centroidal and the whole-body G1 share the robot name "g1"), and logs and counts it
(humanoid_nmpc/humanoid_mpc_config/README.md, "Version skew").

The tabs know nothing of the bus: each is given a publisher (`operator_bus.TopicPublisher`, anything with
`publish(message)`) and builds its message. `OperatorGui` in `base_velocity_controller_gui.py` runs the window against
the bus: every tick, on Tk's thread, it applies the FSM states the bus's receive thread left in a mailbox, and publishes
one walking command. Tk is never called from another thread, and `Bus.publish()` never waits for the network.

The **Open Rerun viewer** button starts the Rerun bridge,
`.bazel/bin/humanoid_nmpc/humanoid_rerun_viewer/humanoid_rerun_viewer --network_config=<the GUI's> [--urdf=<--urdf_file>]`,
which opens the native viewer (without `--urdf_file` it draws no robot); the GUI stops it when it closes. It does not build it: when the binary is missing, the button says to
run `bazel build //humanoid_nmpc/humanoid_rerun_viewer`.

## Tuning the configuration

### The files

The tabs tune the textprotos of a robot package's `config/` directory. Each is parsed strictly into a message of
`humanoid_nmpc/humanoid_mpc_config` (its README holds the conventions; the field comments of the `.proto` files say what
each value means) and starts with the header that names it, `# proto-file:` and `# proto-message:`:

| File | Message | Tab | Read back while the stack runs by |
|---|---|---|---|
| `config/mpc/task.textproto` | `TaskFile` | MPC Parameters | the parameter updater of the running MPC, centroidal or whole-body; the robot process (the contact estimator and the contact wrench gate), from its stored copy |
| `config/mpc/contact_planning.textproto` (Atlas, SA01) | `ContactPlanningFile` | MPC Parameters, block `contact_planning` | the centroidal MPC's parameter updater; the MPC's alone, never sent to the robot |
| `config/command/reference.textproto` | `ReferenceFile` | Command Limits; Joint Targets reads its `default_joint_state` | the parameter updater of the running MPC (the command limits); the robot process at its next start, from its stored copy |
| `config/controller/joint_pd_gains.textproto` | `JointPdGainsFile` | Joint PD Gains; Joint Targets takes its joints | the robot's MRT joint controller, from its stored copy |

The shared gait file, `humanoid_nmpc/humanoid_common_mpc/config/command/gait.textproto` (`GaitFile`), is read at start-up
only and has no tab.

### Tuning a running robot

1. Start the stack with its launch file (`make launch-<robot>-sim`), which opens the GUI on that robot's files.
2. Move a slider, tick a checkbox or pick a name. On the MPC Parameters and Joint PD Gains tabs, 300 ms after the last
   change, the tab publishes the whole edited file (the topics above); nothing is written to disk yet. The running
   stack applies the fields marked `RELOAD_HOT` at once, on either formulation: the parameter updater of the
   centroidal and of the whole-body MPC applies them before its next solve, and the robot process applies the contact
   estimator, the contact wrench gate and the PD gains. A field marked `RELOAD_START_UP` is labeled "restart": the
   running stack reports that it differs and keeps its value until the next start. A field the file's formulation
   does not read (its schema option `formulations`) is labeled "not applicable: the whole-body MPC does not read it"
   (or the centroidal MPC).
3. **Save** writes the changes into the laptop's file and sends the same text to the robot's persistent store (below).
   The MPC watches the laptop's files and the robot its stored copies (roughly once a second), so a saved value
   reaches a stack that missed the publish too, and the next start reads it. The Command Limits tab publishes nothing:
   saving is how its changes reach the MPC. **Reset All** returns every value to the file as loaded or last saved
   (and a publishing tab publishes that).

A value the schema refuses is never sent: every publish re-parses the edited text strictly, and a refusal is shown on
the tab's status line. A file that does not parse (an unknown field, a retired key, a value of the wrong type) is
reported with its line and column, and its tab stays empty until the file is fixed and reloaded.

The files can also be edited by hand, in any editor: the stack reads them as the GUI writes them, and a file the
stack cannot parse is refused with its file, line and column. An editor's save reaches the laptop's MPC through its
file watcher; it reaches the robot's copy at the robot's next start (from the deployed bundle, by its seed policy) or
at once with `push_robot_config` (below). `bazel test //humanoid_nmpc/remote_control:test_config_schema` parses every
shipped file strictly.

### Two copies: the laptop's and the robot's

The MPC runs on the laptop and reads the laptop's files; the robot process reads its own copies, from a persistent
store on the robot (`/var/lib/wb-humanoid-robot/config/<robot>/` in its container, seeded from the deployed bundle;
humanoid_nmpc/docs/distributed_runtime/README.md, "Saving the configuration"). So **Save** keeps two copies equal:

1. It writes the laptop's file (below). When that fails, the status line says why and nothing is sent.
2. It sends exactly the text it wrote (`TunedFile.save()` returns it) as a `ConfigFileSave` on `operator/config_save`:
   the file's kind (task, reference, PD gains), the robot name of its configuration's task file, the file's identity
   (`config_path`, its path from `robot_models/` on, `robot_config_save.config_path_of()`), the fingerprint of the
   file's schema and a sequence (the wall clock in nanoseconds). The contact-planning file is the MPC's and stays on
   the laptop. A Save without changes sends the file too, which re-synchronizes a robot that refused an earlier save
   or missed an editor's.
3. The robot checks it as it checks the file at start-up, stores it atomically and answers on
   `robot/config_save_status`. The tab polls the answer every 200 ms and shows it:

| Status line | Meaning |
|---|---|
| Saved on the laptop · saving on the robot… | sent; no answer yet |
| Saved on the laptop and on the robot (`<stored path>`) | both copies hold the text |
| Saved on the laptop and on the robot (`<stored path>`) … robot unchanged | the robot's copy was the same text already |
| Saved on the laptop · the robot refused it: `<message>` | another robot, configuration or schema version, or a file the robot would not start with: its copy is untouched |
| Saved on the laptop · the robot could not store it: `<message>` | a write failed: its copy is the one it had |
| Saved on the laptop · the robot did not answer in 5 s: the robot's copy is unknown; Save again to retry (a repeat answers unchanged) | no answer yet; a late answer still replaces the line |
| Saved on the laptop · not sent to the robot: `<reason>` | the bus dropped it (its send queue was full), the GUI has no bus, or no task file beside the file names the robot |
| Saved on the laptop · the robot has no store and reads this file (`<path>`) | a robot process started without a store on the same checkout (`bazel run` in the dev container) reads the laptop's file itself |
| Saved on the laptop · the robot has no store: it reads `<path>` in place; this Save did not change it | a robot process without a store that reads a file of its own (another computer, another checkout): its copy differs, shown as a problem |

`remote_control/robot_config_save.py` holds the saver (`RobotConfigSaver`, one for the GUI, matched by sequence) and
the status lines, without Tk; `tk_app/robot_save_status.py` polls it for a tab.

`push_robot_config` sends a file the same way from the command line - after an editor's save, or as a scripted check
of the two-process simulation - and prints the robot's answer:

```bash
bazel run //humanoid_nmpc/remote_control:push_robot_config -- robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto
```

It tells the kind by the file's `# proto-message:` header, parses the file strictly first, and publishes as the
`config_push` node of the network file (a running GUI holds `operator`). The first messages of a freshly bound socket
can be lost before the robot's subscriber connects, so it repeats the same save, under the same sequence, every 0.5 s
until the robot answers that sequence or 10 s pass; the robot answers a repeat with the status it recorded. It exits 0
(`EXIT_IN_SYNC`) when the robot's copy is the file - stored, already the same, or the very file a robot without a store
reads in place - 1 (`EXIT_NOT_SAVED`) when the robot refused it, could not store it or did not answer, 2
(`EXIT_CANNOT_SEND`) when the file, the network file or the bus do not let it send, and 3 (`EXIT_NOT_STORED`) when a
robot without a store reads another file, which the push did not change. A deployment's own
network file needs the `config_push` node only for this tool, not for the GUI's saves.

Anything that can publish on the bus can now replace the robot's three configuration files, as it can already command
the FSM or change the PD gains: the bus has no authentication and must run on the robot's private network. The
identity and the robot name guard against a GUI opened on another configuration, not against an attacker
(humanoid_nmpc/docs/distributed_runtime/README.md, "Saving the configuration", for the trust model and what the robot
checks).

### How the GUI saves

**Save** replaces only the value tokens of the changed fields, and inserts a field the file leaves out at the end of
its block (`humanoid_mpc_config/python/config_textproto/textproto_save.py`). Every other byte - comments, blank lines,
LINT directives, the spelling of the values not changed - is kept. Before writing, the result is parsed strictly again
and compared with the edited message; on any difference nothing is written. The file is re-read for the save, so an
edit made by hand since it was loaded is kept unless it is to one of the changed values. It is replaced atomically
(a temporary file, `fsync`, rename), so the stack's file watchers never read half a file, and the first save keeps the
file as it was as `<file>.bak`. The text written is what the robot's copy receives (above).

### What a tab shows

The MPC Parameters, Joint PD Gains and Command Limits tabs are built from the schemas of their files, not from lists in
the tabs. `remote_control/config_schema.py` walks a file's message descriptor and yields a parameter per number, bool,
enum and registry name, with what the field's `(humanoid_mpc_config.tuning)` option says about it
(`tuning_options.proto`): its unit, its slider range or log scale, whether the running stack applies it (`RELOAD_HOT`)
or only the next start (`RELOAD_START_UP`), which formulations read it (`formulations`: "centroidal", "whole_body";
none named: every formulation), the conditions under which it applies at all, and the reason it gets no widget, where
it gets none (thread counts, log switches, model selections, names, the initial pose, the torque limits). A block's
options apply to every field below it unless that field sets its own.

A registry name is a drop-down of the names its registry accepts, which the GUI reads from
`humanoid_nmpc/humanoid_mpc_config/config_registries.textproto` (written by `bazel run
//tools/config_registries:print_config_registries`, tools/config_registries/README.md); without that file the
operator types the name, and the receiving registry checks it. A term list shows its names, read-only.

A row is labeled by its field names: the dotted path from below its block (`joint_positions[back_bkz]`,
`base_position.x`), each element of a list by its key. The label adds the unit and, where they apply, "restart" (a
`RELOAD_START_UP` field; every formulation applies its hot fields live), "not applicable: the whole-body MPC does not
read it" (a task file field of the other formulation: a task file that names a `centroidal_model` is the centroidal
MPC's, any other the whole-body MPC's, `config_schema.formulation_of()`), "not applicable:
contact_input_parameterization is ...", "default" (the file leaves the field out) or "unset". No option renames a
field, and the comments of the file are not shown.

An "unset" parameter (an optional field without a default, whose absence means something of its own, such as a
`com_height` left to the robot model) shows no value: its box is empty, a check box shows its third state, a drop-down
no name. Nothing but the operator gives it one - leaving an untouched box changes nothing, for any row: the box shows the
value rounded, and only a typed text is taken - and its reset button makes it unset again.

**Adding a hyperparameter** needs a field in its schema (with its default and, where the defaults do not fit, its
tuning options) and its conversion in C++: the tab shows it with no GUI code
(humanoid_nmpc/humanoid_mpc_config/README.md, "Adding a hyperparameter"). `test/test_gui_parameter_coverage.py` fails
for a field that neither renders nor states why not, or that does not say who applies it, and
`test/test_config_registries.py` for a registry the registries' file does not list.

`remote_control/tuned_file.py` holds a file while it is tuned (`TunedFile`), without Tk: a slider moved is a change of
it, the edited file is made from the changes on demand, and only `save()` writes. The Joint Targets tab takes the
joints of the PD gains file and their defaults from the reference file's `default_joint_state`; the MPC tab's contact
estimator is the Base Controller tab's "Contact estimator" drop-down too.

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

`test/test_two_copy_save_end_to_end.py` runs the two copies end to end: the MPC Parameters, Joint PD Gains and Command
Limits tabs save a copy of the DRC Atlas files to the robot process of the simulation (`humanoid_centroidal_mpc_robot`,
headless, with a store in the test's scratch space) over a loopback bus. Each Save leaves the laptop's bytes in the
robot's store; a save of another configuration, another robot or schema, or of a text that does not parse is refused
with the robot's copy untouched; and the robot started again runs the saved task file.

The GUI ends as closing its window does - the bus closed, the Rerun bridge it started stopped - on Ctrl-C, SIGTERM and
SIGHUP (`tools/launch` stops a process with SIGTERM, and with SIGHUP when its terminal closes). The bridge also gets
SIGTERM from the kernel when the GUI dies without that cleanup (`PR_SET_PDEATHSIG`), since it runs in a session of its
own that no teardown of the GUI's process group reaches.
