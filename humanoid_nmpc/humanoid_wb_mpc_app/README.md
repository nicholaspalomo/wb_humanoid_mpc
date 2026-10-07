# humanoid_wb_mpc_app

The processes of the distributed runtime
([`humanoid_nmpc/docs/distributed_runtime/README.md`](../docs/distributed_runtime/README.md)) that run the whole-body
MPC. They replace the ROS-era nodes of that MPC, which were deleted with ROS (see the distributed runtime's
README, "From ROS 2").

## The MPC node and the dummy simulator (laptop side)

<!-- LINT.IfChange(laptop_binaries) -->
| Binary | Bus node | Replaces | What |
|---|---|---|---|
| `:humanoid_wb_mpc_node` | `mpc` | `humanoid_wb_mpc_sqp_node` | the whole-body MPC, solving the robot's observations |
| `:humanoid_wb_mpc_dummy_sim` | `robot` | `humanoid_wb_mpc_dummy_sim_node` | the robot played by the MPC model, against the MPC node |
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_wb_mpc_app/BUILD.bazel:laptop_binaries) -->

```bash
G1=robot_models/unitree_g1
bazel run //humanoid_nmpc/humanoid_wb_mpc_app:humanoid_wb_mpc_node -- --robot_name=g1 \
    --task_file=$PWD/$G1/g1_wb_mpc/config/mpc/task.textproto \
    --reference_file=$PWD/$G1/g1_wb_mpc/config/command/reference.textproto \
    --urdf_file=$PWD/$G1/g1_description/urdf/g1_29dof.urdf \
    --gait_file=$PWD/humanoid_nmpc/humanoid_common_mpc/config/command/gait.textproto
bazel run //humanoid_nmpc/humanoid_wb_mpc_app:humanoid_wb_mpc_dummy_sim -- --robot_name=g1 \
    --task_file=... --reference_file=... --urdf_file=...    # the same files
```

**`humanoid_wb_mpc_node`** (`WBMpcNode.h`) builds what the whole-body SQP node of the ROS era built, and serves it with
`node::MpcNodeRuntime` ([`humanoid_common_mpc_app/node`](../humanoid_common_mpc_app/node/README.md)):

- `WBMpcInterface::Create()` and an `SqpMpc`;
- the procedural motion manager over `WBMpcTargetTrajectoriesCalculator`, fed from `operator/walking_velocity_command`;
  a reset of the MPC resets it and, through its reset hook, the calculator;
- resets served from `wbMpcResetTargetTrajectories()`, the reset target the MRT joint controller hands its in-process
  link (`humanoid_wb_mpc/mrt/WBMpcResetTarget.h`);
- with every policy, the scaled velocity command (`MpcPolicy.annotations`).

As in the ROS node, the whole-body MPC has no contact planner. It has the MPC parameter updater the centroidal node has
(`makeWholeBodyMpcParameterUpdater()`, `humanoid_wb_mpc/mrt/WBMpcParameterUpdater.h`), a synchronized module that runs
before each solve: an update on `operator/mpc_parameters` is checked and counted (`MpcNodeRuntime`) and its
`RELOAD_HOT` fields are written into the running problem before the next solve (`wholeBodyHotFieldAppliers()`); the
`RELOAD_START_UP` fields that differ are logged as taking effect at the next start, and the robot process applies the
contact estimator and the contact wrench gate of the same update. The updater also reloads the task file and the
reference file when they are saved. The solver thread runs on the MPC cores of `ThreadAffinity.h`, `SCHED_FIFO` at
`--realtime_priority` when it is positive. The node runs the visualization publisher
([`humanoid_common_mpc_app/visualization`](../humanoid_common_mpc_app/visualization/README.md)): `viz/scene` and
`viz/telemetry` for the Rerun bridge, from every policy it publishes and every `robot/state` sample, on a thread of its
own at a raised nice value.

**`humanoid_wb_mpc_dummy_sim`** rolls the MPC model out at 80 Hz (the ROS dummy's rate) under the node's policies,
synchronized with the task file's `mpc.mpc_desired_frequency` (`node::DummySimLoop`), and publishes `robot/mpc_observation` as the
bus node `robot`.

<!-- LINT.IfChange(node_flags) -->
The node's own flags: `--ipc_node` (default `mpc`), `--realtime_priority` (SCHED_FIFO priority of the solver thread,
0 = off).
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_wb_mpc_app/src/WBMpcNodeMain.cpp:node_flags) -->

<!-- LINT.IfChange(dummy_sim_flags) -->
The dummy's own flag: `--ipc_node` (default `robot`). It does not read `--gait_file`.
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_wb_mpc_app/src/WBMpcDummySimMain.cpp:dummy_sim_flags) -->

Both take the shared flags of [`humanoid_common_mpc_app/node`](../humanoid_common_mpc_app/node/README.md#the-command-line).

`:test_wb_mpc_node` (large: it builds the G1 whole-body CppAD libraries on a cold cache) runs the node with the real SQP
solver against a scripted robot and operator over loopback buses - observation in and policy out, resets through the
observation counters, the velocity command reaching the motion manager and the next policy - checks that it resets to
exactly the target the controller hands its in-process link, and runs the dummy simulator against it for four seconds:
it stands, the policies flow, and no reset after the start-up one.

## The robot process (robot side)

<!-- LINT.IfChange(robot_binaries) -->
| Binary | Bus node | Replaces | What |
|---|---|---|---|
| `:humanoid_wb_mpc_robot` | `robot` | the ROS-era MuJoCo sim node (`WBMpcRobotSim`) | the realtime loop: the whole-body MRT joint controller on a robot backend (`--backend=mujoco`) |
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_wb_mpc_app/BUILD.bazel:robot_binaries) -->

The binary that runs on the robot's realtime computer and, with the MuJoCo backend, in simulation: see
[`humanoid_common_mpc_app/robot`](../humanoid_common_mpc_app/robot/README.md) for its flags, its realtime thread and the
task-file fields it reads. It reaches its MPC, the MPC node, over the bus only, and builds no optimal control problem:
`WBMpcInterface::CreateControllerModels()`. The in-process MPC it once had (`--mpc_link=in_process`) was removed; the
retired flag is refused at start-up. The whole-body controller computes its feedforward with the inverse dynamics only:
`wb_mpc_feedforward: "gravity_compensation"` is refused.

## The configuration files

The whole-body MPC of the Unitree G1 is configured by textprotos, each a message of the schemas in
[`humanoid_nmpc/humanoid_mpc_config`](../humanoid_mpc_config/README.md) named in its first two lines
(`# proto-file:`, `# proto-message:`). They are read strictly: a field the schema does not have, a retired one or a value
of the wrong type stops the process with the file, line and column, and a retired field's error names its replacement.

| File | Message | Read by |
|---|---|---|
| `robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto` | `TaskFile` | the MPC node and the dummy simulator (`WBMpcInterface::Create()`); the robot process (`WBMpcInterface::CreateControllerModels()` and its own settings) |
| `robot_models/unitree_g1/g1_wb_mpc/config/command/reference.textproto` | `ReferenceFile` | the gait schedule, the target calculator and the motion manager |
| `robot_models/unitree_g1/g1_wb_mpc/config/controller/joint_pd_gains.textproto` | `JointPdGainsFile` | the robot process's MRT joint controller, beside the task file (`config/controller/`) |
| `humanoid_nmpc/humanoid_common_mpc/config/command/gait.textproto` | `GaitFile` | the MPC node's motion manager |

**Tuning.** The weights and values are addressed by name, not by position: `state_weights { base_position { z: ... } }`,
the base orientation in Euler angles ZYX (`base_orientation { yaw: ... pitch: ... roll: ... }`), a joint as
`joint_positions { joint: "left_knee_joint" value: ... }`. The MPC node's parameter updater applies an edit of the task
file's `RELOAD_HOT` fields, and of the reference file's command limits, before the next solve; every other field takes
effect at its next start (humanoid_nmpc/humanoid_mpc_config/README.md, "Live updates"). The robot process applies what
it reads while it runs: the task file's `contact_estimator` and `contact_wrench_gate` when the file changes, and the PD
gains file, which it checks about once a second.

**The GUI** ([`remote_control`](../remote_control/README.md), "Tuning the configuration") edits the same files, rendered from
their schemas. Its Joint PD Gains tab publishes the whole gains file to the robot process (`operator/pd_gains`), and the
task file its MPC Parameters tab publishes (`operator/mpc_parameters`) reaches the MPC node's parameter updater, which
applies its `RELOAD_HOT` fields before the next solve, and the robot process's contact estimator and contact wrench gate.
**Save** writes the edited values into the file and keeps every other byte of it (comments, blank lines, LINT
directives), replacing it atomically and keeping the file as first loaded as `<file>.bak`.

The formulation choices only the centroidal MPC implements are refused at start-up, by name: the contact-implicit terms,
`com_and_acom_tracking_cost`, `dcm_terminal_cost`, `contact_input_parameterization: "basis_vectors"` and
`contact_schedule_source: "contact_planner"`.
