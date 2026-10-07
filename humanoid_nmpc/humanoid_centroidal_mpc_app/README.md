# humanoid_centroidal_mpc_app

The processes of the distributed runtime
([`humanoid_nmpc/docs/distributed_runtime/README.md`](../docs/distributed_runtime/README.md)) that run the centroidal
MPC. They replace the ROS-era nodes of that MPC, which were deleted with ROS (see the distributed runtime's
README, "From ROS 2").

## The MPC node and the dummy simulator (laptop side)

<!-- LINT.IfChange(laptop_binaries) -->
| Binary | Bus node | Replaces | What |
|---|---|---|---|
| `:humanoid_centroidal_mpc_node` | `mpc` | `humanoid_centroidal_mpc_sqp_node` | the centroidal MPC, solving the robot's observations |
| `:humanoid_centroidal_mpc_dummy_sim` | `robot` | `humanoid_centroidal_mpc_dummy_sim_node` | the robot played by the MPC model, against the MPC node |
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_centroidal_mpc_app/BUILD.bazel:laptop_binaries) -->

```bash
ROBOT=robot_models/drc_atlas
bazel run //humanoid_nmpc/humanoid_centroidal_mpc_app:humanoid_centroidal_mpc_node -- --robot_name=drc_atlas \
    --task_file=$PWD/$ROBOT/drc_atlas_centroidal_mpc/config/mpc/task.textproto \
    --reference_file=$PWD/$ROBOT/drc_atlas_centroidal_mpc/config/command/reference.textproto \
    --urdf_file=$PWD/$ROBOT/drc_atlas_description/urdf/atlas.urdf \
    --gait_file=$PWD/humanoid_nmpc/humanoid_common_mpc/config/command/gait.textproto
bazel run //humanoid_nmpc/humanoid_centroidal_mpc_app:humanoid_centroidal_mpc_dummy_sim -- --robot_name=drc_atlas \
    --task_file=... --reference_file=... --urdf_file=...    # the same files
```

**`humanoid_centroidal_mpc_node`** (`CentroidalMpcNode.h`) builds what the SQP node of the ROS era built, and serves it
with `node::MpcNodeRuntime` ([`humanoid_common_mpc_app/node`](../humanoid_common_mpc_app/node/README.md)):

- `CentroidalMpcInterface::Create()` and an `SqpMpc`;
- the procedural motion manager over `CentroidalMpcTargetTrajectoriesCalculator`, fed from
  `operator/walking_velocity_command`; a reset of the MPC resets it and, through its reset hook, the calculator;
- under `contact_schedule_source: "contact_planner"`, the contact planner module;
- the MPC parameter updater (`makeCentroidalMpcParameterUpdater()`: the live-tuning machinery both formulations share,
  `humanoid_common_mpc/parameter_update/`, with the centroidal formulation's appliers; `makeCommandLimitsReloaders()`
  hands the command limits of a reloaded reference file to the calculator and the motion manager), fed from
  `operator/mpc_parameters` and applied before the next solve (see "The configuration files" below);
- resets served from `centroidalMpcResetTargetTrajectories()`, the reset target the MRT joint controller hands its
  in-process link (`humanoid_centroidal_mpc/mrt/CentroidalMpcResetTarget.h`), at the counters of the robot's
  observations (MpcServer);
- with every policy, the planner's target contact poses and the scaled velocity command (`MpcPolicy.annotations`).

The solver thread runs on the MPC cores of `ThreadAffinity.h`, `SCHED_FIFO` at `--realtime_priority` when it is
positive. The node runs the visualization publisher
([`humanoid_common_mpc_app/visualization`](../humanoid_common_mpc_app/visualization/README.md)) on the effective model:
`viz/scene` and `viz/telemetry` for the Rerun bridge, from every policy it publishes and every `robot/state` sample, on
a thread of its own at a raised nice value.

**`humanoid_centroidal_mpc_dummy_sim`** rolls the MPC model out at 100 Hz under the node's policies, synchronized with
the MPC's `mpc.mpc_desired_frequency` as the ROS dummy was (`node::DummySimLoop`), and publishes `robot/mpc_observation` as the
bus node `robot`. Start the node and the dummy in either order.

<!-- LINT.IfChange(node_flags) -->
The node's own flags: `--ipc_node` (default `mpc`), `--realtime_priority` (SCHED_FIFO priority of the solver thread,
0 = off).
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_centroidal_mpc_app/src/CentroidalMpcNodeMain.cpp:node_flags) -->

<!-- LINT.IfChange(dummy_sim_flags) -->
The dummy's own flag: `--ipc_node` (default `robot`). It does not read `--gait_file`.
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_centroidal_mpc_app/src/CentroidalMpcDummySimMain.cpp:dummy_sim_flags) -->

Both take the shared flags of [`humanoid_common_mpc_app/node`](../humanoid_common_mpc_app/node/README.md#the-command-line)
(`--robot_name`, `--task_file`, `--reference_file`, `--urdf_file`, `--gait_file`, `--network_config`).

`:test_centroidal_mpc_node` (large: it builds the DRC Atlas CppAD libraries on a cold cache) runs the node with the real
SQP solver against a scripted robot and operator over loopback buses - observation in and policy out, a solver and a
full reset through the observation counters, the velocity command reaching the motion manager and the next policy, a
task file sent on `operator/mpc_parameters` applied in the next solve and not before - checks that it resets to exactly
the target the controller hands its in-process link, and runs the dummy simulator against it for four seconds: it
stands, one policy per MPC update, and no reset after the start-up one. Its last case runs the hardware topology end to
end: `:humanoid_centroidal_mpc_node` and `:humanoid_centroidal_mpc_robot` (MuJoCo, headless) as two processes on a
network file of the test, and a scripted operator that takes the robot through JOINT_PD into WB_MPC, releases the
gantry, and commands it to stand, walk forward at 0.3 m/s and stop: the robot stays up and travels forward, the loop
holds its period, the policy in use stays fresh, `viz/scene` reaches the bus, and both processes end cleanly on
SIGTERM.

## The robot process (robot side)

<!-- LINT.IfChange(robot_binaries) -->
| Binary | Bus node | Replaces | What |
|---|---|---|---|
| `:humanoid_centroidal_mpc_robot` | `robot` | the ROS-era MuJoCo sim node (`CentroidalMpcRobotSim`) | the realtime loop: the centroidal MRT joint controller on a robot backend (`--backend=mujoco`) |
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_centroidal_mpc_app/BUILD.bazel:robot_binaries) -->

The binary that runs on the robot's realtime computer and, with the MuJoCo backend, in simulation: see
[`humanoid_common_mpc_app/robot`](../humanoid_common_mpc_app/robot/README.md) for its flags, its realtime thread and the
task-file fields it reads. It reaches its MPC, the MPC node, over the bus only, and builds no optimal control problem:
`CentroidalMpcInterface::CreateControllerModels()`. The in-process MPC it once had (`--mpc_link=in_process`, a copy of
`CentroidalMpcNode`'s wiring on a thread of the robot process) was removed; the retired flag is refused at start-up.

`:test_centroidal_mpc_robot_end_to_end` starts the binary as a process (MuJoCo headless, the DRC Atlas files) against an
`MpcServer` around a scripted MPC and an operator, on a loopback network file of its own: the robot comes up in
ZERO_TORQUE, reaches WB_MPC through JOINT_PD, applies the MPC's policy, keeps its 100 Hz period, holds the JOINT_PD
action when the MPC side goes away (link loss, `mpc_healthy` false in `robot/fsm_state`), and ends cleanly on SIGTERM.

## The configuration files

Both processes read a robot's configuration as typed textprotos
([`humanoid_nmpc/humanoid_mpc_config`](../humanoid_mpc_config/README.md), one schema per file), parsed strictly: a field
the schema does not know, a value of the wrong type or a retired key stops the start with the file, the line and the
column.

| File | Read by these binaries as |
|---|---|
| `robot_models/<robot>/<package>/config/mpc/task.textproto` (`--task_file`) | the model, the formulation, the weights, the solver, the robot process's settings |
| `.../config/command/reference.textproto` (`--reference_file`) | the default posture, the gait schedule's start, the command limits |
| `.../config/mpc/contact_planning.textproto`, beside the task file | the contact planner, where the robot ships one (DRC Atlas, EngineAI SA01) |
| `.../config/controller/joint_pd_gains.textproto`, found from the task file's directory | the MRT joint controller's PD gains (robot process) |
| `humanoid_nmpc/humanoid_common_mpc/config/command/gait.textproto` (`--gait_file`) | the gaits of the procedural motion manager (MPC node), at start-up only |

`CentroidalMpcInterface` takes them by path at the composition roots (`Create()`, `CreateControllerModels()`), or
already loaded and typed (`CentroidalMpcConfig`, `loadCentroidalMpcConfig()`), which is how the tests vary a
configuration without writing a file.

**Tuning a running stack.** The MPC node's parameter updater (`MpcParameterUpdaterModule`, shared with the whole-body
MPC) applies a changed
configuration before the next solve, without a restart, from two sources: the files on disk, which it polls about
once a second (the task file, the contact planner's file beside it, and the reference file, whose command limits it
converts once and hands to the target calculator and the motion manager), and the whole task file and contact
planner's file the tuning GUI publishes on `operator/mpc_parameters`. A reload is a whole file: a block it leaves out
means the block's defaults, as at start-up. It applies the fields the schema marks `RELOAD_HOT`
for the centroidal formulation (`centroidalHotFieldNames()`, the fields of the appliers `centroidalHotFieldAppliers()`
builds; `appliedFields()` of the running updater); a `RELOAD_START_UP` field that differs is logged once and takes
effect at the next start. An update of another configuration than the node's task file (its `config_path`) is refused
and counted. A file that does not parse or convert is logged, naming the file and the field, and the
running values are kept. The robot process watches the task file for its controller-side settings (the contact
estimator, the contact wrench gate) and the PD gains file for the joint gains.

**Saving from the GUI.** The tuning GUI (`humanoid_nmpc/remote_control`, "Tuning the configuration" in its
[README](../remote_control/README.md)) edits these files losslessly: its Save replaces only the values that changed,
keeps every comment and LINT directive, re-parses the result strictly before an atomic write, and the watchers above
pick the saved file up. The files can equally be edited by hand.
