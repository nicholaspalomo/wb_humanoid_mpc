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
    --task_file=$PWD/$G1/g1_wb_mpc/config/mpc/task.yaml --reference_file=$PWD/$G1/g1_wb_mpc/config/command/reference.yaml \
    --urdf_file=$PWD/$G1/g1_description/urdf/g1_29dof.urdf \
    --gait_file=$PWD/humanoid_nmpc/humanoid_common_mpc/config/command/gait.yaml
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

As in the ROS node, the whole-body MPC has no contact planner and no MPC parameter updater: `operator/mpc_parameters`
is not subscribed. The solver thread runs on the MPC cores of `ThreadAffinity.h`, `SCHED_FIFO` at
`--realtime_priority` when it is positive. The node runs the visualization publisher
([`humanoid_common_mpc_app/visualization`](../humanoid_common_mpc_app/visualization/README.md)): `viz/scene` and
`viz/telemetry` for the Rerun bridge, from every policy it publishes and every `robot/state` sample, on a thread of its
own at a raised nice value.

**`humanoid_wb_mpc_dummy_sim`** rolls the MPC model out at 80 Hz (the ROS dummy's rate) under the node's policies,
synchronized with the MPC's `mpcDesiredFrequency` (`node::DummySimLoop`), and publishes `robot/mpc_observation` as the
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
[`humanoid_common_mpc_app/robot`](../humanoid_common_mpc_app/robot/README.md) for its flags, its realtime thread and its
task-file keys. It reaches its MPC, the MPC node, over the bus only, and builds no optimal control problem:
`WBMpcInterface::CreateControllerModels()`. The in-process MPC it once had (`--mpc_link=in_process`) was removed; the
retired flag is refused at start-up. The whole-body controller computes its feedforward with the inverse dynamics only: `wbMpcFeedforward:
gravity_compensation` is refused.
