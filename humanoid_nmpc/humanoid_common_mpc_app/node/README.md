# humanoid_common_mpc_app/node

The formulation-independent parts of the laptop-side binaries of the distributed runtime
([`humanoid_nmpc/docs/distributed_runtime/README.md`](../../docs/distributed_runtime/README.md)): the MPC node's
runtime on the bus, the dummy simulator's loop over the remote MPC link, and the process plumbing they share with the
keyboard teleoperation ([`../teleop`](../teleop/README.md)). The binaries are built per formulation, in
[`humanoid_centroidal_mpc_app`](../../humanoid_centroidal_mpc_app/README.md) and
[`humanoid_wb_mpc_app`](../../humanoid_wb_mpc_app/README.md).

| File | Target | What |
|---|---|---|
| `MpcNodeRuntime.h` | `:mpc_node` | the MPC node on the bus: an `ipc::MpcServer` and the operator inputs of the MPC |
| `DummySimLoop.h` | `:mpc_node` | the dummy simulator: the MPC model as the plant, over an `ipc::RemoteMpcLink` |
| `WalkingVelocityCommandConversions.h` | `:mpc_node` | `operator/walking_velocity_command` -> `WalkingVelocityCommand` |
| `ViewerAnnotations.h` | `:mpc_node` | `MpcPolicy.annotations`: target contact patches and the scaled velocity command |
| `NodeBus.h`, `ShutdownSignal.h`, `MpcFiles.h` | `:node_process` | the bus of `--network_config` / `--ipc_node`, SIGINT / SIGTERM, the robot's files |
| `MpcAppFlags.h` | `:mpc_app_flags` | the flags the MPC nodes and dummy simulators share (binaries only) |

## The MPC node: `MpcNodeRuntime`

```text
 bus IO thread                                   solver thread (MpcServer, "mpc_solver", MPC cores)
 robot/mpc_observation ------------------------> resets (observation counters) -> SqpMpc::run() -> mpc/policy, mpc/status
 operator/walking_velocity_command -> motion manager (setAndScaleVelocityCommand)  |  annotations, then the
 operator/mpc_parameters ----------> parameter updater (enqueueParameterUpdate)    |  visualization observer
```

A formulation's node (`CentroidalMpcNode`, `WBMpcNode`) builds the MPC and its synchronized modules and hands the
runtime what it is made of (`MpcNodeRuntime::Components`):

- `mpc`: the `SqpMpc`, with the reference manager and the synchronized modules registered in the order of the ROS
  nodes (motion manager, contact planner, parameter updater);
- `resetTargetTrajectories`: the formulation's reset target, the same free function the MRT joint controller hands its
  in-process link (`centroidalMpcResetTargetTrajectories()`, `wbMpcResetTargetTrajectories()`), so that a reset served
  by the MPC node and one served in process restart the MPC from the same target;
- `motionManager`: `operator/walking_velocity_command` goes to its `setAndScaleVelocityCommand()` on the IO thread
  (`ProceduralMpcMotionManager` keeps the command under a mutex; the solver thread reads it in `preSolverRun()`); a
  command with a value that is not finite is refused, logged and counted;
- `parameterUpdateSink` (optional): the updates of `operator/mpc_parameters` (`humanoid_mpc_config.MpcParameterUpdate`,
  the tuning GUI's whole task file and contact planner's file), converted to their structs on the IO thread, for
  `MpcParameterUpdaterModule::enqueueParameterUpdate()` (`humanoid_common_mpc/parameter_update/`, the one updater
  both formulations' nodes register), which applies the newest one before the next solve. An update of another schema
  version, of another robot (`Config::robotName`) or of another configuration (`Config::taskFileIdentity`, the
  `configFileIdentity()` of the node's task file, against the update's `config_path`: the centroidal and the
  whole-body G1 share their `robot_name`) is refused, logged and counted (`checkMpcParameterUpdate()`);
- `contactPlanningReferenceManager` (optional): the target contact poses sent with every policy;
- `attachVisualization` (optional): the visualization publisher's attacher, below.

The solver thread is configured by `defaultSolverThreadConfig(realtimePriority)`: named `mpc_solver`, pinned to the MPC
cores of `humanoid_common_mpc/common/ThreadAffinity.h`, `SCHED_FIFO` when the node's `--realtime_priority` is positive,
and without `mlockall()` (`MemoryLock::kNone`), because the solver allocates as it runs. The node binaries also pin
their main thread to the MPC cores before they build the MPC, so that the SQP solver's worker threads and the bus's IO
thread inherit them.

### The visualization publisher

`MpcNodeRuntime::VisualizationAttacher` is called once, by `Create()`, with the node's bus before the bus starts: the
visualization publisher ([`../visualization`](../visualization/README.md)) subscribes to `robot/state` there and
publishes `viz/scene` and `viz/telemetry` from a thread of its own. It returns the `MpcServer::PostSolveObserver` the
server calls after every policy it publishes, on the solver thread, with the command, the solution and the performance
index: the observer copies what it needs and returns.

Both formulation nodes always attach it, as the ROS nodes always ran their visualizer:
`CentroidalMpcNode` and `WBMpcNode` hand the runtime `visualization::VisualizationPublisher::MakeBusAttacher()` on the
MPC's own models, keep the publisher as a member declared before the runtime (so that it outlives the server whose
observer feeds it), start it in `start()` before the runtime, and stop it in `stop()` before the runtime, which owns
the bus. `Options::visualization` sets its thread and queue. The runtime's tests attach scripted observers instead.

## The dummy simulator: `DummySimLoop`

OCS2's `MRT_ROS_Dummy_Loop` without ROS. The plant is the MPC's own model, rolled out under the policy in use
(`MRT_BASE::rolloutPolicy()` with the formulation's `RolloutBase`), and the MPC is the MPC node, reached through an
`ipc::RemoteMpcLink` exactly as the robot process reaches it: the dummy publishes as the bus node `robot`, so it runs
against the same MPC node as the robot would.

1. It requests a full reset and sends the initial observation every step until the first policy has arrived.
2. With a positive `mpc.mpc_desired_frequency` of the task file it runs synchronized with the MPC: every
   `max(1, simulationFrequency / mpc_desired_frequency)` steps it waits for the policy solved from the observation it
   sent the step before (one that starts within 0.1 MPC periods of the plant's time), and it sends only that
   observation. The plant's time advances by one MPC period per solve however long a solve takes.
3. With `mpc_desired_frequency <= 0` it runs in real time: every step takes the newest policy and sends its observation.
4. Every step rolls the plant forward by `1 / simulationFrequency` (100 Hz centroidal, 80 Hz whole-body, as before),
   paced on absolute deadlines (`robot::realtime::PeriodicTimer`).

The link's health is not acted on: the plant keeps rolling the policy in use out. The dummy publishes
`robot/mpc_observation` only, no `robot/state`.

## The command line

<!-- LINT.IfChange(mpc_app_flags) -->
| Flag | Default | Meaning |
|---|---|---|
| `--robot_name` | empty | the robot, for the logs |
| `--task_file` | required | `config/mpc/task.textproto` of the robot (`humanoid_mpc_config.TaskFile`) |
| `--reference_file` | required | `config/command/reference.textproto` of the robot (`humanoid_mpc_config.ReferenceFile`) |
| `--urdf_file` | required | the robot's URDF |
| `--gait_file` | required by the MPC nodes | `humanoid_nmpc/humanoid_common_mpc/config/command/gait.textproto` (`humanoid_mpc_config.GaitFile`) |
| `--network_config` | empty: the shipped localhost network | the network file of the bus |
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc_app/node/include/humanoid_common_mpc_app/node/MpcAppFlags.h:mpc_app_flags) -->

Each binary defines `--ipc_node` itself: `mpc` for the MPC nodes, `robot` for the dummy simulators. A file flag that is
not given, or names no file, ends the binary with exit code 2 and a message that names the flag. The files are the
typed textprotos of [`humanoid_mpc_config`](../../humanoid_mpc_config/README.md), read strictly: an unknown field or a
value of the wrong type ends the start-up with its file, line and column.

SIGINT and SIGTERM ask for a clean shutdown (`ShutdownSignal.h`); a second one ends the process at once.

## Tests

| Target | What |
|---|---|
| `:test_node_messages` | the velocity command clamped to the normalized ranges and refused when not finite; the annotations |
| `:test_node_process` | SIGTERM asks for the shutdown; an empty `--network_config` is the shipped network; an unknown node is refused |
| `:test_mpc_node_runtime` | the runtime around the DRC Atlas references and motion manager with OCS2's scripted MPC (no CppAD), against a scripted robot and operator over loopback: observation in and policy out, resets through the observation counters, the velocity command reaching the motion manager and the next policy's annotations, a parameter document applied in the next solve and not before, one of another robot, schema or configuration (or
without a `config_path` at a node with an identity) refused, no parameter subscription without a sink, a target patch per foot under the contact planner, the visualization attacher |
| `:test_dummy_sim_loop` | the loop against an `MpcServer` with the scripted MPC: synchronized (one policy per update, solved for it), in real time, stopping while it waits, no reset after the first |

The formulation packages test the real MPCs end to end (`humanoid_centroidal_mpc_app:test_centroidal_mpc_node`,
`humanoid_wb_mpc_app:test_wb_mpc_node`).
