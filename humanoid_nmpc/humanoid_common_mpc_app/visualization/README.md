# humanoid_common_mpc_app/visualization

The visualization publisher of the MPC process. It computes, with the MPC's robot model and Pinocchio, everything the
Rerun bridge ([`humanoid_rerun_viewer`](../../humanoid_rerun_viewer/README.md)) draws, and publishes it on the bus:

- `viz/scene` (`VisualizationScene`): the robot model instances as link world poses, and the markers;
- `viz/telemetry` (`TelemetrySeries`): the plots, one message per `robot/state` sample.

It runs in the MPC process, on a low-priority thread of its own, so the robot's realtime control thread never draws
anything. It works for both formulations (it only
uses `MpcRobotModelBase`) and for every robot: the frame names come from the task file and the model. The entity paths
and series names are the bridge's contract, tabulated in its README; `SceneContract.h` and the group table of
`TelemetryBuilder.cpp` are tied to the bridge's contract modules with `LINT.IfChange`, and `:test_end_to_end` checks the
two against each other through an `.rrd` file.

## Threads

```text
MPC solver thread ---- setPolicy(command, solution) ----> [triple buffer]  --+
  (MpcServer::Hooks::postSolveObserver)                                       |
                  ---- setObservation(observation) -----> [triple buffer]  --+--> visualization thread (SCHED_IDLE),
bus IO thread -------- pushRobotState(sample) ----------> [bounded SPSC      |    every pollPeriod (5 ms):
  (subscribeRobotState: robot/state, every sample)          queue, 64]      --+    one viz/telemetry per sample,
                                                                                   viz/scene at rerun_scene_frequency
                                                                                   -> [IoThreadOutbox, SPSC, 32]
                                                                                   -> bus IO thread, every pollPeriod:
                                                                                      Bus::publishFromIoThread()
```

Feeding never blocks and never waits for the visualization: each mailbox has one producer and copies into storage it
already holds (no allocation once the shapes are stable). When the visualization falls behind, `robot/state` samples
that do not fit in the queue are dropped and counted (`Statistics::robotStatesDropped`), never queued without bound; a
newer observation or policy replaces one not yet drawn. The visualization thread runs under `SCHED_IDLE`
(`Options::threadPriority`, `robot::web_viewer::ViewerThreadPriority`, as the MuJoCo web viewer's threads): it runs only
on time no other thread wants, and a solver thread that wakes on its core takes the core at once, which a raised nice
value does not do (under EEVDF a woken thread may wait out the running thread's time slice whatever its nice value;
[`mujoco_sim_interface`](../../../robot_runtime/mujoco_sim_interface/README.md) measured it for the viewer).
`kInherited` keeps the starting thread's scheduling, for tests. On a bus, the visualization thread takes none of the
bus's locks: `Bus::publish()` takes its queue and counter mutexes, which the solver thread's own publishes take too, and
a `SCHED_IDLE` thread preempted while it held one would hold the solver up until a core went idle. It posts its
messages to a lock-free outbox (`IoThreadOutbox`) instead, which a periodic callback of the bus drains on the IO thread;
a full outbox drops the message and counts it (`Statistics::publishFailures`). So `Create(model, bus)` comes before
`bus->start()`. A failure to switch is logged as a warning and the
thread runs on. `Options::cores` can pin it away from the solver's cores. Nothing of it runs on the robot's realtime
thread.

A scene is published when something new arrived and its period has passed (on the monotonic clock, so a robot clock
that goes back, as when a simulation restarts, does not stop it).

## Attaching it in the MPC node

```cpp
#include "humanoid_common_mpc_app/visualization/VisualizationPublisher.h"

using ocs2::humanoid::visualization::VisualizationModel;
using ocs2::humanoid::visualization::VisualizationPublisher;

// Before the solver thread starts using the model: the publisher copies the Pinocchio model and clones the robot model.
// It keeps the state manifold (required), which must outlive it.
const VisualizationModel model{taskFile, urdfFile, &interface.getPinocchioInterface(), &mpcRobotModel, stateManifold};
ASSIGN_OR_RETURN(std::unique_ptr<VisualizationPublisher> visualization, VisualizationPublisher::Create(model, *bus));
RETURN_IF_ERROR(visualization->subscribeRobotState(*bus));  // before bus->start()
RETURN_IF_ERROR(visualization->start());                     // publishes once the bus runs

// setPolicy() with every published solution, setObservation() with the observation it was solved from.
ocs2::humanoid::ipc::MpcServer::Hooks hooks;
hooks.postSolveObserver = visualization->postSolveObserver();
// MpcServer::Create(*bus, mpc, resetTarget, config, hooks); bus->start(); server->start();
// Shutdown: server->stop(); visualization->stop(); bus->stop(). The publisher outlives the server.
```

With `MpcNodeRuntime` (`humanoid_common_mpc_app/node`) the steps are its `VisualizationAttacher`, which is called
with the node's bus before it starts: `VisualizationPublisher::MakeBusAttacher(model, options, &publisher)` creates the
publisher on that bus, subscribes it to `robot/state` and returns `postSolveObserver()`. `CentroidalMpcNode` and
`WBMpcNode` always attach it; they keep the publisher as a member declared before the runtime, so that it outlives the
server, start it in their `start()` before the runtime and stop it in their `stop()` before the runtime, which owns
the bus.

`mpcRobotModel` is the model the MPC's dynamics use (for the centroidal MPC the effective one, the
`BasisInputsModelDecorator` when the task file selects `basis_vectors`), and the Pinocchio model the MPC's
(`loadCustomPinocchioInterface()`): `checkVisualizationModel()` refuses one whose `nq` and `nv` are not the robot
model's `getConfigurationDim()` and `getVelocityDim()`, or a robot model whose base is not the
`Composite(Translation, Spherical)` root (`nq = 7 + nj`, `nv = 6 + nj`). `stateManifold` is the manifold the MPC's state
lives on, along which the plan's and the target's states are interpolated (`samplePlan()`, `sampleTarget()`: a slerp of
the base quaternion block; the inputs stay linear); the nodes pass their problem's, and `checkVisualizationModel()`
refuses a null one, since every humanoid state has the quaternion block. `Create()` takes a `PublishFunction` instead
of a bus for tests. Any process that runs a solver feeds it the same three calls.

## What it computes

**Scene** (`SceneBuilder`), from the latest observation, policy and `robot/state` sample:

- `measured`: every link of the URDF with every joint of the sample, from a Pinocchio model of the whole URDF (the
  MPC's model fixes the joints it does not plan for). Without a sample in the last `Options::robotStateTimeout`
  (500 ms; the dummy sim sends none), the observation's state on the MPC's model.
- `terminal_state` (the plan's last node) and `terminal_target` (the reference at the plan's last time), on the MPC's
  model.
- At the observation: the contact force of every stance foot as an arrow ending at its center of pressure, the net
  center of pressure, the four equivalent corner forces of every stance foot (`ContactWrenchMapper<4>` on the task
  file's contact rectangle), and the collision spheres (`collision_constraint` of the task file). The wrenches are the
  policy's input at the observation's time through the state-aware world-frame accessors, so they are right for the
  basis-vector inputs too.
- The plan: the paths of `rerun_plan_frames`, of the base and of the whole-body CoM at its height (not projected to the
  ground, so that the plan's pendulum height shows), and the footholds where a foot lands inside the horizon, in the
  contact colors.

**Telemetry** (`TelemetryBuilder`): every group of the bridge's telemetry contract, at the sample's time, for three
sources: measured (the sample, `RobotStateDecoder`), reference (the target trajectories of the latest policy's command)
and plan (the latest policy). See the bridge's README for the paths, the names and the units.

The MPC's `q` holds the base quaternion and `v` the body angular velocity `w_B`. For the plots the builder computes, into
buffers sized at construction, the ZYX Euler angles of each source's quaternion (`eulerZyxFromQuaternion()`,
`orientation/EulerBoundary.h`), their rates `T_B(Theta)^-1 w_B` (`eulerZyxRatesForDisplay()`: all three NaN where
`|cos(pitch)| < 1e-6`), and display coordinates `[p_W, (yaw, pitch, roll), q_j]` that the `dofs/position` and
`generalized_base` coordinate groups plot; the velocity and force groups plot `v` as it is, named by
`getBaseVelocityDofNames()`. `base_pose/{roll,pitch,yaw}_rate` and `base_pose/angular_velocity_{x,y,z}` sit in the
"Base Pose & Euler" tab next to the angles; `base_twist/angular_*` is the world-frame angular velocity `R(xi) w_B`. All of
it is display only and runs on this thread, never in the robot process; `mpc_observation/state` stays the raw state
(humanoid_nmpc/docs/quaternion_base_orientation/README.md, section 3.4).

**Balance** (`BalanceQuantities.h`), per source: the whole-body CoM `c` and `c_dot` (Pinocchio's, of the source's `q`
and `v`), the DCM / instantaneous capture point `xi = c_xy + c_dot_xy / omega` with `omega = sqrt(g / z0)` (`g` the
Pinocchio model's gravity, `z0 = computeComHeightAboveFeet()` at `VisualizationModel::nominalState`, the MPC
interface's initial state: the repository's one pendulum height, logged at start-up), the ZMP of the contact wrenches at
the contact frames (`p_x = (h F_x - M_y) / F_z`, `p_y = (h F_y + M_x) / F_z`, `h` the normal-force-weighted contact
height; NaN below 1 N), and the ACoM attitude (`AngularCenterOfMass`, NaN for a robot without a network) as roll,
pitch, yaw. The measured wrenches carry no torque (the simulator's force sensors), so the measured ZMP is the
normal-force-weighted mean of the contact frames, not a per-foot center of pressure.

**The MPC solution** (`SolutionTermEvaluator`): with `VisualizationModel::solverProblem`, the solver's own problem (the
first worker's of `SqpSolver::getOcpDefinitions()`, with the weights the parameter updater wrote), the post-solve
observer evaluates every cost and constraint term by name at the solution's first node (the final terms at its last)
on the solver thread, after the policy is published and before the next solve, when no worker runs; the
`mpc_solution/*` groups plot them with the solver's `PerformanceIndex`. One precomputation request and one evaluation
per term per node: tens of microseconds per solve, never on the robot's realtime thread.

## Design rules

- One **measured robot** is drawn from `robot/state`, with every joint of the URDF.
- The **force, center-of-pressure and corner-force markers** use the policy's input at the
  observation's time, not the observation's input, which a simulated robot may leave zero. Below
  `kMinNormalForceForCop` (1 N) the center of pressure is the contact frame, never 0/0.
- The **reference and plan** stay apart: the `reference` groups are the target trajectory's; the `plan` groups
  (`dofs/{position,velocity}/plan`, `frames/*/<frame>/plan`) and the `mpc` wrenches are the MPC policy's.
- **Euler angles** come from the Euler boundary: `eulerZyxFromQuaternion()` returns (yaw, pitch, roll), regular at gimbal
  lock; the builder reorders it to (roll, pitch, yaw) for the plots.
- **Angular velocities** state their frame: the world-frame panels of `base_twist` say "(world)"; the Euler rates and the
  body angular velocity have rows of their own, computed from the quaternion state, with a gap at the pitch
  singularity.
- The **plan's frames** come from `rerun_plan_frames` in the task file, checked against the model at start-up.
- On **clock rewinds**, the scene's rate is kept on the monotonic clock, and the measured acceleration restarts
  after a rewind.
- The publisher runs **off the realtime thread**, on a thread of its own in the MPC process, fed through lock-free
  mailboxes.

## Task-file fields

Read at start-up from the robot's typed task file (`config/mpc/task.textproto`, `humanoid_mpc_config.TaskFile`, parsed
strictly: an unknown field or a value of the wrong type is refused with its file, line and column), together with the
contact polygons (`contacts`) and the collision spheres (`collision_constraint`) the scene draws. The tuning GUI's MPC
Parameters tab shows `rerun_scene_frequency` as a slider like every number of the schema; a change takes effect when the
MPC node restarts. The start-up log names the values.

<!-- LINT.IfChange(task_keys) -->
| Field | Absent | Meaning |
|---|---|---|
| `rerun_scene_frequency` | 30 | [Hz] the most `viz/scene` messages per second; a positive number |
| `telemetry_frames` | the contact frames | the frames of the `frames/<kind>/<frame>/<source>` plots, one entry per frame |
| `rerun_plan_frames` | the contact frames | the frames whose planned paths `world/plan/end_effectors` draws, one strip each (the G1 adds its lidar and palms) |
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_config/task_file.proto:visualization_task_keys, //humanoid_nmpc/humanoid_common_mpc_app/visualization/include/humanoid_common_mpc_app/visualization/VisualizationConfig.h:visualization_task_fields) -->

A frame a list names that the MPC's model does not have, a duplicate, or a name that is not a valid entity path is an
error at start-up that names the field.

## The `robot/state` sample

`RobotStateDecoder` maps the sample's joints by name. The five joint-action arrays are each empty or aligned with
`joint_names`; a joint without an action has a NaN position target (`robot_state_sample.proto`). The effort applied is
`feed-forward + kp (q_des - q) + kd (qd_des - qd)`, 0 without an action. `meas_wrench_contacts_ewrt_world` are in contact
order, in the world frame. The base and the MPC joints become the MPC's generalized coordinates `[p_W, xi, q_j]` and
velocities `[v_W, w_B, dq_j]` through the robot model's measurement helpers (`fillGeneralizedCoordinatesFromMeasurement()`,
`fillGeneralizedVelocitiesFromMeasurement()`), the conversion the MRT controllers use; the generalized forces are sized
as the velocities. Beside them the decoder keeps, for display only, the sample's body angular velocity, the Euler angles
of its quaternion and their rates (`baseEulerAnglesZyx`, `baseEulerZyxRates`).

## Tests

| Target | What it checks |
|---|---|
| `:test_visualization_config` | the fields' defaults, values and refusals, read strictly from a task file; a frame the model lacks is refused, and so are a Pinocchio model or a state manifold of other dimensions than the robot model's and a missing state manifold (by both builders too); every shipped task file sets the fields and its publisher starts |
| `:test_policy_snapshot` | the plan's and the target's sampling: the states along the state manifold's geodesic and the inputs linear (bit for bit `LinearInterpolation`), the node before an event, held outside the horizon, a target without inputs or of other dimensions |
| `:test_robot_state_decoder` | the base in the world frame, `q`'s quaternion and `v`'s body angular velocity, through the robot model's measurement helpers in every formulation; the display's Euler angles and rates: a level turn, tilted bases (the Euler layout's route through the world angular velocity to 1e-12, and the angles' central difference), near the pitch singularity down to `cos(pitch) = 1e-5`, and NaN below 1e-6; joints by name in any order; the PD law of the action; refusals |
| `:test_scene_builder` | for the G1 centroidal, the Atlas with basis-vector inputs and the G1 whole-body MPC: every path once; link poses, plan points, footholds and collision spheres against an independently built Pinocchio model; forces ending at the hand-computed CoP; corner forces equivalent to the wrench; no NaN without force; sample-only and observation-only scenes; the plan recomputed only for a new policy |
| `:test_telemetry_builder` | the group layout against the contract, the display coordinates' and the velocities' names; base, Euler angles, Euler rates (the decoder's, bit for bit), body and world angular velocities, twists, joints, wrenches, the reference and the plan against hand-computed values, the whole-body reference's angular velocity in every frame; the measured acceleration and its reset on a rewind; every value finite, but the Euler rates at the pitch singularity; every formulation |
| `:test_visualization_publisher` | one series per sample in order; a held publisher makes the queue drop and never blocks the feeders; the scene's rate bound; no scene without news; a stale sample giving way to the observation; rejected samples; start and stop; the thread under `SCHED_IDLE` by default, `kInherited` keeping the starting thread's policy; `robot/state` to `viz/telemetry` over loopback buses, published by the bus's IO thread (never `Bus::publish()` on the visualization thread); the MPC node's attacher publishing `viz/telemetry` and `viz/scene` on the node's bus from the robot's samples and its observer, and refusing a null publisher |
| `:test_end_to_end` | the driver (`:visualization_publisher_driver`) -> the bus -> the Rerun bridge's save sink: every link with visuals of all three instances, every marker, exactly the contract's telemetry groups and names, every series in the `.rrd` |
