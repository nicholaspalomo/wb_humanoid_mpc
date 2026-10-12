# humanoid_mpc_ipc

The MPC link's side of the IPC bus ([`humanoid_nmpc/docs/distributed_runtime/README.md`](../docs/distributed_runtime/README.md)).
It builds on OCS2 (`ocs2_core`, `ocs2_oc`, `ocs2_mpc`), the `humanoid_mpc_msgs` protos and Abseil; the two ends of the
link add the bus (`robot_runtime/robot_ipc`), `robot::TripleBuffer`, `robot_runtime/robot_realtime` and the reset
supervisor (`humanoid_common_mpc:mpc_reset_supervisor`). No Pinocchio and no robot model.

| File | Target | What |
|---|---|---|
| `include/humanoid_mpc_ipc/Topics.h` | `:humanoid_mpc_ipc` | the topic names, `ocs2::humanoid::ipc::topics::kMpcPolicy` etc., and `kAllTopics` |
| `python/humanoid_mpc_ipc/topics.py` | `:topics_py` | the same names for the Python tools: `from humanoid_mpc_ipc import topics` |
| `include/humanoid_mpc_ipc/MpcMessageConversions.h` | `:humanoid_mpc_ipc` | OCS2 types <-> messages |
| `include/humanoid_mpc_ipc/SolutionTimeWindow.h` | `:humanoid_mpc_ipc` | cuts a solution to `mpc.solution_time_window` |
| `include/humanoid_mpc_ipc/PolicyControllers.h` | `:humanoid_mpc_ipc` | a policy's controller as the `FeedforwardController`, `LinearController` or `ManifoldLinearController` the link carries, by its exact type |
| `include/humanoid_mpc_ipc/RemoteMpcLink.h` | `:remote_mpc_link` | the robot side of the network MPC link, an `ocs2::MRT_BASE` |
| `include/humanoid_mpc_ipc/RealtimePolicyEvaluator.h` | `:realtime_policy_evaluator` | evaluates the policy in use on the realtime thread, without allocating or logging |
| `include/humanoid_mpc_ipc/MpcServer.h` | `:mpc_server` | the MPC side: the solve loop, driven by the bus |

The C++ and Python topic names are tied together, and to the topic table of the distributed-runtime README, with
`LINT.IfChange(topics)`; `:test_topics_py` reads `Topics.h` and checks that every constant has an equal Python twin.
No topic may be a prefix of another, because ZeroMQ's SUB filter matches prefixes; both topic tests check that. Besides
the MPC link's streams and the operator's commands, the topics carry the GUI's two-copy Save: `operator/config_save`
(`ConfigFileSave`, the saved file's text for the robot's persistent copy) and its answer `robot/config_save_status`
(`ConfigFileSaveStatus`).

## Conversions

```cpp
void toProto(const SystemObservation&, humanoid_mpc_msgs::SystemObservation*);
absl::Status fromProto(const humanoid_mpc_msgs::SystemObservation&, SystemObservation*);
// ... the same pair for ModeSchedule, TargetTrajectories and PerformanceIndex.

absl::Status policyToProto(const CommandData&, const PrimalSolution&, const PerformanceIndex&, humanoid_mpc_msgs::MpcPolicy*);
// The receiver's state on a manifold, or nullptr for a flat state.
absl::Status policyFromProto(const humanoid_mpc_msgs::MpcPolicy&, std::shared_ptr<const StateManifold>, CommandData*,
                             PrimalSolution*, PerformanceIndex*);

absl::Status checkDimensions(const humanoid_mpc_msgs::SystemObservation&, const ModelDimensions&);
absl::Status checkDimensions(const humanoid_mpc_msgs::MpcPolicy&, const ModelDimensions&);
```

**Lossless.** Doubles travel as doubles, and modes and post-event indices as `uint64`. A round trip reproduces every bit, and a policy rebuilt from a message computes the same input as the one it
was made from at any time and state.

**The controller is rebuilt.** `policyFromProto()` builds a `FeedforwardController`, a `LinearController` or a
`ManifoldLinearController` on the time trajectory, so that the received policy always carries a controller.

**The controllers.** Each travels as one `Vector` per node of `time_trajectory`, tagged by `controller_type`
(`humanoid_mpc_msgs/controller_type.proto`; the codes are the wire's own, not `ocs2::ControllerType`'s):

| `controller_type` | OCS2 controller | Node `k` | Doubles per node | Receiver |
|---|---|---|---|---|
| `CONTROLLER_TYPE_FEEDFORWARD` (1) | `FeedforwardController` | `uff_k` | `nu` | flat or on a state manifold |
| `CONTROLLER_TYPE_LINEAR` (2) | `LinearController` | per input `i`, `uff_k(i)` then row `i` of `K_k` | `nu (nx + 1)` | flat only |
| `CONTROLLER_TYPE_MANIFOLD_LINEAR` (3) | `ManifoldLinearController` | `[u*_k, xbar_k, vec K_k]`, `K_k` column-major | `nu + nx + nu ndx` | on a state manifold only |

A `ManifoldLinearController` is the feedback policy of a solver on a state manifold (`ocs2::StateManifold`, a unit
quaternion in the state), `u = u*_k + K_k (x (-) xbar_k)` blended between the nodes, its gains acting on the `ndx`
tangent directions of the state rather than its `nx` stored entries. Its anchors `xbar_k` travel explicitly because they
are not the state trajectory: `multiple_shooting::toPrimalSolution()` gives every pre-event node the anchor, input and
gain of the node before it, and the last node those of the one before. It is sent only as its own nodes, so
`policyToProto()` refuses one whose time stamps are not the time trajectory (the SQP always builds it there). The
encoder needs no manifold; the receiver decodes on its own (`policyFromProto(message, stateManifold, ...)`,
`RemoteMpcLink::Config::stateManifold`), whose ambient dimension must be the states' and whose tangent dimension gives
`ndx`. A receiver on a manifold refuses `CONTROLLER_TYPE_LINEAR`, whose gains would act on the stored state; a flat
receiver refuses `CONTROLLER_TYPE_MANIFOLD_LINEAR`. A robot and an MPC node that disagree on the state therefore surface
as an invalid policy, not as a wrong input.

**Events.** OCS2's multiple-shooting solvers give the pre- and post-event nodes of an event the same time, and build the
controller on that time trajectory. `ControllerBase::flatten()` at the time trajectory samples the pre-event node for
both, so the received policy would have applied the pre-event input after every event.
`policyToProto()` therefore sends such a controller's own node values, in the layout `flattenSingle()` writes and
`unFlatten()` reads (`:test_mpc_message_conversions` checks both). Only a controller on other time stamps is sampled with
`flatten()`.

**Validation.** The `fromProto()` functions check the whole message before they write anything, so the output is
unchanged on error: finite values, sorted times, matching array lengths, post-event indices in range, a known controller
type, and controller data of the size the node dimensions imply. The `absl::InvalidArgumentError` names the field, e.g.
`MpcPolicy.state_trajectory[3].data[2] is nan; every value must be finite`. Whether the dimensions are the robot
model's is for `checkDimensions()`, given the dimensions of the model the receiver evaluates the policy with: the
state and input sizes, and `numModes`, which every mode of the message (the observation's, each of the mode schedule)
must be below. A humanoid has 4, the contact combinations of its two feet; the controller looks the planned contacts up
by mode (`modeNumber2StanceLeg()`), which has no contact flags for any other.

**Allocation.** The `toProto()` functions resize the fields of the message they write into and keep its nested `Vector`
messages, so a caller that keeps one message object allocates only while the message grows
(`:test_mpc_message_conversions_allocations` proves the steady state allocation-free, serialization into a reused
buffer included). Decoding a policy allocates its arrays and a new controller; decoding an observation or target
trajectories into objects of the right sizes does not allocate. None of this belongs on a realtime thread: the robot's
communication thread decodes the policy and hands it over with `MRT_BASE::moveToBuffer()`.

## The network MPC link

```text
 robot process                                                   MPC process
 realtime thread        bus IO thread                            bus IO thread    solver thread
 setCurrentObservation -> triple buffer -> robot/mpc_observation -> mailbox ----> MpcServer: resets, MPC_BASE::run()
 updatePolicy()        <- moveToBuffer <-- mpc/policy, mpc/status <------------- policy; status after every attempt
 (MRT_BASE try-lock)      RemoteMpcLink
```

**`RemoteMpcLink`** (robot side) is an `ocs2::MRT_BASE`: the controller calls `setCurrentObservation()`,
`updatePolicy()`, `initialPolicyReceived()` and `isActivePolicyCurrent()` on it, and `hasOutstandingReset()` /
`isHealthy()` on its own `MpcResetSupervisor`, as with `MPC_MRT_Interface`. It evaluates the policy in use with a
`RealtimePolicyEvaluator`, which computes what `MRT_BASE::evaluatePolicy()` computes, to the bit, without its heap
allocations (OCS2's interpolation and controllers return by value) and without its logging past the end of the plan.
On a state manifold (`RealtimePolicyEvaluator(dimensions, link.getStateManifold().get())`, the manifold retained) it
evaluates a feedforward or a manifold-linear policy as `MRT_BASE::evaluatePolicy()` does after `setStateManifold()`,
through the manifold's allocation-free `StateManifold::differenceInto()` and `interpolateInto()` into buffers sized at
construction; it reports a policy of the other kind as `kUnsupportedController` rather than leaving it to OCS2's
allocating evaluation.

```cpp
RemoteMpcLink::Config config;
config.dimensions = {.stateDim = nx, .inputDim = nu, .numModes = 4};  // checked on every observation and policy
config.stateManifold = stateManifold;                 // nullptr: a flat state; else its ambient dimension is nx
config.policyTimeout = 0.5;                           // [s, robot clock], mpc_link.policy_timeout
absl::StatusOr<std::unique_ptr<RemoteMpcLink>> link = RemoteMpcLink::Create(bus, resetSupervisor, config);
// ... or RemoteMpcLink::Create(busOptions, resetSupervisor, config) for a bus of its own, started and owned.
bus.start();
```

- `setCurrentObservation()` copies into a `robot::TripleBuffer` slot sized from `Config::dimensions`, and checks its
  time against the deadline of the newest accepted policy (the realtime thread's watchdog, below): no lock and no
  allocation. Everything else runs on the bus's IO thread every `Config::pollPeriod` (1 ms) and for every message:
  publishing the newest observation with the supervisor's request counters (`MpcResetSupervisor::resetsRequested()`),
  the reset handshake, decoding policies into the policy buffer, mirroring the health, the policy timeout.
- Resets: on a new request the IO thread drops the buffered policy and keeps the ticket; a policy whose
  `resets_served` / `full_resets_served` are below the ticket is dropped (counted as stale); the first that serves it is
  moved to the buffer and completes the ticket. `resetMpcNode()` requests a full reset and returns at once.
- Health (`MpcResetSupervisor::setRemoteHealth()`): the MPC node's `MpcSolverStatus`, from every policy and from
  `mpc/status` (subscribed after `mpc/policy`, so that a drain hands a status over after a policy of the same drain; a
  status of an attempt no newer than the newest policy of the same MPC server is ignored; `server_instance` tells a
  restarted MPC node, which counts its solves from zero again, from the one before, so that its statuses are mirrored
  before its first policy), and the link's own: lost when the newest policy was
  solved more than `policyTimeout` of robot time ago, or when the robot clock reaches its end. A reported failure and a
  lost link drop the buffered policy, as the in-process solver's reset after a failure does.
- The realtime thread checks the link's own half itself: the IO thread publishes the deadline of every policy it
  accepts, and `setCurrentObservation()` marks the policy expired in the supervisor
  (`MpcResetSupervisor::setRemotePolicyExpired()`) at the first cycle past it. A stalled or starved IO thread therefore
  cannot leave the controller on a plan past its end; when it runs again it finds the same loss.
- A lost link requests a full reset, as the in-process supervisor does once it declares a failing solver unhealthy:
  the hold ends on a policy solved after it, from the robot as it is then, not on one that resumes the gait schedule
  the MPC kept running through the loss. Only that policy ends the loss; a clock that runs backwards (by more than the
  supervisor's `clockRewindTolerance`, as `observeTime()` counts it) restarts the timeout but does not.
- Dropped: policies solved from an observation this robot did not send (`foreign`), policies older than the timeout on
  arrival or whose horizon has ended by then (`late`), undecodable or misdimensioned ones and ones with a mode outside
  the model (`invalid`). `statistics()` counts everything, and gives `policyAge` for `LoopTiming.policy_age_s` (none
  between a rewind of the clock and the first policy of the new clock) and `stalePoliciesDropped` for
  `LoopTiming.stale_policies_dropped`.
- The callbacks reach the link through a guard its destructor clears, so a bus that outlives the link is harmless.

**Viewer annotations.** When the link accepts a policy, it copies the policy's `annotations` (the contact planner's
target contact patches and the scaled walking command) into a `robot::TripleBuffer<msgs::ViewerAnnotations>`;
`takeAnnotations()` hands the newest to one consumer - the robot process's communication thread, which draws them in the
MuJoCo viewer - once each, lock-free, and without allocating once its output holds as many patches.

**`MpcServer`** (MPC side) owns the solve loop of the MRT joint controllers' `solverWorker()`:

```cpp
MpcServer::Config config;
config.dimensions = {.stateDim = nx, .inputDim = nu, .numModes = 4};
config.mpcDesiredFrequency = mpcSettings.mpcDesiredFrequency_;  // <= 0: one solve per new observation
// MemoryLock::kNone, the default here: SCHED_FIFO without mlockall(), whose MCL_FUTURE would make the solver's
// allocations fail at a finite RLIMIT_MEMLOCK. A setting the thread cannot be given is skipped, the rest applied.
config.solverThread = {.name = "mpc_solver", .priority = 0, .cores = mpcCores, .memoryLock = robot::realtime::MemoryLock::kNone};
MpcServer::Hooks hooks;
hooks.annotationsProvider = ...;  // fills MpcPolicy.annotations
hooks.postSolveObserver = ...;    // the visualization publisher
hooks.preSolveObserver = ...;     // the observation of each solve, before it: the motion manager's observe()
// The formulation's reset target, the one its MRT joint controller hands the in-process link
// (centroidalMpcResetTargetTrajectories(), wbMpcResetTargetTrajectories()).
const MpcServer::ResetTargetTrajectoriesFunction resetTargets = [&](const SystemObservation& observation) {
  return centroidalMpcResetTargetTrajectories(observation, info, effectiveModel, pinocchioInterface);
};
absl::StatusOr<std::unique_ptr<MpcServer>> server = MpcServer::Create(bus, mpc, resetTargets, config, hooks);
bus.start();
(*server)->start();
```

- It waits for the first observation, resets fully from it, then solves the newest observation each time a newer one
  has arrived (after a failed attempt at once, on the newest one). Resets: when the observation's counters exceed what
  it served (full when `full_requested` does), when its own `MpcResetSupervisor` asks after a failure, and fully when
  the robot process restarted (its sequence numbers or counters went backwards). A reset is `MPC_BASE::reset()` or
  `resetSolver()`, then the reference manager's target trajectories from the caller's function.
- Failures: its supervisor backs off and declares the MPC unhealthy exactly as in process; a robot request made after
  the failed attempt ends the back-off early. A request the attempt failed to serve (its reset threw) does not: it
  waits out the back-off with the failure, rather than retrying, and publishing `mpc/status`, as fast as it can.
- Every policy is assembled as `MPC_MRT_Interface::copyToBuffer()` does, cut with `trimToSolutionWindow()` and
  stamped with the robot counters served and the solver status; `mpc/status` follows every attempt.
- `stop()` (and the destructor) joins the solver thread after the attempt in progress; a stopped server stays stopped.

## Tests

```bash
bazel test //humanoid_nmpc/humanoid_mpc_ipc/...
```

| Target | What it pins |
|---|---|
| `:test_topics`, `:test_topics_py` | the topic names in C++ and Python |
| `:test_mpc_message_conversions`, `:test_mpc_message_conversions_allocations` | lossless conversions, of the SQP's manifold policy with an event too (anchors, inputs and gains bitwise, the same input on both sides of the event; wrong strides, a missing manifold and a linear policy on a manifold refused); encoding into a kept message does not allocate |
| `:test_solution_time_window` | upstream OCS2 GaussNewtonDDP's window rule on trajectories, events and the three controllers |
| `:test_policy_controllers` | the three controllers are told apart by their exact type, not by `getType()`: a controller of another class that reports `LINEAR` is refused by the conversions and the evaluator and left alone by the window |
| `:test_remote_mpc_link_protocol` | the robot side against a scripted MPC node: which policies are taken, the handshake, health, link loss (the full reset it requests, the realtime thread's watchdog with the IO thread stalled, rewinds), lifetime |
| `:test_mpc_server` | the MPC side against a scripted robot: resets and stamps, restarts, statuses, window, hooks, pacing, back-off |
| `:test_remote_mpc_link` | both ends over loopback with OCS2's `ScriptedMpc`: a reset with a policy in flight, lost observations, solver failures, the policy timeout |
| `:test_remote_mpc_link_parity` | the same script in process (`MPC_MRT_Interface`) and over the network gives the same policies, cycle by cycle; on a state manifold with a `ManifoldLinearController` and an event too, where the robot's `RealtimePolicyEvaluator` matches the in-process `evaluatePolicy()` on both sides of it |
| `:test_remote_mpc_link_annotations` | the annotations of the newest accepted policy reach `takeAnnotations()`, once each |
| `:test_remote_mpc_link_allocations` | the realtime thread's calls on the link, and `RealtimePolicyEvaluator` on its policy, flat or on a state manifold, allocate nothing, nor do `differenceInto()` / `interpolateInto()`; measures OCS2's `evaluatePolicy()` |
| `:test_realtime_policy_evaluator` | `RealtimePolicyEvaluator` computes the doubles `MRT_BASE::evaluatePolicy()` computes, for the flat controllers and, with the MRT's state manifold set, for a feedforward and a manifold-linear one, at events, past both ends and at either sign of the quaternion, and refuses what it cannot evaluate without touching its outputs |
