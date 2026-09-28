# Resetting the MPC after a fall

What the controller does when the plant jumps: a fall caught on the gantry, a reset of the simulator, an observation
clock that runs backwards, and every entry into `WB_MPC`. After any of them the controller comes back as it would from
a fresh start at the robot's current (held) state, and no policy solved before reaches the robot.

## 1. What went wrong

The DRC Atlas fell under dodgeballs about 200 m from the origin. The tilt catch locked the gantry, whose weld then
pulled the base back towards the scene's anchor at the origin; MuJoCo found the step unstable, reset the robot to its
model pose and rewound its clock: the controller's observation time went from 340.8 s back to 52.3 s. From then on every
solve failed with `[SqpSolver] Failed to solve QP`, about 6000 times, each followed by a reset request and another
failure, and the joystick produced no step.

The simulator side (the weld anchored where the robot is caught, MuJoCo's automatic reset replaced by one that keeps the
clock and counts itself in `MujocoSimInterface::resetEpoch()`) is described in `MujocoSimInterface.h`. The controller
side had seven faults:

| # | Fault | Fix |
| --- | --- | --- |
| 1 | The sim loops noticed a catch only as a gantry lock between two reads inside one control cycle. The simulator's own resets lock on the simulation thread, so they went unnoticed, and a reset while locked changed nothing a loop read. | `SimFallRecovery` compares the lock and `resetEpoch()` with the PREVIOUS cycle (section 4). |
| 2 | A robot caught at its standing height hung with its feet loaded where they landed, pitched, off its posture; the MPC entered from there fell again within 2 s of the unlock. | The settle sequence (section 4). |
| 3 | The divergence check requested a reset at every control cycle, ZERO_TORQUE ran the MPC path, and a failed solve requested a reset at every failure with no back-off and no escalation to a fuller reset. | Section 3. |
| 4 | `MPC_MRT_Interface::resetMpcNode()` reset the solver's warm start and nothing else. | The full reset (section 2). |
| 5 | The swing-foot cost's yaw residual `atan2(sin(yaw - ref), cos(yaw - ref))` had a NaN derivative at a foot yaw of exactly zero, the yaw of the reset pose. | `CentroidalMpcEndEffectorFootCost::footYawError()`, the half-angle form (section 5). |
| 6 | The whole-body controller had no modes, could not be destroyed (its solver thread looped on `while (true)`) and kept a failed solution for ever. | Section 6. |
| 7 | `MRTPolicySubscriber::resetMpcNode()` is empty. | Correct: it only listens to policies. Commented. |

## 2. The full reset

```
  any thread                      solver thread (between two solves)                 control thread
  ──────────                      ──────────────────────────────────                 ──────────────
  requestMpcReset()  ──ticket──▶  MpcResetSupervisor::takeResetRequest()
                                  MPC_MRT_Interface::resetMpcNode(target from the
                                  observation current NOW):
                                    MRT_BASE::discardBufferedPolicy()   ── new policy epoch
                                    MPC_BASE::reset():
                                      ReferenceManagerInterface::reset()
                                      SolverSynchronizedModule::reset()  (each module)
                                      SolverBase::reset()
                                    ReferenceManager::setTargetTrajectories(target)
                                  completeReset(ticket)
                                  advanceMpc()  ── policy of the new epoch ──▶       updatePolicy()
                                                                                     postResetPolicyActive =
                                                                                       isActivePolicyCurrent() &&
                                                                                       !hasOutstandingReset()
```

`MPC_BASE::reset()` is where the reset lives, so every path gets it: the MRT joint controllers through
`MPC_MRT_Interface`, and the split MPC/MRT nodes through `MPC_ROS_Interface` and its `/mpc_reset` service.
`MPC_BASE::resetSolver()` (`MPC_MRT_Interface::resetMpcSolver()`) resets the solver alone, for the resets of section 3
that must not change what the robot is doing. What each component clears in the full reset, keeping only its
configuration:

| Component | Cleared by its `reset()` |
| --- | --- |
| `ReferenceManager` | the active mode schedule and target, back to the constructor's; anything still in the buffers |
| `SwitchedModelReferenceManager` | the gait schedule (`GaitSchedule::reset()`: the reference file's initial stance schedule and template), the lift-off positions of the feet, the measured base, CoM and yaw inertia; the target's ground offset |
| `ContactPlanningReferenceManager` | the above, the active and the pending plan, the applied schedule, the lift-off history, the foot bookkeeping, the swing latches, the rule outputs, the NMPC prediction, the schedule shifts, the target contact poses and the operator's target; a new plan epoch |
| `ContactPlannerModule` | the snapshot waiting for the worker and the throttle. The planner's own state (`ContactPlannerInterface::reset()`: its warm start, its previous plan) is reset by the thread that plans, at the first snapshot of the new plan epoch; a snapshot of an older epoch is not planned from |
| `ProceduralMpcMotionManager` | the gait (`stance`), the gait-change hold-off, the acceleration ramp; then its reset hook, which resets the target calculator |
| `TargetTrajectoriesCalculatorBase` / `CentroidalMpcTargetTrajectoriesCalculator` | the velocity filter; the joint-state filter and its clock, so that the joint target starts again from the joints the robot has at the reset, however late it comes (it used to decay over the whole time since the filter's clock was last set, and jumped to the nominal joints) |
| `GaitScheduleUpdater` | a gait received but not yet inserted |
| `MRT_BASE` | the policy waiting in the buffer (the policy in use is replaced by the control thread with the first policy of the new epoch) |

The operator's current command and the limits are inputs, not state, and are kept. The contact planner's drop counters
and the planner module's statistics are diagnostics and keep counting.

A plan the contact planner's worker thread computed from a snapshot taken before the reset is refused when it is handed
over (`ContactPlanningReferenceManager::setContactPlan(plan, planEpoch)`): the reference manager's reset, which runs
first, started a new plan epoch. The planner's reset is keyed to that epoch too, not to a request the module's reset
raises: a plan of the old epoch that was under way when the reset came would have consumed such a request and left its
own warm start and previous plan to the first plan after the reset.

A clock that runs backwards is the controllers' to notice (`MpcResetSupervisor::observeTime()`), and they answer it
with a full reset. The reference manager does not second-guess the times it is asked about: the tests query it at
arbitrary times with schedules of their own. One solve can still run on the rewound clock before the reset request is
served; its policy belongs to the epoch before the reset and never reaches the robot. Without a reset, the motion
manager restarts its command ramp and lifts a gait-change hold-off timed on the old clock, as it did before.

## 3. The MRT joint controllers

Both controllers (`CentroidalMpcMrtJointController`, `WBMpcMrtJointController`) use `MpcResetSupervisor`
(`humanoid_common_mpc/mrt/`) for the hand-over with their solver thread.

<!-- LINT.IfChange(controller_reset_events) -->
| Event | Reset | Until a policy of the new epoch is in use |
| --- | --- | --- |
| Entry into `WB_MPC` from a passive mode | yes, from the observation at that moment | the passive mode's action is held (JOINT_PD for ZERO_TORQUE), then ramped into the MPC action over `mpcEntryBlendTime` (0: at once) |
| A discontinuity reported by the sim loop (`requestMpcResetAndHold()`) | yes | held, as above |
| The observation clock running backwards | yes, one | held, as above; the SAFETY decay keeps the time it has decayed for |
| Gantry unlocked (`requestMpcReset()`) | yes | the policy in use carries the robot |
| Divergence of the policy (`|q_des - q| > 0.5 rad`) | of the solver alone (`resetMpcSolver()`), at most one per post-reset policy and per 0.5 s | the policy in use, clamped |
| A failed solve | of the solver alone, before the next attempt | the policy in use; after `maxConsecutiveFailures` (3) failures in a row the reset becomes a full one and the MPC is **unhealthy**: `WB_MPC` holds the robot with the JOINT_PD action, the attempts back off from 0.1 s to 2 s, and one error says so and how to recover (switch to JOINT_PD and back to WB_MPC to retry at once). The first solve that succeeds ends it, and its policy is ramped in like an entry. |
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/mrt/MpcResetSupervisor.h:reset_supervisor_defaults, //humanoid_nmpc/humanoid_centroidal_mpc/include/humanoid_centroidal_mpc/mrt/CentroidalMpcMrtJointController.h:divergence_reset_interval) -->

The divergence check and the first failures reset the solver alone (`MPC_BASE::resetSolver()`, what a reset was
before), because they happen in the middle of a motion that should go on: a full reset restarts the gait in stance under
a robot in mid-stride, and in a headless run of the walking Atlas that made the robot fall within a second of the first
divergence. Failures that persist through a solver reset come from state the solves inherit, and get the full one.

`MPC_MRT_Interface::advanceMpc()` also reports an MPC that cannot run because the observation time has passed the end
of the previous solution (`FailedPrecondition`); it used to report success and never produce a policy again. ZERO_TORQUE
computes no MPC action and runs no check against the policy.

## 4. The gantry catch in the MuJoCo sims

`SimFallRecovery` (`humanoid_common_mpc_ros2/fsm/`, free of ROS) runs once per control cycle in both sim loops, after the
operator's commands:

<!-- LINT.IfChange(settle_sequence) -->
```
      previous cycle's lock and resetEpoch()                         this cycle
                     │                                                    │
                     ▼                                                    ▼
  resetEpoch() moved?  ── yes ──▶ discontinuity (the simulator reset the robot; it also locked the gantry)
  locked, was unlocked? ── yes ──▶ discontinuity (LOCK_GANTRY)
  unlocked, tilt > simMaxBaseTiltAngle? ── yes ──▶ lockGantry(), discontinuity (the tilt catch)
  unlocked, was locked? ── yes ──▶ gantryUnlocked
                     │
   discontinuity ──▶ JOINT_PD (torques on), the loop calls requestMpcResetAndHold() and publishControllerReset(), which
                     counts the reset in the FSM state it publishes (the remote control re-centers its joysticks on
                     every counted reset, every new lock and every transition into a passive mode)
   caught after a fall, simGantryCatchLift > 0 ──▶ settle sequence:

        lift the gantry by simGantryCatchLift ──▶ wait until at rest ──▶ lower to the height of the catch ──▶ wait until at rest
        (0.25 m/s)                                (tilt < 0.05 rad, base < 0.05 m/s and 0.2 rad/s, the MPC joints within 0.15 rad
                                                   of the nominal posture, for 0.5 s; a 5 s timeout moves on with a warning)

        until it is over, WB_MPC is refused: the mode stays JOINT_PD and one line says why
```
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc_ros2/include/humanoid_common_mpc_ros2/fsm/SimFallRecovery.h:settle_defaults) -->

The lift is what lets JOINT_PD bring the legs back to the nominal posture: caught at its standing height the robot's
feet stay loaded where they landed. The entry into `WB_MPC` then resets the MPC from a robot standing still at the
nominal posture. The operator's `LOCK_GANTRY` of a robot that did not fall is a discontinuity without a sequence;
unlocking the gantry ends a sequence.

Task keys (`config/mpc/task.yaml`, simulation only, tied to both sim loops):

<!-- LINT.IfChange(sim_fall_recovery_keys) -->
| Key | Meaning |
| --- | --- |
| `simMaxBaseTiltAngle` | [rad] tilt past which the robot is caught; 0 disables the catch |
| `simGantryCatchLift` | [m] how far the gantry lifts a caught robot to settle it; 0 skips the sequence and accepts WB_MPC at once. Every robot ships 0.15. |
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_centroidal_mpc_ros2/src/CentroidalMpcRobotSim.cpp:sim_fall_recovery_keys, //humanoid_nmpc/humanoid_wb_mpc_ros2/src/WBMpcRobotSim.cpp:sim_fall_recovery_keys) -->

Rest is judged on joint positions, not joint velocities: with the shipped DRC Atlas JOINT_PD gains an unloaded ankle
chatters at the simulator's step rate (the damping gain exceeds the explicit-integration limit 2 I / dt of a free foot),
which the control loop samples as a large constant velocity while the joint holds its position.

## 5. The swing-foot yaw residual

`CentroidalMpcEndEffectorFootCost` tracks a planned foot yaw with its third orientation residual. The residual is now

$$e = 2 \arctan\frac{s}{1 + c},\qquad s = \mathbf{h} \times \mathbf{r},\quad c = \mathbf{h} \cdot \mathbf{r},$$

with $\mathbf{h} = (R_{00}, R_{10}) / \|(R_{00}, R_{10})\|$ the foot's heading and $\mathbf{r}$ the reference heading,
$(\cos\psi_{ref}, \sin\psi_{ref})$ when a yaw reference is tracked and $\mathbf{h}$ itself when not. It equals the old
$\operatorname{atan2}(\sin(\psi - \psi_{ref}), \cos(\psi - \psi_{ref}))$ for every error short of $\pi$ and is exactly
zero when masked. CppAD's `atan2` evaluates both branches, one of which divides by its first argument; the generated
derivative code does not skip the unselected one, so at $R_{10} = 0$ the old residual's derivative was $0 \cdot \infty$,
and the mask made it $0 \cdot \mathrm{NaN}$. The CppAD library name carries the new version (`_yawRefHalfAngle`), so a
cached library of the old residual is never loaded in its place.

## 6. The whole-body controller

`WBMpcMrtJointController` gained the modes of the centroidal one: ZERO_TORQUE, JOINT_PD (the nominal posture with the
gravity torques of the base-held robot), GRAVITY_COMP, SAFETY (the shared decay law of
`humanoid_common_mpc/mrt/SafetyDecay.h`) and WB_MPC with the hold of section 3. `WBMpcRobotSim` now hands it the FSM
mode and the nominal posture: JOINT_PD, GRAVITY_COMP and SAFETY used to leave the whole-body MPC driving a robot caught
on the gantry. Its reset target has two knots and carries the weight on both feet.

## 7. Tests

| Test | What it pins |
| --- | --- |
| `//lib/ocs2:test_mpc_reset` | `MPC_BASE::reset()` resets the reference manager and every module before the solver, `resetSolver()` the solver alone; a policy solved before a reset is never swapped in after it; a stalled MPC is a failure a reset cures |
| `//humanoid_nmpc/humanoid_common_mpc:testMpcResetSupervisor` | the reset hand-over and its two kinds, the escalation from solver to full resets, the back-off, one error per failure episode, the clock check |
| `//humanoid_nmpc/humanoid_common_mpc:testGaitScheduleReset` | a reset gait schedule answers as a fresh one; the gait updater's first-event guard; a gait received before a reset is dropped |
| `//humanoid_nmpc/humanoid_centroidal_mpc:testMpcResetState` | a reset reference stack answers exactly as a fresh one (gait schedule and contact planner); a rewound clock does not hold the robot standing; lift-off positions do not survive; per-instance gait thresholds; plan epochs; the target calculator's filters, and a joint target that starts from the current joints after a late reset |
| `//humanoid_nmpc/humanoid_centroidal_mpc:testMrtJointControllerReset` | ZERO_TORQUE requests no reset; one divergence reset per policy; the back-off holds JOINT_PD and logs one error; a rewound clock is one reset and the hand-over completes; after a fall the first policy is planned from the held robot on a stance schedule |
| `//humanoid_nmpc/humanoid_centroidal_mpc:testFootYawResidual` | parity with the old residual, finite generated derivatives at a yaw of exactly zero (and none for the old residual), the swing-foot cost at the reset pose |
| `//humanoid_nmpc/humanoid_centroidal_mpc:testMpcResetSolverStack` | the real SQP solver: after 11 s of trotting and a full reset the stack solves as a fresh one - the first policy bit for bit, and every solve of the walk that follows to the precision two fresh stacks agree to; every solve after the reset at the initial pose succeeds, the first policy stands where the robot is, and the robot walks again after a rewound clock |
| `//humanoid_nmpc/humanoid_wb_mpc:testWBMpcMrtJointController` | the whole-body controller can be destroyed, JOINT_PD, the entry hold and the reset target, the back-off |
| `//humanoid_nmpc/humanoid_common_mpc_ros2:testSimFallRecovery` | against the real simulator: every simulator reset, tilt catch and lock is one discontinuity; the settle sequence runs to completion before WB_MPC is accepted |
