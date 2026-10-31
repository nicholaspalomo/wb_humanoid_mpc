/******************************************************************************
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.
Copyright (c) 2024, 1X Technologies. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************************************************************/

#pragma once

#include <atomic>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_mpc/SystemObservation.h"
#include "ocs2_oc/synchronized_module/SolverSynchronizedModule.h"

#include "humanoid_common_mpc/command/BasePlanarMotion.h"
#include "humanoid_common_mpc/command/WalkingVelocityCommand.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/config/reference/ReferenceSettings.h"
#include "humanoid_common_mpc/gait/GaitSchedule.h"
#include "humanoid_common_mpc/gait/ModeSequenceTemplate.h"
#include "humanoid_common_mpc/reference_manager/BreakFrequencyAlphaFilter.h"
#include "humanoid_common_mpc/reference_manager/StrideMotionHistory.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "humanoid_mpc_config/gait_file.nproto.h"

namespace ocs2::humanoid {

/**
 * Turns the operator's walking velocity command into the gait and the target
 * trajectories of each solve: it selects the gait from the commanded and the
 * measured motion, inserts it into the gait schedule and builds the targets
 * with the VelocityTargetToTargetTrajectories it is given.
 *
 * The measured motion. Every solve appends the measured base pose of its
 * initial state to a StrideMotionHistory and averages it over the cycle of the
 * gait the schedule runs (the stride): the mean velocity of the base in its
 * heading frame and the mean rate of its twist heading
 * (getMeasuredPlanarMotion()). The targets start from that motion (design
 * defects D1, D1b), and the gait switches compare the command with it, its
 * velocity and its turn alike (D2): the base's instantaneous twist reads the
 * gait's sway as motion (on the recorded straight walks of the G1, the SA01,
 * the R1 and the whole-body G1 the pelvis moves sideways faster than the 0.1
 * m/s below which slow_walk gives way to stance in 48 to 57 % of the samples,
 * and turns faster than the 0.15 rad/s that raises slow_walk), and on the
 * centroidal state a twist without the joint velocities is not the base's at
 * all (StrideMotionHistory says why the mean over exactly one cycle). The
 * velocity is that of the base origin, whose pose the velocity commands' knots
 * move. The CoM's mean velocity over a stride (on the centroidal state the
 * switches read the CoM's instantaneous velocity before) differs from it by the
 * mean of psi_dot J r_h, J = R_z(pi / 2) and r_h the CoM's offset from the base
 * origin in the heading frame: about |omega| |r_h| on a turn, |omega| its rate.
 *
 * Measured windows. A gait switch reads the mean only when its window is
 * measured throughout (measuredOverTheWholeWindow(),
 * StrideMotionHistory::isMeasuredOver(): no restart and no stall within it). A
 * window that reaches back before a restarted history reads its seed, one
 * instant's twist, sway and all: the sway's rate, up to 0.53 rad/s on the G1's
 * recorded walk. Until the window is measured the switches wait, except one
 * that reads no measurement (stance's, on the command alone); the targets read
 * the mean as it is, the seed being the exact base twist of the observation the
 * solve starts from, which the runner hands over with observe().
 *
 * The first cycle of a gait. A window that reaches back into the stride of the
 * gait before reads the heading's sway sigma (the heading about its mean slope)
 * as the turn (sigma(t) - sigma(t - W)) / W, which cancels only when t - W and
 * t are a whole cycle of one gait apart: up to sigma's peak-to-peak over W
 * (0.12 and 0.14 rad/s on the G1's and R1's recorded walks, against the 0.15
 * rad/s that raises slow_walk), and likewise the sideways sway's displacement
 * over W. It is the sway's displacement, bounded, where a seed carries its
 * rate. The switches read it as it is: waiting for a whole stride of each new
 * gait would hold every switch for one more cycle, after the 0.7 of a horizon
 * the schedule takes to start it (from stance to trot at least 8 s, whatever
 * the robot's own acceleration).
 *
 * The stride under the contact planner. The window is the cycle of the gait
 * template the schedule runs. The contact planner
 * (ContactPlanningReferenceManager) retimes the steps, so under it the robot's
 * stride can differ from the template's cycle, and a sway of period P' leaks
 * into the mean by |sinc(pi W / P')| of its rate's amplitude
 * (StrideMotionHistory). No shipped robot runs the planner.
 *
 * Added to the MPC as a SolverSynchronizedModule, whose preSolverRun() runs on
 * the solver thread, as observe() does. The velocity command is set from the
 * command thread under a lock, and the command limits are atomics, so the
 * setters are thread-safe. It is held by std::shared_ptr because OCS2's
 * SolverBase::addSynchronizedModule() takes one; the node, its command watchdog
 * and the command-limit reloader share that pointer rather than observe it,
 * since the solver decides how long the module lives.
 */
class ProceduralMpcMotionManager : public SolverSynchronizedModule {
 public:
  /**
   * Builds the target trajectories of the filtered velocity command
   * `velocityCommand` ([v_x, v_y, pelvis height, yaw rate]) for the solve over
   * [initTime, finalTime] from `initState`, the robot moving at
   * `measuredMotion` (its stride mean, getMeasuredPlanarMotion()).
   */
  using VelocityTargetToTargetTrajectories =
      std::function<TargetTrajectories(const vector4_t& velocityCommand,
                                       scalar_t initTime,
                                       scalar_t finalTime,
                                       const vector_t& initState,
                                       const BasePlanarMotion& measuredMotion)>;

  struct GaitModeStateConfig {
    std::string gaitCommand = "stance";
    scalar_t cmdMinLinvelBaseRtWorldEwrtHeadingXy;
    scalar_t cmdMaxLinvelBaseRtWorldEwrtHeadingXy;
    scalar_t cmdMinAngvelBaseRtWorldEwrtWorldZ;
    scalar_t cmdMaxAngvelBaseRtWorldEwrtWorldZ;
    scalar_t maxErrLinvelBaseRtWorldEwrtHeadingXy;
    scalar_t maxErrAngvelBaseRtWorldEwrtWorldZ;
  };

  /**
   * The motion manager of the gaits of the typed gait file `gaitFile`
   * (gaitMapFromConfig()), with the command limits, ramps and filter of
   * `referenceSettings` (referenceSettingsFromConfig() of the robot's reference
   * file).
   *
   * @param [in] velocityCommandToTargetTrajectories,
   * switchedModelReferenceManagerPtr: as for the path form below.
   * @return gaitMapFromConfig()'s errors, InvalidArgument when the gait file
   * lacks a gait the velocity command selects (checkEveryGaitIsLoaded()), and
   * applyCommandLimits()'s.
   */
  static absl::StatusOr<std::unique_ptr<ProceduralMpcMotionManager>> Create(
      const mpc_config::GaitFile& gaitFile,
      const ReferenceSettings& referenceSettings,
      std::shared_ptr<SwitchedModelReferenceManager>
          switchedModelReferenceManagerPtr,
      const MpcRobotModelBase<scalar_t>& mpcRobotModel,
      VelocityTargetToTargetTrajectories velocityCommandToTargetTrajectories);

  /**
   * The motion manager above of the gait file at `gaitFile` and the reference
   * file at `referenceFile` (loadGaitFile(), loadReferenceFile(),
   * referenceSettingsFromConfig()): the path form of a root of the MPC's
   * configuration.
   *
   * @param [in] gaitFile: The file path that contains the different gait
   * patterns.
   * @param [in] referenceFile: The file path containing the default references
   * and velocity limits.
   * @param [in] velocityCommandToTargetTrajectories: A function which
   * transforms the commanded velocities to TargetTrajectories. It runs every
   * solve, right after the reference manager's own pre-solve hook, so a target
   * calculator behind it must build on
   * SwitchedModelReferenceManager::getAppliedTerrainHeight
   *             (TargetTrajectoriesCalculatorBase::setTerrainHeightSource): the
   * commanded base height then stands on the ground this solve applied, and
   * follows a hot reload of `terrain_height`.
   * @param [in] switchedModelReferenceManagerPtr: A pointer to the switched
   * model reference manager used to update gait and references
   * @return The errors of the loaders and of the typed Create(), the
   * conversions' prefixed with their file.
   */
  static absl::StatusOr<std::unique_ptr<ProceduralMpcMotionManager>> Create(
      const std::string& gaitFile,
      const std::string& referenceFile,
      std::shared_ptr<SwitchedModelReferenceManager>
          switchedModelReferenceManagerPtr,
      const MpcRobotModelBase<scalar_t>& mpcRobotModel,
      VelocityTargetToTargetTrajectories velocityCommandToTargetTrajectories);

  ProceduralMpcMotionManager(
      const ProceduralMpcMotionManager& mpcMotionManager) = delete;
  ProceduralMpcMotionManager& operator=(const ProceduralMpcMotionManager&) =
      delete;
  ~ProceduralMpcMotionManager() override = default;

  /**
   * Method called right before the solver runs
   *
   * @param initTime : start time of the MPC horizon
   * @param finalTime : Final time of the MPC horizon
   * @param initState : State at the start of the MPC horizon
   * @param referenceManager : The ReferenceManager which manages both
   * ModeSchedule and TargetTrajectories.
   */
  void preSolverRun(scalar_t initTime,
                    scalar_t finalTime,
                    const vector_t& initState,
                    const ReferenceManagerInterface& referenceManager) override;

  /**
   * Method called right after the solver runs
   *
   * @param primalSolution : primalSolution
   */
  void postSolverRun(const PrimalSolution& /*primalSolution*/) override {};

  /**
   * Hands over the observation `observation` the next solve starts from, on the
   * solver thread right before it: its input carries the measured joint
   * velocities (getJointVelocities() of its state and input; the centroidal MRT
   * writes them into the input), which the exact base twist of a centroidal
   * state needs (MpcRobotModelBase::getTwistBaseRtWorldEwrtWorld()). The
   * runners call it for every solve: ipc::MpcServer through
   * Hooks::preSolveObserver (MpcNodeRuntime wires it), and InProcessMpcLink
   * through Config::solverObservationObserver. preSolverRun() uses it when its
   * time and state are the solve's own, exactly; otherwise the motion before a
   * (re)started history is taken as rest, with a warning.
   */
  void observe(const SystemObservation& observation);

  /**
   * The measured planar motion of the last solve: the mean over the stride (the
   * cycle of the gait the schedule ran) of the base's velocity in its heading
   * frame and of its heading rate (StrideMotionHistory). Solver thread.
   */
  const BasePlanarMotion& getMeasuredPlanarMotion() const {
    return measuredMotion_;
  }

  /** [s] The window of the last solve's measured motion: the cycle of the gait
   * the schedule ran at its initial time. */
  scalar_t getMeasuredMotionWindow() const { return measuredMotionWindow_; }

  /**
   * Whether the last solve's measured motion is measured over its whole window
   * (StrideMotionHistory::isMeasuredOver(): no seed and no stall in it). Only
   * then do the gait switches read it (the class comment). Solver thread.
   */
  bool measuredOverTheWholeWindow() const {
    return measuredOverTheWholeWindow_;
  }

  /**
   * Returns the gait logic and the command state to their state after
   * construction: the gait is "stance" again (the reference manager's reset
   * restarts the gait schedule in stance) and the gait-change hold-off, the
   * command filter and the acceleration ramp start afresh. The operator's
   * current command is kept - it is an input, not state - and so are the
   * limits. Then the reset hook runs (setResetHook), which is where the owner
   * of the target calculator behind the VelocityTargetToTargetTrajectories
   * function resets it. Solver thread, between solves.
   *
   * The MRT joint controllers reset the MPC when their observation clock runs
   * backwards (MpcResetSupervisor); without a reset, preSolverRun() still
   * restarts the ramp and lifts the gait-change hold-off on a clock that went
   * backwards.
   */
  void reset() override;

  /**
   * A function reset() runs last. The nodes pass one that resets the
   * TargetTrajectoriesCalculatorBase their VelocityTargetToTargetTrajectories
   * function calls, whose filters are state of the same command path.
   */
  void setResetHook(std::function<void()> resetHook) {
    resetHook_ = std::move(resetHook);
  }

  /** The gait the manager is in, by name of the gait file ("stance", "walk",
   * ...). Solver thread. */
  const std::string& getCurrentGaitCommand() const {
    return currentGaitCommand_;
  }

  /**
   * The operator's command, normalized (velocities in [-1, 1] of the command
   * limits, the pelvis height in meters), scaled by the command limits and kept
   * for the next solve. Thread-safe: the MPC node calls it on the bus's IO
   * thread for every operator/walking_velocity_command while the solver thread
   * runs preSolverRun().
   */
  virtual void setAndScaleVelocityCommand(
      const WalkingVelocityCommand& rawVelocityCommand);

  /**
   * The scaled command the next solve starts from (the last
   * setAndScaleVelocityCommand()), before the command filter and the
   * acceleration ramps. Thread-safe; the MPC node sends it with every policy
   * (ViewerAnnotations).
   */
  virtual WalkingVelocityCommand getScaledWalkingVelocityCommand();

  /**
   * Whether the gait `cfg` gives way to the next faster one: the filtered
   * command `velCommandVec` ([v_x, v_y, pelvis height, yaw rate]) is beyond the
   * gait's command limits and the robot already moves near them by its
   * stride-mean motion `strideMeanMotion` (its velocity in the heading frame,
   * the frame of the command limits, and its heading rate;
   * getMeasuredPlanarMotion()). Null while its window is not measured
   * throughout (measuredOverTheWholeWindow()): then only a gait whose
   * thresholds every motion passes, one below zero as stance's, gives way, on
   * the command alone.
   */
  static bool transitionToFasterGait(
      const vector4_t& velCommandVec,
      const BasePlanarMotion* absl_nullable strideMeanMotion,
      const GaitModeStateConfig& cfg);

  /**
   * Whether the gait `cfg` gives way to the next slower one: the command is
   * below the gait's lower limits and the robot's stride-mean motion
   * `strideMeanMotion` (as for transitionToFasterGait()), its velocity and its
   * turn, is near them. Null while its window is not measured throughout: no
   * gait gives way, since no threshold holds for every motion.
   */
  static bool transitionToSlowerGait(
      const vector4_t& velCommandVec,
      const BasePlanarMotion* absl_nullable strideMeanMotion,
      const GaitModeStateConfig& cfg);

  /**
   * Acceleration limits of the velocity reference (reference.textproto
   * `cmd_max_linacc_base_rt_world_ewrt_heading_xy` [m/s^2] and
   * `cmd_max_angacc_base_rt_world_ewrt_world_z` [rad/s^2]; 0 disables the
   * limit, the default). The filtered joypad command is rate-limited toward its
   * value at every solve, so a stick jump becomes a ramp the MPC target and the
   * contact planner both follow: the planner then lengthens the steps
   * progressively instead of answering a step change in the command.
   * InvalidArgument, and the limits are kept, when either is NaN.
   */
  absl::Status setVelocityCommandAccelerationLimits(
      scalar_t cmdMaxLinaccBaseRtWorldEwrtHeadingXy,
      scalar_t cmdMaxAngaccBaseRtWorldEwrtWorldZ);

  /**
   * Applies the command limits, the command ramps and the command filter of
   * `settings` (referenceSettingsFromConfig()): at creation, and on a hot
   * reload of the reference file, which replaces every one of them (the file is
   * what start-up would read).
   *
   * Called by the parameter updater from the solver thread while the command
   * path scales a raw command on the subscription thread, so the limits are
   * atomics; see TargetTrajectoriesCalculatorBase::applyCommandLimits. The
   * ramps and the filter are only read by preSolverRun(), on the solver thread.
   *
   * @return InvalidArgument for a NaN acceleration limit or an invalid break
   * frequency, which referenceSettingsFromConfig() already refuses; the limits
   * are applied before them.
   */
  absl::Status applyCommandLimits(const ReferenceSettings& settings);

  scalar_t getCmdMaxLinaccBaseRtWorldEwrtHeadingXy() const {
    return cmdMaxLinaccBaseRtWorldEwrtHeadingXy_;
  }
  scalar_t getCmdMaxAngaccBaseRtWorldEwrtWorldZ() const {
    return cmdMaxAngaccBaseRtWorldEwrtWorldZ_;
  }
  /** The rate-limited reference of the last solve, [v_x, v_y, pelvis height,
   * yaw rate]. */
  const vector4_t& getRampedVelocityCommand() const {
    return rampedVelocityCommand_;
  }
  /** The gaits of the gait file, by name (tools/config_dump reads them). */
  const std::map<std::string, ModeSequenceTemplate>& gaitMap() const {
    return gaitMap_;
  }

  /** [Hz] The break frequency of the command filter; 0: off. */
  scalar_t getVelocityCommandFilterBreakFrequency() const {
    return velocityCommandFilter_.getBreakFrequency();
  }

  /**
   * Moves `current` toward `target` by at most
   * cmdMaxLinaccBaseRtWorldEwrtHeadingXy * dt in the (v_x, v_y) plane (as a
   * vector, so the direction of the change is preserved) and
   * cmdMaxAngaccBaseRtWorldEwrtWorldZ * dt in the yaw rate; the pelvis height
   * passes through. A limit <= 0 lets that part jump to the target. Exposed for
   * the unit test.
   */
  static vector4_t rateLimitVelocityCommand(
      const vector4_t& target,
      const vector4_t& current,
      scalar_t dt,
      scalar_t cmdMaxLinaccBaseRtWorldEwrtHeadingXy,
      scalar_t cmdMaxAngaccBaseRtWorldEwrtWorldZ);

  /**
   * InvalidArgument naming `gaitFile` and the gait when `gaitMap` lacks a gait
   * of `gaitModeStates`, which the velocity command can select. Construction
   * refuses such a gait file with it. Exposed for the unit test.
   */
  static absl::Status checkEveryGaitIsLoaded(
      const std::map<std::string, ModeSequenceTemplate>& gaitMap,
      absl::Span<const GaitModeStateConfig> gaitModeStates,
      absl::string_view gaitFile);

 protected:
  // The gaits the velocity command selects, slowest first, with their command
  // limits and error thresholds in the order of GaitModeStateConfig's fields. A
  // table of tuned values rather than a constexpr array, which
  // GaitModeStateConfig's std::string rules out, and
  // tools/locomotion_heuristics/test_derive_parameters.py reads its rows out of
  // this initializer.
  // NOLINTBEGIN(magic-number): the tuned thresholds of the gait table, one row per gait (the comment above).
  // clang-format off
  const std::vector<GaitModeStateConfig> gaitModeStates_ {
    { "stance",       -0.1,  0.1, -0.1,  0.1,  10.0,   10.0 }, // Large threshold allows switching away from stance purely command based.
    { "slow_walk",     0.05,  0.3,  0.05,  0.2,    0.05,  0.05},
    { "walk",          0.25,  0.5, 0.15,  0.35,    0.05,  0.05},
    { "slower_trot",   0.45, 0.7, 0.3,  0.55,      0.1,  0.1},
    { "slow_trot",     0.65, 0.9,  0.5,  0.7,      0.2,  0.2},
    { "trot",          0.8,  1.3,  0.65,  10.0,    0.2,  0.2},
    { "run",           1.2,  10.0,  0.65,  10.0,   0.2,  0.2}
  };  // clang-format on
  // NOLINTEND(magic-number)

  size_t currentGaitMode_ = 0;

  WalkingVelocityCommand scaleWalkingVelocityCommand(
      const WalkingVelocityCommand& rawVelocityCommand) const;

  std::shared_ptr<SwitchedModelReferenceManager>
      switchedModelReferenceManagerPtr_;
  std::shared_ptr<GaitSchedule> gaitSchedulePtr_;
  const MpcRobotModelBase<scalar_t>* absl_nonnull mpcRobotModelPtr_;

  VelocityTargetToTargetTrajectories velocityCommandToTargetTrajectoriesFun_;

  std::map<std::string, ModeSequenceTemplate> gaitMap_;

  // For velocity control mode. The values before Create()'s
  // applyCommandLimits() replaces them.
  std::atomic<scalar_t> cmdMaxLinvelBaseRtWorldEwrtHeadingX_{
      kInitCmdMaxLinvelBaseRtWorldEwrtHeadingX};
  std::atomic<scalar_t> cmdMaxLinvelBaseRtWorldEwrtHeadingY_{
      kInitCmdMaxLinvelBaseRtWorldEwrtHeadingY};
  std::atomic<scalar_t> cmdMaxErrPosBaseRtWorldEwrtWorldZ_{
      kInitCmdMaxErrPosBaseRtWorldEwrtWorldZ};
  std::atomic<scalar_t> cmdMaxAngvelBaseRtWorldEwrtWorldZ_{
      kInitCmdMaxAngvelBaseRtWorldEwrtWorldZ};

  // The operator's command filtered on the solver time; off unless
  // reference.textproto sets a break frequency.
  BreakFrequencyAlphaFilter velocityCommandFilter_{vector4_t::Zero()};
  // Written by the thread that receives the operator's commands, read by the
  // solver thread.
  absl::Mutex velocityCommandMutex_;
  WalkingVelocityCommand velocityCommand_
      ABSL_GUARDED_BY(velocityCommandMutex_);

  // Acceleration-limited velocity reference
  // (setVelocityCommandAccelerationLimits)
  scalar_t cmdMaxLinaccBaseRtWorldEwrtHeadingXy_ = 0.0;  // [m/s^2] <= 0: off
  scalar_t cmdMaxAngaccBaseRtWorldEwrtWorldZ_ = 0.0;     // [rad/s^2] <= 0: off
  vector4_t rampedVelocityCommand_ = vector4_t::Zero();
  scalar_t lastRampTime_ = 0.0;
  bool rampInitialized_ = false;

  std::string currentGaitCommand_ = "stance";
  std::string lastGaitCommand_ = "stance";
  // Solver time of the last gait change; gait changes are held off for
  // kGaitChangeHoldOff after it. Lowest until the first change, so that the
  // first solve can change the gait whatever the clock reads.
  scalar_t lastGaitChangeTime_{std::numeric_limits<scalar_t>::lowest()};
  static constexpr scalar_t kGaitChangeHoldOff = 0.2;  // [s]

  std::function<void()> resetHook_;

 private:
  /** A gait the schedule runs from `startTime` on, with the cycle
   * `cycleDuration` [s]. Passive data. */
  struct ScheduledGait {
    scalar_t startTime = 0.0;
    scalar_t cycleDuration = 0.0;
  };

  /** The manager of Create(), before its files are read. */
  ProceduralMpcMotionManager(
      std::shared_ptr<SwitchedModelReferenceManager>
          switchedModelReferenceManagerPtr,
      const MpcRobotModelBase<scalar_t>& mpcRobotModel,
      VelocityTargetToTargetTrajectories velocityCommandToTargetTrajectories);

  /** [s] The cycle of the gait `gaitCommand` of the gait map (loaded by
   * Create()): its template's duration. */
  scalar_t cycleDurationOf(const std::string& gaitCommand) const;

  /** [s] The cycle of the gait the schedule runs at `time` (scheduledGaits_),
   * the stride the measured motion is averaged over. */
  scalar_t strideAt(scalar_t time) const;

  /** Appends the base pose of `initState` at `initTime` to the history and
   * returns its mean over the stride at initTime. */
  BasePlanarMotion updateMeasuredMotion(scalar_t initTime,
                                        const vector_t& initState);

  /** The exact instantaneous planar motion of `initState`, with the joint
   * velocities of the observation when it is the solve's. */
  BasePlanarMotion instantaneousMotionOf(scalar_t initTime,
                                         const vector_t& initState) const;

  // The measured base poses of the solves; its retention, the longest cycle of
  // the gaits the manager selects, is set by Create().
  StrideMotionHistory strideMotion_{/*retention=*/1.0};
  // The gaits the schedule runs, in increasing start time, the first one
  // running now: the reference file's initial STANCE (the gait file's "stance"
  // cycle) until the first switch, then each gait updateGaitSchedule()
  // inserted.
  std::vector<ScheduledGait> scheduledGaits_;
  // The last observe(): what the next solve starts from, if its time and state
  // are the solve's. hasObservation_ false: none.
  bool hasObservation_ = false;
  scalar_t observedTime_ = 0.0;
  vector_t observedState_;
  vector_t observedInput_;
  // The last solve's (getMeasuredPlanarMotion(), getMeasuredMotionWindow(),
  // measuredOverTheWholeWindow()).
  BasePlanarMotion measuredMotion_;
  scalar_t measuredMotionWindow_ = 0.0;
  bool measuredOverTheWholeWindow_ = false;
};

}  // namespace ocs2::humanoid
