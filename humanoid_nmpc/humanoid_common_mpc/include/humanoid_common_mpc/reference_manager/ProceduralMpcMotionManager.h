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

#include "humanoid_common_mpc/command/WalkingVelocityCommand.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/config/reference/ReferenceSettings.h"
#include "humanoid_common_mpc/gait/GaitSchedule.h"
#include "humanoid_common_mpc/gait/ModeSequenceTemplate.h"
#include "humanoid_common_mpc/reference_manager/BreakFrequencyAlphaFilter.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "humanoid_mpc_config/gait_file.nproto.h"

namespace ocs2::humanoid {

/**
 * Turns the operator's walking velocity command into the gait and the target trajectories of each solve: it selects the
 * gait from the commanded and the measured velocity, inserts it into the gait schedule and builds the targets with the
 * VelocityTargetToTargetTrajectories it is given.
 *
 * Added to the MPC as a SolverSynchronizedModule, whose preSolverRun() runs on the solver thread. The velocity command is
 * set from the command thread under a lock, and the command limits are atomics, so the setters are thread-safe.
 */
class ProceduralMpcMotionManager : public SolverSynchronizedModule {
 public:
  using VelocityTargetToTargetTrajectories =
      std::function<TargetTrajectories(const vector4_t& velocityTarget, scalar_t initTime, scalar_t finalTime, const vector_t& initState)>;

  struct GaitModeStateConfig {
    std::string gaitCommand = "stance";
    scalar_t minLinVelCmd;
    scalar_t maxLinVelCmd;
    scalar_t minAngVelCmd;
    scalar_t maxAngVelCmd;
    scalar_t linVelErrorThresh;
    scalar_t angVelErrorThresh;
  };

  /**
   * The motion manager of the gaits of the typed gait file `gaitFile` (gaitMapFromConfig()), with the command limits,
   * ramps and filter of `referenceSettings` (referenceSettingsFromConfig() of the robot's reference file).
   *
   * @param [in] velocityTargetToTargetTrajectories, switchedModelReferenceManagerPtr: as for the path form below.
   * @return gaitMapFromConfig()'s errors, InvalidArgument when the gait file lacks a gait the velocity command selects
   *         (checkEveryGaitIsLoaded()), and applyCommandLimits()'s.
   */
  static absl::StatusOr<std::unique_ptr<ProceduralMpcMotionManager>> Create(
      const mpc_config::GaitFile& gaitFile,
      const ReferenceSettings& referenceSettings,
      std::shared_ptr<SwitchedModelReferenceManager> switchedModelReferenceManagerPtr,
      const MpcRobotModelBase<scalar_t>& mpcRobotModel,
      VelocityTargetToTargetTrajectories velocityTargetToTargetTrajectories);

  /**
   * The motion manager above of the gait file at `gaitFile` and the reference file at `referenceFile` (loadGaitFile(),
   * loadReferenceFile(), referenceSettingsFromConfig()): the path form of a root of the MPC's configuration.
   *
   * @param [in] gaitFile: The file path that contains the different gait patterns.
   * @param [in] referenceFile: The file path containing the default references and velocity limits.
   * @param [in] velocityTargetToTargetTrajectories: A function which transforms the commanded velocities to TargetTrajectories.
   *             It runs every solve, right after the reference manager's own pre-solve hook, so a target calculator
   *             behind it must build on SwitchedModelReferenceManager::getAppliedTerrainHeight
   *             (TargetTrajectoriesCalculatorBase::setTerrainHeightSource): the commanded base height then stands on
   *             the ground this solve applied, and follows a hot reload of `terrain_height`.
   * @param [in] switchedModelReferenceManagerPtr: A pointer to the switched model reference manager used to update gait and references
   * @return The errors of the loaders and of the typed Create(), the conversions' prefixed with their file.
   */
  static absl::StatusOr<std::unique_ptr<ProceduralMpcMotionManager>> Create(
      const std::string& gaitFile,
      const std::string& referenceFile,
      std::shared_ptr<SwitchedModelReferenceManager> switchedModelReferenceManagerPtr,
      const MpcRobotModelBase<scalar_t>& mpcRobotModel,
      VelocityTargetToTargetTrajectories velocityTargetToTargetTrajectories);

  ProceduralMpcMotionManager(const ProceduralMpcMotionManager& mpcMotionManager) = delete;
  ProceduralMpcMotionManager& operator=(const ProceduralMpcMotionManager&) = delete;
  ~ProceduralMpcMotionManager() override = default;

  /**
   * Method called right before the solver runs
   *
   * @param initTime : start time of the MPC horizon
   * @param finalTime : Final time of the MPC horizon
   * @param initState : State at the start of the MPC horizon
   * @param referenceManager : The ReferenceManager which manages both ModeSchedule and TargetTrajectories.
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
   * Returns the gait logic and the command state to their state after construction: the gait is "stance" again (the
   * reference manager's reset restarts the gait schedule in stance) and the gait-change hold-off, the command filter
   * and the acceleration ramp start afresh. The operator's current command is kept - it is an input,
   * not state - and so are the limits. Then the reset hook runs (setResetHook), which is where the owner of the target
   * calculator behind the VelocityTargetToTargetTrajectories function resets it. Solver thread, between solves.
   *
   * The MRT joint controllers reset the MPC when their observation clock runs backwards (MpcResetSupervisor); without a
   * reset, preSolverRun() still restarts the ramp and lifts the gait-change hold-off on a clock that went backwards.
   */
  void reset() override;

  /**
   * A function reset() runs last. The nodes pass one that resets the TargetTrajectoriesCalculatorBase their
   * VelocityTargetToTargetTrajectories function calls, whose filters are state of the same command path.
   */
  void setResetHook(std::function<void()> resetHook) { resetHook_ = std::move(resetHook); }

  /** The gait the manager is in, by name of the gait file ("stance", "walk", ...). Solver thread. */
  const std::string& getCurrentGaitCommand() const { return currentGaitCommand_; }

  /**
   * The operator's command, normalized (velocities in [-1, 1] of the command limits, the pelvis height in meters),
   * scaled by the command limits and kept for the next solve. Thread-safe: the MPC node calls it on the bus's IO thread
   * for every operator/walking_velocity_command while the solver thread runs preSolverRun().
   */
  virtual void setAndScaleVelocityCommand(const WalkingVelocityCommand& rawVelocityCommand);

  /**
   * The scaled command the next solve starts from (the last setAndScaleVelocityCommand()), before the command filter
   * and the acceleration ramps. Thread-safe; the MPC node sends it with every policy (ViewerAnnotations).
   */
  virtual WalkingVelocityCommand getScaledWalkingVelocityCommand();

  static bool transitionToFasterGait(const vector4_t& velCommandVec, const vector6_t& baseVelocity, const GaitModeStateConfig& cfg);

  static bool transitionToSlowerGait(const vector4_t& velCommandVec, const vector6_t& baseVelocity, const GaitModeStateConfig& cfg);

  /**
   * Acceleration limits of the velocity reference (reference.textproto `max_linear_acceleration` [m/s^2] and
   * `max_angular_acceleration` [rad/s^2]; 0 disables the limit, the default). The filtered joypad command is rate-limited
   * toward its value at every solve, so a stick jump becomes a ramp the MPC target and the contact planner both follow:
   * the planner then lengthens the steps progressively instead of answering a step change in the command.
   * InvalidArgument, and the limits are kept, when either is NaN.
   */
  absl::Status setVelocityCommandAccelerationLimits(scalar_t maxLinearAcceleration, scalar_t maxAngularAcceleration);

  /**
   * Applies the command limits, the command ramps and the command filter of `settings` (referenceSettingsFromConfig()):
   * at creation, and on a hot reload of the reference file, which replaces every one of them (the file is what start-up
   * would read).
   *
   * Called by the parameter updater from the solver thread while the command path scales a raw command on the
   * subscription thread, so the limits are atomics; see TargetTrajectoriesCalculatorBase::applyCommandLimits. The ramps
   * and the filter are only read by preSolverRun(), on the solver thread.
   *
   * @return InvalidArgument for a NaN acceleration limit or an invalid break frequency, which referenceSettingsFromConfig()
   *         already refuses; the limits are applied before them.
   */
  absl::Status applyCommandLimits(const ReferenceSettings& settings);

  scalar_t getMaxLinearAcceleration() const { return maxLinearAcceleration_; }
  scalar_t getMaxAngularAcceleration() const { return maxAngularAcceleration_; }
  /** The rate-limited reference of the last solve, [v_x, v_y, pelvis height, yaw rate]. */
  const vector4_t& getRampedVelocityCommand() const { return rampedVelocityCommand_; }
  /** The gaits of the gait file, by name (tools/config_dump reads them). */
  const std::map<std::string, ModeSequenceTemplate>& gaitMap() const { return gaitMap_; }

  /** [Hz] The break frequency of the command filter; 0: off. */
  scalar_t getVelocityCommandFilterBreakFrequency() const { return velocityCommandFilter_.getBreakFrequency(); }

  /**
   * Moves `current` toward `target` by at most maxLinearAcceleration * dt in the (v_x, v_y) plane (as a vector, so the
   * direction of the change is preserved) and maxAngularAcceleration * dt in the yaw rate; the pelvis height passes
   * through. A limit <= 0 lets that part jump to the target. Exposed for the unit test.
   */
  static vector4_t rateLimitVelocityCommand(
      const vector4_t& target, const vector4_t& current, scalar_t dt, scalar_t maxLinearAcceleration, scalar_t maxAngularAcceleration);

  /**
   * InvalidArgument naming `gaitFile` and the gait when `gaitMap` lacks a gait of `gaitModeStates`, which the velocity
   * command can select. Construction refuses such a gait file with it. Exposed for the unit test.
   */
  static absl::Status checkEveryGaitIsLoaded(const std::map<std::string, ModeSequenceTemplate>& gaitMap,
                                             absl::Span<const GaitModeStateConfig> gaitModeStates,
                                             absl::string_view gaitFile);

 protected:
  // clang-format off
  const std::vector<GaitModeStateConfig> gaitModeStates_ {
    { "stance",       -0.1,  0.1, -0.1,  0.1,  10.0,   10.0 }, // Large threshold allows switching aw3ay from stance purely command based.
    { "slow_walk",     0.05,  0.3,  0.05,  0.2,    0.05,  0.05},
    { "walk",          0.25,  0.5, 0.15,  0.35,    0.05,  0.05},
    { "slower_trot",   0.45, 0.7, 0.3,  0.55,      0.1,  0.1},
    { "slow_trot",     0.65, 0.9,  0.5,  0.7,      0.2,  0.2},
    { "trot",          0.8,  1.3,  0.65,  10.0,    0.2,  0.2},
    { "run",           1.2,  10.0,  0.65,  10.0,   0.2,  0.2}
  };  // clang-format on

  size_t currentGaitMode_ = 0;

  WalkingVelocityCommand scaleWalkingVelocityCommand(const WalkingVelocityCommand& rawVelocityCommand) const;

  std::shared_ptr<SwitchedModelReferenceManager> switchedModelReferenceManagerPtr_;
  std::shared_ptr<GaitSchedule> gaitSchedulePtr_;
  const MpcRobotModelBase<scalar_t>* absl_nonnull mpcRobotModelPtr_;

  VelocityTargetToTargetTrajectories velocityTargetToTargetTrajectoriesFun_;

  std::map<std::string, ModeSequenceTemplate> gaitMap_;

  // For velocity control mode
  std::atomic<scalar_t> maxDisplacementVelocityX_{0.6};
  std::atomic<scalar_t> maxDisplacementVelocityY_{0.3};
  std::atomic<scalar_t> maxDeltaPelvisHeight_{0.3};
  std::atomic<scalar_t> maxRotationVelocity_{0.6};

  // The operator's command filtered on the solver time; off unless reference.textproto sets a break frequency.
  BreakFrequencyAlphaFilter velocityCommandFilter_{vector4_t::Zero()};
  // Written by the thread that receives the operator's commands, read by the solver thread.
  absl::Mutex velocityCommandMutex_;
  WalkingVelocityCommand velocityCommand_ ABSL_GUARDED_BY(velocityCommandMutex_);

  // Acceleration-limited velocity reference (setVelocityCommandAccelerationLimits)
  scalar_t maxLinearAcceleration_ = 0.0;   // [m/s^2] <= 0: off
  scalar_t maxAngularAcceleration_ = 0.0;  // [rad/s^2] <= 0: off
  vector4_t rampedVelocityCommand_ = vector4_t::Zero();
  scalar_t lastRampTime_ = 0.0;
  bool rampInitialized_ = false;

  std::string currentGaitCommand_ = "stance";
  std::string lastGaitCommand_ = "stance";
  // Solver time of the last gait change; gait changes are held off for kGaitChangeHoldOff after it. Lowest until the
  // first change, so that the first solve can change the gait whatever the clock reads.
  scalar_t lastGaitChangeTime_{std::numeric_limits<scalar_t>::lowest()};
  static constexpr scalar_t kGaitChangeHoldOff = 0.2;  // [s]

  std::function<void()> resetHook_;

 private:
  /** The manager of Create(), before its files are read. */
  ProceduralMpcMotionManager(std::shared_ptr<SwitchedModelReferenceManager> switchedModelReferenceManagerPtr,
                             const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                             VelocityTargetToTargetTrajectories velocityTargetToTargetTrajectories);
};

}  // namespace ocs2::humanoid
