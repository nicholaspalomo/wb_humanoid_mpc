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
#include <memory>
#include <mutex>
#include <string>

#include "absl/status/status.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_mpc/SystemObservation.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/reference/ReferenceSettings.h"

namespace ocs2::humanoid {

/**
 * Turns operator commands - a walking velocity or a target base pose - into the TargetTrajectories the MPC tracks, within
 * the command limits and reference defaults of the reference file.
 *
 * The formulations derive from it and build the state and input trajectories. The command limits are atomics, so that
 * applyCommandLimits() may run on the solver thread while targets are built on the command thread; the members document
 * which functions are thread-safe.
 */
class TargetTrajectoriesCalculatorBase {
 public:
  /**
   * The calculator of the command limits and reference defaults `referenceSettings` (referenceSettingsFromConfig() of the
   * robot's reference file, so already checked) and the nominal joint posture `defaultJointState`
   * (defaultJointStateFromConfig(), one entry per joint of `mpcRobotModel`).
   */
  TargetTrajectoriesCalculatorBase(const ReferenceSettings& referenceSettings,
                                   const vector_t& defaultJointState,
                                   const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                                   scalar_t mpcHorizon);

  TargetTrajectoriesCalculatorBase(const TargetTrajectoriesCalculatorBase& rhs) = delete;
  TargetTrajectoriesCalculatorBase& operator=(const TargetTrajectoriesCalculatorBase&) = delete;

  virtual ~TargetTrajectoriesCalculatorBase() = default;

  /**
   * Forgets the filter state the command path carries from one target to the next, so that the next target is built
   * exactly as the first one of a freshly constructed calculator: here the low-pass filter of the commanded velocity.
   * A derived calculator with filters of its own overrides this and calls it. The limits and the ground are
   * configuration and are kept. Not thread-safe with respect to building a target: call it from the thread that builds
   * them (the solver thread, through ProceduralMpcMotionManager::setResetHook()).
   */
  virtual void reset();

  /**
   * Applies the command limits and reference defaults of `referenceSettings` (the scalars this class is constructed with,
   * not the nominal joint posture, which the joint targets path owns): on a hot reload of the reference file, which
   * replaces every one of them.
   *
   * Called by the parameter updater when the reference file changes on disk, i.e. from the solver thread, while the
   * command path reads these values from the subscription thread. They are therefore atomics: each is an independent
   * scalar, so a reload that lands between two reads can only mix an old limit with a new one, which is what a slider
   * drag looks like anyway.
   */
  void applyCommandLimits(const ReferenceSettings& referenceSettings);

  /**
   * Where the ground under the commanded base height comes from. `defaultBaseHeight` and a commanded pelvis height are
   * heights ABOVE THE GROUND (WalkingVelocityCommand::desired_pelvis_height says so), while the target is a world pose,
   * so every base height a command produces is written at that height above getTerrainHeight() - see
   * commandedBaseHeight(). A base height taken from the measured state is already a world height and is left alone.
   *
   * Without a source the ground is the task file's top-level `terrain_height` (ModelSettings::terrainHeight of the model
   * this calculator was built with), which is all a command node running outside the MPC can know. A calculator that
   * runs inside the MPC - the one ProceduralMpcMotionManager calls every solve - must be handed
   * SwitchedModelReferenceManager::getAppliedTerrainHeight: the ground the reference manager applied in this very
   * solve, so that a target built after a hot reload of `terrain_height` stands on the new ground while the manager
   * moves the target already in use by the change, exactly once (SwitchedModelReferenceManager::adaptToCurrentGroundHeight).
   * getTerrainHeight() would not do: a reload landing between the manager's run and the target being built would then
   * be applied twice. Thread-safe.
   */
  void setTerrainHeightSource(std::function<scalar_t()> terrainHeightSource);

  /** [m] the ground the commanded base heights stand on: the source's, or the task file's without one. Thread-safe. */
  scalar_t getTerrainHeight() const;

  /**
   * [m] The world height of the base a command asks for: the commanded pelvis height above the ground, or
   * `defaultBaseHeight` above it when the command carries none (a value at or below kMinCommandedPelvisHeight).
   */
  scalar_t commandedBaseHeight(scalar_t commandedPelvisHeight) const;

  /** [m] a commanded pelvis height at or below this is no command, and `defaultBaseHeight` is used instead. */
  static constexpr scalar_t kMinCommandedPelvisHeight = 0.1;

  void setTargetDisplacementVelocity(scalar_t targetDisplacementVelocity) { targetDisplacementVelocity_ = targetDisplacementVelocity; }
  /** [rad/s] the yaw rate a pose command turns at; a reload of the reference file resets it to its target_rotation_velocity. */
  void setTargetRotationVelocity(scalar_t targetRotationVelocity) { targetRotationVelocity_ = targetRotationVelocity; }

  /**
   * Converts command line to TargetTrajectories.
   * @param [in] commadLineTarget : [deltaX, deltaY, deltaZ, deltaYaw] defined in pelvis frame
   * @param [in] observation : the current observation
   */
  virtual TargetTrajectories commandedPositionToTargetTrajectories(const vector4_t& commandedVelocities,
                                                                   scalar_t initTime,
                                                                   const vector_t& initState) = 0;

  /**
   * Converts desired velocities to TargetTrajectories.
   * @param [in] commandedVelocities : [v_x, v_y, v_yaw] defined in pelvis frame
   * @param [in] observation : the current observation
   */
  virtual TargetTrajectories commandedVelocityToTargetTrajectories(const vector4_t& commandedVelocities,
                                                                   scalar_t initTime,
                                                                   const vector_t& initState) = 0;

 protected:
  scalar_t estimateTimeToTarget(const vector_t& desiredBaseDisplacement) const;

  virtual vector6_t getCurrentBasePoseTarget(const vector_t& state) const;

  vector6_t getDeltaBaseTarget(const vector4_t& commadLinePoseTarget, const vector6_t& currentPoseTarget) const;

  /**
   * Low-pass filters the commanded velocity and rotates its linear part from the pelvis frame into the world frame.
   *
   * The filter state lives on the instance, so two calculators (or two tests) never share a hidden global.
   */
  vector4_t filterAndTransformVelCommandToLocal(const vector4_t& commandedVelLocal, scalar_t currentEulerZ, scalar_t filterAlpha);

  /**
   * `currentPose` moved on by `averageVel` (x, y, yaw rate) for `deltaT`, at the world base height `baseHeight`
   * (commandedBaseHeight()), level.
   */
  vector6_t integrateTargetBasePose(const vector6_t& currentPose, const vector3_t& averageVel, scalar_t baseHeight, scalar_t deltaT) const;

  /** The calculator's own clone of the model it was built with: owned, so destroyed with the calculator. */
  const std::unique_ptr<const MpcRobotModelBase<scalar_t>> mpcRobotModelPtr_;

  // For pose control mode
  std::atomic<scalar_t> targetDisplacementVelocity_;
  std::atomic<scalar_t> targetRotationVelocity_;

  // For velocity control mode
  std::atomic<scalar_t> maxDisplacementVelocityX_{0.6};
  std::atomic<scalar_t> maxDisplacementVelocityY_{0.3};
  std::atomic<scalar_t> maxDeltaPelvisHeight_{0.3};
  std::atomic<scalar_t> maxRotationVelocity_{0.6};

  std::atomic<scalar_t> defaultBaseHeight_;
  // The ground under the commanded base height (setTerrainHeightSource). Guarded, because the source is replaced from
  // the thread that wires the calculator while a command thread may be building a target.
  mutable std::mutex terrainHeightSourceMutex_;
  std::function<scalar_t()> terrainHeightSource_;
  vector_t targetJointState_;
  scalar_t mpcHorizon_;
  // State of the first-order low-pass filter applied to the commanded velocity [v_x, v_y, dz, yaw rate].
  vector4_t filteredVelocityCommand_ = vector4_t::Zero();
};

}  // namespace ocs2::humanoid
