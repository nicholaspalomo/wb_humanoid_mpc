/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

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

#include <memory>
#include <optional>
#include <string>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "ocs2_core/PhysicalConstants.h"
#include "ocs2_core/PreComputation.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/automatic_differentiation/CppAdInterface.h"
#include "ocs2_core/automatic_differentiation/Types.h"
#include "ocs2_core/cost/StateCost.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

namespace ocs2::humanoid {

/**
 * Terminal cost on the Divergent Component of Motion (DCM, capture point) for
 * terminal viability.
 *
 * At the end of the horizon the DCM  xi = c_xy + v_xy / omega  (omega = sqrt(g
 * / nomPosComRtFeetmeanEwrtWorldZ)) is pulled towards xi_ref = p_support +
 * velocityCommandOffsetFactor * v_ref / omega, where p_support is the
 * (time-blended) center of the feet in contact at the terminal time (from the
 * mode schedule) and v_ref the reference CoM velocity of the target
 * trajectories (the planned one under contact planning). With
 * velocityCommandOffsetFactor = 0 this is the classic capturability condition
 * (the robot can come to rest over its terminal support); the velocity offset
 * keeps the condition consistent with a steadily walking robot, whose DCM leads
 * the support point by v / omega, so that the cost does not decelerate the
 * walk. Under an active reduced-order plan the reference is the plan's own DCM
 * instead, and both DCMs are then taken on the plan's pendulum (getParameters).
 *
 * THE PENDULUM. nomPosComRtFeetmeanEwrtWorldZ is the robot's own by default: a
 * `dcm_terminal_cost` block without `nom_pos_com_rt_feetmean_ewrt_world_z` (a
 * nomPosComRtFeetmeanEwrtWorldZ of nullopt here) means the model's center of
 * mass above its feet at the task file's initial_state
 * (computeProxyPosComRtFeetmeanEwrtWorldZ, the length every LIP consumer
 * shares), which the cost is given at construction and resolves on every
 * setConfig(), so a hot reload of the block keeps meaning the same thing. A
 * value is an explicit override.
 *
 * The cost is 0.5 ||W^(1/2) R_z(psi)^T (xi - xi_ref)||^2 with a Gauss-Newton
 * Hessian, so it stays positive semi-definite for every configuration. W =
 * diag(weight_x, weight_y) weighs the longitudinal and the lateral DCM error:
 * the error in the axes of the heading frame, whose yaw psi is the target's
 * base heading at the terminal time (world x without a target). psi is a
 * parameter, not a function of the state. It replaces the
 * quadratic final_state_weights terminal cost on the full state, whose base
 * pose entries are meaningless for varying gait cadences.
 *
 * Not thread-safe: the generated CppAD model keeps evaluation buffers and
 * setConfig() replaces the configuration, so one object is used by one thread
 * at a time; the solver clones the cost per worker thread and reconfigures it
 * between solves. Build it with Create(), which resolves and validates the
 * configuration.
 */
class DcmTerminalCost final : public StateCost {
 public:
  /** The field path prefix of the block in the task file. */
  static constexpr char kConfigPrefix[] = "dcm_terminal_cost.";
  /** The name the interface adds the cost under to the final costs, and the
   * parameter updater looks it up by. */
  static constexpr char kTermName[] = "dcmTerminalCost";
  /**
   * The name the interface builds the cost's CppAD library under (the library
   * is "<it>_p<parameters>"). The same string as kTermName today, defined apart
   * on purpose: a rename of the term must not rename, and regenerate, the
   * library (testCostTermAndLibraryNames).
   */
  static constexpr char kLibraryName[] = "dcmTerminalCost";

  struct Config {
    /** The default weight of each axis of the DCM error. */
    static constexpr scalar_t kDefaultWeight = 100.0;
    /** [s] supportBlendTime's default. */
    static constexpr scalar_t kDefaultSupportBlendTime = 0.1;

    // [m] pendulum height for omega. nullopt: the model's (resolveConfig()
    // fills it in), what a file without
    // dcm_terminal_cost.nom_pos_com_rt_feetmean_ewrt_world_z converts to; a
    // value: an explicit override, positive.
    std::optional<scalar_t> nomPosComRtFeetmeanEwrtWorldZ;
    // [m/s^2] the robot model's gravity (modelGravity()), the one its dynamics
    // use; not a file key. resolveConfig() writes the model's, at start-up and
    // on every setConfig().
    scalar_t gravityMagnitude = kGravityMagnitude;
    // The weights on the longitudinal and the lateral DCM error, the error in
    // the heading frame's x and y axes.
    vector2_t weights{kDefaultWeight, kDefaultWeight};
    scalar_t velocityCommandOffsetFactor =
        1.0;  // scales the v_ref / omega offset of the DCM reference
    // [s] a foot's support weight ramps 0->1 after touch-down and 1->0 before
    // lift-off
    scalar_t supportBlendTime = kDefaultSupportBlendTime;

    /** [1/s] sqrt(gravity / nomPosComRtFeetmeanEwrtWorldZ); empty while
     * nomPosComRtFeetmeanEwrtWorldZ is not resolved (resolveConfig()). */
    std::optional<scalar_t> omega() const;

    /**
     * OK, or InvalidArgument naming the dcm_terminal_cost field to change: a
     * nom_pos_com_rt_feetmean_ewrt_world_z that is given and is not a positive
     * number, a negative weight or blend time. The gravity is the model's,
     * which modelGravity() checks.
     */
    absl::Status validate() const;
  };

  /**
   * `config` with an unset nomPosComRtFeetmeanEwrtWorldZ set to
   * `modelNomPosComRtFeetmeanEwrtWorldZ`, the pendulum length of the robot's
   * model, its gravity set to `modelGravityMagnitude`, the model's, and
   * validated. InvalidArgument naming
   * dcm_terminal_cost.nom_pos_com_rt_feetmean_ewrt_world_z when neither the
   * file nor the model gives a positive height.
   */
  static absl::StatusOr<Config> resolveConfig(
      Config config,
      scalar_t modelNomPosComRtFeetmeanEwrtWorldZ,
      scalar_t modelGravityMagnitude);

  /**
   * Builds the cost with `config` resolved against
   * `modelNomPosComRtFeetmeanEwrtWorldZ` and the gravity of
   * `pinocchioInterface`'s model (resolveConfig, modelGravity()), or returns
   * the InvalidArgument that names the field to change. A contact set other
   * than the biped's two 6-DoF feet (ModelSettings::isBipedContactSet()) is an
   * InvalidArgument naming dcm_terminal_cost and the contact count.
   *
   * @param modelNomPosComRtFeetmeanEwrtWorldZ [m] the model's pendulum length
   * at the nominal state
   *                       (CentroidalMpcInterface::getNomPosComRtFeetmeanEwrtWorldZ());
   * what an unset nomPosComRtFeetmeanEwrtWorldZ stands for, now and on every
   * later setConfig().
   */
  static absl::StatusOr<std::unique_ptr<DcmTerminalCost>> Create(
      const SwitchedModelReferenceManager& referenceManager,
      const Config& config,
      scalar_t modelNomPosComRtFeetmeanEwrtWorldZ,
      const PinocchioInterface& pinocchioInterface,
      const MpcRobotModelBase<ad_scalar_t>& mpcRobotModelAD,
      const std::string& costName,
      const ModelSettings& modelSettings);

  ~DcmTerminalCost() override = default;
  DcmTerminalCost* absl_nonnull clone() const override {
    return new DcmTerminalCost(*this);
  }
  // Copied only by clone(), whose copy constructor is private; never assigned
  // or moved.
  DcmTerminalCost& operator=(const DcmTerminalCost&) = delete;
  DcmTerminalCost(DcmTerminalCost&&) = delete;
  DcmTerminalCost& operator=(DcmTerminalCost&&) = delete;

  bool isActive(scalar_t /*time*/) const override { return isActive_; }
  void setActive(bool active) { isActive_ = active; }

  scalar_t getValue(scalar_t time,
                    const vector_t& state,
                    const TargetTrajectories& targetTrajectories,
                    const PreComputation& preComputation) const override;
  ScalarFunctionQuadraticApproximation getQuadraticApproximation(
      scalar_t time,
      const vector_t& state,
      const TargetTrajectories& targetTrajectories,
      const PreComputation& preComputation) const override;

  /**
   * Live update of weights / height; the CppAD model is parameterized so no
   * recompilation is needed. An unset nomPosComRtFeetmeanEwrtWorldZ is resolved
   * to the model's height the cost was built with, and the gravity is always
   * the model's. A configuration that does not validate is refused with the
   * InvalidArgument naming its field, and the running one is kept.
   */
  absl::Status setConfig(const Config& config);
  /** The running configuration, nomPosComRtFeetmeanEwrtWorldZ resolved. */
  const Config& getConfig() const { return config_; }
  /** [1/s] the running configuration's omega, sqrt(gravity /
   * nomPosComRtFeetmeanEwrtWorldZ) of the resolved
   * nomPosComRtFeetmeanEwrtWorldZ. */
  scalar_t omega() const { return omega_; }
  /** [m] the model's pendulum length an unset nomPosComRtFeetmeanEwrtWorldZ
   * resolves to. */
  scalar_t getModelNomPosComRtFeetmeanEwrtWorldZ() const {
    return modelNomPosComRtFeetmeanEwrtWorldZ_;
  }
  /** [m/s^2] the model's gravity, which every configuration resolves to. */
  scalar_t getModelGravityMagnitude() const { return modelGravityMagnitude_; }

  /**
   * Parameter vector [w_left, w_right, omega, v_ref_x, v_ref_y, offsetFactor,
   * sqrtW_x, sqrtW_y, plannedWeight, plannedDcm_x, plannedDcm_y,
   * refYawHeadingRtWorld]; public for tests. refYawHeadingRtWorld [rad] is the
   * heading of the target's base at `time` (MpcRobotModelBase::
   * getYawBaseRtWorld()), 0 without a target, and sets the axes sqrtW_x and
   * sqrtW_y act in. The support weights are continuous in time: 1 while a foot
   * is in contact and further than supportBlendTime from a lift-off or
   * touch-down, ramping linearly through the transitions, so that the terminal
   * reference does not jump when the receding horizon end crosses a mode
   * switch. omega is the configuration's, or the plan's while the reference
   * manager supplies a planned DCM, so that the robot's DCM and the plan's are
   * taken on one pendulum.
   */
  vector_t getParameters(scalar_t time,
                         const TargetTrajectories& targetTrajectories) const;

  /** Unweighted DCM error xi - xi_ref in world axes for the given state and
   * parameters (their weights and heading ignored); public for tests. */
  vector2_t computeErrPosDcmRtWorldEwrtWorldXy(
      const vector_t& state, const vector_t& parameters) const;

  /**
   * d computeErrPosDcmRtWorldEwrtWorldXy() / d state (2 x the state dimension),
   * from the same library: the Jacobian whose product with the weights
   * getQuadraticApproximation()'s Gauss-Newton Hessian is; public for tests.
   */
  matrix_t computeJacobianErrPosDcmRtWorldEwrtWorldXy(
      const vector_t& state, const vector_t& parameters) const;

  /** Time-blended support weights [w_left, w_right] at `time`; public for
   * tests. */
  vector2_t computeSupportWeights(scalar_t time) const;

 private:
  /** Private: Create() resolves and validates the configuration first.
   * CHECK-fails on an unresolved one. */
  DcmTerminalCost(const SwitchedModelReferenceManager& referenceManager,
                  Config config,
                  scalar_t modelNomPosComRtFeetmeanEwrtWorldZ,
                  scalar_t modelGravityMagnitude,
                  const PinocchioInterface& pinocchioInterface,
                  const MpcRobotModelBase<ad_scalar_t>& mpcRobotModelAD,
                  const std::string& costName,
                  const ModelSettings& modelSettings);
  DcmTerminalCost(const DcmTerminalCost& other);

  const SwitchedModelReferenceManager* absl_nonnull referenceManagerPtr_;
  Config config_;
  scalar_t modelNomPosComRtFeetmeanEwrtWorldZ_;
  // [m/s^2] the gravity of the model the cost was built on (modelGravity()).
  scalar_t modelGravityMagnitude_;
  scalar_t omega_ = 0.0;  // [1/s] config_.omega() of the resolved config_, set
                          // with every config_
  bool isActive_ = true;
  std::unique_ptr<MpcRobotModelBase<ad_scalar_t>> mpcRobotModelAdPtr_;
  std::unique_ptr<CppAdInterface> adInterfacePtr_;
};

}  // namespace ocs2::humanoid
