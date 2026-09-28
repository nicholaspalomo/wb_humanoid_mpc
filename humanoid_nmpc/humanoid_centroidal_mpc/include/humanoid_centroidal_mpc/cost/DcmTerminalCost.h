/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

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
#include <string>

#include <ocs2_core/automatic_differentiation/CppAdInterface.h>
#include <ocs2_core/cost/StateCost.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

namespace ocs2::humanoid {

/**
 * Terminal cost on the Divergent Component of Motion (DCM, capture point) for terminal viability.
 *
 * At the end of the horizon the DCM  xi = c_xy + v_xy / omega  (omega = sqrt(g / comHeight)) is pulled towards
 *   xi_ref = p_support + velocityOffsetFactor * v_cmd / omega,
 * where p_support is the (time-blended) center of the feet in contact at the terminal time (from the mode schedule) and v_cmd the
 * commanded CoM velocity of the target trajectories. With velocityOffsetFactor = 0 this is the classic capturability
 * condition (the robot can come to rest over its terminal support); the velocity offset keeps the condition consistent with
 * a steadily walking robot, whose DCM leads the support point by v / omega, so that the cost does not decelerate the walk.
 * Under an active reduced-order plan the reference is the plan's own DCM instead, and both DCMs are then taken on the
 * plan's pendulum (getParameters).
 *
 * THE PENDULUM. comHeight is the robot's own by default: a `dcm_terminal_cost.comHeight` of 0 means the model's
 * center of mass above its feet at the task file's initialState (computeComHeightAboveFeet, the length every LIP
 * consumer shares), which the cost is given at construction and resolves on every setConfig(), so a hot reload of the
 * block keeps meaning the same thing. A positive value is an explicit override.
 *
 * The cost is 0.5 ||W^(1/2) (xi - xi_ref)||^2 with a Gauss-Newton Hessian, so it stays positive semi-definite for every
 * configuration. It replaces the quadratic Q_final terminal cost on the full state, whose base pose entries are meaningless
 * for varying gait cadences.
 */
class DcmTerminalCost final : public StateCost {
 public:
  /** The key prefix of the block in the task file. */
  static constexpr const char* kConfigPrefix = "dcm_terminal_cost.";
  /** The name the interface adds the cost under to the final costs, and the parameter updater looks it up by. */
  static constexpr const char* kTermName = "dcmTerminalCost";

  struct Config {
    // [m] pendulum height for omega. 0: the model's (resolved by resolveConfig()); positive: an explicit override.
    scalar_t comHeight = 0.0;
    scalar_t gravity = 9.81;              // [m/s^2]
    vector2_t weights{100.0, 100.0};      // [x, y] weights on the DCM error
    scalar_t velocityOffsetFactor = 1.0;  // scales the v_cmd / omega offset of the DCM reference
    scalar_t supportBlendTime = 0.1;      // [s] a foot's support weight ramps 0->1 after touch-down and 1->0 before lift-off

    /** sqrt(gravity / comHeight); meaningful once comHeight is resolved. */
    scalar_t omega() const;

    /**
     * OK, or InvalidArgument naming the dcm_terminal_cost key to change: a negative or non-finite comHeight (0 is
     * allowed and means "the model's"), a gravity that is not positive, a negative weight or blend time.
     */
    absl::Status validate() const;
  };

  /**
   * `config` with a comHeight of 0 replaced by `modelComHeight`, the pendulum length of the robot's model, and validated.
   * InvalidArgument naming dcm_terminal_cost.comHeight when neither the file nor the model gives a positive height.
   */
  static absl::StatusOr<Config> resolveConfig(Config config, scalar_t modelComHeight);

  /**
   * Builds the cost with `config` resolved against `modelComHeight` (resolveConfig), or returns the InvalidArgument that
   * names the key to change.
   *
   * @param modelComHeight [m] the model's pendulum length at the nominal state
   *                       (CentroidalMpcInterface::getNominalComHeight()); what a comHeight of 0 stands for, now and on
   *                       every later setConfig().
   */
  static absl::StatusOr<std::unique_ptr<DcmTerminalCost>> Create(const SwitchedModelReferenceManager& referenceManager,
                                                                 const Config& config,
                                                                 scalar_t modelComHeight,
                                                                 const PinocchioInterface& pinocchioInterface,
                                                                 const MpcRobotModelBase<ad_scalar_t>& mpcRobotModelAD,
                                                                 const std::string& costName,
                                                                 const ModelSettings& modelSettings);

  ~DcmTerminalCost() override = default;
  DcmTerminalCost* clone() const override { return new DcmTerminalCost(*this); }

  bool isActive(scalar_t /*time*/) const override { return isActive_; }
  void setActive(bool active) { isActive_ = active; }

  scalar_t getValue(scalar_t time,
                    const vector_t& state,
                    const TargetTrajectories& targetTrajectories,
                    const PreComputation& preComputation) const override;
  ScalarFunctionQuadraticApproximation getQuadraticApproximation(scalar_t time,
                                                                 const vector_t& state,
                                                                 const TargetTrajectories& targetTrajectories,
                                                                 const PreComputation& preComputation) const override;

  /**
   * Live update of weights / height; the CppAD model is parameterized so no recompilation is needed. A comHeight of 0 is
   * resolved to the model's height the cost was built with. A configuration that does not validate is refused with
   * the InvalidArgument naming its key, and the running one is kept.
   */
  absl::Status setConfig(const Config& config);
  /** The running configuration, comHeight resolved. */
  const Config& getConfig() const { return config_; }
  /** [m] the model's pendulum length a comHeight of 0 resolves to. */
  scalar_t getModelComHeight() const { return modelComHeight_; }

  /**
   * Loads the config from the `dcm_terminal_cost` section of the task file (missing keys keep their defaults, so a
   * missing comHeight is the model's). comHeight is left as the file writes it: 0 is resolved by resolveConfig() or
   * setConfig(). An unreadable file, a value that is not a number and every rejection of Config::validate() are an
   * InvalidArgument naming the key.
   */
  static absl::StatusOr<Config> loadConfig(const std::string& taskFile, const std::string& prefix = kConfigPrefix, bool verbose = false);

  /**
   * Parameter vector [w_left, w_right, omega, v_cmd_x, v_cmd_y, offsetFactor, sqrtW_x, sqrtW_y, plannedWeight,
   * plannedDcm_x, plannedDcm_y]; public for tests.
   * The support weights are continuous in time: 1 while a foot is in contact and further than supportBlendTime from a
   * lift-off or touch-down, ramping linearly through the transitions, so that the terminal reference does not jump when
   * the receding horizon end crosses a mode switch. omega is the configuration's, or the plan's while the reference
   * manager supplies a planned DCM, so that the robot's DCM and the plan's are taken on one pendulum.
   */
  vector_t getParameters(scalar_t time, const TargetTrajectories& targetTrajectories) const;

  /** Unweighted DCM error xi - xi_ref for the given state and parameters; public for tests. */
  vector2_t computeDcmError(const vector_t& state, const vector_t& parameters) const;

  /** Time-blended support weights [w_left, w_right] at `time`; public for tests. */
  vector2_t computeSupportWeights(scalar_t time) const;

 private:
  /** Private: Create() resolves and validates the configuration first. CHECK-fails on an unresolved one. */
  DcmTerminalCost(const SwitchedModelReferenceManager& referenceManager,
                  Config config,
                  scalar_t modelComHeight,
                  const PinocchioInterface& pinocchioInterface,
                  const MpcRobotModelBase<ad_scalar_t>& mpcRobotModelAD,
                  const std::string& costName,
                  const ModelSettings& modelSettings);
  DcmTerminalCost(const DcmTerminalCost& other);
  ad_vector_t residual(const ad_vector_t& state, const ad_vector_t& parameters);

  const SwitchedModelReferenceManager* referenceManagerPtr_;
  Config config_;
  scalar_t modelComHeight_;
  bool isActive_ = true;
  PinocchioInterfaceCppAd pinocchioInterfaceCppAd_;
  std::unique_ptr<MpcRobotModelBase<ad_scalar_t>> mpcRobotModelAdPtr_;
  std::unique_ptr<CppAdInterface> adInterfacePtr_;
};

}  // namespace ocs2::humanoid
