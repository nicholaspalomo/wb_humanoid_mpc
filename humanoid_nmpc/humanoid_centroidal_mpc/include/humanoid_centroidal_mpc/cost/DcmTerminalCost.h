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
 * where p_support is the (time-blended) centre of the feet in contact at the terminal time (from the mode schedule) and v_cmd the
 * commanded CoM velocity of the target trajectories. With velocityOffsetFactor = 0 this is the classic capturability
 * condition (the robot can come to rest over its terminal support); the velocity offset keeps the condition consistent with
 * a steadily walking robot, whose DCM leads the support point by v / omega, so that the cost does not decelerate the walk.
 *
 * The cost is 0.5 ||W^(1/2) (xi - xi_ref)||^2 with a Gauss-Newton Hessian, so it stays positive semi-definite for every
 * configuration. It replaces the quadratic Q_final terminal cost on the full state, whose base pose entries are meaningless
 * for varying gait cadences.
 */
class DcmTerminalCost final : public StateCost {
 public:
  struct Config {
    scalar_t comHeight = 0.85;            // [m] nominal CoM height for omega
    scalar_t gravity = 9.81;              // [m/s^2]
    vector2_t weights{100.0, 100.0};      // [x, y] weights on the DCM error
    scalar_t velocityOffsetFactor = 1.0;  // scales the v_cmd / omega offset of the DCM reference
    scalar_t supportBlendTime = 0.1;      // [s] a foot's support weight ramps 0->1 after touch-down and 1->0 before lift-off

    scalar_t omega() const;
    void validate() const;
  };

  DcmTerminalCost(const SwitchedModelReferenceManager& referenceManager,
                  Config config,
                  const PinocchioInterface& pinocchioInterface,
                  const MpcRobotModelBase<ad_scalar_t>& mpcRobotModelAD,
                  std::string costName,
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

  /** Live update of weights / height; the CppAD model is parameterised so no recompilation is needed. */
  void setConfig(const Config& config);
  const Config& getConfig() const { return config_; }

  /** Loads the config from the `dcm_terminal_cost` section of the task file (missing keys keep their defaults). */
  static Config loadConfig(const std::string& taskFile, const std::string& prefix = "dcm_terminal_cost.", bool verbose = false);

  /**
   * Parameter vector [w_left, w_right, omega, v_cmd_x, v_cmd_y, offsetFactor, sqrtW_x, sqrtW_y]; public for tests.
   * The support weights are continuous in time: 1 while a foot is in contact and further than supportBlendTime from a
   * lift-off or touch-down, ramping linearly through the transitions, so that the terminal reference does not jump when
   * the receding horizon end crosses a mode switch.
   */
  vector_t getParameters(scalar_t time, const TargetTrajectories& targetTrajectories) const;

  /** Unweighted DCM error xi - xi_ref for the given state and parameters; public for tests. */
  vector2_t computeDcmError(const vector_t& state, const vector_t& parameters) const;

  /** Time-blended support weights [w_left, w_right] at `time`; public for tests. */
  vector2_t computeSupportWeights(scalar_t time) const;

 private:
  DcmTerminalCost(const DcmTerminalCost& other);
  ad_vector_t residual(const ad_vector_t& state, const ad_vector_t& parameters);

  const SwitchedModelReferenceManager* referenceManagerPtr_;
  Config config_;
  bool isActive_ = true;
  PinocchioInterfaceCppAd pinocchioInterfaceCppAd_;
  std::unique_ptr<MpcRobotModelBase<ad_scalar_t>> mpcRobotModelAdPtr_;
  std::unique_ptr<CppAdInterface> adInterfacePtr_;
};

}  // namespace ocs2::humanoid
