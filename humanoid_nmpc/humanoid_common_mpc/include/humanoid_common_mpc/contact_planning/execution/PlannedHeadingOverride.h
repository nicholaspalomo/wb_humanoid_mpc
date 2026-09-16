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

#include "humanoid_common_mpc/acom/AngularCenterOfMass.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningFormulation.h"
#include "humanoid_common_mpc/contact_planning/execution/ExecutionRule.h"

namespace ocs2::humanoid {

/**
 * `planned_heading_override`: the plan's heading replaces the commanded base yaw of the MPC's target trajectory, so
 * that the whole-body controller is asked for the turn the ground can support rather than the raw command. With an
 * ACoM evaluator the reference base yaw is set so that the reference ACoM heading equals the planned heading (the
 * ACoM cost compares the ACoM of the state with the ACoM of the reference state). Lives with the reference manager,
 * which owns the robot model and the evaluator it needs.
 */
class PlannedHeadingOverride final : public ExecutionRule {
 public:
  /** `acom` is the manager's evaluator slot: it may be set after construction and is read at every cycle. */
  PlannedHeadingOverride(const MpcRobotModelBase<scalar_t>& mpcRobotModel, const std::shared_ptr<AngularCenterOfMass>* acom)
      : mpcRobotModel_(&mpcRobotModel), acom_(acom) {}

  std::string describe() const override;
  std::vector<std::string> requiredBlocks() const override { return {term::kHeadingDoubleIntegrator}; }
  void configure(const ContactPlanningConfig& /*config*/) override {}
  void overrideTarget(const ExecutionContext& ctx, TargetTrajectories& targetTrajectories) const override;

 private:
  const MpcRobotModelBase<scalar_t>* mpcRobotModel_;
  const std::shared_ptr<AngularCenterOfMass>* acom_;
};

}  // namespace ocs2::humanoid
