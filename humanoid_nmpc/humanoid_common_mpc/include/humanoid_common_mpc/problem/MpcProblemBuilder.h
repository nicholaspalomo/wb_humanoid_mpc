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

#include <functional>
#include <memory>
#include <string>

#include <ocs2_core/augmented_lagrangian/AugmentedLagrangian.h>
#include <ocs2_core/constraint/StateConstraint.h>
#include <ocs2_core/constraint/StateInputConstraint.h>
#include <ocs2_core/cost/StateCost.h>
#include <ocs2_core/cost/StateInputCost.h>
#include <ocs2_oc/oc_problem/OptimalControlProblem.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/HumanoidCostConstraintFactory.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/problem/MpcProblemDefinition.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

namespace ocs2::humanoid {

/**
 * Builds and populates an OCS2 OptimalControlProblem based on an
 * MpcProblemDefinition.
 */
class MpcProblemBuilder {
 public:
  using FootTrackingCostBuilder =
      std::function<absl::StatusOr<std::unique_ptr<StateInputCost>>(
          size_t contactIndex, absl::string_view name)>;
  using StanceConstraintBuilder =
      std::function<absl::StatusOr<std::unique_ptr<StateInputConstraint>>(
          size_t contactIndex, absl::string_view name)>;
  using NormalVelocityConstraintBuilder =
      std::function<absl::StatusOr<std::unique_ptr<StateInputConstraint>>(
          size_t contactIndex, absl::string_view name)>;
  using JointMimicConstraintBuilder =
      std::function<absl::StatusOr<std::unique_ptr<StateInputConstraint>>(
          size_t mimicIndex, absl::string_view name)>;
  using TaskSpaceKinematicsCostBuilder =
      std::function<absl::Status(OptimalControlProblem& problem)>;
  using IcpCostBuilder =
      std::function<absl::StatusOr<std::unique_ptr<StateInputCost>>()>;

  struct CustomBuilders {
    FootTrackingCostBuilder footTrackingCostBuilder;
    StanceConstraintBuilder stanceFootConstraintBuilder;
    NormalVelocityConstraintBuilder normalVelocityConstraintBuilder;
    JointMimicConstraintBuilder jointMimicConstraintBuilder;
    TaskSpaceKinematicsCostBuilder taskSpaceKinematicsCostBuilder;
    IcpCostBuilder icpCostBuilder;
  };

  MpcProblemBuilder(const MpcProblemDefinition& problemDefinition,
                    const HumanoidCostConstraintFactory& factory,
                    const ModelSettings& modelSettings,
                    CustomBuilders customBuilders = CustomBuilders());

  ~MpcProblemBuilder() = default;

  /**
   * Populates the given OptimalControlProblem with the configured costs, soft
   * constraints, and equality constraints.
   *
   * @param [in,out] problem: Target OptimalControlProblem to populate.
   * @return absl::Status::Ok() on success, or an error status if any term fails
   * to build.
   */
  absl::Status buildProblem(OptimalControlProblem& problem) const;

 private:
  absl::Status addCosts(OptimalControlProblem& problem) const;
  absl::Status addTerminalCosts(OptimalControlProblem& problem) const;
  absl::Status addStateSoftConstraints(OptimalControlProblem& problem) const;
  absl::Status addSoftConstraints(OptimalControlProblem& problem) const;
  absl::Status addEqualityConstraints(OptimalControlProblem& problem) const;

  MpcProblemDefinition problemDefinition_;
  const HumanoidCostConstraintFactory* factoryPtr_;
  const ModelSettings* modelSettingsPtr_;
  CustomBuilders customBuilders_;
};

}  // namespace ocs2::humanoid
