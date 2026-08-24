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

#include "humanoid_common_mpc/problem/MpcProblemBuilder.h"

#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"

namespace ocs2::humanoid {

namespace {

inline constexpr size_t kNumLegs = 2;

}  // namespace

MpcProblemBuilder::MpcProblemBuilder(const MpcProblemDefinition& problemDefinition,
                                     const HumanoidCostConstraintFactory& factory,
                                     const ModelSettings& modelSettings,
                                     CustomBuilders customBuilders)
    : problemDefinition_(problemDefinition),
      factoryPtr_(&factory),
      modelSettingsPtr_(&modelSettings),
      customBuilders_(std::move(customBuilders)) {}

absl::Status MpcProblemBuilder::buildProblem(OptimalControlProblem& problem) const {
  const absl::Status costsStatus = addCosts(problem);
  if (!costsStatus.ok()) {
    return costsStatus;
  }

  const absl::Status terminalCostsStatus = addTerminalCosts(problem);
  if (!terminalCostsStatus.ok()) {
    return terminalCostsStatus;
  }

  const absl::Status stateSoftConstraintsStatus = addStateSoftConstraints(problem);
  if (!stateSoftConstraintsStatus.ok()) {
    return stateSoftConstraintsStatus;
  }

  const absl::Status softConstraintsStatus = addSoftConstraints(problem);
  if (!softConstraintsStatus.ok()) {
    return softConstraintsStatus;
  }

  const absl::Status equalityConstraintsStatus = addEqualityConstraints(problem);
  if (!equalityConstraintsStatus.ok()) {
    return equalityConstraintsStatus;
  }

  return absl::OkStatus();
}

absl::Status MpcProblemBuilder::addCosts(OptimalControlProblem& problem) const {
  for (const ProblemTermConfig& term : problemDefinition_.costs) {
    if (!term.enabled) {
      continue;
    }

    if (term.type == "StateInputQuadraticCost") {
      problem.costPtr->add(term.name, factoryPtr_->getStateInputQuadraticCost());
    } else if (term.type == "ICPCost") {
      if (!customBuilders_.icpCostBuilder) {
        return absl::InvalidArgumentError("ICPCost requested but icpCostBuilder was not provided.");
      }
      absl::StatusOr<std::unique_ptr<StateInputCost>> icpCostStatus = customBuilders_.icpCostBuilder();
      if (!icpCostStatus.ok()) {
        return icpCostStatus.status();
      }
      problem.costPtr->add(term.name, std::move(*icpCostStatus));
    } else if (term.type == "TaskSpaceKinematicsCost") {
      if (!customBuilders_.taskSpaceKinematicsCostBuilder) {
        return absl::InvalidArgumentError("TaskSpaceKinematicsCost requested but taskSpaceKinematicsCostBuilder was not provided.");
      }
      const absl::Status status = customBuilders_.taskSpaceKinematicsCostBuilder(problem);
      if (!status.ok()) {
        return status;
      }
    } else if (term.type == "FootTrackingCost" || term.type == "CentroidalMpcEndEffectorFootCost" ||
               term.type == "EndEffectorDynamicsFootCost") {
      if (!customBuilders_.footTrackingCostBuilder) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Foot tracking cost '%s' requested but footTrackingCostBuilder was not provided.", term.type));
      }
      if (term.perContact) {
        for (size_t i = 0; i < kNumLegs; ++i) {
          const absl::string_view footName = modelSettingsPtr_->contactNames[i];
          const std::string termName = absl::StrCat(footName, "_", term.name);
          absl::StatusOr<std::unique_ptr<StateInputCost>> costStatus = customBuilders_.footTrackingCostBuilder(i, termName);
          if (!costStatus.ok()) {
            return costStatus.status();
          }
          problem.costPtr->add(termName, std::move(*costStatus));
        }
      } else {
        absl::StatusOr<std::unique_ptr<StateInputCost>> costStatus = customBuilders_.footTrackingCostBuilder(0, term.name);
        if (!costStatus.ok()) {
          return costStatus.status();
        }
        problem.costPtr->add(term.name, std::move(*costStatus));
      }
    } else if (term.type == "ExternalTorqueQuadraticCost") {
      if (term.perContact) {
        for (size_t i = 0; i < kNumLegs; ++i) {
          const absl::string_view footName = modelSettingsPtr_->contactNames[i];
          const std::string termName = absl::StrCat(footName, "_", term.name);
          problem.costPtr->add(termName, factoryPtr_->getExternalTorqueQuadraticCost(i));
        }
      } else {
        problem.costPtr->add(term.name, factoryPtr_->getExternalTorqueQuadraticCost(0));
      }
    } else {
      return absl::InvalidArgumentError(absl::StrFormat("Unknown cost type: '%s'", term.type));
    }
  }

  return absl::OkStatus();
}

absl::Status MpcProblemBuilder::addTerminalCosts(OptimalControlProblem& problem) const {
  for (const ProblemTermConfig& term : problemDefinition_.terminalCosts) {
    if (!term.enabled) {
      continue;
    }

    if (term.type == "TerminalCost" || term.type == "QuadraticStateCost") {
      problem.finalCostPtr->add(term.name, factoryPtr_->getTerminalCost());
    } else {
      return absl::InvalidArgumentError(absl::StrFormat("Unknown terminal cost type: '%s'", term.type));
    }
  }

  return absl::OkStatus();
}

absl::Status MpcProblemBuilder::addStateSoftConstraints(OptimalControlProblem& problem) const {
  for (const ProblemTermConfig& term : problemDefinition_.stateSoftConstraints) {
    if (!term.enabled) {
      continue;
    }

    if (term.type == "JointLimitsConstraint" || term.type == "JointLimitsSoftConstraint") {
      problem.stateSoftConstraintPtr->add(term.name, factoryPtr_->getJointLimitsConstraint());
    } else if (term.type == "FootCollisionConstraint" || term.type == "FootCollisionSoftConstraint") {
      problem.stateSoftConstraintPtr->add(term.name, factoryPtr_->getFootCollisionConstraint());
    } else {
      return absl::InvalidArgumentError(absl::StrFormat("Unknown state soft constraint type: '%s'", term.type));
    }
  }

  return absl::OkStatus();
}

absl::Status MpcProblemBuilder::addSoftConstraints(OptimalControlProblem& problem) const {
  for (const ProblemTermConfig& term : problemDefinition_.softConstraints) {
    if (!term.enabled) {
      continue;
    }

    if (term.type == "ContactWrenchConeConstraint") {
      if (term.perContact) {
        for (size_t i = 0; i < kNumLegs; ++i) {
          const absl::string_view footName = modelSettingsPtr_->contactNames[i];
          const std::string termName = absl::StrCat(footName, "_", term.name);
          problem.softConstraintPtr->add(termName, factoryPtr_->getContactWrenchConeConstraint(i));
        }
      } else {
        problem.softConstraintPtr->add(term.name, factoryPtr_->getContactWrenchConeConstraint(0));
      }
    } else if (term.type == "FrictionForceConeConstraint") {
      if (term.perContact) {
        for (size_t i = 0; i < kNumLegs; ++i) {
          const absl::string_view footName = modelSettingsPtr_->contactNames[i];
          const std::string termName = absl::StrCat(footName, "_", term.name);
          problem.softConstraintPtr->add(termName, factoryPtr_->getFrictionForceConeConstraint(i));
        }
      } else {
        problem.softConstraintPtr->add(term.name, factoryPtr_->getFrictionForceConeConstraint(0));
      }
    } else if (term.type == "ContactMomentXYConstraint") {
      if (term.perContact) {
        for (size_t i = 0; i < kNumLegs; ++i) {
          const absl::string_view footName = modelSettingsPtr_->contactNames[i];
          const std::string termName = absl::StrCat(footName, "_", term.name);
          problem.softConstraintPtr->add(termName, factoryPtr_->getContactMomentXYConstraint(i, termName));
        }
      } else {
        problem.softConstraintPtr->add(term.name, factoryPtr_->getContactMomentXYConstraint(0, term.name));
      }
    } else if (term.type == "FootCollisionCbfConstraint") {
      problem.softConstraintPtr->add(term.name, factoryPtr_->getFootCollisionCbfConstraint());
    } else {
      return absl::InvalidArgumentError(absl::StrFormat("Unknown soft constraint type: '%s'", term.type));
    }
  }

  return absl::OkStatus();
}

absl::Status MpcProblemBuilder::addEqualityConstraints(OptimalControlProblem& problem) const {
  for (const ProblemTermConfig& term : problemDefinition_.equalityConstraints) {
    if (!term.enabled) {
      continue;
    }

    if (term.type == "ZeroWrenchConstraint") {
      if (term.perContact) {
        for (size_t i = 0; i < kNumLegs; ++i) {
          const absl::string_view footName = modelSettingsPtr_->contactNames[i];
          const std::string termName = absl::StrCat(footName, "_", term.name);
          problem.equalityConstraintPtr->add(termName, factoryPtr_->getZeroWrenchConstraint(i));
        }
      } else {
        problem.equalityConstraintPtr->add(term.name, factoryPtr_->getZeroWrenchConstraint(0));
      }
    } else if (term.type == "ZeroVelocityConstraint" || term.type == "StanceFootConstraint") {
      if (!customBuilders_.stanceFootConstraintBuilder) {
        return absl::InvalidArgumentError("ZeroVelocityConstraint requested but stanceFootConstraintBuilder was not provided.");
      }
      if (term.perContact) {
        for (size_t i = 0; i < kNumLegs; ++i) {
          const absl::string_view footName = modelSettingsPtr_->contactNames[i];
          const std::string termName = absl::StrCat(footName, "_", term.name);
          absl::StatusOr<std::unique_ptr<StateInputConstraint>> conStatus = customBuilders_.stanceFootConstraintBuilder(i, termName);
          if (!conStatus.ok()) {
            return conStatus.status();
          }
          problem.equalityConstraintPtr->add(termName, std::move(*conStatus));
        }
      } else {
        absl::StatusOr<std::unique_ptr<StateInputConstraint>> conStatus = customBuilders_.stanceFootConstraintBuilder(0, term.name);
        if (!conStatus.ok()) {
          return conStatus.status();
        }
        problem.equalityConstraintPtr->add(term.name, std::move(*conStatus));
      }
    } else if (term.type == "NormalVelocityConstraint") {
      if (!customBuilders_.normalVelocityConstraintBuilder) {
        return absl::InvalidArgumentError("NormalVelocityConstraint requested but normalVelocityConstraintBuilder was not provided.");
      }
      if (term.perContact) {
        for (size_t i = 0; i < kNumLegs; ++i) {
          const absl::string_view footName = modelSettingsPtr_->contactNames[i];
          const std::string termName = absl::StrCat(footName, "_", term.name);
          absl::StatusOr<std::unique_ptr<StateInputConstraint>> conStatus = customBuilders_.normalVelocityConstraintBuilder(i, termName);
          if (!conStatus.ok()) {
            return conStatus.status();
          }
          problem.equalityConstraintPtr->add(termName, std::move(*conStatus));
        }
      } else {
        absl::StatusOr<std::unique_ptr<StateInputConstraint>> conStatus = customBuilders_.normalVelocityConstraintBuilder(0, term.name);
        if (!conStatus.ok()) {
          return conStatus.status();
        }
        problem.equalityConstraintPtr->add(term.name, std::move(*conStatus));
      }
    } else if (term.type == "JointMimicConstraint" || term.type == "JointMimicKinematicConstraint") {
      if (!customBuilders_.jointMimicConstraintBuilder) {
        return absl::InvalidArgumentError("JointMimicConstraint requested but jointMimicConstraintBuilder was not provided.");
      }
      if (term.perContact) {
        for (size_t i = 0; i < kNumLegs; ++i) {
          const absl::string_view footName = modelSettingsPtr_->contactNames[i];
          const std::string termName = absl::StrCat(footName, "_", term.name);
          absl::StatusOr<std::unique_ptr<StateInputConstraint>> conStatus = customBuilders_.jointMimicConstraintBuilder(i, termName);
          if (!conStatus.ok()) {
            return conStatus.status();
          }
          problem.equalityConstraintPtr->add(termName, std::move(*conStatus));
        }
      } else {
        absl::StatusOr<std::unique_ptr<StateInputConstraint>> conStatus = customBuilders_.jointMimicConstraintBuilder(0, term.name);
        if (!conStatus.ok()) {
          return conStatus.status();
        }
        problem.equalityConstraintPtr->add(term.name, std::move(*conStatus));
      }
    } else {
      return absl::InvalidArgumentError(absl::StrFormat("Unknown equality constraint type: '%s'", term.type));
    }
  }

  return absl::OkStatus();
}

}  // namespace ocs2::humanoid
