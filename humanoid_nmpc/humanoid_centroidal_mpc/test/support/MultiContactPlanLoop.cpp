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

#include "support/MultiContactPlanLoop.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "ocs2_core/reference/TargetTrajectories.h"

#include "humanoid_common_mpc/common/StatusMacros.h"

namespace ocs2::humanoid {

absl::StatusOr<std::unique_ptr<MultiContactPlanLoop>>
MultiContactPlanLoop::Create(CentroidalMpcConfig config,
                             const std::string& urdfFile) {
  std::unique_ptr<MultiContactPlanLoop> loop =
      absl::WrapUnique(new MultiContactPlanLoop());
  ASSIGN_OR_RETURN(loop->interface_,
                   CentroidalMpcInterface::Create(std::move(config), urdfFile));
  CentroidalMpcInterface& interface = *loop->interface_;
  if (interface.getPlannedMotionReferenceManagerPtr() == nullptr) {
    return absl::InvalidArgumentError(
        "[MultiContactPlanLoop] the configuration tracks no plan: set "
        "contact_schedule_source: \"multi_contact_plan\".");
  }
  loop->period_ = 1.0 / interface.mpcSettings().mpcDesiredFrequency_;
  loop->mpc_ = std::make_unique<SqpMpc>(
      interface.mpcSettings(), interface.sqpSettings(),
      interface.getOptimalControlProblem(), interface.getInitializer());
  loop->mpc_->getSolverPtr()->setReferenceManager(
      interface.getReferenceManagerPtr());
  loop->mrt_ = std::make_unique<MPC_MRT_Interface>(*loop->mpc_);
  loop->mrt_->setStateManifold(interface.getStateManifold());
  loop->mrt_->initRollout(&interface.getRollout());
  const PlannedMotion& motion =
      interface.getPlannedMotionReferenceManagerPtr()->plannedMotion();
  loop->observation_.time = 0.0;
  loop->observation_.state = interface.getInitialState();
  loop->observation_.input = vector_t::Zero(static_cast<Eigen::Index>(
      interface.getEffectiveMpcRobotModel().getInputDim()));
  loop->observation_.mode = motion.modeSequence().front();
  loop->mrt_->setCurrentObservation(loop->observation_);
  loop->mrt_->resetMpcNode(TargetTrajectories({loop->observation_.time},
                                              {loop->observation_.state},
                                              {loop->observation_.input}));
  return loop;
}

absl::StatusOr<MultiContactPlanLoopStep> MultiContactPlanLoop::step() {
  mrt_->setCurrentObservation(observation_);
  RETURN_IF_ERROR(mrt_->advanceMpc());
  mrt_->updatePolicy();
  MultiContactPlanLoopStep record;
  record.performance = mpc_->getSolverPtr()->getPerformanceIndeces();
  record.planTime = referenceManager().planTimeAt(observation_.time);
  record.holding = referenceManager().clock().isHolding();
  vector_t nextState;
  vector_t nextInput;
  size_t mode = 0;
  mrt_->rolloutPolicy(observation_.time, observation_.state, period_, nextState,
                      nextInput, mode);
  observation_.time += period_;
  observation_.state = std::move(nextState);
  observation_.input = std::move(nextInput);
  observation_.mode = mode;
  return record;
}

}  // namespace ocs2::humanoid
