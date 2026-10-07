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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include "humanoid_nmpc/humanoid_wb_mpc/test/live_tuning/WholeBodySolverStack.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "ocs2_core/reference/ModeSchedule.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_mpc/SystemObservation.h"

#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "humanoid_mpc_config/mpc_parameter_update.nproto.h"
#include "humanoid_nmpc/humanoid_wb_mpc/test/live_tuning/WholeBodyLiveTuningFixture.h"
#include "humanoid_wb_mpc/mrt/WBMpcParameterUpdater.h"
#include "humanoid_wb_mpc/mrt/WBMpcResetTarget.h"

namespace ocs2::humanoid::live_tuning_test {
namespace {

/**
 * Standing until 0.1 s, the right foot in swing over [0.1, 0.45], standing again until 0.55 s, then the left foot in
 * swing until 0.9 s. The brackets far outside keep the gait schedule from tiling its template over it.
 */
// LINT.IfChange(walking_schedule)
ModeSchedule walkingSchedule() {
  contact_flag_t rightSwing = makeFeetArray(true);
  rightSwing[1] = false;
  contact_flag_t leftSwing = makeFeetArray(true);
  leftSwing[0] = false;
  return ModeSchedule(
      /*eventTimesInput=*/{-10.0, 0.1, 0.45, 0.55, 0.9, 10.0},
      /*modeSequenceInput=*/{ModeNumber::kStance, ModeNumber::kStance, stanceLeg2ModeNumber(rightSwing), ModeNumber::kStance,
                             stanceLeg2ModeNumber(leftSwing), ModeNumber::kStance, ModeNumber::kStance});
}
// LINT.ThenChange(//humanoid_nmpc/humanoid_wb_mpc/test/live_tuning/RunningProblemFingerprint.cpp:early_swing_times)

void appendDifference(const vector_t& expected, const vector_t& actual, scalar_t& largest) {
  if (expected.size() != actual.size()) {
    largest = std::numeric_limits<scalar_t>::infinity();
    return;
  }
  for (Eigen::Index i = 0; i < expected.size(); ++i) {
    const scalar_t scale = std::max({1.0, std::abs(expected(i)), std::abs(actual(i))});
    largest = std::max(largest, std::abs(expected(i) - actual(i)) / scale);
  }
}

}  // namespace

WholeBodySolverStack::WholeBodySolverStack(const mpc_config::TaskFile& task, Updater updater) : interface_(createWholeBodyMpc(task)) {
  if (interface_ == nullptr) return;
  WBMpcInterface& interface = *interface_;
  mpc_ = std::make_unique<SqpMpc>(interface.mpcSettings(), interface.sqpSettings(), interface.getOptimalControlProblem(),
                                  interface.getInitializer());
  mpc_->getSolverPtr()->setReferenceManager(interface.getReferenceManagerPtr());
  if (updater == Updater::kRegistered) {
    // No file is watched: the tests hand the updater typed files.
    absl::StatusOr<std::shared_ptr<MpcParameterUpdaterModule>> created =
        makeWholeBodyMpcParameterUpdater(mpc_.get(), interface, /*taskFile=*/"", /*referenceFile=*/"", /*referenceFileReloaders=*/{});
    EXPECT_TRUE(created.ok()) << created.status();
    if (!created.ok()) return;
    updater_ = *std::move(created);
    mpc_->getSolverPtr()->addSynchronizedModule(updater_);
  }
  mrt_ = std::make_unique<MPC_MRT_Interface>(*mpc_);
  observation_.time = 0.0;
  observation_.state = interface.getInitialState();
  observation_.input = vector_t::Zero(static_cast<Eigen::Index>(interface.getMpcRobotModel().getInputDim()));
  observation_.mode = ModeNumber::kStance;
  mrt_->setCurrentObservation(observation_);
  mrt_->resetMpcNode(wbMpcResetTargetTrajectories(observation_, interface.getMpcRobotModel(), interface.getPinocchioInterface()));
  // After the reset, which puts the gait schedule back to the file's (a schedule installed before it was lost, and the
  // stack stood still throughout).
  interface.getSwitchedModelReferenceManagerPtr()->getGaitSchedule()->updateModeSchedule(walkingSchedule());
}

void WholeBodySolverStack::applyNow(const mpc_config::TaskFile& task) {
  ASSERT_NE(updater_, nullptr) << "the stack has no updater";
  mpc_config::MpcParameterUpdate update;
  update.task = task;
  updater_->enqueueParameterUpdate(update);
  updater_->preSolverRun(observation_.time, observation_.time + interface_->mpcSettings().timeHorizon_, observation_.state,
                         *interface_->getReferenceManagerPtr());
}

void WholeBodySolverStack::replanReferences() {
  interface_->getReferenceManagerPtr()->preSolverRun(observation_.time, observation_.time + interface_->mpcSettings().timeHorizon_,
                                                     observation_.state, observation_.mode);
}

std::vector<PrimalSolution> WholeBodySolverStack::solve(size_t count) {
  std::vector<PrimalSolution> policies;
  for (size_t i = 0; i < count; ++i) {
    mrt_->setCurrentObservation(observation_);
    const absl::Status status = mrt_->advanceMpc();
    EXPECT_TRUE(status.ok()) << "the solve at t = " << observation_.time << ": " << status;
    mrt_->updatePolicy();
    policies.push_back(mrt_->getPolicy());
    if (status.ok() && mrt_->initialPolicyReceived()) {
      vector_t state;
      vector_t input;
      size_t mode = 0;
      mrt_->evaluatePolicy(observation_.time + kSolvePeriod, observation_.state, state, input, mode);
      observation_.state = state;
      observation_.mode = mode;
    }
    observation_.time += kSolvePeriod;
  }
  return policies;
}

scalar_t policyDifference(const std::vector<PrimalSolution>& expected, const std::vector<PrimalSolution>& actual) {
  if (expected.size() != actual.size()) return std::numeric_limits<scalar_t>::infinity();
  scalar_t largest = 0.0;
  for (size_t i = 0; i < expected.size(); ++i) {
    const PrimalSolution& a = expected[i];
    const PrimalSolution& b = actual[i];
    if (a.timeTrajectory_ != b.timeTrajectory_ || a.stateTrajectory_.size() != b.stateTrajectory_.size() ||
        a.inputTrajectory_.size() != b.inputTrajectory_.size() || a.modeSchedule_.eventTimes != b.modeSchedule_.eventTimes ||
        a.modeSchedule_.modeSequence != b.modeSchedule_.modeSequence) {
      return std::numeric_limits<scalar_t>::infinity();
    }
    for (size_t k = 0; k < a.stateTrajectory_.size(); ++k) appendDifference(a.stateTrajectory_[k], b.stateTrajectory_[k], largest);
    for (size_t k = 0; k < a.inputTrajectory_.size(); ++k) appendDifference(a.inputTrajectory_[k], b.inputTrajectory_[k], largest);
  }
  return largest;
}

}  // namespace ocs2::humanoid::live_tuning_test
