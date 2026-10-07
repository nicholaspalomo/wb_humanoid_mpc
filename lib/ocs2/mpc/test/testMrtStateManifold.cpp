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

#include <gtest/gtest.h>

#include <memory>

#include <ocs2_core/control/ManifoldLinearController.h>
#include <ocs2_core/manifold/ProductStateManifold.h>
#include <ocs2_core/misc/LinearInterpolation.h>

#include "ocs2_mpc/MRT_BASE.h"

namespace ocs2 {
namespace {

/** An MRT whose policy the test hands over directly. */
class HandOverMrt final : public MRT_BASE {
 public:
  void resetMpcNode(const TargetTrajectories& /*initTargetTrajectories*/) override {}
  void setCurrentObservation(const SystemObservation& /*observation*/) override {}
  void handOver(const PrimalSolution& primalSolution) {
    moveToBuffer(std::make_unique<CommandData>(), std::make_unique<PrimalSolution>(primalSolution), std::make_unique<PerformanceIndex>());
  }
};

std::shared_ptr<const ProductStateManifold> attitudeManifold() {
  return ProductStateManifold::create({StateManifoldSegment::euclidean(1), StateManifoldSegment::unitQuaternion()}).value();
}

PrimalSolution planOn(const std::shared_ptr<const ProductStateManifold>& manifold) {
  PrimalSolution plan;
  plan.timeTrajectory_ = {0.0, 0.1, 0.2};
  for (size_t k = 0; k < 3; ++k) {
    vector_t x = vector_t::Random(5);
    manifold->project(x);
    plan.stateTrajectory_.push_back(x);
    plan.inputTrajectory_.push_back(vector_t::Random(2));
  }
  plan.modeSchedule_ = ModeSchedule();
  matrix_array_t gains(3, matrix_t::Random(2, 4));
  plan.controllerPtr_ =
      std::make_unique<ManifoldLinearController>(plan.timeTrajectory_, plan.stateTrajectory_, plan.inputTrajectory_, gains, manifold);
  return plan;
}

TEST(MrtStateManifold, InterpolatesThePlannedStateOnTheManifold) {
  const std::shared_ptr<const ProductStateManifold> manifold = attitudeManifold();
  const PrimalSolution plan = planOn(manifold);
  HandOverMrt mrt;
  mrt.setStateManifold(manifold);
  mrt.handOver(plan);
  ASSERT_TRUE(mrt.updatePolicy());

  vector_t mpcState;
  vector_t mpcInput;
  size_t mode = 0;
  const vector_t measured = plan.stateTrajectory_[1];
  mrt.evaluatePolicy(/*currentTime=*/0.13, measured, mpcState, mpcInput, mode);
  EXPECT_NEAR(mpcState.tail<4>().norm(), 1.0, 1e-15);
  EXPECT_TRUE(mpcState.isApprox(manifold->interpolate(plan.stateTrajectory_[1], plan.stateTrajectory_[2], /*alpha=*/0.3), 1e-12));
  EXPECT_TRUE(mpcInput.isApprox(plan.controllerPtr_->computeInput(/*t=*/0.13, measured), 1e-15));

  // reset() clears the policy, not the manifold.
  mrt.reset();
  EXPECT_EQ(mrt.getStateManifold(), manifold);
}

TEST(MrtStateManifold, WithoutAManifoldThePlannedStateIsInterpolatedLinearly) {
  const PrimalSolution plan = planOn(attitudeManifold());
  HandOverMrt mrt;
  mrt.handOver(plan);
  ASSERT_TRUE(mrt.updatePolicy());
  vector_t mpcState;
  vector_t mpcInput;
  size_t mode = 0;
  mrt.evaluatePolicy(/*currentTime=*/0.13, plan.stateTrajectory_[0], mpcState, mpcInput, mode);
  EXPECT_EQ(mpcState, LinearInterpolation::interpolate(/*enquiryTime=*/0.13, plan.timeTrajectory_, plan.stateTrajectory_));
}

}  // namespace
}  // namespace ocs2
