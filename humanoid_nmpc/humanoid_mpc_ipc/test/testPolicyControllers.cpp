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

// asFeedforwardController() and asLinearController() find the exact type of a policy's controller: each finds its own
// type and no other, from a const or a mutable controller, and a StateBasedLinearController, whose getType() reports the
// LinearController it wraps, is neither. So the conversions, the solution time window and the realtime evaluator all
// refuse or skip it alike, as they do every controller the link does not carry.

#include <cstddef>
#include <random>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/control/ControllerBase.h"
#include "ocs2_core/control/ControllerType.h"
#include "ocs2_core/control/FeedforwardController.h"
#include "ocs2_core/control/LinearController.h"
#include "ocs2_core/control/StateBasedLinearController.h"
#include "ocs2_mpc/CommandData.h"
#include "ocs2_oc/oc_data/PerformanceIndex.h"
#include "ocs2_oc/oc_data/PrimalSolution.h"

#include "humanoid_mpc_ipc/MpcMessageConversions.h"
#include "humanoid_mpc_ipc/PolicyControllers.h"
#include "humanoid_mpc_ipc/RealtimePolicyEvaluator.h"
#include "humanoid_mpc_ipc/SolutionTimeWindow.h"
#include "humanoid_mpc_msgs/mpc_policy.pb.h"
#include "humanoid_nmpc/humanoid_mpc_ipc/test/PolicyTestData.h"

namespace ocs2::humanoid::ipc {
namespace {

using test_data::PolicyShape;

/** A controller of a type the link does not carry, which claims to be a LinearController. */
class ImpostorController final : public ControllerBase {
 public:
  vector_t computeInput(scalar_t /*t*/, const vector_t& /*x*/) override { return vector_t(); }
  void concatenate(const ControllerBase* absl_nonnull /*otherController*/, int /*index*/, int /*length*/) override {}
  int size() const override { return 1; }
  ControllerType getType() const override { return ControllerType::LINEAR; }
  void clear() override {}
  bool empty() const override { return false; }
  ImpostorController* absl_nonnull clone() const override { return new ImpostorController(*this); }
};

TEST(PolicyControllersTest, AFeedforwardControllerIsOneAndNoLinearController) {
  FeedforwardController controller(/*controllerTime=*/{0.0, 1.0}, /*controllerFeedforward=*/{vector_t::Ones(2), vector_t::Zero(2)});
  const ControllerBase& constBase = controller;
  ControllerBase& base = controller;
  EXPECT_EQ(asFeedforwardController(constBase), &controller);
  EXPECT_EQ(asFeedforwardController(base), &controller);
  EXPECT_EQ(asLinearController(constBase), nullptr);
  EXPECT_EQ(asLinearController(base), nullptr);
}

TEST(PolicyControllersTest, ALinearControllerIsOneAndNoFeedforwardController) {
  LinearController controller(/*controllerTime=*/{0.0, 1.0}, /*controllerBias=*/{vector_t::Ones(2), vector_t::Zero(2)},
                              /*controllerGain=*/{matrix_t::Identity(2, 3), matrix_t::Zero(2, 3)});
  const ControllerBase& constBase = controller;
  ControllerBase& base = controller;
  EXPECT_EQ(asLinearController(constBase), &controller);
  EXPECT_EQ(asLinearController(base), &controller);
  EXPECT_EQ(asFeedforwardController(constBase), nullptr);
  EXPECT_EQ(asFeedforwardController(base), nullptr);
}

TEST(PolicyControllersTest, AControllerThatOnlyReportsTheTypeIsNeither) {
  ImpostorController controller;
  ASSERT_EQ(controller.getType(), ControllerType::LINEAR);
  EXPECT_EQ(asLinearController(controller), nullptr);
  EXPECT_EQ(asFeedforwardController(controller), nullptr);
}

TEST(PolicyControllersTest, AStateBasedLinearControllerIsRefusedEverywhereAlthoughItReportsLinear) {
  std::mt19937 generator(/*sd=*/7);
  const PolicyShape shape{.nodes = 10, .stateDim = 3, .inputDim = 2, .events = 1, .controllerType = ControllerType::LINEAR};
  // Owns the LinearController the wrapper points to; outlives every copy of the wrapper.
  PrimalSolution source = test_data::randomPrimalSolution(generator, shape);
  StateBasedLinearController wrapper;
  wrapper.setController(source.controllerPtr_.get());
  ASSERT_EQ(wrapper.getType(), ControllerType::LINEAR) << "the reason PolicyControllers.h does not trust getType()";
  EXPECT_EQ(asLinearController(wrapper), nullptr);
  EXPECT_EQ(asFeedforwardController(wrapper), nullptr);

  PrimalSolution policy = source;
  policy.controllerPtr_.reset(wrapper.clone());

  humanoid_mpc_msgs::MpcPolicy message;
  const absl::Status encoded = policyToProto(CommandData(), policy, PerformanceIndex(), &message);
  EXPECT_EQ(encoded.code(), absl::StatusCode::kInvalidArgument) << encoded;

  RealtimePolicyEvaluator evaluator(ModelDimensions{.stateDim = shape.stateDim, .inputDim = shape.inputDim, .numModes = 2});
  const vector_t state = vector_t::Zero(static_cast<Eigen::Index>(shape.stateDim));
  vector_t mpcState;
  vector_t mpcInput;
  size_t mode = 0;
  EXPECT_EQ(evaluator.evaluate(policy, policy.timeTrajectory_.front(), state, mpcState, mpcInput, mode),
            RealtimePolicyEvaluator::Outcome::kUnsupportedController);

  // The solution is cut and the controller it does not recognize is left whole: the wrapped one keeps every node.
  const LinearController* absl_nullable wrapped = asLinearController(*source.controllerPtr_);
  ASSERT_NE(wrapped, nullptr);
  const size_t wrappedNodes = wrapped->timeStamp_.size();
  trimToSolutionWindow(policy.timeTrajectory_[shape.nodes / 2], &policy);
  EXPECT_LT(policy.timeTrajectory_.size(), shape.nodes);
  EXPECT_EQ(wrapped->timeStamp_.size(), wrappedNodes);
}

}  // namespace
}  // namespace ocs2::humanoid::ipc
