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

// asFeedforwardController(), asLinearController() and
// asManifoldLinearController() find the exact type of a policy's controller:
// each finds its own type and no other, from a const or a mutable controller,
// and a controller of another class whose getType() reports one of the three is
// none of them. So the conversions, the solution time window and the realtime
// evaluator all refuse or skip it alike, as they do every controller the link
// does not carry.

#include <cstddef>
#include <memory>
#include <random>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/control/ControllerBase.h"
#include "ocs2_core/control/ControllerType.h"
#include "ocs2_core/control/FeedforwardController.h"
#include "ocs2_core/control/LinearController.h"
#include "ocs2_core/control/ManifoldLinearController.h"
#include "ocs2_core/manifold/StateManifold.h"
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

/** A controller of a type the link does not carry, which claims to be one it
 * does carry. */
class ImpostorController final : public ControllerBase {
 public:
  explicit ImpostorController(ControllerType claimedType)
      : claimedType_(claimedType) {}
  vector_t computeInput(scalar_t /*t*/, const vector_t& /*x*/) override {
    return vector_t();
  }
  int size() const override { return 1; }
  ControllerType getType() const override { return claimedType_; }
  void clear() override {}
  bool empty() const override { return false; }
  ImpostorController* absl_nonnull clone() const override {
    return new ImpostorController(*this);
  }

 private:
  ControllerType claimedType_;
};

TEST(PolicyControllersTest, AFeedforwardControllerIsOneAndNoLinearController) {
  FeedforwardController controller(
      /*controllerTime=*/{0.0, 1.0},
      /*controllerFeedforward=*/{vector_t::Ones(2), vector_t::Zero(2)});
  const ControllerBase& constBase = controller;
  ControllerBase& base = controller;
  EXPECT_EQ(asFeedforwardController(constBase), &controller);
  EXPECT_EQ(asFeedforwardController(base), &controller);
  EXPECT_EQ(asLinearController(constBase), nullptr);
  EXPECT_EQ(asLinearController(base), nullptr);
}

TEST(PolicyControllersTest, ALinearControllerIsOneAndNoFeedforwardController) {
  LinearController controller(
      /*controllerTime=*/{0.0, 1.0},
      /*controllerBias=*/{vector_t::Ones(2), vector_t::Zero(2)},
      /*controllerGain=*/{matrix_t::Identity(2, 3), matrix_t::Zero(2, 3)});
  const ControllerBase& constBase = controller;
  ControllerBase& base = controller;
  EXPECT_EQ(asLinearController(constBase), &controller);
  EXPECT_EQ(asLinearController(base), &controller);
  EXPECT_EQ(asFeedforwardController(constBase), nullptr);
  EXPECT_EQ(asFeedforwardController(base), nullptr);
  EXPECT_EQ(asManifoldLinearController(constBase), nullptr);
  EXPECT_EQ(asManifoldLinearController(base), nullptr);
}

TEST(PolicyControllersTest,
     AManifoldLinearControllerIsOneAndNeitherOfTheOthers) {
  const std::shared_ptr<const StateManifold> manifold =
      test_data::toyStateManifold();
  const vector_t anchor = vector_t::Unit(
      static_cast<Eigen::Index>(manifold->getAmbientDim()), /*i=*/6);
  const matrix_t gain =
      matrix_t::Ones(2, static_cast<Eigen::Index>(manifold->getTangentDim()));
  ManifoldLinearController controller(
      /*timeStamp=*/{0.0, 1.0}, /*anchorStates=*/{anchor, anchor},
      /*nominalInputs=*/{vector_t::Ones(2), vector_t::Zero(2)},
      /*feedbackGains=*/{gain, gain}, manifold);
  const ControllerBase& constBase = controller;
  ControllerBase& base = controller;
  EXPECT_EQ(asManifoldLinearController(constBase), &controller);
  EXPECT_EQ(asManifoldLinearController(base), &controller);
  EXPECT_EQ(asLinearController(constBase), nullptr);
  EXPECT_EQ(asLinearController(base), nullptr);
  EXPECT_EQ(asFeedforwardController(constBase), nullptr);
  EXPECT_EQ(asFeedforwardController(base), nullptr);

  FeedforwardController feedforward(
      /*controllerTime=*/{0.0}, /*controllerFeedforward=*/{vector_t::Ones(2)});
  EXPECT_EQ(asManifoldLinearController(feedforward), nullptr);
}

TEST(PolicyControllersTest, AControllerThatOnlyReportsTheTypeIsNone) {
  for (const ControllerType claimedType :
       {ControllerType::FEEDFORWARD, ControllerType::LINEAR,
        ControllerType::MANIFOLD_LINEAR}) {
    ImpostorController controller(claimedType);
    ASSERT_EQ(controller.getType(), claimedType);
    EXPECT_EQ(asLinearController(controller), nullptr);
    EXPECT_EQ(asFeedforwardController(controller), nullptr);
    EXPECT_EQ(asManifoldLinearController(controller), nullptr);
  }
}

TEST(PolicyControllersTest, AControllerThatReportsLinearIsRefusedEverywhere) {
  std::mt19937 generator(/*sd=*/7);
  const PolicyShape shape{.nodes = 10,
                          .stateDim = 3,
                          .inputDim = 2,
                          .events = 1,
                          .controllerType = ControllerType::LINEAR};
  PrimalSolution policy = test_data::randomPrimalSolution(generator, shape);
  policy.controllerPtr_ =
      std::make_unique<ImpostorController>(ControllerType::LINEAR);
  ASSERT_EQ(policy.controllerPtr_->getType(), ControllerType::LINEAR)
      << "the reason PolicyControllers.h does not trust getType()";

  humanoid_mpc_msgs::MpcPolicy message;
  const absl::Status encoded =
      policyToProto(CommandData(), policy, PerformanceIndex(), &message);
  EXPECT_EQ(encoded.code(), absl::StatusCode::kInvalidArgument) << encoded;

  RealtimePolicyEvaluator evaluator(ModelDimensions{
      .stateDim = shape.stateDim, .inputDim = shape.inputDim, .numModes = 2});
  const vector_t state =
      vector_t::Zero(static_cast<Eigen::Index>(shape.stateDim));
  vector_t mpcState;
  vector_t mpcInput;
  size_t mode = 0;
  EXPECT_EQ(evaluator.evaluate(policy, policy.timeTrajectory_.front(), state,
                               mpcState, mpcInput, mode),
            RealtimePolicyEvaluator::Outcome::kUnsupportedController);

  // The solution is cut and the controller it does not recognize is left in
  // place.
  const ControllerBase* absl_nonnull impostor = policy.controllerPtr_.get();
  trimToSolutionWindow(policy.timeTrajectory_[shape.nodes / 2], &policy);
  EXPECT_LT(policy.timeTrajectory_.size(), shape.nodes);
  EXPECT_EQ(policy.controllerPtr_.get(), impostor);
}

}  // namespace
}  // namespace ocs2::humanoid::ipc
