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

// RealtimePolicyEvaluator computes what OCS2's MRT_BASE::evaluatePolicy()
// computes, to the bit, for the controllers the MPC link carries, across the
// plan, at its events and past both ends: a feedforward and a linear policy on
// a flat state, and a feedforward and a manifold-linear policy on a state
// manifold, with the MRT's setStateManifold(); and leaves its outputs alone for
// a policy it cannot evaluate. Its allocations are pinned in
// test_remote_mpc_link_allocations, which replaces malloc.

#include <cmath>
#include <cstddef>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/control/ControllerType.h"
#include "ocs2_core/control/FeedforwardController.h"
#include "ocs2_core/control/LinearController.h"
#include "ocs2_core/control/ManifoldLinearController.h"
#include "ocs2_core/manifold/StateManifold.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_mpc/CommandData.h"
#include "ocs2_mpc/MRT_BASE.h"
#include "ocs2_mpc/SystemObservation.h"
#include "ocs2_oc/oc_data/PerformanceIndex.h"
#include "ocs2_oc/oc_data/PrimalSolution.h"

#include "humanoid_mpc_ipc/MpcMessageConversions.h"
#include "humanoid_mpc_ipc/RealtimePolicyEvaluator.h"
#include "humanoid_nmpc/humanoid_mpc_ipc/test/PolicyTestData.h"

namespace ocs2::humanoid::ipc {
namespace {

using Outcome = RealtimePolicyEvaluator::Outcome;
using test_data::PolicyShape;

/** An MRT whose buffer the test fills directly, to evaluate a policy with
 * OCS2's own evaluatePolicy(). */
class BufferedMrt final : public MRT_BASE {
 public:
  void resetMpcNode(
      const TargetTrajectories& /*initTargetTrajectories*/) override {}
  void setCurrentObservation(
      const SystemObservation& /*observation*/) override {}
  void fill(const PrimalSolution& solution) {
    moveToBuffer(std::make_unique<CommandData>(),
                 std::make_unique<PrimalSolution>(solution),
                 std::make_unique<PerformanceIndex>());
  }
};

/**
 * Before the plan, at every node, halfway between nodes (the two nodes of an
 * event share one time), just around every event, and past it.
 */
std::vector<scalar_t> evaluationTimes(const PrimalSolution& policy) {
  const scalar_array_t& times = policy.timeTrajectory_;
  std::vector<scalar_t> result = {times.front() - 0.1, times.front()};
  for (size_t node = 1; node < times.size(); ++node) {
    result.push_back(0.5 * (times[node - 1] + times[node]));
    result.push_back(times[node]);
  }
  for (const size_t postEvent : policy.postEventIndices_) {
    result.push_back(times[postEvent] - 1.0e-9);
    result.push_back(times[postEvent] + 1.0e-9);
  }
  result.push_back(times.back() + 0.1);
  return result;
}

void expectSameEvaluationAsOcs2(const PolicyShape& shape, unsigned int seed) {
  std::mt19937 generator(seed);
  const PrimalSolution policy =
      test_data::randomPrimalSolution(generator, shape);
  BufferedMrt mrt;
  mrt.fill(policy);
  ASSERT_TRUE(mrt.updatePolicy());
  RealtimePolicyEvaluator evaluator(ModelDimensions{
      .stateDim = shape.stateDim, .inputDim = shape.inputDim, .numModes = 1});

  const vector_t state = test_data::randomVector(generator, shape.stateDim);
  vector_t expectedState;
  vector_t expectedInput;
  size_t expectedMode = 0;
  vector_t mpcState = vector_t::Zero(static_cast<Eigen::Index>(shape.stateDim));
  vector_t mpcInput = vector_t::Zero(static_cast<Eigen::Index>(shape.inputDim));
  size_t mode = 0;
  for (const scalar_t time : evaluationTimes(policy)) {
    mrt.evaluatePolicy(time, state, expectedState, expectedInput, expectedMode);
    ASSERT_EQ(evaluator.evaluate(mrt.getPolicy(), time, state, mpcState,
                                 mpcInput, mode),
              Outcome::kEvaluated)
        << "t = " << time;
    // The same doubles, not merely close ones: the controller cannot tell which
    // evaluated its policy.
    EXPECT_TRUE(mpcState == expectedState) << "t = " << time << "\n"
                                           << mpcState.transpose() << "\n"
                                           << expectedState.transpose();
    EXPECT_TRUE(mpcInput == expectedInput) << "t = " << time << "\n"
                                           << mpcInput.transpose() << "\n"
                                           << expectedInput.transpose();
    EXPECT_EQ(mode, expectedMode) << "t = " << time;
  }
}

TEST(RealtimePolicyEvaluatorTest,
     AFeedforwardPolicyEvaluatesAsOcs2EvaluatesIt) {
  expectSameEvaluationAsOcs2({.nodes = 31,
                              .stateDim = 7,
                              .inputDim = 5,
                              .events = 3,
                              .controllerType = ControllerType::FEEDFORWARD},
                             /*seed=*/3);
}

TEST(RealtimePolicyEvaluatorTest, ALinearPolicyEvaluatesAsOcs2EvaluatesIt) {
  expectSameEvaluationAsOcs2({.nodes = 31,
                              .stateDim = 7,
                              .inputDim = 5,
                              .events = 3,
                              .controllerType = ControllerType::LINEAR},
                             /*seed=*/5);
  // The README's bandwidth example, nx = nu = 30 over 60 nodes.
  expectSameEvaluationAsOcs2({.nodes = 60,
                              .stateDim = 30,
                              .inputDim = 30,
                              .events = 4,
                              .controllerType = ControllerType::LINEAR},
                             /*seed=*/7);
}

TEST(RealtimePolicyEvaluatorTest, APolicyOfOneNodeIsHeld) {
  expectSameEvaluationAsOcs2({.nodes = 1,
                              .stateDim = 3,
                              .inputDim = 2,
                              .events = 0,
                              .controllerType = ControllerType::LINEAR},
                             /*seed=*/9);
}

/** The shape of a policy the SQP solves on the toy manifold [E(3), Q, E(5)]. */
PolicyShape manifoldShape(size_t nodes,
                          size_t events,
                          ControllerType controllerType) {
  const std::shared_ptr<const StateManifold> manifold =
      test_data::toyStateManifold();
  return {.nodes = nodes,
          .stateDim = manifold->getAmbientDim(),
          .inputDim = 4,
          .events = events,
          .controllerType = controllerType,
          .stateManifold = manifold};
}

/**
 * The evaluation of OCS2's MRT with the state manifold set, against the
 * evaluator on that manifold: the same doubles at every time of
 * evaluationTimes(), across the events, at states of the manifold and at their
 * other quaternion sign, which the policy cannot tell apart (the difference
 * takes the shortest path).
 */
void expectSameEvaluationAsOcs2OnTheManifold(const PolicyShape& shape,
                                             unsigned int seed) {
  std::mt19937 generator(seed);
  const PrimalSolution policy =
      test_data::randomPrimalSolution(generator, shape);
  BufferedMrt mrt;
  mrt.setStateManifold(shape.stateManifold);
  mrt.fill(policy);
  ASSERT_TRUE(mrt.updatePolicy());
  RealtimePolicyEvaluator evaluator(ModelDimensions{.stateDim = shape.stateDim,
                                                    .inputDim = shape.inputDim,
                                                    .numModes = 1},
                                    shape.stateManifold.get());

  const vector_t state =
      test_data::randomManifoldState(generator, *shape.stateManifold);
  vector_t otherSign = state;
  otherSign.segment<4>(3) *= -1.0;
  vector_t expectedState;
  vector_t expectedInput;
  size_t expectedMode = 0;
  vector_t mpcState = vector_t::Zero(static_cast<Eigen::Index>(shape.stateDim));
  vector_t mpcInput = vector_t::Zero(static_cast<Eigen::Index>(shape.inputDim));
  size_t mode = 0;
  for (const scalar_t time : evaluationTimes(policy)) {
    mrt.evaluatePolicy(time, state, expectedState, expectedInput, expectedMode);
    ASSERT_EQ(evaluator.evaluate(mrt.getPolicy(), time, state, mpcState,
                                 mpcInput, mode),
              Outcome::kEvaluated)
        << "t = " << time;
    EXPECT_TRUE(mpcState == expectedState) << "t = " << time << "\n"
                                           << mpcState.transpose() << "\n"
                                           << expectedState.transpose();
    EXPECT_TRUE(mpcInput == expectedInput) << "t = " << time << "\n"
                                           << mpcInput.transpose() << "\n"
                                           << expectedInput.transpose();
    EXPECT_EQ(mode, expectedMode) << "t = " << time;

    const vector_t inputAtState = mpcInput;
    ASSERT_EQ(evaluator.evaluate(mrt.getPolicy(), time, otherSign, mpcState,
                                 mpcInput, mode),
              Outcome::kEvaluated)
        << "t = " << time;
    EXPECT_TRUE(mpcInput == inputAtState)
        << "the other sign of the quaternion at t = " << time;
  }
}

TEST(RealtimePolicyEvaluatorTest,
     AManifoldLinearPolicyEvaluatesAsOcs2EvaluatesItAcrossItsEvents) {
  expectSameEvaluationAsOcs2OnTheManifold(
      manifoldShape(/*nodes=*/31, /*events=*/3,
                    ControllerType::MANIFOLD_LINEAR),
      /*seed=*/21);
  // An event right after the first node and right before the last.
  expectSameEvaluationAsOcs2OnTheManifold(
      manifoldShape(/*nodes=*/4, /*events=*/1, ControllerType::MANIFOLD_LINEAR),
      /*seed=*/22);
  expectSameEvaluationAsOcs2OnTheManifold(
      manifoldShape(/*nodes=*/2, /*events=*/0, ControllerType::MANIFOLD_LINEAR),
      /*seed=*/23);
}

TEST(RealtimePolicyEvaluatorTest,
     AFeedforwardPolicyOnAManifoldEvaluatesAsOcs2EvaluatesIt) {
  // The planned state is interpolated along the manifold (a slerp of the
  // quaternion), the input linearly.
  expectSameEvaluationAsOcs2OnTheManifold(
      manifoldShape(/*nodes=*/31, /*events=*/3, ControllerType::FEEDFORWARD),
      /*seed=*/24);
  expectSameEvaluationAsOcs2OnTheManifold(
      manifoldShape(/*nodes=*/1, /*events=*/0, ControllerType::FEEDFORWARD),
      /*seed=*/25);
}

TEST(RealtimePolicyEvaluatorTest,
     ThePlannedStateIsInterpolatedAlongTheManifold) {
  // Positive control for the parity above: between two nodes the planned
  // quaternion is of unit length, which the linear interpolation of a flat
  // evaluator would not give it.
  const PolicyShape shape =
      manifoldShape(/*nodes=*/5, /*events=*/0, ControllerType::MANIFOLD_LINEAR);
  std::mt19937 generator(/*sd=*/26);
  const PrimalSolution policy =
      test_data::randomPrimalSolution(generator, shape);
  RealtimePolicyEvaluator evaluator(ModelDimensions{.stateDim = shape.stateDim,
                                                    .inputDim = shape.inputDim,
                                                    .numModes = 1},
                                    shape.stateManifold.get());
  vector_t mpcState;
  vector_t mpcInput;
  size_t mode = 0;
  const scalar_t midpoint =
      0.5 * (policy.timeTrajectory_[1] + policy.timeTrajectory_[2]);
  ASSERT_EQ(evaluator.evaluate(policy, midpoint, policy.stateTrajectory_[1],
                               mpcState, mpcInput, mode),
            Outcome::kEvaluated);
  EXPECT_NEAR(mpcState.segment<4>(3).norm(), 1.0, 1.0e-12);
  const vector_t linear =
      0.5 * (policy.stateTrajectory_[1] + policy.stateTrajectory_[2]);
  EXPECT_GT(std::abs(linear.segment<4>(3).norm() - 1.0), 1.0e-6);
}

class RealtimePolicyEvaluatorRefusalTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::mt19937 generator(/*sd=*/13);
    policy_ = test_data::randomPrimalSolution(
        generator, {.nodes = 10,
                    .stateDim = 3,
                    .inputDim = 2,
                    .events = 1,
                    .controllerType = ControllerType::LINEAR});
    state_ = vector_t::Ones(3);
  }

  /** Evaluates `policy` into outputs holding sentinels, and checks that they
   * still hold them. */
  Outcome evaluateUntouched(const PrimalSolution& policy,
                            const vector_t& state) {
    vector_t mpcState = vector_t::Constant(3, 42.0);
    vector_t mpcInput = vector_t::Constant(2, 42.0);
    size_t mode = 42;
    const Outcome outcome = evaluator_.evaluate(
        policy,
        policy.timeTrajectory_.empty() ? 0.0 : policy.timeTrajectory_[2], state,
        mpcState, mpcInput, mode);
    EXPECT_TRUE(mpcState == vector_t::Constant(3, 42.0));
    EXPECT_TRUE(mpcInput == vector_t::Constant(2, 42.0));
    EXPECT_EQ(mode, 42u);
    return outcome;
  }

  RealtimePolicyEvaluator evaluator_{
      ModelDimensions{.stateDim = 3, .inputDim = 2, .numModes = 1}};
  PrimalSolution policy_;
  vector_t state_;
};

TEST_F(RealtimePolicyEvaluatorRefusalTest,
       APolicyWithoutAControllerOfAKnownTypeIsNotEvaluated) {
  PrimalSolution noController = policy_;
  noController.controllerPtr_.reset();
  EXPECT_EQ(evaluateUntouched(noController, state_),
            Outcome::kUnsupportedController);
}

TEST_F(RealtimePolicyEvaluatorRefusalTest,
       APolicyOrStateOfOtherDimensionsIsNotEvaluated) {
  EXPECT_EQ(evaluateUntouched(policy_, vector_t::Ones(4)),
            Outcome::kMismatchedPolicy)
      << "the measured state";

  PrimalSolution wideState = policy_;
  wideState.stateTrajectory_[2] = vector_t::Ones(4);
  EXPECT_EQ(evaluateUntouched(wideState, state_), Outcome::kMismatchedPolicy);

  PrimalSolution wideGain = policy_;
  wideGain.controllerPtr_ = std::make_unique<LinearController>(
      policy_.timeTrajectory_,
      vector_array_t(policy_.timeTrajectory_.size(), vector_t::Ones(2)),
      matrix_array_t(policy_.timeTrajectory_.size(), matrix_t::Ones(2, 4)));
  EXPECT_EQ(evaluateUntouched(wideGain, state_), Outcome::kMismatchedPolicy);

  PrimalSolution wideInput = policy_;
  wideInput.controllerPtr_ = std::make_unique<FeedforwardController>(
      policy_.timeTrajectory_,
      vector_array_t(policy_.timeTrajectory_.size(), vector_t::Ones(3)));
  EXPECT_EQ(evaluateUntouched(wideInput, state_), Outcome::kMismatchedPolicy);
}

/** Evaluates `policy` into outputs holding sentinels, checks that they still
 * hold them, and returns the outcome. */
Outcome evaluateWithSentinels(RealtimePolicyEvaluator& evaluator,
                              const PrimalSolution& policy,
                              scalar_t time,
                              const vector_t& state,
                              const ModelDimensions& dimensions) {
  const vector_t stateSentinel =
      vector_t::Constant(static_cast<Eigen::Index>(dimensions.stateDim), 42.0);
  const vector_t inputSentinel =
      vector_t::Constant(static_cast<Eigen::Index>(dimensions.inputDim), 42.0);
  vector_t mpcState = stateSentinel;
  vector_t mpcInput = inputSentinel;
  size_t mode = 42;
  const Outcome outcome =
      evaluator.evaluate(policy, time, state, mpcState, mpcInput, mode);
  EXPECT_TRUE(mpcState == stateSentinel);
  EXPECT_TRUE(mpcInput == inputSentinel);
  EXPECT_EQ(mode, 42u);
  return outcome;
}

TEST(RealtimePolicyEvaluatorManifoldRefusalTest,
     EachStateRefusesTheFeedbackPolicyOfTheOtherAndLeavesTheOutputsAlone) {
  std::mt19937 generator(/*sd=*/27);
  const PolicyShape manifoldPolicyShape = manifoldShape(
      /*nodes=*/10, /*events=*/1, ControllerType::MANIFOLD_LINEAR);
  const PrimalSolution manifoldPolicy =
      test_data::randomPrimalSolution(generator, manifoldPolicyShape);
  PolicyShape linearShape = manifoldPolicyShape;
  linearShape.controllerType = ControllerType::LINEAR;
  linearShape.stateManifold = nullptr;
  const PrimalSolution linearPolicy =
      test_data::randomPrimalSolution(generator, linearShape);
  const ModelDimensions dimensions{.stateDim = manifoldPolicyShape.stateDim,
                                   .inputDim = manifoldPolicyShape.inputDim,
                                   .numModes = 1};
  RealtimePolicyEvaluator onTheManifold(
      dimensions, manifoldPolicyShape.stateManifold.get());
  RealtimePolicyEvaluator flat(dimensions);
  const vector_t state = test_data::randomManifoldState(
      generator, *manifoldPolicyShape.stateManifold);
  const scalar_t time = manifoldPolicy.timeTrajectory_[3];

  // A linear policy's gains act on the stored state; on a manifold they would
  // act on 12 entries for 11 directions.
  EXPECT_EQ(evaluateWithSentinels(onTheManifold, linearPolicy, time, state,
                                  dimensions),
            Outcome::kUnsupportedController);
  // A manifold-linear policy has no law on a flat state.
  EXPECT_EQ(
      evaluateWithSentinels(flat, manifoldPolicy, time, state, dimensions),
      Outcome::kUnsupportedController);
  // Positive controls: each evaluates its own.
  vector_t mpcState;
  vector_t mpcInput;
  size_t mode = 0;
  EXPECT_EQ(onTheManifold.evaluate(manifoldPolicy, time, state, mpcState,
                                   mpcInput, mode),
            Outcome::kEvaluated);
  EXPECT_EQ(flat.evaluate(linearPolicy, time, state, mpcState, mpcInput, mode),
            Outcome::kEvaluated);

  // A manifold-linear controller whose gains or anchors are not of the
  // manifold's dimensions.
  const ManifoldLinearController& controller =
      dynamic_cast<const ManifoldLinearController&>(
          *manifoldPolicy.controllerPtr_);
  const size_t nodes = controller.getTimeStamps().size();
  PrimalSolution ambientGains = manifoldPolicy;
  ambientGains.controllerPtr_ = std::make_unique<ManifoldLinearController>(
      controller.getTimeStamps(), controller.getAnchorStates(),
      controller.getNominalInputs(),
      matrix_array_t(
          nodes,
          matrix_t::Ones(static_cast<Eigen::Index>(dimensions.inputDim),
                         static_cast<Eigen::Index>(dimensions.stateDim))),
      manifoldPolicyShape.stateManifold);
  EXPECT_EQ(evaluateWithSentinels(onTheManifold, ambientGains, time, state,
                                  dimensions),
            Outcome::kMismatchedPolicy);
  PrimalSolution shortAnchors = manifoldPolicy;
  shortAnchors.controllerPtr_ = std::make_unique<ManifoldLinearController>(
      controller.getTimeStamps(), vector_array_t(nodes, vector_t::Ones(11)),
      controller.getNominalInputs(), controller.getFeedbackGains(),
      manifoldPolicyShape.stateManifold);
  EXPECT_EQ(evaluateWithSentinels(onTheManifold, shortAnchors, time, state,
                                  dimensions),
            Outcome::kMismatchedPolicy);
}

TEST(RealtimePolicyEvaluatorManifoldRefusalTest,
     AManifoldOfAnotherStateSizeIsRefusedAtConstruction) {
  const std::shared_ptr<const StateManifold> manifold =
      test_data::toyStateManifold();
  EXPECT_DEATH(
      {
        const RealtimePolicyEvaluator evaluator(
            ModelDimensions{.stateDim = 13, .inputDim = 2, .numModes = 1},
            manifold.get());
      },
      "ambient dimension is not the model's state dimension");
}

TEST_F(RealtimePolicyEvaluatorRefusalTest,
       AnEmptyOrInconsistentPolicyIsNotEvaluated) {
  EXPECT_EQ(evaluateUntouched(PrimalSolution(), state_),
            Outcome::kMismatchedPolicy);

  PrimalSolution shortStates = policy_;
  shortStates.stateTrajectory_.pop_back();
  EXPECT_EQ(evaluateUntouched(shortStates, state_), Outcome::kMismatchedPolicy);

  PrimalSolution noModes = policy_;
  noModes.modeSchedule_.modeSequence.clear();
  EXPECT_EQ(evaluateUntouched(noModes, state_), Outcome::kMismatchedPolicy);

  PrimalSolution shortGains = policy_;
  shortGains.controllerPtr_ = std::make_unique<LinearController>(
      policy_.timeTrajectory_,
      vector_array_t(policy_.timeTrajectory_.size(), vector_t::Ones(2)),
      matrix_array_t(1, matrix_t::Ones(2, 3)));
  EXPECT_EQ(evaluateUntouched(shortGains, state_), Outcome::kMismatchedPolicy);
}

}  // namespace
}  // namespace ocs2::humanoid::ipc
