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

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "gtest/gtest.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/control/ControllerBase.h"
#include "ocs2_core/control/ControllerType.h"
#include "ocs2_core/control/FeedforwardController.h"
#include "ocs2_core/control/LinearController.h"
#include "ocs2_core/control/ManifoldLinearController.h"
#include "ocs2_core/manifold/EuclideanStateManifold.h"
#include "ocs2_core/manifold/ProductStateManifold.h"
#include "ocs2_core/manifold/StateManifold.h"
#include "ocs2_core/manifold/StateManifoldSegment.h"

#include "humanoid_mpc_ipc/MpcMessageConversions.h"
#include "humanoid_mpc_ipc/PolicyControllers.h"
#include "humanoid_mpc_msgs/controller_type.pb.h"
#include "humanoid_mpc_msgs/mode_schedule.pb.h"
#include "humanoid_mpc_msgs/mpc_policy.pb.h"
#include "humanoid_mpc_msgs/performance_index.pb.h"
#include "humanoid_mpc_msgs/system_observation.pb.h"
#include "humanoid_mpc_msgs/target_contact_patch.pb.h"
#include "humanoid_mpc_msgs/target_trajectories.pb.h"
#include "humanoid_mpc_msgs/vector.pb.h"
#include "humanoid_nmpc/humanoid_mpc_ipc/test/PolicyTestData.h"

namespace ocs2::humanoid::ipc {
namespace {

using test_data::PolicyShape;
using test_data::randomSize;

constexpr scalar_t kNaN = std::numeric_limits<scalar_t>::quiet_NaN();
constexpr scalar_t kInfinity = std::numeric_limits<scalar_t>::infinity();
constexpr std::array<ControllerType, 2> kControllerTypes = {ControllerType::FEEDFORWARD, ControllerType::LINEAR};

// =====================================================================================================================
// Bitwise comparison: a lossless conversion reproduces every double exactly, signed zeros and NaN payloads included.
// =====================================================================================================================

::testing::AssertionResult bitwiseEqual(scalar_t expected, scalar_t actual) {
  if (std::bit_cast<uint64_t>(expected) == std::bit_cast<uint64_t>(actual)) {
    return ::testing::AssertionSuccess();
  }
  return ::testing::AssertionFailure() << expected << " != " << actual;
}

::testing::AssertionResult bitwiseEqual(const matrix_t& expected, const matrix_t& actual) {
  if (expected.rows() != actual.rows() || expected.cols() != actual.cols()) {
    return ::testing::AssertionFailure() << expected.rows() << "x" << expected.cols() << " != " << actual.rows() << "x" << actual.cols();
  }
  for (Eigen::Index j = 0; j < expected.cols(); ++j) {
    for (Eigen::Index i = 0; i < expected.rows(); ++i) {
      if (!bitwiseEqual(expected(i, j), actual(i, j))) {
        return ::testing::AssertionFailure() << "(" << i << ", " << j << "): " << expected(i, j) << " != " << actual(i, j);
      }
    }
  }
  return ::testing::AssertionSuccess();
}

::testing::AssertionResult bitwiseEqual(const vector_t& expected, const vector_t& actual) {
  return bitwiseEqual(matrix_t(expected), matrix_t(actual));
}

template <typename T, typename Allocator>
::testing::AssertionResult bitwiseEqual(const std::vector<T, Allocator>& expected, const std::vector<T, Allocator>& actual) {
  if (expected.size() != actual.size()) {
    return ::testing::AssertionFailure() << expected.size() << " elements != " << actual.size() << " elements";
  }
  for (size_t k = 0; k < expected.size(); ++k) {
    const ::testing::AssertionResult element = bitwiseEqual(expected[k], actual[k]);
    if (!element) {
      return ::testing::AssertionFailure() << "element " << k << ": " << element.message();
    }
  }
  return ::testing::AssertionSuccess();
}

void expectSameObservation(const SystemObservation& expected, const SystemObservation& actual) {
  EXPECT_TRUE(bitwiseEqual(expected.time, actual.time));
  EXPECT_TRUE(bitwiseEqual(expected.state, actual.state));
  EXPECT_TRUE(bitwiseEqual(expected.input, actual.input));
  EXPECT_EQ(expected.mode, actual.mode);
}

void expectSameModeSchedule(const ModeSchedule& expected, const ModeSchedule& actual) {
  EXPECT_TRUE(bitwiseEqual(expected.eventTimes, actual.eventTimes));
  EXPECT_EQ(expected.modeSequence, actual.modeSequence);
}

void expectSameTargetTrajectories(const TargetTrajectories& expected, const TargetTrajectories& actual) {
  EXPECT_TRUE(bitwiseEqual(expected.timeTrajectory, actual.timeTrajectory));
  EXPECT_TRUE(bitwiseEqual(expected.stateTrajectory, actual.stateTrajectory));
  EXPECT_TRUE(bitwiseEqual(expected.inputTrajectory, actual.inputTrajectory));
}

void expectSamePerformanceIndex(const PerformanceIndex& expected, const PerformanceIndex& actual) {
  EXPECT_TRUE(bitwiseEqual(expected.merit, actual.merit));
  EXPECT_TRUE(bitwiseEqual(expected.cost, actual.cost));
  EXPECT_TRUE(bitwiseEqual(expected.dualFeasibilitiesSSE, actual.dualFeasibilitiesSSE));
  EXPECT_TRUE(bitwiseEqual(expected.dynamicsViolationSSE, actual.dynamicsViolationSSE));
  EXPECT_TRUE(bitwiseEqual(expected.equalityConstraintsSSE, actual.equalityConstraintsSSE));
  EXPECT_TRUE(bitwiseEqual(expected.inequalityConstraintsSSE, actual.inequalityConstraintsSSE));
  EXPECT_TRUE(bitwiseEqual(expected.equalityLagrangian, actual.equalityLagrangian));
  EXPECT_TRUE(bitwiseEqual(expected.inequalityLagrangian, actual.inequalityLagrangian));
}

void expectSameController(const ControllerBase* absl_nullable expected, const ControllerBase* absl_nullable actual) {
  ASSERT_NE(expected, nullptr);
  ASSERT_NE(actual, nullptr);
  ASSERT_EQ(expected->getType(), actual->getType());
  if (expected->getType() == ControllerType::MANIFOLD_LINEAR) {
    const ManifoldLinearController& expectedManifold = dynamic_cast<const ManifoldLinearController&>(*expected);
    const ManifoldLinearController& actualManifold = dynamic_cast<const ManifoldLinearController&>(*actual);
    EXPECT_TRUE(bitwiseEqual(expectedManifold.getTimeStamps(), actualManifold.getTimeStamps()));
    EXPECT_TRUE(bitwiseEqual(expectedManifold.getAnchorStates(), actualManifold.getAnchorStates()));
    EXPECT_TRUE(bitwiseEqual(expectedManifold.getNominalInputs(), actualManifold.getNominalInputs()));
    EXPECT_TRUE(bitwiseEqual(expectedManifold.getFeedbackGains(), actualManifold.getFeedbackGains()));
    EXPECT_EQ(expectedManifold.getStateManifold(), actualManifold.getStateManifold());
  } else if (expected->getType() == ControllerType::LINEAR) {
    const LinearController& expectedLinear = dynamic_cast<const LinearController&>(*expected);
    const LinearController& actualLinear = dynamic_cast<const LinearController&>(*actual);
    EXPECT_TRUE(bitwiseEqual(expectedLinear.timeStamp_, actualLinear.timeStamp_));
    EXPECT_TRUE(bitwiseEqual(expectedLinear.biasArray_, actualLinear.biasArray_));
    EXPECT_TRUE(bitwiseEqual(expectedLinear.gainArray_, actualLinear.gainArray_));
  } else {
    const FeedforwardController& expectedFeedforward = dynamic_cast<const FeedforwardController&>(*expected);
    const FeedforwardController& actualFeedforward = dynamic_cast<const FeedforwardController&>(*actual);
    EXPECT_TRUE(bitwiseEqual(expectedFeedforward.timeStamp_, actualFeedforward.timeStamp_));
    EXPECT_TRUE(bitwiseEqual(expectedFeedforward.uffArray_, actualFeedforward.uffArray_));
  }
}

void expectSamePrimalSolution(const PrimalSolution& expected, const PrimalSolution& actual) {
  EXPECT_TRUE(bitwiseEqual(expected.timeTrajectory_, actual.timeTrajectory_));
  EXPECT_TRUE(bitwiseEqual(expected.stateTrajectory_, actual.stateTrajectory_));
  EXPECT_TRUE(bitwiseEqual(expected.inputTrajectory_, actual.inputTrajectory_));
  EXPECT_EQ(expected.postEventIndices_, actual.postEventIndices_);
  expectSameModeSchedule(expected.modeSchedule_, actual.modeSchedule_);
  expectSameController(expected.controllerPtr_.get(), actual.controllerPtr_.get());
}

// =====================================================================================================================
// Policies
// =====================================================================================================================

/** The three parts of one MPC solution, as MPC_MRT_Interface::copyToBuffer() hands them to the MRT. */
struct Policy {
  CommandData commandData;
  PrimalSolution primalSolution;
  PerformanceIndex performanceIndex;
};

void expectSamePolicy(const Policy& expected, const Policy& actual) {
  expectSameObservation(expected.commandData.mpcInitObservation_, actual.commandData.mpcInitObservation_);
  expectSameTargetTrajectories(expected.commandData.mpcTargetTrajectories_, actual.commandData.mpcTargetTrajectories_);
  expectSamePrimalSolution(expected.primalSolution, actual.primalSolution);
  expectSamePerformanceIndex(expected.performanceIndex, actual.performanceIndex);
}

Policy randomPolicy(std::mt19937& generator, const PolicyShape& shape, size_t targetNodes) {
  Policy policy;
  policy.commandData = test_data::randomCommandData(generator, shape.stateDim, shape.inputDim, targetNodes);
  policy.primalSolution = test_data::randomPrimalSolution(generator, shape);
  policy.performanceIndex = test_data::randomPerformanceIndex(generator);
  return policy;
}

PolicyShape randomShape(std::mt19937& generator, ControllerType controllerType) {
  PolicyShape shape;
  shape.nodes = randomSize(generator, /*low=*/1, /*high=*/40);
  shape.stateDim = randomSize(generator, /*low=*/1, /*high=*/12);
  shape.inputDim = randomSize(generator, /*low=*/1, /*high=*/8);
  shape.events = randomSize(generator, /*low=*/0, /*high=*/(shape.nodes - 1) / 3);
  shape.controllerType = controllerType;
  return shape;
}

humanoid_mpc_msgs::MpcPolicy encode(const Policy& policy) {
  humanoid_mpc_msgs::MpcPolicy message;
  const absl::Status status = policyToProto(policy.commandData, policy.primalSolution, policy.performanceIndex, &message);
  EXPECT_TRUE(status.ok()) << status;
  return message;
}

/** Decodes as a receiver whose state is flat. */
absl::Status decode(const humanoid_mpc_msgs::MpcPolicy& message, Policy* absl_nonnull policy) {
  return policyFromProto(message, /*stateManifold=*/nullptr, &policy->commandData, &policy->primalSolution, &policy->performanceIndex);
}

/** Decodes as a receiver whose state lives on `stateManifold`. */
absl::Status decode(const humanoid_mpc_msgs::MpcPolicy& message,
                    std::shared_ptr<const StateManifold> stateManifold,
                    Policy* absl_nonnull policy) {
  return policyFromProto(message, std::move(stateManifold), &policy->commandData, &policy->primalSolution, &policy->performanceIndex);
}

// Encodes `policy`, sends it through the wire format and decodes it, on `stateManifold` when it is not null.
Policy roundTrip(const Policy& policy, std::shared_ptr<const StateManifold> stateManifold = nullptr) {
  humanoid_mpc_msgs::MpcPolicy parsed;
  EXPECT_TRUE(parsed.ParseFromString(encode(policy).SerializeAsString()));
  Policy decoded;
  const absl::Status status = decode(parsed, std::move(stateManifold), &decoded);
  EXPECT_TRUE(status.ok()) << status;
  return decoded;
}

// ControllerBase::flatten() at `time`, the sampling of a controller on other time stamps.
std::vector<std::vector<double>> flatten(const ControllerBase& controller, const scalar_array_t& time) {
  std::vector<std::vector<double>> samples(time.size());
  std::vector<std::vector<double>* absl_nonnull> sampleRefs;
  for (std::vector<double>& sample : samples) {
    sampleRefs.push_back(&sample);
  }
  controller.flatten(time, sampleRefs);
  return samples;
}

std::vector<double> asStdVector(const humanoid_mpc_msgs::Vector& vector) {
  return std::vector<double>(vector.data().begin(), vector.data().end());
}

// The times a policy is evaluated at: every node, both sides of every node, before and after the horizon, and random
// times in between.
scalar_array_t queryTimes(std::mt19937& generator, const scalar_array_t& timeTrajectory) {
  scalar_array_t times;
  for (const scalar_t time : timeTrajectory) {
    times.push_back(time);
    times.push_back(time - 1.0e-9);
    times.push_back(time + 1.0e-9);
  }
  times.push_back(timeTrajectory.front() - 1.0);
  times.push_back(timeTrajectory.back() + 1.0);
  std::uniform_real_distribution<scalar_t> inside(timeTrajectory.front(), timeTrajectory.back());
  for (int i = 0; i < 50; ++i) {
    times.push_back(inside(generator));
  }
  return times;
}

/** A controller of a type the conversions do not send. */
class UnsupportedController final : public ControllerBase {
 public:
  vector_t computeInput(scalar_t /*t*/, const vector_t& /*x*/) override { return vector_t(); }
  int size() const override { return 1; }
  ControllerType getType() const override { return ControllerType::UNKNOWN; }
  void clear() override {}
  bool empty() const override { return false; }
  UnsupportedController* absl_nonnull clone() const override { return new UnsupportedController(*this); }
};

// =====================================================================================================================
// SystemObservation, ModeSchedule, TargetTrajectories, PerformanceIndex
// =====================================================================================================================

TEST(SystemObservationConversionTest, RoundTripReproducesEveryBit) {
  std::mt19937 generator(/*sd=*/1);
  for (int trial = 0; trial < 50; ++trial) {
    SystemObservation observation = test_data::randomObservation(generator, randomSize(generator, /*low=*/0, /*high=*/40),
                                                                 randomSize(generator, /*low=*/0, /*high=*/40));
    if (observation.state.size() >= 4) {
      observation.state(0) = -0.0;
      observation.state(1) = std::numeric_limits<scalar_t>::denorm_min();
      observation.state(2) = std::numeric_limits<scalar_t>::max();
      observation.state(3) = std::numeric_limits<scalar_t>::lowest();
    }
    if (trial == 0) {
      observation.mode = std::numeric_limits<size_t>::max();
    }
    humanoid_mpc_msgs::SystemObservation message;
    toProto(observation, &message);
    humanoid_mpc_msgs::SystemObservation parsed;
    ASSERT_TRUE(parsed.ParseFromString(message.SerializeAsString()));
    SystemObservation decoded;
    ASSERT_TRUE(fromProto(parsed, &decoded).ok());
    expectSameObservation(observation, decoded);
  }
}

TEST(SystemObservationConversionTest, EmptyMessageIsTheDefaultObservation) {
  std::mt19937 generator(/*sd=*/2);
  SystemObservation decoded = test_data::randomObservation(generator, /*stateDim=*/3, /*inputDim=*/2);
  ASSERT_TRUE(fromProto(humanoid_mpc_msgs::SystemObservation(), &decoded).ok());
  expectSameObservation(SystemObservation(), decoded);
}

TEST(SystemObservationConversionTest, RejectsNonFiniteValuesAndLeavesTheOutputUnchanged) {
  std::mt19937 generator(/*sd=*/3);
  const SystemObservation sentinel = test_data::randomObservation(generator, /*stateDim=*/4, /*inputDim=*/2);
  humanoid_mpc_msgs::SystemObservation valid;
  toProto(test_data::randomObservation(generator, /*stateDim=*/4, /*inputDim=*/2), &valid);

  const std::vector<std::pair<std::function<void(humanoid_mpc_msgs::SystemObservation* absl_nonnull)>, std::string>> cases = {
      {[](humanoid_mpc_msgs::SystemObservation* absl_nonnull message) { message->set_time(kNaN); }, "SystemObservation.time"},
      {[](humanoid_mpc_msgs::SystemObservation* absl_nonnull message) { message->set_state(/*index=*/2, kInfinity); },
       "SystemObservation.state[2]"},
      {[](humanoid_mpc_msgs::SystemObservation* absl_nonnull message) { message->set_input(/*index=*/1, -kInfinity); },
       "SystemObservation.input[1]"},
  };
  for (const std::pair<std::function<void(humanoid_mpc_msgs::SystemObservation* absl_nonnull)>, std::string>& testCase : cases) {
    humanoid_mpc_msgs::SystemObservation message = valid;
    testCase.first(&message);
    SystemObservation output = sentinel;
    const absl::Status status = fromProto(message, &output);
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << status;
    EXPECT_TRUE(absl::StrContains(status.message(), testCase.second)) << status;
    expectSameObservation(sentinel, output);
  }
}

TEST(ModeScheduleConversionTest, RoundTripKeepsModesBeyondOneByte) {
  std::mt19937 generator(/*sd=*/4);
  for (int trial = 0; trial < 20; ++trial) {
    ModeSchedule modeSchedule =
        test_data::randomModeSchedule(generator, randomSize(generator, /*low=*/0, /*high=*/10), /*startTime=*/static_cast<scalar_t>(trial));
    // Modes that do not fit in an octet, which a one-byte encoding would wrap.
    modeSchedule.modeSequence.front() = (trial % 2 == 0) ? 256 : std::numeric_limits<size_t>::max();
    humanoid_mpc_msgs::ModeSchedule message;
    toProto(modeSchedule, &message);
    humanoid_mpc_msgs::ModeSchedule parsed;
    ASSERT_TRUE(parsed.ParseFromString(message.SerializeAsString()));
    ModeSchedule decoded;
    ASSERT_TRUE(fromProto(parsed, &decoded).ok());
    expectSameModeSchedule(modeSchedule, decoded);
  }
}

TEST(ModeScheduleConversionTest, RejectsInconsistentSchedules) {
  humanoid_mpc_msgs::ModeSchedule valid;
  toProto(ModeSchedule({1.0, 2.0}, {3, 4, 5}), &valid);
  const std::vector<std::pair<std::function<void(humanoid_mpc_msgs::ModeSchedule* absl_nonnull)>, std::string>> cases = {
      {[](humanoid_mpc_msgs::ModeSchedule* absl_nonnull message) { message->clear_mode_sequence(); },
       "ModeSchedule.mode_sequence is empty"},
      {[](humanoid_mpc_msgs::ModeSchedule* absl_nonnull message) { message->add_mode_sequence(6); },
       "ModeSchedule.mode_sequence has 4 modes"},
      {[](humanoid_mpc_msgs::ModeSchedule* absl_nonnull message) { message->set_event_times(1, 0.5); }, "ModeSchedule.event_times[1]"},
      {[](humanoid_mpc_msgs::ModeSchedule* absl_nonnull message) { message->set_event_times(/*index=*/0, kNaN); },
       "ModeSchedule.event_times[0]"},
  };
  for (const std::pair<std::function<void(humanoid_mpc_msgs::ModeSchedule* absl_nonnull)>, std::string>& testCase : cases) {
    humanoid_mpc_msgs::ModeSchedule message = valid;
    testCase.first(&message);
    ModeSchedule output;
    const absl::Status status = fromProto(message, &output);
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << status;
    EXPECT_TRUE(absl::StrContains(status.message(), testCase.second)) << status;
    expectSameModeSchedule(ModeSchedule(), output);
  }
}

TEST(TargetTrajectoriesConversionTest, RoundTripReproducesEmptySingleNodeAndLongTrajectories) {
  std::mt19937 generator(/*sd=*/5);
  for (const size_t nodes : {size_t{0}, size_t{1}, size_t{2}, randomSize(generator, /*low=*/3, /*high=*/30)}) {
    for (const size_t inputDim : {size_t{0}, size_t{4}}) {
      const TargetTrajectories targetTrajectories = test_data::randomTargetTrajectories(generator, nodes, /*stateDim=*/6, inputDim);
      humanoid_mpc_msgs::TargetTrajectories message;
      toProto(targetTrajectories, &message);
      humanoid_mpc_msgs::TargetTrajectories parsed;
      ASSERT_TRUE(parsed.ParseFromString(message.SerializeAsString()));
      TargetTrajectories decoded = test_data::randomTargetTrajectories(generator, /*nodes=*/3, /*stateDim=*/2, /*inputDim=*/2);
      ASSERT_TRUE(fromProto(parsed, &decoded).ok());
      expectSameTargetTrajectories(targetTrajectories, decoded);
    }
  }
}

TEST(TargetTrajectoriesConversionTest, RejectsMismatchedLengthsUnsortedTimesAndNonFiniteValues) {
  std::mt19937 generator(/*sd=*/6);
  humanoid_mpc_msgs::TargetTrajectories valid;
  toProto(test_data::randomTargetTrajectories(generator, /*nodes=*/4, /*stateDim=*/3, /*inputDim=*/2), &valid);
  const std::vector<std::pair<std::function<void(humanoid_mpc_msgs::TargetTrajectories* absl_nonnull)>, std::string>> cases = {
      {[](humanoid_mpc_msgs::TargetTrajectories* absl_nonnull message) { message->mutable_state()->RemoveLast(); },
       "TargetTrajectories.state has 3 entries but TargetTrajectories.time has 4"},
      {[](humanoid_mpc_msgs::TargetTrajectories* absl_nonnull message) { message->mutable_input()->RemoveLast(); },
       "TargetTrajectories.input has 3"},
      {[](humanoid_mpc_msgs::TargetTrajectories* absl_nonnull message) { message->set_time(/*index=*/2, message->time(0) - 1.0); },
       "TargetTrajectories.time[2]"},
      {[](humanoid_mpc_msgs::TargetTrajectories* absl_nonnull message) { message->mutable_state(1)->set_data(/*index=*/2, kNaN); },
       "TargetTrajectories.state[1].data[2]"},
      {[](humanoid_mpc_msgs::TargetTrajectories* absl_nonnull message) { message->mutable_input(3)->set_data(/*index=*/0, kInfinity); },
       "TargetTrajectories.input[3].data[0]"},
  };
  for (const std::pair<std::function<void(humanoid_mpc_msgs::TargetTrajectories* absl_nonnull)>, std::string>& testCase : cases) {
    humanoid_mpc_msgs::TargetTrajectories message = valid;
    testCase.first(&message);
    TargetTrajectories output;
    const absl::Status status = fromProto(message, &output);
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << status;
    EXPECT_TRUE(absl::StrContains(status.message(), testCase.second)) << status;
    EXPECT_TRUE(output.empty());
  }
}

TEST(PerformanceIndexConversionTest, RoundTripKeepsNonFiniteDiagnostics) {
  std::mt19937 generator(/*sd=*/7);
  PerformanceIndex performanceIndex = test_data::randomPerformanceIndex(generator);
  performanceIndex.merit = kNaN;  // what a diverged solve reports
  performanceIndex.cost = kInfinity;
  humanoid_mpc_msgs::PerformanceIndex message;
  toProto(performanceIndex, &message);
  humanoid_mpc_msgs::PerformanceIndex parsed;
  ASSERT_TRUE(parsed.ParseFromString(message.SerializeAsString()));
  PerformanceIndex decoded;
  ASSERT_TRUE(fromProto(parsed, &decoded).ok());
  expectSamePerformanceIndex(performanceIndex, decoded);
}

// =====================================================================================================================
// The policy: round trips
// =====================================================================================================================

TEST(PolicyConversionTest, RoundTripReproducesThePolicyNodeForNode) {
  std::mt19937 generator(/*sd=*/8);
  for (const ControllerType controllerType : kControllerTypes) {
    for (int trial = 0; trial < 30; ++trial) {
      const PolicyShape shape = randomShape(generator, controllerType);
      const Policy policy = randomPolicy(generator, shape, /*targetNodes=*/randomSize(generator, /*low=*/0, /*high=*/5));
      expectSamePolicy(policy, roundTrip(policy));
    }
  }
}

TEST(PolicyConversionTest, RoundTripReproducesASingleNodePolicy) {
  std::mt19937 generator(/*sd=*/9);
  for (const ControllerType controllerType : kControllerTypes) {
    const Policy policy = randomPolicy(generator, {.nodes = 1, .stateDim = 5, .inputDim = 3, .controllerType = controllerType},
                                       /*targetNodes=*/1);
    const Policy decoded = roundTrip(policy);
    expectSamePolicy(policy, decoded);
    // A single node is a constant policy, before and after the trip.
    const vector_t state = test_data::randomVector(generator, /*size=*/5);
    EXPECT_TRUE(bitwiseEqual(policy.primalSolution.controllerPtr_->computeInput(/*t=*/-1.0, state),
                             decoded.primalSolution.controllerPtr_->computeInput(/*t=*/1.0e3, state)));
  }
}

TEST(PolicyConversionTest, RebuildsTheController) {
  // The decoded policy carries a controller of the encoded type and size.
  std::mt19937 generator(/*sd=*/10);
  for (const ControllerType controllerType : kControllerTypes) {
    const Policy policy = randomPolicy(generator, {.nodes = 7, .stateDim = 4, .inputDim = 2, .events = 2, .controllerType = controllerType},
                                       /*targetNodes=*/2);
    // Decoding replaces whatever controller the output held, here one of the other type.
    Policy decoded;
    const ControllerType otherType = (controllerType == ControllerType::LINEAR) ? ControllerType::FEEDFORWARD : ControllerType::LINEAR;
    decoded.primalSolution =
        test_data::randomPrimalSolution(generator, {.nodes = 3, .stateDim = 1, .inputDim = 1, .controllerType = otherType});
    ASSERT_TRUE(decode(encode(policy), &decoded).ok());
    ASSERT_NE(decoded.primalSolution.controllerPtr_, nullptr);
    EXPECT_EQ(decoded.primalSolution.controllerPtr_->getType(), controllerType);
    EXPECT_EQ(decoded.primalSolution.controllerPtr_->size(), 7);
  }
}

TEST(PolicyConversionTest, RebuiltPolicyComputesTheSameInputs) {
  std::mt19937 generator(/*sd=*/11);
  for (const ControllerType controllerType : kControllerTypes) {
    for (int trial = 0; trial < 20; ++trial) {
      const PolicyShape shape = randomShape(generator, controllerType);
      const Policy policy = randomPolicy(generator, shape, /*targetNodes=*/2);
      const Policy decoded = roundTrip(policy);
      for (const scalar_t time : queryTimes(generator, policy.primalSolution.timeTrajectory_)) {
        const vector_t state = test_data::randomVector(generator, shape.stateDim);
        EXPECT_TRUE(bitwiseEqual(policy.primalSolution.controllerPtr_->computeInput(time, state),
                                 decoded.primalSolution.controllerPtr_->computeInput(time, state)))
            << "t = " << time << ", trial " << trial;
      }
    }
  }
}

TEST(PolicyConversionTest, KeepsBothNodesOfAnEvent) {
  // At an event the pre- and post-event nodes share one time. ControllerBase::flatten() at the time trajectory samples
  // the pre-event node for both, so a policy rebuilt from it would apply the pre-event input after the event. The
  // conversion sends the nodes themselves instead.
  std::mt19937 generator(/*sd=*/12);
  for (const ControllerType controllerType : kControllerTypes) {
    const PolicyShape shape{.nodes = 10, .stateDim = 3, .inputDim = 2, .events = 1, .controllerType = controllerType};
    const Policy policy = randomPolicy(generator, shape, /*targetNodes=*/1);
    const PrimalSolution& primalSolution = policy.primalSolution;
    ASSERT_EQ(primalSolution.postEventIndices_.size(), size_t{1});
    const size_t postEvent = primalSolution.postEventIndices_.front();
    ASSERT_EQ(primalSolution.timeTrajectory_[postEvent - 1], primalSolution.timeTrajectory_[postEvent]);

    const std::vector<std::vector<double>> sampled = flatten(*primalSolution.controllerPtr_, primalSolution.timeTrajectory_);
    EXPECT_EQ(sampled[postEvent], sampled[postEvent - 1]) << "flatten() no longer collapses an event; revisit writeController()";

    const humanoid_mpc_msgs::MpcPolicy message = encode(policy);
    EXPECT_NE(asStdVector(message.controller_data(static_cast<int>(postEvent))),
              asStdVector(message.controller_data(static_cast<int>(postEvent - 1))));

    // Just after the event the received policy interpolates from the post-event node, as the sent one does.
    Policy decoded;
    ASSERT_TRUE(decode(message, &decoded).ok());
    const scalar_t afterEvent = primalSolution.timeTrajectory_[postEvent] + 1.0e-6;
    const vector_t state = test_data::randomVector(generator, shape.stateDim);
    const vector_t expected = primalSolution.controllerPtr_->computeInput(afterEvent, state);
    EXPECT_TRUE(bitwiseEqual(expected, decoded.primalSolution.controllerPtr_->computeInput(afterEvent, state)));
  }
}

TEST(PolicyConversionTest, ControllerDataHasTheLayoutOfFlattenAndUnflatten) {
  // Without events every time is distinct, and the nodes sent are exactly what ControllerBase::flatten() writes; the
  // controller rebuilt from them is the one LinearController::unFlatten() or FeedforwardController::unFlatten() builds.
  std::mt19937 generator(/*sd=*/13);
  for (const ControllerType controllerType : kControllerTypes) {
    const PolicyShape shape{.nodes = 12, .stateDim = 5, .inputDim = 3, .events = 0, .controllerType = controllerType};
    const Policy policy = randomPolicy(generator, shape, /*targetNodes=*/1);
    const scalar_array_t& time = policy.primalSolution.timeTrajectory_;
    const std::vector<std::vector<double>> flat = flatten(*policy.primalSolution.controllerPtr_, time);

    const humanoid_mpc_msgs::MpcPolicy message = encode(policy);
    ASSERT_EQ(message.controller_data_size(), static_cast<int>(shape.nodes));
    std::vector<const std::vector<double>* absl_nonnull> flatRefs;
    for (size_t k = 0; k < shape.nodes; ++k) {
      EXPECT_EQ(asStdVector(message.controller_data(static_cast<int>(k))), flat[k]) << "node " << k;
      flatRefs.push_back(&flat[k]);
    }

    Policy decoded;
    ASSERT_TRUE(decode(message, &decoded).ok());
    if (controllerType == ControllerType::LINEAR) {
      const size_array_t stateDims(shape.nodes, shape.stateDim);
      const size_array_t inputDims(shape.nodes, shape.inputDim);
      const LinearController unflattened = LinearController::unFlatten(stateDims, inputDims, time, flatRefs);
      expectSameController(&unflattened, decoded.primalSolution.controllerPtr_.get());
    } else {
      const FeedforwardController unflattened = FeedforwardController::unFlatten(time, flatRefs);
      expectSameController(&unflattened, decoded.primalSolution.controllerPtr_.get());
    }
  }
}

TEST(PolicyConversionTest, SamplesAControllerOnOtherTimeStampsWithFlatten) {
  std::mt19937 generator(/*sd=*/14);
  for (const ControllerType controllerType : kControllerTypes) {
    const PolicyShape shape{.nodes = 9, .stateDim = 4, .inputDim = 2, .events = 0, .controllerType = controllerType};
    Policy policy = randomPolicy(generator, shape, /*targetNodes=*/1);
    // A controller on twice as many time stamps as the trajectory, as upstream OCS2's DDP rollout may have.
    PolicyShape denser = shape;
    denser.nodes = 2 * shape.nodes;
    policy.primalSolution.controllerPtr_ = std::move(test_data::randomPrimalSolution(generator, denser).controllerPtr_);

    const humanoid_mpc_msgs::MpcPolicy message = encode(policy);
    const std::vector<std::vector<double>> flat = flatten(*policy.primalSolution.controllerPtr_, policy.primalSolution.timeTrajectory_);
    ASSERT_EQ(message.controller_data_size(), static_cast<int>(shape.nodes));
    for (size_t k = 0; k < shape.nodes; ++k) {
      EXPECT_EQ(asStdVector(message.controller_data(static_cast<int>(k))), flat[k]) << "node " << k;
    }
  }
}

// =====================================================================================================================
// The policy: encoding
// =====================================================================================================================

TEST(PolicyConversionTest, EncodingIntoAReusedMessageGivesTheBytesOfAFreshOne) {
  // The encoder resizes fields rather than clearing them; nothing of an earlier, larger or other-typed policy remains.
  std::mt19937 generator(/*sd=*/15);
  humanoid_mpc_msgs::MpcPolicy reused;
  for (int trial = 0; trial < 20; ++trial) {
    const ControllerType controllerType = kControllerTypes[static_cast<size_t>(trial) % kControllerTypes.size()];
    const Policy policy =
        randomPolicy(generator, randomShape(generator, controllerType), /*targetNodes=*/randomSize(generator, /*low=*/0, /*high=*/4));
    ASSERT_TRUE(policyToProto(policy.commandData, policy.primalSolution, policy.performanceIndex, &reused).ok());
    EXPECT_EQ(reused.SerializeAsString(), encode(policy).SerializeAsString()) << "trial " << trial;
  }
}

TEST(PolicyConversionTest, LeavesTheFieldsTheMpcNodeOwns) {
  std::mt19937 generator(/*sd=*/16);
  const Policy policy = randomPolicy(generator, {.nodes = 4, .stateDim = 2, .inputDim = 1}, /*targetNodes=*/1);
  humanoid_mpc_msgs::MpcPolicy message;
  message.set_resets_served(7);
  message.set_full_resets_served(3);
  message.mutable_solver_status()->set_healthy(true);
  message.mutable_annotations()->add_target_contact_patches()->mutable_pose()->set_x(1.0);
  ASSERT_TRUE(policyToProto(policy.commandData, policy.primalSolution, policy.performanceIndex, &message).ok());
  EXPECT_EQ(message.resets_served(), 7);
  EXPECT_EQ(message.full_resets_served(), 3);
  EXPECT_TRUE(message.solver_status().healthy());
  ASSERT_EQ(message.annotations().target_contact_patches_size(), 1);
}

TEST(PolicyConversionTest, RejectsPrimalSolutionsItCannotSendAndLeavesTheMessageUnchanged) {
  std::mt19937 generator(/*sd=*/17);
  const Policy policy =
      randomPolicy(generator, {.nodes = 5, .stateDim = 3, .inputDim = 2, .events = 1, .controllerType = ControllerType::LINEAR},
                   /*targetNodes=*/1);
  const humanoid_mpc_msgs::MpcPolicy previous = encode(randomPolicy(generator, {.nodes = 3}, /*targetNodes=*/1));

  const std::vector<std::pair<std::function<void(PrimalSolution* absl_nonnull)>, std::string>> cases = {
      {[](PrimalSolution* absl_nonnull primalSolution) { primalSolution->controllerPtr_.reset(); }, "is null"},
      {[](PrimalSolution* absl_nonnull primalSolution) { primalSolution->controllerPtr_ = std::make_unique<UnsupportedController>(); },
       "only a FeedforwardController, a LinearController or a ManifoldLinearController"},
      {[](PrimalSolution* absl_nonnull primalSolution) { primalSolution->controllerPtr_->clear(); },
       "is empty but the time trajectory has 5 nodes"},
      {[](PrimalSolution* absl_nonnull primalSolution) {
         dynamic_cast<LinearController&>(*primalSolution->controllerPtr_).gainArray_.pop_back();
       },
       "5 time stamps, 5 biases and 4 gains"},
      {[](PrimalSolution* absl_nonnull primalSolution) {
         LinearController& linear = dynamic_cast<LinearController&>(*primalSolution->controllerPtr_);
         linear.gainArray_[2] = matrix_t::Zero(3, 3);
       },
       "node 2 of the LinearController has a gain of 3 rows but a bias of 2 entries"},
  };
  for (const std::pair<std::function<void(PrimalSolution* absl_nonnull)>, std::string>& testCase : cases) {
    PrimalSolution primalSolution = policy.primalSolution;
    testCase.first(&primalSolution);
    humanoid_mpc_msgs::MpcPolicy message = previous;
    const absl::Status status = policyToProto(policy.commandData, primalSolution, policy.performanceIndex, &message);
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << status;
    EXPECT_TRUE(absl::StrContains(status.message(), testCase.second)) << status;
    EXPECT_EQ(message.SerializeAsString(), previous.SerializeAsString());
  }
}

// =====================================================================================================================
// The policy: decoding malformed messages
// =====================================================================================================================

struct MalformedPolicy {
  std::string description;
  ControllerType controllerType;
  std::function<void(humanoid_mpc_msgs::MpcPolicy* absl_nonnull)> corrupt;
  // A part of the error message, naming the offending field.
  std::string expectedError;
};

TEST(PolicyConversionTest, RejectsMalformedMessagesAndLeavesTheOutputsUnchanged) {
  using humanoid_mpc_msgs::MpcPolicy;
  constexpr ControllerType kLinear = ControllerType::LINEAR;
  constexpr ControllerType kFeedforward = ControllerType::FEEDFORWARD;
  const std::vector<MalformedPolicy> cases = {
      {"no nodes", kLinear,
       [](MpcPolicy* absl_nonnull message) {
         message->clear_time_trajectory();
         message->clear_state_trajectory();
         message->clear_input_trajectory();
         message->clear_controller_data();
         message->clear_post_event_indices();
       },
       "MpcPolicy.time_trajectory is empty"},
      {"one state too few", kLinear, [](MpcPolicy* absl_nonnull message) { message->mutable_state_trajectory()->RemoveLast(); },
       "MpcPolicy.state_trajectory has 7 entries but MpcPolicy.time_trajectory has 8"},
      {"one input too few", kLinear, [](MpcPolicy* absl_nonnull message) { message->mutable_input_trajectory()->RemoveLast(); },
       "MpcPolicy.input_trajectory has 7 entries"},
      {"one controller node too many", kLinear, [](MpcPolicy* absl_nonnull message) { message->add_controller_data(); },
       "MpcPolicy.controller_data has 9 entries"},
      {"NaN state", kLinear, [](MpcPolicy* absl_nonnull message) { message->mutable_state_trajectory(2)->set_data(/*index=*/1, kNaN); },
       "MpcPolicy.state_trajectory[2].data[1]"},
      {"infinite input", kLinear,
       [](MpcPolicy* absl_nonnull message) { message->mutable_input_trajectory(0)->set_data(/*index=*/0, kInfinity); },
       "MpcPolicy.input_trajectory[0].data[0]"},
      {"NaN gain", kLinear, [](MpcPolicy* absl_nonnull message) { message->mutable_controller_data(1)->set_data(/*index=*/3, kNaN); },
       "MpcPolicy.controller_data[1].data[3]"},
      {"infinite time", kLinear, [](MpcPolicy* absl_nonnull message) { message->set_time_trajectory(/*index=*/1, kInfinity); },
       "MpcPolicy.time_trajectory[1]"},
      {"time going back", kLinear,
       [](MpcPolicy* absl_nonnull message) { message->set_time_trajectory(/*index=*/5, message->time_trajectory(0) - 1.0); },
       "MpcPolicy.time_trajectory[5]"},
      {"linear node one value short", kLinear,
       [](MpcPolicy* absl_nonnull message) { message->mutable_controller_data(3)->mutable_data()->RemoveLast(); },
       "MpcPolicy.controller_data[3] has 11 entries but a LinearController node with 3 inputs and 3 states has 12"},
      {"feedforward node one value long", kFeedforward,
       [](MpcPolicy* absl_nonnull message) { message->mutable_controller_data(0)->add_data(1.0); },
       "MpcPolicy.controller_data[0] has 4 entries but a FeedforwardController node with 3 inputs has 3"},
      {"linear data labeled feedforward", kLinear,
       [](MpcPolicy* absl_nonnull message) { message->set_controller_type(humanoid_mpc_msgs::CONTROLLER_TYPE_FEEDFORWARD); },
       "MpcPolicy.controller_data[0] has 12 entries but a FeedforwardController node"},
      {"unknown controller type", kLinear,
       [](MpcPolicy* absl_nonnull message) { message->set_controller_type(humanoid_mpc_msgs::CONTROLLER_TYPE_UNKNOWN); },
       "MpcPolicy.controller_type is 0"},
      {"controller type from a newer sender", kLinear,
       [](MpcPolicy* absl_nonnull message) { message->set_controller_type(static_cast<humanoid_mpc_msgs::ControllerType>(7)); },
       "MpcPolicy.controller_type is 7"},
      {"post-event index zero", kLinear, [](MpcPolicy* absl_nonnull message) { message->set_post_event_indices(0, 0); },
       "MpcPolicy.post_event_indices[0] is 0"},
      {"post-event index past the end", kLinear, [](MpcPolicy* absl_nonnull message) { message->set_post_event_indices(0, 9); },
       "MpcPolicy.post_event_indices[0] is 9; a post-event index lies in [1, 7]"},
      // A post-event index names a node: the node count itself names none.
      {"post-event index at the node count", kLinear, [](MpcPolicy* absl_nonnull message) { message->set_post_event_indices(0, 8); },
       "MpcPolicy.post_event_indices[0] is 8; a post-event index lies in [1, 7]"},
      {"post-event indices repeating", kLinear,
       [](MpcPolicy* absl_nonnull message) { message->add_post_event_indices(message->post_event_indices(0)); },
       "MpcPolicy.post_event_indices[1] = 4 does not exceed"},
      {"mode schedule without modes", kLinear,
       [](MpcPolicy* absl_nonnull message) {
         message->mutable_mode_schedule()->clear_mode_sequence();
         message->mutable_mode_schedule()->clear_event_times();
       },
       "MpcPolicy.mode_schedule.mode_sequence is empty"},
      {"mode schedule with a mode too few", kLinear,
       [](MpcPolicy* absl_nonnull message) { message->mutable_mode_schedule()->add_event_times(1.0e3); },
       "MpcPolicy.mode_schedule.mode_sequence has 2 modes but MpcPolicy.mode_schedule.event_times has 2"},
      {"NaN event time", kLinear,
       [](MpcPolicy* absl_nonnull message) { message->mutable_mode_schedule()->set_event_times(/*index=*/0, kNaN); },
       "MpcPolicy.mode_schedule.event_times[0]"},
      {"NaN initial state", kLinear,
       [](MpcPolicy* absl_nonnull message) { message->mutable_init_observation()->set_state(/*index=*/0, kNaN); },
       "MpcPolicy.init_observation.state[0]"},
      {"infinite initial time", kLinear, [](MpcPolicy* absl_nonnull message) { message->mutable_init_observation()->set_time(kInfinity); },
       "MpcPolicy.init_observation.time"},
      {"target state missing", kLinear,
       [](MpcPolicy* absl_nonnull message) { message->mutable_target_trajectories()->mutable_state()->RemoveLast(); },
       "MpcPolicy.target_trajectories.state has 2 entries but MpcPolicy.target_trajectories.time has 3"},
      {"target input missing", kLinear,
       [](MpcPolicy* absl_nonnull message) { message->mutable_target_trajectories()->mutable_input()->RemoveLast(); },
       "MpcPolicy.target_trajectories.input has 2 entries"},
      {"target time going back", kLinear,
       [](MpcPolicy* absl_nonnull message) {
         message->mutable_target_trajectories()->set_time(/*index=*/2, message->target_trajectories().time(0) - 1.0);
       },
       "MpcPolicy.target_trajectories.time[2]"},
  };

  std::mt19937 generator(/*sd=*/18);
  for (const MalformedPolicy& testCase : cases) {
    const PolicyShape shape{.nodes = 8, .stateDim = 3, .inputDim = 3, .events = 1, .controllerType = testCase.controllerType};
    humanoid_mpc_msgs::MpcPolicy message = encode(randomPolicy(generator, shape, /*targetNodes=*/3));
    testCase.corrupt(&message);

    const Policy sentinel = randomPolicy(generator, {.nodes = 2, .stateDim = 1, .inputDim = 1}, /*targetNodes=*/1);
    Policy output = {
        .commandData = sentinel.commandData, .primalSolution = sentinel.primalSolution, .performanceIndex = sentinel.performanceIndex};
    const absl::Status status = decode(message, &output);
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << testCase.description << ": " << status;
    EXPECT_TRUE(absl::StrContains(status.message(), testCase.expectedError)) << testCase.description << ": " << status;
    expectSamePolicy(sentinel, output);
  }
}

// =====================================================================================================================
// The policy of a solver on a state manifold: CONTROLLER_TYPE_MANIFOLD_LINEAR
// =====================================================================================================================

constexpr size_t kManifoldInputDim = 3;

/** A policy as the SQP solves it on the toy manifold [E(3), Q, E(5)], with a ManifoldLinearController. */
Policy randomManifoldPolicy(std::mt19937& generator,
                            std::shared_ptr<const StateManifold> manifold,
                            size_t nodes,
                            size_t events,
                            ControllerType controllerType = ControllerType::MANIFOLD_LINEAR) {
  const PolicyShape shape{.nodes = nodes,
                          .stateDim = manifold->getAmbientDim(),
                          .inputDim = kManifoldInputDim,
                          .events = events,
                          .controllerType = controllerType,
                          .stateManifold = std::move(manifold)};
  return randomPolicy(generator, shape, /*targetNodes=*/2);
}

const ManifoldLinearController& manifoldControllerOf(const Policy& policy) {
  const ManifoldLinearController* absl_nullable controller = asManifoldLinearController(*policy.primalSolution.controllerPtr_);
  CHECK(controller != nullptr);
  return *controller;
}

// The times a manifold policy is evaluated at: queryTimes() and, at each event, its time and the times just around it.
scalar_array_t manifoldQueryTimes(std::mt19937& generator, const PrimalSolution& primalSolution) {
  scalar_array_t times = queryTimes(generator, primalSolution.timeTrajectory_);
  for (const size_t postEvent : primalSolution.postEventIndices_) {
    const scalar_t eventTime = primalSolution.timeTrajectory_[postEvent];
    for (const scalar_t offset : {-1.0e-6, -1.0e-12, 0.0, 1.0e-12, 1.0e-6}) {
      times.push_back(eventTime + offset);
    }
  }
  return times;
}

TEST(ManifoldPolicyConversionTest, ASqpPolicyWithAnEventRoundTripsItsAnchorsInputsAndGainsToTheBit) {
  const std::shared_ptr<const StateManifold> manifold = test_data::toyStateManifold();
  std::mt19937 generator(/*sd=*/40);
  const Policy policy = randomManifoldPolicy(generator, manifold, /*nodes=*/12, /*events=*/1);
  const PrimalSolution& primalSolution = policy.primalSolution;
  const ManifoldLinearController& sent = manifoldControllerOf(policy);

  // Positive controls: toPrimalSolution() gave the pre-event node and the last node the anchor of the node before them,
  // so the anchors are not the state trajectory, which the receiver would otherwise have been free to use.
  ASSERT_EQ(primalSolution.postEventIndices_.size(), size_t{1});
  const size_t preEvent = primalSolution.postEventIndices_.front() - 1;
  ASSERT_EQ(primalSolution.timeTrajectory_[preEvent], primalSolution.timeTrajectory_[preEvent + 1]);
  EXPECT_EQ(sent.getAnchorStates()[preEvent], sent.getAnchorStates()[preEvent - 1]);
  EXPECT_NE(sent.getAnchorStates()[preEvent], primalSolution.stateTrajectory_[preEvent]);
  EXPECT_EQ(sent.getAnchorStates().back(), sent.getAnchorStates()[sent.getAnchorStates().size() - 2]);
  EXPECT_NE(sent.getAnchorStates().back(), primalSolution.stateTrajectory_.back());

  const humanoid_mpc_msgs::MpcPolicy message = encode(policy);
  EXPECT_EQ(message.controller_type(), humanoid_mpc_msgs::CONTROLLER_TYPE_MANIFOLD_LINEAR);
  const size_t nodeSize = kManifoldInputDim + manifold->getAmbientDim() + kManifoldInputDim * manifold->getTangentDim();
  for (const humanoid_mpc_msgs::Vector& node : message.controller_data()) {
    EXPECT_EQ(static_cast<size_t>(node.data_size()), nodeSize);
  }

  const Policy decoded = roundTrip(policy, manifold);
  expectSamePolicy(policy, decoded);
  // The same law at every node, on both sides of the event and between the nodes, at states of the manifold.
  ManifoldLinearController sentCopy = sent;
  ManifoldLinearController& received = dynamic_cast<ManifoldLinearController&>(*decoded.primalSolution.controllerPtr_);
  for (const scalar_t time : manifoldQueryTimes(generator, primalSolution)) {
    const vector_t state = test_data::randomManifoldState(generator, *manifold);
    EXPECT_TRUE(bitwiseEqual(sentCopy.computeInput(time, state), received.computeInput(time, state))) << "t = " << time;
  }
  // Positive control: the law changes across the event, which a policy that lost its post-event node would not show.
  const scalar_t eventTime = primalSolution.timeTrajectory_[preEvent];
  const vector_t state = test_data::randomManifoldState(generator, *manifold);
  EXPECT_FALSE(bitwiseEqual(received.computeInput(eventTime, state), received.computeInput(eventTime + 1.0e-6, state)));
}

TEST(ManifoldPolicyConversionTest, NodeArraysAreWhatFlattenWritesAtTheNodeTimes) {
  // Without events every time is distinct: the nodes sent are what ManifoldLinearController::flatten() writes there, and
  // the controller rebuilt from them is the one its unFlatten() builds.
  const std::shared_ptr<const StateManifold> manifold = test_data::toyStateManifold();
  std::mt19937 generator(/*sd=*/41);
  const Policy policy = randomManifoldPolicy(generator, manifold, /*nodes=*/9, /*events=*/0);
  const scalar_array_t& time = policy.primalSolution.timeTrajectory_;
  const std::vector<std::vector<double>> flat = flatten(*policy.primalSolution.controllerPtr_, time);
  const humanoid_mpc_msgs::MpcPolicy message = encode(policy);
  ASSERT_EQ(message.controller_data_size(), static_cast<int>(time.size()));
  std::vector<const std::vector<double>* absl_nullable> flatRefs;
  for (size_t k = 0; k < time.size(); ++k) {
    EXPECT_EQ(asStdVector(message.controller_data(static_cast<int>(k))), flat[k]) << "node " << k;
    flatRefs.push_back(&flat[k]);
  }
  Policy decoded;
  ASSERT_TRUE(decode(message, manifold, &decoded).ok());
  absl::StatusOr<ManifoldLinearController> unflattened = ManifoldLinearController::unFlatten(
      size_array_t(time.size(), manifold->getAmbientDim()), size_array_t(time.size(), kManifoldInputDim),
      size_array_t(time.size(), manifold->getTangentDim()), time, flatRefs, manifold);
  ASSERT_TRUE(unflattened.ok()) << unflattened.status();
  expectSameController(&*unflattened, decoded.primalSolution.controllerPtr_.get());
}

TEST(ManifoldPolicyConversionTest, AFeedforwardPolicyOnAManifoldRoundTrips) {
  const std::shared_ptr<const StateManifold> manifold = test_data::toyStateManifold();
  std::mt19937 generator(/*sd=*/42);
  const Policy policy = randomManifoldPolicy(generator, manifold, /*nodes=*/10, /*events=*/2, ControllerType::FEEDFORWARD);
  expectSamePolicy(policy, roundTrip(policy, manifold));
}

TEST(ManifoldPolicyConversionTest, RefusesToSendAManifoldPolicyItCannotSendAsItsNodes) {
  const std::shared_ptr<const StateManifold> manifold = test_data::toyStateManifold();
  std::mt19937 generator(/*sd=*/43);
  const Policy policy = randomManifoldPolicy(generator, manifold, /*nodes=*/8, /*events=*/1);
  const ManifoldLinearController& controller = manifoldControllerOf(policy);
  const humanoid_mpc_msgs::MpcPolicy previous = encode(randomPolicy(generator, {.nodes = 3}, /*targetNodes=*/1));

  scalar_array_t shifted = controller.getTimeStamps();
  for (scalar_t& time : shifted) time += 1.0e-3;
  matrix_array_t wideGains = controller.getFeedbackGains();
  wideGains[2] = matrix_t::Zero(kManifoldInputDim, manifold->getTangentDim() + 1);
  vector_array_t flatAnchors = controller.getAnchorStates();
  flatAnchors[1] = vector_t::Zero(manifold->getTangentDim());
  const std::vector<std::pair<ManifoldLinearController, std::string>> cases = {
      {ManifoldLinearController(shifted, controller.getAnchorStates(), controller.getNominalInputs(), controller.getFeedbackGains(),
                                manifold),
       "time stamps are not the time trajectory"},
      {ManifoldLinearController(controller.getTimeStamps(), controller.getAnchorStates(), controller.getNominalInputs(), wideGains,
                                manifold),
       "node 2 of the ManifoldLinearController has an anchor state of 12 entries, a nominal input of 3 and a gain of 3x12"},
      {ManifoldLinearController(controller.getTimeStamps(), flatAnchors, controller.getNominalInputs(), controller.getFeedbackGains(),
                                manifold),
       "node 1 of the ManifoldLinearController has an anchor state of 11 entries"},
  };
  for (const std::pair<ManifoldLinearController, std::string>& testCase : cases) {
    PrimalSolution primalSolution = policy.primalSolution;
    primalSolution.controllerPtr_.reset(testCase.first.clone());
    humanoid_mpc_msgs::MpcPolicy message = previous;
    const absl::Status status = policyToProto(policy.commandData, primalSolution, policy.performanceIndex, &message);
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << status;
    EXPECT_TRUE(absl::StrContains(status.message(), testCase.second)) << status;
    EXPECT_EQ(message.SerializeAsString(), previous.SerializeAsString());
  }
}

TEST(ManifoldPolicyConversionTest, DecodingRefusesWrongStridesAMissingManifoldAndALinearPolicyOnAManifold) {
  const std::shared_ptr<const StateManifold> manifold = test_data::toyStateManifold();
  std::mt19937 generator(/*sd=*/44);
  const humanoid_mpc_msgs::MpcPolicy manifoldMessage = encode(randomManifoldPolicy(generator, manifold, /*nodes=*/8, /*events=*/1));
  const humanoid_mpc_msgs::MpcPolicy linearMessage = encode(randomPolicy(
      generator, {.nodes = 8, .stateDim = 12, .inputDim = 3, .events = 1, .controllerType = ControllerType::LINEAR}, /*targetNodes=*/2));
  humanoid_mpc_msgs::MpcPolicy shortNode = manifoldMessage;
  shortNode.mutable_controller_data(3)->mutable_data()->RemoveLast();
  humanoid_mpc_msgs::MpcPolicy longNode = manifoldMessage;
  longNode.mutable_controller_data(0)->add_data(1.0);
  // The same ambient dimension, but no rotation: its tangent has one dimension more, so every node is one gain column short.
  const std::shared_ptr<const StateManifold> flatOfTheSameSize = std::make_shared<EuclideanStateManifold>(manifold->getAmbientDim());
  const std::shared_ptr<const StateManifold> wider = std::make_shared<EuclideanStateManifold>(manifold->getAmbientDim() + 1);

  struct Case {
    std::string description;
    const humanoid_mpc_msgs::MpcPolicy* absl_nonnull message;
    std::shared_ptr<const StateManifold> stateManifold;
    std::string expectedError;
  };
  const std::vector<Case> cases = {
      {"a node one value short", &shortNode, manifold,
       "MpcPolicy.controller_data[3] has 47 entries but a ManifoldLinearController node with 3 inputs, 12 states and 11 tangent "
       "dimensions has 48 = inputs + states + inputs * tangent"},
      {"a node one value long", &longNode, manifold, "MpcPolicy.controller_data[0] has 49 entries"},
      {"no manifold", &manifoldMessage, nullptr, "CONTROLLER_TYPE_MANIFOLD_LINEAR, but the receiver's state has no state manifold"},
      {"a linear policy on a manifold", &linearMessage, manifold, "CONTROLLER_TYPE_LINEAR, whose gains act on the stored state"},
      {"another tangent dimension", &manifoldMessage, flatOfTheSameSize,
       "a ManifoldLinearController node with 3 inputs, 12 states and 12 tangent dimensions has 51"},
      {"another ambient dimension", &manifoldMessage, wider,
       "MpcPolicy.state_trajectory[0] has 12 entries; the receiver's state manifold has 13 ambient dimensions"},
  };
  for (const Case& testCase : cases) {
    const Policy sentinel = randomPolicy(generator, {.nodes = 2, .stateDim = 1, .inputDim = 1}, /*targetNodes=*/1);
    Policy output = {
        .commandData = sentinel.commandData, .primalSolution = sentinel.primalSolution, .performanceIndex = sentinel.performanceIndex};
    const absl::Status status = decode(*testCase.message, testCase.stateManifold, &output);
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << testCase.description << ": " << status;
    EXPECT_TRUE(absl::StrContains(status.message(), testCase.expectedError)) << testCase.description << ": " << status;
    expectSamePolicy(sentinel, output);
  }
  // The overload without a manifold is the flat receiver.
  Policy output;
  EXPECT_FALSE(decode(manifoldMessage, &output).ok());
  EXPECT_TRUE(decode(linearMessage, &output).ok());
}

// =====================================================================================================================
// Model dimensions
// =====================================================================================================================

// randomMode() draws from [0, 2^40]: a model with one mode more accepts every random policy's modes.
constexpr size_t kModesOfRandomPolicies = (size_t{1} << 40) + 1;

TEST(CheckDimensionsTest, AcceptsThePolicyOfTheModelAndNamesTheFirstMismatch) {
  std::mt19937 generator(/*sd=*/19);
  const PolicyShape shape{.nodes = 6, .stateDim = 5, .inputDim = 3, .events = 1, .controllerType = ControllerType::LINEAR};
  const humanoid_mpc_msgs::MpcPolicy message = encode(randomPolicy(generator, shape, /*targetNodes=*/2));
  const ModelDimensions model{.stateDim = 5, .inputDim = 3, .numModes = kModesOfRandomPolicies};
  EXPECT_TRUE(checkDimensions(message, model).ok());
  EXPECT_TRUE(checkDimensions(message.init_observation(), model).ok());

  const absl::Status wrongState = checkDimensions(message, {.stateDim = 6, .inputDim = 3, .numModes = kModesOfRandomPolicies});
  EXPECT_EQ(wrongState.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(wrongState.message(), "MpcPolicy.init_observation.state has 5 entries; the model's state has 6"))
      << wrongState;
  const absl::Status wrongInput =
      checkDimensions(message.init_observation(), {.stateDim = 5, .inputDim = 2, .numModes = kModesOfRandomPolicies});
  EXPECT_TRUE(absl::StrContains(wrongInput.message(), "SystemObservation.input has 3 entries; the model's input has 2")) << wrongInput;

  humanoid_mpc_msgs::MpcPolicy oneNodeOff = message;
  oneNodeOff.mutable_state_trajectory(4)->add_data(0.0);
  const absl::Status nodeOff = checkDimensions(oneNodeOff, model);
  EXPECT_TRUE(absl::StrContains(nodeOff.message(), "MpcPolicy.state_trajectory[4] has 6 entries; the model's state has 5")) << nodeOff;

  humanoid_mpc_msgs::MpcPolicy targetsWithoutInputs = message;
  targetsWithoutInputs.mutable_target_trajectories()->clear_input();
  EXPECT_TRUE(checkDimensions(targetsWithoutInputs, model).ok());
}

TEST(CheckDimensionsTest, AModeOutsideTheModelIsRejected) {
  // A humanoid's four contact modes: a policy from the network may name no other, whatever it decodes to.
  const ModelDimensions humanoid{.stateDim = 5, .inputDim = 3, .numModes = 4};
  std::mt19937 generator(/*sd=*/23);
  const PolicyShape shape{.nodes = 7, .stateDim = 5, .inputDim = 3, .events = 2, .controllerType = ControllerType::FEEDFORWARD};
  humanoid_mpc_msgs::MpcPolicy message = encode(randomPolicy(generator, shape, /*targetNodes=*/2));
  message.mutable_init_observation()->set_mode(/*value=*/3);
  for (int index = 0; index < message.mode_schedule().mode_sequence_size(); ++index) {
    message.mutable_mode_schedule()->set_mode_sequence(index, static_cast<uint64_t>(index % 4));
  }
  ASSERT_TRUE(checkDimensions(message, humanoid).ok()) << checkDimensions(message, humanoid);

  humanoid_mpc_msgs::MpcPolicy scheduled = message;
  scheduled.mutable_mode_schedule()->set_mode_sequence(/*index=*/2, /*value=*/4);
  const absl::Status outsideSchedule = checkDimensions(scheduled, humanoid);
  EXPECT_EQ(outsideSchedule.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(outsideSchedule.message(), "MpcPolicy.mode_schedule.mode_sequence[2] is 4; the model's modes are 0 to 3"))
      << outsideSchedule;

  humanoid_mpc_msgs::MpcPolicy observed = message;
  observed.mutable_init_observation()->set_mode(uint64_t{1} << 40);
  EXPECT_TRUE(absl::StrContains(checkDimensions(observed, humanoid).message(), "MpcPolicy.init_observation.mode is"))
      << checkDimensions(observed, humanoid);
  EXPECT_FALSE(checkDimensions(observed.init_observation(), humanoid).ok());

  const absl::Status noModes = checkDimensions(message, {.stateDim = 5, .inputDim = 3, .numModes = 0});
  EXPECT_TRUE(absl::StrContains(noModes.message(), "ModelDimensions::numModes is 0")) << noModes;
}

// =====================================================================================================================
// The wire format
// =====================================================================================================================

// The number of doubles a policy puts on the wire, and of the Vector messages that carry them.
struct PayloadCount {
  size_t doubles = 0;
  size_t vectors = 0;
};

PayloadCount countPayload(const Policy& policy) {
  PayloadCount count;
  const SystemObservation& observation = policy.commandData.mpcInitObservation_;
  count.doubles += 1 + static_cast<size_t>(observation.state.size() + observation.input.size());
  const TargetTrajectories& targets = policy.commandData.mpcTargetTrajectories_;
  count.doubles += targets.timeTrajectory.size();
  for (const vector_array_t* absl_nonnull trajectory : {&targets.stateTrajectory, &targets.inputTrajectory,
                                                        &policy.primalSolution.stateTrajectory_, &policy.primalSolution.inputTrajectory_}) {
    for (const vector_t& vector : *trajectory) {
      count.doubles += static_cast<size_t>(vector.size());
      ++count.vectors;
    }
  }
  count.doubles += policy.primalSolution.timeTrajectory_.size() + policy.primalSolution.modeSchedule_.eventTimes.size();
  const humanoid_mpc_msgs::MpcPolicy message = encode(policy);
  for (const humanoid_mpc_msgs::Vector& node : message.controller_data()) {
    count.doubles += static_cast<size_t>(node.data_size());
    ++count.vectors;
  }
  count.doubles += 8;  // the performance index
  return count;
}

TEST(PolicyWireFormatTest, BytesOfAFixedPolicyAreStableAndWithinTheirSizeBound) {
  // The bandwidth example of humanoid_nmpc/docs/distributed_runtime/README.md: nx = nu = 30 over 60 nodes.
  for (const ControllerType controllerType : kControllerTypes) {
    std::mt19937 generator(/*sd=*/2026);
    const PolicyShape shape{.nodes = 60, .stateDim = 30, .inputDim = 30, .events = 4, .controllerType = controllerType};
    const Policy policy = randomPolicy(generator, shape, /*targetNodes=*/2);
    const std::string bytes = encode(policy).SerializeAsString();

    // The same input gives the same bytes, and decoding then encoding them again gives them back.
    EXPECT_EQ(encode(policy).SerializeAsString(), bytes);
    humanoid_mpc_msgs::MpcPolicy parsed;
    ASSERT_TRUE(parsed.ParseFromString(bytes));
    Policy decoded;
    ASSERT_TRUE(decode(parsed, &decoded).ok());
    EXPECT_EQ(encode(decoded).SerializeAsString(), bytes);

    // Every double is 8 bytes, packed; the framing adds at most a tag and two lengths per Vector, plus the top-level
    // fields, the modes and the post-event indices.
    const PayloadCount payload = countPayload(policy);
    const size_t lowerBound = 8 * payload.doubles;
    const size_t upperBound = lowerBound + 8 * payload.vectors + 512;
    EXPECT_GE(bytes.size(), lowerBound);
    EXPECT_LE(bytes.size(), upperBound);
    std::cout << (controllerType == ControllerType::LINEAR ? "linear" : "feedforward")
              << " policy, nx = nu = 30, 60 nodes: " << bytes.size() << " bytes\n";
  }
}

TEST(PolicyWireFormatTest, AManifoldPolicyCarriesItsAnchorsBesideItsTangentGains) {
  // The bandwidth example on a state manifold: a quaternion in a state of nx = 31, ndx = nu = 30, over 60 nodes.
  absl::StatusOr<std::shared_ptr<const ProductStateManifold>> created = ProductStateManifold::create(
      {StateManifoldSegment::euclidean(12), StateManifoldSegment::unitQuaternion(), StateManifoldSegment::euclidean(15)});
  ASSERT_TRUE(created.ok()) << created.status();
  const std::shared_ptr<const StateManifold> manifold = *std::move(created);
  std::mt19937 generator(/*sd=*/2026);
  const PolicyShape shape{.nodes = 60,
                          .stateDim = 31,
                          .inputDim = 30,
                          .events = 4,
                          .controllerType = ControllerType::MANIFOLD_LINEAR,
                          .stateManifold = manifold};
  const Policy policy = randomPolicy(generator, shape, /*targetNodes=*/2);
  const humanoid_mpc_msgs::MpcPolicy message = encode(policy);
  const std::string bytes = message.SerializeAsString();
  // u*, xbar and vec K: nu + nx + nu ndx doubles per node.
  for (const humanoid_mpc_msgs::Vector& node : message.controller_data()) {
    EXPECT_EQ(node.data_size(), 30 + 31 + 30 * 30);
  }

  humanoid_mpc_msgs::MpcPolicy parsed;
  ASSERT_TRUE(parsed.ParseFromString(bytes));
  Policy decoded;
  ASSERT_TRUE(decode(parsed, manifold, &decoded).ok());
  EXPECT_EQ(encode(decoded).SerializeAsString(), bytes);

  const PayloadCount payload = countPayload(policy);
  EXPECT_GE(bytes.size(), 8 * payload.doubles);
  EXPECT_LE(bytes.size(), 8 * payload.doubles + 8 * payload.vectors + 512);
  std::cout << "manifold linear policy, nx = 31, ndx = nu = 30, 60 nodes: " << bytes.size() << " bytes\n";
}

}  // namespace
}  // namespace ocs2::humanoid::ipc
