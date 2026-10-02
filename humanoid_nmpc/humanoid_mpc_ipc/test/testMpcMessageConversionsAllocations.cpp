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

// The encoders reuse the capacity of the message they write into: a caller that keeps one message object, as the MPC
// node keeps its outgoing policy, allocates only while the message grows. A test binary of its own, because the
// allocation counter replaces malloc for the whole process.

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <random>
#include <string>

#include "absl/status/status.h"

#include <ocs2_core/Types.h>
#include <ocs2_core/control/ControllerType.h>

#include "humanoid_mpc_ipc/MpcMessageConversions.h"
#include "humanoid_mpc_msgs/mpc_policy.pb.h"
#include "humanoid_mpc_msgs/system_observation.pb.h"
#include "humanoid_mpc_msgs/target_trajectories.pb.h"
#include "humanoid_nmpc/humanoid_mpc_ipc/test/PolicyTestData.h"
#include "humanoid_state_estimation/humanoid_state_estimator/test/AllocationCounter.h"

namespace ocs2::humanoid::ipc {
namespace {

using estimation::heapAllocationCount;
using test_data::PolicyShape;

constexpr std::array<ControllerType, 2> kControllerTypes = {ControllerType::FEEDFORWARD, ControllerType::LINEAR};

struct Policy {
  CommandData commandData;
  PrimalSolution primalSolution;
  PerformanceIndex performanceIndex;
};

Policy randomPolicy(std::mt19937& generator, const PolicyShape& shape) {
  return {test_data::randomCommandData(generator, shape.stateDim, shape.inputDim, /*targetNodes=*/3),
          test_data::randomPrimalSolution(generator, shape), test_data::randomPerformanceIndex(generator)};
}

// The heap allocations of encoding `policy` into `message`.
size_t allocationsOfEncoding(const Policy& policy, humanoid_mpc_msgs::MpcPolicy* message) {
  const size_t before = heapAllocationCount();
  const absl::Status status = policyToProto(policy.commandData, policy.primalSolution, policy.performanceIndex, message);
  const size_t allocations = heapAllocationCount() - before;
  EXPECT_TRUE(status.ok()) << status;
  return allocations;
}

TEST(ConversionAllocationTest, EncodingAPolicyIntoAMessageThatHeldOneOfItsSizeDoesNotAllocate) {
  // The README's bandwidth example, nx = nu = 30 over 60 nodes with four events.
  for (const ControllerType controllerType : kControllerTypes) {
    std::mt19937 generator(/*sd=*/1);
    const PolicyShape shape{.nodes = 60, .stateDim = 30, .inputDim = 30, .events = 4, .controllerType = controllerType};
    const Policy first = randomPolicy(generator, shape);
    const Policy second = randomPolicy(generator, shape);
    PolicyShape shorter = shape;
    shorter.nodes = 31;
    const Policy shorterPolicy = randomPolicy(generator, shorter);

    humanoid_mpc_msgs::MpcPolicy message;
    EXPECT_GT(allocationsOfEncoding(first, &message), size_t{0}) << "the first encoding sizes the message";
    EXPECT_EQ(allocationsOfEncoding(second, &message), size_t{0});
    // A shorter horizon keeps the surplus nodes for later, so returning to the full one allocates nothing either.
    EXPECT_EQ(allocationsOfEncoding(shorterPolicy, &message), size_t{0});
    EXPECT_EQ(allocationsOfEncoding(first, &message), size_t{0});
  }
}

TEST(ConversionAllocationTest, SerializingIntoAReusedBufferDoesNotAllocate) {
  std::mt19937 generator(/*sd=*/2);
  const PolicyShape shape{.nodes = 60, .stateDim = 30, .inputDim = 30, .events = 4, .controllerType = ControllerType::LINEAR};
  const Policy first = randomPolicy(generator, shape);
  const Policy second = randomPolicy(generator, shape);
  humanoid_mpc_msgs::MpcPolicy message;
  allocationsOfEncoding(first, &message);
  std::string buffer(message.ByteSizeLong(), '\0');
  allocationsOfEncoding(second, &message);

  const size_t before = heapAllocationCount();
  const size_t size = message.ByteSizeLong();
  buffer.resize(size);
  const bool serialized = message.SerializeToArray(buffer.data(), static_cast<int>(size));
  const size_t allocations = heapAllocationCount() - before;
  EXPECT_TRUE(serialized);
  EXPECT_EQ(allocations, size_t{0});
}

TEST(ConversionAllocationTest, ObservationsAndTargetsRoundTripWithoutAllocatingOnceSized) {
  std::mt19937 generator(/*sd=*/3);
  const SystemObservation first = test_data::randomObservation(generator, /*stateDim=*/30, /*inputDim=*/30);
  const SystemObservation second = test_data::randomObservation(generator, /*stateDim=*/30, /*inputDim=*/30);
  const TargetTrajectories firstTargets = test_data::randomTargetTrajectories(generator, /*nodes=*/5, /*stateDim=*/30, /*inputDim=*/30);
  const TargetTrajectories secondTargets = test_data::randomTargetTrajectories(generator, /*nodes=*/5, /*stateDim=*/30, /*inputDim=*/30);

  humanoid_mpc_msgs::SystemObservation observationMessage;
  humanoid_mpc_msgs::TargetTrajectories targetsMessage;
  SystemObservation observation;
  TargetTrajectories targets;
  toProto(first, &observationMessage);
  toProto(firstTargets, &targetsMessage);
  ASSERT_TRUE(fromProto(observationMessage, &observation).ok());
  ASSERT_TRUE(fromProto(targetsMessage, &targets).ok());

  // The MPC node decodes every observation into the one it keeps; the robot encodes into the message it keeps.
  const size_t before = heapAllocationCount();
  toProto(second, &observationMessage);
  toProto(secondTargets, &targetsMessage);
  const absl::Status observationStatus = fromProto(observationMessage, &observation);
  const absl::Status targetsStatus = fromProto(targetsMessage, &targets);
  const size_t allocations = heapAllocationCount() - before;
  EXPECT_TRUE(observationStatus.ok()) << observationStatus;
  EXPECT_TRUE(targetsStatus.ok()) << targetsStatus;
  EXPECT_EQ(allocations, size_t{0});
  EXPECT_EQ(observation.state, second.state);
  EXPECT_EQ(targets.stateTrajectory, secondTargets.stateTrajectory);
}

}  // namespace
}  // namespace ocs2::humanoid::ipc
