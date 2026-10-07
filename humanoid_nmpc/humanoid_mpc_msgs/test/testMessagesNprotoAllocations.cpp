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

// The messages of the realtime loop and the MPC link convert without a heap allocation once the struct or message they
// convert into has the value's shape, so that the robot's communication thread can decode a policy and encode an
// observation into objects it keeps. A binary of its own, because the allocation counter replaces malloc for the whole
// process.

#include "gtest/gtest.h"

#include "humanoid_mpc_msgs/fsm_state.nproto.pb.h"
#include "humanoid_mpc_msgs/loop_timing.nproto.pb.h"
#include "humanoid_mpc_msgs/mpc_observation.nproto.pb.h"
#include "humanoid_mpc_msgs/mpc_policy.nproto.pb.h"
#include "humanoid_mpc_msgs/robot_state_sample.nproto.pb.h"
#include "humanoid_mpc_msgs/system_observation.nproto.pb.h"
#include "tools/nproto/test/ProtoAllocations.h"

namespace ocs2::humanoid::msgs {
namespace {

template <typename StructType, typename ProtoType>
struct Conversion {
  using Struct = StructType;
  using Proto = ProtoType;
};

template <typename T>
class RealtimeMessageAllocationTest : public ::testing::Test {};

using RealtimeMessages = ::testing::Types<Conversion<RobotStateSample, humanoid_mpc_msgs::RobotStateSample>,
                                          Conversion<MpcObservation, humanoid_mpc_msgs::MpcObservation>,
                                          Conversion<SystemObservation, humanoid_mpc_msgs::SystemObservation>,
                                          Conversion<LoopTiming, humanoid_mpc_msgs::LoopTiming>,
                                          Conversion<FsmState, humanoid_mpc_msgs::FsmState>,
                                          Conversion<MpcPolicy, humanoid_mpc_msgs::MpcPolicy>>;
TYPED_TEST_SUITE(RealtimeMessageAllocationTest, RealtimeMessages);

TYPED_TEST(RealtimeMessageAllocationTest, ConvertingIntoObjectsOfTheValuesShapeDoesNotAllocate) {
  nproto::test_support::ExpectConversionsIntoSizedObjectsDoNotAllocate<typename TypeParam::Struct, typename TypeParam::Proto>();
}

// A policy of the size the MPC sends: 60 nodes of 60-dimensional vectors.
TEST(RealtimeMessageAllocationTest, AFullSizePolicyConvertsWithoutAllocatingOnceSized) {
  nproto::test_support::ExpectConversionsIntoSizedObjectsDoNotAllocate<MpcPolicy, humanoid_mpc_msgs::MpcPolicy>(
      /*repeatedSize=*/60);
}

}  // namespace
}  // namespace ocs2::humanoid::msgs
