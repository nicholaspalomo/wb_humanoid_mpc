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
#include <vector>

#include <ocs2_mpc/SystemObservation.h>

#include "humanoid_common_mpc_app/robot/RemoteMpcLinkAdapter.h"
#include "humanoid_common_mpc_app/robot/RobotAppOptions.h"
#include "humanoid_nmpc/humanoid_common_mpc_app/robot/test/RobotTestSupport.h"

/*
 * The command line of the robot binaries (the retired MPC link flag, the core lists) and the MpcLink over the remote link:
 * created on a bus that is not running yet, handed to the controller's factory exactly once, its first observation sent.
 */

namespace ocs2::humanoid {
namespace {

TEST(RobotAppOptions, TheRetiredMpcLinkFlagIsRefusedWithItsReplacement) {
  EXPECT_TRUE(checkRetiredMpcLinkFlag("").ok()) << "not given";
  for (const char* value : {"in_process", "remote", "ros"}) {
    const absl::Status refused = checkRetiredMpcLinkFlag(value);
    EXPECT_EQ(refused.code(), absl::StatusCode::kFailedPrecondition) << value;
    EXPECT_NE(refused.message().find("MPC node"), std::string::npos) << "the error names the replacement: " << refused.message();
  }
}

TEST(RobotAppOptions, CoreLists) {
  const std::vector<int> defaults{4, 5};
  EXPECT_EQ(*parseCoreList("default", defaults), defaults);
  EXPECT_TRUE(parseCoreList("none", defaults)->empty());
  EXPECT_TRUE(parseCoreList("", defaults)->empty());
  EXPECT_EQ(*parseCoreList("2, 3,7", defaults), (std::vector<int>{2, 3, 7}));
  EXPECT_FALSE(parseCoreList("2,x", defaults).ok());
  EXPECT_FALSE(parseCoreList("-1", defaults).ok());
}

TEST(RemoteMpcLinkAdapter, IsCreatedBeforeTheBusRunsAndHandedOverOnce) {
  std::unique_ptr<robot::ipc::Bus> bus = robot_test::createLoopbackBus("robot");
  ipc::RemoteMpcLink::Config config;
  config.dimensions = ipc::ModelDimensions{.stateDim = 3, .inputDim = 2, .numModes = 4};
  absl::StatusOr<std::unique_ptr<RemoteMpcLinkAdapter>> adapter = RemoteMpcLinkAdapter::Create(*bus, config);
  ASSERT_TRUE(adapter.ok()) << adapter.status();
  RemoteMpcLinkAdapter* link = adapter->get();
  EXPECT_TRUE(link->isHealthy());
  EXPECT_FALSE(link->initialPolicyReceived());

  const MpcLinkFactory factory = handOverMpcLink(*std::move(adapter));
  std::unique_ptr<MpcLink> taken = factory(/*resetTarget=*/nullptr);
  EXPECT_EQ(taken.get(), link);
  EXPECT_EQ(factory(/*resetTarget=*/nullptr), nullptr) << "the link is handed over once";

  // The first observation goes into the link's triple buffer; the bus's IO thread publishes it once the bus runs.
  SystemObservation observation;
  observation.time = 1.0;
  observation.state = vector_t::Zero(3);
  observation.input = vector_t::Zero(2);
  taken->start(observation);
  ASSERT_TRUE(bus->start().ok());
  EXPECT_TRUE(robot_test::waitFor([&]() { return link->remote().statistics().observationsPublished > 0; }));
  bus->stop();

  // A running bus takes no more subscriptions.
  std::unique_ptr<robot::ipc::Bus> running = robot_test::createLoopbackBus("robot");
  ASSERT_TRUE(running->start().ok());
  EXPECT_FALSE(RemoteMpcLinkAdapter::Create(*running, config).ok());
  running->stop();
}

}  // namespace
}  // namespace ocs2::humanoid
