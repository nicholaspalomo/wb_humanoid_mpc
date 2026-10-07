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

#include <csignal>
#include <memory>
#include <string>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc_app/node/MpcFiles.h"
#include "humanoid_common_mpc_app/node/NodeBus.h"
#include "humanoid_common_mpc_app/node/ShutdownSignal.h"
#include "robot_core/ResourcePaths.h"
#include "robot_ipc/Bus.h"
#include "robot_ipc/NetworkConfig.h"

/*
 * The process plumbing of the laptop-side binaries: SIGTERM asks for a clean shutdown instead of killing the process,
 * an empty --network_config is the shipped single-machine network, and a node the network does not have is refused.
 */

namespace ocs2::humanoid::node {
namespace {

TEST(ShutdownSignal, TheFirstSigtermAsksForTheShutdownInsteadOfEndingTheProcess) {
  installShutdownSignalHandlers();
  installShutdownSignalHandlers();  // idempotent
  EXPECT_FALSE(shutdownRequested());
  ASSERT_EQ(std::raise(SIGTERM), 0);
  // Still running, and the request is recorded.
  EXPECT_TRUE(shutdownRequested());
  waitForShutdown();  // returns at once
}

TEST(NodeBus, AnEmptyNetworkFileIsTheShippedLocalhostNetwork) {
  const absl::StatusOr<robot::ipc::NetworkConfig> network = loadNodeNetwork("");
  ASSERT_TRUE(network.ok()) << network.status();
  EXPECT_EQ(*network, robot::ipc::localhostNetworkConfig());

  const absl::StatusOr<std::string> shippedFile = robot::resolveResourcePath("config/ipc/network.textproto");
  ASSERT_TRUE(shippedFile.ok()) << shippedFile.status();
  const absl::StatusOr<robot::ipc::NetworkConfig> shipped = loadNodeNetwork(*shippedFile);
  ASSERT_TRUE(shipped.ok()) << shipped.status();
  EXPECT_EQ(*shipped, *network);
  for (const char* absl_nonnull node : {"robot", "mpc", "operator", "teleop", "config_push"}) {
    EXPECT_NE(network->find(node), nullptr) << node;
  }
}

TEST(NodeBus, RefusesANodeTheNetworkDoesNotHave) {
  const absl::StatusOr<std::unique_ptr<robot::ipc::Bus>> bus = createNodeBus(/*networkConfigPath=*/"", "planner");
  EXPECT_EQ(bus.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(bus.status().message(), "planner")) << bus.status();
}

TEST(NodeBus, ReportsANetworkFileThatDoesNotExist) {
  const absl::StatusOr<robot::ipc::NetworkConfig> network = loadNodeNetwork("/nonexistent/network.textproto");
  EXPECT_FALSE(network.ok());
}

TEST(MpcFiles, NamesTheFlagOfAFileThatIsNotGivenOrDoesNotExist) {
  const absl::StatusOr<std::string> network = robot::resolveResourcePath("config/ipc/network.textproto");
  ASSERT_TRUE(network.ok()) << network.status();
  MpcFiles files{.taskFile = *network, .referenceFile = *network, .urdfFile = *network, .gaitFile = *network};
  EXPECT_TRUE(validateMpcFiles(files).ok());

  files.urdfFile.clear();
  const absl::Status missing = validateMpcFiles(files);
  EXPECT_EQ(missing.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(missing.message(), "--urdf_file")) << missing;

  files.urdfFile = "/nonexistent/robot.urdf";
  const absl::Status notFound = validateMpcFiles(files);
  EXPECT_EQ(notFound.code(), absl::StatusCode::kNotFound);
  EXPECT_TRUE(absl::StrContains(notFound.message(), "/nonexistent/robot.urdf")) << notFound;

  EXPECT_EQ(validateFileFlag("--gait_file", /*path=*/"").code(), absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace ocs2::humanoid::node
