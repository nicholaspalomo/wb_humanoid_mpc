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
#include <cstddef>

#include "absl/base/nullability.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"

#include "humanoid_mpc_ipc/Topics.h"
#include "robot_ipc/NetworkConfig.h"

namespace ocs2::humanoid::ipc::topics {
namespace {

// The prefixes of the four kinds of publisher
// (humanoid_nmpc/docs/distributed_runtime/README.md, "Topics").
constexpr std::array<absl::string_view, 4> kPublisherPrefixes = {
    "robot/", "mpc/", "viz/", "operator/"};

bool isLowerSnakeCase(absl::string_view name) {
  if (name.empty()) {
    return false;
  }
  for (const char c : name) {
    if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_')) {
      return false;
    }
  }
  return true;
}

TEST(TopicsTest, EveryTopicIsAPublisherPrefixFollowedByASnakeCaseName) {
  for (const absl::string_view topic : kAllTopics) {
    bool matched = false;
    for (const absl::string_view prefix : kPublisherPrefixes) {
      if (absl::StartsWith(topic, prefix)) {
        matched = isLowerSnakeCase(topic.substr(prefix.size()));
      }
    }
    EXPECT_TRUE(matched) << topic;
  }
}

TEST(TopicsTest,
     EveryTopicIsADdsTopicNameListedOnceWithItsQosInTheShippedNetwork) {
  // Every topic is a DDS topic of the same name, whose QoS profile and
  // publishers the network file names: a topic it did not list would fall back
  // to reliable_events and have no writer before its first sample.
  const robot::ipc::NetworkConfig network =
      robot::ipc::localhostNetworkConfig();
  EXPECT_EQ(network.topics.size(), kAllTopics.size())
      << "config/ipc/network.textproto lists exactly the topics of Topics.h";
  for (const absl::string_view topic : kAllTopics) {
    EXPECT_TRUE(robot::ipc::validateTopicName(topic).ok()) << topic;
    const robot::ipc::TopicQos* absl_nullable entry = network.findTopic(topic);
    ASSERT_NE(entry, nullptr)
        << topic << " is not in config/ipc/network.textproto";
    EXPECT_FALSE(entry->publishers.empty()) << topic;
  }
}

TEST(TopicsTest, TheMpcLinkTopicsAreListed) {
  // The two streams the MPC link cannot work without.
  EXPECT_EQ(kRobotMpcObservation, "robot/mpc_observation");
  EXPECT_EQ(kMpcPolicy, "mpc/policy");
  int listed = 0;
  for (const absl::string_view topic : kAllTopics) {
    listed += (topic == kRobotMpcObservation || topic == kMpcPolicy) ? 1 : 0;
  }
  EXPECT_EQ(listed, 2);
}

TEST(TopicsTest, TheSaveAndItsStatusAreListedOnceEach) {
  // The two-copy Save: the GUI's file on the operator's topic, the robot's
  // answer on its own.
  EXPECT_EQ(kOperatorConfigSave, "operator/config_save");
  EXPECT_EQ(kRobotConfigSaveStatus, "robot/config_save_status");
  int listed = 0;
  for (const absl::string_view topic : kAllTopics) {
    listed += (topic == kOperatorConfigSave || topic == kRobotConfigSaveStatus)
                  ? 1
                  : 0;
  }
  EXPECT_EQ(listed, 2);
}

}  // namespace
}  // namespace ocs2::humanoid::ipc::topics
