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

#include "humanoid_mpc_ipc/Topics.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>

#include "absl/strings/match.h"
#include "absl/strings/string_view.h"

namespace ocs2::humanoid::ipc::topics {
namespace {

// The prefixes of the four kinds of publisher (humanoid_nmpc/docs/distributed_runtime/README.md, "Topics").
constexpr std::array<absl::string_view, 4> kPublisherPrefixes = {"robot/", "mpc/", "viz/", "operator/"};

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

TEST(TopicsTest, NoTopicIsAPrefixOfAnother) {
  // ZeroMQ's SUB filter matches a prefix of the topic frame: a topic that began another would also receive it.
  for (size_t i = 0; i < kAllTopics.size(); ++i) {
    for (size_t j = 0; j < kAllTopics.size(); ++j) {
      if (i != j) {
        EXPECT_FALSE(absl::StartsWith(kAllTopics[j], kAllTopics[i])) << kAllTopics[i] << " begins " << kAllTopics[j];
      }
    }
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

}  // namespace
}  // namespace ocs2::humanoid::ipc::topics
