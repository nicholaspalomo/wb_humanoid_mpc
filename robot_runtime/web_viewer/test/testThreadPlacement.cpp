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

#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <string>
#include <thread>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"

#include "web_viewer/ThreadPlacement.h"
#include "web_viewer/ViewerThreadPriority.h"

namespace robot::web_viewer {
namespace {

// The CPUs the calling thread may run on.
std::vector<int> allowedCores() {
  cpu_set_t set;
  CPU_ZERO(&set);
  std::vector<int> cores;
  if (pthread_getaffinity_np(pthread_self(), sizeof(set), &set) != 0)
    return cores;
  for (int core = 0; core < CPU_SETSIZE; ++core) {
    if (CPU_ISSET(core, &set)) cores.push_back(core);
  }
  return cores;
}

// What a thread reports about itself after placing itself: its scheduling
// policies are sched_getscheduler()'s.
struct Placement {
  absl::Status status;
  std::vector<int> cores;
  int policyBefore = -1;
  int policyAfter = -1;
  // The same, read by a thread the placed thread starts afterwards.
  std::vector<int> childCores;
  int childPolicy = -1;
};

Placement placeNewThread(const std::vector<int>& cores,
                         ViewerThreadPriority priority) {
  Placement placement;
  std::thread worker([&placement, &cores, priority] {
    placement.policyBefore = sched_getscheduler(/*pid=*/0);
    placement.status = placeCurrentThread(cores, priority);
    placement.cores = allowedCores();
    placement.policyAfter = sched_getscheduler(/*pid=*/0);
    std::thread child([&placement] {
      placement.childCores = allowedCores();
      placement.childPolicy = sched_getscheduler(/*pid=*/0);
    });
    child.join();
  });
  worker.join();
  return placement;
}

TEST(ThreadPlacementTest, PinsAndIdlesTheThreadAndItsChildren) {
  const std::vector<int> allowed = allowedCores();
  ASSERT_FALSE(allowed.empty());
  const std::vector<int> oneCore = {allowed.back()};
  const Placement placement =
      placeNewThread(oneCore, ViewerThreadPriority::kIdle);
  ASSERT_TRUE(placement.status.ok()) << placement.status;
  EXPECT_EQ(placement.policyBefore, SCHED_OTHER);
  EXPECT_EQ(placement.cores, oneCore);
  EXPECT_EQ(placement.policyAfter, SCHED_IDLE);
  EXPECT_EQ(placement.childCores, oneCore);
  EXPECT_EQ(placement.childPolicy, SCHED_IDLE);
  // The calling thread is untouched.
  EXPECT_EQ(allowedCores(), allowed);
  EXPECT_EQ(sched_getscheduler(/*pid=*/0), SCHED_OTHER);
}

TEST(ThreadPlacementTest,
     AnEmptyCoreListKeepsTheAffinityAndInheritedKeepsTheScheduling) {
  const std::vector<int> allowed = allowedCores();
  const Placement placement =
      placeNewThread(/*cores=*/{}, ViewerThreadPriority::kInherited);
  ASSERT_TRUE(placement.status.ok()) << placement.status;
  EXPECT_EQ(placement.cores, allowed);
  EXPECT_EQ(placement.policyAfter, placement.policyBefore);
  EXPECT_EQ(placement.childPolicy, placement.policyBefore);
}

TEST(ThreadPlacementTest, RefusesBadArgumentsAndAppliesNothing) {
  const Placement badCore = placeNewThread({-1}, ViewerThreadPriority::kIdle);
  EXPECT_EQ(badCore.status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(badCore.policyAfter, badCore.policyBefore);
  const Placement tooLarge =
      placeNewThread({CPU_SETSIZE}, ViewerThreadPriority::kInherited);
  EXPECT_EQ(tooLarge.status.code(), absl::StatusCode::kInvalidArgument);
  const std::vector<int> oneCore = {allowedCores().back()};
  const Placement badPriority =
      placeNewThread(oneCore, static_cast<ViewerThreadPriority>(7));
  EXPECT_EQ(badPriority.status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(badPriority.cores, allowedCores())
      << "a refused placement pins nothing";
  EXPECT_EQ(badPriority.policyAfter, badPriority.policyBefore);
}

TEST(ThreadPlacementTest,
     ACoreOutsideTheAllowedSetFailsButTheSchedulingApplies) {
  const std::vector<int> allowed = allowedCores();
  int outside = -1;
  for (int core = 0; core < CPU_SETSIZE; ++core) {
    if (std::find(allowed.begin(), allowed.end(), core) == allowed.end()) {
      outside = core;
      break;
    }
  }
  ASSERT_GE(outside, 0);
  const Placement placement =
      placeNewThread({outside}, ViewerThreadPriority::kIdle);
  EXPECT_FALSE(placement.status.ok());
  EXPECT_NE(placement.status.message().find("pthread_setaffinity_np"),
            std::string::npos)
      << placement.status;
  EXPECT_EQ(placement.cores, allowed);
  EXPECT_EQ(placement.policyAfter, SCHED_IDLE);
}

}  // namespace
}  // namespace robot::web_viewer
