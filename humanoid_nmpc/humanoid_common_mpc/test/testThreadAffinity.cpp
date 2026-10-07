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
#include <thread>
#include <vector>

#include "gtest/gtest.h"

#include "humanoid_common_mpc/common/ThreadAffinity.h"

/*
 * The default core partitions of the robot and MPC binaries (`--realtime_cores=default`, `--backend_cores=default`)
 * name cores of the process's own CPU set, not of the machine: a container started with a cpuset used to get cores
 * outside it, which the realtime thread could not be pinned to, and it then floated over the cpuset unpinned.
 */

namespace ocs2::humanoid {
namespace {

bool within(const std::vector<int>& cores, const std::vector<int>& allowed) {
  return std::all_of(cores.begin(), cores.end(),
                     [&allowed](int core) { return std::find(allowed.begin(), allowed.end(), core) != allowed.end(); });
}

std::vector<int> range(int first, int count) {
  std::vector<int> cores;
  for (int core = first; core < first + count; ++core) cores.push_back(core);
  return cores;
}

TEST(ThreadAffinity, ThePartitionsOfAWholeMachineAreTheShippedOnes) {
  const SystemCoreAllocation twenty = coreAllocationFor(range(/*first=*/0, /*count=*/20));
  EXPECT_EQ(twenty.simCores, range(/*first=*/0, /*count=*/4));
  EXPECT_EQ(twenty.mrtCores, range(/*first=*/4, /*count=*/2));
  EXPECT_EQ(twenty.mpcCores, range(/*first=*/6, /*count=*/10));
  const SystemCoreAllocation eight = coreAllocationFor(range(/*first=*/0, /*count=*/8));
  EXPECT_EQ(eight.simCores, range(/*first=*/0, /*count=*/2));
  EXPECT_EQ(eight.mrtCores, range(/*first=*/2, /*count=*/1));
  EXPECT_EQ(eight.mpcCores, range(/*first=*/3, /*count=*/5));
}

TEST(ThreadAffinity, ThePartitionsOfACpusetNameItsCoresOnly) {
  // deploy_robot.sh --cpuset 12-15 on a 16-core robot.
  const std::vector<int> cpuset = range(/*first=*/12, /*count=*/4);
  const SystemCoreAllocation allocation = coreAllocationFor(cpuset);
  EXPECT_EQ(allocation.simCores, std::vector<int>{12});
  EXPECT_EQ(allocation.mrtCores, std::vector<int>{13});
  EXPECT_EQ(allocation.mpcCores, (std::vector<int>{14, 15}));
  // A sparse set is taken in order.
  const std::vector<int> sparse{1, 3, 5, 7, 9, 11, 13, 15};
  const SystemCoreAllocation sparseAllocation = coreAllocationFor(sparse);
  EXPECT_EQ(sparseAllocation.mrtCores, std::vector<int>{5});
  for (const std::vector<int>& partition : {sparseAllocation.simCores, sparseAllocation.mrtCores, sparseAllocation.mpcCores}) {
    EXPECT_TRUE(within(partition, sparse));
  }
  // Under four cores nothing is restricted, within the set.
  const SystemCoreAllocation two = coreAllocationFor({6, 7});
  EXPECT_EQ(two.mrtCores, (std::vector<int>{6, 7}));
}

TEST(ThreadAffinity, TheDefaultFollowsTheCallersAffinity) {
  const std::vector<int> all = availableCores();
  ASSERT_FALSE(all.empty());
  const SystemCoreAllocation unrestricted = getDefaultCoreAllocation();
  for (const std::vector<int>& partition : {unrestricted.simCores, unrestricted.mrtCores, unrestricted.mpcCores}) {
    EXPECT_TRUE(within(partition, all));
  }

  // A thread restricted to the last available core, as a container's cpuset restricts the process.
  const int last = all.back();
  std::vector<int> seen;
  SystemCoreAllocation restricted;
  std::thread thread([&]() {
    cpu_set_t only;
    CPU_ZERO(&only);
    CPU_SET(last, &only);
    ASSERT_EQ(pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &only), 0);
    seen = availableCores();
    restricted = getDefaultCoreAllocation();
  });
  thread.join();
  EXPECT_EQ(seen, std::vector<int>{last});
  EXPECT_EQ(restricted.mrtCores, std::vector<int>{last});
  EXPECT_EQ(restricted.simCores, std::vector<int>{last});
}

}  // namespace
}  // namespace ocs2::humanoid
