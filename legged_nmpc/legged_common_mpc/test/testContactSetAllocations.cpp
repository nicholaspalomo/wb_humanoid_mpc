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

#include <cstddef>

#include "gtest/gtest.h"
#include "ocs2_core/Types.h"

#include "legged_common_mpc/common/ContactArray.h"
#include "legged_common_mpc/common/ContactMask.h"
#include "legged_common_mpc/common/Types.h"
#include "robot_runtime/robot_realtime/test/AllocationCounter.h"

/**
 * The per-tick operations of the contact set types on the realtime path
 * allocate nothing: a ContactArray of Eigen fixed-size entries and a
 * ContactMask, built before the counted region as a controller builds them at
 * construction.
 */
namespace ocs2::legged {
namespace {

constexpr size_t kNumContactsOfTheSet = 6;
constexpr size_t kTicks = 100;

TEST(ContactSetAllocations, PerTickOperationsAllocateNothing) {
  ContactArray<vector6_t> sixVectors(kNumContactsOfTheSet, vector6_t::Zero());
  ContactArray<vector6_t> sixVectorsCopy(kNumContactsOfTheSet);
  ContactArray<matrix3_t> matrices(kNumContactsOfTheSet, matrix3_t::Identity());
  ContactMask mask(kNumContactsOfTheSet);
  ContactMask modeMask;
  size_t numActive = 0;

  const size_t before = robot::realtime::heapAllocationCountOnThisThread();
  for (size_t tick = 0; tick < kTicks; ++tick) {
    sixVectors.fill(vector6_t::Zero());
    for (size_t i = 0; i < sixVectors.size(); ++i) {
      sixVectors[i](2) = static_cast<scalar_t>(tick + i);
      matrices[i] = matrix3_t::Identity() * static_cast<scalar_t>(i);
    }
    sixVectorsCopy = sixVectors;
    sixVectors.assign(kNumContactsOfTheSet, vector6_t::Ones());
    modeMask = ContactMask::fromModeNumber(tick, kNumContactsOfTheSet);
    mask.set(tick % kNumContactsOfTheSet, tick % 2 == 0);
    numActive += mask.numActive() + modeMask.numActive();
  }
  const size_t allocations =
      robot::realtime::heapAllocationCountOnThisThread() - before;

  EXPECT_EQ(allocations, 0U);
  EXPECT_EQ(sixVectorsCopy[1](2), static_cast<scalar_t>(kTicks - 1 + 1));
  EXPECT_EQ(sixVectors[0], vector6_t::Ones());
  EXPECT_EQ(matrices[2], matrix3_t::Identity() * 2.0);
  EXPECT_GT(numActive, 0U);
}

}  // namespace
}  // namespace ocs2::legged
