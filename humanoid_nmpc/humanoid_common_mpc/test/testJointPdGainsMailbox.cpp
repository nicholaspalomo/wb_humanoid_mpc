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

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <string>
#include <thread>
#include <vector>

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"

#include "humanoid_common_mpc/mrt/JointPdGains.h"
#include "humanoid_common_mpc/mrt/JointPdGainsMailbox.h"
#include "humanoid_state_estimation/humanoid_state_estimator/test/AllocationCounter.h"

/*
 * The hand-over of joint PD gains to the realtime control thread. The MRT joint controllers used to take a mutex the
 * ROS callback held, write the YAML to <gains file>.live.yaml and parse it with YAML::LoadFile inside
 * computeJointControlAction(); the control thread now only receives a preallocated set of gains. This binary links the
 * allocation counter, which interposes malloc for the whole process.
 */

namespace ocs2::humanoid {

/** Reaches the producers' mutex, to hold it as a producer that is slow to post would. */
class JointPdGainsMailboxTestPeer {
 public:
  static absl::Mutex& producerMutex(JointPdGainsMailbox& mailbox) { return mailbox.producerMutex_; }
};

namespace {

using ::ocs2::humanoid::estimation::heapAllocationCount;

const std::vector<std::string> kMpcJoints{"a", "b", "c"};
const std::vector<std::string> kOtherJoints{"d", "e"};

JointPdGains gainsWithKp(scalar_t kp) {
  JointPdGainsDefaults defaults;
  defaults.kp = kp;
  defaults.kd = kp / 10.0;
  defaults.torqueLimit = 100.0;
  return defaultJointPdGains(defaults, kMpcJoints, kOtherJoints);
}

TEST(JointPdGainsMailbox, GainsPostedOnAnotherThreadAreReceivedOnce) {
  JointPdGainsMailbox mailbox(gainsWithKp(/*kp=*/1.0));
  JointPdGains active = gainsWithKp(/*kp=*/1.0);
  EXPECT_FALSE(mailbox.receive(active)) << "nothing was posted";

  std::thread producer([&mailbox]() { EXPECT_TRUE(mailbox.post(gainsWithKp(/*kp=*/2.0)).ok()); });
  producer.join();
  ASSERT_TRUE(mailbox.receive(active));
  EXPECT_EQ(active.mpcJointKp, gainsWithKp(/*kp=*/2.0).mpcJointKp);
  EXPECT_EQ(active.otherJointKd, gainsWithKp(/*kp=*/2.0).otherJointKd);
  EXPECT_EQ(active.defaults.kp, 2.0);

  active.mpcJointKp[0] = -7.0;  // the control thread's own copy
  EXPECT_FALSE(mailbox.receive(active)) << "the same gains were received twice";
  EXPECT_EQ(active.mpcJointKp[0], -7.0) << "receive() without new gains changed the active ones";
}

TEST(JointPdGainsMailbox, TheNewestPostWins) {
  JointPdGainsMailbox mailbox(gainsWithKp(/*kp=*/1.0));
  JointPdGains active = gainsWithKp(/*kp=*/1.0);
  std::thread producer([&mailbox]() {
    for (int k = 2; k <= 50; ++k) EXPECT_TRUE(mailbox.post(gainsWithKp(static_cast<scalar_t>(k))).ok());
  });
  // Receives while the producer posts see increasing gains, never a torn set. Bounded, so that a lost final post fails
  // here, naming it, instead of spinning until the test times out.
  const std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  scalar_t last = 1.0;
  while (last < 50.0 && std::chrono::steady_clock::now() < deadline) {
    if (mailbox.receive(active)) {
      EXPECT_GE(active.defaults.kp, last);
      for (Eigen::Index i = 0; i < active.mpcJointKp.size(); ++i) {
        EXPECT_EQ(active.mpcJointKp[i], active.defaults.kp) << "a set mixed from two posts";
      }
      for (Eigen::Index i = 0; i < active.otherJointKd.size(); ++i) {
        EXPECT_EQ(active.otherJointKd[i], active.defaults.kd * kOtherJointDefaultGainScale) << "a set mixed from two posts";
      }
      last = active.defaults.kp;
    }
  }
  producer.join();
  ASSERT_EQ(last, 50.0) << "the last post never reached the control thread";
  EXPECT_FALSE(mailbox.receive(active));
  EXPECT_EQ(active.defaults.kp, 50.0);
}

TEST(JointPdGainsMailbox, TheDocumentHandedInLastWinsEvenWhenItIsParsedFirst) {
  JointPdGainsMailbox mailbox(gainsWithKp(/*kp=*/1.0));
  JointPdGains active = gainsWithKp(/*kp=*/1.0);
  // Two producers take their tickets as their documents arrive: the file watcher first, then the GUI. The GUI's
  // document parses faster and is posted first; the file's, handed in earlier, must not overwrite it.
  const uint64_t fileTicket = mailbox.takeTicket();
  const uint64_t guiTicket = mailbox.takeTicket();
  ASSERT_LT(fileTicket, guiTicket);
  ASSERT_TRUE(mailbox.post(gainsWithKp(/*kp=*/3.0), guiTicket).ok());
  EXPECT_TRUE(mailbox.post(gainsWithKp(/*kp=*/2.0), fileTicket).ok()) << "a superseded document is not an error";
  ASSERT_TRUE(mailbox.receive(active));
  EXPECT_EQ(active.defaults.kp, 3.0) << "a document handed in earlier overwrote a newer one";
  EXPECT_FALSE(mailbox.receive(active)) << "the superseded document was posted after all";

  // A newer document that is refused, and so never posted, leaves an older one free to post.
  const uint64_t olderTicket = mailbox.takeTicket();
  const uint64_t refusedTicket = mailbox.takeTicket();
  ASSERT_LT(olderTicket, refusedTicket);
  ASSERT_TRUE(mailbox.post(gainsWithKp(/*kp=*/4.0), olderTicket).ok());
  ASSERT_TRUE(mailbox.receive(active));
  EXPECT_EQ(active.defaults.kp, 4.0);
}

TEST(JointPdGainsMailbox, AConcurrentlyTakenTicketIsNeverReused) {
  JointPdGainsMailbox mailbox(gainsWithKp(/*kp=*/1.0));
  constexpr int kTicketsPerThread = 1000;
  std::vector<uint64_t> first(kTicketsPerThread);
  std::vector<uint64_t> second(kTicketsPerThread);
  std::thread a([&]() {
    for (int k = 0; k < kTicketsPerThread; ++k) first[k] = mailbox.takeTicket();
  });
  std::thread b([&]() {
    for (int k = 0; k < kTicketsPerThread; ++k) second[k] = mailbox.takeTicket();
  });
  a.join();
  b.join();
  std::vector<uint64_t> all(first);
  all.insert(all.end(), second.begin(), second.end());
  std::sort(all.begin(), all.end());
  EXPECT_EQ(std::adjacent_find(all.begin(), all.end()), all.end()) << "two documents were given the same place in line";
  EXPECT_GT(all.front(), 0u) << "ticket 0 would never be newer than the initial gains";
}

TEST(JointPdGainsMailbox, ReceivingMakesNoHeapAllocation) {
  JointPdGainsMailbox mailbox(gainsWithKp(/*kp=*/1.0));
  JointPdGains active = gainsWithKp(/*kp=*/1.0);
  std::thread producer([&mailbox]() { EXPECT_TRUE(mailbox.post(gainsWithKp(/*kp=*/3.0)).ok()); });
  producer.join();

  // Nothing else runs in this process now: every allocation counted is the control thread's.
  const std::size_t before = heapAllocationCount();
  const bool received = mailbox.receive(active);
  const bool receivedAgain = mailbox.receive(active);
  const std::size_t after = heapAllocationCount();
  EXPECT_TRUE(received);
  EXPECT_FALSE(receivedAgain);
  EXPECT_EQ(after - before, 0u) << "receive() allocated on the control thread";
  EXPECT_EQ(active.mpcJointKp[0], 3.0);

  // Positive control: the counter sees an allocation of this thread.
  const std::size_t beforeCopy = heapAllocationCount();
  const JointPdGains copy = active;
  EXPECT_GT(heapAllocationCount(), beforeCopy) << "the allocation counter is not linked into this binary";
  EXPECT_EQ(copy.mpcJointKp[0], 3.0);
}

TEST(JointPdGainsMailbox, ReceivingDoesNotWaitForAProducerHoldingTheLock) {
  JointPdGainsMailbox mailbox(gainsWithKp(/*kp=*/1.0));
  ASSERT_TRUE(mailbox.post(gainsWithKp(/*kp=*/4.0)).ok());
  JointPdGains active = gainsWithKp(/*kp=*/1.0);

  // A producer in the middle of a post holds the mutex; the control thread receives what was posted before, at once.
  absl::Mutex& producerMutex = JointPdGainsMailboxTestPeer::producerMutex(mailbox);
  producerMutex.Lock();
  std::future<bool> received = std::async(std::launch::async, [&mailbox, &active]() { return mailbox.receive(active); });
  const std::future_status status = received.wait_for(std::chrono::seconds(2));
  producerMutex.Unlock();
  ASSERT_EQ(status, std::future_status::ready) << "receive() waited for the producers' lock";
  EXPECT_TRUE(received.get());
  EXPECT_EQ(active.mpcJointKp[0], 4.0);
}

TEST(JointPdGainsMailbox, GainsOfOtherDimensionsAreRefusedAndNothingIsPosted) {
  JointPdGainsMailbox mailbox(gainsWithKp(/*kp=*/1.0));
  JointPdGains wrong = gainsWithKp(/*kp=*/5.0);
  wrong.otherJointKp.resize(7);
  EXPECT_EQ(mailbox.post(wrong).code(), absl::StatusCode::kInvalidArgument);
  JointPdGains active = gainsWithKp(/*kp=*/1.0);
  EXPECT_FALSE(mailbox.receive(active)) << "refused gains reached the control thread";
}

}  // namespace
}  // namespace ocs2::humanoid
