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

#include <atomic>
#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "gtest/gtest.h"

#include "web_viewer/EncodedFrame.h"
#include "web_viewer/ViewerFrameMailbox.h"
#include "web_viewer/ViewerStatus.h"

namespace robot::web_viewer {
namespace {

EncodedFrame frameWith(const std::string& jpeg, double simTime) {
  return EncodedFrame{.sequence = 0, .width = 4, .height = 3, .simTime = simTime, .publishTime = absl::UnixEpoch(), .jpeg = jpeg};
}

TEST(ViewerFrameMailboxTest, PublishStampsSequenceAndTime) {
  ViewerFrameMailbox mailbox(/*maxClients=*/2);
  EXPECT_FALSE(mailbox.latestFrameMetadata().has_value());
  const absl::Time before = absl::Now();
  mailbox.publish(frameWith("a", /*simTime=*/0.5));
  mailbox.publish(frameWith("bb", /*simTime=*/1.0));
  const std::optional<EncodedFrame> latest = mailbox.copyNewer(/*afterSequence=*/0, absl::ZeroDuration());
  if (!latest.has_value()) FAIL() << "no frame";
  EXPECT_EQ(latest->sequence, 2);
  EXPECT_EQ(latest->jpeg, "bb");
  EXPECT_DOUBLE_EQ(latest->simTime, 1.0);
  EXPECT_GE(latest->publishTime, before);
  const std::optional<EncodedFrame> metadata = mailbox.latestFrameMetadata();
  if (!metadata.has_value()) FAIL() << "no metadata";
  EXPECT_EQ(metadata->sequence, 2);
  EXPECT_EQ(metadata->width, 4);
  EXPECT_TRUE(metadata->jpeg.empty());
}

TEST(ViewerFrameMailboxTest, CopyNewerWaitsForTheNextFrameOrTimesOut) {
  ViewerFrameMailbox mailbox(/*maxClients=*/1);
  mailbox.publish(frameWith("first", /*simTime=*/0.0));
  const absl::Time start = absl::Now();
  EXPECT_FALSE(mailbox.copyNewer(/*afterSequence=*/1, absl::Milliseconds(100)).has_value());
  EXPECT_GE(absl::Now() - start, absl::Milliseconds(90));

  std::thread publisher([&mailbox] {
    absl::SleepFor(absl::Milliseconds(50));
    mailbox.publish(frameWith("second", /*simTime=*/0.1));
  });
  const std::optional<EncodedFrame> next = mailbox.copyNewer(/*afterSequence=*/1, absl::Seconds(5));
  publisher.join();
  if (!next.has_value()) FAIL() << "no next frame";
  EXPECT_EQ(next->sequence, 2);
  EXPECT_EQ(next->jpeg, "second");
}

TEST(ViewerFrameMailboxTest, CloseWakesWaitersForGood) {
  ViewerFrameMailbox mailbox(/*maxClients=*/1);
  std::thread closer([&mailbox] {
    absl::SleepFor(absl::Milliseconds(50));
    mailbox.close();
  });
  const absl::Time start = absl::Now();
  EXPECT_FALSE(mailbox.copyNewer(/*afterSequence=*/0, absl::Seconds(10)).has_value());
  EXPECT_LT(absl::Now() - start, absl::Seconds(2));
  closer.join();
  EXPECT_TRUE(mailbox.closed());
  mailbox.publish(frameWith("late", /*simTime=*/0.0));
  EXPECT_FALSE(mailbox.copyNewer(/*afterSequence=*/0, absl::Seconds(1)).has_value());
  EXPECT_FALSE(mailbox.waitForDemand(absl::Seconds(1)));
}

TEST(ViewerFrameMailboxTest, AdmitsAtMostMaxClients) {
  ViewerFrameMailbox mailbox(/*maxClients=*/2);
  EXPECT_TRUE(mailbox.tryAcquireClient());
  EXPECT_TRUE(mailbox.tryAcquireClient());
  EXPECT_FALSE(mailbox.tryAcquireClient());
  EXPECT_EQ(mailbox.activeClients(), 2);
  mailbox.releaseClient();
  EXPECT_EQ(mailbox.activeClients(), 1);
  EXPECT_TRUE(mailbox.tryAcquireClient());
  mailbox.releaseClient();
  mailbox.releaseClient();
  EXPECT_EQ(mailbox.activeClients(), 0);
  mailbox.releaseClient();  // logged, and the count stays 0
  EXPECT_EQ(mailbox.activeClients(), 0);
}

TEST(ViewerFrameMailboxTest, DemandFollowsClientsAndTheLease) {
  ViewerFrameMailbox mailbox(/*maxClients=*/1);
  const absl::Time start = absl::Now();
  EXPECT_FALSE(mailbox.waitForDemand(absl::Milliseconds(150)));
  EXPECT_GE(absl::Now() - start, absl::Milliseconds(140));

  mailbox.noteFrameRequest();
  EXPECT_TRUE(mailbox.waitForDemand(absl::ZeroDuration()));
  absl::SleepFor(ViewerFrameMailbox::kDemandLease + absl::Milliseconds(100));
  EXPECT_FALSE(mailbox.waitForDemand(absl::ZeroDuration()));

  ASSERT_TRUE(mailbox.tryAcquireClient());
  absl::SleepFor(ViewerFrameMailbox::kDemandLease + absl::Milliseconds(100));
  EXPECT_TRUE(mailbox.waitForDemand(absl::ZeroDuration()));
  mailbox.releaseClient();
}

TEST(ViewerFrameMailboxTest, AFrameRequestWakesTheRenderer) {
  ViewerFrameMailbox mailbox(/*maxClients=*/1);
  std::thread page([&mailbox] {
    absl::SleepFor(absl::Milliseconds(50));
    mailbox.noteFrameRequest();
  });
  const absl::Time start = absl::Now();
  EXPECT_TRUE(mailbox.waitForDemand(absl::Seconds(5)));
  EXPECT_LT(absl::Now() - start, absl::Seconds(2));
  page.join();

  ViewerFrameMailbox admitted(/*maxClients=*/1);
  std::thread stream([&admitted] {
    absl::SleepFor(absl::Milliseconds(50));
    admitted.tryAcquireClient();
  });
  EXPECT_TRUE(admitted.waitForDemand(absl::Seconds(5)));
  stream.join();
}

TEST(ViewerFrameMailboxTest, ViewersAreThePageIdsSeenWithinTheLease) {
  ViewerFrameMailbox mailbox(/*maxClients=*/1);
  EXPECT_EQ(mailbox.viewerCount(), 0);
  mailbox.noteViewer("a");
  mailbox.noteViewer("a");
  mailbox.noteViewer("b");
  EXPECT_EQ(mailbox.viewerCount(), 2);
  absl::SleepFor(ViewerFrameMailbox::kDemandLease + absl::Milliseconds(100));
  EXPECT_EQ(mailbox.viewerCount(), 0);
  // Ids beyond the cap replace the oldest.
  for (int viewer = 0; viewer < ViewerFrameMailbox::kMaxTrackedViewers + 5; ++viewer) mailbox.noteViewer(absl::StrCat(viewer));
  EXPECT_EQ(mailbox.viewerCount(), ViewerFrameMailbox::kMaxTrackedViewers);
}

// The renderer resumes when an admission finds no demand before it: no client admitted and the lease lapsed.
TEST(ViewerFrameMailboxTest, DemandResumesOnlyAfterALapse) {
  ViewerFrameMailbox mailbox(/*maxClients=*/2);
  EXPECT_FALSE(mailbox.demandResumedAt().has_value());
  ASSERT_TRUE(mailbox.tryAcquireClient());
  mailbox.noteFrameRequest();
  const std::optional<std::chrono::steady_clock::time_point> first = mailbox.demandResumedAt();
  ASSERT_TRUE(first.has_value());
  // A second client while the first is admitted, and a request within the lease, find the renderer busy.
  ASSERT_TRUE(mailbox.tryAcquireClient());
  mailbox.releaseClient();
  mailbox.releaseClient();
  absl::SleepFor(absl::Milliseconds(100));
  ASSERT_TRUE(mailbox.tryAcquireClient());
  mailbox.noteFrameRequest();
  mailbox.releaseClient();
  EXPECT_EQ(mailbox.demandResumedAt(), first);
  absl::SleepFor(ViewerFrameMailbox::kDemandLease + absl::Milliseconds(100));
  ASSERT_TRUE(mailbox.tryAcquireClient());
  mailbox.releaseClient();
  const std::optional<std::chrono::steady_clock::time_point> second = mailbox.demandResumedAt();
  ASSERT_TRUE(second.has_value());
  // value_or: clang-tidy's optional-access analysis does not follow the ASSERTs above; the fallbacks fail the check.
  EXPECT_GT(second.value_or(std::chrono::steady_clock::time_point::min()), first.value_or(std::chrono::steady_clock::time_point::max()));
}

TEST(ViewerFrameMailboxTest, ReadersAlwaysCopyWholeFrames) {
  ViewerFrameMailbox mailbox(/*maxClients=*/4);
  std::atomic<bool> done = false;
  std::atomic<int> torn = 0;
  std::vector<std::thread> readers;
  for (int reader = 0; reader < 4; ++reader) {
    readers.emplace_back([&mailbox, &done, &torn] {
      int64_t last = 0;
      while (!done.load()) {
        const std::optional<EncodedFrame> frame = mailbox.copyNewer(last, absl::Milliseconds(10));
        if (!frame.has_value()) continue;
        // Each frame's bytes repeat the character of its sequence, sized by it: a torn copy would mix two.
        const char expected = static_cast<char>('a' + frame->sequence % 26);
        if (frame->jpeg != std::string(static_cast<size_t>(1000 + frame->sequence % 7), expected)) torn.fetch_add(1);
        last = frame->sequence;
      }
    });
  }
  for (int64_t sequence = 1; sequence <= 2000; ++sequence) {
    mailbox.publish(frameWith(std::string(static_cast<size_t>(1000 + sequence % 7), static_cast<char>('a' + sequence % 26)),
                              /*simTime=*/0.0));
  }
  done.store(true);
  for (std::thread& reader : readers) reader.join();
  EXPECT_EQ(torn.load(), 0);
}

TEST(ViewerFrameMailboxTest, KeepsTheNewestStatus) {
  ViewerFrameMailbox mailbox(/*maxClients=*/1);
  EXPECT_EQ(mailbox.status().render.width, 0);
  ViewerStatus status;
  status.render.width = 640;
  status.visualizations.push_back({.name = "zmp", .hotkey = 'z', .enabled = true, .description = "the ZMP"});
  mailbox.publishStatus(status);
  EXPECT_EQ(mailbox.status().render.width, 640);
  ASSERT_EQ(mailbox.status().visualizations.size(), 1U);
  EXPECT_EQ(mailbox.status().visualizations[0].name, "zmp");
}

}  // namespace
}  // namespace robot::web_viewer
