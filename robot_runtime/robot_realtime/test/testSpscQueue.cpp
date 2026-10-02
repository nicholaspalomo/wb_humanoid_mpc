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
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include <Eigen/Core>

#include "robot_realtime/SpscQueue.h"

namespace robot::realtime {
namespace {

// The indices sit on cache lines of their own, and the queue occupies whole lines.
static_assert(alignof(SpscQueue<int>) == kCacheLineSize);
static_assert(sizeof(SpscQueue<int>) % kCacheLineSize == 0);
static_assert(sizeof(SpscQueue<int>) >= 3 * kCacheLineSize);

TEST(SpscQueueTest, startsEmptyWithTheRequestedCapacity) {
  SpscQueue<int> queue(/*capacity=*/4);
  EXPECT_TRUE(queue.empty());
  EXPECT_EQ(queue.capacity(), 4u);
  EXPECT_EQ(queue.sizeApprox(), 0u);
  EXPECT_EQ(queue.droppedCount(), 0u);

  int value = 7;
  EXPECT_FALSE(queue.tryPop(value));
  EXPECT_EQ(value, 7) << "a failed pop leaves its output untouched";
}

TEST(SpscQueueTest, popsInPushOrder) {
  SpscQueue<int> queue(/*capacity=*/4);
  for (int value = 0; value < 4; ++value) {
    ASSERT_TRUE(queue.tryPush(value));
  }
  EXPECT_EQ(queue.sizeApprox(), 4u);
  for (int expected = 0; expected < 4; ++expected) {
    int value = -1;
    ASSERT_TRUE(queue.tryPop(value));
    EXPECT_EQ(value, expected);
  }
  EXPECT_TRUE(queue.empty());
}

TEST(SpscQueueTest, aFullQueueRejectsAndCountsEveryFurtherPush) {
  SpscQueue<int> queue(/*capacity=*/8);
  int accepted = 0;
  for (int value = 0; value < 20; ++value) {
    accepted += queue.tryPush(value) ? 1 : 0;
  }
  EXPECT_EQ(accepted, 8);
  EXPECT_EQ(queue.droppedCount(), 12u);
  EXPECT_EQ(queue.sizeApprox(), 8u);

  // The values that fit are intact and in order; the rejected ones never entered.
  for (int expected = 0; expected < 8; ++expected) {
    int value = -1;
    ASSERT_TRUE(queue.tryPop(value));
    EXPECT_EQ(value, expected);
  }
  int value = -1;
  EXPECT_FALSE(queue.tryPop(value));

  // Space freed by a pop is usable again, and the drop count only grows.
  EXPECT_TRUE(queue.tryPush(100));
  EXPECT_EQ(queue.droppedCount(), 12u);
}

TEST(SpscQueueTest, keepsOrderAcrossManyWrapArounds) {
  SpscQueue<int> queue(/*capacity=*/3);
  int nextPush = 0;
  int nextPop = 0;
  for (int round = 0; round < 1000; ++round) {
    // Alternate between filling the queue and leaving it partly full, so that the indices meet in every position.
    const int pushes = round % 2 == 0 ? 3 : 2;
    for (int i = 0; i < pushes; ++i) {
      ASSERT_TRUE(queue.tryPush(nextPush++));
    }
    for (int i = 0; i < pushes; ++i) {
      int value = -1;
      ASSERT_TRUE(queue.tryPop(value));
      ASSERT_EQ(value, nextPop++);
    }
  }
  EXPECT_TRUE(queue.empty());
  EXPECT_EQ(queue.droppedCount(), 0u);
}

TEST(SpscQueueTest, everySlotStartsAsACopyOfThePrototype) {
  const std::vector<double> prototype(16, 1.5);
  SpscQueue<std::vector<double>> queue(/*capacity=*/2, prototype);
  for (int i = 0; i < 2; ++i) {
    std::size_t slotSize = 0;
    double slotValue = 0.0;
    ASSERT_TRUE(queue.tryPushInPlace([&slotSize, &slotValue](std::vector<double>& slot) {
      slotSize = slot.size();
      slotValue = slot.front();
    }));
    EXPECT_EQ(slotSize, prototype.size());
    EXPECT_EQ(slotValue, 1.5);
  }
}

TEST(SpscQueueTest, inPlaceAccessCallsBackOnlyWhenThereIsASlot) {
  SpscQueue<std::array<int, 4>> queue(/*capacity=*/1);
  int writes = 0;
  int reads = 0;

  EXPECT_FALSE(queue.tryPopInPlace([&reads](const std::array<int, 4>&) { ++reads; }));
  EXPECT_TRUE(queue.tryPushInPlace([&writes](std::array<int, 4>& slot) {
    slot = {1, 2, 3, 4};
    ++writes;
  }));
  EXPECT_FALSE(queue.tryPushInPlace([&writes](std::array<int, 4>&) { ++writes; }));
  EXPECT_EQ(queue.droppedCount(), 1u);

  std::array<int, 4> seen{};
  EXPECT_TRUE(queue.tryPopInPlace([&reads, &seen](const std::array<int, 4>& slot) {
    seen = slot;
    ++reads;
  }));
  EXPECT_EQ(writes, 1);
  EXPECT_EQ(reads, 1);
  EXPECT_EQ(seen, (std::array<int, 4>{1, 2, 3, 4}));
}

TEST(SpscQueueTest, carriesEigenPayloads) {
  SpscQueue<Eigen::VectorXd> queue(/*capacity=*/2, Eigen::VectorXd::Zero(30));
  const Eigen::VectorXd pushed = Eigen::VectorXd::LinSpaced(30, 0.0, 29.0);
  ASSERT_TRUE(queue.tryPush(pushed));
  Eigen::VectorXd popped = Eigen::VectorXd::Zero(30);
  ASSERT_TRUE(queue.tryPop(popped));
  EXPECT_EQ(popped, pushed);
}

TEST(SpscQueueDeathTest, refusesAZeroCapacity) {
  EXPECT_DEATH(SpscQueue<int>(/*capacity=*/0), "at least one value");
}

// ---------------------------------------------------------------------------------------------------------------------
// Two threads
// ---------------------------------------------------------------------------------------------------------------------

// A payload of two cache lines whose words all repeat the sequence number, so that a slot read while it is being
// written (a missing acquire or release) shows up as words that disagree.
struct SequencedPayload {
  std::uint64_t sequence = 0;
  std::array<std::uint64_t, 15> copies{};

  void set(std::uint64_t value) {
    sequence = value;
    copies.fill(value);
  }
  bool consistent() const {
    for (const std::uint64_t copy : copies) {
      if (copy != sequence) {
        return false;
      }
    }
    return true;
  }
};

struct StressResult {
  std::uint64_t received = 0;
  std::uint64_t outOfOrder = 0;
  std::uint64_t torn = 0;
  std::uint64_t producerFailures = 0;
};

/**
 * Pushes `messages` sequence numbers from a producer thread and pops them on this one. With `retryWhenFull` the
 * producer retries a push until it succeeds, as a non-realtime producer may; without, it drops the value and moves on,
 * as the realtime thread does.
 */
StressResult runStress(SpscQueue<SequencedPayload>& queue, std::uint64_t messages, bool retryWhenFull) {
  StressResult result;
  std::atomic<bool> producerDone{false};
  std::thread producer([&queue, &result, &producerDone, messages, retryWhenFull]() {
    SequencedPayload payload;
    for (std::uint64_t sequence = 0; sequence < messages; ++sequence) {
      payload.set(sequence);
      while (!queue.tryPush(payload)) {
        ++result.producerFailures;
        if (!retryWhenFull) {
          break;
        }
        std::this_thread::yield();
      }
    }
    producerDone.store(/*desired=*/true, std::memory_order_release);
  });

  SequencedPayload popped;
  bool first = true;
  std::uint64_t previous = 0;
  for (;;) {
    if (queue.tryPop(popped)) {
      ++result.received;
      result.torn += popped.consistent() ? 0 : 1;
      // Retrying keeps every value, so each must follow the previous one; dropping may leave gaps, never reorder.
      const bool inOrder =
          first ? (!retryWhenFull || popped.sequence == 0) : (retryWhenFull ? popped.sequence == previous + 1 : popped.sequence > previous);
      result.outOfOrder += inOrder ? 0 : 1;
      previous = popped.sequence;
      first = false;
      continue;
    }
    if (producerDone.load(std::memory_order_acquire) && queue.empty()) {
      break;
    }
    std::this_thread::yield();
  }
  producer.join();
  return result;
}

TEST(SpscQueueStressTest, aProducerThatRetriesLosesNothingAndKeepsOrder) {
  constexpr std::uint64_t kMessages = 1'000'000;
  auto queue = std::make_unique<SpscQueue<SequencedPayload>>(/*capacity=*/256);
  const StressResult result = runStress(*queue, kMessages, /*retryWhenFull=*/true);

  EXPECT_EQ(result.received, kMessages);
  EXPECT_EQ(result.outOfOrder, 0u);
  EXPECT_EQ(result.torn, 0u);
  EXPECT_EQ(queue->droppedCount(), result.producerFailures) << "every rejected push is counted";
}

TEST(SpscQueueStressTest, aSingleSlotQueueStillHandsOverEveryValue) {
  constexpr std::uint64_t kMessages = 100'000;
  auto queue = std::make_unique<SpscQueue<SequencedPayload>>(/*capacity=*/1);
  const StressResult result = runStress(*queue, kMessages, /*retryWhenFull=*/true);

  EXPECT_EQ(result.received, kMessages);
  EXPECT_EQ(result.outOfOrder, 0u);
  EXPECT_EQ(result.torn, 0u);
  EXPECT_EQ(queue->droppedCount(), result.producerFailures);
}

TEST(SpscQueueStressTest, aProducerThatNeverWaitsLosesExactlyWhatItWasToldWasDropped) {
  constexpr std::uint64_t kMessages = 1'000'000;
  auto queue = std::make_unique<SpscQueue<SequencedPayload>>(/*capacity=*/16);
  const StressResult result = runStress(*queue, kMessages, /*retryWhenFull=*/false);

  EXPECT_EQ(result.received + queue->droppedCount(), kMessages);
  EXPECT_EQ(queue->droppedCount(), result.producerFailures);
  EXPECT_EQ(result.outOfOrder, 0u);
  EXPECT_EQ(result.torn, 0u);
}

// A queue that never holds more than one value - the producer pushes the next one only once the consumer has taken the
// last - watched by a third thread: sizeApprox() must never report a size the queue did not hold. Loading the write
// index before the read index, as it once did, reported a full queue whenever a push and a pop fell between the loads.
TEST(SpscQueueStressTest, sizeApproxNeverReportsASizeTheQueueDidNotHold) {
  constexpr std::uint64_t kMessages = 100'000;
  auto queue = std::make_unique<SpscQueue<std::uint64_t>>(/*capacity=*/64);
  std::atomic<std::uint64_t> consumed{0};
  std::thread producer([&queue, &consumed]() {
    for (std::uint64_t value = 0; value < kMessages; ++value) {
      while (consumed.load(std::memory_order_acquire) != value) {
        std::this_thread::yield();
      }
      queue->tryPush(value);
    }
  });
  std::thread consumer([&queue, &consumed]() {
    std::uint64_t value = 0;
    while (consumed.load(std::memory_order_relaxed) < kMessages) {
      if (queue->tryPop(value)) {
        // Only this thread writes the counter.
        consumed.store(consumed.load(std::memory_order_relaxed) + 1, std::memory_order_release);
      } else {
        std::this_thread::yield();
      }
    }
  });

  std::size_t largest = 0;
  std::uint64_t observations = 0;
  while (consumed.load(std::memory_order_acquire) < kMessages) {
    largest = std::max(largest, queue->sizeApprox());
    ++observations;
  }
  producer.join();
  consumer.join();
  EXPECT_LE(largest, 1u) << "over " << observations << " observations";
  EXPECT_GT(observations, 0u);
  EXPECT_EQ(queue->sizeApprox(), 0u);
  EXPECT_EQ(queue->droppedCount(), 0u);
}

}  // namespace
}  // namespace robot::realtime
