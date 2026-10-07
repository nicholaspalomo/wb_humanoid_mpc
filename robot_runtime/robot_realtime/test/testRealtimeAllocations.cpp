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

// The realtime property of this package: once constructed, the SPSC queue, the periodic timer and the loop timing
// statistics make no heap allocation. The allocation counter interposes malloc for this whole binary, which is why
// these tests have a target of their own.

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include "Eigen/Core"
#include "gtest/gtest.h"

#include "robot_core/TripleBuffer.h"
#include "robot_realtime/LoopTimingSnapshot.h"
#include "robot_realtime/LoopTimingStats.h"
#include "robot_realtime/PeriodicTimer.h"
#include "robot_realtime/SpscQueue.h"
#include "robot_runtime/robot_realtime/test/AllocationCounter.h"

namespace robot::realtime {
namespace {

using std::chrono::microseconds;
using std::chrono::milliseconds;
using std::chrono::nanoseconds;

// A telemetry sample as the realtime thread would hand it to the communication thread: storage on the heap, of a size
// fixed when the queue is built.
struct TelemetrySample {
  double time = 0.0;
  Eigen::VectorXd jointPositions;
  std::vector<double> jointTorques;
};

TelemetrySample makePrototype() {
  TelemetrySample prototype;
  prototype.jointPositions = Eigen::VectorXd::Zero(30);
  prototype.jointTorques.assign(30, 0.0);
  return prototype;
}

// Without this, a counter that saw nothing would make every test below pass.
TEST(RealtimeAllocationTest, theCounterSeesTheQueueAllocateItsSlots) {
  const TelemetrySample prototype = makePrototype();
  const size_t before = heapAllocationCount();
  auto queue = std::make_unique<SpscQueue<TelemetrySample>>(/*capacity=*/8, prototype);
  const size_t allocations = heapAllocationCount() - before;
  ASSERT_NE(queue, nullptr);
  // The over-aligned queue itself, its slot array, and two buffers per slot.
  EXPECT_GE(allocations, 2u + 2u * 9u);
}

// The per-thread count sees this thread's allocations and none of another thread's.
TEST(RealtimeAllocationTest, thePerThreadCounterCountsTheCallingThreadOnly) {
  const size_t before = heapAllocationCountOnThisThread();
  size_t otherThreadAllocations = 0;
  std::thread other([&otherThreadAllocations]() {
    const size_t otherBefore = heapAllocationCountOnThisThread();
    std::vector<std::unique_ptr<int>> values;
    for (int value = 0; value < 100; ++value) values.push_back(std::make_unique<int>(value));
    otherThreadAllocations = heapAllocationCountOnThisThread() - otherBefore;
  });
  other.join();
  EXPECT_GE(otherThreadAllocations, 100u);
  // std::thread allocates its state on the creating thread; none of the 100 values may show up here.
  const size_t afterThread = heapAllocationCountOnThisThread();
  EXPECT_LT(afterThread - before, 100u);
  const std::unique_ptr<int> mine = std::make_unique<int>(1);
  EXPECT_EQ(heapAllocationCountOnThisThread() - afterThread, 1u);
}

TEST(RealtimeAllocationTest, pushingAndPoppingPayloadsOfThePrototypesSizeDoesNotAllocate) {
  const TelemetrySample prototype = makePrototype();
  auto queue = std::make_unique<SpscQueue<TelemetrySample>>(/*capacity=*/16, prototype);
  TelemetrySample sample = prototype;
  TelemetrySample received = prototype;
  uint64_t pushed = 0;
  uint64_t popped = 0;

  const size_t before = heapAllocationCount();
  for (int round = 0; round < 1000; ++round) {
    // Three pushes per two pops: the queue fills up, so the full (dropping) path runs as well as the normal one.
    for (int i = 0; i < 3; ++i) {
      sample.time = round;
      sample.jointPositions.setConstant(round);
      sample.jointTorques.assign(30, static_cast<double>(round));
      pushed += queue->tryPush(sample) ? 1 : 0;
    }
    const bool pushedInPlace = queue->tryPushInPlace([round](TelemetrySample& slot) {
      slot.time = round;
      slot.jointPositions.setConstant(round);
    });
    pushed += pushedInPlace ? 1 : 0;
    popped += queue->tryPop(received) ? 1 : 0;
    const bool poppedInPlace = queue->tryPopInPlace([&received](const TelemetrySample& slot) { received.time = slot.time; });
    popped += poppedInPlace ? 1 : 0;
  }
  while (queue->tryPop(received)) {
    ++popped;
  }
  const size_t allocations = heapAllocationCount() - before;

  EXPECT_EQ(allocations, 0u);
  EXPECT_EQ(pushed, popped);
  EXPECT_GT(queue->droppedCount(), 0u) << "the full path was exercised";
  EXPECT_EQ(received.jointPositions.size(), 30);
}

TEST(RealtimeAllocationTest, waitingForTheNextPeriodDoesNotAllocate) {
  PeriodicTimer timer(microseconds(500), OverrunPolicy::kSkipMissedPeriods);
  timer.start();
  const size_t before = heapAllocationCount();
  nanoseconds totalLateness(0);
  for (int cycle = 0; cycle < 20; ++cycle) {
    totalLateness += timer.waitForNextPeriod().lateness;
  }
  const size_t allocations = heapAllocationCount() - before;
  EXPECT_EQ(allocations, 0u);
  EXPECT_GE(totalLateness, nanoseconds(0));
}

TEST(RealtimeAllocationTest, accumulatingAndPublishingLoopTimingDoesNotAllocate) {
  LoopTimingStats stats(milliseconds(2), milliseconds(100));
  TripleBuffer<LoopTimingSnapshot> buffer;
  uint64_t windows = 0;

  const size_t before = heapAllocationCount();
  nanoseconds start = std::chrono::seconds(1);
  for (int cycle = 0; cycle < 10'000; ++cycle) {
    if (stats.addCycle(CycleTiming{.start = start, .end = start + microseconds(300 + cycle % 7), .lateness = microseconds(cycle % 11)})) {
      buffer.writeSlot() = stats.lastSnapshot();
      buffer.publishWrite();
    }
    if (buffer.acquireRead()) {
      windows = buffer.readSlot().window;
    }
    start += milliseconds(2);
  }
  const size_t allocations = heapAllocationCount() - before;

  EXPECT_EQ(allocations, 0u);
  EXPECT_GT(windows, 100u);
}

// The pieces together, as the robot process's realtime thread uses them: pace, compute, hand a telemetry sample to the
// communication thread, account for the cycle and publish the timing once per window.
TEST(RealtimeAllocationTest, aWholeRealtimeCycleDoesNotAllocate) {
  const TelemetrySample prototype = makePrototype();
  auto telemetry = std::make_unique<SpscQueue<TelemetrySample>>(/*capacity=*/64, prototype);
  auto timing = std::make_unique<TripleBuffer<LoopTimingSnapshot>>();
  size_t allocations = 1;
  uint64_t windows = 0;

  // On a thread of its own, as in the robot process; the test's main thread only waits in join() meanwhile.
  std::thread realtime([&telemetry, &timing, &allocations, &windows]() {
    PeriodicTimer timer(milliseconds(1), OverrunPolicy::kSkipMissedPeriods);
    LoopTimingStats stats(milliseconds(1), milliseconds(10));
    timer.start();
    const size_t before = heapAllocationCount();
    for (int cycle = 0; cycle < 50; ++cycle) {
      const TimerWakeup wakeup = timer.waitForNextPeriod();
      telemetry->tryPushInPlace([&wakeup](TelemetrySample& slot) {
        slot.time = std::chrono::duration<double>(wakeup.deadline).count();
        slot.jointPositions.setConstant(slot.time);
      });
      if (stats.addCycle(wakeup, monotonicNow())) {
        timing->writeSlot() = stats.lastSnapshot();
        timing->publishWrite();
      }
    }
    allocations = heapAllocationCount() - before;
    windows = stats.lastSnapshot().window;
  });
  realtime.join();

  EXPECT_EQ(allocations, 0u);
  EXPECT_GE(windows, 1u);
}

}  // namespace
}  // namespace robot::realtime
