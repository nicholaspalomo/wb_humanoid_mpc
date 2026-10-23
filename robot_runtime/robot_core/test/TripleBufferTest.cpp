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
#include <thread>
#include <vector>

#include "gtest/gtest.h"

#include "robot_core/TripleBuffer.h"

namespace robot {
namespace {

// Simple POD type for testing
struct TestData {
  int value = 0;
  double extra = 0.0;
};

// ---------------------------------------------------------------------------
// Basic functionality
// ---------------------------------------------------------------------------

TEST(TripleBufferTest, DefaultConstructedReadsDefaultValue) {
  TripleBuffer<TestData> buf;
  const TestData& data = buf.readSlot();
  EXPECT_EQ(data.value, 0);
  EXPECT_DOUBLE_EQ(data.extra, 0.0);
}

TEST(TripleBufferTest, InitialValueAllSlotsInitialized) {
  TestData init{.value = 42, .extra = 3.14};
  TripleBuffer<TestData> buf(init);
  const TestData& data = buf.readSlot();
  EXPECT_EQ(data.value, 42);
  EXPECT_DOUBLE_EQ(data.extra, 3.14);
}

TEST(TripleBufferTest, WriteAndPublishConsumerSeesLatest) {
  TripleBuffer<int> buf(0);

  buf.writeSlot() = 100;
  buf.publishWrite();

  ASSERT_TRUE(buf.acquireRead());
  EXPECT_EQ(buf.readSlot(), 100);
}

TEST(TripleBufferTest, MultipleWritesConsumerSeesOnlyLatest) {
  TripleBuffer<int> buf(0);

  buf.writeSlot() = 1;
  buf.publishWrite();
  buf.writeSlot() = 2;
  buf.publishWrite();
  buf.writeSlot() = 3;
  buf.publishWrite();

  // Consumer should see the most recent value
  ASSERT_TRUE(buf.acquireRead());
  EXPECT_EQ(buf.readSlot(), 3);
}

TEST(TripleBufferTest, AcquireWithoutPublishReturnsFalse) {
  TripleBuffer<int> buf(0);
  EXPECT_FALSE(buf.acquireRead());
}

TEST(TripleBufferTest, DoubleAcquireSecondReturnsFalse) {
  TripleBuffer<int> buf(0);

  buf.writeSlot() = 42;
  buf.publishWrite();

  ASSERT_TRUE(buf.acquireRead());
  EXPECT_EQ(buf.readSlot(), 42);

  // No new data published since last acquire
  EXPECT_FALSE(buf.acquireRead());
  // Read slot still valid
  EXPECT_EQ(buf.readSlot(), 42);
}

TEST(TripleBufferTest, HasNewDataReflectsState) {
  TripleBuffer<int> buf(0);

  EXPECT_FALSE(buf.hasNewData());

  buf.writeSlot() = 1;
  buf.publishWrite();
  EXPECT_TRUE(buf.hasNewData());

  buf.acquireRead();
  EXPECT_FALSE(buf.hasNewData());
}

TEST(TripleBufferTest, AlternatingWriteReadCorrectValues) {
  TripleBuffer<int> buf(0);

  for (int i = 1; i <= 100; ++i) {
    buf.writeSlot() = i;
    buf.publishWrite();
    ASSERT_TRUE(buf.acquireRead());
    EXPECT_EQ(buf.readSlot(), i);
  }
}

// ---------------------------------------------------------------------------
// Concurrent SPSC correctness
// ---------------------------------------------------------------------------

TEST(TripleBufferTest, ConcurrentSPSCReaderSeesMonotonicallyIncreasingValues) {
  TripleBuffer<int> buf(0);
  constexpr int kNumWrites = 100000;
  std::atomic<bool> done{false};

  // Producer thread: write 1..N
  std::thread producer([&]() {
    for (int i = 1; i <= kNumWrites; ++i) {
      buf.writeSlot() = i;
      buf.publishWrite();
    }
    done.store(true, std::memory_order_release);
  });

  // Consumer thread: read values, verify monotonically increasing
  int lastSeen = 0;
  int readCount = 0;
  while (!done.load(std::memory_order_acquire) || buf.hasNewData()) {
    if (buf.acquireRead()) {
      int val = buf.readSlot();
      ASSERT_GE(val, lastSeen) << "Non-monotonic value at read " << readCount;
      lastSeen = val;
      ++readCount;
    }
  }
  // Drain any remaining
  if (buf.acquireRead()) {
    int val = buf.readSlot();
    ASSERT_GE(val, lastSeen);
    lastSeen = val;
  }

  producer.join();

  // We must have eventually seen the final value
  EXPECT_EQ(lastSeen, kNumWrites);
  // Consumer should have read at least 1 value (though likely far fewer than kNumWrites
  // because the triple buffer drops intermediate values)
  EXPECT_GT(readCount, 0);
}

TEST(TripleBufferTest, ConcurrentSPSCNoDataRaceConsistentStruct) {
  struct BigData {
    int header = 0;
    int payload[64]{};
    int footer = 0;

    void fill(int val) {
      header = val;
      for (int& p : payload) p = val;
      footer = val;
    }

    bool isConsistent() const {
      if (header != footer) return false;
      for (const int& p : payload) {
        if (p != header) return false;
      }
      return true;
    }
  };

  TripleBuffer<BigData> buf;
  constexpr int kNumWrites = 50000;
  std::atomic<bool> done{false};

  std::thread producer([&]() {
    for (int i = 1; i <= kNumWrites; ++i) {
      buf.writeSlot().fill(i);
      buf.publishWrite();
    }
    done.store(true, std::memory_order_release);
  });

  int readCount = 0;
  while (!done.load(std::memory_order_acquire) || buf.hasNewData()) {
    if (buf.acquireRead()) {
      ASSERT_TRUE(buf.readSlot().isConsistent())
          << "Torn read detected at read " << readCount << ", header=" << buf.readSlot().header << ", footer=" << buf.readSlot().footer;
      ++readCount;
    }
  }
  // Final drain
  if (buf.acquireRead()) {
    ASSERT_TRUE(buf.readSlot().isConsistent());
  }

  producer.join();
  EXPECT_GT(readCount, 0);
}

// ---------------------------------------------------------------------------
// Bounded latency
// ---------------------------------------------------------------------------

TEST(TripleBufferTest, ReaderNeverBlocksBoundedLatency) {
  TripleBuffer<int> buf(0);
  constexpr int kNumReads = 10000;

  // Measure reader latency without any writer
  const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  for (int i = 0; i < kNumReads; ++i) {
    buf.acquireRead();
    (void)buf.readSlot();
  }
  const std::chrono::steady_clock::duration elapsed = std::chrono::steady_clock::now() - start;
  const std::chrono::nanoseconds::rep avgNs = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count() / kNumReads;

  // Reader should never block — avg latency should be well under 1µs
  EXPECT_LT(avgNs, 1000) << "Average read latency " << avgNs << "ns exceeds 1µs";
}

}  // namespace
}  // namespace robot
