#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include "robot_core/TripleBuffer.h"

namespace robot {
namespace {

// Simple POD type for testing
struct TestData {
  int value{0};
  double extra{0.0};
};

// ---------------------------------------------------------------------------
// Basic functionality
// ---------------------------------------------------------------------------

TEST(TripleBufferTest, DefaultConstructed_ReadsDefaultValue) {
  TripleBuffer<TestData> buf;
  const auto& data = buf.readSlot();
  EXPECT_EQ(data.value, 0);
  EXPECT_DOUBLE_EQ(data.extra, 0.0);
}

TEST(TripleBufferTest, InitialValue_AllSlotsInitialized) {
  TestData init{42, 3.14};
  TripleBuffer<TestData> buf(init);
  const auto& data = buf.readSlot();
  EXPECT_EQ(data.value, 42);
  EXPECT_DOUBLE_EQ(data.extra, 3.14);
}

TEST(TripleBufferTest, WriteAndPublish_ConsumerSeesLatest) {
  TripleBuffer<int> buf(0);

  buf.writeSlot() = 100;
  buf.publishWrite();

  ASSERT_TRUE(buf.acquireRead());
  EXPECT_EQ(buf.readSlot(), 100);
}

TEST(TripleBufferTest, MultipleWrites_ConsumerSeesOnlyLatest) {
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

TEST(TripleBufferTest, AcquireWithoutPublish_ReturnsFalse) {
  TripleBuffer<int> buf(0);
  EXPECT_FALSE(buf.acquireRead());
}

TEST(TripleBufferTest, DoubleAcquire_SecondReturnsFalse) {
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

TEST(TripleBufferTest, HasNewData_ReflectsState) {
  TripleBuffer<int> buf(0);

  EXPECT_FALSE(buf.hasNewData());

  buf.writeSlot() = 1;
  buf.publishWrite();
  EXPECT_TRUE(buf.hasNewData());

  buf.acquireRead();
  EXPECT_FALSE(buf.hasNewData());
}

TEST(TripleBufferTest, AlternatingWriteRead_CorrectValues) {
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

TEST(TripleBufferTest, ConcurrentSPSC_ReaderSeesMonotonicallyIncreasingValues) {
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

TEST(TripleBufferTest, ConcurrentSPSC_NoDataRace_ConsistentStruct) {
  struct BigData {
    int header{0};
    int payload[64]{};
    int footer{0};

    void fill(int val) {
      header = val;
      for (auto& p : payload) p = val;
      footer = val;
    }

    bool isConsistent() const {
      if (header != footer) return false;
      for (const auto& p : payload) {
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

TEST(TripleBufferTest, ReaderNeverBlocks_BoundedLatency) {
  TripleBuffer<int> buf(0);
  constexpr int kNumReads = 10000;

  // Measure reader latency without any writer
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < kNumReads; ++i) {
    buf.acquireRead();
    (void)buf.readSlot();
  }
  auto elapsed = std::chrono::steady_clock::now() - start;
  auto avgNs = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count() / kNumReads;

  // Reader should never block — avg latency should be well under 1µs
  EXPECT_LT(avgNs, 1000) << "Average read latency " << avgNs << "ns exceeds 1µs";
}

}  // namespace
}  // namespace robot
