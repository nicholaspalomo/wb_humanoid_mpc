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

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/log/check.h"

namespace robot::realtime {

/**
 * The alignment that keeps data written by different threads off each other's cache lines. 128 rather than 64 bytes:
 * x86's adjacent-line prefetcher pulls cache lines in pairs, and some ARM cores have 128-byte lines.
 */
inline constexpr std::size_t kCacheLineSize = 128;

/**
 * A bounded, lock-free ring buffer between exactly one producer thread and exactly one consumer thread.
 *
 * Every slot is constructed once, as a copy of a prototype, when the queue is constructed. tryPush() copy-assigns the
 * value into a slot and tryPop() copy-assigns a slot out, so a payload that owns heap storage (an Eigen dynamic vector,
 * a std::vector) moves through the queue without allocating as long as it has the prototype's size: the assignment
 * reuses the storage the slot already owns. Neither call blocks or waits. A push into a full queue fails, is counted as
 * a drop, and leaves the queue unchanged, so a realtime producer is never held up by a slow consumer.
 *
 * The producer publishes a slot with a release store of its write index, which the consumer reads with an acquire load
 * before reading the slot; the consumer hands a slot back the same way through its read index. Each side keeps a
 * private copy of the other side's index and only reloads it when the copy says the queue is full (or empty), so in
 * steady state the two threads do not bounce each other's cache lines. The two indices live on separate cache lines.
 *
 * T must be copy-assignable. The queue is neither copyable nor movable.
 *
 * Producer:  tryPush(), tryPushInPlace().
 * Consumer:  tryPop(), tryPopInPlace(), empty().
 * Any thread: capacity(), sizeApprox(), droppedCount().
 */
template <typename T>
class SpscQueue {
 public:
  /**
   * @param capacity The number of values the queue holds when full; at least 1.
   * @param prototype Every slot starts as a copy of it. Give it the size of the payloads that will be pushed, so that
   *                  pushing them does not allocate.
   */
  explicit SpscQueue(std::size_t capacity, const T& prototype = T()) : slots_(capacity + 1, prototype) {
    CHECK_GT(capacity, 0u) << "an SpscQueue must hold at least one value";
  }

  SpscQueue(const SpscQueue&) = delete;
  SpscQueue& operator=(const SpscQueue&) = delete;
  SpscQueue(SpscQueue&&) = delete;
  SpscQueue& operator=(SpscQueue&&) = delete;

  // ---------------------------------------------------------------------------------------------------------------
  // Producer (exactly one thread)
  // ---------------------------------------------------------------------------------------------------------------

  /// Copies `value` into the queue. False, and counted as a drop, when the queue is full.
  bool tryPush(const T& value) {
    return tryPushInPlace([&value](T& slot) { slot = value; });
  }

  /**
   * Fills the next free slot in place, without an intermediate copy: `writer(T& slot)` is called with a slot that holds
   * whatever was last written to it (initially the prototype) and must overwrite the fields the consumer reads. False,
   * and counted as a drop, when the queue is full; `writer` is then not called. A writer that grows the slot's storage
   * allocates, as any assignment would.
   */
  template <typename Writer>
  bool tryPushInPlace(Writer&& writer) {
    const std::size_t writeIndex = writeIndex_.load(std::memory_order_relaxed);
    const std::size_t nextWriteIndex = advance(writeIndex);
    if (nextWriteIndex == readIndexCache_) {
      readIndexCache_ = readIndex_.load(std::memory_order_acquire);
      if (nextWriteIndex == readIndexCache_) {
        // Only the producer writes the counter, so a load and a store suffice; other threads read it relaxed.
        droppedCount_.store(droppedCount_.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
        return false;
      }
    }
    std::forward<Writer>(writer)(slots_[writeIndex]);
    writeIndex_.store(nextWriteIndex, std::memory_order_release);
    return true;
  }

  // ---------------------------------------------------------------------------------------------------------------
  // Consumer (exactly one thread)
  // ---------------------------------------------------------------------------------------------------------------

  /// Copies the oldest value into `value` and removes it. False, leaving `value` untouched, when the queue is empty.
  bool tryPop(T& value) {
    return tryPopInPlace([&value](const T& slot) { value = slot; });
  }

  /**
   * Reads the oldest value in place, then removes it: `reader(const T& slot)` may serialize the slot directly instead
   * of copying it out first. The slot stays reserved for the consumer until `reader` returns. False when the queue is
   * empty; `reader` is then not called.
   */
  template <typename Reader>
  bool tryPopInPlace(Reader&& reader) {
    const std::size_t readIndex = readIndex_.load(std::memory_order_relaxed);
    if (readIndex == writeIndexCache_) {
      writeIndexCache_ = writeIndex_.load(std::memory_order_acquire);
      if (readIndex == writeIndexCache_) {
        return false;
      }
    }
    std::forward<Reader>(reader)(static_cast<const T&>(slots_[readIndex]));
    readIndex_.store(advance(readIndex), std::memory_order_release);
    return true;
  }

  /// True when there is nothing to pop. Exact on the consumer thread; a hint anywhere else.
  bool empty() const { return readIndex_.load(std::memory_order_acquire) == writeIndex_.load(std::memory_order_acquire); }

  // ---------------------------------------------------------------------------------------------------------------
  // Any thread
  // ---------------------------------------------------------------------------------------------------------------

  /// The number of values the queue holds when full.
  std::size_t capacity() const { return slots_.size() - 1; }

  /**
   * The number of values the queue held at one instant during the call, possibly out of date by the time it returns.
   *
   * The two indices are read as a pair the queue actually held: the read index, then the write index, then the read
   * index again, until it has not moved in between (on the consumer thread it never does). Reading them one after the
   * other without that check can produce a size no state of the queue ever had: an empty queue that the producer and
   * the consumer each step once between the two loads reads as full. Should the consumer pop between every pair of
   * loads kSizeSnapshotAttempts times in a row, the last read index loaded before the write index is used. Never more
   * than capacity(); it does not block, and it writes nothing either side reads.
   */
  std::size_t sizeApprox() const {
    std::size_t readIndex = readIndex_.load(std::memory_order_acquire);
    for (int attempt = 1;; ++attempt) {
      const std::size_t writeIndex = writeIndex_.load(std::memory_order_acquire);
      const std::size_t readIndexAfter = readIndex_.load(std::memory_order_acquire);
      if (readIndexAfter == readIndex || attempt == kSizeSnapshotAttempts) {
        return distance(readIndex, writeIndex);
      }
      readIndex = readIndexAfter;
    }
  }

  /// The number of pushes that failed because the queue was full, since construction.
  std::uint64_t droppedCount() const { return droppedCount_.load(std::memory_order_relaxed); }

 private:
  static_assert(std::atomic<std::size_t>::is_always_lock_free, "the indices must be lock-free atomics");
  static_assert(std::atomic<std::uint64_t>::is_always_lock_free, "the drop counter must be a lock-free atomic");

  // How often sizeApprox() reads the indices again while the consumer keeps moving the read index.
  static constexpr int kSizeSnapshotAttempts = 16;

  // One slot more than the capacity, so that a full queue (write index just behind the read index) and an empty one
  // (equal indices) are told apart without a shared counter.
  std::size_t advance(std::size_t index) const {
    const std::size_t next = index + 1;
    return next == slots_.size() ? 0 : next;
  }

  // The values from `readIndex` up to `writeIndex`, around the ring: at most capacity().
  std::size_t distance(std::size_t readIndex, std::size_t writeIndex) const {
    return writeIndex >= readIndex ? writeIndex - readIndex : writeIndex + slots_.size() - readIndex;
  }

  // Read-only after construction.
  alignas(kCacheLineSize) std::vector<T> slots_;

  // Written by the producer only.
  alignas(kCacheLineSize) std::atomic<std::size_t> writeIndex_{0};
  std::size_t readIndexCache_ = 0;
  std::atomic<std::uint64_t> droppedCount_{0};

  // Written by the consumer only. The queue's alignment rounds its size up to whole cache lines, so whatever follows
  // the queue in memory does not share this line either.
  alignas(kCacheLineSize) std::atomic<std::size_t> readIndex_{0};
  std::size_t writeIndexCache_ = 0;
};

}  // namespace robot::realtime
