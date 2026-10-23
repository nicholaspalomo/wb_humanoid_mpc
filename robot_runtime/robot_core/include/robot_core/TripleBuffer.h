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

namespace robot {

/// Lock-free triple buffer for single-producer, single-consumer (SPSC) scenarios.
///
/// Three slots are maintained: one for writing (producer), one for reading (consumer),
/// and one "clean" slot that serves as the exchange point. The producer writes to its
/// slot and atomically swaps it with the clean slot. The consumer atomically swaps the
/// clean slot with its read slot, then reads. Producer and consumer never touch the
/// same slot simultaneously.
///
/// State encoding: a single atomic uint8_t encodes [newData(1 bit) | clean(2 bits) | read(2 bits) | write(2 bits)]
/// using 7 bits total. The newData flag indicates whether the clean slot has been updated
/// since the last consumer read.
template <typename T>
class TripleBuffer {
 public:
  /// Construct with default-constructed slots.
  TripleBuffer() : state_(encodeState(kInitialWriteSlot, kInitialCleanSlot, kInitialReadSlot, /*newData=*/false)) {}

  /// Construct with initial value copied into all three slots.
  explicit TripleBuffer(const T& initial)
      : slots_{initial, initial, initial}, state_(encodeState(kInitialWriteSlot, kInitialCleanSlot, kInitialReadSlot, /*newData=*/false)) {}

  /// Non-copyable, non-movable (contains atomics).
  TripleBuffer(const TripleBuffer&) = delete;
  TripleBuffer& operator=(const TripleBuffer&) = delete;
  ~TripleBuffer() = default;

  // ---------------------------------------------------------------------------
  // Producer API (call from exactly one thread)
  // ---------------------------------------------------------------------------

  /// Get a mutable reference to the current write slot.
  /// The producer can modify this freely; the consumer never sees it until publishWrite().
  T& writeSlot() { return slots_[getWriteIdx()]; }

  /// Publish the current write slot: atomically swap write ↔ clean and set newData flag.
  void publishWrite() {
    uint8_t expected = state_.load(std::memory_order_acquire);
    uint8_t desired = expected;
    do {
      // Swap write and clean indices, set newData = true
      uint8_t w = decodeWrite(expected);
      uint8_t c = decodeClean(expected);
      uint8_t r = decodeRead(expected);
      desired = encodeState(c, w, r, /*newData=*/true);  // write becomes clean, clean becomes write
    } while (!state_.compare_exchange_weak(expected, desired, std::memory_order_acq_rel, std::memory_order_acquire));
  }

  // ---------------------------------------------------------------------------
  // Consumer API (call from exactly one thread)
  // ---------------------------------------------------------------------------

  /// Acquire the latest published data: atomically swap clean ↔ read if new data is available.
  /// Returns true if new data was acquired, false if the read slot already has the latest.
  bool acquireRead() {
    uint8_t expected = state_.load(std::memory_order_acquire);
    if (!decodeNewData(expected)) {
      return false;  // No new data since last acquire
    }
    uint8_t desired = expected;
    do {
      if (!decodeNewData(expected)) {
        return false;  // Another acquireRead() raced ahead (shouldn't happen in SPSC)
      }
      uint8_t w = decodeWrite(expected);
      uint8_t c = decodeClean(expected);
      uint8_t r = decodeRead(expected);
      desired = encodeState(w, r, c, /*newData=*/false);  // read becomes clean, clean becomes read, clear newData
    } while (!state_.compare_exchange_weak(expected, desired, std::memory_order_acq_rel, std::memory_order_acquire));
    return true;
  }

  /// Get a const reference to the current read slot.
  const T& readSlot() const { return slots_[getReadIdx()]; }

  /// Get a mutable reference to the current read slot (for in-place operations like mj_copyData).
  T& readSlot() { return slots_[getReadIdx()]; }

  /// Check if there is new data available without acquiring it.
  bool hasNewData() const { return decodeNewData(state_.load(std::memory_order_acquire)); }

 private:
  /** The number of slots: one written, one read and the clean one between them. */
  static constexpr size_t kNumSlots = 3;
  /** The slots the producer writes, the exchange holds and the consumer reads at construction. */
  static constexpr uint8_t kInitialWriteSlot = 0;
  static constexpr uint8_t kInitialCleanSlot = 1;
  static constexpr uint8_t kInitialReadSlot = 2;
  // State encoding: [newData(1) | clean(2) | read(2) | write(2)] = 7 bits. A slot index takes two bits.
  static constexpr int kSlotIndexMask = 0x3;
  static constexpr int kReadSlotShift = 2;
  static constexpr int kCleanSlotShift = 4;
  static constexpr int kNewDataFlag = 0x40;

  static constexpr uint8_t encodeState(uint8_t write, uint8_t clean, uint8_t read, bool newData) {
    return static_cast<uint8_t>((write & kSlotIndexMask) | ((read & kSlotIndexMask) << kReadSlotShift) |
                                ((clean & kSlotIndexMask) << kCleanSlotShift) | (newData ? kNewDataFlag : 0));
  }

  static constexpr uint8_t decodeWrite(uint8_t s) { return s & kSlotIndexMask; }
  static constexpr uint8_t decodeRead(uint8_t s) { return (s >> kReadSlotShift) & kSlotIndexMask; }
  static constexpr uint8_t decodeClean(uint8_t s) { return (s >> kCleanSlotShift) & kSlotIndexMask; }
  static constexpr bool decodeNewData(uint8_t s) { return (s & kNewDataFlag) != 0; }

  uint8_t getWriteIdx() const { return decodeWrite(state_.load(std::memory_order_acquire)); }
  uint8_t getReadIdx() const { return decodeRead(state_.load(std::memory_order_acquire)); }

  T slots_[kNumSlots];
  std::atomic<uint8_t> state_;
};

}  // namespace robot
