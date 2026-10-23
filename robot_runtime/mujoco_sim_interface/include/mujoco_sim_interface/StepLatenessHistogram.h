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

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>

#include "absl/time/time.h"

namespace robot::mujoco_sim_interface {

/**
 * How late the physics loop finished its steps: one sample per step, the time by which the step overran the start of
 * the next one (zero or less when it finished in time). The physics thread records; any thread reads the count, the
 * percentiles and the maximum, to check that nothing else on the machine (a viewer, say) delays the simulation.
 *
 * Every bin is preallocated and every operation is a relaxed atomic, so record() is wait-free and allocation-free on
 * the physics thread. One thread records at a time; reads may run concurrently and see a sample count that is a few
 * samples behind the bins. reset() while samples are recorded loses or keeps those samples.
 */
class StepLatenessHistogram {
 public:
  static constexpr int64_t kBinWidthNanoseconds = 5'000;  // 5 us
  static constexpr int kNumBins = 4'000;                  // up to 20 ms; later samples share the last bin

  StepLatenessHistogram() = default;
  StepLatenessHistogram(const StepLatenessHistogram&) = delete;
  StepLatenessHistogram& operator=(const StepLatenessHistogram&) = delete;
  StepLatenessHistogram(StepLatenessHistogram&&) = delete;
  StepLatenessHistogram& operator=(StepLatenessHistogram&&) = delete;
  ~StepLatenessHistogram() = default;

  /** Records one step that finished `lateness` after the start of the next one; a negative one counts as zero. */
  void record(std::chrono::nanoseconds lateness);

  /** Forgets every sample. */
  void reset();

  /** The number of samples recorded since construction or the last reset(). */
  int64_t count() const;

  /**
   * The lateness that `fraction` (in [0, 1]) of the samples do not exceed, to the upper edge of its 5 us bin (the
   * maximum for the last bin and for fraction 1); zero without samples.
   */
  absl::Duration percentile(double fraction) const;

  /** The largest lateness recorded, exactly; zero without samples. */
  absl::Duration max() const;

  /** The number of samples later than `lateness`, to the 5 us bin: those of the bins that start at or after it. */
  int64_t countLaterThan(absl::Duration lateness) const;

 private:
  std::array<std::atomic<int64_t>, kNumBins> bins_{};
  std::atomic<int64_t> count_ = 0;
  std::atomic<int64_t> maxNanoseconds_ = 0;
};

}  // namespace robot::mujoco_sim_interface
