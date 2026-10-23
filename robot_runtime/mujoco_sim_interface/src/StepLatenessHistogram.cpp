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

#include "mujoco_sim_interface/StepLatenessHistogram.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>

#include "absl/time/time.h"

namespace robot::mujoco_sim_interface {

void StepLatenessHistogram::record(std::chrono::nanoseconds lateness) {
  const int64_t nanoseconds = std::max<int64_t>(lateness.count(), 0);
  const int64_t bin = std::min<int64_t>(nanoseconds / kBinWidthNanoseconds, kNumBins - 1);
  bins_[bin].fetch_add(1, std::memory_order_relaxed);
  // One writer, so a plain compare and store keeps the maximum.
  if (nanoseconds > maxNanoseconds_.load(std::memory_order_relaxed)) maxNanoseconds_.store(nanoseconds, std::memory_order_relaxed);
  count_.fetch_add(1, std::memory_order_relaxed);
}

void StepLatenessHistogram::reset() {
  for (std::atomic<int64_t>& bin : bins_) bin.store(0, std::memory_order_relaxed);
  maxNanoseconds_.store(0, std::memory_order_relaxed);
  count_.store(0, std::memory_order_relaxed);
}

int64_t StepLatenessHistogram::count() const {
  return count_.load(std::memory_order_relaxed);
}

absl::Duration StepLatenessHistogram::percentile(double fraction) const {
  int64_t total = 0;
  for (const std::atomic<int64_t>& bin : bins_) total += bin.load(std::memory_order_relaxed);
  if (total == 0) return absl::ZeroDuration();
  if (!(fraction < 1.0)) return max();
  const int64_t wanted = std::max<int64_t>(1, static_cast<int64_t>(std::ceil(std::max(fraction, 0.0) * static_cast<double>(total))));
  int64_t seen = 0;
  for (int bin = 0; bin < kNumBins; ++bin) {
    seen += bins_[bin].load(std::memory_order_relaxed);
    if (seen < wanted) continue;
    // The last bin holds everything later than the others, so its upper edge is the maximum.
    return bin + 1 == kNumBins ? max() : std::min(absl::Nanoseconds((bin + 1) * kBinWidthNanoseconds), max());
  }
  return max();
}

int64_t StepLatenessHistogram::countLaterThan(absl::Duration lateness) const {
  const int64_t nanoseconds = std::max<int64_t>(absl::ToInt64Nanoseconds(lateness), 0);
  const int64_t firstBin = std::min<int64_t>((nanoseconds + kBinWidthNanoseconds - 1) / kBinWidthNanoseconds, kNumBins - 1);
  int64_t later = 0;
  for (int64_t bin = firstBin; bin < kNumBins; ++bin) later += bins_[bin].load(std::memory_order_relaxed);
  return later;
}

absl::Duration StepLatenessHistogram::max() const {
  return absl::Nanoseconds(maxNanoseconds_.load(std::memory_order_relaxed));
}

}  // namespace robot::mujoco_sim_interface
