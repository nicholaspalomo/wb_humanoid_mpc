/******************************************************************************
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.

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

#include <chrono>
#include <optional>

#include "absl/log/absl_check.h"

namespace robot {

/**
 * The rate at which tick() is called, smoothed exponentially with `alpha` (0 < alpha <= 1; 1 keeps only the newest
 * period), for the simulator and renderer frame rates of the MuJoCo metrics overlay. Each rate is one over the period
 * between two ticks, so the first tick after construction or reset() only starts the clock, and fps() is 0 until the
 * second. A period of zero or less (two ticks within one tick of the steady clock) carries no rate and is skipped.
 * Not thread-safe: one thread ticks and reads it.
 */
class FPSTracker {
 public:
  /** The default smoothing factor `alpha`. */
  static constexpr double kDefaultAlpha = 0.1;

  explicit FPSTracker(double alpha = kDefaultAlpha) : alpha_(alpha) {
    ABSL_CHECK(alpha > 0.0 && alpha <= 1.0) << "FPSTracker: alpha must be in (0, 1], got " << alpha;
  }

  FPSTracker(const FPSTracker&) = default;
  FPSTracker& operator=(const FPSTracker&) = default;
  FPSTracker(FPSTracker&&) = default;
  FPSTracker& operator=(FPSTracker&&) = default;
  ~FPSTracker() = default;

  /** Records a tick now. */
  void tick() { tickAt(std::chrono::steady_clock::now()); }

  /** Records a tick at `now`, which must not be earlier than the previous tick's. */
  void tickAt(std::chrono::steady_clock::time_point now) {
    if (!lastTimePoint_.has_value()) {
      lastTimePoint_ = now;
      return;
    }
    const double period = std::chrono::duration<double>(now - *lastTimePoint_).count();
    if (!(period > 0.0)) return;
    lastTimePoint_ = now;
    const double rate = 1.0 / period;
    fps_ = hasRate_ ? alpha_ * rate + (1.0 - alpha_) * fps_ : rate;
    hasRate_ = true;
  }

  /** Forgets the ticks so far: the next tick starts the clock again, and fps() is 0 until the one after it. */
  void reset() {
    lastTimePoint_.reset();
    hasRate_ = false;
    fps_ = 0.0;
  }

  /** [Hz] The smoothed rate; 0 before two ticks. */
  double fps() const { return fps_; }

 private:
  double alpha_;  // Smoothing factor (0 < alpha <= 1)
  /// Whether fps_ holds a rate yet; the first rate is taken as it is rather than smoothed against 0.
  bool hasRate_ = false;
  double fps_ = 0.0;
  /// The previous tick; empty before the first and after reset().
  std::optional<std::chrono::steady_clock::time_point> lastTimePoint_;
};
}  // namespace robot
