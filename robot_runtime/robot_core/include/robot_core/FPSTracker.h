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

#include "absl/log/absl_check.h"

namespace robot {

/**
 * The rate at which tick() is called, smoothed exponentially with `alpha` (0 < alpha <= 1; 1 keeps only the newest
 * period), for the simulator and renderer frame rates of the MuJoCo metrics overlay. Not thread-safe: one thread ticks and reads it.
 */
class FPSTracker {
 public:
  explicit FPSTracker(double alpha = 0.1) : alpha_(alpha) {
    ABSL_CHECK(alpha > 0.0 && alpha <= 1.0) << "FPSTracker: alpha must be in (0, 1], got " << alpha;
    lastTimePoint_ = std::chrono::steady_clock::now();
  }

  ~FPSTracker() = default;

  void tick() {
    const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
    double deltaTime = std::chrono::duration<double>(now - lastTimePoint_).count();
    lastTimePoint_ = now;

    double currentFPS = 1.0 / deltaTime;

    if (initialized_) {
      fps_ = alpha_ * currentFPS + (1.0 - alpha_) * fps_;
    } else {
      fps_ = currentFPS;
      initialized_ = true;
    }
  }

  void reset() { initialized_ = false; }

  double fps() const { return fps_; }

 private:
  bool initialized_ = false;

  double alpha_;  // Smoothing factor (0 < alpha <= 1)
  double fps_ = 0.0;

  std::chrono::time_point<std::chrono::steady_clock> lastTimePoint_;
};
}  // namespace robot
