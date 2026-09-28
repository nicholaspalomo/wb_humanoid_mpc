/******************************************************************************
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.
Copyright (c) 2024, 1X Technologies. All rights reserved.

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

#include <cassert>
#include <cmath>
#include <optional>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid {

/**
 * A first-order low-pass filter, dy/dt = (x - y) / tau with the time constant tau = 1 / (2 pi f_c) of the break
 * frequency f_c, run on the time it is given - the solver time of the MPC, never the wall clock.
 *
 * It is discretized exactly for an input held between samples: y_k = y_{k-1} + alpha (x_k - y_{k-1}) with
 * alpha = 1 - exp(-dt / tau) and dt the time since the previous sample, so that the step response is 1 - exp(-t / tau)
 * at every sample however the samples are spaced.
 *
 * A break frequency of 0 switches the filter off: the output is the input, sample by sample. The output still tracks the
 * input then, so that a filter switched on later (setBreakFrequency, a hot reload) starts from the last input instead of
 * jumping back to where it was.
 *
 * The first sample after construction or reset() starts the clock and returns the initial output unchanged. A sample
 * earlier than the previous one - a clock that ran backwards - restarts the clock the same way, holding the output.
 *
 * This replaces a filter that blended its input with its initial output by the wall-clock time since construction and
 * never updated either: the "5 Hz" command filter of every robot scaled the command by t / (t + 0.032 s), within 3% of
 * the identity a second after start-up. The robots now configure the filter in reference.yaml
 * (ProceduralMpcMotionManager::kVelocityCommandFilterBreakFrequencyKey) and ship it off.
 */
class BreakFrequencyAlphaFilter final {
 public:
  /** A filter that is off (break frequency 0), whose output starts at `initialOutput`. */
  explicit BreakFrequencyAlphaFilter(vector_t initialOutput) : initialOutput_(initialOutput), output_(std::move(initialOutput)) {}

  /** OK for a finite break frequency >= 0 [Hz], 0 meaning off; InvalidArgument otherwise. */
  static absl::Status validateBreakFrequency(scalar_t breakFrequency) {
    if (!std::isfinite(breakFrequency) || breakFrequency < 0.0) {
      return absl::InvalidArgumentError(absl::StrCat("the break frequency of a first-order low-pass filter must be a finite number of Hz, ",
                                                     ">= 0 (0 switches the filter off); got ", breakFrequency, "."));
    }
    return absl::OkStatus();
  }

  /**
   * Sets the break frequency [Hz], 0 switching the filter off. Keeps the output and the clock, so that the filter goes on
   * from where it is. An invalid value (validateBreakFrequency) is rejected and leaves the filter as it was.
   */
  absl::Status setBreakFrequency(scalar_t breakFrequency) {
    const absl::Status status = validateBreakFrequency(breakFrequency);
    if (status.ok()) breakFrequency_ = breakFrequency;
    return status;
  }

  scalar_t getBreakFrequency() const { return breakFrequency_; }
  bool isEnabled() const { return breakFrequency_ > 0.0; }

  /**
   * Filters `input`, sampled at `time` [s], and returns the output. `input` has the size of the initial output.
   */
  const vector_t& update(scalar_t time, const vector_t& input) {
    assert(input.size() == output_.size());
    if (!isEnabled()) {
      output_ = input;
    } else if (lastTime_.has_value() && time > *lastTime_) {
      // alpha = 1 - exp(-dt / tau), as expm1 so that it keeps its digits for a dt much shorter than tau.
      const scalar_t alpha = -std::expm1(-2.0 * M_PI * breakFrequency_ * (time - *lastTime_));
      output_ += alpha * (input - output_);
    }
    lastTime_ = time;
    return output_;
  }

  /** The output of the last update(), or the initial output before the first one. */
  const vector_t& getOutput() const { return output_; }

  /** Back to the state after construction: the initial output and no clock. The break frequency is kept. */
  void reset() {
    output_ = initialOutput_;
    lastTime_.reset();
  }

 private:
  scalar_t breakFrequency_ = 0.0;  // [Hz] 0: off
  vector_t initialOutput_;
  vector_t output_;
  std::optional<scalar_t> lastTime_;
};

}  // namespace ocs2::humanoid
