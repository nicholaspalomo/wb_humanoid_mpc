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

#include "mujoco_sim_interface/BaseAccelerometer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "robot_core/Types.h"

namespace robot::mujoco_sim_interface {
namespace {

/** The samples a mean acceleration takes: two velocities. */
constexpr size_t kMinSamplesForMean = 2;

}  // namespace

int BaseAccelerometer::windowStepsFor(double window, double dt) {
  if (!(std::isfinite(window) && std::isfinite(dt) && window > 0.0 &&
        dt > 0.0)) {
    return 1;
  }
  return std::max(1, static_cast<int>(std::lround(window / dt)));
}

BaseAccelerometer::BaseAccelerometer(int windowSteps)
    : samples_(static_cast<size_t>(std::max(1, windowSteps)) + 1) {}

void BaseAccelerometer::restart(double time,
                                const vector3_t& linvelBaseRtWorldEwrtWorld) {
  count_ = 0;
  record(time, linvelBaseRtWorldEwrtWorld);
}

void BaseAccelerometer::record(double time,
                               const vector3_t& linvelBaseRtWorldEwrtWorld) {
  if (count_ > 0 && !(time > samples_[newest_].time)) {
    samples_[newest_] = {
        .time = time, .linvelBaseRtWorldEwrtWorld = linvelBaseRtWorldEwrtWorld};
    return;
  }
  newest_ = count_ == 0 ? 0 : (newest_ + 1) % samples_.size();
  samples_[newest_] = {
      .time = time, .linvelBaseRtWorldEwrtWorld = linvelBaseRtWorldEwrtWorld};
  count_ = std::min(count_ + 1, samples_.size());
}

vector3_t BaseAccelerometer::linaccBaseRtWorldEwrtWorld() const {
  if (count_ < kMinSamplesForMean) return vector3_t::Zero();
  const size_t oldest =
      (newest_ + samples_.size() - (count_ - 1)) % samples_.size();
  const Sample& now = samples_[newest_];
  const Sample& then = samples_[oldest];
  return (now.linvelBaseRtWorldEwrtWorld - then.linvelBaseRtWorldEwrtWorld) /
         (now.time - then.time);
}

}  // namespace robot::mujoco_sim_interface
