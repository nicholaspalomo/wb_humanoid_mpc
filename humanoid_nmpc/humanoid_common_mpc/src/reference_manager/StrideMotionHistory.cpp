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

#include "humanoid_common_mpc/reference_manager/StrideMotionHistory.h"

#include <algorithm>
#include <cmath>
#include <deque>

#include "Eigen/Geometry"
#include "absl/functional/function_ref.h"
#include "ocs2_core/Dimensions.h"

#include "humanoid_common_mpc/common/Sinc.h"
#include "humanoid_common_mpc/orientation/BaseOrientation.h"

namespace ocs2::humanoid {
namespace {

/** `vector` (x, y) rotated by -`angle` about z: R_z(angle)^T vector. */
vector2_t rotatedBack(scalar_t angle, const vector2_t& vector) {
  const scalar_t cosine = std::cos(angle);
  const scalar_t sine = std::sin(angle);
  return vector2_t(cosine * vector(0) + sine * vector(1), -sine * vector(0) + cosine * vector(1));
}

}  // namespace

BasePlanarMotion instantaneousPlanarMotion(const vector4_t& orientation, const vector6_t& twistWorld) {
  const scalar_t heading = twistHeading(orientation);
  const quaternion_t rotation(orientation);
  const vector3_t angularVelocityLocal = rotation.conjugate() * vector3_t(twistWorld.tail<kRotationTangentDim>());
  return BasePlanarMotion{.velocityInHeadingFrame = rotatedBack(heading, vector2_t(twistWorld.head<kHorizontalPositionDim>())),
                          .headingRate = headingTiltJacobian(orientation).row(0).transpose().dot(angularVelocityLocal)};
}

StrideMotionHistory::StrideMotionHistory(scalar_t retention) : retention_(retention) {}

void StrideMotionHistory::reset() {
  samples_.clear();
  firstSampleKept_ = false;
}

void StrideMotionHistory::addSample(scalar_t time,
                                    const vector2_t& positionXy,
                                    scalar_t heading,
                                    absl::FunctionRef<BasePlanarMotion()> seed) {
  if (!samples_.empty() && time == samples_.back().time) return;
  if (samples_.empty() || time < samples_.back().time || time - samples_.back().time > kMaxBridgedGap) {
    samples_.clear();
    samples_.push_back(Sample{.time = time, .heading = heading, .displacement = vector2_t::Zero()});
    lastPosition_ = positionXy;
    seed_ = seed();
    firstSampleKept_ = true;
    measuredSince_ = time;
    return;
  }
  const Sample& last = samples_.back();
  // A bridged gap: the samples are dense again from this one on.
  if (time - last.time > kMaxSampleGap) measuredSince_ = time;
  // The heading change since the last sample, the branch of |dPsi| <= pi: the true one below the heading rates the class
  // comment states, kMaxBridgedGap included.
  const scalar_t headingChange = std::remainder(heading - last.heading, 2.0 * M_PI);
  const scalar_t midHeading = last.heading + 0.5 * headingChange;
  // The chord of a constant body twist, dp = dt sinc(dPsi / 2) R_z(Psi_mid) v_h, solved for dD = v_h dt.
  const vector2_t displacementStep = rotatedBack(midHeading, vector2_t(positionXy - lastPosition_)) / sinc(0.5 * headingChange);
  samples_.push_back(Sample{.time = time, .heading = last.heading + headingChange, .displacement = last.displacement + displacementStep});
  lastPosition_ = positionXy;
  // Keep one sample at or before time - retention, for the interpolation at the start of the longest window.
  while (samples_.size() > 1 && samples_[1].time <= time - retention_) {
    samples_.pop_front();
    firstSampleKept_ = false;
  }
}

StrideMotionHistory::Sample StrideMotionHistory::at(scalar_t time) const {
  const Sample& first = samples_.front();
  if (time <= first.time) {
    // Before the first sample the base moves at the seed's constant motion; a pruned history is never asked there
    // (mean() cuts its window to the retention, which the kept samples cover), so its first sample is held.
    const scalar_t before = firstSampleKept_ ? time - first.time : 0.0;
    return Sample{.time = time,
                  .heading = first.heading + before * seed_.headingRate,
                  .displacement = first.displacement + before * seed_.velocityInHeadingFrame};
  }
  const std::deque<Sample>::const_iterator after =
      std::upper_bound(samples_.begin(), samples_.end(), time, [](scalar_t value, const Sample& sample) { return value < sample.time; });
  if (after == samples_.end()) return samples_.back();
  const Sample& previous = *(after - 1);
  const scalar_t fraction = (time - previous.time) / (after->time - previous.time);
  return Sample{.time = time,
                .heading = previous.heading + fraction * (after->heading - previous.heading),
                .displacement = previous.displacement + fraction * (after->displacement - previous.displacement)};
}

bool StrideMotionHistory::isMeasuredOver(scalar_t window) const {
  // measuredSince_ is at or after the first sample, so a window that starts at or after it reads no seed.
  return !samples_.empty() && samples_.back().time - std::min(window, retention_) >= measuredSince_;
}

BasePlanarMotion StrideMotionHistory::mean(scalar_t window) const {
  if (samples_.empty()) return BasePlanarMotion{};
  const Sample& last = samples_.back();
  const scalar_t span = std::min(window, retention_);
  const Sample start = at(last.time - span);
  return BasePlanarMotion{.velocityInHeadingFrame = (last.displacement - start.displacement) / span,
                          .headingRate = (last.heading - start.heading) / span};
}

}  // namespace ocs2::humanoid
