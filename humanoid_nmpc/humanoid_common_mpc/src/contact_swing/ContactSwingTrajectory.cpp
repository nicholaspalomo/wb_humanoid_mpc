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

#include "humanoid_common_mpc/contact_swing/ContactSwingTrajectory.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace ocs2::humanoid {
namespace {

/** How far from 1 the norm of an endpoint's normal may be. */
constexpr scalar_t kUnitNormalTolerance = 1.0e-6;
/** The degree of the Bezier arc: the factor of its derivative. */
constexpr scalar_t kBezierDegree = 3.0;
/** The smoothstep s = 3 sigma^2 - 2 sigma^3 and its derivative 6 sigma (1 -
 * sigma). */
constexpr scalar_t kSmoothstepSquareFactor = 3.0;
constexpr scalar_t kSmoothstepCubeFactor = 2.0;
constexpr scalar_t kSmoothstepRateFactor = 6.0;
/** The control points of the arc, by role. */
constexpr size_t kLiftOffPoint = 0;
constexpr size_t kLiftControlPoint = 1;
constexpr size_t kTouchDownControlPoint = 2;
constexpr size_t kTouchDownPoint = 3;
static_assert(kTouchDownPoint + 1 == ContactSwingTrajectory::kNumControlPoints);

/** Whether every coefficient of `vector` is finite. */
bool isFinite(const vector3_t& vector) {
  return vector.allFinite();
}

/** InvalidArgument naming `what` unless `endpoint` is finite with a unit
 * normal. */
absl::Status checkEndpoint(const ContactSwingEndpoint& endpoint,
                           absl::string_view what) {
  if (!isFinite(endpoint.posContactRtWorldEwrtWorld) ||
      !isFinite(endpoint.normal)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "[ContactSwingTrajectory] the ", what, " point is not finite."));
  }
  if (std::abs(endpoint.normal.norm() - 1.0) > kUnitNormalTolerance) {
    return absl::InvalidArgumentError(absl::StrCat(
        "[ContactSwingTrajectory] the ", what, " normal has length ",
        endpoint.normal.norm(), ", but it must be a unit vector."));
  }
  return absl::OkStatus();
}

/**
 * The clearance the arc must keep at the arc parameter `s`: `clearance`,
 * ramped linearly from zero at either end over
 * ContactSwingTrajectory::kClearanceRampFraction.
 */
scalar_t requiredClearance(scalar_t s, scalar_t clearance) {
  const scalar_t ramp =
      std::min({1.0, s / ContactSwingTrajectory::kClearanceRampFraction,
                (1.0 - s) / ContactSwingTrajectory::kClearanceRampFraction});
  return clearance * ramp;
}

}  // namespace

absl::StatusOr<ContactSwingTrajectory> ContactSwingTrajectory::Create(
    scalar_t liftOffTime,
    scalar_t touchDownTime,
    const ContactSwingEndpoint& liftOff,
    const ContactSwingEndpoint& touchDown,
    const ContactSwingSettings& settings,
    absl::FunctionRef<scalar_t(const vector3_t&)> signedDistance) {
  if (!std::isfinite(liftOffTime) || !std::isfinite(touchDownTime) ||
      !(touchDownTime > liftOffTime)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "[ContactSwingTrajectory] the touch-down at ", touchDownTime,
        " s must be after the lift-off at ", liftOffTime, " s."));
  }
  if (absl::Status status = checkEndpoint(liftOff, "lift-off"); !status.ok()) {
    return status;
  }
  if (absl::Status status = checkEndpoint(touchDown, "touch-down");
      !status.ok()) {
    return status;
  }
  if (!std::isfinite(settings.liftHeight) || !(settings.liftHeight > 0.0)) {
    return absl::InvalidArgumentError(
        absl::StrCat("[ContactSwingTrajectory] the lift height is ",
                     settings.liftHeight, " m, but it must be positive."));
  }
  if (!std::isfinite(settings.clearance) || settings.clearance < 0.0) {
    return absl::InvalidArgumentError(
        absl::StrCat("[ContactSwingTrajectory] the clearance is ",
                     settings.clearance, " m, but it must not be negative."));
  }

  ContactSwingTrajectory trajectory;
  trajectory.liftOffTime_ = liftOffTime;
  trajectory.touchDownTime_ = touchDownTime;
  const scalar_t largestControlDistance =
      kMaxControlDistanceGrowth * settings.liftHeight;
  for (scalar_t controlDistance = settings.liftHeight;;
       controlDistance *= kControlDistanceGrowth) {
    controlDistance = std::min(controlDistance, largestControlDistance);
    trajectory.setControlPoints(liftOff, touchDown, controlDistance);
    scalar_t margin = std::numeric_limits<scalar_t>::infinity();
    for (int sample = 1; sample < kClearanceSamples; ++sample) {
      const scalar_t s = static_cast<scalar_t>(sample) /
                         static_cast<scalar_t>(kClearanceSamples);
      margin = std::min(margin, signedDistance(trajectory.pointAt(s)) -
                                    requiredClearance(s, settings.clearance));
    }
    trajectory.clearanceMargin_ = margin;
    trajectory.clearanceSatisfied_ = margin >= 0.0;
    if (trajectory.clearanceSatisfied_ ||
        controlDistance >= largestControlDistance) {
      break;
    }
  }
  return trajectory;
}

void ContactSwingTrajectory::setControlPoints(
    const ContactSwingEndpoint& liftOff,
    const ContactSwingEndpoint& touchDown,
    scalar_t controlDistance) {
  controlDistance_ = controlDistance;
  controlPoints_[kLiftOffPoint] = liftOff.posContactRtWorldEwrtWorld;
  controlPoints_[kLiftControlPoint] =
      liftOff.posContactRtWorldEwrtWorld + controlDistance * liftOff.normal;
  controlPoints_[kTouchDownControlPoint] =
      touchDown.posContactRtWorldEwrtWorld + controlDistance * touchDown.normal;
  controlPoints_[kTouchDownPoint] = touchDown.posContactRtWorldEwrtWorld;
}

vector3_t ContactSwingTrajectory::pointAt(scalar_t s) const {
  const scalar_t r = 1.0 - s;
  return r * r * r * controlPoints_[kLiftOffPoint] +
         kBezierDegree * r * r * s * controlPoints_[kLiftControlPoint] +
         kBezierDegree * r * s * s * controlPoints_[kTouchDownControlPoint] +
         s * s * s * controlPoints_[kTouchDownPoint];
}

vector3_t ContactSwingTrajectory::tangentAt(scalar_t s) const {
  const scalar_t r = 1.0 - s;
  return kBezierDegree * (r * r *
                              (controlPoints_[kLiftControlPoint] -
                               controlPoints_[kLiftOffPoint]) +
                          2.0 * r * s *
                              (controlPoints_[kTouchDownControlPoint] -
                               controlPoints_[kLiftControlPoint]) +
                          s * s *
                              (controlPoints_[kTouchDownPoint] -
                               controlPoints_[kTouchDownControlPoint]));
}

ContactSwingReference ContactSwingTrajectory::reference(scalar_t time) const {
  const scalar_t duration = touchDownTime_ - liftOffTime_;
  const scalar_t sigma = std::clamp((time - liftOffTime_) / duration, 0.0, 1.0);
  const scalar_t s =
      sigma * sigma * (kSmoothstepSquareFactor - kSmoothstepCubeFactor * sigma);
  const scalar_t sRate =
      kSmoothstepRateFactor * sigma * (1.0 - sigma) / duration;
  return ContactSwingReference{
      .posContactRtWorldEwrtWorld = pointAt(s),
      .linvelContactRtWorldEwrtWorld = sRate * tangentAt(s)};
}

}  // namespace ocs2::humanoid
