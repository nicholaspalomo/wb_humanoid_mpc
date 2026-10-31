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
#include <cstddef>

#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"

#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid {

/** Where a swing starts or ends: a point on a surface and the surface's
 * outward unit normal there, in the world frame. Passive data. */
struct ContactSwingEndpoint {
  vector3_t posContactRtWorldEwrtWorld = vector3_t::Zero();  // [m]
  vector3_t normal = vector3_t::UnitZ();
};

/** The reference of a swinging contact at one instant, in the world frame.
 * Passive data. */
struct ContactSwingReference {
  vector3_t posContactRtWorldEwrtWorld = vector3_t::Zero();     // [m]
  vector3_t linvelContactRtWorldEwrtWorld = vector3_t::Zero();  // [m/s]
};

/** How a swing is shaped (the task file's multi_contact_plan.swing). Passive
 * data. */
struct ContactSwingSettings {
  /** The defaults of the fields below. */
  static constexpr scalar_t kDefaultLiftHeight = 0.1;  // [m]
  static constexpr scalar_t kDefaultClearance = 0.05;  // [m]

  // [m] Positive: how far along the two normals the inner control points of
  // the arc start.
  scalar_t liftHeight = kDefaultLiftHeight;
  // [m] Not negative: the signed distance to the terrain every point of the
  // arc away from its ends keeps.
  scalar_t clearance = kDefaultClearance;
};

/**
 * One swing of a contact from a lift-off point to a touch-down point (design
 * section 7.2, humanoid_nmpc/docs/multi_contact_planning/README.md): a cubic
 * Bezier arc
 *
 *   B(s) = (1-s)^3 P0 + 3 (1-s)^2 s P1 + 3 (1-s) s^2 P2 + s^3 P3,
 *   P0 = p_lift, P1 = p_lift + a n_lift, P2 = p_touch + a n_touch,
 *   P3 = p_touch,
 *
 * whose tangent at lift-off is along the normal n_lift of the surface it
 * leaves and at touch-down along -n_touch, into the target surface, timed by
 * s = 3 sigma^2 - 2 sigma^3 of the normalized time sigma, so that it starts
 * and ends at rest. Create() raises the control distance a from the settings'
 * lift height until every sampled point of the arc keeps the terrain's signed
 * distance at least the clearance, ramped down linearly to zero over the first
 * and the last quarter of the arc, where it must meet the surfaces.
 *
 * A value type, immutable after Create(): every method is const and safe to
 * call concurrently.
 */
class ContactSwingTrajectory {
 public:
  /**
   * The swing from `liftOff` at `liftOffTime` [s] to `touchDown` at
   * `touchDownTime`, clear of the terrain whose signed distance [m] at a world
   * point is `signedDistance` (read only within Create()). When no control
   * distance up to kMaxControlDistanceGrowth times the lift height clears the
   * terrain, the swing of the largest is returned and clearanceSatisfied() is
   * false.
   *
   * @return InvalidArgument for a touch-down not after the lift-off, a value
   * that is not finite, a normal that is not unit (within 1e-6), a lift height
   * that is not positive or a negative clearance.
   */
  static absl::StatusOr<ContactSwingTrajectory> Create(
      scalar_t liftOffTime,
      scalar_t touchDownTime,
      const ContactSwingEndpoint& liftOff,
      const ContactSwingEndpoint& touchDown,
      const ContactSwingSettings& settings,
      absl::FunctionRef<scalar_t(const vector3_t&)> signedDistance);

  ContactSwingTrajectory(const ContactSwingTrajectory&) = default;
  ContactSwingTrajectory& operator=(const ContactSwingTrajectory&) = default;
  ContactSwingTrajectory(ContactSwingTrajectory&&) noexcept = default;
  ContactSwingTrajectory& operator=(ContactSwingTrajectory&&) noexcept =
      default;
  ~ContactSwingTrajectory() = default;

  /** The reference at `time` [s], clamped to the swing: the lift-off point
   * before it, the touch-down point after it, at rest at both. */
  ContactSwingReference reference(scalar_t time) const;

  /** Whether `time` lies in [liftOffTime(), touchDownTime()]. */
  bool contains(scalar_t time) const {
    return time >= liftOffTime_ && time <= touchDownTime_;
  }

  scalar_t liftOffTime() const { return liftOffTime_; }
  scalar_t touchDownTime() const { return touchDownTime_; }
  /** [m] The control distance a the arc was built with. */
  scalar_t controlDistance() const { return controlDistance_; }
  /** Whether every sampled point keeps the required clearance. */
  bool clearanceSatisfied() const { return clearanceSatisfied_; }
  /** [m] The smallest excess of the sampled points' signed distance over the
   * required clearance: negative when clearanceSatisfied() is false. */
  scalar_t clearanceMargin() const { return clearanceMargin_; }

  /** The point of the arc at the arc parameter `s` in [0, 1]. */
  vector3_t pointAt(scalar_t s) const;
  /** dB/ds at the arc parameter `s`. */
  vector3_t tangentAt(scalar_t s) const;

  /** The interior samples of the arc parameter the clearance is checked at:
   * s = k / kClearanceSamples for k = 1 .. kClearanceSamples - 1. */
  static constexpr int kClearanceSamples = 32;
  /** The fraction of the arc at each end over which the required clearance
   * ramps from zero to the full clearance. */
  static constexpr scalar_t kClearanceRampFraction = 0.25;
  /** The factor the control distance grows by per attempt, and the largest
   * multiple of the lift height it reaches. */
  static constexpr scalar_t kControlDistanceGrowth = 1.25;
  static constexpr scalar_t kMaxControlDistanceGrowth = 40.0;
  /** The control points of a cubic Bezier arc. */
  static constexpr size_t kNumControlPoints = 4;

 private:
  ContactSwingTrajectory() = default;

  /** Sets the control points for the control distance `controlDistance`. */
  void setControlPoints(const ContactSwingEndpoint& liftOff,
                        const ContactSwingEndpoint& touchDown,
                        scalar_t controlDistance);

  scalar_t liftOffTime_ = 0.0;
  scalar_t touchDownTime_ = 0.0;  // > liftOffTime_
  std::array<vector3_t, kNumControlPoints> controlPoints_;
  scalar_t controlDistance_ = 0.0;
  bool clearanceSatisfied_ = false;
  scalar_t clearanceMargin_ = 0.0;
};

}  // namespace ocs2::humanoid
