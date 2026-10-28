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

#include <vector>

#include "Eigen/Core"
#include "absl/status/statusor.h"
#include "absl/types/span.h"

#include "humanoid_multi_contact/contact_set/ContactSet.h"
#include "humanoid_multi_contact/plan/multi_contact_plan.nproto.h"
#include "humanoid_multi_contact/stability/RobotStanceModel.h"

namespace ocs2::humanoid::multi_contact {

/** The defaults of TimingOptions; its fields say what each is. */
inline constexpr double kDefaultMaxComSpeed = 0.15;
inline constexpr double kDefaultMaxComAcceleration = 0.15;
inline constexpr double kDefaultMaxBaseSpeed = 0.2;
inline constexpr double kDefaultMaxPadSpeed = 0.3;
inline constexpr double kDefaultMaxAngularSpeed = 0.5;
inline constexpr double kDefaultMinSegmentDuration = 0.8;
inline constexpr double kDefaultDwell = 0.4;

/** The speed limits of the timing law (design section 5, "timing"). */
struct TimingOptions {
  // [m/s, m/s^2] The CoM's speed and acceleration.
  double maxComSpeed = kDefaultMaxComSpeed;
  double maxComAcceleration = kDefaultMaxComAcceleration;
  // [m/s] The base's and any pad's speed; [rad/s] the base's turn and any
  // joint's speed.
  double maxBaseSpeed = kDefaultMaxBaseSpeed;
  double maxPadSpeed = kDefaultMaxPadSpeed;
  double maxAngularSpeed = kDefaultMaxAngularSpeed;
  // [s] The shortest segment, and the hold at rest between two segments, in
  // which a contact is loaded or unloaded.
  double minSegmentDuration = kDefaultMinSegmentDuration;
  double dwell = kDefaultDwell;
};

/** One knot of a motion: a generalized position and its contact mode.
 * Passive data. */
struct MotionKnot {
  Eigen::VectorXd q;
  ContactMask mask = 0;
};

/** A motion from rest to rest: its first knot is where the previous segment
 * ended. Passive data. */
struct MotionSegment {
  std::vector<MotionKnot> knots;
};

/** s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5, the minimum-jerk profile from rest
 * to rest over tau in [0, 1]. */
double minimumJerk(double tau);

/** The tau in [0, 1] with minimumJerk(tau) = s, s clamped into [0, 1]. */
double inverseMinimumJerk(double s);

/**
 * The times of a segment's knots from its start, by the timing law: the path
 * parameter u of a knot is the fraction of the segment's "time at the speed
 * limits" (each interval's largest displacement of the CoM, the base, a pad or
 * a joint over its speed limit), and u(t) follows the minimum-jerk profile over
 * the segment's duration T, the longest of
 *  - 1.875 times the time at the speed limits (the profile's peak speed),
 *  - sqrt(5.774 L / a) for the CoM's path length L (its peak acceleration),
 *  - the shortest segment,
 * all scaled by `slowdown` (at least 1). The motion starts and ends at rest.
 * Sets `model`'s configuration.
 */
absl::StatusOr<std::vector<double>> segmentKnotTimes(
    RobotStanceModel& model,
    const MotionSegment& segment,
    const TimingOptions& options,
    double slowdown);

/**
 * The plan's knots of consecutive segments: each segment's knots at
 * segmentKnotTimes() with its entry of `slowdowns`, a segment's first knot
 * repeated `dwell` after the previous segment's last (the hold in which a
 * contact is loaded or unloaded), and a final hold of `dwell`, each dwell
 * scaled by the slowdown of the segment after it (the final one by the last
 * segment's). Joint positions and velocities are those of the model's actuated
 * joints (RobotStanceModel::actuatedJoints()), the velocities central
 * differences of the positions, zero at a segment's ends; the momenta are
 * A_G(q) v. Contact wrenches are left empty. InvalidArgument for an empty
 * segment or a `slowdowns` of another size than `segments`; sets `model`'s
 * configuration.
 */
absl::StatusOr<std::vector<config::MultiContactPlan::Knot>> timedPlanKnots(
    RobotStanceModel& model,
    const std::vector<MotionSegment>& segments,
    const TimingOptions& options,
    absl::Span<const double> slowdowns);

}  // namespace ocs2::humanoid::multi_contact
