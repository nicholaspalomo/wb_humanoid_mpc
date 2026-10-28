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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include "humanoid_multi_contact/planner/PlanTiming.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "ocs2_core/Dimensions.h"
#include "pinocchio/algorithm/joint-configuration.hpp"

#include "humanoid_common_mpc/common/StatusMacros.h"

namespace ocs2::humanoid::multi_contact {
namespace {

// The coefficients of the minimum-jerk profile 10 tau^3 - 15 tau^4 + 6 tau^5.
constexpr double kCubic = 10.0;
constexpr double kQuartic = 15.0;
constexpr double kQuintic = 6.0;
// Its peak speed, ds/dtau at tau = 1/2, and its peak acceleration,
// d2s/dtau2 at tau = 1/2 - sqrt(3)/6.
constexpr double kPeakSpeed = 1.875;
constexpr double kPeakAcceleration = 5.773502691896258;
// Bisections of inverseMinimumJerk(): 2^-50 of the unit interval.
constexpr int kBisections = 50;
// [s] Below this, a segment's "time at the speed limits" is none.
constexpr double kNoMotion = 1.0e-9;
// [s] The shortest interval between two knots: the times must increase.
constexpr double kMinKnotInterval = 1.0e-3;

/** The time one interval needs at the speed limits. */
double intervalTime(const RobotStanceModel& model,
                    const Eigen::VectorXd& from,
                    const Eigen::VectorXd& to,
                    const Eigen::Vector3d& comFrom,
                    const Eigen::Vector3d& comTo,
                    const std::vector<Eigen::Vector3d>& padsFrom,
                    const std::vector<Eigen::Vector3d>& padsTo,
                    const TimingOptions& options) {
  const Eigen::VectorXd tangent =
      pinocchio::difference(model.model(), from, to);
  double time = (comTo - comFrom).norm() / options.maxComSpeed;
  time = std::max(time,
                  (to.head<kPositionDim>() - from.head<kPositionDim>()).norm() /
                      options.maxBaseSpeed);
  time =
      std::max(time, tangent.segment<kRotationTangentDim>(kPositionDim).norm() /
                         options.maxAngularSpeed);
  time = std::max(
      time,
      tangent.tail(tangent.size() - kPoseTangentDim).lpNorm<Eigen::Infinity>() /
          options.maxAngularSpeed);
  for (size_t pad = 0; pad < padsFrom.size(); ++pad) {
    time = std::max(time,
                    (padsTo[pad] - padsFrom[pad]).norm() / options.maxPadSpeed);
  }
  return time;
}

}  // namespace

double minimumJerk(double tau) {
  const double t = std::clamp(tau, 0.0, 1.0);
  const double cube = t * t * t;
  return cube * (kCubic - kQuartic * t + kQuintic * t * t);
}

double inverseMinimumJerk(double s) {
  // The ends exactly: the profile is flat there, so a bisection would leave
  // the cube root of its rounding.
  if (!(s > 0.0)) return 0.0;
  if (!(s < 1.0)) return 1.0;
  const double target = s;
  double low = 0.0;
  double high = 1.0;
  for (int i = 0; i < kBisections; ++i) {
    const double middle = 0.5 * (low + high);
    if (minimumJerk(middle) < target) {
      low = middle;
    } else {
      high = middle;
    }
  }
  return 0.5 * (low + high);
}

absl::StatusOr<std::vector<double>> segmentKnotTimes(
    RobotStanceModel& model,
    const MotionSegment& segment,
    const TimingOptions& options,
    double slowdown) {
  if (segment.knots.empty()) {
    return absl::InvalidArgumentError("a motion segment has no knot");
  }
  const int count = static_cast<int>(segment.knots.size());
  std::vector<Eigen::Vector3d> coms(count);
  std::vector<std::vector<Eigen::Vector3d>> pads(count);
  for (int k = 0; k < count; ++k) {
    RETURN_IF_ERROR(model.setGeneralizedPosition(segment.knots[k].q));
    coms[k] = model.centerOfMass();
    for (int pad = 0; pad < model.contactSet().size(); ++pad) {
      pads[k].push_back(model.padPosition(pad));
    }
  }
  std::vector<double> cumulative = {0.0};
  double comPath = 0.0;
  for (int k = 1; k < count; ++k) {
    cumulative.push_back(cumulative.back() +
                         intervalTime(model, segment.knots[k - 1].q,
                                      segment.knots[k].q, coms[k - 1], coms[k],
                                      pads[k - 1], pads[k], options));
    comPath += (coms[k] - coms[k - 1]).norm();
  }
  const double atLimits = cumulative.back();
  const double duration =
      std::max(slowdown, 1.0) *
      std::max({options.minSegmentDuration, kPeakSpeed * atLimits,
                std::sqrt(kPeakAcceleration * comPath /
                          options.maxComAcceleration)});
  std::vector<double> times;
  for (int k = 0; k < count; ++k) {
    const double u = atLimits < kNoMotion
                         ? static_cast<double>(k) / std::max(1, count - 1)
                         : cumulative[k] / atLimits;
    times.push_back(duration * inverseMinimumJerk(u));
    if (k > 0 && !(times[k] > times[k - 1])) {
      times[k] = times[k - 1] + kMinKnotInterval;
    }
  }
  return times;
}

namespace {

/** Appends one knot of `q` at `time` with velocity `v`. */
absl::Status appendKnot(RobotStanceModel& model,
                        const MotionKnot& knot,
                        double time,
                        const Eigen::VectorXd& v,
                        std::vector<config::MultiContactPlan::Knot>& knots) {
  ASSIGN_OR_RETURN(const CentroidalMomentum momentum,
                   model.centroidalMomentum(knot.q, v));
  config::MultiContactPlan::Knot planKnot;
  planKnot.time = time;
  planKnot.base_position = knot.q.head<kPositionDim>();
  planKnot.base_orientation_xyzw =
      knot.q.segment<kQuaternionCoefficients>(kPositionDim);
  const std::vector<ActuatedJoint>& joints = model.actuatedJoints();
  planKnot.joint_positions.resize(static_cast<Eigen::Index>(joints.size()));
  planKnot.joint_velocities.resize(static_cast<Eigen::Index>(joints.size()));
  for (size_t j = 0; j < joints.size(); ++j) {
    planKnot.joint_positions(static_cast<Eigen::Index>(j)) =
        knot.q(joints[j].positionIndex);
    planKnot.joint_velocities(static_cast<Eigen::Index>(j)) =
        v(joints[j].velocityIndex);
  }
  planKnot.linear_momentum = momentum.linear;
  planKnot.angular_momentum = momentum.angular;
  planKnot.active_contacts = knot.mask;
  knots.push_back(std::move(planKnot));
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::vector<config::MultiContactPlan::Knot>> timedPlanKnots(
    RobotStanceModel& model,
    const std::vector<MotionSegment>& segments,
    const TimingOptions& options,
    absl::Span<const double> slowdowns) {
  if (slowdowns.size() != segments.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("the plan's timing needs one slowdown per segment, ",
                     segments.size(), ", not ", slowdowns.size()));
  }
  std::vector<config::MultiContactPlan::Knot> knots;
  const Eigen::VectorXd rest = Eigen::VectorXd::Zero(model.model().nv);
  double start = 0.0;
  for (size_t index = 0; index < segments.size(); ++index) {
    const MotionSegment& segment = segments[index];
    const double slowdown = slowdowns[index];
    ASSIGN_OR_RETURN(const std::vector<double> times,
                     segmentKnotTimes(model, segment, options, slowdown));
    if (index > 0) start += std::max(slowdown, 1.0) * options.dwell;
    const int count = static_cast<int>(segment.knots.size());
    for (int k = 0; k < count; ++k) {
      Eigen::VectorXd v = rest;
      if (k > 0 && k + 1 < count) {
        v = pinocchio::difference(model.model(), segment.knots[k - 1].q,
                                  segment.knots[k + 1].q) /
            (times[k + 1] - times[k - 1]);
      }
      RETURN_IF_ERROR(
          appendKnot(model, segment.knots[k], start + times[k], v, knots));
    }
    start += times.back();
  }
  if (!knots.empty()) {
    config::MultiContactPlan::Knot hold = knots.back();
    hold.time += std::max(slowdowns.back(), 1.0) * options.dwell;
    knots.push_back(std::move(hold));
  }
  return knots;
}

}  // namespace ocs2::humanoid::multi_contact
