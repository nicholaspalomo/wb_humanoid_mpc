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

#include "humanoid_mpc_validation/closed_loop/ClosedLoopMetrics.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Geometry"

#include "humanoid_common_mpc/orientation/BaseOrientation.h"
#include "humanoid_mpc_validation/closed_loop/ClosedLoopMetricsSchema.h"

namespace ocs2::humanoid::validation {
namespace {

std::optional<double> meanOf(double sum, size_t count) {
  if (count == 0) return std::nullopt;
  return sum / static_cast<double>(count);
}

std::optional<double> rmsOf(double squaredSum, size_t count) {
  if (count == 0) return std::nullopt;
  return std::sqrt(squaredSum / static_cast<double>(count));
}

/** The nearest-rank percentile `percent` of sorted `values`. */
double percentile(const std::vector<double>& sorted, double percent) {
  const size_t rank = static_cast<size_t>(std::ceil(percent / 100.0 * static_cast<double>(sorted.size())));
  return sorted[std::min(sorted.size() - 1, rank == 0 ? 0 : rank - 1)];
}

}  // namespace

double wrapAngle(double angle) {
  const double wrapped = std::remainder(angle, 2.0 * M_PI);  // in [-pi, pi]
  return wrapped <= -M_PI ? wrapped + 2.0 * M_PI : wrapped;
}

JsonValue summarizeTimes(std::vector<double> values) {
  JsonValue summary = JsonValue::object();
  if (values.empty()) {
    summary.set("mean", JsonValue());
    summary.set("p50", JsonValue());
    summary.set("p99", JsonValue());
    summary.set("max", JsonValue());
    return summary;
  }
  std::sort(values.begin(), values.end());
  double sum = 0.0;
  for (const double value : values) sum += value;
  summary.set("mean", JsonValue::number(sum / static_cast<double>(values.size())));
  summary.set("p50", JsonValue::number(percentile(values, /*percent=*/50.0)));
  summary.set("p99", JsonValue::number(percentile(values, /*percent=*/99.0)));
  summary.set("max", JsonValue::number(values.back()));
  return summary;
}

void ClosedLoopMetrics::addControlCycle(const ControlCycleSample& sample) {
  if (numCycles_ == 0) firstTime_ = sample.time;
  lastTime_ = sample.time;
  ++numCycles_;

  const double height = sample.basePosition.z();
  heightSum_ += height;
  heightSquaredSum_ += height * height;
  const double heightError = height - sample.referenceBaseHeight;
  heightErrorSquaredSum_ += heightError * heightError;

  const vector4_t xi = sample.baseQuaternion;
  const double tilt = tiltVector(xi).norm();
  tiltSquaredSum_ += tilt * tilt;
  tiltMax_ = std::max(tiltMax_, tilt);

  // The base velocity in the heading frame, and the yaw rate in the world, against the reference.
  const double heading = twistHeading(xi);
  const double cosHeading = std::cos(heading);
  const double sinHeading = std::sin(heading);
  const Eigen::Vector3d& velocity = sample.baseLinearVelocityWorld;
  const Eigen::Vector2d headingVelocity(cosHeading * velocity.x() + sinHeading * velocity.y(),
                                        -sinHeading * velocity.x() + cosHeading * velocity.y());
  velocityErrorSquaredSum_ += (headingVelocity - sample.referenceVelocity.head<2>()).squaredNorm();
  Eigen::Quaterniond rotation;
  rotation.coeffs() = sample.baseQuaternion;
  const double yawRate = (rotation.normalized().toRotationMatrix() * sample.baseAngularVelocityLocal).z();
  const double yawRateError = yawRate - sample.referenceVelocity(2);
  yawRateErrorSquaredSum_ += yawRateError * yawRateError;

  // Heading unwrapped from the first sample on.
  if (haveHeading_) {
    cumulativeHeading_ += wrapAngle(heading - lastHeading_);
    cumulativeHeadingMax_ = std::max(cumulativeHeadingMax_, cumulativeHeading_);
    cumulativeHeadingMin_ = std::min(cumulativeHeadingMin_, cumulativeHeading_);
  }
  haveHeading_ = true;
  lastHeading_ = heading;

  // Stance-foot slip: the horizontal drift of a contact point from where it touched down, while it stays in contact.
  for (size_t contact = 0; contact < 2; ++contact) {
    const Eigen::Vector2d position = sample.contactPositions[contact].head<2>();
    StancePhase& phase = stance_[contact];
    if (sample.contactFlags[contact]) {
      if (!phase.active) {
        phase.active = true;
        phase.touchDownPosition = position;
        phase.maxDrift = 0.0;
      }
      phase.maxDrift = std::max(phase.maxDrift, (position - phase.touchDownPosition).norm());
    } else if (phase.active) {
      closeStancePhase(contact, lastContactPosition_[contact]);
    }
    lastContactPosition_[contact] = position;
  }

  for (Eigen::Index joint = 0; joint < sample.jointTorques.size(); ++joint) {
    torqueSquaredSum_ += sample.jointTorques(joint) * sample.jointTorques(joint);
  }
  torqueCount_ += static_cast<size_t>(sample.jointTorques.size());
  nonFiniteValues_ += sample.nonFiniteValues;
  if (!sample.mpcHealthy) ++unhealthyCycles_;
}

void ClosedLoopMetrics::closeStancePhase(size_t contact, const Eigen::Vector2d& position) {
  StancePhase& phase = stance_[contact];
  stanceFinalDrifts_.push_back((position - phase.touchDownPosition).norm());
  slipMax_ = std::max(slipMax_, phase.maxDrift);
  phase.active = false;
}

void ClosedLoopMetrics::addSolve(const SolveSample& sample) {
  wallTimesMs_.push_back(sample.wallTimeMs);
  lqApproximationMs_.push_back(sample.lqApproximationMs);
  solveQpMs_.push_back(sample.solveQpMs);
  linesearchMs_.push_back(sample.linesearchMs);
  computeControllerMs_.push_back(sample.computeControllerMs);
  if (!sample.succeeded) ++failedSolves_;
  if (sample.initialStateRotationGap.has_value() && (!maxRotationGap_.has_value() || *sample.initialStateRotationGap > *maxRotationGap_)) {
    maxRotationGap_ = sample.initialStateRotationGap;
    maxRotationGapTime_ = sample.time;
  }
  if (sample.initialStateGapNorm.has_value() && (!maxGapNorm_.has_value() || *sample.initialStateGapNorm > *maxGapNorm_)) {
    maxGapNorm_ = sample.initialStateGapNorm;
  }
  if (sample.quaternionNormDeviation.has_value() &&
      (!maxQuaternionNormDeviation_.has_value() || *sample.quaternionNormDeviation > *maxQuaternionNormDeviation_)) {
    maxQuaternionNormDeviation_ = sample.quaternionNormDeviation;
  }
}

void ClosedLoopMetrics::setFall(double time, std::string reason) {
  fallTime_ = time;
  fallReason_ = std::move(reason);
}

JsonValue ClosedLoopMetrics::report(const ClosedLoopRunInfo& info) const {
  // The stance phases still open at the end count as ended there.
  std::vector<double> finalDrifts = stanceFinalDrifts_;
  double slipMax = slipMax_;
  for (size_t contact = 0; contact < 2; ++contact) {
    if (!stance_[contact].active) continue;
    finalDrifts.push_back((lastContactPosition_[contact] - stance_[contact].touchDownPosition).norm());
    slipMax = std::max(slipMax, stance_[contact].maxDrift);
  }
  double finalDriftSquaredSum = 0.0;
  for (const double drift : finalDrifts) finalDriftSquaredSum += drift * drift;

  const std::optional<double> heightMean = meanOf(heightSum_, numCycles_);
  std::optional<double> heightStd;
  if (heightMean.has_value()) {
    heightStd = std::sqrt(std::max(0.0, heightSquaredSum_ / static_cast<double>(numCycles_) - *heightMean * *heightMean));
  }
  const bool sampled = numCycles_ > 0;

  // LINT.IfChange(metrics_document)
  JsonValue document = JsonValue::object();
  document.set("schema", JsonValue::string(std::string(kClosedLoopMetricsSchemaName)));
  document.set("label", JsonValue::string(info.label));
  document.set("robot", JsonValue::string(info.robot));
  document.set("formulation", JsonValue::string(info.formulation));
  document.set("scenario", JsonValue::string(info.scenario));
  document.set("provenance", info.provenance);
  document.set("settings", info.settings);

  JsonValue& evaluation = document.set("evaluation", JsonValue::object());
  evaluation.set("start_time_s", JsonValue::optionalNumber(sampled ? std::optional<double>(firstTime_) : std::nullopt));
  evaluation.set("end_time_s", JsonValue::optionalNumber(sampled ? std::optional<double>(lastTime_) : std::nullopt));
  evaluation.set("control_cycles", JsonValue::number(static_cast<double>(numCycles_)));
  evaluation.set("solves", JsonValue::number(static_cast<double>(wallTimesMs_.size())));

  JsonValue& survival = document.set("survival", JsonValue::object());
  survival.set("survived", JsonValue::boolean(survived()));
  survival.set("fall_time_s", JsonValue::optionalNumber(fallTime_));
  survival.set("fall_reason", JsonValue::string(fallReason_));

  JsonValue& baseHeight = document.set("base_height", JsonValue::object());
  baseHeight.set("mean_m", JsonValue::optionalNumber(heightMean));
  baseHeight.set("std_m", JsonValue::optionalNumber(heightStd));
  baseHeight.set("rms_error_m", JsonValue::optionalNumber(rmsOf(heightErrorSquaredSum_, numCycles_)));

  JsonValue& tilt = document.set("tilt", JsonValue::object());
  tilt.set("rms_rad", JsonValue::optionalNumber(rmsOf(tiltSquaredSum_, numCycles_)));
  tilt.set("max_rad", JsonValue::optionalNumber(sampled ? std::optional<double>(tiltMax_) : std::nullopt));

  document.set("velocity", JsonValue::object())
      .set("rms_error_mps", JsonValue::optionalNumber(rmsOf(velocityErrorSquaredSum_, numCycles_)));
  document.set("yaw_rate", JsonValue::object())
      .set("rms_error_radps", JsonValue::optionalNumber(rmsOf(yawRateErrorSquaredSum_, numCycles_)));

  JsonValue& heading = document.set("heading", JsonValue::object());
  heading.set("cumulative_final_rad", JsonValue::optionalNumber(sampled ? std::optional<double>(cumulativeHeading_) : std::nullopt));
  heading.set("cumulative_max_rad", JsonValue::optionalNumber(sampled ? std::optional<double>(cumulativeHeadingMax_) : std::nullopt));
  heading.set("cumulative_min_rad", JsonValue::optionalNumber(sampled ? std::optional<double>(cumulativeHeadingMin_) : std::nullopt));

  JsonValue& slip = document.set("stance_foot_slip", JsonValue::object());
  slip.set("max_m", JsonValue::optionalNumber(finalDrifts.empty() ? std::nullopt : std::optional<double>(slipMax)));
  slip.set("rms_m", JsonValue::optionalNumber(rmsOf(finalDriftSquaredSum, finalDrifts.size())));
  slip.set("stance_phases", JsonValue::number(static_cast<double>(finalDrifts.size())));

  document.set("joint_torque", JsonValue::object()).set("rms_nm", JsonValue::optionalNumber(rmsOf(torqueSquaredSum_, torqueCount_)));

  JsonValue& gap = document.set("initial_state_gap", JsonValue::object());
  gap.set("max_rotation_rad", JsonValue::optionalNumber(maxRotationGap_));
  gap.set("max_rotation_time_s",
          JsonValue::optionalNumber(maxRotationGap_.has_value() ? std::optional<double>(maxRotationGapTime_) : std::nullopt));
  gap.set("max_norm", JsonValue::optionalNumber(maxGapNorm_));

  document.set("quaternion_norm", JsonValue::object()).set("max_deviation", JsonValue::optionalNumber(maxQuaternionNormDeviation_));
  document.set("non_finite_values", JsonValue::number(static_cast<double>(nonFiniteValues_)));

  JsonValue& solveTime = document.set("solve_time_ms", JsonValue::object());
  solveTime.set("total", summarizeTimes(wallTimesMs_));
  solveTime.set("lq_approximation", summarizeTimes(lqApproximationMs_));
  solveTime.set("solve_qp", summarizeTimes(solveQpMs_));
  solveTime.set("linesearch", summarizeTimes(linesearchMs_));
  solveTime.set("compute_controller", summarizeTimes(computeControllerMs_));

  JsonValue& failures = document.set("failures", JsonValue::object());
  failures.set("failed_solves", JsonValue::number(static_cast<double>(failedSolves_)));
  failures.set("resets_served", JsonValue::number(static_cast<double>(counters_.resetsServed)));
  failures.set("full_resets_served", JsonValue::number(static_cast<double>(counters_.fullResetsServed)));
  failures.set("simulator_resets", JsonValue::number(static_cast<double>(counters_.simulatorResets)));
  failures.set("unhealthy_cycles", JsonValue::number(static_cast<double>(unhealthyCycles_)));
  // LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_validation/src/closed_loop/ClosedLoopMetricsSchema.cpp:metrics_schema)
  return document;
}

}  // namespace ocs2::humanoid::validation
