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

#include "pinocchio/fwd.hpp"

#include "humanoid_nmpc/humanoid_common_mpc/test/equivalence/TermProbePoints.h"

#include <array>
#include <cmath>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "ocs2_core/ComputationRequest.h"
#include "ocs2_core/constraint/StateConstraint.h"
#include "ocs2_core/constraint/StateInputConstraint.h"
#include "ocs2_core/penalties/penalties/PieceWisePolynomialBarrierPenalty.h"
#include "ocs2_core/penalties/penalties/QuadraticPenalty.h"
#include "ocs2_core/soft_constraint/StateInputSoftConstraint.h"
#include "ocs2_core/soft_constraint/StateSoftConstraint.h"

#include "humanoid_common_mpc/common/CostTermNames.h"
#include "humanoid_common_mpc/constraint/BasisScalingNonNegativityConstraint.h"
#include "humanoid_common_mpc/constraint/JointLimitsSoftConstraint.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/orientation/EulerBoundary.h"
#include "humanoid_common_mpc/pinocchio_model/pinocchioUtils.h"
#include "humanoid_mpc_validation/chart/EulerChart.h"

namespace ocs2::humanoid {
namespace {

/** [s] The time of the points in the first double support of the walking reference. */
constexpr scalar_t kDoubleSupportTime = 0.05;
/** [input units] Every entry of a point's input is the weight-compensating input's plus a uniform draw within this. */
constexpr scalar_t kInputPerturbation = 0.1;
/** [rad/s] Each ZYX Euler rate of a whole-body point (but the initial one) is drawn within this; norm below 1 rad/s. */
constexpr scalar_t kMaxAbsEulerRate = 0.5;
/** [m/s] The forward velocity of the mid-swing points (the normalized linear momentum on the centroidal state). */
constexpr scalar_t kMidSwingForwardVelocity = 0.5;

/** A point's name and base attitude (yaw, pitch, roll). Passive data. */
struct NamedAttitude {
  std::string name;
  vector3_t angles;
};

/** A point's name and time. Passive data. */
struct NamedTime {
  std::string name;
  scalar_t time = 0.0;
};

/** A point at rest: its name and, when set, the pitch of its attitude at a yaw of 0.7 rad. Passive data. */
struct RestAttitude {
  std::string name;
  std::optional<scalar_t> pitch;
};

/** `low + (high - low) U` for a uniform draw U of `generator`. */
scalar_t uniform(scalar_t low, scalar_t high, std::mt19937& generator) {
  std::uniform_real_distribution<scalar_t> distribution(low, high);
  return distribution(generator);
}

/** `eulerState` with its base angles (yaw, pitch, roll) set to `angles` and, on the whole body, Euler rates drawn. */
vector_t withAttitude(const MpcEquivalenceFixture& mpc, vector_t eulerState, const vector3_t& angles, std::mt19937& generator) {
  Eigen::Index row = 0;
  for (const validation::EulerStateSegment& segment : mpc.eulerLayout().segments) {
    if (segment.kind == validation::EulerStateSegmentKind::kEulerZyxAngles) eulerState.segment<3>(row) = angles;
    if (segment.kind == validation::EulerStateSegmentKind::kEulerZyxRates) {
      for (Eigen::Index i = 0; i < 3; ++i) eulerState(row + i) = uniform(-kMaxAbsEulerRate, kMaxAbsEulerRate, generator);
    }
    row += static_cast<Eigen::Index>(segment.dim);
  }
  return eulerState;
}

/**
 * `eulerState` perturbed as a random point: a heading anywhere, pitch and roll within 0.2 rad (a tilt of up to 0.28
 * rad), the base position within 2 cm, the joint angles within 0.05 rad, the momenta and velocities within 0.2 and, on
 * the whole body, Euler rates drawn.
 */
vector_t randomlyPerturbed(const MpcEquivalenceFixture& mpc, vector_t eulerState, std::mt19937& generator) {
  Eigen::Index row = 0;
  for (const validation::EulerStateSegment& segment : mpc.eulerLayout().segments) {
    const Eigen::Index dim = static_cast<Eigen::Index>(segment.dim);
    if (segment.kind == validation::EulerStateSegmentKind::kEulerZyxAngles) {
      eulerState.segment<3>(row) << uniform(-M_PI, M_PI, generator), uniform(/*low=*/-0.2, /*high=*/0.2, generator),
          uniform(/*low=*/-0.2, /*high=*/0.2, generator);
    } else if (segment.kind == validation::EulerStateSegmentKind::kEulerZyxRates) {
      for (Eigen::Index i = 0; i < dim; ++i) eulerState(row + i) = uniform(-kMaxAbsEulerRate, kMaxAbsEulerRate, generator);
    } else {
      scalar_t halfWidth = 0.2;
      if (segment.name == "base_position") halfWidth = 0.02;
      if (segment.name == "joint_positions") halfWidth = 0.05;
      for (Eigen::Index i = 0; i < dim; ++i) eulerState(row + i) += uniform(-halfWidth, halfWidth, generator);
    }
    row += dim;
  }
  return eulerState;
}

/** `eulerState` moving forward along its base's x axis at kMidSwingForwardVelocity. */
vector_t walkingForward(const MpcEquivalenceFixture& mpc, vector_t eulerState) {
  Eigen::Index row = 0;
  for (const validation::EulerStateSegment& segment : mpc.eulerLayout().segments) {
    if (segment.name == "normalized_linear_momentum" || segment.name == "base_linear_velocity") {
      eulerState(row) = kMidSwingForwardVelocity;
    }
    row += static_cast<Eigen::Index>(segment.dim);
  }
  return eulerState;
}

/**
 * The point `name` at `time`: the state Phi(`eulerState`), the input that carries the weight on the feet in stance at
 * `time` under `reference`, each entry perturbed within kInputPerturbation.
 */
TermProbePoint probePoint(const MpcEquivalenceFixture& mpc,
                          const EquivalenceReference& reference,
                          std::string name,
                          scalar_t time,
                          const vector_t& eulerState,
                          std::mt19937& generator) {
  TermProbePoint point{
      .name = std::move(name), .time = time, .state = mpc.quaternionStateOf(eulerState), .input = vector_t(), .atRest = false};
  point.input = mpc.weightCompensatingInput(point.state, modeNumber2StanceLeg(reference.modeSchedule.modeAtTime(time)));
  for (Eigen::Index i = 0; i < point.input.size(); ++i) point.input(i) += uniform(-kInputPerturbation, kInputPerturbation, generator);
  return point;
}

/** The initial point at rest - its momentum zero, its joint velocities set to zero - level and at gimbal lock. */
void appendRestingPoints(const MpcEquivalenceFixture& mpc, std::vector<TermProbePoint>& probes) {
  const scalar_t restTime = probes.front().time;
  vector_t restState = probes.front().state;
  vector_t restInput = probes.front().input;
  const vector_t zeroJointVelocities = vector_t::Zero(static_cast<Eigen::Index>(mpc.model().getJointDim()));
  mpc.model().setJointVelocities(restState, restInput, zeroJointVelocities);
  const std::vector<RestAttitude> restAttitudes = {{.name = "initial_at_rest", .pitch = std::nullopt},
                                                   {.name = "gimbal_lock_pitch_p90deg_at_rest", .pitch = 0.5 * M_PI},
                                                   {.name = "gimbal_lock_pitch_m90deg_at_rest", .pitch = -0.5 * M_PI}};
  for (const RestAttitude& attitude : restAttitudes) {
    vector_t state = restState;
    if (attitude.pitch.has_value()) mpc.model().setBaseOrientation(state, quaternionFromEulerZyx(vector3_t(0.7, *attitude.pitch, 0.0)));
    probes.push_back(
        TermProbePoint{.name = attitude.name, .time = restTime, .state = std::move(state), .input = restInput, .atRest = true});
  }
}

/** The labels `<name>.<term>` of `collection`'s terms, sorted by term name. */
template <typename CollectionT>
std::vector<std::string> collectionTermLabels(absl::string_view name, const CollectionT& collection) {
  std::vector<std::string> labels;
  for (const std::string& term : sortedTermNames(collection)) labels.push_back(absl::StrCat(name, ".", term));
  return labels;
}

/** The labels `<name>.<term>` of the terms of `collection` that are a `TermT`. */
template <typename TermT, typename CollectionT>
std::vector<std::string> labelsOfClass(absl::string_view name, CollectionT& collection) {
  std::vector<std::string> labels;
  for (const std::string& term : sortedTermNames(collection)) {
    if (dynamic_cast<const TermT*>(&collection.get(term)) != nullptr) labels.push_back(absl::StrCat(name, ".", term));
  }
  return labels;
}

/** `initial` moved into the bands of the joint-limit and the basis-scaling barriers of `problem` (barrierBandPoints()). */
TargetedProbePoint barrierBandPoint(OptimalControlProblem& problem, const MpcEquivalenceFixture& mpc, const TermProbePoint& initial) {
  TargetedProbePoint band{.point = initial, .labels = {}};
  TermProbePoint& point = band.point;
  point.name = "barrier_bands";
  const MpcRobotModelBase<scalar_t>& model = mpc.model();
  for (const std::string& name : sortedTermNames(*problem.stateSoftConstraintPtr)) {
    const JointLimitsSoftConstraint* absl_nullable limits =
        dynamic_cast<const JointLimitsSoftConstraint*>(&problem.stateSoftConstraintPtr->get(name));
    if (limits == nullptr) continue;
    const scalar_t delta = limits->getGains().delta;
    const JointPositionLimits bounds = readPinocchioJointLimits(mpc.pinocchioInterface(), mpc.modelSettings(), Verbosity::kQuiet);
    vector_t joints = model.getJointAngles(point.state);
    for (Eigen::Index joint = 0; joint < joints.size(); ++joint) {
      const scalar_t lower = bounds.lower(joint);
      const scalar_t upper = bounds.upper(joint);
      if (!std::isfinite(lower) || !std::isfinite(upper) || upper - lower < 2.0 * delta) continue;
      joints(joint) = joints(joint) - lower < upper - joints(joint) ? lower + 0.5 * delta : upper - 0.5 * delta;
    }
    model.setJointAngles(point.state, joints);
  }
  const std::vector<std::string>& contactNames = mpc.modelSettings().contactNames;
  for (size_t contact = 0; contact < contactNames.size(); ++contact) {
    const std::string name = basisNonNegativityTermName(contactNames[contact]);
    size_t index = 0;
    if (!problem.costPtr->getTermIndex(name, index)) continue;
    const BasisScalingNonNegativityConstraint* absl_nullable barrier =
        dynamic_cast<const BasisScalingNonNegativityConstraint*>(&problem.costPtr->get(name));
    if (barrier == nullptr || !barrier->isScheduleGated() || !barrier->isActive(point.time)) continue;
    point.input
        .segment(static_cast<Eigen::Index>(model.getContactWrenchStartIndices(contact)),
                 static_cast<Eigen::Index>(model.getContactInputDim(contact)))
        .setConstant(0.5 * barrier->getBarrierConfig().delta);
  }
  const std::vector<std::vector<std::string>> groups = {
      labelsOfClass<JointLimitsSoftConstraint>("stateCost", *problem.stateCostPtr),
      labelsOfClass<JointLimitsSoftConstraint>("stateSoftConstraint", *problem.stateSoftConstraintPtr),
      labelsOfClass<BasisScalingNonNegativityConstraint>("cost", *problem.costPtr),
      labelsOfClass<BasisScalingNonNegativityConstraint>("softConstraint", *problem.softConstraintPtr)};
  for (const std::vector<std::string>& group : groups) band.labels.insert(group.begin(), group.end());
  return band;
}

/**
 * The point `start` with its joint angles moved into the band of the foot-collision barrier (barrierBandPoints()), or
 * nullopt when `problem` has no foot-collision soft constraint of a piecewise polynomial barrier.
 */
std::optional<TargetedProbePoint> footCollisionBandPoint(OptimalControlProblem& problem,
                                                         const MpcEquivalenceFixture& mpc,
                                                         const TermProbePoint& start) {
  size_t index = 0;
  if (!problem.stateSoftConstraintPtr->getTermIndex(kFootCollisionTerm, index)) return std::nullopt;
  StateSoftConstraint* absl_nullable soft = dynamic_cast<StateSoftConstraint*>(&problem.stateSoftConstraintPtr->get(kFootCollisionTerm));
  if (soft == nullptr || soft->getPenalty().getPenaltyPtrArray().empty()) return std::nullopt;
  const PieceWisePolynomialBarrierPenalty* absl_nullable barrier =
      dynamic_cast<const PieceWisePolynomialBarrierPenalty*>(soft->getPenalty().getPenaltyPtrArray().front().get());
  if (barrier == nullptr) return std::nullopt;
  PieceWisePolynomialBarrierPenalty::Config config;
  barrier->getConfig(config);
  TargetedProbePoint band{.point = start, .labels = {absl::StrCat("stateSoftConstraint.", kFootCollisionTerm)}};
  TermProbePoint& point = band.point;
  point.name = "foot_collision_band";
  const MpcRobotModelBase<scalar_t>& model = mpc.model();
  const Eigen::Index jointStart = static_cast<Eigen::Index>(model.getJointStartindex());
  const Eigen::Index jointDim = static_cast<Eigen::Index>(model.getJointDim());
  const StateConstraint& constraint = soft->get();
  const scalar_t middle = 0.5 * config.delta;
  for (int step = 0; step < kMaxBandSteps; ++step) {
    requestIntermediateTermsAt(problem, point.time, point.state, point.input, TermRequest::kApproximation);
    const VectorFunctionLinearApproximation linear = constraint.getLinearApproximation(point.time, point.state, *problem.preComputationPtr);
    Eigen::Index row = 0;
    const scalar_t smallest = linear.f.minCoeff(&row);
    if (std::abs(smallest - middle) <= 0.25 * config.delta) break;
    const vector_t gradient = linear.dfdx.row(row).segment(jointStart, jointDim).transpose();
    if (gradient.isZero(/*prec=*/0.0)) break;
    model.setJointAngles(point.state, model.getJointAngles(point.state) + ((middle - smallest) / gradient.squaredNorm()) * gradient);
  }
  return band;
}

/**
 * The findings of `exemptions` that no longer hold: one for every exemption that names no label of `expected`, and one
 * for every exemption whose term `met` lists, which `what` says of it.
 */
std::vector<std::string> staleExemptionFindings(absl::Span<const std::string> expected,
                                                const absl::flat_hash_set<std::string>& met,
                                                absl::Span<const TermExemption> exemptions,
                                                absl::string_view what) {
  const absl::flat_hash_set<std::string> labels(expected.begin(), expected.end());
  std::vector<std::string> findings;
  for (const TermExemption& exemption : exemptions) {
    if (!labels.contains(exemption.label)) {
      findings.push_back(absl::StrCat(exemption.label, " is exempted (", exemption.reason,
                                      ") but names no term of the collections the test covers: correct or delete its exemption"));
    } else if (met.contains(exemption.label)) {
      findings.push_back(absl::StrCat(exemption.label, " is exempted (", exemption.reason, ") but ", what, ": delete its exemption"));
    }
  }
  return findings;
}

/** The labels `exemptions` name. */
absl::flat_hash_set<std::string> exemptedLabels(absl::Span<const TermExemption> exemptions) {
  absl::flat_hash_set<std::string> labels;
  for (const TermExemption& exemption : exemptions) labels.insert(exemption.label);
  return labels;
}

}  // namespace

std::vector<TermProbePoint> termProbePoints(const MpcEquivalenceFixture& mpc, const EquivalenceReference& reference) {
  std::mt19937 generator(kTermProbeSeed);
  const vector_t initial = mpc.eulerStateOf(mpc.initialState());
  std::vector<TermProbePoint> probes;
  probes.push_back(probePoint(mpc, reference, "initial", kDoubleSupportTime, initial, generator));
  const std::vector<NamedAttitude> attitudes = {
      {.name = "pitch_p0.2", .angles = vector3_t(0.0, 0.2, 0.0)},
      {.name = "roll_pitch_0.05_m0.03", .angles = vector3_t(0.0, -0.03, 0.05)},
      {.name = "roll_pitch_0.2_0.1", .angles = vector3_t(0.0, 0.1, 0.2)},
      {.name = "yaw_2.5_tilt_0.05", .angles = vector3_t(2.5, 0.04, 0.03)},
      {.name = "yaw_m3.0_tilt_0.05", .angles = vector3_t(-3.0, 0.04, 0.03)},
      {.name = "pitch_80deg", .angles = vector3_t(0.0, 80.0 * M_PI / 180.0, 0.0)},
  };
  for (const NamedAttitude& attitude : attitudes) {
    probes.push_back(
        probePoint(mpc, reference, attitude.name, kDoubleSupportTime, withAttitude(mpc, initial, attitude.angles, generator), generator));
  }
  probes.push_back(probePoint(mpc, reference, "random_0", kDoubleSupportTime, randomlyPerturbed(mpc, initial, generator), generator));
  // The middle of the first single support (the left foot in stance) and of the second (the right foot).
  const scalar_t firstSwing = kWalkingFirstLiftOff + 0.5 * kWalkingSingleSupport;
  const scalar_t secondSwing = firstSwing + kWalkingSingleSupport + kWalkingDoubleSupport;
  const vector_t walking = walkingForward(mpc, initial);
  const std::vector<NamedTime> swings = {{.name = "mid_swing_right", .time = firstSwing}, {.name = "mid_swing_left", .time = secondSwing}};
  for (const NamedTime& swing : swings) {
    probes.push_back(
        probePoint(mpc, reference, swing.name, swing.time, withAttitude(mpc, walking, vector3_t(0.0, -0.03, 0.05), generator), generator));
  }
  appendRestingPoints(mpc, probes);
  return probes;
}

std::vector<TargetedProbePoint> barrierBandPoints(OptimalControlProblem& problem,
                                                  const MpcEquivalenceFixture& mpc,
                                                  absl::Span<const TermProbePoint> points) {
  std::vector<TargetedProbePoint> bands;
  if (points.empty()) return bands;
  TargetedProbePoint barriers = barrierBandPoint(problem, mpc, points.front());
  if (!barriers.labels.empty()) bands.push_back(std::move(barriers));
  for (const TermProbePoint& point : points) {
    if (point.name != "mid_swing_right") continue;
    if (std::optional<TargetedProbePoint> collision = footCollisionBandPoint(problem, mpc, point); collision.has_value()) {
      bands.push_back(*std::move(collision));
    }
  }
  return bands;
}

bool hasFrozenStateJacobian(const MpcEquivalenceFixture& mpc, absl::string_view label) {
  return !mpc.isWholeBody() && absl::EndsWith(label, kZeroVelocityTermSuffix);
}

Differenced extrapolate(const std::array<vector_t, kStepFractions.size()>& samples, scalar_t step, scalar_t evaluationRelativeError) {
  const vector_t coarseDifference = (samples[0] - samples[1]) / (2.0 * step);
  const vector_t middleDifference = (samples[2] - samples[3]) / step;
  const vector_t fineDifference = (samples[4] - samples[5]) / (0.5 * step);
  const vector_t coarse = (4.0 * middleDifference - coarseDifference) / 3.0;
  vector_t fine = (4.0 * fineDifference - middleDifference) / 3.0;
  vector_t magnitude = vector_t::Zero(fine.size());
  for (const vector_t& sample : samples) magnitude = magnitude.cwiseMax(sample.cwiseAbs());
  const vector_t evaluationScale = (magnitude.array() > 0.0).select(magnitude.cwiseMax(1.0), vector_t::Zero(fine.size()));
  vector_t truncation = (fine - coarse).cwiseAbs();
  vector_t errorBound = truncation + (6.0 * evaluationRelativeError / step) * evaluationScale;
  return Differenced{.value = std::move(fine), .errorBound = std::move(errorBound), .truncation = std::move(truncation)};
}

void requestIntermediateTermsAt(
    OptimalControlProblem& problem, scalar_t time, const vector_t& state, const vector_t& input, TermRequest request) {
  const RequestSet values = Request::Cost + Request::SoftConstraint + Request::Constraint;
  problem.preComputationPtr->request(request == TermRequest::kApproximation ? values + Request::Approximation : values, time, state, input);
}

void requestFinalTermsAt(OptimalControlProblem& problem, scalar_t finalTime, const vector_t& state, TermRequest request) {
  const RequestSet values = Request::Cost + Request::SoftConstraint + Request::Constraint;
  problem.preComputationPtr->requestFinal(request == TermRequest::kApproximation ? values + Request::Approximation : values, finalTime,
                                          state);
}

std::vector<std::string> termLabels(const OptimalControlProblem& problem, TermCollections collections) {
  std::vector<std::vector<std::string>> groups = {
      collectionTermLabels("cost", *problem.costPtr),
      collectionTermLabels("softConstraint", *problem.softConstraintPtr),
      collectionTermLabels("stateCost", *problem.stateCostPtr),
      collectionTermLabels("stateSoftConstraint", *problem.stateSoftConstraintPtr),
      collectionTermLabels("preJumpCost", *problem.preJumpCostPtr),
      collectionTermLabels("preJumpSoftConstraint", *problem.preJumpSoftConstraintPtr),
      collectionTermLabels("finalCost", *problem.finalCostPtr),
      collectionTermLabels("finalSoftConstraint", *problem.finalSoftConstraintPtr),
  };
  if (collections == TermCollections::kEveryTerm) {
    groups.push_back(collectionTermLabels("equalityConstraint", *problem.equalityConstraintPtr));
    groups.push_back(collectionTermLabels("inequalityConstraint", *problem.inequalityConstraintPtr));
    groups.push_back(collectionTermLabels("stateEqualityConstraint", *problem.stateEqualityConstraintPtr));
    groups.push_back(collectionTermLabels("stateInequalityConstraint", *problem.stateInequalityConstraintPtr));
    groups.push_back(collectionTermLabels("preJumpEqualityConstraint", *problem.preJumpEqualityConstraintPtr));
    groups.push_back(collectionTermLabels("preJumpInequalityConstraint", *problem.preJumpInequalityConstraintPtr));
    groups.push_back(collectionTermLabels("finalEqualityConstraint", *problem.finalEqualityConstraintPtr));
    groups.push_back(collectionTermLabels("finalInequalityConstraint", *problem.finalInequalityConstraintPtr));
  }
  std::vector<std::string> labels;
  for (const std::vector<std::string>& group : groups) labels.insert(labels.end(), group.begin(), group.end());
  return labels;
}

std::vector<std::string> coverageFindings(absl::Span<const std::string> expected,
                                          const absl::flat_hash_set<std::string>& checked,
                                          absl::Span<const TermExemption> exemptions) {
  std::vector<std::string> findings = staleExemptionFindings(expected, checked, exemptions, "was checked");
  const absl::flat_hash_set<std::string> exempted = exemptedLabels(exemptions);
  for (const std::string& label : expected) {
    if (!checked.contains(label) && !exempted.contains(label)) {
      findings.push_back(absl::StrCat(label,
                                      " was checked at no point: give the test a point where it is active, or exempt it by name with "
                                      "the reason no point can reach it"));
    }
  }
  return findings;
}

std::vector<std::string> nonZeroFindings(absl::Span<const std::string> expected,
                                         const absl::flat_hash_set<std::string>& checked,
                                         const absl::flat_hash_set<std::string>& nonZero,
                                         absl::Span<const TermExemption> exemptions,
                                         absl::string_view quantity) {
  std::vector<std::string> findings =
      staleExemptionFindings(expected, nonZero, exemptions, absl::StrCat("its ", quantity, " was not zero"));
  const absl::flat_hash_set<std::string> exempted = exemptedLabels(exemptions);
  for (const std::string& label : expected) {
    if (checked.contains(label) && !nonZero.contains(label) && !exempted.contains(label)) {
      findings.push_back(absl::StrCat(label, ": its ", quantity,
                                      " was zero at every point it was checked at, so its check compared zeros: give the test a "
                                      "point where it is not (barrierBandPoints()), or exempt it by name with the reason no point "
                                      "can reach one"));
    }
  }
  return findings;
}

std::optional<std::string> addSoftStanceFootConstraint(OptimalControlProblem& problem,
                                                       absl::Span<const std::string> contactNames,
                                                       scalar_t softConstraintWeight) {
  for (const std::string& contact : contactNames) {
    const std::string name = zeroVelocityTermName(contact);
    size_t index = 0;
    if (!problem.equalityConstraintPtr->getTermIndex(name, index)) continue;
    if (problem.softConstraintPtr->getTermIndex(name, index)) return std::nullopt;
    const StateInputConstraint& hard = problem.equalityConstraintPtr->get(name);
    problem.softConstraintPtr->add(name,
                                   std::make_unique<StateInputSoftConstraint>(std::unique_ptr<StateInputConstraint>(hard.clone()),
                                                                              std::make_unique<QuadraticPenalty>(softConstraintWeight)));
    return absl::StrCat("softConstraint.", name);
  }
  return std::nullopt;
}

}  // namespace ocs2::humanoid
