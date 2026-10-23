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

// Pinocchio forward declarations must be included first.
#include "pinocchio/fwd.hpp"

#include <algorithm>
#include <array>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "ocs2_oc/oc_problem/OptimalControlProblem.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/equivalence/EulerMpcOracle.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/equivalence/MpcEquivalenceFixture.h"

/*
 * What the two tests of every term of the optimal control problem on the quaternion root share
 * (testQuaternionRootTermDerivatives.cpp, first order; testQuaternionRootTermHessians.cpp, second order): their
 * deterministic evaluation points, built from the configuration's shipped task file (termProbePoints()), and the points
 * inside the bands of the piecewise polynomial barriers, which the shared points leave at zero (barrierBandPoints()); the
 * precomputation requests that precede every evaluation of a term; the sorted term names of a collection; the coverage
 * checks that every term of the collections a test covers was checked, and checked where its checked quantity is not
 * zero (termLabels(), coverageFindings(), nonZeroFindings()); the soft stance-foot constraint of the problem copies
 * they test (addSoftStanceFootConstraint(), which the first-order test adds)
 * and the bounded extrapolation of their differences (extrapolate()).
 */

namespace ocs2::humanoid {

/** The seed of the points' random perturbations: the points are the same in every run. */
inline constexpr unsigned kTermProbeSeed = 20261007;

/** One evaluation point: a name, a time, a state of today's layout and an input. Passive data. */
struct TermProbePoint {
  std::string name;
  scalar_t time = 0.0;
  vector_t state;
  vector_t input;
  /** Whether every velocity of the robot is zero there: its momentum and its joint velocities. */
  bool atRest = false;
};

/**
 * The points of the term tests, deterministic (kTermProbeSeed), from the task file's initial state under `reference`
 * (MpcEquivalenceFixture::walkingReference()): the initial state; single- and two-axis tilts, headings of 2.5 and -3.0
 * rad with a tilt of 0.05 rad and a pitch of 80 degrees; a random state (any heading, a tilt of up to 0.28 rad); all of
 * them in the first double support; the middle of the walk's first two single supports (the right foot swinging, then
 * the left) walking forward and tilted; and the initial state at rest at a level attitude and at the two gimbal-lock
 * attitudes of the Euler coordinates (pitch +-90 degrees). Each input carries the weight on the feet in stance, each
 * entry perturbed within 0.1; the points at rest keep the first point's input with zero joint velocities.
 */
std::vector<TermProbePoint> termProbePoints(const MpcEquivalenceFixture& mpc, const EquivalenceReference& reference);

/** The Newton steps of the foot-collision band point (barrierBandPoints()) at most. */
inline constexpr int kMaxBandSteps = 20;

/** A point of the term tests for some of the problem's terms alone. Passive data. */
struct TargetedProbePoint {
  TermProbePoint point;
  /** The labels (termLabels()) of the terms the point is for. */
  absl::flat_hash_set<std::string> labels;
};

/**
 * The points for the terms whose piecewise polynomial barriers lie in their zero region at every point of `points`
 * (termProbePoints()'s), built in `problem` (a term test's copy, the reference installed), so that the barriers' first
 * and second orders are not zero there:
 *   - barrier_bands, for every JointLimitsSoftConstraint (stateCost, stateSoftConstraint) and every
 *     BasisScalingNonNegativityConstraint (cost, softConstraint): the first point with every joint whose limits lie at
 *     least two of the joint-limit barrier's bands apart moved to half a band inside its nearer limit, and the scalings
 *     of every schedule-gated lambda >= 0 barrier active there set to half its band;
 *   - foot_collision_band, for the foot-collision soft constraint (stateSoftConstraint.kFootCollisionTerm), which is
 *     active in single support only: the point mid_swing_right with its joint angles moved, by Newton steps on the
 *     smallest row of the constraint (a distance between collision points of the feet or the knees, less twice their
 *     radius), until that row lies within a quarter of the barrier's band of the band's middle, at most kMaxBandSteps
 *     steps.
 * A point none of whose terms the problem has is left out. A foot_collision_band whose steps did not reach the band is
 * still returned: the term tests' non-zero check then fails the term by name.
 */
std::vector<TargetedProbePoint> barrierBandPoints(OptimalControlProblem& problem,
                                                  const MpcEquivalenceFixture& mpc,
                                                  absl::Span<const TermProbePoint> points);

/**
 * Whether the term `label` of `mpc`'s problem may report a state Jacobian that is a documented approximation: a stance
 * foot's zero-velocity constraint, hard or soft, by the suffix of its name, of the centroidal MPC, whose
 * ZeroVelocityConstraintCppAd evaluates an EndEffectorKinematicsTwistConstraint, which holds its map M from the foot's
 * angular velocity to the rate of its orientation residual constant at the linearization point, so its state Jacobian is
 * the derivative only where the feet do not turn. The whole body's ZeroAccelerationConstraintCppAd, under the same
 * suffix, tapes the rates exactly (humanoid_wb_mpc/end_effector/OrientationErrorWrtPlaneRates.h) and is not
 * classified. testQuaternionRootTermHessians.cpp holds the state blocks of such a soft constraint at the points at rest
 * only; testQuaternionRootTermDerivatives.cpp does not use it, and holds every state Jacobian at every point, the twist
 * constraint's with the dropped term Av_ang (dM/dx) omega added.
 */
bool hasFrozenStateJacobian(const MpcEquivalenceFixture& mpc, absl::string_view label);

/** The steps of one bounded extrapolation (extrapolate()), in units of the step h: central differences at h, h/2 and h/4. */
inline constexpr std::array<scalar_t, 6> kStepFractions = {1.0, -1.0, 0.5, -0.5, 0.25, -0.25};

/** A derivative by differences and a bound on its error, entry by entry. Passive data. */
struct Differenced {
  vector_t value;
  /** The estimate of the truncation error plus the round-off bound. */
  vector_t errorBound;
  /** The estimate of the truncation error alone: large where the differences straddle a jump or a kink. */
  vector_t truncation;
};

/**
 * d f(s) / ds at 0 from samples[i] = f(kStepFractions[i] h), h = `step`: the Richardson extrapolation
 * R(h/2) = (4 D(h/4) - D(h/2)) / 3 of the central differences D(t) = (f(t) - f(-t)) / 2t, bounded by
 * |R(h/2) - R(h)| + 6 delta / h. The first term is, asymptotically, 15 times the O(h^4) truncation of R(h/2), whose
 * own is 1/16 of R(h)'s; the second is R(h/2)'s round-off, (4 * 4 delta / h + 2 delta / h) / 3, for evaluations within
 * delta = `evaluationRelativeError` of each row's largest magnitude over the samples, floor 1 - none for a row that is
 * zero at every sample, whose differences are exact. kFunctionEvaluationRelativeError (EulerMpcOracle.h) is a hundred
 * times the evaluation error measured on the shipped robots.
 */
Differenced extrapolate(const std::array<vector_t, kStepFractions.size()>& samples,
                        scalar_t step,
                        scalar_t evaluationRelativeError = kFunctionEvaluationRelativeError);

/** The names of `collection`'s terms, sorted: the order of a hash map must not reach a test's messages. */
template <typename CollectionT>
std::vector<std::string> sortedTermNames(const CollectionT& collection) {
  std::vector<std::string> names;
  for (const std::pair<const std::string, size_t>& entry : collection.getTermNameMap()) names.push_back(entry.first);
  std::sort(names.begin(), names.end());
  return names;
}

/** What a precomputation request prepares the terms for. */
enum class TermRequest {
  /** Their values. */
  kValues,
  /** Their values and derivatives. */
  kApproximation,
};

/**
 * Requests the precomputation of every intermediate term (costs, soft constraints and constraints) at (state, input)
 * and `time`, for `request`, as a solver requests it before it evaluates the terms there.
 */
void requestIntermediateTermsAt(
    OptimalControlProblem& problem, scalar_t time, const vector_t& state, const vector_t& input, TermRequest request);

/** The same of the final terms at `finalTime`. */
void requestFinalTermsAt(OptimalControlProblem& problem, scalar_t finalTime, const vector_t& state, TermRequest request);

/** Which collections of the optimal control problem a term test covers. */
enum class TermCollections {
  /** The cost and soft-constraint collections, intermediate and final: testQuaternionRootTermHessians's. */
  kCostsAndSoftConstraints,
  /** Those and the constraint collections, intermediate and final: testQuaternionRootTermDerivatives's. */
  kEveryTerm,
};

/**
 * The labels `<collection>.<term>` of every term of `problem`'s `collections`, collection by collection, each sorted by
 * name: the terms a term test must check at one point at least. A collection is named after its OptimalControlProblem
 * member without the `Ptr` (cost, softConstraint, stateCost, stateSoftConstraint, preJumpCost, preJumpSoftConstraint,
 * finalCost, finalSoftConstraint and, for kEveryTerm, equalityConstraint, inequalityConstraint, stateEqualityConstraint,
 * stateInequalityConstraint, preJumpEqualityConstraint, preJumpInequalityConstraint, finalEqualityConstraint,
 * finalInequalityConstraint), as the term tests label the terms they check. No term test evaluates a pre-jump
 * collection, which no MPC here fills, so a term added to one fails the coverage instead of going unchecked.
 */
std::vector<std::string> termLabels(const OptimalControlProblem& problem, TermCollections collections);

/** A term a requirement of a term test does not apply to, by its label, and why. Passive data. */
struct TermExemption {
  std::string label;
  std::string reason;
};

/**
 * What a term test's coverage lacks: one finding for every label of `expected` that is not in `checked` and that no
 * exemption names; one for every exemption whose term was checked after all, whose reason no longer holds; and one for
 * every exemption that names no label of `expected`. Empty when the test checked every term it covers.
 */
std::vector<std::string> coverageFindings(absl::Span<const std::string> expected,
                                          const absl::flat_hash_set<std::string>& checked,
                                          absl::Span<const TermExemption> exemptions);

/**
 * Which terms a term test checked only against zeros: one finding for every label of `expected` that is in `checked`
 * but not in `nonZero` - its `quantity` (the reported first derivative, the reported Hessian) was zero at every point it
 * was checked at - and that no exemption names; one for every exemption whose term was non-zero after all; and one for
 * every exemption that names no label of `expected`. Empty when every checked term was checked where it is not zero.
 */
std::vector<std::string> nonZeroFindings(absl::Span<const std::string> expected,
                                         const absl::flat_hash_set<std::string>& checked,
                                         const absl::flat_hash_set<std::string>& nonZero,
                                         absl::Span<const TermExemption> exemptions,
                                         absl::string_view quantity);

/**
 * Adds to `problem` the soft form of the first stance foot's zero-velocity constraint it holds as a hard equality
 * constraint (`<foot>_zeroVelocity`: the centroidal MPC's ZeroVelocityConstraintCppAd, the whole body's
 * ZeroAccelerationConstraintCppAd), wrapped as the MPC interfaces wrap it under the zero_velocity soft constraint: a
 * StateInputSoftConstraint of a copy of the constraint with a QuadraticPenalty of `softConstraintWeight` (the foot
 * constraint's soft_constraint_weight). No shipped robot lists the soft form, which would leave the first-order test's
 * checks of it unexercised: the twist constraint's dropped term contracted with the penalty's slope, and the whole
 * body's soft gradient. Returns the soft term's label (`softConstraint.<foot>_zeroVelocity`), or nullopt when `problem`
 * holds no hard zero-velocity constraint or already holds the soft form.
 */
std::optional<std::string> addSoftStanceFootConstraint(OptimalControlProblem& problem,
                                                       absl::Span<const std::string> contactNames,
                                                       scalar_t softConstraintWeight);

}  // namespace ocs2::humanoid
