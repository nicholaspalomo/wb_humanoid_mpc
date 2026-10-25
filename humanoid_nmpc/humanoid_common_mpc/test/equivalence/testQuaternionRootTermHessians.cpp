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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Eigenvalues"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "ocs2_core/constraint/ConstraintOrder.h"
#include "ocs2_core/constraint/StateConstraint.h"
#include "ocs2_core/constraint/StateInputConstraint.h"
#include "ocs2_core/cost/StateCost.h"
#include "ocs2_core/cost/StateCostCollection.h"
#include "ocs2_core/cost/StateInputCost.h"
#include "ocs2_core/cost/StateInputCostCollection.h"
#include "ocs2_core/cost/StateInputGaussNewtonCostAd.h"
#include "ocs2_core/manifold/StateManifold.h"
#include "ocs2_core/penalties/MultidimensionalPenalty.h"
#include "ocs2_core/penalties/penalties/PenaltyBase.h"
#include "ocs2_core/penalties/penalties/PieceWisePolynomialBarrierPenalty.h"
#include "ocs2_core/penalties/penalties/SquaredHingePenalty.h"
#include "ocs2_core/soft_constraint/StateInputSoftConstraint.h"
#include "ocs2_core/soft_constraint/StateSoftConstraint.h"
#include "ocs2_oc/oc_problem/OptimalControlProblem.h"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/model.hpp"

#include "humanoid_centroidal_mpc/cost/DcmTerminalCost.h"
#include "humanoid_common_mpc/acom/AngularCenterOfMass.h"
#include "humanoid_common_mpc/common/CostTermNames.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/constraint/BasisScalingNonNegativityConstraint.h"
#include "humanoid_common_mpc/constraint/FrictionForceConeConstraint.h"
#include "humanoid_common_mpc/constraint/JointLimitsSoftConstraint.h"
#include "humanoid_common_mpc/cost/BasePoseShapedQuadraticStateCost.h"
#include "humanoid_common_mpc/cost/ComAndAcomTrackingCost.h"
#include "humanoid_common_mpc/cost/InputQuadraticCost.h"
#include "humanoid_common_mpc/cost/StateInputQuadraticCost.h"
#include "humanoid_common_mpc/cost/StateQuadraticCost.h"
#include "humanoid_common_mpc/cost/TuningQuadraticStateInputCost.h"
#include "humanoid_common_mpc/orientation/BaseOrientation.h"
#include "humanoid_common_mpc/orientation/TuningLayoutDeviation.h"
#include "humanoid_common_mpc/pinocchio_model/pinocchioUtils.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/equivalence/EulerMpcOracle.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/equivalence/MpcEquivalenceFixture.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/equivalence/TermProbePoints.h"

/*
 * The second order of every cost and soft constraint of both MPCs on the
 * quaternion root, on every shipped configuration, at the points of
 * testQuaternionRootTermDerivatives (TermProbePoints.h: the initial state, its
 * tilts, headings and a pitch of 80 degrees, a seeded random state, both
 * mid-swing points, and at rest level and at pitch
 * +-90 degrees) and, for the piecewise polynomial barriers alone, at two more
 * inside the bands of those barriers (barrierBandPoints(): the initial point
 * with the joints and the basis scalings in the bands of the joint-limit and
 * the basis-scaling barriers, and the right foot's mid-swing point with the
 * feet, or the knees, in the band of the foot-collision barrier), against
 * today's math alone: no recorded data, no hash of a shipped file, so a retuned
 * robot is tested as it is.
 *
 * Every term of the problem's cost and soft-constraint collections (cost,
 * softConstraint, stateCost, stateSoftConstraint, finalCost,
 * finalSoftConstraint), active at the point's time (the final ones at the end
 * of the horizon), is enumerated by name and given the rule its class
 * documents, L = phi(v(x, u)) with an inner vector function v and an outer
 * function phi:
 *   - a quadratic tracking cost phi(v) = 1/2 v' W v of a residual v: the
 * Tuning* costs (v = the tuning deviation of TuningLayoutDeviation and u -
 * u_ref, W = blockdiag(Q, R)), InputQuadraticCost (v = u - u_ref, W = R), every
 * CppAD Gauss-Newton cost (v its cost vector, W = I: the foot, torso,
 * external-torque and whole-body foot costs), DcmTerminalCost (v the DCM error,
 * W its weights) and ComAndAcomTrackingCost (v = [c - c_ref; e_HT of the ACoM],
 * W = blockdiag(Q_com, Q_acom), built here from the public pieces the cost
 * composes);
 *   - a penalty phi(v) = sum_i p(v_i) of a constraint v:
 * JointLimitsSoftConstraint (v = [q_max - q; q - q_min]),
 *     BasisScalingNonNegativityConstraint (v = lambda) and every
 * StateSoftConstraint / StateInputSoftConstraint (v its constraint, p its
 * MultidimensionalPenalty); and the second order the rule prescribes is the
 * Gauss-Newton model J' phi''(v) J of the inner Jacobian J, plus, for a soft
 * constraint of ConstraintOrder::Quadratic, sum_i p'(v_i) H_c,i of the
 * constraint's own reported second order, whose documented regularization -
 * FrictionForceConeConstraint's hessianDiagonalShift s, subtracted from the
 * diagonal of its state and input Hessians - is the friction cone's state shift
 * c I, c = -s sum_i p'(v_i). A term of no known class fails the test by name,
 * and so does a term no point checks (coverageFindings()) and a term whose
 * reported Hessian is zero at every point that checks it, whose comparisons
 * would be of zeros (nonZeroFindings(); a term no point can reach is exempted
 * by name with the reason).
 *
 * At every point, for every term, in the tangent the solver works in ([dx; du],
 * x (+) dx the retraction): (1) the term's value is phi(v) (1e-10 relative), so
 * the rule describes this term; (2) the radial part xi' dL/dxi of its ambient
 * gradient vanishes (1e-10), the condition under which the pull-back E' H E the
 * solver applies is the tangent Hessian with no retraction curvature -1/4 (xi'
 * dL/dxi) I added (StateManifold.h); (3) J, pulled back, is the derivative of v
 * along the retraction and the inputs (Richardson-extrapolated central
 *       differences, as testQuaternionRootTermDerivatives: 1e-6 relative to
 * max(1, max |J|); one-sided where the central ones straddle a jump of v, as a
 * residual read from a quaternion has where the rotation-matrix-to-quaternion
 *       conversion switches branch and flips its sign, or a kink of a barrier;
 * and with the sample rows whose sign that branch leaves undetermined - at a
 * pitch of +-90 degrees round-off decides it - turned back to the point's,
 *       alignSigns()), and a penalty's p'(v_i) and p''(v_i) are the derivatives
 * of p and p' (a quadratic phi's W v and W are the term's own weights); (4) the
 * reported gradient is J' phi'(v), pulled back; (5) the reported Hessian,
 * pulled back as the solver pulls it back (E' H_xx E, H_ux E, H_uu), is the
 * rule's J' phi'' J (+ sum_i p'_i H_c,i), built here independently from the
 * verified J, entry by entry within the round-off of the two assemblies (1e-12
 * times the magnitude of their summands), and it is symmetric and positive
 *       semi-definite, as a Gauss-Newton model of a convex outer function is;
 *   (6) for a ConstraintOrder::Quadratic constraint its reported second order
 * is its exact one (differences of its verified Jacobian) minus its documented
 * shift s I, and for the friction cone the state block is exactly that shift,
 * -s I_a in the ambient state, so the soft constraint's state Hessian is c I_a
 * with c >= 0, which the pull-back leaves as c E' E: c on every tangent row but
 * a quarter of it, c / 4, on the rotation's three rows; (7) the reported
 * Hessian is the term's exact Hessian - differences of its reported gradient,
 * verified first order (testQuaternionRootTermDerivatives) - minus the
 * curvature its rule drops, sum_i phi'(v_i) d2v_i (differences of J' phi'(v)
 * with phi'(v) held at the point), plus the constraint curvature it keeps. Both
 * sides of the Gauss-Newton condition are exercised and counted: where the
 * dropped curvature is zero - a linear v (the input blocks of the quadratic
 * input costs, the joint limits, the Euclidean blocks of the tuning deviation)
 * or phi'(v) = 0 - the reported Hessian is the exact Hessian; where phi'(v) is
 * nonzero and v nonlinear it differs from it by exactly that curvature. Every
 * Hessian comparison of (3), (5), (6) and (7) is shown to fail with one entry,
 * the largest (and its mirror image in a symmetric block), moved by 1e-6: of
 * itself for the round-off comparisons of (5) and (6), whose bound is far below
 * it; of max(1, itself) for the comparisons with differences, whose round-off
 * floor is absolute, as testQuaternionRootTermDerivatives's scale is; and each
 * symmetry check with its largest entry off the diagonal moved alone by 1e-6 of
 * the block's largest entry, the scale its bound is relative to.
 *
 * Tolerances: the round-off of an assembly is bounded by 1e-12 of the magnitude
 * of its summands, |E~|' (|J|' |phi''| |J| + sum_i |phi'_i| |H_c,i|) |E~|,
 * which exceeds the gamma_n ~ n 1.1e-16 of n <= 100 products and sums by two
 * orders, with the term's own J allowed to differ from the rule's by 1e-12 of
 * each row's largest entry (the DCM cost assembles from its weighted library,
 * the CoM and ACoM rule composes its J here: entries that vanish mathematically
 * come out as round-off of the row's scale, differently in the two); the error
 * of a difference by the Richardson estimate |R(h/2) - R(h)| (15 times R(h/2)'s
 * O(h^4) truncation) plus its round-off 6 delta / h for evaluations within
 * delta = kFunctionEvaluationRelativeError (EulerMpcOracle.h, 1e-12 of each
 * row's largest magnitude, floor 1, a hundred times the evaluation error
 * measured on the shipped robots; for the outer function phi, a few hundred
 * products, 1e-13); a second order compared with differences is held to the sum
 * of the two. A diagonal block is differenced along its rows' and its columns'
 * directions and the two are averaged; the cross block ux is the differences of
 * the input gradient along the state alone, so that its round-off is the input
 * gradient's and not the (often far larger) state gradient's.
 */

namespace ocs2::humanoid {
namespace {

/** [rad], [state units], [input units] The step h of the differences, as
 * testQuaternionRootTermDerivatives's. */
constexpr scalar_t kDifferenceStep = 1.0e-4;
/**
 * A first derivative against its differences, relative to max(1, its largest
 * entry): testQuaternionRootTermDerivatives's kDerivativeTolerance, for the
 * same reasons (the extrapolation's truncation and round-off, and -ffast-math's
 * reassociation in the generated code).
 */
constexpr scalar_t kFirstDerivativeTolerance = 1.0e-6;
/** A term's value against its rule's phi(v), relative to max(1, |phi|): the
 * same sums in another order. */
constexpr scalar_t kValueTolerance = 1.0e-10;
/** The round-off of an assembly, relative to the magnitude of its summands (the
 * file comment). */
constexpr scalar_t kRoundOffFactor = 1.0e-12;
/**
 * |xi' dL/dxi| relative to max(1, |dL/dxi|): zero for a term that reads the
 * quaternion through xi / |xi| alone, up to the round-off of the products that
 * make the ambient gradient.
 */
constexpr scalar_t kRadialGradientTolerance = 1.0e-10;
/**
 * The relative evaluation error of an outer function phi and its derivatives,
 * for the round-off of their differences: phi is a sum of at most a few hundred
 * products (a weighted square, one scalar penalty per row), whose error is
 * within gamma_n of the magnitudes summed; 1e-13 is a thousand unit round-offs.
 */
constexpr scalar_t kOuterEvaluationRelativeError = 1.0e-13;
/**
 * A row of an inner function is turned back to the sign it has at the point
 * only where its value there exceeds this many times the change one step can
 * make, h max(1, max |its Jacobian row|): a smooth row cannot change sign
 * within the step there, so only a sign the function does not determine is
 * turned (alignSigns()).
 */
constexpr scalar_t kSignAlignmentMargin = 10.0;
/** The change of one Hessian entry that every Hessian comparison must reject
 * (the file comment). */
constexpr scalar_t kHessianMutation = 1.0e-6;

class QuaternionRootTermHessians : public MpcEquivalenceTest {};

/** Which terms are evaluated together: at the node's time and its
 * precomputation, or at the end of the horizon. */
enum class TermGroup {
  kIntermediate,
  kFinal,
};

/** A term's inner vector function v at one point: its value and its ambient
 * Jacobians. Passive data. */
struct InnerLinearization {
  vector_t value;
  /** dv/dx, rows x the ambient state. */
  matrix_t stateJacobian;
  /** dv/du, rows x the inputs; no columns for a state term. */
  matrix_t inputJacobian;
};

/** The outer function phi of a term at v: its value, gradient and Hessian with
 * respect to v. Passive data. */
struct OuterApproximation {
  scalar_t value = 0.0;
  vector_t gradient;
  matrix_t hessian;
};

/** A ConstraintOrder::Quadratic constraint's own second order at one point.
 * Passive data. */
struct ConstraintSecondOrder {
  /** Its dfdxx, dfdux and dfduu of every row, ambient, as it reports them. */
  VectorFunctionQuadraticApproximation reported;
  /** s: its class documents the reported Hessians as the exact ones minus s I
   * on the state and input diagonals. */
  scalar_t diagonalShift = 0.0;
  /** Whether its class documents its state Hessian as that shift alone
   * (FrictionForceConeConstraint reads only u). */
  bool stateHessianIsShift = false;
};

/** The second order a term's class prescribes (the file comment): L = phi(v(x,
 * u)). Passive data. */
struct SecondOrderRule {
  std::string name;
  /** v and its Jacobians at (x, u), the precomputation requested there. */
  std::function<InnerLinearization(const vector_t&, const vector_t&)> inner;
  /** phi at v. */
  std::function<OuterApproximation(const vector_t&)> outer;
  /**
   * Whether phi is a penalty sum_i p(v_i), whose p' and p'' are held to
   * differences of p and p'. A quadratic phi's derivatives are W v and W of the
   * term's own weights by construction, which (1) and (7) hold to the term.
   */
  bool penaltyOuter = false;
  /** For a soft constraint of ConstraintOrder::Quadratic: the constraint's
   * second order at (x, u); empty otherwise. */
  std::function<ConstraintSecondOrder(const vector_t&, const vector_t&)>
      constraintSecondOrder;
};

/** One term of the problem and its rule. Its functions take (x, u); a state
 * term ignores u and reports no input. */
struct TermUnderTest {
  std::string label;
  bool hasInput = false;
  std::function<scalar_t(const vector_t&, const vector_t&)> value;
  /** The term's own quadratic approximation, ambient; for a state term with
   * empty input blocks. */
  std::function<ScalarFunctionQuadraticApproximation(const vector_t&,
                                                     const vector_t&)>
      approximation;
  /** NotFound naming the term when its class has no rule. */
  absl::StatusOr<SecondOrderRule> rule;
};

// -------------------------------------------------------------------------------------------------
// outer functions

/** phi(v) = 1/2 v' W v. */
OuterApproximation quadraticOuter(const vector_t& value,
                                  const matrix_t& weights) {
  vector_t weighted = weights * value;
  const scalar_t half = 0.5 * value.dot(weighted);
  return OuterApproximation{
      .value = half, .gradient = std::move(weighted), .hessian = weights};
}

/** phi(v) = sum_i p(v_i) of one scalar penalty p. */
OuterApproximation penaltyOuter(const PenaltyBase& penalty,
                                const vector_t& value) {
  OuterApproximation outer{
      .value = 0.0,
      .gradient = vector_t(value.size()),
      .hessian = matrix_t::Zero(value.size(), value.size())};
  for (Eigen::Index i = 0; i < value.size(); ++i) {
    outer.value += penalty.getValue(/*t=*/0.0, value(i));
    outer.gradient(i) = penalty.getDerivative(/*t=*/0.0, value(i));
    outer.hessian(i, i) = penalty.getSecondDerivative(/*t=*/0.0, value(i));
  }
  return outer;
}

/** phi of a soft constraint: its MultidimensionalPenalty's approximation of the
 * identity map v -> v. */
OuterApproximation multidimensionalPenaltyOuter(
    const MultidimensionalPenalty& penalty,
    scalar_t time,
    const vector_t& value) {
  VectorFunctionLinearApproximation identity;
  identity.f = value;
  identity.dfdx = matrix_t::Identity(value.size(), value.size());
  identity.dfdu.resize(value.size(), 0);
  ScalarFunctionQuadraticApproximation approximation =
      penalty.getQuadraticApproximation(time, identity);
  return OuterApproximation{.value = approximation.f,
                            .gradient = std::move(approximation.dfdx),
                            .hessian = std::move(approximation.dfdxx)};
}

matrix_t blockDiagonal(const matrix_t& first, const matrix_t& second) {
  matrix_t result = matrix_t::Zero(first.rows() + second.rows(),
                                   first.cols() + second.cols());
  result.topLeftCorner(first.rows(), first.cols()) = first;
  result.bottomRightCorner(second.rows(), second.cols()) = second;
  return result;
}

// -------------------------------------------------------------------------------------------------
// rules by class

/** The dense dr/dx of a block-sparse tuning deviation. */
matrix_t denseTuningJacobian(const TuningDeviationApproximation& deviation,
                             Eigen::Index stateDim) {
  matrix_t jacobian = matrix_t::Zero(deviation.residual.size(), stateDim);
  for (const TuningJacobianIdentityRun& run : deviation.identityRuns) {
    const Eigen::Index size = static_cast<Eigen::Index>(run.size);
    jacobian.block(static_cast<Eigen::Index>(run.tuningRow),
                   static_cast<Eigen::Index>(run.stateColumn), size, size) +=
        matrix_t::Identity(size, size);
  }
  for (const TuningJacobianDenseBlock& block : deviation.denseBlocks) {
    jacobian.block(static_cast<Eigen::Index>(block.tuningRow),
                   static_cast<Eigen::Index>(block.stateColumn),
                   block.value.rows(), block.value.cols()) += block.value;
  }
  return jacobian;
}

/** Where a Tuning* cost takes its references at (x, u): the state's and, for a
 * state-input cost, the input's. */
using TuningReference =
    std::function<StateInputReference(const vector_t&, const vector_t&)>;

/** What a Tuning* cost does with the input. */
enum class TuningInput {
  /** A state cost: no input. */
  kNone,
  /** A state-input cost whose input reference is the input itself
     (StateQuadraticCost): u - u_ref vanishes identically. */
  kUntracked,
  /** A state-input cost that weighs u - u_ref with R. */
  kTracked,
};

/**
 * A Tuning* cost (TuningQuadraticStateInputCost.h, TuningQuadraticStateCost.h):
 * v = r(x, x_ref), the tuning deviation, and, for a tracked input, u - u_ref; W
 * = Q, or blockdiag(Q, R).
 */
SecondOrderRule tuningRule(std::string name,
                           const MpcEquivalenceFixture& mpc,
                           TuningReference reference,
                           const matrix_t& stateWeights,
                           const matrix_t& inputWeights,
                           TuningInput input) {
  const matrix_t weights = input == TuningInput::kTracked
                               ? blockDiagonal(stateWeights, inputWeights)
                               : stateWeights;
  return SecondOrderRule{
      .name = std::move(name),
      .inner =
          [deviation = TuningLayoutDeviation(&mpc.model()),
           reference = std::move(reference),
           input](const vector_t& x, const vector_t& u) {
            const StateInputReference references = reference(x, u);
            const TuningDeviationApproximation approximation =
                deviation.linearize(x, references.state);
            const Eigen::Index tuningDim = approximation.residual.size();
            const Eigen::Index inputColumns =
                input == TuningInput::kNone ? 0 : u.size();
            const Eigen::Index inputRows =
                input == TuningInput::kTracked ? u.size() : 0;
            InnerLinearization inner{.value = vector_t(tuningDim + inputRows),
                                     .stateJacobian = matrix_t::Zero(
                                         tuningDim + inputRows, x.size()),
                                     .inputJacobian = matrix_t::Zero(
                                         tuningDim + inputRows, inputColumns)};
            inner.value.head(tuningDim) = approximation.residual;
            inner.stateJacobian.topRows(tuningDim) =
                denseTuningJacobian(approximation, x.size());
            if (input == TuningInput::kTracked) {
              inner.value.tail(inputRows) = u - references.input;
              inner.inputJacobian.bottomRows(inputRows).setIdentity();
            }
            return inner;
          },
      .outer =
          [weights](const vector_t& v) { return quadraticOuter(v, weights); },
      .penaltyOuter = false,
      .constraintSecondOrder = nullptr};
}

/** InputQuadraticCost: v = u - u_ref of the reference manager's
 * getDesiredInput(), W = R. */
SecondOrderRule inputQuadraticRule(const InputQuadraticCost& cost,
                                   const MpcEquivalenceFixture& mpc,
                                   const TargetTrajectories& targets,
                                   scalar_t time) {
  const SwitchedModelReferenceManager* absl_nonnull referenceManager =
      &mpc.referenceManager();
  const TargetTrajectories* absl_nonnull targetTrajectories = &targets;
  return SecondOrderRule{
      .name = "input quadratic: R",
      .inner =
          [referenceManager, targetTrajectories, time](const vector_t& x,
                                                       const vector_t& u) {
            return InnerLinearization{
                .value = u - referenceManager->getDesiredInput(
                                 *targetTrajectories, x, time),
                .stateJacobian = matrix_t::Zero(u.size(), x.size()),
                .inputJacobian = matrix_t::Identity(u.size(), u.size())};
          },
      .outer = [weights = cost.getInputWeights()](
                   const vector_t& v) { return quadraticOuter(v, weights); },
      .penaltyOuter = false,
      .constraintSecondOrder = nullptr};
}

/** A CppAD Gauss-Newton cost: v its cost vector f(x, u, p), W = I. */
SecondOrderRule cppAdGaussNewtonRule(const StateInputCostGaussNewtonAd& cost,
                                     const TargetTrajectories& targets,
                                     const PreComputation& preComputation,
                                     scalar_t time) {
  const StateInputCostGaussNewtonAd* absl_nonnull term = &cost;
  const TargetTrajectories* absl_nonnull targetTrajectories = &targets;
  const PreComputation* absl_nonnull precomputation = &preComputation;
  return SecondOrderRule{
      .name = "CppAD Gauss-Newton: J' J",
      .inner =
          [term, targetTrajectories, precomputation, time](const vector_t& x,
                                                           const vector_t& u) {
            VectorFunctionLinearApproximation costVector =
                term->getCostVectorLinearApproximation(
                    time, x, u, *targetTrajectories, *precomputation);
            return InnerLinearization{
                .value = std::move(costVector.f),
                .stateJacobian = std::move(costVector.dfdx),
                .inputJacobian = std::move(costVector.dfdu)};
          },
      .outer =
          [](const vector_t& v) {
            return quadraticOuter(v, matrix_t::Identity(v.size(), v.size()));
          },
      .penaltyOuter = false,
      .constraintSecondOrder = nullptr};
}

/** DcmTerminalCost: v its unweighted DCM error at getParameters(), W = diag(its
 * weights). */
SecondOrderRule dcmRule(const DcmTerminalCost& cost,
                        const TargetTrajectories& targets,
                        scalar_t time) {
  const DcmTerminalCost* absl_nonnull term = &cost;
  const TargetTrajectories* absl_nonnull targetTrajectories = &targets;
  const matrix_t weights = cost.getConfig().weights.asDiagonal();
  return SecondOrderRule{
      .name = "DCM Gauss-Newton: J' W J",
      .inner =
          [term, targetTrajectories, time](const vector_t& x,
                                           const vector_t& /*u*/) {
            const vector_t parameters =
                term->getParameters(time, *targetTrajectories);
            return InnerLinearization{
                .value = term->computeDcmError(x, parameters),
                .stateJacobian = term->computeDcmErrorJacobian(x, parameters),
                .inputJacobian = matrix_t(2, 0)};
          },
      .outer =
          [weights](const vector_t& v) { return quadraticOuter(v, weights); },
      .penaltyOuter = false,
      .constraintSecondOrder = nullptr};
}

/**
 * ComAndAcomTrackingCost's residual at `state` against `stateReference`,
 * composed from the public pieces the cost composes (its class comment): e_com
 * = c(q) - c(q_ref) with Pinocchio's CoM Jacobian through
 * getPinocchioTangentJacobian(), and e_acom = e_HT(xi_a(x), xi_a(x_ref)) of the
 * ACoM attitude with J_HT R(Delta_theta)' on the base rotation and J_HT J_r
 * J_Delta_theta on the joints.
 */
InnerLinearization comAndAcomResidual(const MpcRobotModelBase<scalar_t>& model,
                                      const pinocchio::Model& pinocchioModel,
                                      const AngularCenterOfMass& acom,
                                      const vector_t& stateReference,
                                      const vector_t& state) {
  pinocchio::Data data(pinocchioModel);
  pinocchio::jacobianCenterOfMass(pinocchioModel, data,
                                  model.getGeneralizedCoordinates(state));
  const vector3_t com = data.com[0];
  const matrix_t comJacobian =
      data.Jcom * model.getPinocchioTangentJacobian(state);
  const vector3_t comReference = pinocchio::centerOfMass(
      pinocchioModel, data, model.getGeneralizedCoordinates(stateReference));
  const vector_t joints = model.getJointAngles(state);
  const HeadingTiltError acomError = headingTiltError(
      acom.acomAttitude(model.getBaseOrientation(state), joints),
      acom.acomAttitude(model.getBaseOrientation(stateReference),
                        model.getJointAngles(stateReference)));
  const AcomAttitudeTangentJacobian tangent =
      acom.acomAttitudeTangentJacobian(joints);
  const matrix_t rotation = model.getBaseRotationTangentJacobian(state);
  InnerLinearization inner{.value = vector_t(6),
                           .stateJacobian = matrix_t::Zero(6, state.size()),
                           .inputJacobian = matrix_t(6, 0)};
  inner.value << com - comReference, acomError.residual;
  inner.stateJacobian.topRows(3) = comJacobian;
  inner.stateJacobian.block(
      3, static_cast<Eigen::Index>(model.getBaseOrientationStartIndex()), 3,
      rotation.cols()) =
      acomError.jacobianBodyTangent * tangent.baseOrientation * rotation;
  inner.stateJacobian.block(
      3, static_cast<Eigen::Index>(model.getJointStartindex()), 3,
      tangent.jointAngles.cols()) =
      acomError.jacobianBodyTangent * tangent.jointAngles;
  return inner;
}

/** ComAndAcomTrackingCost: v = [e_com; e_acom] against the model's
 * getDesiredState(), W = blockdiag(Q_com, Q_acom). */
absl::StatusOr<SecondOrderRule> comAndAcomRule(
    const ComAndAcomTrackingCost& cost,
    const MpcEquivalenceFixture& mpc,
    const TargetTrajectories& targets,
    scalar_t time) {
  ASSIGN_OR_RETURN(
      std::unique_ptr<AngularCenterOfMass> acom,
      AngularCenterOfMass::Create(mpc.modelSettings().robotName,
                                  mpc.modelSettings().mpcModelJointNames));
  const MpcRobotModelBase<scalar_t>* absl_nonnull model = &mpc.model();
  const pinocchio::Model* absl_nonnull pinocchioModel =
      &mpc.pinocchioInterface().getModel();
  const TargetTrajectories* absl_nonnull targetTrajectories = &targets;
  const matrix_t weights = blockDiagonal(cost.getQCom(), cost.getQAcom());
  return SecondOrderRule{
      .name = "CoM and ACoM Gauss-Newton: J' Q J",
      .inner =
          [model, pinocchioModel, targetTrajectories, time,
           network = AngularCenterOfMass(*acom)](const vector_t& x,
                                                 const vector_t& /*u*/) {
            return comAndAcomResidual(
                *model, *pinocchioModel, network,
                model->getDesiredState(*targetTrajectories, time), x);
          },
      .outer =
          [weights](const vector_t& v) { return quadraticOuter(v, weights); },
      .penaltyOuter = false,
      .constraintSecondOrder = nullptr};
}

/** JointLimitsSoftConstraint: v = [q_max - q; q - q_min] of the model's limits,
 * p its piecewise polynomial barrier. */
SecondOrderRule jointLimitsRule(const JointLimitsSoftConstraint& cost,
                                const MpcEquivalenceFixture& mpc) {
  const PieceWisePolynomialBarrierPenalty::Config config = cost.getGains();
  const MpcRobotModelBase<scalar_t>* absl_nonnull model = &mpc.model();
  return SecondOrderRule{
      .name = "joint-limit barrier: J' p'' J",
      .inner =
          [model, limits = readPinocchioJointLimits(mpc.pinocchioInterface(),
                                                    mpc.modelSettings(),
                                                    Verbosity::kQuiet)](
              const vector_t& x, const vector_t& /*u*/) {
            const vector_t joints = model->getJointAngles(x);
            const Eigen::Index jointDim = joints.size();
            const Eigen::Index start =
                static_cast<Eigen::Index>(model->getJointStartindex());
            InnerLinearization inner{
                .value = vector_t(2 * jointDim),
                .stateJacobian = matrix_t::Zero(2 * jointDim, x.size()),
                .inputJacobian = matrix_t(2 * jointDim, 0)};
            inner.value << limits.upper - joints, joints - limits.lower;
            inner.stateJacobian.block(0, start, jointDim, jointDim) =
                -matrix_t::Identity(jointDim, jointDim);
            inner.stateJacobian.block(jointDim, start, jointDim, jointDim)
                .setIdentity();
            return inner;
          },
      .outer =
          [config](const vector_t& v) {
            return penaltyOuter(PieceWisePolynomialBarrierPenalty(config), v);
          },
      .penaltyOuter = true,
      .constraintSecondOrder = nullptr};
}

/**
 * BasisScalingNonNegativityConstraint of the contact `name` belongs to: v = its
 * scalings lambda, p its class's penalty (the configured barrier when schedule
 * gated, the squared hinge of mu at 0 otherwise).
 */
absl::StatusOr<SecondOrderRule> basisScalingRule(
    const BasisScalingNonNegativityConstraint& cost,
    absl::string_view name,
    const MpcEquivalenceFixture& mpc) {
  const std::vector<std::string>& contactNames =
      mpc.modelSettings().contactNames;
  size_t contact = 0;
  while (contact < contactNames.size() &&
         basisNonNegativityTermName(contactNames[contact]) != name)
    ++contact;
  if (contact == contactNames.size())
    return absl::NotFoundError(
        absl::StrCat("no contact owns the scalings of ", name));
  const Eigen::Index start = static_cast<Eigen::Index>(
      mpc.model().getContactWrenchStartIndices(contact));
  const Eigen::Index scalings =
      static_cast<Eigen::Index>(mpc.model().getContactInputDim(contact));
  const PieceWisePolynomialBarrierPenalty::Config config =
      cost.getBarrierConfig();
  const bool gated = cost.isScheduleGated();
  return SecondOrderRule{
      .name = "basis-scaling barrier: p''",
      .inner =
          [start, scalings](const vector_t& x, const vector_t& u) {
            InnerLinearization inner{
                .value = u.segment(start, scalings),
                .stateJacobian = matrix_t::Zero(scalings, x.size()),
                .inputJacobian = matrix_t::Zero(scalings, u.size())};
            inner.inputJacobian.middleCols(start, scalings).setIdentity();
            return inner;
          },
      .outer =
          [config, gated](const vector_t& v) {
            if (gated)
              return penaltyOuter(PieceWisePolynomialBarrierPenalty(config), v);
            return penaltyOuter(SquaredHingePenalty(SquaredHingePenalty::Config(
                                    config.mu, /*deltaParam=*/0.0)),
                                v);
          },
      .penaltyOuter = true,
      .constraintSecondOrder = nullptr};
}

/** The second order of a ConstraintOrder::Quadratic state-input constraint,
 * with the shift its class documents. */
std::function<ConstraintSecondOrder(const vector_t&, const vector_t&)>
stateInputConstraintSecondOrder(const StateInputConstraint& constraint,
                                const PreComputation& preComputation,
                                scalar_t time) {
  if (constraint.getOrder() != ConstraintOrder::Quadratic) return nullptr;
  const FrictionForceConeConstraint* absl_nullable cone =
      dynamic_cast<const FrictionForceConeConstraint*>(&constraint);
  const scalar_t shift =
      cone != nullptr ? cone->getConfig().hessianDiagonalShift : 0.0;
  const bool stateHessianIsShift = cone != nullptr;
  const StateInputConstraint* absl_nonnull term = &constraint;
  const PreComputation* absl_nonnull precomputation = &preComputation;
  return [term, precomputation, time, shift, stateHessianIsShift](
             const vector_t& x, const vector_t& u) {
    return ConstraintSecondOrder{.reported = term->getQuadraticApproximation(
                                     time, x, u, *precomputation),
                                 .diagonalShift = shift,
                                 .stateHessianIsShift = stateHessianIsShift};
  };
}

/** A StateInputSoftConstraint: v its constraint, p its MultidimensionalPenalty.
 */
SecondOrderRule stateInputSoftConstraintRule(
    StateInputSoftConstraint& softConstraint,
    const PreComputation& preComputation,
    scalar_t time) {
  const StateInputConstraint* absl_nonnull constraint = &softConstraint.get();
  const MultidimensionalPenalty* absl_nonnull penalty =
      &softConstraint.getPenalty();
  const PreComputation* absl_nonnull precomputation = &preComputation;
  std::function<ConstraintSecondOrder(const vector_t&, const vector_t&)>
      secondOrder =
          stateInputConstraintSecondOrder(*constraint, preComputation, time);
  return SecondOrderRule{
      .name = secondOrder == nullptr
                  ? "soft constraint: J' p'' J"
                  : "soft constraint: J' p'' J + sum p' H_c (shift s I)",
      .inner =
          [constraint, precomputation, time](const vector_t& x,
                                             const vector_t& u) {
            VectorFunctionLinearApproximation linear =
                constraint->getLinearApproximation(time, x, u, *precomputation);
            return InnerLinearization{.value = std::move(linear.f),
                                      .stateJacobian = std::move(linear.dfdx),
                                      .inputJacobian = std::move(linear.dfdu)};
          },
      .outer =
          [penalty, time](const vector_t& v) {
            return multidimensionalPenaltyOuter(*penalty, time, v);
          },
      .penaltyOuter = true,
      .constraintSecondOrder = std::move(secondOrder)};
}

/** A StateSoftConstraint: v its constraint, p its MultidimensionalPenalty. */
SecondOrderRule stateSoftConstraintRule(StateSoftConstraint& softConstraint,
                                        const PreComputation& preComputation,
                                        scalar_t time) {
  const StateConstraint* absl_nonnull constraint = &softConstraint.get();
  const MultidimensionalPenalty* absl_nonnull penalty =
      &softConstraint.getPenalty();
  const PreComputation* absl_nonnull precomputation = &preComputation;
  SecondOrderRule rule{
      .name = "soft constraint: J' p'' J",
      .inner =
          [constraint, precomputation, time](const vector_t& x,
                                             const vector_t& /*u*/) {
            VectorFunctionLinearApproximation linear =
                constraint->getLinearApproximation(time, x, *precomputation);
            const Eigen::Index rows = linear.f.size();
            return InnerLinearization{.value = std::move(linear.f),
                                      .stateJacobian = std::move(linear.dfdx),
                                      .inputJacobian = matrix_t(rows, 0)};
          },
      .outer =
          [penalty, time](const vector_t& v) {
            return multidimensionalPenaltyOuter(*penalty, time, v);
          },
      .penaltyOuter = true,
      .constraintSecondOrder = nullptr};
  if (constraint->getOrder() == ConstraintOrder::Quadratic) {
    rule.name = "soft constraint: J' p'' J + sum p' H_c";
    rule.constraintSecondOrder = [constraint, precomputation, time](
                                     const vector_t& x, const vector_t& /*u*/) {
      return ConstraintSecondOrder{
          .reported =
              constraint->getQuadraticApproximation(time, x, *precomputation),
          .diagonalShift = 0.0,
          .stateHessianIsShift = false};
    };
  }
  return rule;
}

/** The rule of a state-input term by its class; NotFound naming it when there
 * is none. */
absl::StatusOr<SecondOrderRule> stateInputTermRule(
    StateInputCost& term,
    const std::string& label,
    const MpcEquivalenceFixture& mpc,
    const TargetTrajectories& targets,
    const PreComputation& preComputation,
    scalar_t time) {
  const SwitchedModelReferenceManager* absl_nonnull referenceManager =
      &mpc.referenceManager();
  const TargetTrajectories* absl_nonnull targetTrajectories = &targets;
  if (const StateQuadraticCost* absl_nullable cost =
          dynamic_cast<const StateQuadraticCost*>(&term);
      cost != nullptr) {
    // Its input deviation is identically zero
    // (StateQuadraticCost::getReference), so its R, set to zero, weighs
    // nothing.
    if (!cost->getInputWeights().isZero(/*prec=*/0.0)) {
      return absl::FailedPreconditionError(
          absl::StrCat(label,
                       " weighs an input deviation that is identically zero: R "
                       "is not zero"));
    }
    return tuningRule(
        "tuning Gauss-Newton: J' Q J", mpc,
        [referenceManager, targetTrajectories, time](const vector_t& x,
                                                     const vector_t& u) {
          return StateInputReference{.state = referenceManager->getDesiredState(
                                         *targetTrajectories, x, time),
                                     .input = u};
        },
        cost->getStateWeights(), cost->getInputWeights(),
        TuningInput::kUntracked);
  }
  if (const StateInputQuadraticCost* absl_nullable cost =
          dynamic_cast<const StateInputQuadraticCost*>(&term);
      cost != nullptr) {
    return tuningRule(
        "tuning Gauss-Newton: J' Q J, R", mpc,
        [referenceManager, targetTrajectories, time](const vector_t& x,
                                                     const vector_t& /*u*/) {
          return StateInputReference{.state = referenceManager->getDesiredState(
                                         *targetTrajectories, x, time),
                                     .input = referenceManager->getDesiredInput(
                                         *targetTrajectories, x, time)};
        },
        cost->getStateWeights(), cost->getInputWeights(),
        TuningInput::kTracked);
  }
  if (const InputQuadraticCost* absl_nullable cost =
          dynamic_cast<const InputQuadraticCost*>(&term);
      cost != nullptr) {
    return inputQuadraticRule(*cost, mpc, targets, time);
  }
  if (const StateInputCostGaussNewtonAd* absl_nullable cost =
          dynamic_cast<const StateInputCostGaussNewtonAd*>(&term);
      cost != nullptr) {
    return cppAdGaussNewtonRule(*cost, targets, preComputation, time);
  }
  if (const BasisScalingNonNegativityConstraint* absl_nullable cost =
          dynamic_cast<const BasisScalingNonNegativityConstraint*>(&term);
      cost != nullptr) {
    return basisScalingRule(*cost, label.substr(label.find('.') + 1), mpc);
  }
  if (StateInputSoftConstraint* absl_nullable softConstraint =
          dynamic_cast<StateInputSoftConstraint*>(&term);
      softConstraint != nullptr) {
    return stateInputSoftConstraintRule(*softConstraint, preComputation, time);
  }
  return absl::NotFoundError(
      absl::StrCat(label,
                   " is of no class with a second-order rule: give it one in "
                   "testQuaternionRootTermHessians"));
}

/** The rule of a state term by its class; NotFound naming it when there is
 * none. */
absl::StatusOr<SecondOrderRule> stateTermRule(
    StateCost& term,
    const std::string& label,
    const MpcEquivalenceFixture& mpc,
    const TargetTrajectories& targets,
    const PreComputation& preComputation,
    scalar_t time) {
  if (const BasePoseShapedQuadraticStateCost* absl_nullable cost =
          dynamic_cast<const BasePoseShapedQuadraticStateCost*>(&term);
      cost != nullptr) {
    const SwitchedModelReferenceManager* absl_nonnull referenceManager =
        &mpc.referenceManager();
    const MpcRobotModelBase<scalar_t>* absl_nonnull model = &mpc.model();
    const TargetTrajectories* absl_nonnull targetTrajectories = &targets;
    return tuningRule(
        "tuning Gauss-Newton: J' Q J", mpc,
        [referenceManager, model, targetTrajectories, time](
            const vector_t& /*x*/, const vector_t& /*u*/) {
          return StateInputReference{
              .state = referenceManager->shapeBasePose(
                  time, model->getDesiredState(*targetTrajectories, time)),
              .input = vector_t()};
        },
        cost->getStateWeights(), /*inputWeights=*/matrix_t(),
        TuningInput::kNone);
  }
  if (const DcmTerminalCost* absl_nullable cost =
          dynamic_cast<const DcmTerminalCost*>(&term);
      cost != nullptr) {
    return dcmRule(*cost, targets, time);
  }
  if (const ComAndAcomTrackingCost* absl_nullable cost =
          dynamic_cast<const ComAndAcomTrackingCost*>(&term);
      cost != nullptr) {
    return comAndAcomRule(*cost, mpc, targets, time);
  }
  if (const JointLimitsSoftConstraint* absl_nullable cost =
          dynamic_cast<const JointLimitsSoftConstraint*>(&term);
      cost != nullptr) {
    return jointLimitsRule(*cost, mpc);
  }
  if (StateSoftConstraint* absl_nullable softConstraint =
          dynamic_cast<StateSoftConstraint*>(&term);
      softConstraint != nullptr) {
    return stateSoftConstraintRule(*softConstraint, preComputation, time);
  }
  return absl::NotFoundError(
      absl::StrCat(label,
                   " is of no class with a second-order rule: give it one in "
                   "testQuaternionRootTermHessians"));
}

/** The active terms of `group`'s collections of `problem` at `time`, by name,
 * each with its rule. */
std::vector<TermUnderTest> termsUnderTest(OptimalControlProblem& problem,
                                          const MpcEquivalenceFixture& mpc,
                                          TermGroup group,
                                          scalar_t time) {
  std::vector<TermUnderTest> terms;
  const PreComputation* absl_nonnull preComputation =
      problem.preComputationPtr.get();
  const TargetTrajectories* absl_nonnull targets =
      problem.targetTrajectoriesPtr;
  if (group == TermGroup::kIntermediate) {
    const std::vector<
        std::pair<std::string, StateInputCostCollection* absl_nonnull>>
        inputCollections = {
            {"cost", problem.costPtr.get()},
            {"softConstraint", problem.softConstraintPtr.get()}};
    for (const std::pair<std::string, StateInputCostCollection* absl_nonnull>&
             collection : inputCollections) {
      for (const std::string& name : sortedTermNames(*collection.second)) {
        StateInputCost* absl_nonnull term = &collection.second->get(name);
        if (!term->isActive(time)) continue;
        const std::string label = absl::StrCat(collection.first, ".", name);
        terms.push_back(TermUnderTest{
            .label = label,
            .hasInput = true,
            .value =
                [term, time, targets, preComputation](const vector_t& x,
                                                      const vector_t& u) {
                  return term->getValue(time, x, u, *targets, *preComputation);
                },
            .approximation =
                [term, time, targets, preComputation](const vector_t& x,
                                                      const vector_t& u) {
                  return term->getQuadraticApproximation(time, x, u, *targets,
                                                         *preComputation);
                },
            .rule = stateInputTermRule(*term, label, mpc, *targets,
                                       *preComputation, time)});
      }
    }
  }
  const std::vector<std::pair<std::string, StateCostCollection* absl_nonnull>>
      stateCollections =
          group == TermGroup::kIntermediate
              ? std::vector<std::pair<
                    std::string,
                    StateCostCollection* absl_nonnull>>{{"stateCost",
                                                         problem.stateCostPtr
                                                             .get()},
                                                        {"stateSoftConstraint",
                                                         problem
                                                             .stateSoftConstraintPtr
                                                             .get()}}
              : std::vector<
                    std::pair<std::string, StateCostCollection* absl_nonnull>>{
                    {"finalCost", problem.finalCostPtr.get()},
                    {"finalSoftConstraint",
                     problem.finalSoftConstraintPtr.get()}};
  for (const std::pair<std::string, StateCostCollection* absl_nonnull>&
           collection : stateCollections) {
    for (const std::string& name : sortedTermNames(*collection.second)) {
      StateCost* absl_nonnull term = &collection.second->get(name);
      if (!term->isActive(time)) continue;
      const std::string label = absl::StrCat(collection.first, ".", name);
      terms.push_back(TermUnderTest{
          .label = label,
          .hasInput = false,
          .value =
              [term, time, targets, preComputation](const vector_t& x,
                                                    const vector_t& /*u*/) {
                return term->getValue(time, x, *targets, *preComputation);
              },
          .approximation =
              [term, time, targets, preComputation](const vector_t& x,
                                                    const vector_t& /*u*/) {
                ScalarFunctionQuadraticApproximation approximation =
                    term->getQuadraticApproximation(time, x, *targets,
                                                    *preComputation);
                approximation.dfdu.resize(0);
                approximation.dfdux.resize(0, x.size());
                approximation.dfduu.resize(0, 0);
                return approximation;
              },
          .rule = stateTermRule(*term, label, mpc, *targets, *preComputation,
                                time)});
    }
  }
  return terms;
}

// -------------------------------------------------------------------------------------------------
// differences

// The bounded extrapolation of the differences, extrapolate(), and its
// Differenced are TermProbePoints.h's: the term tests share them
// (kFunctionEvaluationRelativeError for a term's functions,
// kOuterEvaluationRelativeError for phi).

/**
 * d f(s) / ds at 0 from one side, `side` = +1 or -1, from `center` = f(0) and
 * samples[i] = f(kStepFractions[i] h): the forward (backward) differences F(t)
 * = (f(side t) - f(0)) / (side t) at t = h, h/2, h/4 extrapolated twice, (8
 * F(h/4) - 6 F(h/2) + F(h)) / 3, bounded by its difference from the
 * once-extrapolated 2 F(h/4) - F(h/2) plus its round-off 30 delta / h (as
 * extrapolate()).
 */
Differenced extrapolateOneSided(
    const vector_t& center,
    const std::array<vector_t, kStepFractions.size()>& samples,
    scalar_t side,
    scalar_t step,
    scalar_t evaluationRelativeError) {
  const size_t offset = side > 0.0 ? 0 : 1;
  const vector_t coarse = (samples[0 + offset] - center) / (side * step);
  const vector_t middle = (samples[2 + offset] - center) / (side * 0.5 * step);
  const vector_t fine = (samples[4 + offset] - center) / (side * 0.25 * step);
  vector_t value = (8.0 * fine - 6.0 * middle + coarse) / 3.0;
  const vector_t once = 2.0 * fine - middle;
  vector_t magnitude = center.cwiseAbs();
  for (size_t i = offset; i < samples.size(); i += 2)
    magnitude = magnitude.cwiseMax(samples[i].cwiseAbs());
  const vector_t evaluationScale =
      (magnitude.array() > 0.0)
          .select(magnitude.cwiseMax(1.0), vector_t::Zero(value.size()));
  vector_t truncation = (value - once).cwiseAbs();
  vector_t errorBound =
      truncation + (30.0 * evaluationRelativeError / step) * evaluationScale;
  return Differenced{.value = std::move(value),
                     .errorBound = std::move(errorBound),
                     .truncation = std::move(truncation)};
}

/** The largest entry of `differenced`'s truncation estimate relative to max(1,
 * the entry). */
scalar_t relativeTruncation(const Differenced& differenced) {
  if (differenced.value.size() == 0) return 0.0;
  return (differenced.truncation.array() /
          differenced.value.cwiseAbs().cwiseMax(1.0).array())
      .maxCoeff();
}

/**
 * extrapolate(), unless the central differences straddle a jump or a kink of f,
 * which their two extrapolations then disagree on by more than
 * kFirstDerivativeTolerance: the one-sided differences of the side whose
 * truncation estimate is the smaller, the side of the point's own branch. A
 * jump is how a residual read from a quaternion appears where the
 * rotation-matrix-to-quaternion conversion switches branch, which flips the
 * sign of the quaternion and of the residual and leaves the cost, its gradient
 * and its Gauss-Newton model unchanged (CppAD's conditional expressions take
 * the branch at run time); a kink, where a piecewise polynomial barrier changes
 * piece.
 */
Differenced extrapolatePiecewise(
    const vector_t& center,
    const std::array<vector_t, kStepFractions.size()>& samples,
    scalar_t step,
    scalar_t evaluationRelativeError = kFunctionEvaluationRelativeError) {
  Differenced central = extrapolate(samples, step, evaluationRelativeError);
  if (relativeTruncation(central) <= kFirstDerivativeTolerance) return central;
  Differenced forward = extrapolateOneSided(center, samples, /*side=*/1.0, step,
                                            evaluationRelativeError);
  Differenced backward = extrapolateOneSided(center, samples, /*side=*/-1.0,
                                             step, evaluationRelativeError);
  return relativeTruncation(forward) <= relativeTruncation(backward)
             ? std::move(forward)
             : std::move(backward);
}

/** A state and an input. Passive data. */
struct StateInput {
  vector_t state;
  vector_t input;
};

/** `point` moved by `step` along `direction`: tangent row `direction` through
 * the retraction, or input `direction - n_dx`. */
StateInput movedPoint(const StateManifold& manifold,
                      const TermProbePoint& point,
                      Eigen::Index direction,
                      scalar_t step) {
  const Eigen::Index tangentDim =
      static_cast<Eigen::Index>(manifold.getTangentDim());
  StateInput moved{.state = vector_t(), .input = point.input};
  if (direction < tangentDim) {
    manifold.retract(point.state, vector_t::Unit(tangentDim, direction), step,
                     moved.state);
  } else {
    moved.state = point.state;
    moved.input(direction - tangentDim) += step;
  }
  return moved;
}

void requestGroup(OptimalControlProblem& problem,
                  TermGroup group,
                  scalar_t time,
                  const StateInput& at,
                  TermRequest request) {
  if (group == TermGroup::kFinal) {
    requestFinalTermsAt(problem, time, at.state, request);
  } else {
    requestIntermediateTermsAt(problem, time, at.state, at.input, request);
  }
}

/** [dv/dx E(x) | dv/du]: the inner Jacobian in the term's tangent. */
matrix_t tangentJacobian(const StateManifold& manifold,
                         const vector_t& state,
                         const InnerLinearization& inner) {
  matrix_t stateJacobian = inner.stateJacobian;
  manifold.pullBackStateColumns(state, stateJacobian);
  matrix_t jacobian(stateJacobian.rows(),
                    stateJacobian.cols() + inner.inputJacobian.cols());
  jacobian.leftCols(stateJacobian.cols()) = stateJacobian;
  jacobian.rightCols(inner.inputJacobian.cols()) = inner.inputJacobian;
  return jacobian;
}

/** [E(x)' dL/dx; dL/du]: the reported gradient as the solver pulls it back. */
vector_t tangentGradient(
    const StateManifold& manifold,
    const vector_t& state,
    const ScalarFunctionQuadraticApproximation& approximation) {
  vector_t stateGradient = approximation.dfdx;
  manifold.pullBackGradient(state, stateGradient);
  vector_t gradient(stateGradient.size() + approximation.dfdu.size());
  gradient.head(stateGradient.size()) = stateGradient;
  gradient.tail(approximation.dfdu.size()) = approximation.dfdu;
  return gradient;
}

/** One term at one point of the differences, in its tangent. Passive data. */
struct TermSample {
  vector_t inner;
  /** The tangent inner Jacobian, column-major. */
  vector_t innerJacobian;
  vector_t gradient;
};

TermSample sampleTerm(const StateManifold& manifold,
                      const TermUnderTest& term,
                      const StateInput& at) {
  const InnerLinearization inner = term.rule->inner(at.state, at.input);
  const matrix_t jacobian = tangentJacobian(manifold, at.state, inner);
  return TermSample{
      .inner = inner.value,
      .innerJacobian =
          Eigen::Map<const vector_t>(jacobian.data(), jacobian.size()),
      .gradient = tangentGradient(manifold, at.state,
                                  term.approximation(at.state, at.input))};
}

/** The differences of one term along every direction of its tangent, column by
 * column, with their bounds. Passive data. */
struct TermDifferences {
  /** dv, rows x n: the inner Jacobian's differences. */
  matrix_t innerJacobian;
  matrix_t innerJacobianBound;
  /** d [E' dL/dx; dL/du], n x n: the exact Hessian's (unsymmetrized). */
  matrix_t gradientJacobian;
  matrix_t gradientJacobianBound;
  /** Per row i of v, n x n: d (row i of the tangent inner Jacobian)', the exact
   * Hessian of v_i (unsymmetrized). */
  std::vector<matrix_t> innerHessians;
  std::vector<matrix_t> innerHessiansBound;
  /** Rows of samples whose sign alignSigns() turned back. */
  size_t turnedRows = 0;
};

using SampleSet = std::array<std::vector<TermSample>, kStepFractions.size()>;

/** The samples of term `term`'s `field` at the steps of one extrapolation. */
std::array<vector_t, kStepFractions.size()> samplesOf(
    const SampleSet& samples,
    size_t term,
    vector_t TermSample::*absl_nonnull field) {
  std::array<vector_t, kStepFractions.size()> values;
  for (size_t i = 0; i < values.size(); ++i)
    values[i] = samples[i][term].*field;
  return values;
}

/** Stores the extrapolated derivatives of term `term` along `direction` in
 * `differences`. */
/**
 * Turns back, in the samples of an inner function and of its Jacobian, the rows
 * whose sign differs from the point's where the function does not determine it:
 * a residual read from a quaternion takes the sign the
 * rotation-matrix-to-quaternion conversion's branch gives it, and at a pitch of
 * +-90 degrees that branch is decided by the round-off of a rotation entry that
 * vanishes there, so neighboring samples come out in either sign. The cost, its
 * gradient and its Gauss-Newton model do not depend on the sign (r and J turn
 * together). A row is turned only where it exceeds kSignAlignmentMargin times
 * the change a step can make, so a row that crosses zero is never turned; a
 * row's Jacobian entries are turned with it. Returns how many rows it turned.
 */
size_t alignSigns(const TermSample& center,
                  Eigen::Index rows,
                  std::array<vector_t, kStepFractions.size()>& inner,
                  std::array<vector_t, kStepFractions.size()>& jacobian) {
  const Eigen::Index columns =
      rows == 0 ? 0 : center.innerJacobian.size() / rows;
  const matrix_t centerJacobian =
      Eigen::Map<const matrix_t>(center.innerJacobian.data(), rows, columns);
  size_t turned = 0;
  for (Eigen::Index row = 0; row < rows; ++row) {
    const scalar_t reach =
        kDifferenceStep *
        std::max(1.0, columns == 0
                          ? 0.0
                          : centerJacobian.row(row).cwiseAbs().maxCoeff());
    if (!(std::abs(center.inner(row)) > kSignAlignmentMargin * reach)) continue;
    for (size_t i = 0; i < inner.size(); ++i) {
      if (!(inner[i](row) * center.inner(row) < 0.0)) continue;
      inner[i](row) = -inner[i](row);
      for (Eigen::Index column = 0; column < columns; ++column)
        jacobian[i](row + rows * column) *= -1.0;
      ++turned;
    }
  }
  return turned;
}

void storeDirection(const TermSample& center,
                    const SampleSet& samples,
                    size_t term,
                    Eigen::Index direction,
                    TermDifferences& differences) {
  const Eigen::Index rows = differences.innerJacobian.rows();
  std::array<vector_t, kStepFractions.size()> innerSamples =
      samplesOf(samples, term, &TermSample::inner);
  std::array<vector_t, kStepFractions.size()> jacobianSamples =
      samplesOf(samples, term, &TermSample::innerJacobian);
  differences.turnedRows +=
      alignSigns(center, rows, innerSamples, jacobianSamples);
  const Differenced inner =
      extrapolatePiecewise(center.inner, innerSamples, kDifferenceStep);
  differences.innerJacobian.col(direction) = inner.value;
  differences.innerJacobianBound.col(direction) = inner.errorBound;
  const Differenced gradient = extrapolatePiecewise(
      center.gradient, samplesOf(samples, term, &TermSample::gradient),
      kDifferenceStep);
  differences.gradientJacobian.col(direction) = gradient.value;
  differences.gradientJacobianBound.col(direction) = gradient.errorBound;
  const Differenced jacobian = extrapolatePiecewise(
      center.innerJacobian, jacobianSamples, kDifferenceStep);
  const Eigen::Index columns = differences.innerJacobian.cols();
  for (Eigen::Index row = 0; row < rows; ++row) {
    for (Eigen::Index column = 0; column < columns; ++column) {
      differences.innerHessians[static_cast<size_t>(row)](column, direction) =
          jacobian.value(row + rows * column);
      differences.innerHessiansBound[static_cast<size_t>(row)](
          column, direction) = jacobian.errorBound(row + rows * column);
    }
  }
}

/**
 * The differences of every term of a group along every direction of the point's
 * tangent (the inputs too for the intermediate group), each sample taken after
 * the group's precomputation request at its moved point.
 */
std::vector<TermDifferences> differenceTerms(
    OptimalControlProblem& problem,
    const StateManifold& manifold,
    const std::vector<TermUnderTest>& terms,
    TermGroup group,
    scalar_t time,
    const TermProbePoint& point) {
  const Eigen::Index tangentDim =
      static_cast<Eigen::Index>(manifold.getTangentDim());
  const Eigen::Index inputDim = point.input.size();
  std::vector<TermDifferences> differences;
  std::vector<Eigen::Index> termDims;
  std::vector<TermSample> centers;
  const StateInput at{.state = point.state, .input = point.input};
  requestGroup(problem, group, time, at, TermRequest::kApproximation);
  for (const TermUnderTest& term : terms) {
    centers.push_back(sampleTerm(manifold, term, at));
    const Eigen::Index rows = centers.back().inner.size();
    const Eigen::Index dim = tangentDim + (term.hasInput ? inputDim : 0);
    termDims.push_back(dim);
    differences.push_back(TermDifferences{
        .innerJacobian = matrix_t::Zero(rows, dim),
        .innerJacobianBound = matrix_t::Zero(rows, dim),
        .gradientJacobian = matrix_t::Zero(dim, dim),
        .gradientJacobianBound = matrix_t::Zero(dim, dim),
        .innerHessians = std::vector<matrix_t>(static_cast<size_t>(rows),
                                               matrix_t::Zero(dim, dim)),
        .innerHessiansBound = std::vector<matrix_t>(static_cast<size_t>(rows),
                                                    matrix_t::Zero(dim, dim))});
  }
  const Eigen::Index directions =
      tangentDim + (group == TermGroup::kIntermediate ? inputDim : 0);
  for (Eigen::Index direction = 0; direction < directions; ++direction) {
    SampleSet samples;
    for (size_t i = 0; i < kStepFractions.size(); ++i) {
      const StateInput moved = movedPoint(manifold, point, direction,
                                          kStepFractions[i] * kDifferenceStep);
      requestGroup(problem, group, time, moved, TermRequest::kApproximation);
      for (const TermUnderTest& term : terms)
        samples[i].push_back(sampleTerm(manifold, term, moved));
    }
    for (size_t term = 0; term < terms.size(); ++term) {
      if (direction < termDims[term])
        storeDirection(centers[term], samples, term, direction,
                       differences[term]);
    }
  }
  return differences;
}

// -------------------------------------------------------------------------------------------------
// comparisons

/** The largest deviation of a comparison: absolute, and relative to the
 * expectation's largest entry. Passive data. */
struct Deviation {
  scalar_t absolute = 0.0;
  scalar_t relative = 0.0;
};

void accumulate(Deviation& into, const Deviation& deviation) {
  into.absolute = std::max(into.absolute, deviation.absolute);
  into.relative = std::max(into.relative, deviation.relative);
}

Deviation deviationOf(const matrix_t& actual, const matrix_t& expected) {
  if (actual.size() == 0) return Deviation{};
  const scalar_t absolute = (actual - expected).cwiseAbs().maxCoeff();
  const scalar_t scale = expected.cwiseAbs().maxCoeff();
  return Deviation{.absolute = absolute,
                   .relative = scale > 0.0 ? absolute / scale : absolute};
}

/** What one test run compared and found. Passive data. */
struct Tally {
  size_t comparisons = 0;
  size_t mutations = 0;
  std::vector<std::string> findings;
};

/** |actual - expected| <= bound entry by entry, or the worst entry relative to
 * its bound (a NaN fails); nullopt when within. */
std::optional<std::string> boundViolation(const std::string& what,
                                          const matrix_t& actual,
                                          const matrix_t& expected,
                                          const matrix_t& bound) {
  if (actual.rows() != expected.rows() || actual.cols() != expected.cols()) {
    return absl::StrFormat("%s: %d x %d, expected %d x %d", what, actual.rows(),
                           actual.cols(), expected.rows(), expected.cols());
  }
  for (Eigen::Index row = 0; row < actual.rows(); ++row) {
    for (Eigen::Index column = 0; column < actual.cols(); ++column) {
      const scalar_t difference =
          std::abs(actual(row, column) - expected(row, column));
      if (!(difference <= bound(row, column))) {
        return absl::StrFormat(
            "%s: entry (%d, %d) is %.17g, expected %.17g: |difference| %.3e "
            "above its bound %.3e",
            what, row, column, actual(row, column), expected(row, column),
            difference, bound(row, column));
      }
    }
  }
  return std::nullopt;
}

/** How a comparison's mutation is sized and placed. */
enum class Mutation {
  /** The largest entry, by kHessianMutation of itself (absolute if the matrix
     is zero). */
  kRelative,
  /** The largest entry and its mirror image, likewise: a symmetric block. */
  kRelativeSymmetric,
  /** The largest entry and its mirror image, by kHessianMutation of max(1,
     itself): a comparison with differences. */
  kFloorOneSymmetric,
  /** The largest entry alone, likewise. */
  kFloorOne,
  /**
   * The largest entry off the diagonal alone, by kHessianMutation of the
   * matrix's largest entry: a symmetry check, whose bound is relative to the
   * scale of the whole block.
   */
  kOffDiagonalOfScale,
};

/** A matrix with one entry (and perhaps its mirror image) moved: where and by
 * how much. Passive data. */
struct MutatedMatrix {
  matrix_t matrix;
  Eigen::Index row = 0;
  Eigen::Index column = 0;
  scalar_t change = 0.0;
};

/** `matrix` mutated as `mutation` says. */
MutatedMatrix mutated(const matrix_t& matrix, Mutation mutation) {
  matrix_t candidates = matrix.cwiseAbs();
  const scalar_t largest = matrix.size() == 0 ? 0.0 : candidates.maxCoeff();
  if (mutation == Mutation::kOffDiagonalOfScale)
    candidates.diagonal().setConstant(-1.0);
  MutatedMatrix result{.matrix = matrix, .row = 0, .column = 0, .change = 0.0};
  candidates.maxCoeff(&result.row, &result.column);
  const scalar_t magnitude = mutation == Mutation::kOffDiagonalOfScale
                                 ? largest
                                 : std::abs(matrix(result.row, result.column));
  const bool floorOne = mutation == Mutation::kFloorOne ||
                        mutation == Mutation::kFloorOneSymmetric;
  result.change =
      kHessianMutation * (floorOne ? std::max(1.0, magnitude)
                                   : (magnitude > 0.0 ? magnitude : 1.0));
  result.matrix(result.row, result.column) += result.change;
  const bool mirrored = mutation == Mutation::kRelativeSymmetric ||
                        mutation == Mutation::kFloorOneSymmetric;
  if (mirrored && result.row != result.column)
    result.matrix(result.column, result.row) += result.change;
  return result;
}

/** Compares `actual` with `expected` within `bound`, and that the comparison
 * rejects `actual` mutated as `mutation` says. */
Deviation compareHessian(const std::string& what,
                         const matrix_t& actual,
                         const matrix_t& expected,
                         const matrix_t& bound,
                         Mutation mutation,
                         Tally& tally) {
  ++tally.comparisons;
  if (std::optional<std::string> violation =
          boundViolation(what, actual, expected, bound);
      violation.has_value()) {
    tally.findings.push_back(*std::move(violation));
  }
  if (actual.size() > 0) {
    ++tally.mutations;
    const MutatedMatrix changed = mutated(actual, mutation);
    if (!boundViolation(what, changed.matrix, expected, bound).has_value()) {
      tally.findings.push_back(
          absl::StrFormat("%s: accepts the Hessian with entry (%d, %d) = %.6e "
                          "moved by %.3e, within its bound %.3e",
                          what, changed.row, changed.column,
                          actual(changed.row, changed.column), changed.change,
                          bound(changed.row, changed.column)));
    }
  }
  return deviationOf(actual, expected);
}

/** A first derivative against its differences within kFirstDerivativeTolerance
 * of max(1, its largest entry). */
Deviation compareFirstDerivative(const std::string& what,
                                 const matrix_t& reported,
                                 const matrix_t& differenced,
                                 Tally& tally) {
  ++tally.comparisons;
  const scalar_t scale = reported.size() == 0
                             ? 1.0
                             : std::max(1.0, reported.cwiseAbs().maxCoeff());
  const matrix_t bound = matrix_t::Constant(reported.rows(), reported.cols(),
                                            kFirstDerivativeTolerance * scale);
  if (std::optional<std::string> violation =
          boundViolation(what, reported, differenced, bound);
      violation.has_value()) {
    tally.findings.push_back(*std::move(violation));
  }
  return deviationOf(reported, differenced);
}

/** A block of a term's tangent Hessian [[xx, xu], [ux, uu]]. Passive data. */
struct HessianBlock {
  std::string name;
  Eigen::Index row = 0;
  Eigen::Index column = 0;
  Eigen::Index rows = 0;
  Eigen::Index columns = 0;
  bool symmetric = false;
  /** Whether it is a state block, whose differences a frozen-map term does not
   * have away from rest. */
  bool readsStateDifferences = false;
};

std::vector<HessianBlock> hessianBlocks(Eigen::Index tangentDim,
                                        Eigen::Index inputDim) {
  std::vector<HessianBlock> blocks = {{.name = "xx",
                                       .row = 0,
                                       .column = 0,
                                       .rows = tangentDim,
                                       .columns = tangentDim,
                                       .symmetric = true,
                                       .readsStateDifferences = true}};
  if (inputDim > 0) {
    blocks.push_back({.name = "ux",
                      .row = tangentDim,
                      .column = 0,
                      .rows = inputDim,
                      .columns = tangentDim,
                      .symmetric = false,
                      .readsStateDifferences = true});
    blocks.push_back({.name = "uu",
                      .row = tangentDim,
                      .column = tangentDim,
                      .rows = inputDim,
                      .columns = inputDim,
                      .symmetric = true,
                      .readsStateDifferences = false});
  }
  return blocks;
}

matrix_t blockOf(const matrix_t& matrix, const HessianBlock& block) {
  return matrix.block(block.row, block.column, block.rows, block.columns);
}

matrix_t symmetricPart(const matrix_t& matrix) {
  return 0.5 * (matrix + matrix.transpose());
}

// -------------------------------------------------------------------------------------------------
// one term at one point

/** A term's quantities at the point itself. Passive data. */
struct TermAtPoint {
  /** getValue() after a request of the values. */
  scalar_t value = 0.0;
  /** getQuadraticApproximation() after a request of the approximation, ambient.
   */
  ScalarFunctionQuadraticApproximation approximation;
  InnerLinearization inner;
  OuterApproximation outer;
  std::optional<ConstraintSecondOrder> secondOrder;
};

/** A term's second-order model in its tangent [dx; du] at the point (the file
 * comment). Passive data. */
struct TangentModel {
  /** E~ = blockdiag(E(x), I): the term's tangent into its ambient [x; u],
   * dense. */
  matrix_t embedding;
  /** J~ = [dv/dx E(x) | dv/du]. */
  matrix_t jacobian;
  /** The reported Hessian blocks as the solver pulls them back. */
  matrix_t reported;
  /** J~' phi'' J~. */
  matrix_t gaussNewton;
  /** sum_i phi'_i E~' H_c,i E~ of a Quadratic-order constraint's reported
   * second order; zero otherwise. */
  matrix_t constraintCurvature;
  /**
   * |E~|' (|J|' |phi''| |J| + |J|' |phi''| S + S' |phi''| |J| + sum_i |phi'_i|
   * |H_c,i|) |E~|, the magnitude of the summands of either assembly, S the
   * matrix of J's row maxima: the two assemblies' J may be evaluated apart.
   */
  matrix_t magnitude;
  /** The reported gradient as the solver pulls it back, and |E~|' (|J| + S)'
   * |phi'|, the magnitude of its summands. */
  vector_t gradient;
  vector_t gradientMagnitude;
};

/** Row `row`'s Hessian [[xx, ux'], [ux, uu]] of a constraint's second order,
 * ambient. */
matrix_t constraintRowHessian(
    const VectorFunctionQuadraticApproximation& secondOrder,
    size_t row,
    Eigen::Index inputDim) {
  const matrix_t& stateBlock = secondOrder.dfdxx[row];
  const Eigen::Index stateDim = stateBlock.rows();
  matrix_t hessian = matrix_t::Zero(stateDim + inputDim, stateDim + inputDim);
  hessian.topLeftCorner(stateDim, stateDim) = stateBlock;
  if (inputDim > 0) {
    hessian.bottomLeftCorner(inputDim, stateDim) = secondOrder.dfdux[row];
    hessian.topRightCorner(stateDim, inputDim) =
        secondOrder.dfdux[row].transpose();
    hessian.bottomRightCorner(inputDim, inputDim) = secondOrder.dfduu[row];
  }
  return hessian;
}

matrix_t reportedTangentHessian(
    const StateManifold& manifold,
    const vector_t& state,
    const ScalarFunctionQuadraticApproximation& approximation) {
  const Eigen::Index tangentDim =
      static_cast<Eigen::Index>(manifold.getTangentDim());
  const Eigen::Index inputDim = approximation.dfduu.rows();
  matrix_t stateBlock = approximation.dfdxx;
  manifold.pullBackHessian(state, stateBlock);
  matrix_t hessian(tangentDim + inputDim, tangentDim + inputDim);
  hessian.topLeftCorner(tangentDim, tangentDim) = stateBlock;
  if (inputDim > 0) {
    matrix_t cross = approximation.dfdux;
    manifold.pullBackStateColumns(state, cross);
    hessian.bottomLeftCorner(inputDim, tangentDim) = cross;
    hessian.topRightCorner(tangentDim, inputDim) = cross.transpose();
    hessian.bottomRightCorner(inputDim, inputDim) = approximation.dfduu;
  }
  return hessian;
}

TangentModel tangentModelOf(const StateManifold& manifold,
                            const vector_t& state,
                            const TermAtPoint& at) {
  const Eigen::Index stateDim = state.size();
  const Eigen::Index inputDim = at.inner.inputJacobian.cols();
  matrix_t stateEmbedding = matrix_t::Identity(stateDim, stateDim);
  manifold.pullBackStateColumns(state, stateEmbedding);
  TangentModel model;
  model.embedding =
      blockDiagonal(stateEmbedding, matrix_t::Identity(inputDim, inputDim));
  matrix_t ambientJacobian(at.inner.value.size(), stateDim + inputDim);
  ambientJacobian.leftCols(stateDim) = at.inner.stateJacobian;
  ambientJacobian.rightCols(inputDim) = at.inner.inputJacobian;
  model.jacobian = ambientJacobian * model.embedding;
  model.gaussNewton =
      model.jacobian.transpose() * at.outer.hessian * model.jacobian;
  // |J| and, for the Jacobian the term assembles from, which may be evaluated
  // apart from the one the rule reads (DcmTerminalCost's weighted library, the
  // CoM and ACoM composed here), |J| + |dJ| with |dJ| up to its row's largest
  // entry: products J' W J of the two differ by J' W dJ + dJ' W J.
  const matrix_t absoluteJacobian = ambientJacobian.cwiseAbs();
  const vector_t rowScale = absoluteJacobian.rowwise().maxCoeff();
  const matrix_t rowScaleJacobian =
      rowScale * matrix_t::Ones(1, absoluteJacobian.cols());
  const matrix_t absoluteHessian = at.outer.hessian.cwiseAbs();
  matrix_t ambientMagnitude =
      absoluteJacobian.transpose() * absoluteHessian * absoluteJacobian +
      absoluteJacobian.transpose() * absoluteHessian * rowScaleJacobian +
      rowScaleJacobian.transpose() * absoluteHessian * absoluteJacobian;
  model.constraintCurvature =
      matrix_t::Zero(model.jacobian.cols(), model.jacobian.cols());
  if (at.secondOrder.has_value()) {
    for (size_t row = 0; row < at.secondOrder->reported.dfdxx.size(); ++row) {
      const matrix_t hessian =
          constraintRowHessian(at.secondOrder->reported, row, inputDim);
      const scalar_t weight = at.outer.gradient(static_cast<Eigen::Index>(row));
      model.constraintCurvature +=
          weight * model.embedding.transpose() * hessian * model.embedding;
      ambientMagnitude += std::abs(weight) * hessian.cwiseAbs();
    }
  }
  model.magnitude = model.embedding.cwiseAbs().transpose() * ambientMagnitude *
                    model.embedding.cwiseAbs();
  model.reported = reportedTangentHessian(manifold, state, at.approximation);
  model.gradient = tangentGradient(manifold, state, at.approximation);
  model.gradientMagnitude = model.embedding.cwiseAbs().transpose() *
                            (absoluteJacobian + rowScaleJacobian).transpose() *
                            at.outer.gradient.cwiseAbs();
  return model;
}

/** What the comparisons of one term found over the points, for the summary.
 * Passive data. */
struct TermStatistics {
  std::string rule;
  size_t points = 0;
  /** The points at which its state blocks were held too: every point, a
   * frozen-map term's at rest only. */
  size_t statePoints = 0;
  Deviation innerJacobian;
  Deviation assembly;
  Deviation exactMinusCurvature;
  /** Hessian blocks whose dropped curvature is zero (the Gauss-Newton Hessian
   * is the exact one), and the others. */
  size_t exactBlocks = 0;
  size_t curvedBlocks = 0;
  /** The largest dropped curvature, relative to the reported Hessian block's
   * largest entry (floor 1). */
  scalar_t largestDroppedCurvature = 0.0;
  /** Sample rows whose undetermined sign alignSigns() turned back. */
  size_t turnedRows = 0;
  /** The points at which its reported Hessian was not zero, so that its
   * comparisons were not of zeros. */
  size_t nonZeroHessianPoints = 0;
};

/** How one term is checked at one point. Passive data. */
struct CheckContext {
  /** The point's and the term's names, for the messages. */
  std::string what;
  /** Whether its state differences hold: everywhere but where a frozen-map
   * term's feet turn (TermProbePoints.h). */
  bool stateDifferencesHold = true;
};

/** (1) The value is phi(v), and (2) the ambient gradient has no radial part on
 * the quaternion. */
void checkValueAndRadialGradient(const TermAtPoint& at,
                                 const CheckContext& context,
                                 const MpcEquivalenceFixture& mpc,
                                 const vector_t& state,
                                 Tally& tally) {
  const scalar_t phi = at.outer.value;
  const scalar_t valueBound = kValueTolerance * std::max(1.0, std::abs(phi));
  for (const scalar_t value : {at.value, at.approximation.f}) {
    ++tally.comparisons;
    if (!(std::abs(value - phi) <= valueBound)) {
      tally.findings.push_back(
          absl::StrFormat("%s: value %.17g, its rule's phi(v) %.17g",
                          context.what, value, phi));
    }
  }
  const Eigen::Index orientation =
      static_cast<Eigen::Index>(mpc.model().getBaseOrientationStartIndex());
  const vector4_t gradient = at.approximation.dfdx.segment<4>(orientation);
  const vector4_t xi = mpc.model().getBaseOrientation(state).normalized();
  ++tally.comparisons;
  if (!(std::abs(xi.dot(gradient)) <=
        kRadialGradientTolerance *
            std::max(1.0, gradient.cwiseAbs().maxCoeff()))) {
    tally.findings.push_back(
        absl::StrFormat("%s: the ambient gradient has a radial part xi' dL/dxi "
                        "= %.3e: the pull-back E' H E lacks "
                        "the retraction curvature -1/4 (xi' dL/dxi) I",
                        context.what, xi.dot(gradient)));
  }
}

/** (3) J against the differences of v, and a penalty's p' and p'' against the
 * differences of p and p'. */
void checkFirstOrders(const SecondOrderRule& rule,
                      const TermAtPoint& at,
                      const TermDifferences& differences,
                      const TangentModel& model,
                      const CheckContext& context,
                      TermStatistics& statistics,
                      Tally& tally) {
  const Eigen::Index tangentDim =
      model.jacobian.cols() - at.inner.inputJacobian.cols();
  const Eigen::Index columns = context.stateDifferencesHold
                                   ? model.jacobian.cols()
                                   : model.jacobian.cols() - tangentDim;
  const Eigen::Index first = model.jacobian.cols() - columns;
  accumulate(statistics.innerJacobian,
             compareFirstDerivative(
                 absl::StrCat(context.what, " inner Jacobian (tangent)"),
                 model.jacobian.rightCols(columns),
                 differences.innerJacobian.middleCols(first, columns), tally));
  if (!rule.penaltyOuter) return;
  // A penalty's p'(v_i) against differences of p, and p''(v_i) against
  // differences of p', row by row: each p reads its own row alone (phi'' is
  // diagonal by construction).
  const vector_t& v = at.inner.value;
  for (Eigen::Index i = 0; i < v.size(); ++i) {
    const scalar_t step = kDifferenceStep * std::max(1.0, std::abs(v(i)));
    std::array<vector_t, kStepFractions.size()> values;
    std::array<vector_t, kStepFractions.size()> derivatives;
    for (size_t s = 0; s < kStepFractions.size(); ++s) {
      const OuterApproximation moved = rule.outer(
          v + kStepFractions[s] * step * vector_t::Unit(v.size(), i));
      values[s] = vector_t::Constant(/*rows=*/1, moved.value);
      derivatives[s] = vector_t::Constant(/*rows=*/1, moved.gradient(i));
    }
    const vector_t centerValue = vector_t::Constant(/*rows=*/1, at.outer.value);
    const vector_t centerDerivative =
        vector_t::Constant(/*rows=*/1, at.outer.gradient(i));
    const Differenced first = extrapolatePiecewise(
        centerValue, values, step, kOuterEvaluationRelativeError);
    compareFirstDerivative(absl::StrCat(context.what, " p'(v_", i, ")"),
                           centerDerivative, first.value, tally);
    const Differenced second = extrapolatePiecewise(
        centerDerivative, derivatives, step, kOuterEvaluationRelativeError);
    const matrix_t reported =
        matrix_t::Constant(/*rows=*/1, /*cols=*/1, at.outer.hessian(i, i));
    compareHessian(absl::StrCat(context.what, " p''(v_", i, ")"), reported,
                   second.value,
                   second.errorBound + kRoundOffFactor * reported.cwiseAbs(),
                   Mutation::kFloorOne, tally);
  }
}

/** (4) The reported gradient is J~' phi'(v), and (5) the reported Hessian the
 * rule's J~' phi'' J~ + its constraint curvature. */
void checkAssembly(const TermAtPoint& at,
                   const TangentModel& model,
                   const CheckContext& context,
                   const std::vector<HessianBlock>& blocks,
                   TermStatistics& statistics,
                   Tally& tally) {
  const vector_t assembledGradient =
      model.jacobian.transpose() * at.outer.gradient;
  ++tally.comparisons;
  if (std::optional<std::string> violation = boundViolation(
          absl::StrCat(context.what, " gradient against J~' phi'(v)"),
          model.gradient, assembledGradient,
          kRoundOffFactor * model.gradientMagnitude);
      violation.has_value()) {
    tally.findings.push_back(*std::move(violation));
  }
  const matrix_t expected = model.gaussNewton + model.constraintCurvature;
  const matrix_t bound = kRoundOffFactor * model.magnitude;
  for (const HessianBlock& block : blocks) {
    accumulate(statistics.assembly,
               compareHessian(
                   absl::StrCat(context.what, " ", block.name,
                                " Hessian against J~' phi'' J~ + sum p' H_c"),
                   blockOf(model.reported, block), blockOf(expected, block),
                   blockOf(bound, block),
                   block.symmetric ? Mutation::kRelativeSymmetric
                                   : Mutation::kRelative,
                   tally));
    if (block.symmetric && block.rows >= 2) {
      const matrix_t reported = blockOf(model.reported, block);
      compareHessian(absl::StrCat(context.what, " ", block.name,
                                  " Hessian against its transpose"),
                     reported, reported.transpose(),
                     blockOf(bound, block) + blockOf(bound, block).transpose(),
                     Mutation::kOffDiagonalOfScale, tally);
    }
  }
  const scalar_t smallest =
      Eigen::SelfAdjointEigenSolver<matrix_t>(symmetricPart(model.reported),
                                              Eigen::EigenvaluesOnly)
          .eigenvalues()
          .minCoeff();
  const scalar_t eigenvalueBound =
      kRoundOffFactor * static_cast<scalar_t>(model.reported.rows()) *
      (model.magnitude.size() == 0 ? 0.0 : model.magnitude.maxCoeff());
  ++tally.comparisons;
  if (!(smallest >= -eigenvalueBound)) {
    tally.findings.push_back(
        absl::StrFormat("%s: the Hessian is not positive semi-definite: "
                        "smallest eigenvalue %.3e",
                        context.what, smallest));
  }
}

/** The Gram matrix E' E of the state's tangent on the unit sphere: the
 * identity, but 1/4 on the rotation's rows. */
matrix_t unitTangentGram(const StateManifold& manifold) {
  const Eigen::Index tangentDim =
      static_cast<Eigen::Index>(manifold.getTangentDim());
  matrix_t gram = matrix_t::Identity(tangentDim, tangentDim);
  for (Eigen::Index row = 0; row < tangentDim; ++row) {
    if (manifold.getMaximumRotationAngle(vector_t::Unit(tangentDim, row)) > 0.0)
      gram(row, row) = 0.25;
  }
  return gram;
}

/** (6) A Quadratic-order constraint's reported second order is its exact one
 * minus its documented shift. */
void checkConstraintSecondOrder(const TermAtPoint& at,
                                const TermDifferences& differences,
                                const TangentModel& model,
                                const CheckContext& context,
                                const StateManifold& manifold,
                                Tally& tally) {
  if (!at.secondOrder.has_value()) return;
  const ConstraintSecondOrder& secondOrder = *at.secondOrder;
  const Eigen::Index inputDim = at.inner.inputJacobian.cols();
  const Eigen::Index stateDim = at.inner.stateJacobian.cols();
  const matrix_t& embedding = model.embedding;
  const matrix_t shift =
      secondOrder.diagonalShift * embedding.transpose() * embedding;
  scalar_t regularization = 0.0;
  for (size_t row = 0; row < secondOrder.reported.dfdxx.size(); ++row) {
    const matrix_t hessian =
        constraintRowHessian(secondOrder.reported, row, inputDim);
    const matrix_t exact = embedding.transpose() * hessian * embedding + shift;
    const matrix_t bound =
        symmetricPart(differences.innerHessiansBound[row]) +
        kRoundOffFactor * (embedding.cwiseAbs().transpose() *
                               hessian.cwiseAbs() * embedding.cwiseAbs() +
                           shift.cwiseAbs());
    const Eigen::Index first =
        context.stateDifferencesHold ? 0 : embedding.cols() - inputDim;
    const Eigen::Index size = embedding.cols() - first;
    compareHessian(
        absl::StrCat(context.what, " constraint row ", row,
                     " second order + its shift against its differences"),
        exact.bottomRightCorner(size, size),
        symmetricPart(differences.innerHessians[row])
            .bottomRightCorner(size, size),
        bound.bottomRightCorner(size, size), Mutation::kFloorOneSymmetric,
        tally);
    regularization -= secondOrder.diagonalShift *
                      at.outer.gradient(static_cast<Eigen::Index>(row));
    if (!secondOrder.stateHessianIsShift) continue;
    // The documented form: the state Hessian is the shift alone, -s I_a, and
    // the cross block is zero.
    compareHessian(
        absl::StrCat(context.what, " constraint row ", row,
                     " ambient state Hessian against -s I"),
        secondOrder.reported.dfdxx[row],
        -secondOrder.diagonalShift * matrix_t::Identity(stateDim, stateDim),
        matrix_t::Constant(stateDim, stateDim,
                           kRoundOffFactor * secondOrder.diagonalShift),
        Mutation::kRelativeSymmetric, tally);
    compareHessian(
        absl::StrCat(context.what, " constraint row ", row,
                     " ambient cross Hessian against 0"),
        secondOrder.reported.dfdux[row], matrix_t::Zero(inputDim, stateDim),
        matrix_t::Zero(inputDim, stateDim), Mutation::kRelative, tally);
  }
  if (!secondOrder.stateHessianIsShift) return;
  // The soft constraint's state Hessian: the Gauss-Newton part and c I_a, c =
  // -s sum_i p'_i >= 0, pulled back to c E' E.
  ++tally.comparisons;
  if (!(regularization >= 0.0)) {
    tally.findings.push_back(
        absl::StrFormat("%s: the regularization c = %.3e is negative",
                        context.what, regularization));
  }
  const Eigen::Index tangentDim = embedding.cols() - inputDim;
  const matrix_t gram =
      embedding.topLeftCorner(stateDim, tangentDim).transpose() *
      embedding.topLeftCorner(stateDim, tangentDim);
  compareHessian(
      absl::StrCat(context.what,
                   " E' E against the identity with 1/4 on the rotation"),
      gram, unitTangentGram(manifold),
      matrix_t::Constant(tangentDim, tangentDim, kRoundOffFactor),
      Mutation::kRelativeSymmetric, tally);
  const matrix_t stateRegularization =
      model.reported.topLeftCorner(tangentDim, tangentDim) -
      model.gaussNewton.topLeftCorner(tangentDim, tangentDim);
  compareHessian(
      absl::StrCat(context.what,
                   " state Hessian beyond J~' p'' J~ against c E' E (c / 4 on "
                   "the rotation)"),
      stateRegularization, regularization * unitTangentGram(manifold),
      kRoundOffFactor * (model.magnitude.topLeftCorner(tangentDim, tangentDim) +
                         matrix_t::Constant(tangentDim, tangentDim,
                                            std::abs(regularization))),
      Mutation::kRelativeSymmetric, tally);
}

/**
 * (7) The reported Hessian against the exact Hessian minus the curvature the
 * rule drops plus the constraint curvature it keeps, and on which side of the
 * Gauss-Newton condition each block lies.
 */
void checkExactness(const TermAtPoint& at,
                    const TermDifferences& differences,
                    const TangentModel& model,
                    const CheckContext& context,
                    const std::vector<HessianBlock>& blocks,
                    TermStatistics& statistics,
                    Tally& tally) {
  const Eigen::Index dim = differences.gradientJacobian.rows();
  matrix_t curvature = matrix_t::Zero(dim, dim);
  matrix_t curvatureBound = matrix_t::Zero(dim, dim);
  for (size_t row = 0; row < differences.innerHessians.size(); ++row) {
    const scalar_t weight = at.outer.gradient(static_cast<Eigen::Index>(row));
    curvature += weight * differences.innerHessians[row];
    curvatureBound += std::abs(weight) * differences.innerHessiansBound[row];
  }
  const matrix_t roundOff = kRoundOffFactor * model.magnitude;
  const matrix_t gaussNewtonPart = model.reported - model.constraintCurvature;
  for (const HessianBlock& block : blocks) {
    if (block.readsStateDifferences && !context.stateDifferencesHold) continue;
    // A diagonal block is differenced twice, along its rows' and its columns'
    // directions: their mean. The cross block ux is taken as the differences of
    // the input gradient along the state, whose round-off is the input
    // gradient's.
    const std::function<matrix_t(const matrix_t&)> estimate =
        [&block](const matrix_t& matrix) {
          return block.symmetric ? symmetricPart(blockOf(matrix, block))
                                 : blockOf(matrix, block);
        };
    accumulate(
        statistics.exactMinusCurvature,
        compareHessian(
            absl::StrCat(context.what, " ", block.name,
                         " Hessian against the exact Hessian minus the dropped "
                         "curvature"),
            blockOf(gaussNewtonPart, block),
            estimate(differences.gradientJacobian - curvature),
            estimate(differences.gradientJacobianBound + curvatureBound) +
                blockOf(roundOff, block),
            block.symmetric ? Mutation::kFloorOneSymmetric
                            : Mutation::kFloorOne,
            tally));
    const matrix_t droppedBlock =
        estimate(curvature) - blockOf(model.constraintCurvature, block);
    if (boundViolation("dropped curvature", droppedBlock,
                       matrix_t::Zero(block.rows, block.columns),
                       estimate(curvatureBound) + blockOf(roundOff, block))
            .has_value()) {
      ++statistics.curvedBlocks;
      const scalar_t scale =
          std::max(1.0, blockOf(model.reported, block).cwiseAbs().maxCoeff());
      statistics.largestDroppedCurvature =
          std::max(statistics.largestDroppedCurvature,
                   droppedBlock.cwiseAbs().maxCoeff() / scale);
    } else {
      ++statistics.exactBlocks;
    }
  }
}

/** The terms of `group` at `point`: their quantities there, after the requests
 * a solver makes. */
std::vector<TermAtPoint> evaluateAtPoint(
    OptimalControlProblem& problem,
    const std::vector<TermUnderTest>& terms,
    TermGroup group,
    scalar_t time,
    const TermProbePoint& point) {
  const StateInput at{.state = point.state, .input = point.input};
  std::vector<TermAtPoint> evaluated(terms.size());
  requestGroup(problem, group, time, at, TermRequest::kValues);
  for (size_t j = 0; j < terms.size(); ++j)
    evaluated[j].value = terms[j].value(point.state, point.input);
  requestGroup(problem, group, time, at, TermRequest::kApproximation);
  for (size_t j = 0; j < terms.size(); ++j) {
    const SecondOrderRule& rule = *terms[j].rule;
    evaluated[j].approximation =
        terms[j].approximation(point.state, point.input);
    evaluated[j].inner = rule.inner(point.state, point.input);
    evaluated[j].outer = rule.outer(evaluated[j].inner.value);
    if (rule.constraintSecondOrder != nullptr)
      evaluated[j].secondOrder =
          rule.constraintSecondOrder(point.state, point.input);
  }
  return evaluated;
}

/**
 * Every check of every term of `group` at `point`, or of the terms `only` names
 * alone when it is not null (a TargetedProbePoint's; not retained).
 */
void checkGroupAtPoint(
    OptimalControlProblem& problem,
    const MpcEquivalenceFixture& mpc,
    TermGroup group,
    scalar_t time,
    const TermProbePoint& point,
    const absl::flat_hash_set<std::string>* absl_nullable only,
    std::map<std::string, TermStatistics>& statistics,
    Tally& tally) {
  std::vector<TermUnderTest> terms;
  for (TermUnderTest& term : termsUnderTest(problem, mpc, group, time)) {
    if (only != nullptr && !only->contains(term.label)) continue;
    if (term.rule.ok()) {
      terms.push_back(std::move(term));
    } else {
      tally.findings.push_back(absl::StrCat(point.name, " ", term.label, ": ",
                                            term.rule.status().message()));
    }
  }
  if (terms.empty()) return;
  const StateManifold& manifold = mpc.manifold();
  const std::vector<TermAtPoint> evaluated =
      evaluateAtPoint(problem, terms, group, time, point);
  const std::vector<TermDifferences> differences =
      differenceTerms(problem, manifold, terms, group, time, point);
  for (size_t j = 0; j < terms.size(); ++j) {
    const TangentModel model =
        tangentModelOf(manifold, point.state, evaluated[j]);
    const CheckContext context{
        .what = absl::StrCat(point.name, " ", terms[j].label),
        .stateDifferencesHold =
            point.atRest || !hasFrozenStateJacobian(mpc, terms[j].label)};
    TermStatistics& termStatistics = statistics[terms[j].label];
    termStatistics.rule = terms[j].rule->name;
    ++termStatistics.points;
    if (context.stateDifferencesHold) ++termStatistics.statePoints;
    if (!model.reported.isZero(/*prec=*/0.0))
      ++termStatistics.nonZeroHessianPoints;
    termStatistics.turnedRows += differences[j].turnedRows;
    const std::vector<HessianBlock> blocks =
        hessianBlocks(static_cast<Eigen::Index>(manifold.getTangentDim()),
                      evaluated[j].inner.inputJacobian.cols());
    checkValueAndRadialGradient(evaluated[j], context, mpc, point.state, tally);
    checkFirstOrders(*terms[j].rule, evaluated[j], differences[j], model,
                     context, termStatistics, tally);
    checkAssembly(evaluated[j], model, context, blocks, termStatistics, tally);
    checkConstraintSecondOrder(evaluated[j], differences[j], model, context,
                               manifold, tally);
    checkExactness(evaluated[j], differences[j], model, context, blocks,
                   termStatistics, tally);
  }
}

TEST_P(QuaternionRootTermHessians,
       EverySecondOrderIsItsRulesModelOfVerifiedFirstOrders) {
  MpcEquivalenceFixture& mpc = this->mpc();
  const EquivalenceReference reference = mpc.walkingReference();
  std::vector<TermProbePoint> points = termProbePoints(mpc, reference);
  const scalar_t finalTime = reference.initTime + mpc.timeHorizon();
  // A copy of the problem, whose collections hand out mutable terms; its terms
  // share the interface's reference manager, on which the reference is
  // installed as an MPC solve installs it.
  OptimalControlProblem problem(mpc.problem());
  mpc.installReference(reference);
  problem.targetTrajectoriesPtr =
      &mpc.referenceManager().getTargetTrajectories();
  const std::vector<TargetedProbePoint> bands =
      barrierBandPoints(problem, mpc, points);

  std::map<std::string, TermStatistics> statistics;
  Tally tally;
  for (const TermProbePoint& point : points) {
    checkGroupAtPoint(problem, mpc, TermGroup::kIntermediate, point.time, point,
                      /*only=*/nullptr, statistics, tally);
    checkGroupAtPoint(problem, mpc, TermGroup::kFinal, finalTime, point,
                      /*only=*/nullptr, statistics, tally);
  }
  // The barriers, which the shared points leave in their zero region, inside
  // their bands.
  for (const TargetedProbePoint& band : bands) {
    checkGroupAtPoint(problem, mpc, TermGroup::kIntermediate, band.point.time,
                      band.point, &band.labels, statistics, tally);
  }
  size_t exactBlocks = 0;
  size_t curvedBlocks = 0;
  absl::flat_hash_set<std::string> checked;
  absl::flat_hash_set<std::string> nonZero;
  for (const std::pair<const std::string, TermStatistics>& entry : statistics) {
    const TermStatistics& term = entry.second;
    exactBlocks += term.exactBlocks;
    curvedBlocks += term.curvedBlocks;
    if (term.statePoints > 0) checked.insert(entry.first);
    if (term.nonZeroHessianPoints > 0) nonZero.insert(entry.first);
    absl::PrintF(
        "[hessians] %s | %s | %s | %d points (Hessian not zero at %d) | inner "
        "J %.1e (%.1e rel) | assembly %.1e (%.1e rel) | exact - "
        "curvature %.1e (%.1e rel) | exact blocks %d, curved blocks %d "
        "(largest dropped curvature %.1e) | rows of undetermined sign "
        "turned %d\n",
        GetParam(), entry.first, term.rule, term.points,
        term.nonZeroHessianPoints, term.innerJacobian.absolute,
        term.innerJacobian.relative, term.assembly.absolute,
        term.assembly.relative, term.exactMinusCurvature.absolute,
        term.exactMinusCurvature.relative, term.exactBlocks, term.curvedBlocks,
        term.largestDroppedCurvature, term.turnedRows);
  }
  // Every term of the cost and soft-constraint collections, checked at one
  // point at least, and at one where its reported Hessian is not zero.
  const std::vector<std::string> expected =
      termLabels(problem, TermCollections::kCostsAndSoftConstraints);
  const std::vector<std::string> coverage =
      coverageFindings(expected, checked, /*exemptions=*/{});
  const std::vector<std::string> zeros = nonZeroFindings(
      expected, checked, nonZero, /*exemptions=*/{}, "reported Hessian");
  absl::PrintF(
      "[hessians] %s | %d of the problem's %d costs and soft constraints "
      "checked, %d where their reported Hessian is not zero\n",
      GetParam(), checked.size(), expected.size(), nonZero.size());
  testing::Test::RecordProperty("terms", static_cast<int>(statistics.size()));
  testing::Test::RecordProperty("collection_terms",
                                static_cast<int>(expected.size()));
  testing::Test::RecordProperty("checked_terms",
                                static_cast<int>(checked.size()));
  testing::Test::RecordProperty("comparisons",
                                static_cast<int>(tally.comparisons));
  testing::Test::RecordProperty("mutations", static_cast<int>(tally.mutations));
  testing::Test::RecordProperty("exact_blocks", static_cast<int>(exactBlocks));
  testing::Test::RecordProperty("curved_blocks",
                                static_cast<int>(curvedBlocks));
  EXPECT_FALSE(statistics.empty());
  EXPECT_TRUE(coverage.empty())
      << coverage.size() << " terms of the problem unchecked, the first:\n"
      << (coverage.empty() ? std::string() : coverage.front());
  for (const std::string& finding : coverage) ADD_FAILURE() << finding;
  EXPECT_TRUE(zeros.empty())
      << zeros.size()
      << " terms of the problem checked against zeros only, the first:\n"
      << (zeros.empty() ? std::string() : zeros.front());
  for (const std::string& finding : zeros) ADD_FAILURE() << finding;
  // Both sides of the Gauss-Newton condition occur on every configuration.
  EXPECT_GT(exactBlocks, 0u);
  EXPECT_GT(curvedBlocks, 0u);
  EXPECT_TRUE(tally.findings.empty())
      << tally.findings.size() << " findings, the first:\n"
      << (tally.findings.empty() ? std::string() : tally.findings.front());
  for (size_t i = 0; i < tally.findings.size() && i < 20; ++i)
    ADD_FAILURE() << tally.findings[i];
}

INSTANTIATE_TEST_SUITE_P(
    Shipped,
    QuaternionRootTermHessians,
    ::testing::ValuesIn(mpcEquivalenceConfigurationNames()),
    mpcEquivalenceTestName);

}  // namespace
}  // namespace ocs2::humanoid
