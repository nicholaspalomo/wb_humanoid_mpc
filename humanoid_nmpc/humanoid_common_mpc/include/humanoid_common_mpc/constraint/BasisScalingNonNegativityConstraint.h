/******************************************************************************
Copyright (c) 2025. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

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

#include <memory>

#include <ocs2_core/cost/StateInputCost.h>
#include <ocs2_core/penalties/penalties/PenaltyBase.h>
#include <ocs2_core/penalties/penalties/PieceWisePolynomialBarrierPenalty.h>

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

namespace ocs2::humanoid {

/**
 * @brief Soft constraint enforcing λ ≥ 0 for basis-vector scaling inputs.
 *
 * In the basis-vector formulation, each contact wrench is parameterized as
 * W = B * λ where B columns are wrench-cone generators. For the wrench to
 * lie inside the cone, all λ_i must be non-negative.
 *
 * This cost applies a one-sided barrier penalty (lower bound = 0) to each
 * λ element in the input vector.
 *
 * Whether it is gated on the mode schedule follows the hard `zero_wrench` constraint. With `zero_wrench` listed the
 * swing-phase λ are already pinned to zero by that equality, so bounding them again is redundant and the term switches
 * itself off. The contact-implicit formulation removes `zero_wrench` - loadMpcFormulationTasks() insists on it - and
 * this term is then the ONLY thing standing between the solver and a negative λ, because in basis-vector mode
 * CentroidalMpcInterface skips ContactWrenchConeConstraint entirely and the wrench cone is enforced structurally by
 * λ ≥ 0 alone. A negative λ is an adhesive, outside-the-cone wrench, and the complementarity penalty (f_n h)² is
 * sign-blind and would not object. It must therefore be active at every node whenever the gate is off.
 * See humanoid_nmpc/docs/contact_implicit_mpc/README.md.
 *
 * WHICH PENALTY, and why it depends on the gate. The barrier this term is configured with is a
 * PieceWisePolynomialBarrierPenalty, whose derivative at λ = 0 is `-mu * delta / 2`, i.e. strictly NEGATIVE: the
 * objective falls as λ grows away from zero. That is a bribe to invent contact force on a foot in flight, and it is
 * the same defect that made the ground-penetration log barrier hover every loaded foot. It is harmless while the term
 * is GATED, because the swing-phase λ are pinned to zero by the `zero_wrench` equality and the term is switched off
 * there anyway.
 *
 * Un-gated it is not harmless in principle, so the term switches to a SquaredHingePenalty whose zero sits exactly at
 * λ = 0 in value AND gradient - the same substitution makeContactConePenalty() performs for the three un-gated cone
 * terms. The two penalties agree to within `delta` of the boundary for every λ that matters (at the shipped
 * mu = 0.01, delta = 1e-3 they differ by under 0.1 % at |λ| = 1 N), so this is not a retuning: it removes a 1 mN dead
 * zone and changes nothing else.
 */
class BasisScalingNonNegativityConstraint final : public StateInputCost {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @param referenceManager Used to query contact flags at evaluation time.
   * @param contactIndex     Which foot this constraint applies to (0 = left, 1 = right).
   * @param lambdaStartIdx   Starting index of this foot's λ block in the input vector.
   * @param numBasis         Number of basis vectors (λ elements) per foot.
   * @param barrierSettings  Configuration for the barrier penalty function.
   * @param scheduleGated    Whether to switch the term off while the mode schedule calls this foot a swing foot; see
   *                         the class documentation. Must be false under the contact-implicit formulation.
   */
  BasisScalingNonNegativityConstraint(const SwitchedModelReferenceManager& referenceManager,
                                      size_t contactIndex,
                                      size_t lambdaStartIdx,
                                      size_t numBasis,
                                      PieceWisePolynomialBarrierPenalty::Config barrierSettings,
                                      bool scheduleGated = true);

  /** Whether this term is gated on the mode schedule's contact flag; see the class documentation. */
  bool isScheduleGated() const { return scheduleGated_; }

  BasisScalingNonNegativityConstraint* clone() const override { return new BasisScalingNonNegativityConstraint(*this); }

  bool isActive(scalar_t time) const override;

  scalar_t getValue(scalar_t time,
                    const vector_t& state,
                    const vector_t& input,
                    const TargetTrajectories& targetTrajectories,
                    const PreComputation& preComp) const override;

  ScalarFunctionQuadraticApproximation getQuadraticApproximation(scalar_t time,
                                                                 const vector_t& state,
                                                                 const vector_t& input,
                                                                 const TargetTrajectories& targetTrajectories,
                                                                 const PreComputation& preComp) const override;

  /**
   * @brief Update the penalty parameters in place, so the key stays live in the tuning dashboard.
   *
   * `delta` reaches the penalty only while this term is gated. Un-gated the penalty is a hinge whose zero must sit
   * exactly on λ = 0, so a positive `delta` there would reinstate the very bribe the hinge exists to remove; it is
   * dropped deliberately, exactly as MpcParameterUpdaterModule does for the un-gated cone terms.
   */
  void setBarrierPenalty(const PieceWisePolynomialBarrierPenalty::Config& barrierSettings);

  /** The CONFIGURED parameters, which is what a hot reload round-trips; see setBarrierPenalty on `delta`. */
  PieceWisePolynomialBarrierPenalty::Config getBarrierConfig() const { return barrierConfig_; }

  /** The penalty actually in use: "SquaredHingePenalty" when un-gated, "PieceWisePolynomialBarrierPenalty" when gated. */
  std::string getPenaltyName() const { return penaltyPtr_->name(); }

 private:
  BasisScalingNonNegativityConstraint(const BasisScalingNonNegativityConstraint& rhs);

  const SwitchedModelReferenceManager* referenceManagerPtr_;
  size_t contactIndex_;
  size_t lambdaStartIdx_;
  size_t numBasis_;
  // Fixed by the formulation at load time rather than tuned, so it is const and the parallel solve reads it without
  // synchronisation. It has to survive the copy the SQP solver makes of the whole problem per worker thread.
  const bool scheduleGated_;
  PieceWisePolynomialBarrierPenalty::Config barrierConfig_;
  std::unique_ptr<PenaltyBase> penaltyPtr_;
};

}  // namespace ocs2::humanoid
