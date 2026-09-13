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

#include <ocs2_core/cost/StateInputCost.h>
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
 * λ element in the input vector. The penalty is only active for feet in
 * contact (swing-phase λ are zeroed by the ZeroWrench hard constraint).
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
   */
  BasisScalingNonNegativityConstraint(const SwitchedModelReferenceManager& referenceManager,
                                      size_t contactIndex,
                                      size_t lambdaStartIdx,
                                      size_t numBasis,
                                      PieceWisePolynomialBarrierPenalty::Config barrierSettings);

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
   * @brief Update the barrier penalty parameters in place.
   */
  void setBarrierPenalty(const PieceWisePolynomialBarrierPenalty::Config& barrierSettings);

  PieceWisePolynomialBarrierPenalty::Config getBarrierConfig() const;

 private:
  BasisScalingNonNegativityConstraint(const BasisScalingNonNegativityConstraint& rhs);

  const SwitchedModelReferenceManager* referenceManagerPtr_;
  size_t contactIndex_;
  size_t lambdaStartIdx_;
  size_t numBasis_;
  std::unique_ptr<PieceWisePolynomialBarrierPenalty> penaltyPtr_;
};

}  // namespace ocs2::humanoid
