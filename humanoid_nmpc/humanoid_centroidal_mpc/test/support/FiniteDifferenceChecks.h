/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

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

#include <gtest/gtest.h>

#include <ocs2_core/PreComputation.h>
#include <ocs2_core/Types.h>

#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid {

/** Central-difference step and tolerance that every contact term in this package is checked against. */
constexpr scalar_t kFiniteDifferenceStep = 1e-6;
constexpr scalar_t kDerivativeTol = 1e-5;

/**
 * Checks a state-input term's analytic linear approximation against central finite differences of its own value.
 *
 * This is the check that catches the one class of mistake unit tests otherwise miss entirely: a scale, a frame change
 * or a sign applied to the value but not to the Jacobian. Such a term reports the right number, passes every test that
 * reads getValue(), and hands the solver a wrong gradient.
 *
 * Templated on the term rather than taking a StateInputConstraint&, so that it serves both the constraints and any
 * other object exposing getValue / getLinearApproximation with the same signatures.
 */
template <typename TERM_T>
void expectStateInputDerivativesMatchFiniteDifferences(const TERM_T& term, const vector_t& state, const vector_t& input) {
  const PreComputation preComp;
  const VectorFunctionLinearApproximation approximation = term.getLinearApproximation(0.0, state, input, preComp);
  EXPECT_TRUE(approximation.f.isApprox(term.getValue(0.0, state, input, preComp), 1e-12))
      << "the linear approximation's value disagrees with getValue()";

  for (long index = 0; index < state.size(); ++index) {
    vector_t perturbed = state;
    perturbed(index) += kFiniteDifferenceStep;
    const vector_t forward = term.getValue(0.0, perturbed, input, preComp);
    perturbed(index) -= 2.0 * kFiniteDifferenceStep;
    const vector_t backward = term.getValue(0.0, perturbed, input, preComp);
    const vector_t numerical = (forward - backward) / (2.0 * kFiniteDifferenceStep);
    for (long row = 0; row < numerical.size(); ++row) {
      EXPECT_NEAR(approximation.dfdx(row, index), numerical(row), kDerivativeTol) << "dfdx(" << row << ", " << index << ")";
    }
  }
  for (long index = 0; index < input.size(); ++index) {
    vector_t perturbed = input;
    perturbed(index) += kFiniteDifferenceStep;
    const vector_t forward = term.getValue(0.0, state, perturbed, preComp);
    perturbed(index) -= 2.0 * kFiniteDifferenceStep;
    const vector_t backward = term.getValue(0.0, state, perturbed, preComp);
    const vector_t numerical = (forward - backward) / (2.0 * kFiniteDifferenceStep);
    for (long row = 0; row < numerical.size(); ++row) {
      EXPECT_NEAR(approximation.dfdu(row, index), numerical(row), kDerivativeTol) << "dfdu(" << row << ", " << index << ")";
    }
  }
}

/** The same check for a state-only term. */
template <typename TERM_T>
void expectStateDerivativesMatchFiniteDifferences(const TERM_T& term, const vector_t& state) {
  const PreComputation preComp;
  const VectorFunctionLinearApproximation approximation = term.getLinearApproximation(0.0, state, preComp);
  EXPECT_TRUE(approximation.f.isApprox(term.getValue(0.0, state, preComp), 1e-12))
      << "the linear approximation's value disagrees with getValue()";

  for (long index = 0; index < state.size(); ++index) {
    vector_t perturbed = state;
    perturbed(index) += kFiniteDifferenceStep;
    const vector_t forward = term.getValue(0.0, perturbed, preComp);
    perturbed(index) -= 2.0 * kFiniteDifferenceStep;
    const vector_t backward = term.getValue(0.0, perturbed, preComp);
    const vector_t numerical = (forward - backward) / (2.0 * kFiniteDifferenceStep);
    for (long row = 0; row < numerical.size(); ++row) {
      EXPECT_NEAR(approximation.dfdx(row, index), numerical(row), kDerivativeTol) << "dfdx(" << row << ", " << index << ")";
    }
  }
}

}  // namespace ocs2::humanoid
