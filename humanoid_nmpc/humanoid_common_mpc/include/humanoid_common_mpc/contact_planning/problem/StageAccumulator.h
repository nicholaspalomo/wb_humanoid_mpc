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

#include "humanoid_common_mpc/contact_planning/OcpQpHpipm.h"
#include "humanoid_common_mpc/contact_planning/problem/LipCoefficients.h"

namespace ocs2::humanoid {

/**
 * Accumulates the quadratic cost of one stage: every term adds w (l_x' x + l_u' u + c)^2 to 0.5 x'Qx + 0.5 u'Ru + u'Sx +
 * q'x + r'u + constant (the OcpQpStage convention), in the order the terms are applied. The order is the accumulation
 * order of the floating point sums, so the same term list gives bit-identical matrices on every assembly.
 *
 * The constant part of the expansion, w c^2, is accumulated into OcpQpStage::constant rather than discarded. It is
 * irrelevant to the solver, but it is what makes the assembled objective the functional the terms document, and
 * therefore what makes the objectives of two problems assembled on different node grids comparable.
 */
class StageAccumulator {
 public:
  explicit StageAccumulator(OcpQpStage& stage) : stage_(stage) {}

  /**
   * Adds weight * (sum_i xc_i x_i + sum_j uc_j u_j + offset)^2; nothing for weight <= 0. The quadratic and linear parts
   * go into Q, q, R, S, r and the constant weight * offset^2 into OcpQpStage::constant, so that the stage evaluates to
   * the full residual and not only to its solution-relevant part.
   */
  void addQuadraticResidual(const Coefficients& xCoefficients, const Coefficients& uCoefficients, scalar_t offset, scalar_t weight);
  /** Adds weight to the diagonal of Q. */
  void addStateRegularization(scalar_t weight) { stage_.Q.diagonal().array() += weight; }
  /** Adds weight to the diagonal of R (no-op on a node without inputs). */
  void addInputRegularization(scalar_t weight) {
    if (stage_.numInputs() > 0) stage_.R.diagonal().array() += weight;
  }

  OcpQpStage& stage() { return stage_; }
  int numStates() const { return stage_.numStates(); }
  int numInputs() const { return stage_.numInputs(); }

 private:
  OcpQpStage& stage_;
};

}  // namespace ocs2::humanoid
