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

#include "humanoid_common_mpc/contact_planning/problem/StageAccumulator.h"

namespace ocs2::humanoid {

void StageAccumulator::addQuadraticResidual(const Coefficients& xCoefficients,
                                            const Coefficients& uCoefficients,
                                            scalar_t offset,
                                            scalar_t weight) {
  if (weight <= 0.0) return;
  vector_t lx = vector_t::Zero(stage_.numStates());
  vector_t lu = vector_t::Zero(stage_.numInputs());
  for (const std::pair<int, scalar_t>& coefficient : xCoefficients) lx(coefficient.first) += coefficient.second;
  for (const std::pair<int, scalar_t>& coefficient : uCoefficients) lu(coefficient.first) += coefficient.second;
  stage_.Q.noalias() += 2.0 * weight * lx * lx.transpose();
  stage_.q.noalias() += 2.0 * weight * offset * lx;
  if (stage_.numInputs() > 0) {
    stage_.R.noalias() += 2.0 * weight * lu * lu.transpose();
    stage_.S.noalias() += 2.0 * weight * lu * lx.transpose();
    stage_.r.noalias() += 2.0 * weight * offset * lu;
  }
  // The expansion of w (l_x' x + l_u' u + c)^2 has a third part besides the quadratic and the linear one, the constant
  // w c^2, and it used to be dropped here. The reason it was dropped is sound as far as the QP itself goes: a constant
  // is invisible to the solver, it moves neither the minimiser nor the KKT residuals, and it cancels out of every
  // comparison between two solutions of ONE assembled problem - which covers the branch-and-bound bounds and their
  // absoluteGap test, EventShiftLocalSearchStage and HeadingRelinearisationStage, where both sides carry the same
  // constant. What the reasoning misses is that CadenceStretchStage::afterSearch compares objectives across node
  // grids: it re-assembles the problem at s * dt and scores the result against the incumbent assembled at dt. The
  // residual offsets of the shipped terms depend on dt - StepLengthCost's nominal displacement is
  // v_cmd * dt * T_stride / T_swing and TerminalDcmCost's weight carries exp(2 omega dt) - so the dropped constant
  // grows with the stretch, every stretched candidate's reported objective was depressed by an amount monotone in the
  // stretch, and the stage accepted stretches whose true cost was higher than the incumbent's, overshooting the
  // cadence it is supposed to pick. Accumulating the constant here makes the assembled objective the functional this
  // class documents. It leaves the solutions themselves bit-identical, because the constant never reaches HPIPM.
  stage_.constant += weight * offset * offset;
}

}  // namespace ocs2::humanoid
