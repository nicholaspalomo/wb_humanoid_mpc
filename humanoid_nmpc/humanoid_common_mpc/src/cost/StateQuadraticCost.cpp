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

#include "humanoid_common_mpc/cost/StateQuadraticCost.h"

namespace ocs2::humanoid {

StateQuadraticCost::StateQuadraticCost(matrix_t Q,
                                       size_t inputDim,
                                       const SwitchedModelReferenceManager& referenceManager)
    : QuadraticStateInputCost(std::move(Q), matrix_t::Zero(inputDim, inputDim)),
      inputDim_(inputDim),
      referenceManagerPtr_(&referenceManager) {}

StateQuadraticCost::StateQuadraticCost(const StateQuadraticCost& rhs)
    : QuadraticStateInputCost(rhs),
      inputDim_(rhs.inputDim_),
      referenceManagerPtr_(rhs.referenceManagerPtr_) {}

std::pair<vector_t, vector_t> StateQuadraticCost::getStateInputDeviation(
    scalar_t time,
    const vector_t& state,
    const vector_t& input,
    const TargetTrajectories& targetTrajectories) const {
  const vector_t xNominal = referenceManagerPtr_->getDesiredState(targetTrajectories, state, time);
  return {state - xNominal, vector_t::Zero(inputDim_)};
}

}  // namespace ocs2::humanoid
