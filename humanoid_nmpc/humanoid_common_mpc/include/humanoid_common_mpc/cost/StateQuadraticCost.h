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

#include <ocs2_core/cost/QuadraticStateInputCost.h>
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

namespace ocs2::humanoid {

/**
 * @brief Quadratic state tracking cost: 0.5 * (x - x_nom)^T * Q * (x - x_nom)
 */
class StateQuadraticCost final : public QuadraticStateInputCost {
 public:
  StateQuadraticCost(matrix_t Q,
                     size_t inputDim,
                     const SwitchedModelReferenceManager& referenceManager);

  ~StateQuadraticCost() override = default;
  StateQuadraticCost* clone() const override { return new StateQuadraticCost(*this); }

 private:
  StateQuadraticCost(const StateQuadraticCost& rhs);

  std::pair<vector_t, vector_t> getStateInputDeviation(scalar_t time,
                                                       const vector_t& state,
                                                       const vector_t& input,
                                                       const TargetTrajectories& targetTrajectories) const override;

  const size_t inputDim_;
  const SwitchedModelReferenceManager* referenceManagerPtr_;
};

}  // namespace ocs2::humanoid
