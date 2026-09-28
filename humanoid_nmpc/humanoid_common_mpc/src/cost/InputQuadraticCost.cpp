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

#include "humanoid_common_mpc/cost/InputQuadraticCost.h"

#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"

namespace ocs2::humanoid {

InputQuadraticCost::InputQuadraticCost(matrix_t R,
                                       size_t stateDim,
                                       const SwitchedModelReferenceManager& referenceManager,
                                       const MpcRobotModelBase<scalar_t>& mpcRobotModel)
    : QuadraticStateInputCost(matrix_t::Zero(stateDim, stateDim), std::move(R)),
      stateDim_(stateDim),
      referenceManagerPtr_(&referenceManager),
      mpcRobotModelPtr_(&mpcRobotModel) {}

InputQuadraticCost::InputQuadraticCost(const InputQuadraticCost& rhs)
    : QuadraticStateInputCost(rhs),
      stateDim_(rhs.stateDim_),
      referenceManagerPtr_(rhs.referenceManagerPtr_),
      mpcRobotModelPtr_(rhs.mpcRobotModelPtr_) {}

std::pair<vector_t, vector_t> InputQuadraticCost::getStateInputDeviation(scalar_t time,
                                                                         const vector_t& state,
                                                                         const vector_t& input,
                                                                         const TargetTrajectories& targetTrajectories) const {
  // The contact-force reference, from the reference manager rather than computed here. It is still weight compensation
  // and nothing else on every robot shipped in this repository - getDesiredInput() makes exactly the call this line
  // used to make when no wrench heuristic is listed, so this is a no-op refactor there - but it is now the one place
  // that reference is decided, which is what lets Bledt's H_f heuristics shape it
  // (humanoid_nmpc/docs/locomotion_heuristics/README.md). The reference manager is also the only object that knows
  // the contact flags at an arbitrary node time, which is why the reference cannot live in targetTrajectories: the
  // stance set changes several times inside one horizon and that trajectory has three knots.
  const vector_t uNominal = referenceManagerPtr_->getDesiredInput(targetTrajectories, state, time);
  return {vector_t::Zero(stateDim_), input - uNominal};
}

}  // namespace ocs2::humanoid
