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
#include <ocs2_pinocchio_interface/PinocchioInterface.h>
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

namespace ocs2::humanoid {

/**
 * @brief Quadratic input effort cost: 0.5 * (u - u_nom)^T * R * (u - u_nom)
 */
class InputQuadraticCost final : public QuadraticStateInputCost {
 public:
  InputQuadraticCost(matrix_t R,
                     size_t stateDim,
                     const SwitchedModelReferenceManager& referenceManager,
                     const PinocchioInterface& pinocchioInterface,
                     const MpcRobotModelBase<scalar_t>& mpcRobotModel);

  ~InputQuadraticCost() override = default;
  InputQuadraticCost* clone() const override { return new InputQuadraticCost(*this); }

 private:
  InputQuadraticCost(const InputQuadraticCost& rhs);

  std::pair<vector_t, vector_t> getStateInputDeviation(scalar_t time,
                                                       const vector_t& state,
                                                       const vector_t& input,
                                                       const TargetTrajectories& targetTrajectories) const override;

  const size_t stateDim_;
  const SwitchedModelReferenceManager* referenceManagerPtr_;
  const PinocchioInterface& pinInterface_;
  const MpcRobotModelBase<scalar_t>* mpcRobotModelPtr_;
};

}  // namespace ocs2::humanoid
