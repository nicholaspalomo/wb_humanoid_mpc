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

#include "humanoid_common_mpc/contact/ContactInputJacobian.h"

#include <cmath>

#include "absl/log/check.h"

namespace ocs2::humanoid {

matrix_t contactForceInputJacobian(const MpcRobotModelBase<scalar_t>& mpcRobotModel, size_t contactPointIndex) {
  constexpr size_t kForceDim = 3;
  constexpr scalar_t kLinearityTolerance = 1e-12;

  const size_t inputDim = mpcRobotModel.getInputDim();
  const vector_t zeroInput = vector_t::Zero(inputDim);
  const vector3_t offset = mpcRobotModel.getContactForce(zeroInput, contactPointIndex);
  CHECK_LE(offset.cwiseAbs().maxCoeff(), kLinearityTolerance)
      << "[contactForceInputJacobian] the contact force of contact " << contactPointIndex
      << " must be linear in the input for it to be probed by unit inputs; the zero input produced " << offset.transpose();

  matrix_t jacobian = matrix_t::Zero(kForceDim, inputDim);
  for (size_t index = 0; index < inputDim; ++index) {
    vector_t probe = vector_t::Zero(inputDim);
    probe(static_cast<long>(index)) = 1.0;
    jacobian.col(static_cast<long>(index)) = mpcRobotModel.getContactForce(probe, contactPointIndex);
  }
  return jacobian;
}

vector_t normalContactForceRow(const MpcRobotModelBase<scalar_t>& mpcRobotModel, size_t contactPointIndex) {
  constexpr long kNormalRow = 2;
  constexpr scalar_t kSignTolerance = 1e-12;

  const vector_t row = contactForceInputJacobian(mpcRobotModel, contactPointIndex).row(kNormalRow);
  CHECK((row.array() >= -kSignTolerance).all()) << "[normalContactForceRow] the normal contact force must be non-negative in the "
                                                   "model's input parameterization, or it cannot serve as a load indicator";
  CHECK_GT(row.cwiseAbs().maxCoeff(), kSignTolerance)
      << "[normalContactForceRow] no input of contact " << contactPointIndex << " produces a normal force";
  return row;
}

}  // namespace ocs2::humanoid
