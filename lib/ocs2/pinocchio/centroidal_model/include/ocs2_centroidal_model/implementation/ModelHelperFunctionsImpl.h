/******************************************************************************
Copyright (c) 2020, Farbod Farshidian. All rights reserved.

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

#include <ocs2_core/manifold/UnitQuaternionMath.h>

#include "ocs2_core/Dimensions.h"

#include "ocs2_centroidal_model/AccessHelperFunctions.h"

namespace ocs2 {

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
template <typename SCALAR_T>
Eigen::Matrix<SCALAR_T, centroidal_model::kBaseVelocityDim, centroidal_model::kCentroidalMomentumDim>
computeFloatingBaseCentroidalMomentumMatrixInverse(
    const Eigen::Matrix<SCALAR_T, centroidal_model::kCentroidalMomentumDim, centroidal_model::kBaseVelocityDim>& Ab) {
  using centroidal_model::kAngularMomentumDim;
  using centroidal_model::kAngularMomentumStartIndex;
  using centroidal_model::kBaseAngularVelocityStartInVelocity;
  using centroidal_model::kBaseVelocityDim;
  using centroidal_model::kCentroidalMomentumDim;
  using centroidal_model::kLinearMomentumDim;
  // Ab = [m I, Ab_12; 0, Ab_22]: its rows are the linear and the angular momentum, its columns dp_W and w_B.
  const SCALAR_T mass = Ab(0, 0);
  Eigen::Matrix<SCALAR_T, kRotationTangentDim, kAngularMomentumDim> Ab_22_inv =
      Ab.template block<kAngularMomentumDim, kRotationTangentDim>(kAngularMomentumStartIndex, kBaseAngularVelocityStartInVelocity)
          .inverse();
  Eigen::Matrix<SCALAR_T, kBaseVelocityDim, kCentroidalMomentumDim> Ab_inv =
      Eigen::Matrix<SCALAR_T, kBaseVelocityDim, kCentroidalMomentumDim>::Zero();
  Ab_inv << 1.0 / mass * Eigen::Matrix<SCALAR_T, kPositionDim, kLinearMomentumDim>::Identity(),
      -1.0 / mass * Ab.template block<kLinearMomentumDim, kRotationTangentDim>(0, kBaseAngularVelocityStartInVelocity) * Ab_22_inv,
      Eigen::Matrix<SCALAR_T, kRotationTangentDim, kLinearMomentumDim>::Zero(), Ab_22_inv;
  return Ab_inv;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
template <typename SCALAR_T>
Eigen::Matrix<SCALAR_T, Eigen::Dynamic, 1> computeConfigurationDerivative(const CentroidalModelInfoTpl<SCALAR_T>& info,
                                                                          const Eigen::Matrix<SCALAR_T, Eigen::Dynamic, 1>& state,
                                                                          const Eigen::Matrix<SCALAR_T, Eigen::Dynamic, 1>& vPinocchio) {
  assert(state.rows() == info.stateDim);
  assert(vPinocchio.rows() == info.velocityDim);
  const quaternion_coeffs_t<SCALAR_T> xi = centroidal_model::getBaseQuaternion(state, info);
  Eigen::Matrix<SCALAR_T, Eigen::Dynamic, 1> configurationRate(info.configurationDim);
  configurationRate.template head<kPositionDim>() = vPinocchio.template head<kPositionDim>();
  configurationRate.template segment<kQuaternionCoefficients>(centroidal_model::kBaseQuaternionStartInConfiguration).noalias() =
      SCALAR_T(kQuaternionHalfAngleFactor) *  // d/dt xi = 1/2 G(xi) w_B
      (quaternionRateMatrix(xi) * vPinocchio.template segment<kRotationTangentDim>(centroidal_model::kBaseAngularVelocityStartInVelocity));
  configurationRate.tail(info.actuatedDofNum) = vPinocchio.tail(info.actuatedDofNum);
  return configurationRate;
}

}  // namespace ocs2
