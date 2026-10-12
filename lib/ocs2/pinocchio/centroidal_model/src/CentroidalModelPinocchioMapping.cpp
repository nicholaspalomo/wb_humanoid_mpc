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

#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include "absl/base/nullability.h"

#include "ocs2_centroidal_model/CentroidalModelPinocchioMapping.h"

#include <stdexcept>

#include <ocs2_core/manifold/UnitQuaternionMath.h>

#include "ocs2_centroidal_model/AccessHelperFunctions.h"
#include "ocs2_centroidal_model/ModelHelperFunctions.h"

namespace ocs2 {

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
template <typename SCALAR>
CentroidalModelPinocchioMappingTpl<SCALAR>::CentroidalModelPinocchioMappingTpl(CentroidalModelInfoTpl<SCALAR> centroidalModelInfo)
    : pinocchioInterfacePtr_(nullptr), centroidalModelInfo_(std::move(centroidalModelInfo)) {}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
template <typename SCALAR>
CentroidalModelPinocchioMappingTpl<SCALAR>::CentroidalModelPinocchioMappingTpl(const CentroidalModelPinocchioMappingTpl& rhs)
    : pinocchioInterfacePtr_(nullptr), centroidalModelInfo_(rhs.centroidalModelInfo_) {}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
template <typename SCALAR>
CentroidalModelPinocchioMappingTpl<SCALAR>* absl_nonnull CentroidalModelPinocchioMappingTpl<SCALAR>::clone() const {
  return new CentroidalModelPinocchioMappingTpl<SCALAR>(*this);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
template <typename SCALAR>
void CentroidalModelPinocchioMappingTpl<SCALAR>::setPinocchioInterface(const PinocchioInterfaceTpl<SCALAR>& pinocchioInterface) {
  pinocchioInterfacePtr_ = &pinocchioInterface;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
template <typename SCALAR>
auto CentroidalModelPinocchioMappingTpl<SCALAR>::getPinocchioJointPosition(const vector_t& state) const -> vector_t {
  vector_t qPinocchio = centroidal_model::getGeneralizedCoordinates(state, centroidalModelInfo_);
  const quaternion_coeffs_t<SCALAR> orientation = centroidal_model::getBaseQuaternion(state, centroidalModelInfo_);
  qPinocchio.template segment<4>(3) = quaternionSafeNormalize(orientation);
  return qPinocchio;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
template <typename SCALAR>
auto CentroidalModelPinocchioMappingTpl<SCALAR>::getPinocchioJointVelocity(const vector_t& state, const vector_t& input) const -> vector_t {
  const auto& info = centroidalModelInfo_;
  assert(info.stateDim == state.rows());
  assert(info.inputDim == input.rows());

  const auto& A = getCentroidalMomentumMatrix(*pinocchioInterfacePtr_);
  const Eigen::Matrix<SCALAR, 6, 6> Ab = A.template leftCols<6>();
  const auto Ab_inv = computeFloatingBaseCentroidalMomentumMatrixInverse(Ab);

  const auto jointVelocities = centroidal_model::getJointVelocities(input, info).head(info.actuatedDofNum);

  Eigen::Matrix<SCALAR, 6, 1> momentum = info.robotMass * centroidal_model::getNormalizedMomentum(state, info);
  if (info.centroidalModelType == CentroidalModelType::FullCentroidalDynamics) {
    momentum.noalias() -= A.rightCols(info.actuatedDofNum) * jointVelocities;
  }

  vector_t vPinocchio(info.velocityDim);
  vPinocchio.template head<6>().noalias() = Ab_inv * momentum;
  vPinocchio.tail(info.actuatedDofNum) = jointVelocities;

  return vPinocchio;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
template <typename SCALAR>
auto CentroidalModelPinocchioMappingTpl<SCALAR>::getOcs2Jacobian(const vector_t& /*state*/, const matrix_t& /*Jq*/,
                                                                const matrix_t& /*Jv*/) const -> std::pair<matrix_t, matrix_t> {
  throw std::runtime_error(
      "[CentroidalModelPinocchioMapping] getOcs2Jacobian is not supported on the quaternion base orientation: the terms of "
      "the centroidal model are differentiated by CppAD only.");
}

// explicit template instantiation
template class ocs2::CentroidalModelPinocchioMappingTpl<ocs2::scalar_t>;
template class ocs2::CentroidalModelPinocchioMappingTpl<ocs2::ad_scalar_t>;

}  // namespace ocs2
