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

#include "ocs2_centroidal_model/ModelHelperFunctions.h"

#include "ocs2_centroidal_model/AccessHelperFunctions.h"

#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/frames.hpp>

#include <ocs2_core/manifold/UnitQuaternionMath.h>
#include <ocs2_robotic_tools/common/SkewSymmetricMatrix.h>

#include "ocs2_core/Dimensions.h"
#include "ocs2_core/PhysicalConstants.h"

namespace ocs2 {

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
template <typename SCALAR_T>
Eigen::Matrix<SCALAR_T, centroidal_model::kCentroidalMomentumDim, centroidal_model::kBaseVelocityDim> getSingleRigidBodyBaseMomentumMatrix(
    const CentroidalModelInfoTpl<SCALAR_T>& info,
    const Eigen::Matrix<SCALAR_T, Eigen::Dynamic, 1>& q) {
  using vector3_t = Eigen::Matrix<SCALAR_T, kPositionDim, 1>;
  using matrix3_t = Eigen::Matrix<SCALAR_T, kPositionDim, kPositionDim>;
  using matrix6_t = Eigen::Matrix<SCALAR_T, centroidal_model::kCentroidalMomentumDim, centroidal_model::kBaseVelocityDim>;
  // The base velocity v_b = [dp_W, w_B] of the Composite(Translation, Spherical) root: the CoM c = p_W - R r_nom moves
  // at dc = dp_W + [R r_nom]x R w_B, and the nominal inertia about it turns with the base, L = R I_nom R' R w_B.
  const quaternion_coeffs_t<SCALAR_T> baseOrientation =
      q.template segment<kQuaternionCoefficients>(centroidal_model::kBaseQuaternionStartInConfiguration);
  const matrix3_t rotationBaseToWorld = quaternionRotationMatrix(baseOrientation);
  const vector3_t comToBasePositionInWorld = rotationBaseToWorld * info.comToBasePositionNominal;
  const matrix3_t skewSymmetricMap = skewSymmetricMatrix(comToBasePositionInWorld);
  matrix6_t Ab = matrix6_t::Zero();
  Ab.template topLeftCorner<centroidal_model::kLinearMomentumDim, kPositionDim>().diagonal().array() = info.robotMass;
  Ab.template topRightCorner<centroidal_model::kLinearMomentumDim, kRotationTangentDim>().noalias() =
      info.robotMass * skewSymmetricMap * rotationBaseToWorld;
  Ab.template bottomRightCorner<centroidal_model::kAngularMomentumDim, kRotationTangentDim>().noalias() =
      rotationBaseToWorld * info.centroidalInertiaNominal;
  return Ab;
}
template Eigen::Matrix<scalar_t, centroidal_model::kCentroidalMomentumDim, centroidal_model::kBaseVelocityDim>
getSingleRigidBodyBaseMomentumMatrix<scalar_t>(const CentroidalModelInfo&, const vector_t&);
template Eigen::Matrix<ad_scalar_t, centroidal_model::kCentroidalMomentumDim, centroidal_model::kBaseVelocityDim>
getSingleRigidBodyBaseMomentumMatrix<ad_scalar_t>(const CentroidalModelInfoCppAd&, const ad_vector_t&);

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
template <typename SCALAR_T>
void updateCentroidalDynamics(PinocchioInterfaceTpl<SCALAR_T>& interface, const CentroidalModelInfoTpl<SCALAR_T>& info,
                              const Eigen::Matrix<SCALAR_T, Eigen::Dynamic, 1>& q) {
  using vector3_t = Eigen::Matrix<SCALAR_T, kPositionDim, 1>;

  const auto& model = interface.getModel();
  auto& data = interface.getData();

  switch (info.centroidalModelType) {
    case CentroidalModelType::FullCentroidalDynamics: {
      pinocchio::computeCentroidalMap(model, data, q);
      pinocchio::updateFramePlacements(model, data);
      break;
    }
    case CentroidalModelType::SingleRigidBodyDynamics: {
      const vector3_t basePosition = q.template head<kPositionDim>();
      const quaternion_coeffs_t<SCALAR_T> baseOrientation =
          q.template segment<kQuaternionCoefficients>(centroidal_model::kBaseQuaternionStartInConfiguration);
      const vector3_t comToBasePositionInWorld = quaternionRotationMatrix(baseOrientation) * info.comToBasePositionNominal;
      data.Ag = Eigen::Matrix<SCALAR_T, -1, -1>::Zero(centroidal_model::kCentroidalMomentumDim, info.velocityDim);
      data.Ag.template leftCols<centroidal_model::kBaseVelocityDim>() = getSingleRigidBodyBaseMomentumMatrix(info, q);
      data.com[0] = basePosition - comToBasePositionInWorld;
      pinocchio::forwardKinematics(model, data, q);
      pinocchio::updateFramePlacements(model, data);
      break;
    }
    default: {
      throw std::runtime_error("The chosen centroidal model type is not supported.");
      break;
    }
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
template <typename SCALAR_T>
const Eigen::Matrix<SCALAR_T, centroidal_model::kCentroidalMomentumDim, Eigen::Dynamic>& getCentroidalMomentumMatrix(
    const PinocchioInterfaceTpl<SCALAR_T>& interface) {
  return interface.getData().Ag;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
template <typename SCALAR_T>
Eigen::Matrix<SCALAR_T, kPositionDim, 1> getPositionComToContactPointInWorldFrame(const PinocchioInterfaceTpl<SCALAR_T>& interface,
                                                                                  const CentroidalModelInfoTpl<SCALAR_T>& info,
                                                                                  size_t contactIndex) {
  const auto& data = interface.getData();
  return (data.oMf[info.endEffectorFrameIndices[contactIndex]].translation() - data.com[0]);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
template <typename SCALAR_T>
Eigen::Matrix<SCALAR_T, centroidal_model::kCentroidalMomentumDim, 1> getNormalizedCentroidalMomentumRate(
    const PinocchioInterfaceTpl<SCALAR_T>& interface,
    const CentroidalModelInfoTpl<SCALAR_T>& info,
    const Eigen::Matrix<SCALAR_T, Eigen::Dynamic, 1>& input) {
  const Eigen::Matrix<SCALAR_T, kPositionDim, 1> gravityVector(SCALAR_T(0.0), SCALAR_T(0.0), SCALAR_T(-kGravitationalAcceleration));
  Eigen::Matrix<SCALAR_T, centroidal_model::kCentroidalMomentumDim, 1> centroidalMomentumRate;
  centroidalMomentumRate << info.robotMass * gravityVector, Eigen::Matrix<SCALAR_T, centroidal_model::kAngularMomentumDim, 1>::Zero();

  for (size_t i = 0; i < info.numThreeDofContacts; i++) {
    const auto contactForceInWorldFrame = centroidal_model::getContactForces(input, i, info);
    const auto positionComToContactPointInWorldFrame = getPositionComToContactPointInWorldFrame(interface, info, i);
    centroidalMomentumRate.template head<centroidal_model::kLinearMomentumDim>() += contactForceInWorldFrame;
    centroidalMomentumRate.template tail<centroidal_model::kAngularMomentumDim>().noalias() +=
        positionComToContactPointInWorldFrame.cross(contactForceInWorldFrame);
  }  // end of i loop

  for (size_t i = info.numThreeDofContacts; i < info.numThreeDofContacts + info.numSixDofContacts; i++) {
    const auto contactForceInWorldFrame = centroidal_model::getContactForces(input, i, info);
    const auto contactTorqueInWorldFrame = centroidal_model::getContactTorques(input, i, info);
    const auto positionComToContactPointInWorldFrame = getPositionComToContactPointInWorldFrame(interface, info, i);
    centroidalMomentumRate.template head<centroidal_model::kLinearMomentumDim>() += contactForceInWorldFrame;
    centroidalMomentumRate.template tail<centroidal_model::kAngularMomentumDim>().noalias() +=
        positionComToContactPointInWorldFrame.cross(contactForceInWorldFrame) + contactTorqueInWorldFrame;
  }  // end of i loop

  // normalize by the total mass
  centroidalMomentumRate /= info.robotMass;

  return centroidalMomentumRate;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
// Explicit template instantiation
template void updateCentroidalDynamics<scalar_t>(PinocchioInterface&, const CentroidalModelInfo&, const vector_t&);
template void updateCentroidalDynamics<ad_scalar_t>(PinocchioInterfaceCppAd&, const CentroidalModelInfoCppAd&, const ad_vector_t&);

template const Eigen::Matrix<scalar_t, centroidal_model::kCentroidalMomentumDim, Eigen::Dynamic>& getCentroidalMomentumMatrix<scalar_t>(
    const PinocchioInterface&);
template const Eigen::Matrix<ad_scalar_t, centroidal_model::kCentroidalMomentumDim, Eigen::Dynamic>&
getCentroidalMomentumMatrix<ad_scalar_t>(const PinocchioInterfaceCppAd&);

template Eigen::Matrix<scalar_t, kPositionDim, 1> getPositionComToContactPointInWorldFrame<scalar_t>(const PinocchioInterface&,
                                                                                                     const CentroidalModelInfo&,
                                                                                                     size_t);
template Eigen::Matrix<ad_scalar_t, kPositionDim, 1> getPositionComToContactPointInWorldFrame<ad_scalar_t>(const PinocchioInterfaceCppAd&,
                                                                                                           const CentroidalModelInfoCppAd&,
                                                                                                           size_t);

template Eigen::Matrix<scalar_t, centroidal_model::kCentroidalMomentumDim, 1> getNormalizedCentroidalMomentumRate<scalar_t>(
    const PinocchioInterface&, const CentroidalModelInfo&, const vector_t&);
template Eigen::Matrix<ad_scalar_t, centroidal_model::kCentroidalMomentumDim, 1> getNormalizedCentroidalMomentumRate<ad_scalar_t>(
    const PinocchioInterfaceCppAd&, const CentroidalModelInfoCppAd&, const ad_vector_t&);

}  // namespace ocs2
