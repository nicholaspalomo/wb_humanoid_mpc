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

#include "ocs2_centroidal_model/FactoryFunctions.h"

#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/centroidal.hpp>

#include "ocs2_core/Dimensions.h"

namespace ocs2 {
namespace centroidal_model {

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
CentroidalModelInfo createCentroidalModelInfo(const PinocchioInterface& interface, const CentroidalModelType& type,
                                              const vector_t& nominalJointAngles, const std::vector<std::string>& threeDofContactNames,
                                              const std::vector<std::string>& sixDofContactNames) {
  const auto& model = interface.getModel();
  auto data = interface.getData();

  // The root joint is Composite(Translation, Spherical), the position and the unit quaternion (x, y, z, w) of the base:
  // 7 configuration and 6 velocity entries of the base, the joints after them.
  if (model.nv != nominalJointAngles.size() + kBaseVelocityDim || model.nq != model.nv + 1) {
    throw std::runtime_error("[CentroidalModelInfo] the model has nq = " + std::to_string(model.nq) + " and nv = " + std::to_string(model.nv) +
                             ", but a Composite(Translation, Spherical) root joint with nominalJointAngles.size() = " +
                             std::to_string(nominalJointAngles.size()) + " joints has nq = nv + 1 and nv = joints + 6");
  }

  CentroidalModelInfoTpl<scalar_t> info;
  info.centroidalModelType = type;
  info.numThreeDofContacts = threeDofContactNames.size();
  info.numSixDofContacts = sixDofContactNames.size();
  info.configurationDim = model.nq;
  info.velocityDim = model.nv;
  info.actuatedDofNum = info.velocityDim - kBaseVelocityDim;
  info.stateDim = info.configurationDim + kCentroidalMomentumDim;
  info.stateTangentDim = info.velocityDim + kCentroidalMomentumDim;
  info.inputDim = info.actuatedDofNum + kForceDim * info.numThreeDofContacts + kWrenchDim * info.numSixDofContacts;
  info.robotMass = pinocchio::computeTotalMass(model);

  for (const auto& name : threeDofContactNames) {
    info.endEffectorFrameIndices.push_back(model.getFrameId(name));
  }

  for (const auto& name : sixDofContactNames) {
    info.endEffectorFrameIndices.push_back(model.getFrameId(name));
  }

  // make sure the nominal base frame is aligned with the world frame: the base at the origin, the identity quaternion
  info.qPinocchioNominal.resize(model.nq);
  info.qPinocchioNominal << vector_t::Zero(kPositionDim), 0.0, 0.0, 0.0, 1.0, nominalJointAngles;
  info.centroidalInertiaNominal.setZero();
  info.comToBasePositionNominal.setZero();
  if (info.centroidalModelType == CentroidalModelType::SingleRigidBodyDynamics) {
    const vector_t vPinocchioNominal = vector_t::Zero(info.velocityDim);
    pinocchio::ccrba(model, data, info.qPinocchioNominal, vPinocchioNominal);
    info.centroidalInertiaNominal = data.Ig.inertia().matrix();
    info.comToBasePositionNominal = info.qPinocchioNominal.template head<kPositionDim>() - data.com[0];
  }

  return info;
}

}  // namespace centroidal_model
}  // namespace ocs2
