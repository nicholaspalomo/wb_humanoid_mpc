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

#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include "humanoid_centroidal_mpc/dynamics/CentroidalDynamicsBasisInputsAD.h"

#include <stdexcept>
#include <utility>

#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

#include <ocs2_centroidal_model/AccessHelperFunctions.h>
#include <ocs2_centroidal_model/CentroidalModelPinocchioMapping.h>
#include <ocs2_centroidal_model/ModelHelperFunctions.h>

#include "absl/strings/str_cat.h"

#include "humanoid_common_mpc/contact/ContactWrenchConeBasisMatrix.h"

namespace ocs2::humanoid {

namespace {
constexpr size_t kWrenchDim = 6;
constexpr size_t kForceDim = 3;

/** Frame index of every contact; validate() has established that each frame exists. */
std::array<pinocchio::FrameIndex, N_CONTACTS> contactFrameIndices(const PinocchioInterface& pinocchioInterface,
                                                                  const ModelSettings& modelSettings) {
  const pinocchio::ModelTpl<scalar_t>& model = pinocchioInterface.getModel();
  std::array<pinocchio::FrameIndex, N_CONTACTS> indices;
  for (size_t i = 0; i < N_CONTACTS; ++i) {
    indices[i] = model.getFrameId(modelSettings.contactNames[i]);
  }
  return indices;
}
}  // namespace

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
absl::Status CentroidalDynamicsBasisInputsAD::validate(const PinocchioInterface& pinocchioInterface,
                                                       const CentroidalModelInfo& info,
                                                       const ModelSettings& modelSettings,
                                                       const std::array<matrix_t, N_CONTACTS>& localBasisMatrices) {
  if (info.numThreeDofContacts != 0 || info.numSixDofContacts != N_CONTACTS || modelSettings.contactNames.size() != N_CONTACTS) {
    return absl::InvalidArgumentError(
        absl::StrCat("[CentroidalDynamicsBasisInputsAD] basis-vector contact inputs need exactly ", N_CONTACTS,
                     " six-DoF contacts and no three-DoF contacts; model_settings.contactNames6DoF gives ", info.numSixDofContacts,
                     " six-DoF and ", info.numThreeDofContacts, " three-DoF contacts."));
  }
  if (static_cast<size_t>(info.actuatedDofNum) != modelSettings.mpc_joint_dim) {
    return absl::InvalidArgumentError(absl::StrCat("[CentroidalDynamicsBasisInputsAD] the centroidal model actuates ", info.actuatedDofNum,
                                                   " joints but model_settings.fixedJointNames leaves ", modelSettings.mpc_joint_dim,
                                                   " MPC joints; they must agree."));
  }
  const pinocchio::ModelTpl<scalar_t>& model = pinocchioInterface.getModel();
  const Eigen::Index numBasisPerFoot = localBasisMatrices[0].cols();
  for (size_t i = 0; i < N_CONTACTS; ++i) {
    if (localBasisMatrices[i].rows() != static_cast<Eigen::Index>(kWrenchDim) || localBasisMatrices[i].cols() != numBasisPerFoot) {
      return absl::InternalError(absl::StrCat("[CentroidalDynamicsBasisInputsAD] basis matrix ", i, " has size ",
                                              localBasisMatrices[i].rows(), "x", localBasisMatrices[i].cols(), ", expected ", kWrenchDim,
                                              "x", numBasisPerFoot, "."));
    }
    const std::string& frameName = modelSettings.contactNames[i];
    if (!model.existFrame(frameName)) {
      return absl::InvalidArgumentError(absl::StrCat("[CentroidalDynamicsBasisInputsAD] model_settings.contactNames6DoF entry '", frameName,
                                                     "' is not a frame of the robot model."));
    }
  }
  return absl::OkStatus();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
absl::StatusOr<std::unique_ptr<CentroidalDynamicsBasisInputsAD>> CentroidalDynamicsBasisInputsAD::Create(
    const PinocchioInterface& pinocchioInterface,
    const CentroidalModelInfo& info,
    const std::string& modelName,
    const ModelSettings& modelSettings,
    const std::array<matrix_t, N_CONTACTS>& localBasisMatrices) {
  const absl::Status status = validate(pinocchioInterface, info, modelSettings, localBasisMatrices);
  if (!status.ok()) {
    return status;
  }
  // Not std::make_unique: it cannot reach the private constructor.
  return std::unique_ptr<CentroidalDynamicsBasisInputsAD>(
      new CentroidalDynamicsBasisInputsAD(pinocchioInterface, info, modelName, modelSettings, localBasisMatrices));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
CentroidalDynamicsBasisInputsAD::CentroidalDynamicsBasisInputsAD(const PinocchioInterface& pinocchioInterface,
                                                                 const CentroidalModelInfo& info,
                                                                 const std::string& modelName,
                                                                 const ModelSettings& modelSettings,
                                                                 const std::array<matrix_t, N_CONTACTS>& localBasisMatrices)
    : SystemDynamicsBaseAD(),
      pinocchioInterfaceCppAd_(pinocchioInterface.toCppAd()),
      infoCppAd_(info.toCppAd()),
      B_local_(localBasisMatrices),
      numBasisPerFoot_(static_cast<size_t>(localBasisMatrices[0].cols())),
      jointDim_(modelSettings.mpc_joint_dim),
      basisInputDim_(static_cast<size_t>(localBasisMatrices[0].cols()) * N_CONTACTS + modelSettings.mpc_joint_dim) {
  // Create() has validated the arguments.
  contactFrameIndices_ = contactFrameIndices(pinocchioInterface, modelSettings);

  initialize(info.stateDim, basisInputDim_, uniqueModelName(modelName, localBasisMatrices), modelSettings.modelFolderCppAd,
             modelSettings.recompileLibrariesCppAd, modelSettings.verboseCppAd);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
CentroidalDynamicsBasisInputsAD::CentroidalDynamicsBasisInputsAD(const CentroidalDynamicsBasisInputsAD& rhs)
    : SystemDynamicsBaseAD(rhs),
      pinocchioInterfaceCppAd_(rhs.pinocchioInterfaceCppAd_),
      infoCppAd_(rhs.infoCppAd_),
      B_local_(rhs.B_local_),
      contactFrameIndices_(rhs.contactFrameIndices_),
      numBasisPerFoot_(rhs.numBasisPerFoot_),
      jointDim_(rhs.jointDim_),
      basisInputDim_(rhs.basisInputDim_) {}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::string CentroidalDynamicsBasisInputsAD::uniqueModelName(const std::string& modelName,
                                                             const std::array<matrix_t, N_CONTACTS>& localBasisMatrices) {
  // A cached CppAD library compiled for a different basis (another generator set, friction coefficient, footprint or
  // number of generators) must never be reused, so the name carries the basis' content key.
  return absl::StrCat(modelName, "_", basisInputsLibraryKey(localBasisMatrices));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
template <typename SCALAR_T>
VECTOR_T<SCALAR_T> CentroidalDynamicsBasisInputsAD::toWorldFrameWrenchInput(const PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface,
                                                                            const CentroidalModelInfoTpl<SCALAR_T>& info,
                                                                            const VECTOR_T<SCALAR_T>& basisInput) const {
  assert(basisInput.size() == static_cast<Eigen::Index>(basisInputDim_));
  const pinocchio::DataTpl<SCALAR_T>& data = pinocchioInterface.getData();

  VECTOR_T<SCALAR_T> wrenchInput = VECTOR_T<SCALAR_T>::Zero(info.inputDim);
  for (size_t i = 0; i < N_CONTACTS; ++i) {
    const VECTOR_T<SCALAR_T> lambda = basisInput.segment(numBasisPerFoot_ * i, numBasisPerFoot_);
    const VECTOR6_T<SCALAR_T> wrenchLocal = B_local_[i].template cast<SCALAR_T>() * lambda;
    const MATRIX3_T<SCALAR_T> w_R_l = data.oMf[contactFrameIndices_[i]].rotation();
    // Contact i is the i-th six-DoF contact (there are no three-DoF contacts, checked in the constructor).
    centroidal_model::getContactForces(wrenchInput, i, info) = w_R_l * wrenchLocal.template head<kForceDim>();
    centroidal_model::getContactTorques(wrenchInput, i, info) = w_R_l * wrenchLocal.template tail<kForceDim>();
  }
  centroidal_model::getJointVelocities(wrenchInput, info) = basisInput.tail(jointDim_);
  return wrenchInput;
}

template vector_t CentroidalDynamicsBasisInputsAD::toWorldFrameWrenchInput<scalar_t>(const PinocchioInterfaceTpl<scalar_t>&,
                                                                                     const CentroidalModelInfoTpl<scalar_t>&,
                                                                                     const vector_t&) const;
template ad_vector_t CentroidalDynamicsBasisInputsAD::toWorldFrameWrenchInput<ad_scalar_t>(const PinocchioInterfaceTpl<ad_scalar_t>&,
                                                                                           const CentroidalModelInfoTpl<ad_scalar_t>&,
                                                                                           const ad_vector_t&) const;

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
ad_vector_t CentroidalDynamicsBasisInputsAD::systemFlowMap(ad_scalar_t time,
                                                           const ad_vector_t& state,
                                                           const ad_vector_t& input,
                                                           const ad_vector_t& parameters) const {
  // Work on local copies: this method is const and the tape must not depend on shared mutable state.
  PinocchioInterfaceCppAd pinocchioInterfaceCppAd = pinocchioInterfaceCppAd_;
  CentroidalModelPinocchioMappingCppAd mappingCppAd(infoCppAd_);
  mappingCppAd.setPinocchioInterface(pinocchioInterfaceCppAd);
  const CentroidalModelInfoCppAd& info = infoCppAd_;

  const ad_vector_t qPinocchio = mappingCppAd.getPinocchioJointPosition(state);
  // Updates the centroidal quantities AND the frame placements needed for the contact frame rotations.
  updateCentroidalDynamics(pinocchioInterfaceCppAd, info, qPinocchio);

  // λ → world-frame wrench input of the underlying centroidal model.
  const ad_vector_t wrenchInput = toWorldFrameWrenchInput<ad_scalar_t>(pinocchioInterfaceCppAd, info, input);

  ad_vector_t stateDerivative(info.stateDim);

  // compute center of mass acceleration and derivative of the normalized angular momentum
  centroidal_model::getNormalizedMomentum(stateDerivative, info) =
      getNormalizedCentroidalMomentumRate(pinocchioInterfaceCppAd, info, wrenchInput);

  // derivatives of the floating base variables + joint velocities
  centroidal_model::getGeneralizedCoordinates(stateDerivative, info) = mappingCppAd.getPinocchioJointVelocity(state, wrenchInput);

  return stateDerivative;
}

}  // namespace ocs2::humanoid
