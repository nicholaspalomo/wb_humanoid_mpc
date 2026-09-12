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

#include <functional>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

#include <ocs2_centroidal_model/AccessHelperFunctions.h>
#include <ocs2_centroidal_model/CentroidalModelPinocchioMapping.h>
#include <ocs2_centroidal_model/ModelHelperFunctions.h>

#include "absl/strings/str_cat.h"

namespace ocs2::humanoid {

namespace {
static constexpr size_t kWrenchDim = 6;
static constexpr size_t kForceDim = 3;
}  // namespace

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
  if (info.numThreeDofContacts != 0 || info.numSixDofContacts != N_CONTACTS) {
    throw std::runtime_error(absl::StrCat("[CentroidalDynamicsBasisInputsAD] Expected exactly ", N_CONTACTS,
                                          " six-DoF contacts and no three-DoF contacts, got ", info.numSixDofContacts, " / ",
                                          info.numThreeDofContacts));
  }
  if (static_cast<size_t>(info.actuatedDofNum) != jointDim_) {
    throw std::runtime_error(absl::StrCat("[CentroidalDynamicsBasisInputsAD] actuatedDofNum (", info.actuatedDofNum,
                                          ") does not match mpc_joint_dim (", jointDim_, ")"));
  }
  const auto& model = pinocchioInterface.getModel();
  for (size_t i = 0; i < N_CONTACTS; ++i) {
    if (B_local_[i].rows() != static_cast<Eigen::Index>(kWrenchDim) || B_local_[i].cols() != static_cast<Eigen::Index>(numBasisPerFoot_)) {
      throw std::runtime_error(absl::StrCat("[CentroidalDynamicsBasisInputsAD] Basis matrix ", i, " has size ", B_local_[i].rows(), "x",
                                            B_local_[i].cols(), ", expected 6x", numBasisPerFoot_));
    }
    const std::string& frameName = modelSettings.contactNames[i];
    if (!model.existFrame(frameName)) {
      throw std::runtime_error(absl::StrCat("[CentroidalDynamicsBasisInputsAD] Contact frame '", frameName, "' does not exist"));
    }
    contactFrameIndices_[i] = model.getFrameId(frameName);
  }

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
  // Encode the basis dimension and a hash of the basis entries so that a cached CppAD library compiled for a
  // different basis (e.g. other friction coefficient, footprint or number of generators) is never reused.
  std::ostringstream oss;
  oss << std::setprecision(12);
  for (const matrix_t& B : localBasisMatrices) {
    oss << B.rows() << 'x' << B.cols() << ':';
    for (Eigen::Index r = 0; r < B.rows(); ++r) {
      for (Eigen::Index c = 0; c < B.cols(); ++c) {
        oss << B(r, c) << ',';
      }
    }
    oss << ';';
  }
  const size_t hash = std::hash<std::string>{}(oss.str());
  std::ostringstream name;
  name << modelName << "_basis" << localBasisMatrices[0].cols() << "_" << std::hex << hash;
  return name.str();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
template <typename SCALAR_T>
VECTOR_T<SCALAR_T> CentroidalDynamicsBasisInputsAD::toWorldFrameWrenchInput(const PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface,
                                                                            const CentroidalModelInfoTpl<SCALAR_T>& info,
                                                                            const VECTOR_T<SCALAR_T>& basisInput) const {
  assert(basisInput.size() == static_cast<Eigen::Index>(basisInputDim_));
  const auto& data = pinocchioInterface.getData();

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
