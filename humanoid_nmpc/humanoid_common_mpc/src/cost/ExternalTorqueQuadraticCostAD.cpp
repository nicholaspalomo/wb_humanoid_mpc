/******************************************************************************
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.

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

#include "humanoid_common_mpc/cost/ExternalTorqueQuadraticCostAD.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

namespace ocs2::humanoid {

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

absl::StatusOr<std::unique_ptr<ExternalTorqueQuadraticCostAD>> ExternalTorqueQuadraticCostAD::Create(
    size_t endEffectorIndex,
    const Config& config,
    const SwitchedModelReferenceManager& referenceManager,
    const PinocchioInterface& pinocchioInterface,
    const MpcRobotModelBase<ad_scalar_t>& mpcRobotModelAD,
    const ModelSettings& modelSettings) {
  if (static_cast<size_t>(config.weights.size()) != config.activeJointNames.size()) {
    return absl::InvalidArgumentError(absl::StrCat("[ExternalTorqueQuadraticCostAD] ", config.weights.size(), " weights for ",
                                                   config.activeJointNames.size(), " activeJointNames: give one weight per joint."));
  }
  // Resolved here, once: the cost function that is taped indexes the joint torques by them.
  std::vector<size_t> activeJointIndices;
  activeJointIndices.reserve(config.activeJointNames.size());
  for (const std::string& jointName : config.activeJointNames) {
    const absl::StatusOr<size_t> index = mpcRobotModelAD.findJointIndex(jointName);
    if (!index.ok()) {
      return absl::NotFoundError(absl::StrCat("[ExternalTorqueQuadraticCostAD] activeJointNames: ", index.status().message()));
    }
    activeJointIndices.push_back(*index);
  }
  // The constructor is private, so std::make_unique cannot reach it.
  return absl::WrapUnique(new ExternalTorqueQuadraticCostAD(endEffectorIndex, config, std::move(activeJointIndices), referenceManager,
                                                            pinocchioInterface, mpcRobotModelAD, modelSettings));
}

ExternalTorqueQuadraticCostAD::ExternalTorqueQuadraticCostAD(size_t endEffectorIndex,
                                                             const Config& config,
                                                             std::vector<size_t> activeJointIndices,
                                                             const SwitchedModelReferenceManager& referenceManager,
                                                             const PinocchioInterface& pinocchioInterface,
                                                             const MpcRobotModelBase<ad_scalar_t>& mpcRobotModelAD,
                                                             const ModelSettings& modelSettings)
    : StateInputCostGaussNewtonAd(),
      contactPointIndex_(endEffectorIndex),
      frameID_(pinocchioInterface.getModel().getFrameId(modelSettings.contactNames[endEffectorIndex])),
      n_parameters_(1 + config.weights.size()),
      sqrtWeights_(config.weights.cwiseSqrt()),
      activeJointNames_(config.activeJointNames),
      activeJointIndices_(std::move(activeJointIndices)),
      referenceManagerPtr_(&referenceManager),
      pinocchioInterfaceCppAd_(pinocchioInterface.toCppAd()),
      mpcRobotModelADPtr_(mpcRobotModelAD.clone()) {
  LOG(INFO) << "Initialized ExternalTorqueQuadraticCostAD with weights: " << config.weights.cwiseSqrt();
  const std::string endEffectorName = modelSettings.contactNames[endEffectorIndex];
  LOG(INFO) << "Frame name: " << endEffectorName;

  LOG(INFO) << "Frame ID: " << frameID_;
  LOG(INFO) << "State dim: " << mpcRobotModelADPtr_->getStateDim();
  LOG(INFO) << "Input dim: " << mpcRobotModelADPtr_->getInputDim();
  LOG(INFO) << "Parameters dim: " << n_parameters_;

  initialize(mpcRobotModelADPtr_->getStateDim(), mpcRobotModelADPtr_->getInputDim(), n_parameters_,
             endEffectorName + "_ExternalTorqueQuadraticCost", modelSettings.modelFolderCppAd, modelSettings.recompileLibrariesCppAd,
             modelSettings.verboseCppAd);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

ExternalTorqueQuadraticCostAD::ExternalTorqueQuadraticCostAD(const ExternalTorqueQuadraticCostAD& other)
    : StateInputCostGaussNewtonAd(other),
      contactPointIndex_(other.contactPointIndex_),
      frameID_(other.frameID_),
      n_parameters_(other.n_parameters_),
      sqrtWeights_(other.sqrtWeights_),
      activeJointNames_(other.activeJointNames_),
      activeJointIndices_(other.activeJointIndices_),
      referenceManagerPtr_(other.referenceManagerPtr_),
      pinocchioInterfaceCppAd_(other.pinocchioInterfaceCppAd_),
      mpcRobotModelADPtr_(other.mpcRobotModelADPtr_->clone()),
      // isActive_ is copied deliberately: the SQP solver clones the whole problem once per worker thread,
      // and a copy constructor that dropped this flag silently reverted a deactivated term to active.
      isActive_(other.isActive_) {}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

bool ExternalTorqueQuadraticCostAD::isActive(scalar_t time) const {
  if (!isActive_) return false;
  return referenceManagerPtr_->getContactFlags(time)[contactPointIndex_];
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

vector_t ExternalTorqueQuadraticCostAD::getParameters(scalar_t time,
                                                      const TargetTrajectories& /*targetTrajectories*/,
                                                      const PreComputation& /*preComputation*/) const {
  vector_t params(n_parameters_);
  const scalar_t impactProximityScaler = referenceManagerPtr_->getSwingTrajectoryPlanner()->getImpactProximityFactor(
      (1 - contactPointIndex_), time);  // Get impactproximity scaler from swing foot.
  params << sqrtWeights_, impactProximityScaler;
  return params;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

ad_vector_t ExternalTorqueQuadraticCostAD::costVectorFunction(ad_scalar_t /*time*/,
                                                              const ad_vector_t& state,
                                                              const ad_vector_t& input,
                                                              const ad_vector_t& parameters) {
  const PinocchioInterfaceCppAd::Model& model = pinocchioInterfaceCppAd_.getModel();
  PinocchioInterfaceCppAd::Data& data = pinocchioInterfaceCppAd_.getData();

  const ad_vector_t q = mpcRobotModelADPtr_->getGeneralizedCoordinates(state);
  ad_matrix_t J_ee = ad_matrix_t::Zero(6, mpcRobotModelADPtr_->getGenCoordinatesDim());
  pinocchio::computeFrameJacobian(model, data, q, frameID_, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J_ee);

  // The Jacobian is LOCAL_WORLD_ALIGNED, so the wrench must be expressed in the world frame. The state-aware
  // accessor is frame-correct for both wrench-space and basis-vector (local-frame) input parameterizations.
  ad_vector_t tauExt = J_ee.transpose() * mpcRobotModelADPtr_->getContactWrenchInWorldFrame(state, input, contactPointIndex_);

  ad_vector_t tauExtActive = ad_vector_t::Zero(sqrtWeights_.size());
  for (Eigen::Index i = 0; i < sqrtWeights_.size(); ++i) {
    tauExtActive[i] = tauExt[6 + activeJointIndices_[static_cast<size_t>(i)]];
  }

  const ad_vector_t sqrtWeightsAD = parameters.head(sqrtWeights_.size());
  const ad_scalar_t midSwingScaler =
      1 - parameters[sqrtWeights_.size()];  // large when in the middle of the swing phase, close when close to impact.

  return tauExtActive.cwiseProduct(sqrtWeightsAD) * midSwingScaler;  // multiply with weights
}

}  // namespace ocs2::humanoid
