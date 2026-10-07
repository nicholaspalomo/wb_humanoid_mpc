/******************************************************************************
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.
Copyright (c) 2024, 1X Technologies. All rights reserved.

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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include "humanoid_centroidal_mpc/cost/ICPCost.h"

#include <cmath>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "ocs2_centroidal_model/ModelHelperFunctions.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "ocs2_pinocchio_interface/PinocchioStateInputMapping.h"
#include "ocs2_robotic_tools/common/RotationTransforms.h"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/model.hpp"

#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"

namespace ocs2::humanoid {

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

ICPCost::ICPCost(const SwitchedModelReferenceManager& referenceManager,
                 vector2_t weights,
                 const PinocchioInterface& pinocchioInterface,
                 const MpcRobotModelBase<ad_scalar_t>& mpcRobotModelAD,
                 const std::string& costName,
                 const ModelSettings& modelSettings)
    : StateInputCostGaussNewtonAd(),
      referenceManagerPtr_(&referenceManager),
      sqrtWeights_(weights.cwiseSqrt()),
      pinocchioInterfaceCppAd_(pinocchioInterface.toCppAd()),
      mpcRobotModelAdPtr_(mpcRobotModelAD.clone()) {
  initialize(mpcRobotModelAD.getStateDim(), mpcRobotModelAD.getInputDim(), /*parameterDim=*/2, costName, modelSettings.modelFolderCppAd,
             modelSettings.recompileLibrariesCppAd);
  LOG(INFO) << "Initialized ICPCost with weights: " << weights.transpose();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

ICPCost::ICPCost(const ICPCost& other)
    : StateInputCostGaussNewtonAd(other),
      referenceManagerPtr_(other.referenceManagerPtr_),
      sqrtWeights_(other.sqrtWeights_),
      // isActive_ is copied deliberately: the SQP solver clones the whole problem once per worker thread,
      // and a copy constructor that dropped this flag silently reverted a deactivated term to active.
      isActive_(other.isActive_),
      pinocchioInterfaceCppAd_(other.pinocchioInterfaceCppAd_),
      mpcRobotModelAdPtr_(other.mpcRobotModelAdPtr_->clone()) {}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

ad_vector_t ICPCost::costVectorFunction(ad_scalar_t /*time*/,
                                        const ad_vector_t& state,
                                        const ad_vector_t& /*input*/,
                                        const ad_vector_t& parameters) {
  const ad_vector_t sqrtWeightParams = parameters.head(2);  // EndEffectorKinematicsWeights vector element

  const PinocchioInterfaceCppAd::Model& model = pinocchioInterfaceCppAd_.getModel();
  PinocchioInterfaceCppAd::Data& data = pinocchioInterfaceCppAd_.getData();

  const ad_vector_t q = mpcRobotModelAdPtr_->getGeneralizedCoordinates(state);

  pinocchio::centerOfMass(model, data, q, /*computeSubtreeComs=*/false);
  ad_vector2_t com = data.com[0].head(2);

  pinocchio::updateFramePlacements(model, data);
  std::vector<VECTOR3_T<ad_scalar_t>> contactPositions = getContactPositions<ad_scalar_t>(pinocchioInterfaceCppAd_, *mpcRobotModelAdPtr_);
  ad_vector2_t desiredCOMPosition = (contactPositions[0] + contactPositions[1]).head(2) / ad_scalar_t(2.0);

  ad_vector_t com_vel(2);
  com_vel[0] = state[0];
  com_vel[1] = state[1];

  ad_vector_t capturePoint = com;
  // The capture point would be com + com_vel / omega, omega = sqrt(g / z_0); the velocity term is not included.
  ad_vector_t errors = desiredCOMPosition - capturePoint;

  return errors.cwiseProduct(sqrtWeightParams);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

vector_t ICPCost::getParameters(scalar_t /*time*/,
                                const TargetTrajectories& /*targetTrajectories*/,
                                const PreComputation& /*preComputation*/) const {
  // TODO(npalomo): update this reference for non-flat ground.
  vector_t parameters = sqrtWeights_;  // EndEffectorKinematicsWeights vector element

  return parameters;
}

}  // namespace ocs2::humanoid
