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

#include "humanoid_wb_mpc/cost/EndEffectorDynamicsFootCost.h"

#include <string>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "ocs2_robotic_tools/common/RotationTransforms.h"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/model.hpp"

#include "humanoid_wb_mpc/dynamics/DynamicsHelperFunctions.h"

namespace ocs2::humanoid {

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

EndEffectorDynamicsFootCost::EndEffectorDynamicsFootCost(const SwitchedModelReferenceManager& referenceManager,
                                                         const EndEffectorDynamicsWeights& weights,
                                                         const PinocchioInterface& pinocchioInterface,
                                                         const EndEffectorDynamics<scalar_t>& endEffectorDynamics,
                                                         const WBAccelMpcRobotModel<ad_scalar_t>& mpcRobotModel,
                                                         size_t contactIndex,
                                                         const std::string& modelName,
                                                         const ModelSettings& modelSettings)
    : StateInputCostGaussNewtonAd(),
      referenceManagerPtr_(&referenceManager),
      sqrtWeights_(weights.toVector().cwiseSqrt()),
      contactIndex_(contactIndex),
      frameID_(pinocchioInterface.getModel().getFrameId(modelSettings.contactNames[contactIndex])),
      pinocchioInterfaceCppAd_(pinocchioInterface.toCppAd()),
      endEffectorDynamicsPtr_(endEffectorDynamics.clone()),
      mpcRobotModelPtr_(mpcRobotModel.clone()) {
  initialize(mpcRobotModel.getStateDim(), mpcRobotModel.getInputDim(), /*parameterDim=*/37, modelName, modelSettings.modelFolderCppAd,
             modelSettings.recompileLibrariesCppAd);
  LOG(INFO) << "Frame ID: " << frameID_;
  LOG(INFO) << "Initialized EndEffectorDynamicsFootCost with weights: " << weights.toVector().transpose();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

// LINT.IfChange(foot_cost_names)
std::string EndEffectorDynamicsFootCost::termName(absl::string_view footName) {
  return absl::StrCat(footName, "_TaskSpaceTrackingCost");
}

std::string EndEffectorDynamicsFootCost::libraryModelName(absl::string_view footName) {
  return absl::StrCat(footName, "_TaskSpaceTrackingCost");
}
// LINT.ThenChange(//humanoid_nmpc/humanoid_wb_mpc/test/testJointTorqueCostLibraryName.cpp:term_and_library_names)

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

EndEffectorDynamicsFootCost::EndEffectorDynamicsFootCost(const EndEffectorDynamicsFootCost& other)
    : StateInputCostGaussNewtonAd(other),
      referenceManagerPtr_(other.referenceManagerPtr_),
      sqrtWeights_(other.sqrtWeights_),
      contactIndex_(other.contactIndex_),
      frameID_(other.frameID_),
      pinocchioInterfaceCppAd_(other.pinocchioInterfaceCppAd_),
      endEffectorDynamicsPtr_(other.endEffectorDynamicsPtr_->clone()),
      mpcRobotModelPtr_(other.mpcRobotModelPtr_->clone()) {}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

ad_vector_t EndEffectorDynamicsFootCost::costVectorFunction(ad_scalar_t /*time*/,
                                                            const ad_vector_t& state,
                                                            const ad_vector_t& input,
                                                            const ad_vector_t& parameters) {
  const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;

  const PinocchioInterfaceCppAd::Model& model = pinocchioInterfaceCppAd_.getModel();
  PinocchioInterfaceCppAd::Data& data = pinocchioInterfaceCppAd_.getData();

  const ad_vector_t q = mpcRobotModelPtr_->getGeneralizedCoordinates(state);
  const ad_vector_t v = mpcRobotModelPtr_->getGeneralizedVelocities(state, input);
  const ad_vector_t a = computeGeneralizedAccelerations<ad_scalar_t>(state, input, pinocchioInterfaceCppAd_, *mpcRobotModelPtr_);

  pinocchio::forwardKinematics(model, data, q, v, a);
  const pinocchio::SE3Tpl<ad_scalar_t> frameData = pinocchio::updateFramePlacement(model, data, frameID_);

  ad_vector3_t linearVelocity = pinocchio::getFrameVelocity(model, data, frameID_, rf).linear();
  ad_quaternion_t orientation = matrixToQuaternion(frameData.rotation());
  ad_vector3_t angularVelocity = pinocchio::getFrameVelocity(model, data, frameID_, rf).angular();
  const pinocchio::MotionTpl<ad_scalar_t> accel = pinocchio::getFrameClassicalAcceleration(model, data, frameID_, rf);
  ad_vector3_t linearAccel = accel.linear();
  ad_vector3_t angularAccel = accel.angular();

  const PlanarEndEffectorDynamicsReference<ad_scalar_t> reference(parameters.head(18));
  const ad_vector_t sqrtWeightParams = parameters.segment(18, 18);  // EndEffectorDynamicsWeights vector element
  const ad_scalar_t& impactProximityScaler = parameters[36];

  // LINT.IfChange(foot_cost_errors)
  ad_vector_t errors(18);
  errors << ad_vector3_t::Zero(), quaternionDistanceToPlane<ad_scalar_t>(orientation, reference.getPlaneNormal()),
      (linearVelocity - reference.getLinearVelocity()), (angularVelocity - reference.getAngularVelocity()),
      (linearAccel - reference.getLinearAcceleration()), (angularAccel - reference.getAngularAcceleration());
  // LINT.ThenChange(//humanoid_nmpc/humanoid_wb_mpc/src/config/costs/EndEffectorDynamicsWeightsFromConfig.cpp:inert_foot_cost_weights)

  return errors.cwiseProduct(sqrtWeightParams) * impactProximityScaler;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

vector_t EndEffectorDynamicsFootCost::getParameters(scalar_t time,
                                                    const TargetTrajectories& /*targetTrajectories*/,
                                                    const PreComputation& /*preComputation*/) const {
  // The references are flat ground at rest, the plane the foot's tilt is held to and the swing trajectory planner's
  // impact proximity; the solver's target is not read.
  const scalar_t impactProximityScaler = referenceManagerPtr_->getSwingTrajectoryPlanner()->getImpactProximityFactor(contactIndex_, time);

  // TODO(npalomo): Update this reference for non flat ground in the future.
  vector_t parameters(37);
  parameters.head(3) = vector3_t(0.0, 0.0, 0.0);  // Reference position
  // The ground's normal, tilted toe-up during a swing by swing_trajectory_config.swing_pitch_angle (exactly (0, 0, 1)
  // in contact and with the default pitch of 0), as the centroidal foot cost tracks it.
  // LINT.IfChange(foot_cost_ground_normal)
  parameters.segment(3, 3) = referenceManagerPtr_->getSwingFootPlaneNormal(contactIndex_, time);
  // LINT.ThenChange(//humanoid_nmpc/humanoid_wb_mpc/src/config/costs/EndEffectorDynamicsWeightsFromConfig.cpp:inert_foot_cost_weights)
  parameters.segment(6, 3) = vector3_t(0.0, 0.0, 0.0);   // Reference linear velocity
  parameters.segment(9, 3) = vector3_t(0.0, 0.0, 0.0);   // Reference angular velocity
  parameters.segment(12, 3) = vector3_t(0.0, 0.0, 0.0);  // Reference linear acceleration
  parameters.segment(15, 3) = vector3_t(0.0, 0.0, 0.0);  // Reference angular acceleration
  parameters.segment(18, 18) = sqrtWeights_;             // EndEffectorDynamicsWeights vector element

  parameters[36] = impactProximityScaler;

  return parameters;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
}  // namespace ocs2::humanoid
