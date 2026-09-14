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

#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include <ocs2_pinocchio_interface/PinocchioInterface.h>

#include "humanoid_centroidal_mpc/cost/CentroidalMpcEndEffectorFootCost.h"

#include <ocs2_centroidal_model/ModelHelperFunctions.h>
#include <ocs2_robotic_tools/common/RotationTransforms.h>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

#include <ocs2_pinocchio_interface/PinocchioStateInputMapping.h>

namespace ocs2::humanoid {

namespace {
// Reference (12) + sqrt weights (12) + impact proximity scaler (1) + foot yaw reference (1) + its flag (1).
constexpr size_t kNumParameters = 27;
}  // namespace

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

CentroidalMpcEndEffectorFootCost::CentroidalMpcEndEffectorFootCost(const SwitchedModelReferenceManager& referenceManager,
                                                                   EndEffectorKinematicsWeights weights,
                                                                   const PinocchioInterface& pinocchioInterface,
                                                                   const MpcRobotModelBase<ad_scalar_t>& mpcRobotModelAD,
                                                                   size_t contactIndex,
                                                                   std::string costName,
                                                                   const ModelSettings& modelSettings,
                                                                   bool activeInStance)
    : StateInputCostGaussNewtonAd(),
      referenceManagerPtr_(&referenceManager),
      sqrtWeights_(weights.toVector().cwiseSqrt()),
      activeInStance_(activeInStance),
      frameID_(pinocchioInterface.getModel().getFrameId(modelSettings.contactNames[contactIndex])),
      pinocchioInterfaceCppAd_(pinocchioInterface.toCppAd()),
      mpcRobotModelAdPtr_(mpcRobotModelAD.clone()),
      contactIndex_(contactIndex) {
  initialize(mpcRobotModelAD.getStateDim(), mpcRobotModelAD.getInputDim(), kNumParameters, costName + "_yawRef",
             modelSettings.modelFolderCppAd, modelSettings.recompileLibrariesCppAd);
  std::cout << "Frame ID: " << frameID_ << std::endl;
  std::cout << "Initialized CentroidalMpcEndEffectorFootCost (activeInStance=" << (activeInStance_ ? "true" : "false")
            << ") with weights: " << weights.toVector().transpose() << std::endl;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

CentroidalMpcEndEffectorFootCost::CentroidalMpcEndEffectorFootCost(const CentroidalMpcEndEffectorFootCost& other)
    : StateInputCostGaussNewtonAd(other),
      referenceManagerPtr_(other.referenceManagerPtr_),
      sqrtWeights_(other.sqrtWeights_),
      activeInStance_(other.activeInStance_),
      frameID_(other.frameID_),
      contactIndex_(other.contactIndex_),
      pinocchioInterfaceCppAd_(other.pinocchioInterfaceCppAd_),
      mpcRobotModelAdPtr_(other.mpcRobotModelAdPtr_->clone()) {}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

ad_vector_t CentroidalMpcEndEffectorFootCost::costVectorFunction(ad_scalar_t time,
                                                                 const ad_vector_t& state,
                                                                 const ad_vector_t& input,
                                                                 const ad_vector_t& parameters) {
  const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;

  const PlanarEndEffectorKinematicsPlanarReference<ad_scalar_t> reference(parameters.head(12));
  const ad_vector_t sqrtWeightParams = parameters.segment(12, 12);  // EndEffectorKinematicsWeights vector element
  const ad_scalar_t impactProximityScaler = parameters[24];
  const ad_scalar_t yawReference = parameters[25];
  const ad_scalar_t hasYawReference = parameters[26];

  const auto& model = pinocchioInterfaceCppAd_.getModel();
  auto& data = pinocchioInterfaceCppAd_.getData();

  const ad_vector_t q = mpcRobotModelAdPtr_->getGeneralizedCoordinates(state);
  const ad_vector_t v = mpcRobotModelAdPtr_->getGeneralizedVelocities(state, input);
  pinocchio::forwardKinematics(model, data, q, v);
  auto frameData = pinocchio::updateFramePlacement(model, data, frameID_);

  // auto oMf = data.oMf;
  ad_vector_t position = frameData.translation();
  ad_vector_t linearVelocity = pinocchio::getFrameVelocity(model, data, frameID_, rf).linear();
  ad_matrix3_t orientation = frameData.rotation();
  ad_vector_t angularVelocity = pinocchio::getFrameVelocity(model, data, frameID_, rf).angular();

  // Orientation error: the distance to the ground plane (roll, pitch). With a planned foot yaw (contact planner heading
  // model) the third component tracks it, wrapped so that a crossing of +-pi is not a discontinuity; the config weight
  // orientation_z selects whether that error costs anything.
  ad_vector_t orientationError = rotationMatrixDistanceToPlane<ad_scalar_t>(orientation, reference.getPlaneNormal());
  const ad_scalar_t footYaw = CppAD::atan2(orientation(1, 0), orientation(0, 0));
  const ad_scalar_t yawError = CppAD::atan2(CppAD::sin(footYaw - yawReference), CppAD::cos(footYaw - yawReference));
  orientationError(2) = hasYawReference * yawError + (ad_scalar_t(1.0) - hasYawReference) * orientationError(2);

  ad_vector_t errors(12);
  errors << (position - reference.getPosition()), orientationError,
      (linearVelocity - reference.getLinearVelocity()) * impactProximityScaler, (angularVelocity - reference.getAngularVelocity());

  return errors.cwiseProduct(sqrtWeightParams);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

vector_t CentroidalMpcEndEffectorFootCost::getParameters(scalar_t time,
                                                         const TargetTrajectories& targetTrajectories,
                                                         const PreComputation& preComputation) const {
  // Interpolate reference
  const vector_t xRef = targetTrajectories.getDesiredState(time);
  const vector_t uRef = targetTrajectories.getDesiredInput(time);

  const scalar_t impactProximityScaler = referenceManagerPtr_->getSwingTrajectoryPlanner()->getImpactProximityFactor(contactIndex_, time);

  // TODO Update this reference for non flat ground in the future
  vector_t parameters(kNumParameters);
  parameters.head(3) = vector3_t(0.0, 0.0, 0.0);  // Reference position
  // Plane the foot orientation is tracked against. Flat ground unless a toe-up swing pitch is configured, in which case
  // the normal is tilted back along the heading for the swing so that the toe -- half a foot length ahead of the
  // tracked sole centre -- clears the ground by more than the height reference alone provides.
  parameters.segment(3, 3) = referenceManagerPtr_->getSwingFootPlaneNormal(contactIndex_, time);
  parameters.segment(6, 3) = vector3_t(0.0, 0.0, 0.0);  // Reference linear velocity
  parameters.segment(9, 3) = vector3_t(0.0, 0.0, 0.0);  // Reference angular velocity
  parameters.segment(12, 12) = sqrtWeights_;            // EndEffectorKinematicsWeights vector element

  parameters[24] = impactProximityScaler;
  parameters[25] = 0.0;  // planned foot yaw reference
  parameters[26] = 0.0;  // 1 when the yaw reference is to be tracked

  // Without a contact planner there is no meaningful xy position reference (the foot is free to land where the
  // whole-body optimization puts it), so the xy position error is switched off. A planned foothold turns it on and
  // replaces the position and linear velocity references with the interpolated swing trajectory towards the target.
  const std::optional<SwingFootReference> swingReference = referenceManagerPtr_->getSwingFootReference(contactIndex_, time);
  if (swingReference.has_value()) {
    parameters.head(3) = swingReference->position;
    parameters.segment(6, 3) = swingReference->linearVelocity;
    if (swingReference->yaw.has_value()) {
      parameters[25] = *swingReference->yaw;
      parameters[26] = 1.0;
    }
  } else {
    parameters[12] = 0.0;  // sqrt weight of the x position error
    parameters[13] = 0.0;  // sqrt weight of the y position error

    // Without a contact planner, provide a heuristic velocity reference to prevent the swing foot from
    // dragging backward relative to the moving body.  The config lin_velocity_x weight is intentionally 0
    // (to avoid penalising stance), so we temporarily override the sqrt-weight here with a small value.
    // The impactProximityScaler already ramps this cost to zero near touchdown, which is the profile we want.
    const std::optional<vector2_t> velocityReference = referenceManagerPtr_->getSwingFootVelocityReference(contactIndex_, time);
    if (velocityReference.has_value()) {
      parameters.segment(6, 2) = *velocityReference;
      parameters[18] = 1.5;  // override sqrt(lin_velocity_x) for swing only
      parameters[19] = 1.5;  // override sqrt(lin_velocity_y) for swing only
    }
  }
  return parameters;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
}  // namespace ocs2::humanoid
