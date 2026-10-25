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

#include "humanoid_centroidal_mpc/cost/CentroidalMpcEndEffectorFootCost.h"

#include <cmath>
#include <optional>
#include <string>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "ocs2_centroidal_model/ModelHelperFunctions.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "ocs2_pinocchio_interface/PinocchioStateInputMapping.h"
#include "ocs2_robotic_tools/common/RotationTransforms.h"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/model.hpp"

#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"

namespace ocs2::humanoid {

namespace {
// Reference (12) + sqrt weights (12) + impact proximity scaler (1) + foot yaw
// reference (1) + its flag (1).
constexpr size_t kNumParameters = 27;

// The residual, kEndEffectorTaskSpaceErrorDim errors: of the position, of the
// orientation (roll, pitch and yaw), of the linear velocity and of the angular
// velocity, in this order; the twist's errors follow the pose's.
constexpr int kLinearVelocityResidualStartIndex = kPoseTangentDim;

// The layout of the parameters (getParameters()): the reference
// (PlanarEndEffectorKinematicsPlanarReference: the position, the normal of the
// plane the orientation is tracked against, the linear and the angular
// velocity), the sqrt weights of the residual (EndEffectorKinematicsWeights),
// the impact proximity scaler, the foot yaw reference and its flag.
using PlanarReference = PlanarEndEffectorKinematicsPlanarReference<scalar_t>;
constexpr int kSqrtWeightsStartIndex = PlanarReference::kDim;
constexpr int kImpactProximityScalerIndex =
    kSqrtWeightsStartIndex + kEndEffectorTaskSpaceErrorDim;
constexpr int kYawReferenceIndex = kImpactProximityScalerIndex + 1;
constexpr int kHasYawReferenceIndex = kYawReferenceIndex + 1;
static_assert(static_cast<size_t>(kHasYawReferenceIndex) + 1 == kNumParameters);
}  // namespace

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

CentroidalMpcEndEffectorFootCost::CentroidalMpcEndEffectorFootCost(
    const SwitchedModelReferenceManager& referenceManager,
    const TaskSpaceFootCostSettings& settings,
    const PinocchioInterface& pinocchioInterface,
    const MpcRobotModelBase<ad_scalar_t>& mpcRobotModelAD,
    size_t contactIndex,
    const std::string& costName,
    const ModelSettings& modelSettings)
    : StateInputCostGaussNewtonAd(),
      referenceManagerPtr_(&referenceManager),
      sqrtWeights_(settings.weights.toVector().cwiseSqrt()),
      swingVelocityReferenceSqrtWeight_(
          std::sqrt(settings.swingVelocityReferenceWeight)),
      activeInStance_(settings.activeInStance),
      contactIndex_(contactIndex),
      frameID_(pinocchioInterface.getModel().getFrameId(
          modelSettings.contactNames[contactIndex])),
      pinocchioInterfaceCppAd_(pinocchioInterface.toCppAd()),
      mpcRobotModelAdPtr_(mpcRobotModelAD.clone()) {
  // The library name carries the residual's version, so that a library taped
  // from an earlier residual is never loaded from the cache in its place (the
  // robots ship recompileLibrariesCppAd: false): kLibraryVersionSuffix.
  initialize(mpcRobotModelAD.getStateDim(), mpcRobotModelAD.getInputDim(),
             kNumParameters, absl::StrCat(costName, kLibraryVersionSuffix),
             modelSettings.modelFolderCppAd,
             modelSettings.recompileLibrariesCppAd);
  LOG(INFO) << "Frame ID: " << frameID_;
  LOG(INFO) << "Initialized CentroidalMpcEndEffectorFootCost (active_phases: "
            << (activeInStance_ ? "swing_and_stance" : "swing")
            << ") with weights: " << settings.weights.toVector().transpose()
            << ", swing velocity reference weight: "
            << settings.swingVelocityReferenceWeight;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

CentroidalMpcEndEffectorFootCost::CentroidalMpcEndEffectorFootCost(
    const CentroidalMpcEndEffectorFootCost& other)
    : StateInputCostGaussNewtonAd(other),
      referenceManagerPtr_(other.referenceManagerPtr_),
      sqrtWeights_(other.sqrtWeights_),
      swingVelocityReferenceSqrtWeight_(
          other.swingVelocityReferenceSqrtWeight_),
      // isActive_ is copied deliberately: the SQP solver clones the whole
      // problem once per worker thread, and a copy constructor that dropped
      // this flag silently reverted a deactivated term to active.
      isActive_(other.isActive_),
      activeInStance_(other.activeInStance_),
      contactIndex_(other.contactIndex_),
      frameID_(other.frameID_),
      pinocchioInterfaceCppAd_(other.pinocchioInterfaceCppAd_),
      mpcRobotModelAdPtr_(other.mpcRobotModelAdPtr_->clone()) {}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

ad_vector_t CentroidalMpcEndEffectorFootCost::costVectorFunction(
    ad_scalar_t /*time*/,
    const ad_vector_t& state,
    const ad_vector_t& input,
    const ad_vector_t& parameters) {
  const pinocchio::ReferenceFrame rf =
      pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;

  const PlanarEndEffectorKinematicsPlanarReference<ad_scalar_t> reference(
      parameters.head(PlanarReference::kDim));
  // The EndEffectorKinematicsWeights vector elements.
  const ad_vector_t sqrtWeightParams =
      parameters.segment(kSqrtWeightsStartIndex, kEndEffectorTaskSpaceErrorDim);
  const ad_scalar_t& impactProximityScaler =
      parameters[kImpactProximityScalerIndex];
  const ad_scalar_t& yawReference = parameters[kYawReferenceIndex];
  const ad_scalar_t& hasYawReference = parameters[kHasYawReferenceIndex];

  const PinocchioInterfaceCppAd::Model& model =
      pinocchioInterfaceCppAd_.getModel();
  PinocchioInterfaceCppAd::Data& data = pinocchioInterfaceCppAd_.getData();

  const ad_vector_t q = mpcRobotModelAdPtr_->getGeneralizedCoordinates(state);
  const ad_vector_t v =
      mpcRobotModelAdPtr_->getGeneralizedVelocities(state, input);
  pinocchio::forwardKinematics(model, data, q, v);
  const pinocchio::SE3Tpl<ad_scalar_t>& frameData =
      pinocchio::updateFramePlacement(model, data, frameID_);

  ad_vector_t position = frameData.translation();
  ad_vector_t linearVelocity =
      pinocchio::getFrameVelocity(model, data, frameID_, rf).linear();
  ad_matrix3_t orientation = frameData.rotation();
  ad_vector_t angularVelocity =
      pinocchio::getFrameVelocity(model, data, frameID_, rf).angular();

  // Orientation error, in radians as the end-effector costs'
  // (computeTaskSpaceErrors()): the tilt from the reference plane as a rotation
  // vector in world axes (tiltAngleToPlane(), roll and pitch on flat ground).
  // With a planned foot yaw (contact planner heading model) the third component
  // tracks it, wrapped so that a crossing of +-pi is not a discontinuity; the
  // config weight orientation_z selects whether that error costs anything.
  // footYawError() is finite, value and derivatives, at every yaw, a yaw of
  // exactly zero included; see there for why the atan2 it replaces was not.
  ad_vector_t orientationError =
      tiltAngleToPlane<ad_scalar_t>(orientation, reference.getPlaneNormal());
  const ad_scalar_t yawError =
      footYawError<ad_scalar_t>(orientation, yawReference, hasYawReference);
  orientationError(kZAxisIndex) =
      hasYawReference * yawError +
      (ad_scalar_t(1.0) - hasYawReference) * orientationError(kZAxisIndex);

  ad_vector_t errors(kEndEffectorTaskSpaceErrorDim);
  errors << (position - reference.getPosition()), orientationError,
      (linearVelocity - reference.getLinearVelocity()) * impactProximityScaler,
      (angularVelocity - reference.getAngularVelocity());

  return errors.cwiseProduct(sqrtWeightParams);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

vector_t CentroidalMpcEndEffectorFootCost::getParameters(
    scalar_t time,
    const TargetTrajectories& /*targetTrajectories*/,
    const PreComputation& /*preComputation*/) const {
  // Every reference comes from the reference manager and the swing trajectory
  // planner; the solver's target is not read.
  const scalar_t impactProximityScaler =
      referenceManagerPtr_->getSwingTrajectoryPlanner()
          ->getImpactProximityFactor(contactIndex_, time);

  // The height and the vertical velocity are the swing trajectory planner's,
  // whose trajectories the reference manager builds on its terrain height: the
  // swing profile in swing, the ground itself in stance (under active_phases
  // "swing_and_stance"). A landing target below replaces only the horizontal
  // references.
  const SwingTrajectoryPlanner& swingPlanner =
      *referenceManagerPtr_->getSwingTrajectoryPlanner();
  vector_t parameters(kNumParameters);
  parameters.head(kPositionDim) =
      vector3_t(0.0, 0.0,
                swingPlanner.getZpositionConstraint(
                    contactIndex_, time));  // Reference position
  // Plane the foot orientation is tracked against. Flat ground unless a toe-up
  // swing pitch is configured, in which case the normal is tilted back along
  // the heading for the swing so that the toe -- half a foot length ahead of
  // the tracked sole center -- clears the ground by more than the height
  // reference alone provides.
  parameters.segment(PlanarReference::kPlaneNormalStart, kPositionDim) =
      referenceManagerPtr_->getSwingFootPlaneNormal(contactIndex_, time);
  parameters.segment(PlanarReference::kLinearVelocityStart, kPositionDim) =
      vector3_t(0.0, 0.0,
                swingPlanner.getZvelocityConstraint(
                    contactIndex_, time));  // Reference linear velocity
  parameters.segment(PlanarReference::kAngularVelocityStart,
                     kRotationTangentDim) =
      vector3_t(0.0, 0.0, 0.0);  // Reference angular velocity
  parameters.segment(kSqrtWeightsStartIndex, kEndEffectorTaskSpaceErrorDim) =
      sqrtWeights_;  // EndEffectorKinematicsWeights vector element

  parameters[kImpactProximityScalerIndex] = impactProximityScaler;
  parameters[kYawReferenceIndex] = 0.0;  // planned foot yaw reference
  parameters[kHasYawReferenceIndex] =
      0.0;  // 1 when the yaw reference is to be tracked

  // Without a landing target there is no meaningful xy position reference (the
  // foot is free to land where the whole-body optimization puts it), so the xy
  // position error is switched off. A landing target turns it on and replaces
  // the position and linear velocity references with the interpolated swing
  // trajectory towards the target.
  const std::optional<SwingFootReference> swingReference =
      referenceManagerPtr_->getSwingFootReference(contactIndex_, time);
  if (swingReference.has_value()) {
    parameters.head(kPositionDim) = swingReference->position;
    parameters.segment(PlanarReference::kLinearVelocityStart, kPositionDim) =
        swingReference->linearVelocity;
    if (swingReference->yaw.has_value()) {
      parameters[kYawReferenceIndex] = *swingReference->yaw;
      parameters[kHasYawReferenceIndex] = 1.0;
    }
  } else {
    parameters[kSqrtWeightsStartIndex + kXAxisIndex] =
        0.0;  // sqrt weight of the x position error
    parameters[kSqrtWeightsStartIndex + kYAxisIndex] =
        0.0;  // sqrt weight of the y position error

    // A swing without a landing target tracks the commanded base velocity, so
    // that the foot does not drag behind the moving body, weighed by
    // swing_velocity_reference_weight in place of lin_velocity_x / _y (which
    // weigh the swing trajectory's velocity when there is a target, and the
    // foot's in stance). The impact proximity scaler ramps the error to zero
    // near touch-down.
    const std::optional<vector2_t> velocityReference =
        referenceManagerPtr_->getSwingFootVelocityReference(contactIndex_,
                                                            time);
    if (velocityReference.has_value()) {
      parameters.segment(PlanarReference::kLinearVelocityStart,
                         kHorizontalPositionDim) = *velocityReference;
      parameters[kSqrtWeightsStartIndex + kLinearVelocityResidualStartIndex +
                 kXAxisIndex] = swingVelocityReferenceSqrtWeight_;
      parameters[kSqrtWeightsStartIndex + kLinearVelocityResidualStartIndex +
                 kYAxisIndex] = swingVelocityReferenceSqrtWeight_;
    }
  }
  return parameters;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
}  // namespace ocs2::humanoid
