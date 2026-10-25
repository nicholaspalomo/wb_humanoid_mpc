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

#include "pinocchio/fwd.hpp"

#include "humanoid_common_mpc/constraint/FootCollisionCbfConstraint.h"

#include <array>
#include <cstddef>
#include <string>

#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/model.hpp"

namespace ocs2::humanoid {

namespace {

// The parameters of the taped function, in the order of getParameters().
constexpr Eigen::Index kFootRadiusParameter = 0;
constexpr Eigen::Index kKneeRadiusParameter = 1;
constexpr Eigen::Index kGammaParameter = 2;

}  // namespace

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

FootCollisionCbfConstraint::FootCollisionCbfConstraint(
    const SwitchedModelReferenceManager& referenceManager,
    const PinocchioInterface& pinocchioInterface,
    const MpcRobotModelBase<ad_scalar_t>& mpcRobotModel,
    const Config& config,
    const ModelSettings& modelSettings)
    : StateInputConstraintCppAd(ConstraintOrder::Linear),
      referenceManagerPtr_(&referenceManager),
      pinocchioInterfaceCppAd_(pinocchioInterface.toCppAd()),
      mpcRobotModelPtr_(&mpcRobotModel),
      config_(config) {
  initialize(mpcRobotModelPtr_->getStateDim(), mpcRobotModelPtr_->getInputDim(),
             kNumParameters, kLibraryName, modelSettings.modelFolderCppAd,
             modelSettings.recompileLibrariesCppAd, modelSettings.verboseCppAd);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

FootCollisionCbfConstraint::FootCollisionCbfConstraint(
    const FootCollisionCbfConstraint& other) = default;

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

bool FootCollisionCbfConstraint::isActive(scalar_t time) const {
  // Off while both feet are in contact, as FootCollisionConstraint: it would
  // fight the stance feet's constraints.
  const contact_flag_t contactFlags =
      referenceManagerPtr_->getContactFlags(time);
  return !(contactFlags[0] && contactFlags[1]);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

vector_t FootCollisionCbfConstraint::getParameters(
    scalar_t /*time*/, const PreComputation& /*preComputation*/) const {
  vector_t parameters(kNumParameters);
  parameters[kFootRadiusParameter] = config_.spheres.footCollisionSphereRadius;
  parameters[kKneeRadiusParameter] = config_.spheres.kneeCollisionSphereRadius;
  parameters[kGammaParameter] = config_.gamma;
  return parameters;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

ad_vector_t FootCollisionCbfConstraint::constraintFunction(
    ad_scalar_t /*time*/,
    const ad_vector_t& state,
    const ad_vector_t& input,
    const ad_vector_t& parameters) const {
  const PinocchioInterfaceCppAd::Model& model =
      pinocchioInterfaceCppAd_.getModel();
  PinocchioInterfaceCppAd::Data data = pinocchioInterfaceCppAd_.getData();

  // The base at rest (the class comment says why that leaves dh/dt exact), the
  // joints at the model's joint velocities.
  const ad_vector_t q = mpcRobotModelPtr_->getGeneralizedCoordinates(state);
  const ad_vector_t jointVelocities =
      mpcRobotModelPtr_->getJointVelocities(state, input);
  ad_vector_t v = ad_vector_t::Zero(model.nv);
  v.tail(jointVelocities.size()) = jointVelocities;
  pinocchio::forwardKinematics(model, data, q, v);

  const std::array<std::string, kNumFootCollisionPoints> frames =
      footCollisionPointFrames(config_.spheres);
  std::array<ad_vector3_t, kNumFootCollisionPoints> positions;
  std::array<ad_vector3_t, kNumFootCollisionPoints> velocities;
  for (size_t point = 0; point < kNumFootCollisionPoints; ++point) {
    const pinocchio::FrameIndex frameId = model.getFrameId(frames[point]);
    positions[point] =
        pinocchio::updateFramePlacement(model, data, frameId).translation();
    velocities[point] = pinocchio::getFrameVelocity(
                            model, data, frameId,
                            pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED)
                            .linear();
  }

  // Two spheres touch at a distance of 2 * radius.
  const ad_scalar_t minDistFoot = 2.0 * parameters[kFootRadiusParameter];
  const ad_scalar_t minDistKnee = 2.0 * parameters[kKneeRadiusParameter];
  const ad_scalar_t minDistFootSquared = minDistFoot * minDistFoot;
  const ad_scalar_t minDistKneeSquared = minDistKnee * minDistKnee;
  const ad_scalar_t gamma = parameters[kGammaParameter];

  ad_vector_t constraintValues(
      static_cast<Eigen::Index>(kNumFootCollisionPairs));
  for (size_t row = 0; row < kNumFootCollisionPairs; ++row) {
    const FootCollisionPair& pair = kFootCollisionPairs[row];
    const size_t first = static_cast<size_t>(pair.first);
    const size_t second = static_cast<size_t>(pair.second);
    const ad_vector3_t relativePosition = positions[first] - positions[second];
    const ad_vector3_t relativeVelocity =
        velocities[first] - velocities[second];
    const ad_scalar_t barrier =
        relativePosition.squaredNorm() -
        (pair.radius == FootCollisionRadius::kKnee ? minDistKneeSquared
                                                   : minDistFootSquared);
    const ad_scalar_t barrierRate =
        2.0 * relativePosition.dot(relativeVelocity);
    constraintValues[static_cast<Eigen::Index>(row)] =
        barrierRate + gamma * barrier;
  }
  return constraintValues;
}

}  // namespace ocs2::humanoid
