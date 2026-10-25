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

#include "pinocchio/fwd.hpp"

#include "humanoid_common_mpc/constraint/FootCollisionConstraint.h"

#include <array>
#include <cstddef>
#include <string>

#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/model.hpp"

namespace ocs2::humanoid {

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

FootCollisionConstraint::FootCollisionConstraint(
    const SwitchedModelReferenceManager& referenceManager,
    const PinocchioInterface& pinocchioInterface,
    const MpcRobotModelBase<ad_scalar_t>& mpcRobotModel,
    const Config& config,
    const std::string& costName,
    const ModelSettings& modelSettings)
    : StateConstraintCppAd(ConstraintOrder::Linear),
      referenceManagerPtr_(&referenceManager),
      pinocchioInterfaceCppAd_(pinocchioInterface.toCppAd()),
      mpcRobotModelPtr_(&mpcRobotModel),
      cfg_(config) {
  initialize(mpcRobotModelPtr_->getStateDim(), kNumParameters, costName,
             modelSettings.modelFolderCppAd,
             modelSettings.recompileLibrariesCppAd, modelSettings.verboseCppAd);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

// isActive_ is copied deliberately: the SQP solver clones the whole problem
// once per worker thread, and a copy constructor that dropped this flag
// silently reverted a deactivated term to active.
FootCollisionConstraint::FootCollisionConstraint(
    const FootCollisionConstraint& other) = default;

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
bool FootCollisionConstraint::isActive(scalar_t time) const {
  if (!isActive_) return false;

  // Inactivate the constraint if both feet are in contact. Prevents it from
  // fighting against the stance foot constraints.
  contact_flag_t contactFlags = referenceManagerPtr_->getContactFlags(time);
  return !(contactFlags[0] && contactFlags[1]);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

ad_vector_t FootCollisionConstraint::constraintFunction(
    ad_scalar_t /*time*/,
    const ad_vector_t& state,
    const ad_vector_t& parameters) const {
  const PinocchioInterfaceCppAd::Model& model =
      pinocchioInterfaceCppAd_.getModel();
  PinocchioInterfaceCppAd::Data data = pinocchioInterfaceCppAd_.getData();

  const ad_vector_t q = mpcRobotModelPtr_->getGeneralizedCoordinates(state);
  pinocchio::forwardKinematics(model, data, q);

  const std::array<std::string, kNumFootCollisionPoints> frames =
      footCollisionPointFrames(cfg_);
  std::array<ad_vector3_t, kNumFootCollisionPoints> positions;
  for (size_t point = 0; point < kNumFootCollisionPoints; ++point) {
    positions[point] = pinocchio::updateFramePlacement(
                           model, data, model.getFrameId(frames[point]))
                           .translation();
  }

  // The minimum distance between two collision spheres, 2 * radius: each row is
  // a distance minus it, not a squared distance. parameters[0] is the radius of
  // the foot spheres, parameters[1] that of the knee spheres.
  const ad_scalar_t minDistFoot = 2.0 * parameters[0];
  const ad_scalar_t minDistKnee = 2.0 * parameters[1];

  ad_vector_t constraintValues(numConstraints_);
  for (size_t row = 0; row < kNumFootCollisionPairs; ++row) {
    const FootCollisionPair& pair = kFootCollisionPairs[row];
    const ad_vector3_t difference = positions[static_cast<size_t>(pair.first)] -
                                    positions[static_cast<size_t>(pair.second)];
    constraintValues[static_cast<Eigen::Index>(row)] =
        difference.norm() -
        (pair.radius == FootCollisionRadius::kKnee ? minDistKnee : minDistFoot);
  }
  return constraintValues;
}

}  // namespace ocs2::humanoid
