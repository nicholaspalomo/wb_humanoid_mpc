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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include "humanoid_centroidal_mpc/constraint/TerrainNonPenetrationConstraint.h"

#include <string>

#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/model.hpp"

namespace ocs2::humanoid {
namespace {

/** Where the radius and the margin follow the center offset among the
 * parameters. */
constexpr int kRadiusParameter = kPositionDim;
constexpr int kMarginParameter = kPositionDim + 1;

}  // namespace

TerrainNonPenetrationConstraint::TerrainNonPenetrationConstraint(
    const SwitchedModelReferenceManager& referenceManager,
    const PinocchioInterface& pinocchioInterface,
    const MpcRobotModelBase<ad_scalar_t>& mpcRobotModel,
    const multi_contact::TerrainModel& terrain,
    size_t contactIndex,
    const Sphere& sphere,
    scalar_t margin,
    const std::string& libraryName,
    const ModelSettings& modelSettings)
    : StateConstraintCppAd(ConstraintOrder::Linear),
      referenceManagerPtr_(&referenceManager),
      pinocchioInterfaceCppAd_(pinocchioInterface.toCppAd()),
      mpcRobotModelPtr_(&mpcRobotModel),
      terrainPtr_(&terrain),
      contactIndex_(contactIndex),
      frameName_(modelSettings.contactNames[contactIndex]),
      sphere_(sphere),
      margin_(margin) {
  initialize(mpcRobotModelPtr_->getStateDim(), kNumParameters, libraryName,
             modelSettings.modelFolderCppAd,
             modelSettings.recompileLibrariesCppAd, modelSettings.verboseCppAd);
}

TerrainNonPenetrationConstraint::TerrainNonPenetrationConstraint(
    const TerrainNonPenetrationConstraint& other) = default;

bool TerrainNonPenetrationConstraint::isActive(scalar_t time) const {
  return !referenceManagerPtr_->isInContact(time, contactIndex_);
}

vector_t TerrainNonPenetrationConstraint::getParameters(
    scalar_t /*time*/, const PreComputation& /*preComputation*/) const {
  vector_t parameters(kNumParameters);
  parameters << sphere_.centerOffset, sphere_.radius, margin_;
  return parameters;
}

ad_vector_t TerrainNonPenetrationConstraint::constraintFunction(
    ad_scalar_t /*time*/,
    const ad_vector_t& state,
    const ad_vector_t& parameters) const {
  const PinocchioInterfaceCppAd::Model& model =
      pinocchioInterfaceCppAd_.getModel();
  PinocchioInterfaceCppAd::Data data = pinocchioInterfaceCppAd_.getData();
  const ad_vector_t q = mpcRobotModelPtr_->getGeneralizedCoordinates(state);
  pinocchio::forwardKinematics(model, data, q);
  const pinocchio::SE3Tpl<ad_scalar_t>& frame = pinocchio::updateFramePlacement(
      model, data, model.getFrameId(frameName_));
  const ad_vector3_t center =
      frame.translation() + frame.rotation() * parameters.head<kPositionDim>();
  ad_vector_t value(1);
  value(0) = terrainPtr_->sphereDistance<ad_scalar_t>(
                 center, parameters(kRadiusParameter)) -
             parameters(kMarginParameter);
  return value;
}

}  // namespace ocs2::humanoid
