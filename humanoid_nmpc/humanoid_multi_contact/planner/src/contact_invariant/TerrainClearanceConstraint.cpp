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

#include "humanoid_multi_contact/planner/contact_invariant/TerrainClearanceConstraint.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "ocs2_centroidal_model/CentroidalModelPinocchioMapping.h"
#include "ocs2_core/Dimensions.h"
#include "pinocchio/algorithm/kinematics.hpp"

namespace ocs2::humanoid::multi_contact {
namespace {

using AdVector3 = Eigen::Matrix<ad_scalar_t, kPositionDim, 1>;

}  // namespace

absl::StatusOr<std::vector<ClearanceSphere>> clearanceSpheresOf(
    const CollisionModel& collision,
    const RobotStanceModel& stanceModel,
    const PinocchioInterface& pinocchio) {
  const pinocchio::Model& mpcModel = pinocchio.getModel();
  std::vector<ClearanceSphere> spheres;
  for (const CollisionSphere& sphere : collision.spheres()) {
    const std::string& name = stanceModel.model().names[sphere.joint];
    if (!mpcModel.existJointName(name)) {
      return absl::NotFoundError(absl::StrCat(
          "terrain clearance: the collision sphere of '",
          collision.linkNames()[sphere.link], "' hangs from the joint '", name,
          "', which the MPC's model does not have"));
    }
    spheres.push_back(ClearanceSphere{.joint = mpcModel.getJointId(name),
                                      .center = sphere.center,
                                      .radius = sphere.radius});
  }
  return spheres;
}

TerrainClearanceConstraint::TerrainClearanceConstraint(
    const PinocchioInterface& pinocchio,
    const CentroidalModelInfo& info,
    std::vector<ClearanceSphere> spheres,
    const TerrainModel& terrain,
    scalar_t margin,
    const std::string& libraryName,
    const std::string& folder,
    bool recompile,
    bool verbose)
    : StateConstraintCppAd(ConstraintOrder::Linear),
      pinocchioCppAd_(pinocchio.toCppAd()),
      infoCppAd_(info.toCppAd()),
      spheres_(std::move(spheres)),
      terrain_(terrain),
      margin_(margin) {
  initialize(info.stateDim, /*parameterDim=*/0, libraryName, folder, recompile,
             verbose);
}

ad_vector_t TerrainClearanceConstraint::constraintFunction(
    ad_scalar_t /*time*/,
    const ad_vector_t& state,
    const ad_vector_t& /*parameters*/) const {
  PinocchioInterfaceCppAd interface = pinocchioCppAd_;
  const CentroidalModelPinocchioMappingCppAd mapping(infoCppAd_);
  const ad_vector_t q = mapping.getPinocchioJointPosition(state);
  pinocchio::forwardKinematics(interface.getModel(), interface.getData(), q);
  ad_vector_t value(spheres_.size());
  for (size_t k = 0; k < spheres_.size(); ++k) {
    const ClearanceSphere& sphere = spheres_[k];
    const pinocchio::SE3Tpl<ad_scalar_t>& placement =
        interface.getData().oMi[sphere.joint];
    const AdVector3 center =
        placement.translation() +
        placement.rotation() * sphere.center.cast<ad_scalar_t>();
    value(k) = terrain_.sphereDistance<ad_scalar_t>(
                   center, ad_scalar_t(sphere.radius)) -
               ad_scalar_t(margin_);
  }
  return value;
}

}  // namespace ocs2::humanoid::multi_contact
