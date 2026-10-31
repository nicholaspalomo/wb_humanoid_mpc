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

#include "humanoid_multi_contact/planner/contact_invariant/RobotContactFeatures.h"

#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "ocs2_centroidal_model/AccessHelperFunctions.h"
#include "ocs2_centroidal_model/CentroidalModelPinocchioMapping.h"
#include "ocs2_centroidal_model/ModelHelperFunctions.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/PhysicalConstants.h"
#include "ocs2_core/automatic_differentiation/Types.h"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/kinematics.hpp"

#include "humanoid_multi_contact/planner/contact_invariant/ContactFeatureRows.h"

namespace ocs2::humanoid::multi_contact {
namespace {

using AdVector3 = Eigen::Matrix<ad_scalar_t, kPositionDim, 1>;
using AdMatrix3 = Eigen::Matrix<ad_scalar_t, kPositionDim, kPositionDim>;

/** A candidate as the tape reads it. Passive data. */
struct TapedCandidate {
  pinocchio::FrameIndex frame = 0;
  pinocchio::JointIndex joint = 0;
  size_t mpcContactIndex = 0;
  bool surface = false;
  Eigen::Matrix3d padRotation = Eigen::Matrix3d::Identity();
  double geomRadius = 0.0;
};

/** The constants of the tape, in the tape's scalar. */
struct TapeConstants {
  double friction = 0.0;
  double coneSmoothing = 0.0;
  double cosineOfMaxTilt = 1.0;
  // [N] The robot's weight, the unit of the cones.
  double weight = 1.0;
};

/** Candidate `entry`'s rows of the features at the kinematics of `data`. */
void candidateFeatures(const PinocchioInterfaceCppAd::Model& model,
                       const PinocchioInterfaceCppAd::Data& data,
                       const CentroidalModelInfoCppAd& info,
                       const TerrainModel& terrain,
                       const TapedCandidate& entry,
                       const TapeConstants& constants,
                       const ad_vector_t& input,
                       int candidate,
                       ad_vector_t& features) {
  const AdVector3 pad = data.oMf[entry.frame].translation();
  const AdMatrix3 rotation =
      data.oMi[entry.joint].rotation() * entry.padRotation.cast<ad_scalar_t>();
  const AdVector3 padAxis = rotation.col(kZAxisIndex);
  const AdVector3 velocity =
      pinocchio::getFrameVelocity(model, data, entry.frame,
                                  pinocchio::LOCAL_WORLD_ALIGNED)
          .linear();
  const AdVector3 normal = terrain.normal<ad_scalar_t>(pad);
  const ad_scalar_t distance = terrain.signedDistance<ad_scalar_t>(pad);
  features(featureRow(candidate, kDistanceFeature)) = distance;
  features.segment<kPositionDim>(featureRow(candidate, kSlipFeature)) =
      velocity - velocity.dot(normal) * normal;
  const ad_scalar_t alignment = padAxis.dot(normal);
  if (entry.surface) {
    features(featureRow(candidate, kTiltFeature)) =
        ad_scalar_t(1.0) - alignment;
    features(featureRow(candidate, kConeFeature)) = ad_scalar_t(0.0);
    features(featureRow(candidate, kClearanceFeature)) = distance;
    return;
  }
  const ad_scalar_t excess = ad_scalar_t(constants.cosineOfMaxTilt) - alignment;
  features(featureRow(candidate, kTiltFeature)) =
      CppAD::CondExpGt(excess, ad_scalar_t(0.0), excess, ad_scalar_t(0.0));
  const AdVector3 force = centroidal_model::getForceContactEwrtWorld(
                              input, entry.mpcContactIndex, info) /
                          ad_scalar_t(constants.weight);
  const ad_scalar_t normalForce = force.dot(normal);
  const AdVector3 tangential = force - normalForce * normal;
  features(featureRow(candidate, kConeFeature)) =
      ad_scalar_t(constants.friction) * normalForce -
      CppAD::sqrt(
          tangential.squaredNorm() +
          ad_scalar_t(constants.coneSmoothing * constants.coneSmoothing));
  // The collision sphere's center is the geom radius inside the pad, along
  // the pad's z axis (the certifier's inactive-pad clearance).
  const ad_scalar_t radius(entry.geomRadius);
  features(featureRow(candidate, kClearanceFeature)) =
      terrain.sphereDistance<ad_scalar_t>(pad + radius * padAxis, radius);
}

}  // namespace

absl::StatusOr<std::unique_ptr<ContactFeatureFunction>> robotContactFeatures(
    const PinocchioInterface& pinocchio,
    const CentroidalModelInfo& info,
    const std::vector<RobotFeatureCandidate>& candidates,
    const TerrainModel& terrain,
    const RobotFeatureSettings& settings,
    const FeatureLibrarySettings& library) {
  std::vector<TapedCandidate> taped;
  const pinocchio::Model& model = pinocchio.getModel();
  for (const RobotFeatureCandidate& candidate : candidates) {
    if (!model.existFrame(candidate.frameName)) {
      return absl::NotFoundError(
          absl::StrCat("contact features: the MPC's model has no frame '",
                       candidate.frameName, "'"));
    }
    if (candidate.mpcContactIndex >=
        info.numThreeDofContacts + info.numSixDofContacts) {
      return absl::InvalidArgumentError(
          absl::StrCat("contact features: '", candidate.frameName,
                       "' is not a contact of the MPC"));
    }
    const pinocchio::FrameIndex frame = model.getFrameId(candidate.frameName);
    taped.push_back(
        TapedCandidate{.frame = frame,
                       .joint = model.frames[frame].parentJoint,
                       .mpcContactIndex = candidate.mpcContactIndex,
                       .surface = candidate.kind == ContactKind::kSurface,
                       .padRotation = candidate.padRotation,
                       .geomRadius = candidate.geomRadius});
  }
  const TapeConstants constants{
      .friction = settings.friction,
      .coneSmoothing = settings.coneSmoothing,
      .cosineOfMaxTilt = std::cos(settings.maxPointTilt),
      .weight = info.robotMass * kGravityMagnitude};
  const int numCandidates = static_cast<int>(candidates.size());
  const ContactFeatureFunction::Function function =
      [pinocchioAd = pinocchio.toCppAd(), infoAd = info.toCppAd(), taped,
       terrain, constants, numCandidates](
          const ad_vector_t& state, const ad_vector_t& input) -> ad_vector_t {
    PinocchioInterfaceCppAd interface = pinocchioAd;
    CentroidalModelPinocchioMappingCppAd mapping(infoAd);
    mapping.setPinocchioInterface(interface);
    const ad_vector_t q = mapping.getPinocchioJointPosition(state);
    updateCentroidalDynamics(interface, infoAd, q);
    const ad_vector_t v = mapping.getPinocchioJointVelocity(state, input);
    const PinocchioInterfaceCppAd::Model& adModel = interface.getModel();
    PinocchioInterfaceCppAd::Data& data = interface.getData();
    pinocchio::forwardKinematics(adModel, data, q, v);
    pinocchio::updateFramePlacements(adModel, data);
    ad_vector_t features = ad_vector_t::Zero(numFeatureRows(numCandidates));
    for (int i = 0; i < numCandidates; ++i) {
      candidateFeatures(adModel, data, infoAd, terrain, taped[i], constants,
                        input, i, features);
    }
    features.segment<kMomentumRateFeatures>(momentumRateRow(numCandidates)) =
        getMnormRateCmomComRtWorldEwrtWorld(interface, infoAd, input);
    return features;
  };
  return ContactFeatureFunction::Create(
      function, static_cast<int>(info.stateDim),
      static_cast<int>(info.inputDim), numCandidates, library);
}

}  // namespace ocs2::humanoid::multi_contact
