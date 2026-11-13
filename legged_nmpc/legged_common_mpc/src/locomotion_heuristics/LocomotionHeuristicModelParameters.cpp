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

#include "legged_common_mpc/locomotion_heuristics/LocomotionHeuristicModelParameters.h"

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/Types.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/multibody/fwd.hpp"

#include "legged_common_mpc/common/BipedFeet.h"
#include "legged_common_mpc/common/ModelSettings.h"
#include "legged_common_mpc/common/MpcRobotModelBase.h"
#include "legged_common_mpc/common/StatusMacros.h"
#include "legged_common_mpc/common/Types.h"
#include "legged_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "legged_common_mpc/pinocchio_model/pinocchioUtils.h"

namespace ocs2::legged {

namespace {

/** `worldOffsetXy` [m], a horizontal offset from the base in the world,
 * turned into the heading frame of the base yaw `yawBaseRtWorld` [rad]. */
vector2_t headingFrameOf(const vector2_t& worldOffsetXy,
                         scalar_t yawBaseRtWorld) {
  const scalar_t cosYaw = std::cos(yawBaseRtWorld);
  const scalar_t sinYaw = std::sin(yawBaseRtWorld);
  return vector2_t(cosYaw * worldOffsetXy.x() + sinYaw * worldOffsetXy.y(),
                   -sinYaw * worldOffsetXy.x() + cosYaw * worldOffsetXy.y());
}

/** The contact each hip entry walks up from: each leg's first contact, or the
 * biped's contacts 0 and 1 on a set without legs. */
std::vector<size_t> hipWalkContactsOf(const ModelSettings& modelSettings) {
  std::vector<size_t> contacts;
  if (modelSettings.legs.empty()) {
    for (size_t foot = 0; foot < static_cast<size_t>(kBipedFootCount); ++foot) {
      contacts.push_back(foot);
    }
    return contacts;
  }
  for (const ModelSettings::Leg& leg : modelSettings.legs) {
    contacts.push_back(leg.contactIndices.empty() ? 0 : leg.contactIndices[0]);
  }
  return contacts;
}

/** Returns the horizontal positions `posContactsRtWorldEwrtWorld` [m] relative
 * to the base at `posBaseRtWorldEwrtWorldXy` [m], in the heading frame of the
 * base yaw `yawBaseRtWorld` [rad]. */
std::vector<vector2_t> computePosContactsRtBaseEwrtHeadingXy(
    absl::Span<const vector3_t> posContactsRtWorldEwrtWorld,
    const vector2_t& posBaseRtWorldEwrtWorldXy,
    scalar_t yawBaseRtWorld) {
  std::vector<vector2_t> posContactsRtBaseEwrtHeadingXy;
  posContactsRtBaseEwrtHeadingXy.reserve(posContactsRtWorldEwrtWorld.size());
  for (const vector3_t& posContactRtWorldEwrtWorld :
       posContactsRtWorldEwrtWorld) {
    posContactsRtBaseEwrtHeadingXy.push_back(headingFrameOf(
        vector2_t(posContactRtWorldEwrtWorld.head<kHorizontalPositionDim>() -
                  posBaseRtWorldEwrtWorldXy),
        yawBaseRtWorld));
  }
  return posContactsRtBaseEwrtHeadingXy;
}

/**
 * Returns the horizontal world position [m] of the hip of leg `leg`, whose hip
 * walk starts at contact `contact`, read from the frame placements in `data`:
 * the leg's hipFrameName when it has one; otherwise the last joint before the
 * floating base above the contact's parent joint (hipJointOf()); otherwise
 * `fallbackPosHipRtWorldEwrtWorldXy`. FailedPrecondition when the hip frame is
 * not a frame of `model`.
 */
absl::StatusOr<vector2_t> findPosHipRtWorldEwrtWorldXy(
    const pinocchio::Model& model,
    const pinocchio::Data& data,
    const ModelSettings& modelSettings,
    size_t leg,
    size_t contact,
    const vector2_t& fallbackPosHipRtWorldEwrtWorldXy) {
  if (leg < modelSettings.legs.size() &&
      !modelSettings.legs[leg].hipFrameName.empty()) {
    const std::string& hipFrameName = modelSettings.legs[leg].hipFrameName;
    if (!model.existFrame(hipFrameName)) {
      return absl::FailedPreconditionError(
          absl::StrCat("[LocomotionHeuristicModelParameters] the hip frame '",
                       hipFrameName, "' of leg '", modelSettings.legs[leg].name,
                       "' is not a frame of the MPC model."));
    }
    return vector2_t(data.oMf[hipFrameIdOf(model, hipFrameName)]
                         .translation()
                         .head<kHorizontalPositionDim>());
  }
  const std::vector<std::string> contactParentJointNames =
      modelSettings.contactParentJointNamesInContactOrder();
  if (contact < contactParentJointNames.size() &&
      model.existJointName(contactParentJointNames[contact])) {
    const pinocchio::JointIndex joint =
        model.getJointId(contactParentJointNames[contact]);
    if (joint > 0) {
      return vector2_t(data.oMi[hipJointOf(model, joint)]
                           .translation()
                           .head<kHorizontalPositionDim>());
    }
  }
  return fallbackPosHipRtWorldEwrtWorldXy;
}

/**
 * Returns r_hip of every hip entry (hipWalkContactsOf()) [m], relative to the
 * base at `posBaseRtWorldEwrtWorldXy` in the heading frame of `yawBaseRtWorld`,
 * from the frame placements in `data` at the nominal posture, where the
 * contacts stand at `posContactsRtWorldEwrtWorld`. r_hip is the landmark
 * hip_centered_stepping places the foot under. A leg's hip is the frame
 * ModelSettings::Leg::hipFrameName names (the task file's hip_frame_name, or
 * the derived hip joint, which a default biped leg has). Without one - a set
 * without legs, or a default leg whose foot names no joint - walk up the
 * kinematic tree from the joint that carries the leg's first contact to the
 * LAST joint before the floating base: that joint is the hip, however the URDF
 * happens to name it. Its placement in the world at the nominal posture,
 * expressed relative to the base and with the vertical dropped, is r_hip. A
 * joint's frame sits at the joint, so a default biped leg gets the same r_hip
 * either way. FailedPrecondition naming a hip frame that is not a frame of
 * `model`.
 */
absl::StatusOr<std::vector<vector2_t>> computePosHipsRtBaseEwrtHeadingXy(
    const pinocchio::Model& model,
    const pinocchio::Data& data,
    const ModelSettings& modelSettings,
    absl::Span<const vector3_t> posContactsRtWorldEwrtWorld,
    const vector2_t& posBaseRtWorldEwrtWorldXy,
    scalar_t yawBaseRtWorld) {
  const std::vector<size_t> hipWalkContacts = hipWalkContactsOf(modelSettings);
  std::vector<vector2_t> posHipsRtBaseEwrtHeadingXy(hipWalkContacts.size(),
                                                    vector2_t::Zero());
  for (size_t leg = 0; leg < hipWalkContacts.size(); ++leg) {
    const size_t contact = hipWalkContacts[leg];
    // Fallback: this leg's contact's own horizontal position relative to the
    // base. It is exact for a leg standing vertically, which is the posture
    // this is evaluated in, so a robot whose tree cannot be walked still gets
    // a sensible landmark rather than the base origin - which would put both
    // feet in the same place.
    const vector2_t fallbackPosHipRtWorldEwrtWorldXy =
        contact < posContactsRtWorldEwrtWorld.size()
            ? vector2_t(posContactsRtWorldEwrtWorld[contact]
                            .head<kHorizontalPositionDim>())
            : posBaseRtWorldEwrtWorldXy;
    ASSIGN_OR_RETURN(
        const vector2_t posHipRtWorldEwrtWorldXy,
        findPosHipRtWorldEwrtWorldXy(model, data, modelSettings, leg, contact,
                                     fallbackPosHipRtWorldEwrtWorldXy));
    // Rotate the world-frame offset back into the heading frame (the base yaw
    // only), so that the stored r_hip is a property of the robot rather than of
    // the heading it happened to be nominally standing at.
    posHipsRtBaseEwrtHeadingXy[leg] = headingFrameOf(
        vector2_t(posHipRtWorldEwrtWorldXy - posBaseRtWorldEwrtWorldXy),
        yawBaseRtWorld);
  }
  return posHipsRtBaseEwrtHeadingXy;
}

}  // namespace

std::string LocomotionHeuristicModelParameters::summary() const {
  std::vector<std::string> hips;
  for (size_t leg = 0; leg < posHipsRtBaseEwrtHeadingXy.size(); ++leg) {
    hips.push_back(absl::StrCat("leg ", leg, " (",
                                posHipsRtBaseEwrtHeadingXy[leg].x(), ", ",
                                posHipsRtBaseEwrtHeadingXy[leg].y(), ") m"));
  }
  return absl::StrCat("  totalMass: ", totalMass, " kg, weight: ", totalWeight,
                      " N, gravity: ", gravity,
                      " m/s^2\n  nomPosComRtFeetmeanEwrtWorldZ: ",
                      nomPosComRtFeetmeanEwrtWorldZ,
                      " m above the mean foot height\n  hip (base frame): ",
                      absl::StrJoin(hips, ", "), "\n");
}

std::vector<vector2_t> computeNomPosContactsRtBaseEwrtHeadingXy(
    PinocchioInterface& pinocchioInterface,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel,
    const vector_t& nominalState) {
  const vector_t q = mpcRobotModel.getGeneralizedCoordinates(nominalState);
  const std::vector<vector3_t> posContactsRtWorldEwrtWorld =
      computePosContactsRtWorldEwrtWorld<scalar_t>(q, pinocchioInterface,
                                                   mpcRobotModel);
  const vector2_t posBaseRtWorldEwrtWorldXy =
      mpcRobotModel.getPosBaseRtWorldEwrtWorld(nominalState)
          .head<kHorizontalPositionDim>();
  const scalar_t yawBaseRtWorld = mpcRobotModel.getYawBaseRtWorld(nominalState);
  return computePosContactsRtBaseEwrtHeadingXy(
      posContactsRtWorldEwrtWorld, posBaseRtWorldEwrtWorldXy, yawBaseRtWorld);
}

absl::StatusOr<LocomotionHeuristicModelParameters>
deriveLocomotionHeuristicModelParameters(
    PinocchioInterface& pinocchioInterface,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel,
    const vector_t& nominalState) {
  const pinocchio::Model& model = pinocchioInterface.getModel();
  const pinocchio::Data& data = pinocchioInterface.getData();
  LocomotionHeuristicModelParameters derived;
  ASSIGN_OR_RETURN(derived.gravity, modelGravity(model));
  derived.totalMass = pinocchio::computeTotalMass(model);
  if (!(derived.totalMass > 0.0)) {
    return absl::FailedPreconditionError(
        "[LocomotionHeuristicModelParameters] the robot model has no mass; "
        "every wrench heuristic divides by the weight.");
  }
  derived.totalWeight = derived.totalMass * derived.gravity;

  const vector_t q = mpcRobotModel.getGeneralizedCoordinates(nominalState);
  // The pendulum length of the capture point: the one every LIP consumer shares
  // (computeProxyPosComRtFeetmeanEwrtWorldZ), measured at the nominal posture
  // and used only as the fallback when the per-solve measurement is unavailable
  // or nonsense. It also leaves the frame placements of the nominal posture in
  // `data`, which the hip walk below reads.
  derived.nomPosComRtFeetmeanEwrtWorldZ =
      computeProxyPosComRtFeetmeanEwrtWorldZ(q, pinocchioInterface,
                                             mpcRobotModel);
  const std::vector<vector3_t> posContactsRtWorldEwrtWorld =
      getPosContactsRtWorldEwrtWorld<scalar_t>(pinocchioInterface,
                                               mpcRobotModel);
  if (posContactsRtWorldEwrtWorld.empty()) {
    return absl::FailedPreconditionError(
        "[LocomotionHeuristicModelParameters] the robot model has no contact "
        "frames.");
  }

  const vector2_t posBaseRtWorldEwrtWorldXy =
      mpcRobotModel.getPosBaseRtWorldEwrtWorld(nominalState)
          .head<kHorizontalPositionDim>();
  const scalar_t yawBaseRtWorld = mpcRobotModel.getYawBaseRtWorld(nominalState);
  ASSIGN_OR_RETURN(
      derived.posHipsRtBaseEwrtHeadingXy,
      computePosHipsRtBaseEwrtHeadingXy(
          model, data, mpcRobotModel.modelSettings, posContactsRtWorldEwrtWorld,
          posBaseRtWorldEwrtWorldXy, yawBaseRtWorld));
  // Where each contact stands at the nominal posture, from the same frame
  // placements.
  derived.nomPosContactsRtBaseEwrtHeadingXy =
      computePosContactsRtBaseEwrtHeadingXy(posContactsRtWorldEwrtWorld,
                                            posBaseRtWorldEwrtWorldXy,
                                            yawBaseRtWorld);
  return derived;
}

}  // namespace ocs2::legged
