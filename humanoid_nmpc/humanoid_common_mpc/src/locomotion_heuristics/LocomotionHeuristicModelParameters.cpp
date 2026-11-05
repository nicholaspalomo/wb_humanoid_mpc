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

#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicModelParameters.h"

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/Types.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/multibody/fwd.hpp"

#include "humanoid_common_mpc/common/BipedFeet.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"

namespace ocs2::humanoid {

std::string LocomotionHeuristicModelParameters::summary() const {
  return absl::StrCat(
      "  totalMass: ", totalMass, " kg, weight: ", totalWeight,
      " N, gravity: ", gravity, " m/s^2\n  nomPosComRtFeetmeanEwrtWorldZ: ",
      nomPosComRtFeetmeanEwrtWorldZ,
      " m above the mean foot height\n  hip (base frame): left (",
      posHipsRtBaseEwrtHeadingXy[kContactLeftIndex].x(), ", ",
      posHipsRtBaseEwrtHeadingXy[kContactLeftIndex].y(), ") m, right (",
      posHipsRtBaseEwrtHeadingXy[kContactRightIndex].x(), ", ",
      posHipsRtBaseEwrtHeadingXy[kContactRightIndex].y(), ") m\n");
}

absl::StatusOr<LocomotionHeuristicModelParameters>
deriveLocomotionHeuristicModelParameters(
    PinocchioInterface& pinocchioInterface,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel,
    const vector_t& nominalState) {
  const pinocchio::Model& model = pinocchioInterface.getModel();
  pinocchio::Data& data = pinocchioInterface.getData();
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

  // The landmark hip_centered_stepping places the foot under. Walk up the
  // kinematic tree from the joint that carries this leg's contact frame to the
  // LAST joint before the floating base - that joint is the hip, however the
  // URDF happens to name it, which is why this does not look for a name. Its
  // placement in the world at the nominal posture, expressed relative to the
  // base and with the vertical dropped, is r_hip.
  const vector2_t posBaseRtWorldEwrtWorldXy =
      mpcRobotModel.getPosBaseRtWorldEwrtWorld(nominalState)
          .head<kHorizontalPositionDim>();
  const scalar_t yawBaseRtWorld = mpcRobotModel.getYawBaseRtWorld(nominalState);
  const scalar_t cosYaw = std::cos(yawBaseRtWorld);
  const scalar_t sinYaw = std::sin(yawBaseRtWorld);
  const std::vector<std::string>& contactParentJointNames =
      mpcRobotModel.modelSettings.contactParentJointNames;
  for (size_t foot = 0; foot < kNumContacts; ++foot) {
    // Fallback: this foot's own horizontal position relative to the base. It is
    // exact for a leg standing vertically, which is the posture this is
    // evaluated in, so a robot whose tree cannot be walked still gets a
    // sensible landmark rather than the base origin - which would put both feet
    // in the same place.
    vector2_t posHipRtWorldEwrtWorldXy =
        foot < posContactsRtWorldEwrtWorld.size()
            ? vector2_t(posContactsRtWorldEwrtWorld[foot]
                            .head<kHorizontalPositionDim>())
            : posBaseRtWorldEwrtWorldXy;
    pinocchio::JointIndex joint = 0;
    if (foot < contactParentJointNames.size() &&
        model.existJointName(contactParentJointNames[foot])) {
      joint = model.getJointId(contactParentJointNames[foot]);
    }
    if (joint > 0) {
      while (model.parents[joint] > 1) {
        joint = model.parents[joint];  // stop at the first child of the
      }
      // floating base
      posHipRtWorldEwrtWorldXy =
          data.oMi[joint].translation().head<kHorizontalPositionDim>();
    }
    const vector2_t posHipRtBaseEwrtWorldXy =
        posHipRtWorldEwrtWorldXy - posBaseRtWorldEwrtWorldXy;
    // Rotate the world-frame offset back into the heading frame (the base yaw
    // only), so that the stored r_hip is a property of the robot rather than of
    // the heading it happened to be nominally standing at.
    derived.posHipsRtBaseEwrtHeadingXy[foot] =
        vector2_t(cosYaw * posHipRtBaseEwrtWorldXy.x() +
                      sinYaw * posHipRtBaseEwrtWorldXy.y(),
                  -sinYaw * posHipRtBaseEwrtWorldXy.x() +
                      cosYaw * posHipRtBaseEwrtWorldXy.y());
  }
  return derived;
}

}  // namespace ocs2::humanoid
