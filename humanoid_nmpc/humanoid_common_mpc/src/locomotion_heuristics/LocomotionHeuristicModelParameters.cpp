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

#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicModelParameters.h"

#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"

namespace ocs2::humanoid {

std::string LocomotionHeuristicModelParameters::summary() const {
  return absl::StrCat("  totalMass: ", totalMass, " kg, weight: ", totalWeight, " N, gravity: ", gravity,
                      " m/s^2\n  nominalComHeight: ", nominalComHeight, " m above the mean foot height\n  hip (base frame): left (",
                      hipPositionInBaseFrame[CONTACT_LEFT_INDEX].x(), ", ", hipPositionInBaseFrame[CONTACT_LEFT_INDEX].y(), ") m, right (",
                      hipPositionInBaseFrame[CONTACT_RIGHT_INDEX].x(), ", ", hipPositionInBaseFrame[CONTACT_RIGHT_INDEX].y(), ") m\n");
}

absl::StatusOr<LocomotionHeuristicModelParameters> deriveLocomotionHeuristicModelParameters(
    PinocchioInterface& pinocchioInterface,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel,
    const vector_t& nominalState,
    scalar_t gravity) {
  if (!(gravity > 0.0)) {
    return absl::InvalidArgumentError(absl::StrCat("[LocomotionHeuristicModelParameters] gravity must be positive, got ", gravity, "."));
  }
  LocomotionHeuristicModelParameters derived;
  derived.gravity = gravity;

  const pinocchio::Model& model = pinocchioInterface.getModel();
  pinocchio::Data& data = pinocchioInterface.getData();
  derived.totalMass = pinocchio::computeTotalMass(model);
  if (!(derived.totalMass > 0.0)) {
    return absl::FailedPreconditionError(
        "[LocomotionHeuristicModelParameters] the robot model has no mass; every wrench heuristic divides by the weight.");
  }
  derived.totalWeight = derived.totalMass * gravity;

  const vector_t q = mpcRobotModel.getGeneralizedCoordinates(nominalState);
  pinocchio::centerOfMass(model, data, q, false);
  pinocchio::updateFramePlacements(model, data);

  // The pendulum length of the capture point: the centre of mass above the feet, not above the world origin, so that
  // it is the same number on a robot standing on a box. Measured at the nominal posture and used only as the fallback
  // when the per-solve measurement is unavailable or nonsense.
  const std::vector<vector3_t> feet = computeContactPositions<scalar_t>(q, pinocchioInterface, mpcRobotModel);
  if (feet.empty()) {
    return absl::FailedPreconditionError("[LocomotionHeuristicModelParameters] the robot model has no contact frames.");
  }
  scalar_t meanFootZ = 0.0;
  for (const vector3_t& foot : feet) meanFootZ += foot(2) / static_cast<scalar_t>(feet.size());
  derived.nominalComHeight = data.com[0](2) - meanFootZ;

  // The landmark hip_centered_stepping places the foot under. Walk up the kinematic tree from the joint that carries
  // this leg's contact frame to the LAST joint before the floating base - that joint is the hip, however the URDF
  // happens to name it, which is why this does not look for a name. Its placement in the world at the nominal posture,
  // expressed relative to the base and with the vertical dropped, is r_hip.
  const vector6_t basePose = mpcRobotModel.getBasePose(nominalState);
  const vector2_t basePosition = basePose.head<2>();
  const scalar_t baseYaw = basePose(3);
  const scalar_t cosYaw = std::cos(baseYaw);
  const scalar_t sinYaw = std::sin(baseYaw);
  const std::vector<std::string>& contactParentJointNames = mpcRobotModel.modelSettings.contactParentJointNames;
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    // Fallback: this foot's own horizontal position relative to the base. It is exact for a leg standing vertically,
    // which is the posture this is evaluated in, so a robot whose tree cannot be walked still gets a sensible
    // landmark rather than the base origin - which would put both feet in the same place.
    vector2_t hipInWorld = foot < feet.size() ? vector2_t(feet[foot].head<2>()) : basePosition;
    pinocchio::JointIndex joint = 0;
    if (foot < contactParentJointNames.size() && model.existJointName(contactParentJointNames[foot])) {
      joint = model.getJointId(contactParentJointNames[foot]);
    }
    if (joint > 0) {
      while (model.parents[joint] > 1) joint = model.parents[joint];  // stop at the first child of the floating base
      hipInWorld = data.oMi[joint].translation().head<2>();
    }
    const vector2_t relative = hipInWorld - basePosition;
    // Rotate the world-frame offset back into the base frame, so that the stored r_hip is a property of the robot
    // rather than of the heading it happened to be nominally standing at.
    derived.hipPositionInBaseFrame[foot] =
        vector2_t(cosYaw * relative.x() + sinYaw * relative.y(), -sinYaw * relative.x() + cosYaw * relative.y());
  }
  return derived;
}

}  // namespace ocs2::humanoid
