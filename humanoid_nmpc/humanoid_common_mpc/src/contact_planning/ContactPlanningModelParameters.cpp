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

#include "humanoid_common_mpc/contact_planning/ContactPlanningModelParameters.h"

#include <cmath>
#include <sstream>

#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/multibody/joint/joint-generic.hpp>

#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"

namespace ocs2::humanoid {

namespace {

/** True for a revolute joint whose axis is the vertical of its parent frame (RZ, or an unaligned axis close to +-z). */
bool isRevoluteAboutVertical(const pinocchio::Model& model, pinocchio::JointIndex joint) {
  const std::string name = model.joints[joint].shortname();
  if (name == "JointModelRZ" || name == "JointModelRUBZ") return true;
  if (name == "JointModelRevoluteUnaligned" || name == "JointModelRevoluteUnboundedUnaligned") {
    // The axis is the last column of the joint's motion subspace for a one-dof joint: read it through the joint data.
    const pinocchio::JointModelRevoluteUnaligned* unaligned =
        boost::get<pinocchio::JointModelRevoluteUnaligned>(&model.joints[joint].toVariant());
    if (unaligned != nullptr) return std::abs(unaligned->axis(2)) > 0.9;
  }
  return false;
}

}  // namespace

void ContactPlanningModelParameters::applyTo(ContactPlanningConfig& config) const {
  config.yawTorqueBudget.torsionalFrictionTorque = torsionalFrictionTorque;
  config.yawTorqueBudget.doubleSupportYawCouple = doubleSupportYawCouple;
  config.hipYawRange.lower = footYawOffsetLower;
  config.hipYawRange.upper = footYawOffsetUpper;
  if (config.shared.comHeight <= 0.0 && comHeight > 0.0) config.shared.comHeight = comHeight;
  if (config.zmpSupportRegion.halfWidthX <= 0.0 && zmpHalfWidthX > 0.0) config.zmpSupportRegion.halfWidthX = zmpHalfWidthX;
  if (config.zmpSupportRegion.halfWidthY <= 0.0 && zmpHalfWidthY > 0.0) config.zmpSupportRegion.halfWidthY = zmpHalfWidthY;
}

std::string ContactPlanningModelParameters::summary() const {
  std::ostringstream out;
  out << "mass " << totalMass << " kg, comHeight " << comHeight << " m, footprint half extents " << zmpHalfWidthX << " x " << zmpHalfWidthY
      << " m, torsionalFrictionTorque " << torsionalFrictionTorque << " N m, doubleSupportYawCouple " << doubleSupportYawCouple
      << " N m, foot yaw bounds";
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    out << " [" << footYawOffsetLower[foot] << ", " << footYawOffsetUpper[foot] << "]";
    if (foot < hipYawJoints.size())
      out << " (" << (hipYawJoints[foot].empty() ? "no hip yaw joint found, fallback" : hipYawJoints[foot]) << ")";
  }
  return out.str();
}

ContactPlanningModelParameters deriveContactPlanningModelParameters(PinocchioInterface& pinocchioInterface,
                                                                    const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                                                                    const vector_t& nominalState,
                                                                    const std::vector<std::string>& contactParentJointNames,
                                                                    const ContactPlanningGroundParameters& ground,
                                                                    scalar_t gravity,
                                                                    scalar_t nominalStepWidth) {
  ContactPlanningModelParameters derived;
  const auto& model = pinocchioInterface.getModel();
  auto& data = pinocchioInterface.getData();
  derived.totalMass = pinocchio::computeTotalMass(model);
  const scalar_t weight = derived.totalMass * gravity;

  {
    const vector_t q = mpcRobotModel.getGeneralizedCoordinates(nominalState);
    pinocchio::centerOfMass(model, data, q, false);
    const scalar_t comZ = data.com[0](2);
    const std::vector<vector3_t> feet = computeContactPositions<scalar_t>(q, pinocchioInterface, mpcRobotModel);
    scalar_t meanFootZ = 0.0;
    for (const vector3_t& foot : feet) meanFootZ += foot(2) / static_cast<scalar_t>(feet.size());
    derived.comHeight = comZ - meanFootZ;
  }
  derived.zmpHalfWidthX = ground.footprintHalfLengthX;
  derived.zmpHalfWidthY = ground.footprintHalfWidthY;
  derived.torsionalFrictionTorque = ground.torsionalFrictionCoefficient * weight;
  derived.doubleSupportYawCouple = ground.frictionCoefficient * 0.5 * weight * nominalStepWidth;

  derived.hipYawJoints.assign(N_CONTACTS, "");
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    pinocchio::JointIndex joint = 0;
    if (foot < contactParentJointNames.size() && model.existJointName(contactParentJointNames[foot])) {
      joint = model.getJointId(contactParentJointNames[foot]);
    }
    // Walk up from the joint that carries the contact frame to the first revolute joint about the vertical: the hip yaw.
    while (joint > 0 && !isRevoluteAboutVertical(model, joint)) joint = model.parents[joint];
    scalar_t lower = -ContactPlanningConfig::kDefaultFootYawOffset;
    scalar_t upper = ContactPlanningConfig::kDefaultFootYawOffset;
    if (joint > 0) {
      const int idx = model.joints[joint].idx_q();
      const scalar_t jointLower = std::max(model.lowerPositionLimit(idx), -M_PI);
      const scalar_t jointUpper = std::min(model.upperPositionLimit(idx), M_PI);
      if (jointLower < 0.0 && jointUpper > 0.0) {
        lower = jointLower;
        upper = jointUpper;
        derived.hipYawJoints[foot] = model.names[joint];
      }
    }
    derived.footYawOffsetLower[foot] = lower;
    derived.footYawOffsetUpper[foot] = upper;
  }
  return derived;
}

}  // namespace ocs2::humanoid
