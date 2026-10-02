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

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>

#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/multibody/joint/joint-generic.hpp>

#include "absl/strings/str_cat.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"

namespace ocs2::humanoid {

namespace {

/**
 * The axis an unaligned revolute joint (bounded or unbounded) rotates about, in the joint's own frame, read through
 * pinocchio's generic joint interface: the motion subspace of a revolute joint is the single unit twist about its
 * axis, so the angular part of its one column is the axis the joint model carries, copied verbatim.
 *
 * JointModelRevoluteUnaligned and JointModelRevoluteUnboundedUnaligned are two UNRELATED alternatives of pinocchio's
 * joint variant - the unbounded one is not a subclass of the bounded one, it merely carries a Vector3 axis of its own -
 * so asking the variant for one alternative cannot serve both. Asking only for the bounded alternative, which an
 * earlier version did while admitting both shortnames, made the second name dead code: for a continuous off-axis
 * joint the axis came back empty, the walk up the kinematic tree ran past the hip yaw all the way to the root, and the
 * leg silently got the symmetric fallback bounds while summary() reported "no hip yaw joint found" for a joint that
 * was right there. The motion subspace is the same for both, and reading it needs no access to the variant.
 */
vector3_t unalignedRevoluteAxis(const PinocchioInterface::Model& model, pinocchio::JointIndex joint) {
  const PinocchioInterface::Model::JointData jointData = model.joints[joint].createData();
  const Eigen::Matrix<scalar_t, 6, Eigen::Dynamic> motionSubspace = jointData.S().matrix();
  return motionSubspace.block<3, 1>(pinocchio::Motion::ANGULAR, 0);
}

/** The rotation axis of a revolute joint in the joint's own frame, or empty for any other kind of joint. */
std::optional<vector3_t> revoluteAxis(const PinocchioInterface::Model& model, pinocchio::JointIndex joint) {
  const std::string name = model.joints[joint].shortname();
  if (name == "JointModelRX" || name == "JointModelRUBX") return vector3_t::UnitX();
  if (name == "JointModelRY" || name == "JointModelRUBY") return vector3_t::UnitY();
  if (name == "JointModelRZ" || name == "JointModelRUBZ") return vector3_t::UnitZ();
  if (name == "JointModelRevoluteUnaligned" || name == "JointModelRevoluteUnboundedUnaligned") {
    return unalignedRevoluteAxis(model, joint);
  }
  return std::nullopt;
}

}  // namespace

HipYawRange deriveHipYawRange(const PinocchioInterface::Model& model, const std::string& contactParentJointName) {
  HipYawRange range;
  if (!model.existJointName(contactParentJointName)) return range;
  // The joint frames in the world at the neutral configuration: what the URDF's joint placements compose to.
  PinocchioInterface::Data data(model);
  pinocchio::forwardKinematics(model, data, pinocchio::neutral(model));
  pinocchio::JointIndex joint = model.getJointId(contactParentJointName);
  scalar_t verticalComponent = 0.0;  // the world z component of the hip yaw axis: +1 up, -1 down
  for (; joint > 0; joint = model.parents[joint]) {
    const std::optional<vector3_t> axis = revoluteAxis(model, joint);
    if (!axis.has_value()) continue;
    verticalComponent = (data.oMi[joint].rotation() * axis->normalized())(2);
    if (std::abs(verticalComponent) > 0.9) break;
  }
  if (joint == 0) return range;
  if (model.joints[joint].nq() > 1) {
    // The position limits are only angles when the joint stores an angle, i.e. when nq == 1. An unbounded
    // ("continuous" in URDF) revolute joint has nq == 2 and stores (cos q, sin q), and pinocchio's URDF parser fills
    // both of those entries of lower/upperPositionLimit with +-1.01. Reading them at idx_q as if they were angles
    // bounded the foot yaw of a joint that has no limit at all to +-1.01 rad, a number that comes from the unit-circle
    // representation and means nothing here. An unbounded joint turns all the way round, so its honest range is the
    // full circle, which the heading model clips to [-pi, pi] regardless.
    range.lower = -M_PI;
    range.upper = M_PI;
    range.joint = model.names[joint];
    return range;
  }
  const int idx = model.joints[joint].idx_q();
  const scalar_t jointLower = std::max(model.lowerPositionLimit(idx), -M_PI);
  const scalar_t jointUpper = std::min(model.upperPositionLimit(idx), M_PI);
  if (!(jointLower < 0.0 && jointUpper > 0.0)) return range;
  // The foot yaw is +q about an upward axis and -q about a downward one, which mirrors the range.
  const bool pointsDown = verticalComponent < 0.0;
  range.lower = pointsDown ? -jointUpper : jointLower;
  range.upper = pointsDown ? -jointLower : jointUpper;
  range.joint = model.names[joint];
  return range;
}

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
  std::string out = absl::StrCat("mass ", totalMass, " kg, comHeight ", comHeight, " m, footprint half extents ", zmpHalfWidthX, " x ",
                                 zmpHalfWidthY, " m, torsionalFrictionTorque ", torsionalFrictionTorque, " N m, doubleSupportYawCouple ",
                                 doubleSupportYawCouple, " N m, foot yaw bounds");
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    absl::StrAppend(&out, " [", footYawOffsetLower[foot], ", ", footYawOffsetUpper[foot], "]");
    if (foot < hipYawJoints.size()) {
      absl::StrAppend(&out, " (", hipYawJoints[foot].empty() ? "no hip yaw joint found, fallback" : hipYawJoints[foot], ")");
    }
  }
  return out;
}

ContactPlanningModelParameters deriveContactPlanningModelParameters(PinocchioInterface& pinocchioInterface,
                                                                    const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                                                                    const vector_t& nominalState,
                                                                    const std::vector<std::string>& contactParentJointNames,
                                                                    const ContactPlanningGroundParameters& ground,
                                                                    scalar_t gravity,
                                                                    scalar_t nominalStepWidth) {
  ContactPlanningModelParameters derived;
  const PinocchioInterface::Model& model = pinocchioInterface.getModel();
  derived.totalMass = pinocchio::computeTotalMass(model);
  const scalar_t weight = derived.totalMass * gravity;

  // The pendulum length every LIP consumer shares (computeComHeightAboveFeet), at the nominal posture.
  derived.comHeight = computeComHeightAboveFeet(mpcRobotModel.getGeneralizedCoordinates(nominalState), pinocchioInterface, mpcRobotModel);
  derived.zmpHalfWidthX = ground.footprintHalfLengthX;
  derived.zmpHalfWidthY = ground.footprintHalfWidthY;
  derived.torsionalFrictionTorque = ground.torsionalFrictionCoefficient * weight;
  derived.doubleSupportYawCouple = ground.frictionCoefficient * 0.5 * weight * nominalStepWidth;

  derived.hipYawJoints.assign(N_CONTACTS, "");
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    const HipYawRange range =
        foot < contactParentJointNames.size() ? deriveHipYawRange(model, contactParentJointNames[foot]) : HipYawRange();
    derived.footYawOffsetLower[foot] = range.lower;
    derived.footYawOffsetUpper[foot] = range.upper;
    derived.hipYawJoints[foot] = range.joint;
  }
  return derived;
}

}  // namespace ocs2::humanoid
