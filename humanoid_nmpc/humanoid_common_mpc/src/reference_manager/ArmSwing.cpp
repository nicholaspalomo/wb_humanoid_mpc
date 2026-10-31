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

#include "humanoid_common_mpc/reference_manager/ArmSwing.h"

#include <cmath>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "ocs2_core/manifold/UnitQuaternionMath.h"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/kinematics.hpp"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/StatusMacros.h"

namespace ocs2::humanoid {

namespace {

// LINT.IfChange(arm_swing_generator_names)
constexpr char kProceduralGeneratorName[] = "procedural";
constexpr char kNoGeneratorName[] = "none";
// LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_config/arm_swing_config.proto:arm_swing_generator)

/**
 * The smallest share of a derived direction's own scale - a shoulder's lever,
 * the part of an elbow's axis across the arm's plane - that is taken as a
 * direction rather than as numerical noise.
 */
constexpr scalar_t kMinimumDirectionShare = 0.1;

/** One driven joint, as the derivation sees it at the nominal posture. Passive
 * data. */
struct ArmJoint {
  vector3_t posLinkRtWorldEwrtWorld =
      vector3_t::Zero();                             // [m] the joint's origin
  vector3_t axisJointEwrtWorld = vector3_t::Zero();  // the unit rotation axis
  vector3_t carriedCom =
      vector3_t::Zero();  // [m] the center of mass of everything the joint
                          // carries, in the world
};

/** `jointName` of the MPC model at the posture the kinematics of `data` were
 * computed at. */
absl::StatusOr<ArmJoint> armJointAt(const pinocchio::Model& model,
                                    const pinocchio::Data& data,
                                    const std::string& jointName) {
  if (!model.existJointName(jointName)) {
    return absl::InvalidArgumentError(
        absl::StrCat("[deriveArmSwingDirections] the arm joint '", jointName,
                     "' is not in the robot model."));
  }
  const pinocchio::JointIndex joint = model.getJointId(jointName);
  if (model.joints[joint].nv() != 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "[deriveArmSwingDirections] the arm joint '", jointName, "' has ",
        model.joints[joint].nv(), " degrees of freedom, not 1."));
  }
  ArmJoint result;
  result.posLinkRtWorldEwrtWorld = data.oMi[joint].translation();
  // The angular rows of the world-frame joint Jacobian
  // (computeJointJacobians()): the joint's axis in the world.
  result.axisJointEwrtWorld = data.J
                                  .block<kRotationTangentDim, 1>(
                                      kPositionDim, model.joints[joint].idx_v())
                                  .normalized();
  scalar_t mass = 0.0;
  for (const pinocchio::JointIndex carried : model.subtrees[joint]) {
    const pinocchio::Inertia& inertia = model.inertias[carried];
    mass += inertia.mass();
    result.carriedCom +=
        inertia.mass() * data.oMi[carried].act(inertia.lever());
  }
  if (!(mass > 0.0)) {
    return absl::InvalidArgumentError(
        absl::StrCat("[deriveArmSwingDirections] the arm joint '", jointName,
                     "' carries no mass."));
  }
  result.carriedCom /= mass;
  return result;
}

/** +1 when a positive rate of `shoulder` swings what it carries forward along
 * the base's x axis `forward`, else -1. */
absl::StatusOr<scalar_t> shoulderDirection(const ArmJoint& shoulder,
                                           const vector3_t& forward,
                                           absl::string_view jointName) {
  const vector3_t lever =
      shoulder.carriedCom - shoulder.posLinkRtWorldEwrtWorld;
  const scalar_t forwardSpeed =
      forward.dot(shoulder.axisJointEwrtWorld.cross(lever));
  if (!(std::abs(forwardSpeed) >= kMinimumDirectionShare * lever.norm())) {
    return absl::InvalidArgumentError(absl::StrCat(
        "[deriveArmSwingDirections] the shoulder joint '", jointName,
        "' hardly swings the arm forward at the nominal posture (",
        forwardSpeed, " m/s per rad/s on a ", lever.norm(), " m lever)."));
  }
  return forwardSpeed > 0.0 ? 1.0 : -1.0;
}

/**
 * +1 when a positive rate of `elbow` closes the angle between the upper arm,
 * towards `shoulder`, and what the elbow carries, else -1. With w the forearm
 * and u the upper arm from the elbow, the angle changes at -a . (w x u) / |w x
 * u|.
 */
absl::StatusOr<scalar_t> elbowDirection(const ArmJoint& elbow,
                                        const ArmJoint& shoulder,
                                        absl::string_view jointName) {
  const vector3_t forearm = elbow.carriedCom - elbow.posLinkRtWorldEwrtWorld;
  const vector3_t upperArm =
      shoulder.posLinkRtWorldEwrtWorld - elbow.posLinkRtWorldEwrtWorld;
  const vector3_t normal = forearm.cross(upperArm);
  const scalar_t normalNorm = normal.norm();
  const scalar_t across =
      (normalNorm > 0.0) ? elbow.axisJointEwrtWorld.dot(normal) / normalNorm
                         : 0.0;
  if (!(std::abs(across) >= kMinimumDirectionShare)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "[deriveArmSwingDirections] the elbow joint '", jointName,
        "' hardly flexes the arm at the nominal posture (its axis is ", across,
        " across the plane of the upper arm and the forearm)."));
  }
  return across > 0.0 ? 1.0 : -1.0;
}

}  // namespace

std::vector<std::string> armSwingGeneratorNames() {
  return {kProceduralGeneratorName, kNoGeneratorName};
}

absl::StatusOr<ArmSwingGenerator> armSwingGeneratorFromName(
    absl::string_view name) {
  if (name == kProceduralGeneratorName) return ArmSwingGenerator::kProcedural;
  if (name == kNoGeneratorName) return ArmSwingGenerator::kNone;
  return absl::InvalidArgumentError(
      absl::StrCat("arm_swing.generator is '", name, "'; the generators are ",
                   absl::StrJoin(armSwingGeneratorNames(), ", "), "."));
}

absl::StatusOr<ArmSwingSettings> armSwingSettingsFromConfig(
    const mpc_config::ArmSwingConfig& config) {
  ArmSwingSettings settings;
  ASSIGN_OR_RETURN(settings.generator,
                   armSwingGeneratorFromName(config.generator));
  // Negated comparisons, so that a NaN is refused rather than accepted.
  if (!(config.shoulder_gain >= 0.0) || !std::isfinite(config.shoulder_gain)) {
    return absl::InvalidArgumentError(
        absl::StrCat("arm_swing.shoulder_gain is ", config.shoulder_gain,
                     "; it must be finite and not negative."));
  }
  if (!(config.elbow_gain >= 0.0) || !std::isfinite(config.elbow_gain)) {
    return absl::InvalidArgumentError(
        absl::StrCat("arm_swing.elbow_gain is ", config.elbow_gain,
                     "; it must be finite and not negative."));
  }
  if (!std::isfinite(config.phase_offset)) {
    return absl::InvalidArgumentError(absl::StrCat("arm_swing.phase_offset is ",
                                                   config.phase_offset,
                                                   "; it must be finite."));
  }
  settings.shoulderGain = config.shoulder_gain;
  settings.elbowGain = config.elbow_gain;
  settings.phaseOffset = config.phase_offset;
  return settings;
}

absl::StatusOr<ArmSwingDirections> deriveArmSwingDirections(
    PinocchioInterface& pinocchioInterface,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel,
    const vector_t& nominalState) {
  const ModelSettings& settings = mpcRobotModel.modelSettings;
  if (!settings.hasArmSwingJoints) {
    return absl::FailedPreconditionError(
        "[deriveArmSwingDirections] the robot names no arm joints "
        "(model_settings.arm_joint_names).");
  }
  const pinocchio::Model& model = pinocchioInterface.getModel();
  pinocchio::Data& data = pinocchioInterface.getData();
  const vector_t q = mpcRobotModel.getGeneralizedCoordinates(nominalState);
  pinocchio::forwardKinematics(model, data, q);
  pinocchio::computeJointJacobians(model, data, q);
  // The base's forward axis at the nominal posture, in the world.
  const vector3_t forward =
      quaternionRotationMatrix<scalar_t>(
          mpcRobotModel.getQuatBaseRtWorldXyzw(nominalState))
          .col(0);

  /** One arm's two joints, by MPC joint index, and where their directions go.
   * Passive data. */
  struct Arm {
    size_t shoulderIndex;
    size_t elbowIndex;
    scalar_t* absl_nonnull shoulderDirectionOut;
    scalar_t* absl_nonnull elbowDirectionOut;
  };
  ArmSwingDirections directions;
  const std::vector<Arm> arms = {
      {.shoulderIndex = settings.j_l_shoulder_y_index,
       .elbowIndex = settings.j_l_elbow_y_index,
       .shoulderDirectionOut = &directions.leftShoulder,
       .elbowDirectionOut = &directions.leftElbow},
      {.shoulderIndex = settings.j_r_shoulder_y_index,
       .elbowIndex = settings.j_r_elbow_y_index,
       .shoulderDirectionOut = &directions.rightShoulder,
       .elbowDirectionOut = &directions.rightElbow},
  };
  for (const Arm& arm : arms) {
    const std::string& shoulderName =
        settings.mpcModelJointNames[arm.shoulderIndex];
    const std::string& elbowName = settings.mpcModelJointNames[arm.elbowIndex];
    ASSIGN_OR_RETURN(const ArmJoint shoulder,
                     armJointAt(model, data, shoulderName));
    ASSIGN_OR_RETURN(const ArmJoint elbow, armJointAt(model, data, elbowName));
    ASSIGN_OR_RETURN(*arm.shoulderDirectionOut,
                     shoulderDirection(shoulder, forward, shoulderName));
    ASSIGN_OR_RETURN(*arm.elbowDirectionOut,
                     elbowDirection(elbow, shoulder, elbowName));
  }
  return directions;
}

}  // namespace ocs2::humanoid
