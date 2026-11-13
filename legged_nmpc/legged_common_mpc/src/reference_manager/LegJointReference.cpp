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

#include "legged_common_mpc/reference_manager/LegJointReference.h"

#include <cmath>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "Eigen/SVD"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/Types.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/multibody/fwd.hpp"

#include "legged_common_mpc/common/ModelSettings.h"
#include "legged_common_mpc/common/MpcRobotModelBase.h"
#include "legged_common_mpc/common/Types.h"
#include "legged_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"

namespace ocs2::legged {

namespace {

// The names of leg_joint_reference (LegJointReference).
// LINT.IfChange(leg_joint_reference_names)
constexpr char kDefaultPostureName[] = "default_posture";
constexpr char kFootMotionName[] = "foot_motion";
// LINT.ThenChange(//legged_nmpc/legged_mpc_config/task_file.proto:leg_joint_reference)

/** [m/rad] A Jacobian column at most this long moves the contact by nothing:
 * the joint is not on the contact's leg. */
constexpr scalar_t kMinJacobianColumnNorm = 1.0e-9;

/** [rad] The step of the central differences of the leg Jacobians. */
constexpr scalar_t kJacobianDifferenceStep = 1.0e-5;

/** The fixed-point iterations of the second-order leg map
 * (FootMotionLegReference::computeLegJointDisplacement()). */
constexpr int kSecondOrderIterations = 3;

/** R_z(yaw)^T `vector`: world axes into the heading frame at `yaw`. */
vector3_t worldToHeading(scalar_t yaw, const vector3_t& vector) {
  const scalar_t cosine = std::cos(yaw);
  const scalar_t sine = std::sin(yaw);
  return vector3_t(cosine * vector.x() + sine * vector.y(),
                   -sine * vector.x() + cosine * vector.y(), vector.z());
}

/**
 * [m/rad] The columns `jointIndices` of the position Jacobian of the frame
 * `frameId` at the state `state` of `mpcRobotModel`, in the heading frame at
 * `yawBaseRtWorld`: the frame's velocity relative to the base per joint rate.
 */
matrix_t computeLegJacobian(const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                            PinocchioInterface& pinocchioInterface,
                            const vector_t& state,
                            pinocchio::FrameIndex frameId,
                            scalar_t yawBaseRtWorld,
                            absl::Span<const size_t> jointIndices) {
  const pinocchio::Model& model = pinocchioInterface.getModel();
  matrix_t jacobian = matrix_t::Zero(kTwistDim, model.nv);
  pinocchio::computeFrameJacobian(
      model, pinocchioInterface.getData(),
      mpcRobotModel.getGeneralizedCoordinates(state), frameId,
      pinocchio::LOCAL_WORLD_ALIGNED, jacobian);
  matrix_t legJacobian(kPositionDim,
                       static_cast<Eigen::Index>(jointIndices.size()));
  for (size_t entry = 0; entry < jointIndices.size(); ++entry) {
    legJacobian.col(static_cast<Eigen::Index>(entry)) = worldToHeading(
        yawBaseRtWorld,
        jacobian.col(static_cast<Eigen::Index>(kTwistDim + jointIndices[entry]))
            .head<kPositionDim>());
  }
  return legJacobian;
}

/** J' (J J')^-1 v: the minimum-norm joint rates of the full-rank 3 x k
 * `jacobian` that move the contact at `linvelContactRtBaseEwrtHeading`. */
vector_t minimumNormJointRates(
    const matrix_t& jacobian, const vector3_t& linvelContactRtBaseEwrtHeading) {
  const matrix3_t gram = jacobian * jacobian.transpose();
  return jacobian.transpose() * gram.inverse() * linvelContactRtBaseEwrtHeading;
}

}  // namespace

std::vector<std::string> legJointReferenceNames() {
  return {kDefaultPostureName, kFootMotionName};
}

absl::string_view legJointReferenceName(LegJointReference reference) {
  switch (reference) {
    case LegJointReference::kDefaultPosture:
      return kDefaultPostureName;
    case LegJointReference::kFootMotion:
      return kFootMotionName;
  }
  // Not a value of the enum.
  return kDefaultPostureName;
}

absl::StatusOr<LegJointReference> legJointReferenceFromName(
    absl::string_view name) {
  if (name == kDefaultPostureName) return LegJointReference::kDefaultPosture;
  if (name == kFootMotionName) return LegJointReference::kFootMotion;
  return absl::InvalidArgumentError(
      absl::StrCat("leg_joint_reference is '", name, "'; the references are ",
                   absl::StrJoin(legJointReferenceNames(), ", "), "."));
}

absl::StatusOr<FootMotionLegReference> FootMotionLegReference::Create(
    PinocchioInterface pinocchioInterface,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel,
    const vector_t& nominalState) {
  const ModelSettings& settings = mpcRobotModel.modelSettings;
  const pinocchio::Model& model = pinocchioInterface.getModel();
  pinocchio::Data& data = pinocchioInterface.getData();
  const size_t jointDim = mpcRobotModel.getJointDim();
  if (static_cast<size_t>(model.nv) != kTwistDim + jointDim) {
    return absl::InvalidArgumentError(absl::StrCat(
        "[FootMotionLegReference] the Pinocchio model has ", model.nv,
        " velocities; the leg joint reference expects the base's ", kTwistDim,
        " and the ", jointDim, " MPC joints."));
  }
  if (mpcRobotModel.getJointVelocitiesStartindex() + jointDim !=
      mpcRobotModel.getInputDim()) {
    return absl::InvalidArgumentError(
        "[FootMotionLegReference] the leg joint reference regularizes joint "
        "velocities of the input, and this model's input does not end in "
        "them (the centroidal MPC's does).");
  }
  const vector_t q = mpcRobotModel.getGeneralizedCoordinates(nominalState);
  const vector_t nomJointPositions = mpcRobotModel.getJointAngles(nominalState);
  const std::vector<vector3_t> posContactsRtWorldEwrtWorld =
      computePosContactsRtWorldEwrtWorld<scalar_t>(q, pinocchioInterface,
                                                   mpcRobotModel);
  const vector3_t posBaseRtWorldEwrtWorld =
      mpcRobotModel.getPosBaseRtWorldEwrtWorld(nominalState);
  const scalar_t yawBaseRtWorld = mpcRobotModel.getYawBaseRtWorld(nominalState);

  std::vector<Leg> legs;
  legs.reserve(settings.contactNames.size());
  std::vector<bool> jointTaken(jointDim, false);
  for (size_t contact = 0; contact < settings.contactNames.size(); ++contact) {
    const std::string& name = settings.contactNames[contact];
    if (!settings.isPointContact(contact)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "[FootMotionLegReference] leg_joint_reference 'foot_motion' "
          "moves 3-DoF point contacts; contact '",
          name, "' is a 6-DoF one."));
    }
    // The leg's joints are solved from the motion of a point fixed on the
    // foot; a rolling contact's frame is a wheel center, whose contact point
    // moves over the rim.
    if (settings.isRollingContact(contact)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "[FootMotionLegReference] leg_joint_reference 'foot_motion' "
          "moves point contacts fixed on their links; contact '",
          name, "' is a rolling one (model_settings.contacts_rolling)."));
    }
    matrix_t jacobian = matrix_t::Zero(kTwistDim, model.nv);
    pinocchio::computeFrameJacobian(model, data, q, model.getFrameId(name),
                                    pinocchio::LOCAL_WORLD_ALIGNED, jacobian);
    Leg leg;
    for (size_t joint = 0; joint < jointDim; ++joint) {
      const Eigen::Index column = static_cast<Eigen::Index>(kTwistDim + joint);
      if (jacobian.col(column).head<kPositionDim>().norm() <=
          kMinJacobianColumnNorm) {
        continue;
      }
      if (jointTaken[joint]) {
        return absl::InvalidArgumentError(absl::StrCat(
            "[FootMotionLegReference] leg_joint_reference "
            "'foot_motion' needs one contact per leg; joint '",
            settings.mpcModelJointNames[joint], "' moves contact '", name,
            "' and an earlier contact."));
      }
      jointTaken[joint] = true;
      leg.jointIndices.push_back(joint);
    }
    // The leg's columns in the heading frame: the base fixed, so the contact's
    // world velocity is its velocity relative to the base.
    const pinocchio::FrameIndex frameId = model.getFrameId(name);
    const matrix_t legJacobian =
        computeLegJacobian(mpcRobotModel, pinocchioInterface, nominalState,
                           frameId, yawBaseRtWorld, leg.jointIndices);
    if (leg.jointIndices.size() < static_cast<size_t>(kPositionDim)) {
      return absl::InvalidArgumentError(
          absl::StrCat("[FootMotionLegReference] contact '", name,
                       "' is moved by ", leg.jointIndices.size(),
                       " joints; its velocity needs ", kPositionDim, "."));
    }
    const Eigen::JacobiSVD<matrix_t> svd(
        legJacobian, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const vector_t& singularValues = svd.singularValues();
    if (!(singularValues(singularValues.size() - 1) >= kMinSingularValue)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "[FootMotionLegReference] the leg of contact '", name,
          "' is singular at the nominal posture (smallest singular value ",
          singularValues(singularValues.size() - 1), " m/rad)."));
    }
    leg.nomJointPositions.resize(
        static_cast<Eigen::Index>(leg.jointIndices.size()));
    for (size_t entry = 0; entry < leg.jointIndices.size(); ++entry) {
      leg.nomJointPositions(static_cast<Eigen::Index>(entry)) =
          nomJointPositions(static_cast<Eigen::Index>(leg.jointIndices[entry]));
    }
    leg.jacobian = legJacobian;
    leg.inverseJacobian = svd.matrixV() *
                          singularValues.cwiseInverse().asDiagonal() *
                          svd.matrixU().transpose();
    // dJ / dq_j by central differences: the second-order term of the map.
    for (size_t entry = 0; entry < leg.jointIndices.size(); ++entry) {
      const Eigen::Index joint =
          static_cast<Eigen::Index>(leg.jointIndices[entry]);
      vector_t forward = nominalState;
      vector_t backward = nominalState;
      vector_t joints = nomJointPositions;
      joints(joint) += kJacobianDifferenceStep;
      mpcRobotModel.setJointAngles(forward, joints);
      joints(joint) -= 2.0 * kJacobianDifferenceStep;
      mpcRobotModel.setJointAngles(backward, joints);
      leg.jacobianDerivatives.push_back(
          (computeLegJacobian(mpcRobotModel, pinocchioInterface, forward,
                              frameId, yawBaseRtWorld, leg.jointIndices) -
           computeLegJacobian(mpcRobotModel, pinocchioInterface, backward,
                              frameId, yawBaseRtWorld, leg.jointIndices)) /
          (2.0 * kJacobianDifferenceStep));
    }
    leg.nomPosContactRtBaseEwrtHeading =
        worldToHeading(yawBaseRtWorld, posContactsRtWorldEwrtWorld[contact] -
                                           posBaseRtWorldEwrtWorld);
    legs.push_back(std::move(leg));
  }
  return FootMotionLegReference(std::move(legs));
}

vector_t FootMotionLegReference::computeLegJointDisplacement(
    const Leg& leg, const vector3_t& errPosContactRtBaseEwrtHeading) {
  // p(q_0 + dq) = p_0 + J dq + 1/2 H[dq, dq] + O(dq^3), solved for dq by
  // fixed-point iteration from the first-order dq = J^+ dp.
  vector_t errJointPositions =
      leg.inverseJacobian * errPosContactRtBaseEwrtHeading;
  for (int iteration = 0; iteration < kSecondOrderIterations; ++iteration) {
    vector3_t secondOrderTerm = vector3_t::Zero();
    for (size_t entry = 0; entry < leg.jacobianDerivatives.size(); ++entry) {
      secondOrderTerm += leg.jacobianDerivatives[entry] * errJointPositions *
                         errJointPositions(static_cast<Eigen::Index>(entry));
    }
    errJointPositions = leg.inverseJacobian * (errPosContactRtBaseEwrtHeading -
                                               0.5 * secondOrderTerm);
  }
  return errJointPositions;
}

void FootMotionLegReference::setLegJointPositions(
    size_t contactIndex,
    const vector3_t& posContactRtBaseEwrtHeading,
    Eigen::Ref<vector_t> jointPositions) const {
  const Leg& leg = legs_[contactIndex];
  const vector_t errJointPositions = computeLegJointDisplacement(
      leg, posContactRtBaseEwrtHeading - leg.nomPosContactRtBaseEwrtHeading);
  for (size_t entry = 0; entry < leg.jointIndices.size(); ++entry) {
    const Eigen::Index row = static_cast<Eigen::Index>(entry);
    jointPositions(static_cast<Eigen::Index>(leg.jointIndices[entry])) =
        leg.nomJointPositions(row) + errJointPositions(row);
  }
}

void FootMotionLegReference::setLegJointVelocities(
    size_t contactIndex,
    const vector3_t& posContactRtBaseEwrtHeading,
    const vector3_t& linvelContactRtBaseEwrtHeading,
    Eigen::Ref<vector_t> jointVelocities) const {
  const Leg& leg = legs_[contactIndex];
  const vector_t errJointPositions = computeLegJointDisplacement(
      leg, posContactRtBaseEwrtHeading - leg.nomPosContactRtBaseEwrtHeading);
  // The Jacobian at the posture setLegJointPositions() gives, to first order.
  matrix_t jacobian = leg.jacobian;
  for (size_t entry = 0; entry < leg.jacobianDerivatives.size(); ++entry) {
    jacobian += leg.jacobianDerivatives[entry] *
                errJointPositions(static_cast<Eigen::Index>(entry));
  }
  const vector_t jointRates =
      minimumNormJointRates(jacobian, linvelContactRtBaseEwrtHeading);
  for (size_t entry = 0; entry < leg.jointIndices.size(); ++entry) {
    jointVelocities(static_cast<Eigen::Index>(leg.jointIndices[entry])) =
        jointRates(static_cast<Eigen::Index>(entry));
  }
}

vector3_t FootMotionLegReference::computeStanceLinvelContactRtBaseEwrtHeading(
    const vector2_t& cmdLinvelBaseRtWorldEwrtHeadingXy,
    scalar_t cmdRateYawBaseRtWorld,
    const vector3_t& posContactRtBaseEwrtHeading) {
  return vector3_t(-cmdLinvelBaseRtWorldEwrtHeadingXy.x() +
                       cmdRateYawBaseRtWorld * posContactRtBaseEwrtHeading.y(),
                   -cmdLinvelBaseRtWorldEwrtHeadingXy.y() -
                       cmdRateYawBaseRtWorld * posContactRtBaseEwrtHeading.x(),
                   0.0);
}

}  // namespace ocs2::legged
