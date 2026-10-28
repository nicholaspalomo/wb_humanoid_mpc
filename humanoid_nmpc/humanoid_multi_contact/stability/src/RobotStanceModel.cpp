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

#include "humanoid_multi_contact/stability/RobotStanceModel.h"

#include <cmath>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "ocs2_core/Dimensions.h"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/algorithm/centroidal.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/multibody/joint/joint-free-flyer.hpp"
#include "pinocchio/parsers/urdf.hpp"

#include "humanoid_common_mpc/common/StatusMacros.h"

namespace ocs2::humanoid::multi_contact {
namespace {

// The joint index of the free-flyer root: Pinocchio's joint 0 is the universe.
constexpr pinocchio::JointIndex kRootJoint = 1;
// The first index of the base orientation in the configuration (x, y, z, w).
constexpr int kBaseQuaternionIndex = kPositionDim;
// How far from 1 the norm of a base quaternion may be.
constexpr double kQuaternionNormTolerance = 1.0e-6;

/** [v]x, the cross-product matrix. */
Eigen::Matrix3d crossMatrix(const Eigen::Vector3d& v) {
  Eigen::Matrix3d matrix;
  matrix << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
  return matrix;
}

}  // namespace

RobotStanceModel::RobotStanceModel(pinocchio::Model model,
                                   ContactSet contacts,
                                   std::vector<pinocchio::JointIndex> padJoints)
    : model_(std::move(model)),
      data_(model_),
      contacts_(std::move(contacts)),
      padJoints_(std::move(padJoints)) {}

absl::StatusOr<std::unique_ptr<RobotStanceModel>> RobotStanceModel::Create(
    absl::string_view urdfPath, const ContactSet& contacts) {
  pinocchio::Model model;
  try {  // NOLINT(exceptions): Pinocchio's URDF parser throws; its exception becomes a Status here.
    pinocchio::urdf::buildModel(std::string(urdfPath),
                                pinocchio::JointModelFreeFlyer(), model);
    // NOLINTNEXTLINE(exceptions): the boundary of the try above.
  } catch (const std::exception& error) {
    return absl::InvalidArgumentError(
        absl::StrCat("the URDF '", urdfPath,
                     "' does not give a Pinocchio model: ", error.what()));
  }
  std::vector<pinocchio::JointIndex> padJoints;
  padJoints.reserve(contacts.size());
  for (const ContactCandidate& candidate : contacts.candidates()) {
    if (!model.existJointName(candidate.parentJoint)) {
      return absl::NotFoundError(
          absl::StrCat("contact candidate '", candidate.name, "' of '",
                       contacts.robot(), "': the URDF '", urdfPath,
                       "' has no joint '", candidate.parentJoint, "'"));
    }
    padJoints.push_back(model.getJointId(candidate.parentJoint));
  }
  std::unique_ptr<RobotStanceModel> stanceModel = absl::WrapUnique(
      new RobotStanceModel(std::move(model), contacts, std::move(padJoints)));
  stanceModel->mass_ = pinocchio::computeTotalMass(stanceModel->model_);
  for (pinocchio::JointIndex joint = kRootJoint + 1;
       joint < static_cast<pinocchio::JointIndex>(stanceModel->model_.njoints);
       ++joint) {
    if (stanceModel->model_.joints[joint].nv() != 1) continue;
    const int velocityIndex = stanceModel->model_.joints[joint].idx_v();
    const int positionIndex = stanceModel->model_.joints[joint].idx_q();
    stanceModel->actuatedJoints_.push_back(ActuatedJoint{
        .name = stanceModel->model_.names[joint],
        .velocityIndex = velocityIndex,
        .positionIndex = positionIndex,
        .effortLimit = stanceModel->model_.effortLimit(velocityIndex),
        .lowerLimit = stanceModel->model_.lowerPositionLimit(positionIndex),
        .upperLimit = stanceModel->model_.upperPositionLimit(positionIndex)});
  }
  const RobotConfiguration neutral;
  if (absl::Status status = stanceModel->setConfiguration(neutral);
      !status.ok()) {
    return status;
  }
  return stanceModel;
}

bool RobotStanceModel::hasJoint(absl::string_view name) const {
  return model_.existJointName(std::string(name));
}

absl::StatusOr<Eigen::VectorXd> RobotStanceModel::generalizedPositionOf(
    const RobotConfiguration& configuration) const {
  if (static_cast<Eigen::Index>(configuration.jointNames.size()) !=
      configuration.jointPositions.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("a configuration has ", configuration.jointNames.size(),
                     " joint names but ", configuration.jointPositions.size(),
                     " positions"));
  }
  if (!configuration.basePosition.allFinite() ||
      !configuration.jointPositions.allFinite() ||
      !configuration.baseOrientation.coeffs().allFinite()) {
    return absl::InvalidArgumentError(
        "a configuration needs finite values and a unit base quaternion");
  }
  Eigen::VectorXd q = pinocchio::neutral(model_);
  q.head<kPositionDim>() = configuration.basePosition;
  q.segment<kQuaternionCoefficients>(kBaseQuaternionIndex) =
      configuration.baseOrientation.coeffs();
  absl::flat_hash_set<std::string> seen;
  for (size_t i = 0; i < configuration.jointNames.size(); ++i) {
    const std::string& name = configuration.jointNames[i];
    if (!model_.existJointName(name) || name == model_.names[kRootJoint] ||
        model_.joints[model_.getJointId(name)].nq() != 1) {
      return absl::InvalidArgumentError(absl::StrCat(
          "'", name, "' is not a joint of one degree of freedom of the model"));
    }
    if (!seen.insert(name).second) {
      return absl::InvalidArgumentError(
          absl::StrCat("joint '", name, "' is given twice"));
    }
    q(model_.joints[model_.getJointId(name)].idx_q()) =
        configuration.jointPositions(static_cast<Eigen::Index>(i));
  }
  return q;
}

RobotConfiguration RobotStanceModel::configurationOf(
    const Eigen::VectorXd& q) const {
  RobotConfiguration configuration;
  configuration.basePosition = q.head<kPositionDim>();
  configuration.baseOrientation = Eigen::Quaterniond(Eigen::Vector4d(
      q.segment<kQuaternionCoefficients>(kBaseQuaternionIndex)));
  configuration.jointPositions.resize(
      static_cast<Eigen::Index>(actuatedJoints_.size()));
  for (size_t i = 0; i < actuatedJoints_.size(); ++i) {
    configuration.jointNames.push_back(actuatedJoints_[i].name);
    configuration.jointPositions(static_cast<Eigen::Index>(i)) =
        q(actuatedJoints_[i].positionIndex);
  }
  return configuration;
}

absl::Status RobotStanceModel::setConfiguration(
    const RobotConfiguration& configuration) {
  if (!(std::abs(configuration.baseOrientation.norm() - 1.0) <=
        kQuaternionNormTolerance)) {
    return absl::InvalidArgumentError(
        "a configuration needs finite values and a unit base quaternion");
  }
  ASSIGN_OR_RETURN(const Eigen::VectorXd q,
                   generalizedPositionOf(configuration));
  return setGeneralizedPosition(q);
}

absl::Status RobotStanceModel::setGeneralizedPosition(
    const Eigen::VectorXd& q) {
  if (q.size() != model_.nq || !q.allFinite() ||
      !(std::abs(
            q.segment<kQuaternionCoefficients>(kBaseQuaternionIndex).norm() -
            1.0) <= kQuaternionNormTolerance)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "a generalized position needs ", model_.nq,
        " finite values and a unit base quaternion, not ", q.size()));
  }
  q_ = q;
  q_.segment<kQuaternionCoefficients>(kBaseQuaternionIndex).normalize();
  // Also computes the forward kinematics and the center of mass.
  centerOfMassJacobian_ = pinocchio::jacobianCenterOfMass(
      model_, data_, q_, /*computeSubtreeComs=*/false);
  centerOfMass_ = data_.com[0];
  pinocchio::computeJointJacobians(model_, data_, q_);
  pinocchio::computeGeneralizedGravity(model_, data_, q_);
  return absl::OkStatus();
}

absl::StatusOr<CentroidalMomentum> RobotStanceModel::centroidalMomentum(
    const Eigen::VectorXd& q, const Eigen::VectorXd& v) {
  if (v.size() != model_.nv || !v.allFinite()) {
    return absl::InvalidArgumentError(
        absl::StrCat("a generalized velocity needs ", model_.nv,
                     " finite values, not ", v.size()));
  }
  RETURN_IF_ERROR(setGeneralizedPosition(q));
  const pinocchio::Force& momentum =
      pinocchio::computeCentroidalMomentum(model_, data_, q_, v);
  return CentroidalMomentum{.linear = momentum.linear(),
                            .angular = momentum.angular()};
}

absl::StatusOr<pinocchio::JointIndex> RobotStanceModel::jointIndex(
    absl::string_view name) const {
  if (!model_.existJointName(std::string(name))) {
    return absl::NotFoundError(
        absl::StrCat("the model has no joint '", name, "'"));
  }
  return model_.getJointId(std::string(name));
}

absl::Status RobotStanceModel::relaxPositionLimits(double relaxation) {
  if (!(relaxation >= 0.0) || !std::isfinite(relaxation)) {
    return absl::InvalidArgumentError(
        "the joint limits' relaxation must be finite and not negative");
  }
  for (ActuatedJoint& joint : actuatedJoints_) {
    joint.lowerLimit -= relaxation;
    joint.upperLimit += relaxation;
  }
  return absl::OkStatus();
}

Eigen::Vector3d RobotStanceModel::padPosition(int index) const {
  const pinocchio::SE3& joint = data_.oMi[padJoints_[index]];
  return joint.act(Eigen::Vector3d(contacts_.candidate(index).offset));
}

Eigen::Matrix3d RobotStanceModel::padRotation(int index) const {
  return data_.oMi[padJoints_[index]].rotation() *
         contacts_.candidate(index).padRotation;
}

Eigen::Matrix<double, kPositionDim, Eigen::Dynamic>
RobotStanceModel::pointJacobian(int index, const Eigen::Vector3d& point) const {
  return jointPointJacobian(padJoints_[index], point);
}

Eigen::Matrix<double, kRotationTangentDim, Eigen::Dynamic>
RobotStanceModel::jointAngularJacobian(pinocchio::JointIndex joint) const {
  pinocchio::Data::Matrix6x jacobian =
      pinocchio::Data::Matrix6x::Zero(kTwistDim, model_.nv);
  pinocchio::getJointJacobian(model_, data_, joint,
                              pinocchio::LOCAL_WORLD_ALIGNED, jacobian);
  return jacobian.bottomRows<kRotationTangentDim>();
}

Eigen::Matrix<double, kPositionDim, Eigen::Dynamic>
RobotStanceModel::jointPointJacobian(pinocchio::JointIndex joint,
                                     const Eigen::Vector3d& point) const {
  pinocchio::Data::Matrix6x jacobian =
      pinocchio::Data::Matrix6x::Zero(kTwistDim, model_.nv);
  // At the joint's origin, in world-aligned axes: (v_o; omega).
  pinocchio::getJointJacobian(model_, data_, joint,
                              pinocchio::LOCAL_WORLD_ALIGNED, jacobian);
  const Eigen::Vector3d lever = point - data_.oMi[joint].translation();
  // v_point = v_o + omega x lever = v_o - [lever]x omega.
  return jacobian.topRows<kPositionDim>() -
         crossMatrix(lever) * jacobian.bottomRows<kRotationTangentDim>();
}

}  // namespace ocs2::humanoid::multi_contact
