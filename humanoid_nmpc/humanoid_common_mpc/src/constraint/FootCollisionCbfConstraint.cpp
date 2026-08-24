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

#include <pinocchio/fwd.hpp>

#include "humanoid_common_mpc/constraint/FootCollisionCbfConstraint.h"

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

#include <ocs2_core/misc/LoadData.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace ocs2::humanoid {

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

FootCollisionCbfConstraint::FootCollisionCbfConstraint(const SwitchedModelReferenceManager& referenceManager,
                                                       const PinocchioInterface& pinocchioInterface,
                                                       const MpcRobotModelBase<ad_scalar_t>& mpcRobotModel,
                                                       const Config& config,
                                                       std::string costName,
                                                       const ModelSettings& modelSettings)
    : StateInputConstraintCppAd(ConstraintOrder::Linear),
      referenceManagerPtr_(&referenceManager),
      pinocchioInterfaceCppAd_(pinocchioInterface.toCppAd()),
      mpcRobotModelPtr_(&mpcRobotModel),
      cfg_(std::move(config)) {
  const auto& model = pinocchioInterfaceCppAd_.getModel();
  leftAnkleId_ = model.getFrameId(cfg_.leftAnkleFrame);
  rightAnkleId_ = model.getFrameId(cfg_.rightAnkleFrame);
  leftFootCenterId_ = model.getFrameId(cfg_.leftFootCenterFrame);
  rightFootCenterId_ = model.getFrameId(cfg_.rightFootCenterFrame);
  leftFoot1Id_ = model.getFrameId(cfg_.leftFootFrame1);
  rightFoot1Id_ = model.getFrameId(cfg_.rightFootFrame1);
  leftFoot2Id_ = model.getFrameId(cfg_.leftFootFrame2);
  rightFoot2Id_ = model.getFrameId(cfg_.rightFootFrame2);
  leftKneeId_ = model.getFrameId(cfg_.leftKneeFrame);
  rightKneeId_ = model.getFrameId(cfg_.rightKneeFrame);

  initialize(mpcRobotModelPtr_->getStateDim(), mpcRobotModelPtr_->getInputDim(), 3, std::move(costName), modelSettings.modelFolderCppAd,
             modelSettings.recompileLibrariesCppAd, modelSettings.verboseCppAd);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

FootCollisionCbfConstraint::FootCollisionCbfConstraint(const FootCollisionCbfConstraint& other)
    : StateInputConstraintCppAd(other),
      referenceManagerPtr_(other.referenceManagerPtr_),
      pinocchioInterfaceCppAd_(other.pinocchioInterfaceCppAd_),
      mpcRobotModelPtr_(other.mpcRobotModelPtr_),
      cfg_(other.cfg_),
      leftAnkleId_(other.leftAnkleId_),
      rightAnkleId_(other.rightAnkleId_),
      leftFootCenterId_(other.leftFootCenterId_),
      rightFootCenterId_(other.rightFootCenterId_),
      leftFoot1Id_(other.leftFoot1Id_),
      rightFoot1Id_(other.rightFoot1Id_),
      leftFoot2Id_(other.leftFoot2Id_),
      rightFoot2Id_(other.rightFoot2Id_),
      leftKneeId_(other.leftKneeId_),
      rightKneeId_(other.rightKneeId_),
      numConstraints_(other.numConstraints_),
      isActive_(other.isActive_) {}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

bool FootCollisionCbfConstraint::isActive(scalar_t time) const {
  if (!isActive_) return false;

  // Inactivate the constraint if both feet are in contact. Prevents it from fighting against the stance foot constraints.
  auto contactFlags = referenceManagerPtr_->getContactFlags(time);
  return !(contactFlags[0] && contactFlags[1]);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

ad_vector_t FootCollisionCbfConstraint::constraintFunction(ad_scalar_t time,
                                                           const ad_vector_t& state,
                                                           const ad_vector_t& input,
                                                           const ad_vector_t& parameters) const {
  const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;

  const auto& model = pinocchioInterfaceCppAd_.getModel();
  auto data = pinocchioInterfaceCppAd_.getData();

  const ad_vector_t q = mpcRobotModelPtr_->getGeneralizedCoordinates(state);
  ad_vector_t v = ad_vector_t::Zero(model.nv);
  v.tail(model.nv - 6) = mpcRobotModelPtr_->getJointVelocities(state, input);
  pinocchio::forwardKinematics(model, data, q, v);

  // Ankle collision points and velocities
  ad_vector3_t pos_ankle_l = pinocchio::updateFramePlacement(model, data, leftAnkleId_).translation();
  ad_vector3_t pos_ankle_r = pinocchio::updateFramePlacement(model, data, rightAnkleId_).translation();
  ad_vector3_t vel_ankle_l = pinocchio::getFrameVelocity(model, data, leftAnkleId_, rf).linear();
  ad_vector3_t vel_ankle_r = pinocchio::getFrameVelocity(model, data, rightAnkleId_, rf).linear();

  // Foot collision points and velocities
  ad_vector3_t pos_f_l = pinocchio::updateFramePlacement(model, data, leftFootCenterId_).translation();
  ad_vector3_t pos_f_r = pinocchio::updateFramePlacement(model, data, rightFootCenterId_).translation();
  ad_vector3_t vel_f_l = pinocchio::getFrameVelocity(model, data, leftFootCenterId_, rf).linear();
  ad_vector3_t vel_f_r = pinocchio::getFrameVelocity(model, data, rightFootCenterId_, rf).linear();

  ad_vector3_t pos_f_l_p1 = pinocchio::updateFramePlacement(model, data, leftFoot1Id_).translation();
  ad_vector3_t pos_f_r_p1 = pinocchio::updateFramePlacement(model, data, rightFoot1Id_).translation();
  ad_vector3_t vel_f_l_p1 = pinocchio::getFrameVelocity(model, data, leftFoot1Id_, rf).linear();
  ad_vector3_t vel_f_r_p1 = pinocchio::getFrameVelocity(model, data, rightFoot1Id_, rf).linear();

  ad_vector3_t pos_f_l_p2 = pinocchio::updateFramePlacement(model, data, leftFoot2Id_).translation();
  ad_vector3_t pos_f_r_p2 = pinocchio::updateFramePlacement(model, data, rightFoot2Id_).translation();
  ad_vector3_t vel_f_l_p2 = pinocchio::getFrameVelocity(model, data, leftFoot2Id_, rf).linear();
  ad_vector3_t vel_f_r_p2 = pinocchio::getFrameVelocity(model, data, rightFoot2Id_, rf).linear();

  // Knee collision points and velocities
  ad_vector3_t pos_k_l = pinocchio::updateFramePlacement(model, data, leftKneeId_).translation();
  ad_vector3_t pos_k_r = pinocchio::updateFramePlacement(model, data, rightKneeId_).translation();
  ad_vector3_t vel_k_l = pinocchio::getFrameVelocity(model, data, leftKneeId_, rf).linear();
  ad_vector3_t vel_k_r = pinocchio::getFrameVelocity(model, data, rightKneeId_, rf).linear();

  // Parameters: [footCollisionSphereRadius, kneeCollisionSphereRadius, gamma]
  ad_scalar_t minDistFootSq = 4.0 * parameters[0] * parameters[0];
  ad_scalar_t minDistKneeSq = 4.0 * parameters[1] * parameters[1];
  ad_scalar_t gamma = parameters[2];

  auto computeCbfPair = [&](const ad_vector3_t& pos_A, const ad_vector3_t& vel_A, const ad_vector3_t& pos_B, const ad_vector3_t& vel_B,
                            ad_scalar_t minDistanceSq) -> ad_scalar_t {
    ad_vector3_t diff_p = pos_A - pos_B;
    ad_vector3_t diff_v = vel_A - vel_B;
    ad_scalar_t distSq = diff_p.squaredNorm();
    ad_scalar_t hSq = distSq - minDistanceSq;
    ad_scalar_t hDot = 2.0 * diff_p.dot(diff_v);
    return hDot + gamma * hSq;
  };

  ad_vector_t constraintValues(numConstraints_);
  constraintValues[0] = computeCbfPair(pos_f_l_p1, vel_f_l_p1, pos_f_r_p1, vel_f_r_p1, minDistFootSq);
  constraintValues[1] = computeCbfPair(pos_f_l_p1, vel_f_l_p1, pos_f_r_p2, vel_f_r_p2, minDistFootSq);
  constraintValues[2] = computeCbfPair(pos_f_l_p2, vel_f_l_p2, pos_f_r_p1, vel_f_r_p1, minDistFootSq);
  constraintValues[3] = computeCbfPair(pos_f_l_p2, vel_f_l_p2, pos_f_r_p2, vel_f_r_p2, minDistFootSq);

  constraintValues[4] = computeCbfPair(pos_f_l, vel_f_l, pos_f_r_p1, vel_f_r_p1, minDistFootSq);
  constraintValues[5] = computeCbfPair(pos_f_l, vel_f_l, pos_f_r_p2, vel_f_r_p2, minDistFootSq);
  constraintValues[6] = computeCbfPair(pos_f_r, vel_f_r, pos_f_l_p1, vel_f_l_p1, minDistFootSq);
  constraintValues[7] = computeCbfPair(pos_f_r, vel_f_r, pos_f_l_p2, vel_f_l_p2, minDistFootSq);
  constraintValues[8] = computeCbfPair(pos_f_l, vel_f_l, pos_f_r, vel_f_r, minDistFootSq);

  constraintValues[9] = computeCbfPair(pos_k_l, vel_k_l, pos_k_r, vel_k_r, minDistKneeSq);

  constraintValues[10] = computeCbfPair(pos_f_l, vel_f_l, pos_ankle_r, vel_ankle_r, minDistFootSq);
  constraintValues[11] = computeCbfPair(pos_f_l_p1, vel_f_l_p1, pos_ankle_r, vel_ankle_r, minDistFootSq);
  constraintValues[12] = computeCbfPair(pos_f_l_p2, vel_f_l_p2, pos_ankle_r, vel_ankle_r, minDistFootSq);
  constraintValues[13] = computeCbfPair(pos_f_r, vel_f_r, pos_ankle_l, vel_ankle_l, minDistFootSq);
  constraintValues[14] = computeCbfPair(pos_f_r_p1, vel_f_r_p1, pos_ankle_l, vel_ankle_l, minDistFootSq);
  constraintValues[15] = computeCbfPair(pos_f_r_p2, vel_f_r_p2, pos_ankle_l, vel_ankle_l, minDistFootSq);

  return constraintValues;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

FootCollisionCbfConstraint::Config FootCollisionCbfConstraint::loadFootCollisionCbfConstraintConfig(absl::string_view taskFile,
                                                                                                    bool verbose) {
  loadData::PropertyTree pt;
  loadData::readPropertyTree(std::string(taskFile), pt);
  constexpr absl::string_view prefix = "collision_cbf_constraint.";

  Config collisionConfig;

  loadData::loadPtreeValue(pt, collisionConfig.gamma, absl::StrCat(prefix, "gamma"), verbose);
  loadData::loadPtreeValue(pt, collisionConfig.leftAnkleFrame, absl::StrCat(prefix, "foot.leftAnkleFrame"), verbose);
  loadData::loadPtreeValue(pt, collisionConfig.rightAnkleFrame, absl::StrCat(prefix, "foot.rightAnkleFrame"), verbose);
  loadData::loadPtreeValue(pt, collisionConfig.footCollisionSphereRadius, absl::StrCat(prefix, "foot.footCollisionSphereRadius"), verbose);
  loadData::loadPtreeValue(pt, collisionConfig.leftKneeFrame, absl::StrCat(prefix, "knee.leftKneeFrame"), verbose);
  loadData::loadPtreeValue(pt, collisionConfig.rightKneeFrame, absl::StrCat(prefix, "knee.rightKneeFrame"), verbose);
  loadData::loadPtreeValue(pt, collisionConfig.kneeCollisionSphereRadius, absl::StrCat(prefix, "knee.kneeCollisionSphereRadius"), verbose);

  return collisionConfig;
}

}  // namespace ocs2::humanoid
