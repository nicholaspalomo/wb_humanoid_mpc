/******************************************************************************
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.
Copyright (c) 2024, 1X Technologies. All rights reserved.

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

#include "pinocchio/fwd.hpp"

#include "humanoid_common_mpc/constraint/FootCollisionConstraint.h"

#include <string>

#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/model.hpp"

namespace ocs2::humanoid {

namespace {

/**
 * The rows of the constraint, in the order the generated library was taped in: the distances between the two feet's
 * collision points (Point1, Point2) and centers, between the knees, and between a foot and the other leg's ankle.
 */
constexpr int kLeftPoint1RightPoint1Row = 0;
constexpr int kLeftPoint1RightPoint2Row = 1;
constexpr int kLeftPoint2RightPoint1Row = 2;
constexpr int kLeftPoint2RightPoint2Row = 3;
constexpr int kLeftCenterRightPoint1Row = 4;
constexpr int kLeftCenterRightPoint2Row = 5;
constexpr int kRightCenterLeftPoint1Row = 6;
constexpr int kRightCenterLeftPoint2Row = 7;
constexpr int kLeftCenterRightCenterRow = 8;
constexpr int kKneesRow = 9;
constexpr int kLeftCenterRightAnkleRow = 10;
constexpr int kLeftPoint1RightAnkleRow = 11;
constexpr int kLeftPoint2RightAnkleRow = 12;
constexpr int kRightCenterLeftAnkleRow = 13;
constexpr int kRightPoint1LeftAnkleRow = 14;
constexpr int kRightPoint2LeftAnkleRow = 15;

}  // namespace

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

FootCollisionConstraint::FootCollisionConstraint(const SwitchedModelReferenceManager& referenceManager,
                                                 const PinocchioInterface& pinocchioInterface,
                                                 const MpcRobotModelBase<ad_scalar_t>& mpcRobotModel,
                                                 const Config& config,
                                                 const std::string& costName,
                                                 const ModelSettings& modelSettings)
    : StateConstraintCppAd(ConstraintOrder::Linear),
      referenceManagerPtr_(&referenceManager),
      pinocchioInterfaceCppAd_(pinocchioInterface.toCppAd()),
      mpcRobotModelPtr_(&mpcRobotModel),
      cfg_(config) {
  initialize(mpcRobotModelPtr_->getStateDim(), kNumParameters, costName, modelSettings.modelFolderCppAd,
             modelSettings.recompileLibrariesCppAd, modelSettings.verboseCppAd);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

// isActive_ is copied deliberately: the SQP solver clones the whole problem once per worker thread,
// and a copy constructor that dropped this flag silently reverted a deactivated term to active.
FootCollisionConstraint::FootCollisionConstraint(const FootCollisionConstraint& other) = default;

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
bool FootCollisionConstraint::isActive(scalar_t time) const {
  if (!isActive_) return false;

  // Inactivate the constraint if both feet are in contact. Prevents it from fighting against the stance foot constraints.
  contact_flag_t contactFlags = referenceManagerPtr_->getContactFlags(time);
  return !(contactFlags[0] && contactFlags[1]);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

ad_vector_t FootCollisionConstraint::constraintFunction(ad_scalar_t /*time*/,
                                                        const ad_vector_t& state,
                                                        const ad_vector_t& parameters) const {
  static_assert(static_cast<size_t>(kRightPoint2LeftAnkleRow) + 1 == kNumConstraints, "one row per distance kept apart");
  const PinocchioInterfaceCppAd::Model& model = pinocchioInterfaceCppAd_.getModel();
  PinocchioInterfaceCppAd::Data data = pinocchioInterfaceCppAd_.getData();

  const ad_vector_t q = mpcRobotModelPtr_->getGeneralizedCoordinates(state);
  pinocchio::forwardKinematics(model, data, q);

  // Ankle collision points
  ad_vector3_t pos_ankle_l = pinocchio::updateFramePlacement(model, data, model.getFrameId(cfg_.leftAnkleFrame)).translation();
  ad_vector3_t pos_ankle_r = pinocchio::updateFramePlacement(model, data, model.getFrameId(cfg_.rightAnkleFrame)).translation();

  // Foot collision points
  ad_vector3_t pos_f_l = pinocchio::updateFramePlacement(model, data, model.getFrameId(cfg_.leftFootCenterFrame)).translation();
  ad_vector3_t pos_f_r = pinocchio::updateFramePlacement(model, data, model.getFrameId(cfg_.rightFootCenterFrame)).translation();
  ad_vector3_t pos_f_l_p1 = pinocchio::updateFramePlacement(model, data, model.getFrameId(cfg_.leftFootFrame1)).translation();
  ad_vector3_t pos_f_r_p1 = pinocchio::updateFramePlacement(model, data, model.getFrameId(cfg_.rightFootFrame1)).translation();
  ad_vector3_t pos_f_l_p2 = pinocchio::updateFramePlacement(model, data, model.getFrameId(cfg_.leftFootFrame2)).translation();
  ad_vector3_t pos_f_r_p2 = pinocchio::updateFramePlacement(model, data, model.getFrameId(cfg_.rightFootFrame2)).translation();

  // Knee collision points
  ad_vector3_t pos_k_l = pinocchio::updateFramePlacement(model, data, model.getFrameId(cfg_.leftKneeFrame)).translation();
  ad_vector3_t pos_k_r = pinocchio::updateFramePlacement(model, data, model.getFrameId(cfg_.rightKneeFrame)).translation();

  // The minimum distance between two collision spheres, 2 * radius: each row is a distance minus it, not a squared
  // distance. parameters[0] is the radius of the foot spheres, parameters[1] that of the knee spheres.
  ad_scalar_t minDistFoot = 2.0 * parameters[0];
  ad_scalar_t minDistKnee = 2.0 * parameters[1];

  ad_vector_t constraintValues(numConstraints_);
  constraintValues[kLeftPoint1RightPoint1Row] = ((pos_f_l_p1 - pos_f_r_p1).norm() - minDistFoot);
  constraintValues[kLeftPoint1RightPoint2Row] = ((pos_f_l_p1 - pos_f_r_p2).norm() - minDistFoot);
  constraintValues[kLeftPoint2RightPoint1Row] = ((pos_f_l_p2 - pos_f_r_p1).norm() - minDistFoot);
  constraintValues[kLeftPoint2RightPoint2Row] = ((pos_f_l_p2 - pos_f_r_p2).norm() - minDistFoot);

  constraintValues[kLeftCenterRightPoint1Row] = ((pos_f_l - pos_f_r_p1).norm() - minDistFoot);
  constraintValues[kLeftCenterRightPoint2Row] = ((pos_f_l - pos_f_r_p2).norm() - minDistFoot);
  constraintValues[kRightCenterLeftPoint1Row] = ((pos_f_r - pos_f_l_p1).norm() - minDistFoot);
  constraintValues[kRightCenterLeftPoint2Row] = ((pos_f_r - pos_f_l_p2).norm() - minDistFoot);
  constraintValues[kLeftCenterRightCenterRow] = ((pos_f_l - pos_f_r).norm() - minDistFoot);

  constraintValues[kKneesRow] = ((pos_k_l - pos_k_r).norm() - minDistKnee);

  constraintValues[kLeftCenterRightAnkleRow] = ((pos_f_l - pos_ankle_r).norm() - minDistFoot);
  constraintValues[kLeftPoint1RightAnkleRow] = ((pos_f_l_p1 - pos_ankle_r).norm() - minDistFoot);
  constraintValues[kLeftPoint2RightAnkleRow] = ((pos_f_l_p2 - pos_ankle_r).norm() - minDistFoot);
  constraintValues[kRightCenterLeftAnkleRow] = ((pos_f_r - pos_ankle_l).norm() - minDistFoot);
  constraintValues[kRightPoint1LeftAnkleRow] = ((pos_f_r_p1 - pos_ankle_l).norm() - minDistFoot);
  constraintValues[kRightPoint2LeftAnkleRow] = ((pos_f_r_p2 - pos_ankle_l).norm() - minDistFoot);

  return constraintValues;
}

}  // namespace ocs2::humanoid
