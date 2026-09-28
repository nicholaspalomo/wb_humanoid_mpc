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

#include <pinocchio/fwd.hpp>

#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"

// Pinnochio
#include <pinocchio/algorithm/contact-dynamics.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

#include <functional>
#include <string>

#include <humanoid_common_mpc/gait/MotionPhaseDefinition.h>

namespace ocs2::humanoid {

template <typename SCALAR_T>
void updateFramePlacements(const VECTOR_T<SCALAR_T>& q, PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface) {
  const typename PinocchioInterfaceTpl<SCALAR_T>::Model& model = pinocchioInterface.getModel();
  typename PinocchioInterfaceTpl<SCALAR_T>::Data& data = pinocchioInterface.getData();
  updateFramePlacements(q, model, data);
}
template void updateFramePlacements(const ad_vector_t& q, PinocchioInterfaceTpl<ad_scalar_t>& pinocchioInterface);
template void updateFramePlacements(const vector_t& q, PinocchioInterfaceTpl<scalar_t>& pinocchioInterface);

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

template <typename SCALAR_T>
void updateFramePlacements(const VECTOR_T<SCALAR_T>& q, const pinocchio::ModelTpl<SCALAR_T>& model, pinocchio::DataTpl<SCALAR_T>& data) {
  pinocchio::forwardKinematics(model, data, q);
  updateFramePlacements(model, data);
}
template void updateFramePlacements(const ad_vector_t& q,
                                    const pinocchio::ModelTpl<ad_scalar_t>& model,
                                    pinocchio::DataTpl<ad_scalar_t>& data);
template void updateFramePlacements(const vector_t& q, const pinocchio::ModelTpl<scalar_t>& model, pinocchio::DataTpl<scalar_t>& data);

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

template <typename SCALAR_T>
std::vector<VECTOR3_T<SCALAR_T>> computeContactPositions(const VECTOR_T<SCALAR_T>& q,
                                                         PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface,
                                                         const MpcRobotModelBase<SCALAR_T>& mpcRobotModel) {
  updateFramePlacements<SCALAR_T>(q, pinocchioInterface);
  return getContactPositions<SCALAR_T>(pinocchioInterface, mpcRobotModel);
}
template std::vector<VECTOR3_T<ad_scalar_t>> computeContactPositions(const VECTOR_T<ad_scalar_t>& q,
                                                                     PinocchioInterfaceTpl<ad_scalar_t>& pinocchioInterface,
                                                                     const MpcRobotModelBase<ad_scalar_t>& mpcRobotModel);
template std::vector<VECTOR3_T<scalar_t>> computeContactPositions(const VECTOR_T<scalar_t>& q,
                                                                  PinocchioInterfaceTpl<scalar_t>& pinocchioInterface,
                                                                  const MpcRobotModelBase<scalar_t>& mpcRobotModel);

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

template <typename SCALAR_T>
std::vector<VECTOR3_T<SCALAR_T>> getContactPositions(const PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface,
                                                     const MpcRobotModelBase<SCALAR_T>& mpcRobotModel) {
  assert(mpcRobotModel.modelSettings.contactNames.size() == N_CONTACTS);
  std::vector<VECTOR3_T<SCALAR_T>> footPositions;
  footPositions.reserve(N_CONTACTS);
  const typename PinocchioInterfaceTpl<SCALAR_T>::Data& data = pinocchioInterface.getData();
  std::vector<pinocchio::FrameIndex> contactFrameIndices = getContactFrameIndices(pinocchioInterface, mpcRobotModel);

  for (size_t i = 0; i < N_CONTACTS; i++) {
    const VECTOR3_T<SCALAR_T>& footPosition = data.oMf[getContactFrameIndex(pinocchioInterface, mpcRobotModel, i)].translation();
    footPositions.emplace_back(footPosition);
  }
  return footPositions;
}
template std::vector<VECTOR3_T<ad_scalar_t>> getContactPositions(const PinocchioInterfaceTpl<ad_scalar_t>& pinocchioInterface,
                                                                 const MpcRobotModelBase<ad_scalar_t>& mpcRobotModel);
template std::vector<VECTOR3_T<scalar_t>> getContactPositions(const PinocchioInterfaceTpl<scalar_t>& pinocchioInterface,
                                                              const MpcRobotModelBase<scalar_t>& mpcRobotModel);

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

template <typename SCALAR_T>
std::vector<VECTOR3_T<SCALAR_T>> computeFramePositions(const VECTOR_T<SCALAR_T>& q,
                                                       PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface,
                                                       std::vector<std::string> frameNames) {
  updateFramePlacements<SCALAR_T>(q, pinocchioInterface);
  return getFramePositions<SCALAR_T>(pinocchioInterface, frameNames);
}
template std::vector<VECTOR3_T<ad_scalar_t>> computeFramePositions(const VECTOR_T<ad_scalar_t>& q,
                                                                   PinocchioInterfaceTpl<ad_scalar_t>& pinocchioInterface,
                                                                   std::vector<std::string> frameNames);
template std::vector<VECTOR3_T<scalar_t>> computeFramePositions(const VECTOR_T<scalar_t>& q,
                                                                PinocchioInterfaceTpl<scalar_t>& pinocchioInterface,
                                                                std::vector<std::string> frameNames);

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

template <typename SCALAR_T>
std::vector<VECTOR3_T<SCALAR_T>> getFramePositions(const PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface,
                                                   std::vector<std::string> frameNames) {
  std::vector<VECTOR3_T<SCALAR_T>> positions;
  positions.reserve(frameNames.size());
  const typename PinocchioInterfaceTpl<SCALAR_T>::Model& model = pinocchioInterface.getModel();
  const typename PinocchioInterfaceTpl<SCALAR_T>::Data& data = pinocchioInterface.getData();
  for (size_t i = 0; i < frameNames.size(); i++) {
    if (frameNames[i].empty() || !model.existFrame(frameNames[i])) {
      positions.emplace_back(VECTOR3_T<SCALAR_T>::Zero());
      continue;
    }
    const pinocchio::FrameIndex frameIndex = model.getFrameId(frameNames[i]);
    const VECTOR3_T<SCALAR_T>& position = data.oMf[frameIndex].translation();
    positions.emplace_back(position);
  }
  return positions;
}
template std::vector<VECTOR3_T<ad_scalar_t>> getFramePositions(const PinocchioInterfaceTpl<ad_scalar_t>& pinocchioInterface,
                                                               std::vector<std::string> frameNames);
template std::vector<VECTOR3_T<scalar_t>> getFramePositions(const PinocchioInterfaceTpl<scalar_t>& pinocchioInterface,
                                                            std::vector<std::string> frameNames);

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

scalar_t computeComHeightAboveFeet(const vector_t& q,
                                   PinocchioInterface& pinocchioInterface,
                                   const MpcRobotModelBase<scalar_t>& mpcRobotModel) {
  pinocchio::centerOfMass(pinocchioInterface.getModel(), pinocchioInterface.getData(), q, /*computeSubtreeComs=*/false);
  const scalar_t comHeight = pinocchioInterface.getData().com[0](2);
  const std::vector<vector3_t> feet = computeContactPositions<scalar_t>(q, pinocchioInterface, mpcRobotModel);
  scalar_t meanFootHeight = 0.0;
  for (const vector3_t& foot : feet) {
    meanFootHeight += foot(2) / static_cast<scalar_t>(feet.size());
  }
  return comHeight - meanFootHeight;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

scalar_t computeGroundHeightEstimate(PinocchioInterfaceTpl<scalar_t>& pinocchioInterface,
                                     const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                                     const vector_t& q,
                                     size_t measuredMode) {
  updateFramePlacements<scalar_t>(q, pinocchioInterface);
  return getGroundHeightEstimate(pinocchioInterface, mpcRobotModel, measuredMode);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

scalar_t getGroundHeightEstimate(PinocchioInterfaceTpl<scalar_t>& pinocchioInterface,
                                 const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                                 size_t measuredMode) {
  contact_flag_t measuredContactFlags = modeNumber2StanceLeg(measuredMode);

  std::vector<vector3_t> contactPositions = getContactPositions<scalar_t>(pinocchioInterface, mpcRobotModel);

  static scalar_t terrainHeight = 0.0;

  // Use right foot if in contact
  if (measuredContactFlags[0] && measuredContactFlags[1]) {
    vector3_t footPosition1 = contactPositions[0];
    vector3_t footPosition2 = contactPositions[1];
    terrainHeight = 0.5 * (footPosition1[2] + footPosition2[2]);
  } else if (measuredContactFlags[0]) {
    vector3_t footPosition = contactPositions[0];
    terrainHeight = footPosition[2];
  } else if (measuredContactFlags[1]) {
    vector3_t footPosition = contactPositions[1];
    terrainHeight = footPosition[2];
  }
  return terrainHeight;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

template <typename SCALAR_T>
VECTOR6_T<SCALAR_T> computeBaseAcceleration(const MATRIX_T<SCALAR_T>& M,
                                            const VECTOR_T<SCALAR_T>& nle,
                                            const VECTOR_T<SCALAR_T>& qdd_joints,
                                            const VECTOR_T<SCALAR_T>& externalForcesInJointSpace) {
  // The base rows of M qdd + nle = J^T F, solved for the base acceleration: M_bb a_b = r with
  // r = -nle_b - M_bj qdd_j + (J^T F)_b.
  const MATRIX_T<SCALAR_T> M_bj = M.block(0, 6, 6, qdd_joints.size());
  const VECTOR6_T<SCALAR_T> r = -nle.head(6) - M_bj * qdd_joints + externalForcesInJointSpace.head(6);

  // M_bb = [A B; C D] is NOT block diagonal: the translational and rotational coordinates of the base couple through the
  // offset of the whole-body center of mass from the base, so inverting A and D separately is wrong. M_bb is solved
  // exactly through the Schur complement S = D - C A^-1 B, with 3x3 inverses only. That is deliberate: Eigen inverts a
  // fixed-size 3x3 matrix by cofactors, with no comparison, whereas a 6x6 inverse goes through a pivoting LU that compares
  // entries of the matrix. A comparison between CppAD variables is recorded on the tape, and CppADCodeGen's code
  // generator cannot evaluate it: the whole-body dynamics then fail to compile with "GreaterThanZero cannot be called for
  // non-parameters". A is the translational block, the total mass times the identity, and S is the rotational inertia
  // about the center of mass in the base's Euler-angle rates, so both are invertible wherever M_bb is. crba() fills
  // both triangles of this 6x6 root block (it writes the root joint's rows over its whole subtree), so C is read, not
  // assumed to be B^T.
  const MATRIX3_T<SCALAR_T> A = M.template block<3, 3>(0, 0);
  const MATRIX3_T<SCALAR_T> B = M.template block<3, 3>(0, 3);
  const MATRIX3_T<SCALAR_T> C = M.template block<3, 3>(3, 0);
  const MATRIX3_T<SCALAR_T> D = M.template block<3, 3>(3, 3);
  const MATRIX3_T<SCALAR_T> AInverse = A.inverse();
  const MATRIX3_T<SCALAR_T> schurComplementInverse = (D - C * AInverse * B).inverse();

  const VECTOR3_T<SCALAR_T> r1 = r.template head<3>();
  const VECTOR3_T<SCALAR_T> r2 = r.template tail<3>();
  const VECTOR3_T<SCALAR_T> angular = schurComplementInverse * (r2 - C * (AInverse * r1));
  const VECTOR3_T<SCALAR_T> linear = AInverse * (r1 - B * angular);

  VECTOR6_T<SCALAR_T> baseAcceleration;
  baseAcceleration << linear, angular;
  return baseAcceleration;
}
template VECTOR6_T<scalar_t> computeBaseAcceleration(const MATRIX_T<scalar_t>& M,
                                                     const VECTOR_T<scalar_t>& nle,
                                                     const VECTOR_T<scalar_t>& qdd_joints,
                                                     const VECTOR_T<scalar_t>& externalForcesInJointSpace);
template VECTOR6_T<ad_scalar_t> computeBaseAcceleration(const MATRIX_T<ad_scalar_t>& M,
                                                        const VECTOR_T<ad_scalar_t>& nle,
                                                        const VECTOR_T<ad_scalar_t>& qdd_joints,
                                                        const VECTOR_T<ad_scalar_t>& externalForcesInJointSpace);

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

namespace {

///
/// Writes the full joint-space inertia matrix M(q) into data.M and the nonlinear effects nle(q, qd) into data.nle, and
/// returns the generalized contact force J_l^T W_l + J_r^T W_r: the terms of M qdd + nle = S^T tau + J^T W that the
/// inverse dynamics below read.
///
/// pinocchio::crba() writes, for every joint, its rows over the columns of its own subtree: the UPPER TRIANGLE of M, and
/// only the part of it where one joint supports the other (the full 6x6 block of the composite base joint included).
/// Every other entry - the lower triangle, and the coupling of two joints on different branches, such as the two legs -
/// it leaves at whatever the data held, zero for a fresh one but not for a data another algorithm has written. The joint
/// rows of M qdd read both: the joint-base coupling M_jb and the lower triangle of M_jj. So M is cleared first, and its
/// strictly lower triangle mirrored from the upper one after, which is exact: M is symmetric. Neither records an
/// operation on a CppAD tape, only copies.
///
template <typename SCALAR_T>
VECTOR_T<SCALAR_T> updateInverseDynamicsTerms(const VECTOR_T<SCALAR_T>& q,
                                              const VECTOR_T<SCALAR_T>& qd,
                                              const std::array<VECTOR6_T<SCALAR_T>, 2>& footWrenches,
                                              const pinocchio::ModelTpl<SCALAR_T>& model,
                                              pinocchio::DataTpl<SCALAR_T>& data) {
  data.M.setZero();
  pinocchio::crba(model, data, q);
  data.M.template triangularView<Eigen::StrictlyLower>() = data.M.transpose().template triangularView<Eigen::StrictlyLower>();
  pinocchio::nonLinearEffects(model, data, q, qd);

  // LOCAL_WORLD_ALIGNED Jacobians of the sole frames, which take the world-frame wrenches of the MPC.
  MATRIX_T<SCALAR_T> J_foot_l = MATRIX_T<SCALAR_T>::Zero(6, qd.size());
  MATRIX_T<SCALAR_T> J_foot_r = MATRIX_T<SCALAR_T>::Zero(6, qd.size());
  pinocchio::computeFrameJacobian(model, data, q, model.getFrameId("foot_l_contact"), pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
                                  J_foot_l);
  pinocchio::computeFrameJacobian(model, data, q, model.getFrameId("foot_r_contact"), pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
                                  J_foot_r);
  return J_foot_l.transpose() * footWrenches[0] + J_foot_r.transpose() * footWrenches[1];
}

}  // namespace

template <typename SCALAR_T>
VECTOR_T<SCALAR_T> computeJointTorques(const VECTOR_T<SCALAR_T>& q,
                                       const VECTOR_T<SCALAR_T>& qd,
                                       const VECTOR_T<SCALAR_T>& qdd_joints,
                                       const std::array<VECTOR6_T<SCALAR_T>, 2>& footWrenches,
                                       PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface) {
  const typename PinocchioInterfaceTpl<SCALAR_T>::Model& model = pinocchioInterface.getModel();
  pinocchio::DataTpl<SCALAR_T>& data = pinocchioInterface.getData();
  const VECTOR_T<SCALAR_T> externalForcesInJointSpace = updateInverseDynamicsTerms<SCALAR_T>(q, qd, footWrenches, model, data);

  // The base is unactuated: its rows fix the base acceleration the wrenches produce, and the joint rows then carry it.
  const VECTOR6_T<SCALAR_T> baseAcceleration = computeBaseAcceleration<SCALAR_T>(data.M, data.nle, qdd_joints, externalForcesInJointSpace);
  VECTOR_T<SCALAR_T> q_dd(qd.size());
  q_dd << baseAcceleration, qdd_joints;

  const Eigen::Index numJoints = qdd_joints.size();
  return data.M.bottomRows(numJoints) * q_dd + data.nle.tail(numJoints) - externalForcesInJointSpace.tail(numJoints);
}
template VECTOR_T<scalar_t> computeJointTorques(const VECTOR_T<scalar_t>& q,
                                                const VECTOR_T<scalar_t>& qd,
                                                const VECTOR_T<scalar_t>& qdd_joints,
                                                const std::array<VECTOR6_T<scalar_t>, 2>& footWrenches,
                                                PinocchioInterfaceTpl<scalar_t>& pinocchioInterface);
template VECTOR_T<ad_scalar_t> computeJointTorques(const VECTOR_T<ad_scalar_t>& q,
                                                   const VECTOR_T<ad_scalar_t>& qd,
                                                   const VECTOR_T<ad_scalar_t>& qdd_joints,
                                                   const std::array<VECTOR6_T<ad_scalar_t>, 2>& footWrenches,
                                                   PinocchioInterfaceTpl<ad_scalar_t>& pinocchioInterface);

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

template <typename SCALAR_T>
VECTOR_T<SCALAR_T> computeBaseHeldJointTorques(const VECTOR_T<SCALAR_T>& q,
                                               const VECTOR_T<SCALAR_T>& qd,
                                               const VECTOR_T<SCALAR_T>& qdd_joints,
                                               const std::array<VECTOR6_T<SCALAR_T>, 2>& footWrenches,
                                               PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface) {
  const typename PinocchioInterfaceTpl<SCALAR_T>::Model& model = pinocchioInterface.getModel();
  pinocchio::DataTpl<SCALAR_T>& data = pinocchioInterface.getData();
  const VECTOR_T<SCALAR_T> externalForcesInJointSpace = updateInverseDynamicsTerms<SCALAR_T>(q, qd, footWrenches, model, data);

  // Whatever holds the base (the gantry) supplies the base rows; the base acceleration is zero.
  const Eigen::Index numJoints = qdd_joints.size();
  return data.M.bottomRightCorner(numJoints, numJoints) * qdd_joints + data.nle.tail(numJoints) -
         externalForcesInJointSpace.tail(numJoints);
}
template VECTOR_T<scalar_t> computeBaseHeldJointTorques(const VECTOR_T<scalar_t>& q,
                                                        const VECTOR_T<scalar_t>& qd,
                                                        const VECTOR_T<scalar_t>& qdd_joints,
                                                        const std::array<VECTOR6_T<scalar_t>, 2>& footWrenches,
                                                        PinocchioInterfaceTpl<scalar_t>& pinocchioInterface);

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

template <typename SCALAR_T>
VECTOR_T<SCALAR_T> computeJointTorquesRNEA(const VECTOR_T<SCALAR_T>& q,
                                           const VECTOR_T<SCALAR_T>& qd,
                                           const VECTOR_T<SCALAR_T>& qdd_joints,
                                           const std::array<VECTOR6_T<SCALAR_T>, 2>& footWrenches,
                                           PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface) {
  const typename PinocchioInterfaceTpl<SCALAR_T>::Model& model = pinocchioInterface.getModel();
  typename PinocchioInterfaceTpl<SCALAR_T>::Data& data = pinocchioInterface.getData();

  pinocchio::container::aligned_vector<pinocchio::Force> fextDesired(model.njoints, pinocchio::Force::Zero());

  pinocchio::forwardKinematics(model, data, q, qd);
  pinocchio::updateFramePlacements(model, data);

  const std::function<void(const std::string&, size_t)> setExternalForce = [&](const std::string& frameName, size_t i) {
    const pinocchio::FrameIndex frameIndex = model.getFrameId(frameName);
    const pinocchio::JointIndex jointIndex = model.frames[frameIndex].parentJoint;
    const VECTOR3_T<SCALAR_T> translationJointFrameToContactFrame = model.frames[frameIndex].placement.translation();
    const MATRIX3_T<SCALAR_T> rotationWorldFrameToJointFrame = data.oMi[jointIndex].rotation().transpose();
    const VECTOR3_T<SCALAR_T> contactForce = rotationWorldFrameToJointFrame * footWrenches[i].head(3);
    const VECTOR3_T<SCALAR_T> contactTorque = rotationWorldFrameToJointFrame * footWrenches[i].tail(3);
    fextDesired[jointIndex].linear() = contactForce;
    fextDesired[jointIndex].angular() = translationJointFrameToContactFrame.cross(contactForce) + contactTorque;
  };

  setExternalForce("foot_l_contact", /*i=*/0);
  setExternalForce("foot_r_contact", /*i=*/1);

  pinocchio::crba(model, data, q);
  pinocchio::nonLinearEffects(model, data, q, qd);

  // Compute Jacobians for the foot frames
  MATRIX_T<SCALAR_T> J_foot_l = MATRIX_T<SCALAR_T>::Zero(6, qd.size());
  MATRIX_T<SCALAR_T> J_foot_r = MATRIX_T<SCALAR_T>::Zero(6, qd.size());

  ////////////////////////////////////////////////////////////////////////////

  pinocchio::computeFrameJacobian(model, data, q, model.getFrameId("foot_l_contact"), pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
                                  J_foot_l);
  pinocchio::computeFrameJacobian(model, data, q, model.getFrameId("foot_r_contact"), pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
                                  J_foot_r);

  // Project contact wrenches into the joint space

  VECTOR_T<SCALAR_T> externalForcesInJointSpace = J_foot_l.transpose() * footWrenches[0] + J_foot_r.transpose() * footWrenches[1];

  // Repalce q with external forces in joint space.

  VECTOR6_T<SCALAR_T> baseAccelerations = computeBaseAcceleration(data.M, data.nle, qdd_joints, externalForcesInJointSpace);

  VECTOR_T<SCALAR_T> q_dd(qd.size());
  q_dd << baseAccelerations, qdd_joints;

  vector_t torques = pinocchio::rnea(model, data, q, qd, q_dd, fextDesired);

  return torques.tail(qdd_joints.size());
}
template VECTOR_T<scalar_t> computeJointTorquesRNEA(const VECTOR_T<scalar_t>& q,
                                                    const VECTOR_T<scalar_t>& qd,
                                                    const VECTOR_T<scalar_t>& qdd_joints,
                                                    const std::array<VECTOR6_T<scalar_t>, 2>& footWrenches,
                                                    PinocchioInterfaceTpl<scalar_t>& pinocchioInterface);
// template VECTOR_T<ad_scalar_t> computeJointTorquesRNEA(const VECTOR_T<ad_scalar_t>& q,
//                                                        const VECTOR_T<ad_scalar_t>& qd,
//                                                        const VECTOR_T<ad_scalar_t>& qdd_joints,
//                                                        const std::array<VECTOR6_T<ad_scalar_t>, 2>& footWrenches,
//                                                        PinocchioInterfaceTpl<ad_scalar_t>& pinocchioInterface);

}  // namespace ocs2::humanoid
