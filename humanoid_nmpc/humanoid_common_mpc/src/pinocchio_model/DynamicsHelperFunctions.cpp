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

#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"

// Pinnochio
#include <algorithm>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/Types.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/multibody.hpp"

#include "humanoid_common_mpc/common/ContactArray.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid {

template <typename SCALAR_T>
void updateFramePlacements(
    const VECTOR_T<SCALAR_T>& q,
    PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface) {
  const typename PinocchioInterfaceTpl<SCALAR_T>::Model& model =
      pinocchioInterface.getModel();
  typename PinocchioInterfaceTpl<SCALAR_T>::Data& data =
      pinocchioInterface.getData();
  updateFramePlacements(q, model, data);
}
template void updateFramePlacements(
    const ad_vector_t& q,
    PinocchioInterfaceTpl<ad_scalar_t>& pinocchioInterface);
template void updateFramePlacements(
    const vector_t& q, PinocchioInterfaceTpl<scalar_t>& pinocchioInterface);

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

template <typename SCALAR_T>
void updateFramePlacements(const VECTOR_T<SCALAR_T>& q,
                           const pinocchio::ModelTpl<SCALAR_T>& model,
                           pinocchio::DataTpl<SCALAR_T>& data) {
  pinocchio::forwardKinematics(model, data, q);
  updateFramePlacements(model, data);
}
template void updateFramePlacements(
    const ad_vector_t& q,
    const pinocchio::ModelTpl<ad_scalar_t>& model,
    pinocchio::DataTpl<ad_scalar_t>& data);
template void updateFramePlacements(const vector_t& q,
                                    const pinocchio::ModelTpl<scalar_t>& model,
                                    pinocchio::DataTpl<scalar_t>& data);

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

template <typename SCALAR_T>
std::vector<VECTOR3_T<SCALAR_T>> computePosContactsRtWorldEwrtWorld(
    const VECTOR_T<SCALAR_T>& q,
    PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface,
    const MpcRobotModelBase<SCALAR_T>& mpcRobotModel) {
  updateFramePlacements<SCALAR_T>(q, pinocchioInterface);
  return getPosContactsRtWorldEwrtWorld<SCALAR_T>(pinocchioInterface,
                                                  mpcRobotModel);
}
template std::vector<VECTOR3_T<ad_scalar_t>> computePosContactsRtWorldEwrtWorld(
    const VECTOR_T<ad_scalar_t>& q,
    PinocchioInterfaceTpl<ad_scalar_t>& pinocchioInterface,
    const MpcRobotModelBase<ad_scalar_t>& mpcRobotModel);
template std::vector<VECTOR3_T<scalar_t>> computePosContactsRtWorldEwrtWorld(
    const VECTOR_T<scalar_t>& q,
    PinocchioInterfaceTpl<scalar_t>& pinocchioInterface,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel);

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

template <typename SCALAR_T>
std::vector<VECTOR3_T<SCALAR_T>> getPosContactsRtWorldEwrtWorld(
    const PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface,
    const MpcRobotModelBase<SCALAR_T>& mpcRobotModel) {
  // Every contact of the set in contact order (3-DoF contacts first): the
  // feet of a biped, the four feet of a quadruped, the knees, hands and feet of
  // a multi-contact set.
  const size_t numContacts = mpcRobotModel.modelSettings.numContacts();
  std::vector<VECTOR3_T<SCALAR_T>> posContactsRtWorldEwrtWorld;
  posContactsRtWorldEwrtWorld.reserve(numContacts);
  const typename PinocchioInterfaceTpl<SCALAR_T>::Data& data =
      pinocchioInterface.getData();

  for (size_t i = 0; i < numContacts; ++i) {
    const VECTOR3_T<SCALAR_T>& posContactRtWorldEwrtWorld =
        data.oMf[getContactFrameIndex(pinocchioInterface, mpcRobotModel, i)]
            .translation();
    posContactsRtWorldEwrtWorld.emplace_back(posContactRtWorldEwrtWorld);
  }
  return posContactsRtWorldEwrtWorld;
}
template std::vector<VECTOR3_T<ad_scalar_t>> getPosContactsRtWorldEwrtWorld(
    const PinocchioInterfaceTpl<ad_scalar_t>& pinocchioInterface,
    const MpcRobotModelBase<ad_scalar_t>& mpcRobotModel);
template std::vector<VECTOR3_T<scalar_t>> getPosContactsRtWorldEwrtWorld(
    const PinocchioInterfaceTpl<scalar_t>& pinocchioInterface,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel);

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

template <typename SCALAR_T>
std::vector<VECTOR3_T<SCALAR_T>> computePosLinksRtWorldEwrtWorld(
    const VECTOR_T<SCALAR_T>& q,
    PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface,
    absl::Span<const std::string> frameNames) {
  updateFramePlacements<SCALAR_T>(q, pinocchioInterface);
  return getPosLinksRtWorldEwrtWorld<SCALAR_T>(pinocchioInterface, frameNames);
}
template std::vector<VECTOR3_T<ad_scalar_t>> computePosLinksRtWorldEwrtWorld(
    const VECTOR_T<ad_scalar_t>& q,
    PinocchioInterfaceTpl<ad_scalar_t>& pinocchioInterface,
    absl::Span<const std::string> frameNames);
template std::vector<VECTOR3_T<scalar_t>> computePosLinksRtWorldEwrtWorld(
    const VECTOR_T<scalar_t>& q,
    PinocchioInterfaceTpl<scalar_t>& pinocchioInterface,
    absl::Span<const std::string> frameNames);

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

template <typename SCALAR_T>
std::vector<VECTOR3_T<SCALAR_T>> getPosLinksRtWorldEwrtWorld(
    const PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface,
    absl::Span<const std::string> frameNames) {
  std::vector<VECTOR3_T<SCALAR_T>> posLinksRtWorldEwrtWorld;
  posLinksRtWorldEwrtWorld.reserve(frameNames.size());
  const typename PinocchioInterfaceTpl<SCALAR_T>::Model& model =
      pinocchioInterface.getModel();
  const typename PinocchioInterfaceTpl<SCALAR_T>::Data& data =
      pinocchioInterface.getData();
  for (size_t i = 0; i < frameNames.size(); ++i) {
    if (frameNames[i].empty() || !model.existFrame(frameNames[i])) {
      posLinksRtWorldEwrtWorld.emplace_back(VECTOR3_T<SCALAR_T>::Zero());
      continue;
    }
    const pinocchio::FrameIndex frameIndex = model.getFrameId(frameNames[i]);
    const VECTOR3_T<SCALAR_T>& posLinkRtWorldEwrtWorld =
        data.oMf[frameIndex].translation();
    posLinksRtWorldEwrtWorld.emplace_back(posLinkRtWorldEwrtWorld);
  }
  return posLinksRtWorldEwrtWorld;
}
template std::vector<VECTOR3_T<ad_scalar_t>> getPosLinksRtWorldEwrtWorld(
    const PinocchioInterfaceTpl<ad_scalar_t>& pinocchioInterface,
    absl::Span<const std::string> frameNames);
template std::vector<VECTOR3_T<scalar_t>> getPosLinksRtWorldEwrtWorld(
    const PinocchioInterfaceTpl<scalar_t>& pinocchioInterface,
    absl::Span<const std::string> frameNames);

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

scalar_t computeProxyPosComRtFeetmeanEwrtWorldZ(
    const vector_t& q,
    PinocchioInterface& pinocchioInterface,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel) {
  const ModelSettings& settings = mpcRobotModel.modelSettings;
  // The biped keeps the mean over its two feet, by the arithmetic it always
  // had.
  if (settings.isBipedContactSet()) {
    return computePosComRtFeetmeanEwrtWorldZ(q, pinocchioInterface,
                                             mpcRobotModel);
  }
  // Another set with legs: the mean over the legs' contacts.
  if (!settings.legs.empty()) {
    std::vector<size_t> footContactIndices;
    for (const ModelSettings::Leg& leg : settings.legs) {
      footContactIndices.insert(footContactIndices.end(),
                                leg.contactIndices.begin(),
                                leg.contactIndices.end());
    }
    return computePosComRtFeetmeanEwrtWorldZ(q, pinocchioInterface,
                                             mpcRobotModel, footContactIndices);
  }
  // A set without legs (a robot sitting or lying): the height above its lowest
  // contact frame stands in for the one above the feet.
  return computePosComRtLowcontactEwrtWorldZ(q, pinocchioInterface,
                                             mpcRobotModel);
}

scalar_t computePosComRtFeetmeanEwrtWorldZ(
    const vector_t& q,
    PinocchioInterface& pinocchioInterface,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel) {
  pinocchio::centerOfMass(pinocchioInterface.getModel(),
                          pinocchioInterface.getData(), q,
                          /*computeSubtreeComs=*/false);
  const scalar_t posComRtWorldEwrtWorldZ =
      pinocchioInterface.getData().com[0](kZAxisIndex);
  const std::vector<vector3_t> posContactsRtWorldEwrtWorld =
      computePosContactsRtWorldEwrtWorld<scalar_t>(q, pinocchioInterface,
                                                   mpcRobotModel);
  scalar_t posFeetmeanRtWorldEwrtWorldZ = 0.0;
  for (const vector3_t& posContactRtWorldEwrtWorld :
       posContactsRtWorldEwrtWorld) {
    posFeetmeanRtWorldEwrtWorldZ +=
        posContactRtWorldEwrtWorld(kZAxisIndex) /
        static_cast<scalar_t>(posContactsRtWorldEwrtWorld.size());
  }
  return posComRtWorldEwrtWorldZ - posFeetmeanRtWorldEwrtWorldZ;
}

scalar_t computePosComRtFeetmeanEwrtWorldZ(
    const vector_t& q,
    PinocchioInterface& pinocchioInterface,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel,
    absl::Span<const size_t> footContactIndices) {
  pinocchio::centerOfMass(pinocchioInterface.getModel(),
                          pinocchioInterface.getData(), q,
                          /*computeSubtreeComs=*/false);
  const scalar_t posComRtWorldEwrtWorldZ =
      pinocchioInterface.getData().com[0](kZAxisIndex);
  const std::vector<vector3_t> posContactsRtWorldEwrtWorld =
      computePosContactsRtWorldEwrtWorld<scalar_t>(q, pinocchioInterface,
                                                   mpcRobotModel);
  scalar_t posFeetmeanRtWorldEwrtWorldZ = 0.0;
  for (const size_t contact : footContactIndices) {
    posFeetmeanRtWorldEwrtWorldZ +=
        posContactsRtWorldEwrtWorld[contact](kZAxisIndex) /
        static_cast<scalar_t>(footContactIndices.size());
  }
  return posComRtWorldEwrtWorldZ - posFeetmeanRtWorldEwrtWorldZ;
}

scalar_t computePosComRtLowcontactEwrtWorldZ(
    const vector_t& q,
    PinocchioInterface& pinocchioInterface,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel) {
  pinocchio::centerOfMass(pinocchioInterface.getModel(),
                          pinocchioInterface.getData(), q,
                          /*computeSubtreeComs=*/false);
  const scalar_t posComRtWorldEwrtWorldZ =
      pinocchioInterface.getData().com[0](kZAxisIndex);
  updateFramePlacements<scalar_t>(q, pinocchioInterface);
  scalar_t minPosContactsRtWorldEwrtWorldZ =
      std::numeric_limits<scalar_t>::infinity();
  for (size_t i = 0; i < mpcRobotModel.modelSettings.contactNames.size(); ++i) {
    minPosContactsRtWorldEwrtWorldZ = std::min(
        minPosContactsRtWorldEwrtWorldZ,
        pinocchioInterface.getData()
            .oMf[getContactFrameIndex(pinocchioInterface, mpcRobotModel, i)]
            .translation()(kZAxisIndex));
  }
  return posComRtWorldEwrtWorldZ - minPosContactsRtWorldEwrtWorldZ;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

template <typename SCALAR_T>
VECTOR6_T<SCALAR_T> computeBaseGeneralizedAcceleration(
    const MATRIX_T<SCALAR_T>& M,
    const VECTOR_T<SCALAR_T>& nle,
    const VECTOR_T<SCALAR_T>& qdd_joints,
    const VECTOR_T<SCALAR_T>& generalizedExternalForce) {
  // The base rows of M qdd + nle = J^T F, solved for the base acceleration:
  // M_bb a_b = r with r = -nle_b - M_bj qdd_j + (J^T F)_b. The base has the
  // kPoseTangentDim velocities of a pose: linear, then angular.
  const MATRIX_T<SCALAR_T> M_bj =
      M.block(0, kPoseTangentDim, kPoseTangentDim, qdd_joints.size());
  const VECTOR6_T<SCALAR_T> r = -nle.head(kPoseTangentDim) - M_bj * qdd_joints +
                                generalizedExternalForce.head(kPoseTangentDim);

  // M_bb = [A B; C D] is NOT block diagonal: the translational and rotational
  // coordinates of the base couple through the offset of the whole-body center
  // of mass from the base, so inverting A and D separately is wrong. M_bb is
  // solved exactly through the Schur complement S = D - C A^-1 B, with 3x3
  // inverses only. That is deliberate: Eigen inverts a fixed-size 3x3 matrix by
  // cofactors, with no comparison, whereas a 6x6 inverse goes through a
  // pivoting LU that compares entries of the matrix. A comparison between CppAD
  // variables is recorded on the tape, and CppADCodeGen's code generator cannot
  // evaluate it: the whole-body dynamics then fail to compile with
  // "GreaterThanZero cannot be called for non-parameters". A is the
  // translational block, the total mass times the identity, and S is the
  // rotational inertia about the center of mass in the base's body axes (the
  // Spherical root's velocity is the body angular velocity w_B), positive
  // definite at every attitude: unlike the Euler-angle rates of a SphericalZYX
  // root, it has no singularity at a pitch of +-90 degrees, so both are
  // invertible everywhere. crba() fills both triangles of this 6x6 root block
  // (it writes the root joint's rows over its whole subtree), so C is read, not
  // assumed to be B^T.
  const MATRIX3_T<SCALAR_T> A =
      M.template block<kPositionDim, kPositionDim>(0, 0);
  const MATRIX3_T<SCALAR_T> B =
      M.template block<kPositionDim, kRotationTangentDim>(0, kPositionDim);
  const MATRIX3_T<SCALAR_T> C =
      M.template block<kRotationTangentDim, kPositionDim>(kPositionDim, 0);
  const MATRIX3_T<SCALAR_T> D =
      M.template block<kRotationTangentDim, kRotationTangentDim>(kPositionDim,
                                                                 kPositionDim);
  const MATRIX3_T<SCALAR_T> AInverse = A.inverse();
  const MATRIX3_T<SCALAR_T> schurComplementInverse =
      (D - C * AInverse * B).inverse();

  const VECTOR3_T<SCALAR_T> r1 = r.template head<kPositionDim>();
  const VECTOR3_T<SCALAR_T> r2 = r.template tail<kRotationTangentDim>();
  const VECTOR3_T<SCALAR_T> angular =
      schurComplementInverse * (r2 - C * (AInverse * r1));
  const VECTOR3_T<SCALAR_T> linear = AInverse * (r1 - B * angular);

  VECTOR6_T<SCALAR_T> baseGeneralizedAcceleration;
  baseGeneralizedAcceleration << linear, angular;
  return baseGeneralizedAcceleration;
}
template VECTOR6_T<scalar_t> computeBaseGeneralizedAcceleration(
    const MATRIX_T<scalar_t>& M,
    const VECTOR_T<scalar_t>& nle,
    const VECTOR_T<scalar_t>& qdd_joints,
    const VECTOR_T<scalar_t>& generalizedExternalForce);
template VECTOR6_T<ad_scalar_t> computeBaseGeneralizedAcceleration(
    const MATRIX_T<ad_scalar_t>& M,
    const VECTOR_T<ad_scalar_t>& nle,
    const VECTOR_T<ad_scalar_t>& qdd_joints,
    const VECTOR_T<ad_scalar_t>& generalizedExternalForce);

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

namespace {

///
/// Writes the full joint-space inertia matrix M(q) into data.M and the
/// nonlinear effects nle(q, qd) into data.nle, and returns the generalized
/// contact force sum_i J_i^T W_i over the contacts: the terms of M qdd + nle =
/// S^T tau +
/// J^T W that the inverse dynamics below read.
///
/// pinocchio::crba() writes, for every joint, its rows over the columns of its
/// own subtree: the UPPER TRIANGLE of M, and only the part of it where one
/// joint supports the other (the full 6x6 block of the composite base joint
/// included). Every other entry - the lower triangle, and the coupling of two
/// joints on different branches, such as the two legs - it leaves at whatever
/// the data held, zero for a fresh one but not for a data another algorithm has
/// written. The joint rows of M qdd read both: the joint-base coupling M_jb and
/// the lower triangle of M_jj. So M is cleared first, and its strictly lower
/// triangle mirrored from the upper one after, which is exact: M is symmetric.
/// Neither records an operation on a CppAD tape, only copies.
///
template <typename SCALAR_T>
VECTOR_T<SCALAR_T> updateInverseDynamicsTerms(
    const VECTOR_T<SCALAR_T>& q,
    const VECTOR_T<SCALAR_T>& qd,
    const ContactArray<VECTOR6_T<SCALAR_T>>& wrenchContactsEwrtWorld,
    const ContactArray<pinocchio::FrameIndex>& contactFrames,
    const pinocchio::ModelTpl<SCALAR_T>& model,
    pinocchio::DataTpl<SCALAR_T>& data) {
  data.M.setZero();
  pinocchio::crba(model, data, q);
  data.M.template triangularView<Eigen::StrictlyLower>() =
      data.M.transpose().template triangularView<Eigen::StrictlyLower>();
  pinocchio::nonLinearEffects(model, data, q, qd);

  // LOCAL_WORLD_ALIGNED Jacobians of the contact frames, which take the
  // world-frame wrenches of the MPC.
  VECTOR_T<SCALAR_T> generalizedContactForce =
      VECTOR_T<SCALAR_T>::Zero(qd.size());
  for (size_t i = 0; i < contactFrames.size(); ++i) {
    MATRIX_T<SCALAR_T> jacContactRtWorldEwrtWorld =
        MATRIX_T<SCALAR_T>::Zero(kTwistDim, qd.size());
    pinocchio::computeFrameJacobian(
        model, data, q, contactFrames[i],
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
        jacContactRtWorldEwrtWorld);
    generalizedContactForce +=
        jacContactRtWorldEwrtWorld.transpose() * wrenchContactsEwrtWorld[i];
  }
  return generalizedContactForce;
}

}  // namespace

template <typename SCALAR_T>
VECTOR_T<SCALAR_T> computeJointTorques(
    const VECTOR_T<SCALAR_T>& q,
    const VECTOR_T<SCALAR_T>& qd,
    const VECTOR_T<SCALAR_T>& qdd_joints,
    const ContactArray<VECTOR6_T<SCALAR_T>>& wrenchContactsEwrtWorld,
    const ContactArray<pinocchio::FrameIndex>& contactFrames,
    PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface) {
  const typename PinocchioInterfaceTpl<SCALAR_T>::Model& model =
      pinocchioInterface.getModel();
  pinocchio::DataTpl<SCALAR_T>& data = pinocchioInterface.getData();
  const VECTOR_T<SCALAR_T> generalizedExternalForce =
      updateInverseDynamicsTerms<SCALAR_T>(q, qd, wrenchContactsEwrtWorld,
                                           contactFrames, model, data);

  // The base is unactuated: its rows fix the base acceleration the wrenches
  // produce, and the joint rows then carry it.
  const VECTOR6_T<SCALAR_T> baseGeneralizedAcceleration =
      computeBaseGeneralizedAcceleration<SCALAR_T>(data.M, data.nle, qdd_joints,
                                                   generalizedExternalForce);
  VECTOR_T<SCALAR_T> q_dd(qd.size());
  q_dd << baseGeneralizedAcceleration, qdd_joints;

  const Eigen::Index numJoints = qdd_joints.size();
  return data.M.bottomRows(numJoints) * q_dd + data.nle.tail(numJoints) -
         generalizedExternalForce.tail(numJoints);
}
template VECTOR_T<scalar_t> computeJointTorques(
    const VECTOR_T<scalar_t>& q,
    const VECTOR_T<scalar_t>& qd,
    const VECTOR_T<scalar_t>& qdd_joints,
    const ContactArray<VECTOR6_T<scalar_t>>& wrenchContactsEwrtWorld,
    const ContactArray<pinocchio::FrameIndex>& contactFrames,
    PinocchioInterfaceTpl<scalar_t>& pinocchioInterface);
template VECTOR_T<ad_scalar_t> computeJointTorques(
    const VECTOR_T<ad_scalar_t>& q,
    const VECTOR_T<ad_scalar_t>& qd,
    const VECTOR_T<ad_scalar_t>& qdd_joints,
    const ContactArray<VECTOR6_T<ad_scalar_t>>& wrenchContactsEwrtWorld,
    const ContactArray<pinocchio::FrameIndex>& contactFrames,
    PinocchioInterfaceTpl<ad_scalar_t>& pinocchioInterface);

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

template <typename SCALAR_T>
VECTOR_T<SCALAR_T> computeBaseHeldJointTorques(
    const VECTOR_T<SCALAR_T>& q,
    const VECTOR_T<SCALAR_T>& qd,
    const VECTOR_T<SCALAR_T>& qdd_joints,
    const ContactArray<VECTOR6_T<SCALAR_T>>& wrenchContactsEwrtWorld,
    const ContactArray<pinocchio::FrameIndex>& contactFrames,
    PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface) {
  const typename PinocchioInterfaceTpl<SCALAR_T>::Model& model =
      pinocchioInterface.getModel();
  pinocchio::DataTpl<SCALAR_T>& data = pinocchioInterface.getData();
  const VECTOR_T<SCALAR_T> generalizedExternalForce =
      updateInverseDynamicsTerms<SCALAR_T>(q, qd, wrenchContactsEwrtWorld,
                                           contactFrames, model, data);

  // Whatever holds the base (the gantry) supplies the base rows; the base
  // acceleration is zero.
  const Eigen::Index numJoints = qdd_joints.size();
  return data.M.bottomRightCorner(numJoints, numJoints) * qdd_joints +
         data.nle.tail(numJoints) - generalizedExternalForce.tail(numJoints);
}
template VECTOR_T<scalar_t> computeBaseHeldJointTorques(
    const VECTOR_T<scalar_t>& q,
    const VECTOR_T<scalar_t>& qd,
    const VECTOR_T<scalar_t>& qdd_joints,
    const ContactArray<VECTOR6_T<scalar_t>>& wrenchContactsEwrtWorld,
    const ContactArray<pinocchio::FrameIndex>& contactFrames,
    PinocchioInterfaceTpl<scalar_t>& pinocchioInterface);

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

absl::Status checkContactFramesExist(
    const pinocchio::Model& model, absl::Span<const std::string> contactNames) {
  if (contactNames.empty()) {
    return absl::InvalidArgumentError(
        "model_settings names no contact frame; the MPC needs at least one "
        "contact.");
  }
  for (size_t i = 0; i < contactNames.size(); ++i) {
    if (!model.existFrame(contactNames[i])) {
      return absl::NotFoundError(
          absl::StrCat("contact frame \"", contactNames[i], "\" (contact ", i,
                       " of model_settings.contact_names) is not a frame of "
                       "the robot model."));
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<scalar_t> modelGravity(
    const pinocchio::ModelTpl<scalar_t>& model) {
  const vector3_t gravityEwrtWorld = model.gravity.linear();
  if (!gravityEwrtWorld.allFinite() || gravityEwrtWorld.x() != 0.0 ||
      gravityEwrtWorld.y() != 0.0 || !(gravityEwrtWorld.z() < 0.0)) {
    return absl::InvalidArgumentError(
        absl::StrCat("The robot model's gravity (", gravityEwrtWorld.x(), ", ",
                     gravityEwrtWorld.y(), ", ", gravityEwrtWorld.z(),
                     ") m/s^2 does not point down the world z axis; the "
                     "pendulums of the MPC need it to."));
  }
  return -gravityEwrtWorld.z();
}

}  // namespace ocs2::humanoid
