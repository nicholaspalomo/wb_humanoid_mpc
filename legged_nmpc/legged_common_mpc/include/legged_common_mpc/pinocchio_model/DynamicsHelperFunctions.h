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

#pragma once

#include "pinocchio/fwd.hpp"

#include <cstddef>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "cppad/cg.hpp"
#include "ocs2_core/PhysicalConstants.h"
#include "ocs2_core/Types.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/multibody.hpp"

#include "legged_common_mpc/common/BipedFeet.h"
#include "legged_common_mpc/common/ContactArray.h"
#include "legged_common_mpc/common/ContactMask.h"
#include "legged_common_mpc/common/MpcRobotModelBase.h"
#include "legged_common_mpc/common/Types.h"
#include "legged_common_mpc/pinocchio_model/PinocchioFrameConversions.h"

namespace ocs2::legged {

///
/// @brief Updates all frame placement in the PinocchioInterface.
///
/// @tparam SCALAR_T Scalar type [scalar_t/ad_scalar_t].
/// @param q Current generalized coordinates.
/// @param pinocchioInterface Pinocchio interface.
///

template <typename SCALAR_T>
void updateFramePlacements(const VECTOR_T<SCALAR_T>& q,
                           PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface);

///
/// @brief Computes and returns all the contact positions in the inertial frame.
///
/// @tparam SCALAR_T Scalar type [scalar_t/ad_scalar_t].
/// @param q Current generalized coordinates.
/// @param pinocchioInterface Pinocchio interface.
///
/// @return Vector of contact positions in the inertial frame.

template <typename SCALAR_T>
void updateFramePlacements(const VECTOR_T<SCALAR_T>& q,
                           const pinocchio::ModelTpl<SCALAR_T>& model,
                           pinocchio::DataTpl<SCALAR_T>& data);

///
/// @brief Computes and returns all the contact positions in the inertial frame.
///
/// @tparam SCALAR_T Scalar type [scalar_t/ad_scalar_t].
/// @param q Current generalized coordinates.
/// @param pinocchioInterface Pinocchio interface.
///
/// @return Vector of contact positions in the inertial frame.

template <typename SCALAR_T>
std::vector<VECTOR3_T<SCALAR_T>> computePosContactsRtWorldEwrtWorld(
    const VECTOR_T<SCALAR_T>& q,
    PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface,
    const MpcRobotModelBase<SCALAR_T>& mpcRobotModel);

///
/// @brief Returns the positions of every contact of the set
/// (ModelSettings::contactNames, in contact order) in the inertial frame: the
/// origin of each contact frame, and for a rolling contact the wheel's lowest
/// point, its frame's origin less the wheel radius along world z
/// (contact/RollingContact.h). Assumes that the frame placements are up to
/// date (e.g. updateFramePlacements has been called).
///
/// @tparam SCALAR_T Scalar type [scalar_t/ad_scalar_t].
/// @param pinocchioInterface Pinocchio interface.
///
/// @return Vector of contact positions in the inertial frame.

template <typename SCALAR_T>
std::vector<VECTOR3_T<SCALAR_T>> getPosContactsRtWorldEwrtWorld(
    const PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface,
    const MpcRobotModelBase<SCALAR_T>& mpcRobotModel);

///
/// @brief Computes and returns all the frame positions in the inertial frame.
///
/// @tparam SCALAR_T Scalar type [scalar_t/ad_scalar_t].
/// @param q Current generalized coordinates.
/// @param pinocchioInterface Pinocchio interface.
/// @param frameNames Names of the frames for which the positions should be
/// computed.
///
/// @return Vector of contact positions in the inertial frame.

template <typename SCALAR_T>
std::vector<VECTOR3_T<SCALAR_T>> computePosLinksRtWorldEwrtWorld(
    const VECTOR_T<SCALAR_T>& q,
    PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface,
    absl::Span<const std::string> frameNames);

///
/// @brief Returns all the frame positions in the inertial frame.
/// Assumes that the frame placements are up to date (e.g. updateFramePlacements
/// has been called).
///
/// @tparam SCALAR_T Scalar type [scalar_t/ad_scalar_t].
/// @param pinocchioInterface Pinocchio interface.
/// @param frameNames Names of the frames for which the positions should be
/// returned.
///
/// @return Vector of contact positions in the inertial frame.

template <typename SCALAR_T>
std::vector<VECTOR3_T<SCALAR_T>> getPosLinksRtWorldEwrtWorld(
    const PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface,
    absl::Span<const std::string> frameNames);

///
/// @brief The pendulum length of the robot: its whole-body center of mass above
/// the mean height of its contact frames.
///
/// The one definition of the linear inverted pendulum's height in this
/// repository. Every consumer that needs a nominal pendulum length evaluates it
/// at the task file's `initialState` - the contact planner's
/// `shared.nomPosComRtFeetmeanEwrtWorldZ` and the DCM terminal cost's
/// `dcm_terminal_cost.nomPosComRtFeetmeanEwrtWorldZ` when they are 0
/// (CentroidalMpcInterface::getNomPosComRtFeetmeanEwrtWorldZ), and the
/// locomotion heuristics' nominal CoM height - so that they cannot disagree;
/// the capture-point heuristic also measures it at every solve.
/// tools/locomotion_heuristics/derive_parameters.py computes the same quantity
/// the same way.
///
/// Above the mean foot height rather than above the world origin, so that it is
/// the same number on a robot standing on a box as on one standing on the
/// floor. The contact frames are the sole centers (createPinocchioModel adds
/// them), and a rolling contact is its wheel's lowest point
/// (getPosContactsRtWorldEwrtWorld()). `feetmean` is the mean over the legs'
/// contacts (ModelSettings::legs): on the biped contact set
/// (ModelSettings::isBipedContactSet()) its two feet, on another set with legs
/// every contact of a leg. On a set without legs, such as a robot sitting on a
/// seat, it is the height above the lowest contact frame of the set instead.
/// Updates the center of mass and the frame placements of
/// `pinocchioInterface`'s data.
///
/// @param q Generalized coordinates
/// (MpcRobotModelBase::getGeneralizedCoordinates of a state).
/// @param pinocchioInterface Pinocchio interface of the MPC model.
/// @param mpcRobotModel The MPC robot model, which names the contact frames.
///
/// @return [m] computePosComRtFeetmeanEwrtWorldZ() on the biped contact set
/// and, over the legs' contacts, on another set with legs;
/// computePosComRtLowcontactEwrtWorldZ() on a set without legs, where it stands
/// in for the height above the feet (the `proxy` of the frame naming
/// convention).

scalar_t computeProxyPosComRtFeetmeanEwrtWorldZ(
    const vector_t& q,
    PinocchioInterface& pinocchioInterface,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel);

///
/// @brief Computes the center of mass height above the mean of foot contact
/// frames' origins (the soles): every contact of the set (the biped's two
/// feet, ModelSettings::isBipedContactSet()), or the contacts
/// `footContactIndices` (indices into ModelSettings::contactNames, at least
/// one, each valid). Updates the center of mass and the frame placements of
/// `pinocchioInterface`'s data.
///
/// @return [m] the center of mass height above the mean height of the foot
/// contact frames.

scalar_t computePosComRtFeetmeanEwrtWorldZ(
    const vector_t& q,
    PinocchioInterface& pinocchioInterface,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel);
scalar_t computePosComRtFeetmeanEwrtWorldZ(
    const vector_t& q,
    PinocchioInterface& pinocchioInterface,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel,
    absl::Span<const size_t> footContactIndices);

///
/// @brief Computes the center of mass height above the lowest contact frame of
/// the contact set (for a rolling contact, its wheel's lowest point): the
/// support of a robot sitting or lying on a contact set without legs. Updates
/// the center of mass and the frame placements of `pinocchioInterface`'s data.
///
/// @return [m] the center of mass height above the lowest contact frame.

scalar_t computePosComRtLowcontactEwrtWorldZ(
    const vector_t& q,
    PinocchioInterface& pinocchioInterface,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel);

///
/// @brief Return amount of legs in contact
///
/// @return number of clodes leg contacts as unsigned integer

inline size_t numberOfLegsInContacts(const BipedFeetArray<bool>& contactFlags) {
  size_t numStanceLegs = 0;
  for (bool legInContact : contactFlags) {
    if (legInContact) {
      ++numStanceLegs;
    }
  }
  return numStanceLegs;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

///
/// @brief The magnitude g [m/s^2] of the gravity of `model`
/// (pinocchio::Model::gravity): the gravity the robot's dynamics use, and the
/// one every pendulum of the MPC (the DCM terminal cost, the contact planner,
/// the capture point) is made with, so that none of them is a configuration key
/// that could disagree with the model.
///
/// @return InvalidArgument unless the gravity points straight down the world z
/// axis with a finite, positive magnitude,
///         which the pendulums' sqrt(g / z) assumes.
///
absl::StatusOr<scalar_t> modelGravity(
    const pinocchio::ModelTpl<scalar_t>& model);

///
/// @brief The weight of the robot that `pinocchioInterface` models [N]: its
/// total mass times kGravityMagnitude.
///
/// Evaluated per call rather than cached in a function-static: a static is
/// initialized by the first model that reaches it and then answers for every
/// other model in the process (the test binaries hold several robots at once).
///
inline scalar_t computeRobotWeight(
    const PinocchioInterface& pinocchioInterface) {
  return computeTotalMass(pinocchioInterface.getModel()) * kGravityMagnitude;
}

inline vector_t weightCompensatingInput(
    const PinocchioInterface& pinocchioInterface,
    const BipedFeetArray<bool>& contactFlags,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel) {
  const scalar_t weight = computeRobotWeight(pinocchioInterface);
  const size_t numStanceLegs = numberOfLegsInContacts(contactFlags);
  vector_t input = vector_t::Zero(mpcRobotModel.getInputDim());
  if (numStanceLegs > 0) {
    // NOTE: uses the input-only setter, i.e. the force is written in the frame
    // of the input parameterization (world frame for wrench-space models, local
    // contact frame for basis-vector inputs). For a flat foot the vertical
    // force is identical in both frames. Prefer the state-aware overload below
    // when a state is available.
    const vector3_t forceContactEwrtParam(0.0, 0.0, weight / numStanceLegs);
    for (size_t i = 0; i < contactFlags.size(); ++i) {
      if (contactFlags[i]) {
        mpcRobotModel.setForceContactEwrtParam(input, forceContactEwrtParam, i);
      }
    }
  }
  return input;
}

/**
 * weightCompensatingInput() of the contacts `contactFlags` says are active, for
 * a contact set of any size: the robot's weight shared evenly by them as
 * vertical forces through the input-only setter, and zero joint velocities.
 * Zero when no contact is active. Contact i of the mask is contact i of
 * `mpcRobotModel`'s set.
 */
inline vector_t weightCompensatingInput(
    const PinocchioInterface& pinocchioInterface,
    ContactMask contactFlags,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel) {
  const scalar_t weight = computeRobotWeight(pinocchioInterface);
  const size_t numStanceContacts = contactFlags.numActive();
  vector_t input = vector_t::Zero(mpcRobotModel.getInputDim());
  if (numStanceContacts == 0) return input;
  // The same arithmetic as the two-foot overload above, so that a biped's
  // reference is bit-identical whichever of the two it is built with.
  const vector3_t forceContactEwrtParam(
      0.0, 0.0, weight / static_cast<scalar_t>(numStanceContacts));
  for (size_t i = 0; i < contactFlags.size(); ++i) {
    if (contactFlags[i]) {
      mpcRobotModel.setForceContactEwrtParam(input, forceContactEwrtParam, i);
    }
  }
  return input;
}

///
/// @brief Computes the weight-compensating input (evenly distributed vertical
/// world-frame forces) for the given state. The state is used to express the
/// world-frame force correctly in the input parameterization (e.g. rotated into
/// the local contact frame for basis-vector inputs).
///
/// @param pinocchioInterface Pinocchio interface.
/// @param contactFlags Contact flags.
/// @param mpcRobotModel MPC robot model.
/// @param state Current MPC state.
///
/// @return input with weight-compensating contact forces and zero joint
/// velocities.

inline vector_t weightCompensatingInput(
    const PinocchioInterface& pinocchioInterface,
    const BipedFeetArray<bool>& contactFlags,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel,
    const vector_t& state) {
  const scalar_t weight = computeRobotWeight(pinocchioInterface);
  const size_t numStanceLegs = numberOfLegsInContacts(contactFlags);
  vector_t input = vector_t::Zero(mpcRobotModel.getInputDim());
  if (numStanceLegs > 0) {
    const vector3_t forceContactEwrtWorld(0.0, 0.0, weight / numStanceLegs);
    for (size_t i = 0; i < contactFlags.size(); ++i) {
      if (contactFlags[i]) {
        mpcRobotModel.setForceContactEwrtWorld(state, input,
                                               forceContactEwrtWorld, i);
      }
    }
  }
  return input;
}

/** The state-aware weightCompensatingInput() of the contacts `contactFlags`
 * says are active, for a contact set of any size: vertical world-frame forces
 * written through the world-frame setter at `state`. */
inline vector_t weightCompensatingInput(
    const PinocchioInterface& pinocchioInterface,
    ContactMask contactFlags,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel,
    const vector_t& state) {
  const scalar_t weight = computeRobotWeight(pinocchioInterface);
  const size_t numStanceContacts = contactFlags.numActive();
  vector_t input = vector_t::Zero(mpcRobotModel.getInputDim());
  if (numStanceContacts == 0) return input;
  const vector3_t forceContactEwrtWorld(
      0.0, 0.0, weight / static_cast<scalar_t>(numStanceContacts));
  for (size_t i = 0; i < contactFlags.size(); ++i) {
    if (contactFlags[i]) {
      mpcRobotModel.setForceContactEwrtWorld(state, input,
                                             forceContactEwrtWorld, i);
    }
  }
  return input;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

template <typename SCALAR_T>
inline pinocchio::FrameIndex getContactFrameIndex(
    const PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface,
    const MpcRobotModelBase<SCALAR_T>& mpcRobotModel,
    size_t contactIndex) {
  return pinocchioInterface.getModel().getFrameId(
      mpcRobotModel.modelSettings.contactNames[contactIndex]);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

template <typename SCALAR_T>
inline std::vector<pinocchio::FrameIndex> getContactFrameIndices(
    const PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface,
    const MpcRobotModelBase<SCALAR_T>& mpcRobotModel) {
  // Sized, not reserved: reserve leaves size() at 0, so indexing below would be
  // out of bounds and the returned vector would be empty - which silently
  // turned every range-for over these indices into a no-op.
  const size_t numContacts = mpcRobotModel.modelSettings.contactNames.size();
  std::vector<pinocchio::FrameIndex> contactFrameIndices(numContacts);
  for (size_t i = 0; i < numContacts; ++i) {
    contactFrameIndices[i] =
        getContactFrameIndex<SCALAR_T>(pinocchioInterface, mpcRobotModel, i);
  }
  return contactFrameIndices;
}

/**
 * Returns the Pinocchio frame of every contact of `mpcRobotModel`
 * (modelSettings.contactNames), in contact order (the 3-DoF contacts first,
 * then the 6-DoF ones): the frames the inverse dynamics below apply the contact
 * wrenches at. The frames must exist (checkContactFramesExist(), which the MPC
 * interfaces call once at construction), and the set has at most
 * kMaxNumContacts contacts (ModelSettings::Create() refuses more).
 */
template <typename SCALAR_T>
inline ContactArray<pinocchio::FrameIndex> getContactFrameIndexArray(
    const PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface,
    const MpcRobotModelBase<SCALAR_T>& mpcRobotModel) {
  const size_t numContacts = mpcRobotModel.modelSettings.contactNames.size();
  ContactArray<pinocchio::FrameIndex> contactFrameIndices(numContacts);
  for (size_t i = 0; i < numContacts; ++i) {
    contactFrameIndices[i] =
        getContactFrameIndex<SCALAR_T>(pinocchioInterface, mpcRobotModel, i);
  }
  return contactFrameIndices;
}

/**
 * Checks that `model` has a frame of every name of `contactNames`
 * (modelSettings.contactNames).
 *
 * @return InvalidArgument when there are no names; NotFound naming the first
 * contact frame that `model` does not have, which
 * every term that reads a contact frame by its index would otherwise read out
 * of bounds (pinocchio::Model::getFrameId() returns nframes for an unknown
 * name).
 */
absl::Status checkContactFramesExist(
    const pinocchio::Model& model, absl::Span<const std::string> contactNames);

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

///
/// @brief Computes the position of a contact's center of pressure (CoP) in the
/// world, from the state-aware world-frame wrench accessor, which is
/// frame-correct for every input parameterization (wrench-space and
/// basis-vector inputs).
///
/// @warning Assumes that the frame placements are up to date. Does not work
/// when frame is not in contact -> f_z = 0 results in NaN.
///
/// @tparam SCALAR_T Scalar type [scalar_t/ad_scalar_t].
/// @param state Current state.
/// @param input Current input.
/// @param pinocchioInterface Pinocchio interface.
///
/// @return Location of center of pressure in the inertial frame.

template <typename SCALAR_T>
inline VECTOR3_T<SCALAR_T> computePosCopRtWorldEwrtWorld(
    const VECTOR_T<SCALAR_T>& state,
    const VECTOR_T<SCALAR_T>& input,
    const PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface,
    size_t contactIndex,
    const MpcRobotModelBase<SCALAR_T>& mpcRobotModel) {
  const pinocchio::FrameIndex frameIndex =
      getContactFrameIndex(pinocchioInterface, mpcRobotModel, contactIndex);
  const VECTOR6_T<SCALAR_T> wrenchContactEwrtContact =
      rotateVectorWorldToLocal<SCALAR_T>(
          mpcRobotModel.getWrenchContactEwrtWorld(state, input, contactIndex),
          pinocchioInterface.getData(), frameIndex);
  // The wrench is [force, torque]: the CoP is (-tau_y, tau_x) / f_z.
  SCALAR_T posCopRtContactEwrtContactX =
      -wrenchContactEwrtContact[kWrenchTorqueYIndex] /
      wrenchContactEwrtContact[kWrenchForceZIndex];
  SCALAR_T posCopRtContactEwrtContactY =
      wrenchContactEwrtContact[kWrenchTorqueXIndex] /
      wrenchContactEwrtContact[kWrenchForceZIndex];
  VECTOR3_T<SCALAR_T> posCopRtContactEwrtContact(
      posCopRtContactEwrtContactX, posCopRtContactEwrtContactY, /*z=*/0.0);
  return transformPointLocalToWorld(posCopRtContactEwrtContact,
                                    pinocchioInterface.getData(), frameIndex);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

///
/// @brief Computes the base acceleration
///
/// @param M Given mass matrix
/// @param nle nonlinear effects in joint space inlcuding gravity comp,
/// centrifugal, corriolis
/// @param qdd_joints Generalized accelerations
/// @param generalizedExternalForce Sum of J^T F_ext
///
/// @return linear and angular base acceleration.

template <typename SCALAR_T>
VECTOR6_T<SCALAR_T> computeBaseGeneralizedAcceleration(
    const MATRIX_T<SCALAR_T>& M,
    const VECTOR_T<SCALAR_T>& nle,
    const VECTOR_T<SCALAR_T>& qdd_joints,
    const VECTOR_T<SCALAR_T>& generalizedExternalForce);

///
/// @brief Floating-base inverse dynamics: the joint torques that produce the
/// joint accelerations `qdd_joints` while the contacts push on the ground with
/// `wrenchContactsEwrtWorld`, the base being free.
///
/// The unactuated base rows of M qdd + nle = S^T tau + J^T W fix the base
/// acceleration those wrenches produce (computeBaseGeneralizedAcceleration),
/// and the joint rows then return tau_j = M_jb a_b + M_jj qdd_j + nle_j - (J^T
/// W)_j: the joint rows of pinocchio::rnea() at that base acceleration and with
/// those external forces. It used to multiply the joint rows of crba()'s
/// upper-triangular M and so left out the base coupling M_jb a_b (and the lower
/// triangle of M_jj); testJointTorqueInverseDynamics pins it against RNEA.
///
/// With no wrench at all the base is in free fall and gravity loads no joint;
/// for a robot whose base is held from outside (the gantry) use
/// computeBaseHeldJointTorques.
///
/// Updates M, nle and the frame Jacobians in `pinocchioInterface`'s data.
///
/// @param q Generalized coordinates [base position (3), base quaternion (x, y,
/// z, w) (4), joint angles]: nq = 7 + nj.
/// @param qd Generalized velocities [base linear velocity in the world (3),
/// body angular velocity (3), joint velocities]: nv = 6 + nj.
/// @param qdd_joints Joint accelerations.
/// @param wrenchContactsEwrtWorld The wrench of each contact, in contact order,
/// world frame, at the origin of its frame in `contactFrames` (a point contact
/// has a zero moment).
/// @param contactFrames The frame of each contact, in contact order
/// (getContactFrameIndexArray()); each must be a frame of the model, and it has
/// as many entries as `wrenchContactsEwrtWorld`.
/// @param pinocchioInterface Pinocchio interface of the MPC model.
///
/// @return joint torques of same dimension as qdd_joints

template <typename SCALAR_T>
VECTOR_T<SCALAR_T> computeJointTorques(
    const VECTOR_T<SCALAR_T>& q,
    const VECTOR_T<SCALAR_T>& qd,
    const VECTOR_T<SCALAR_T>& qdd_joints,
    const ContactArray<VECTOR6_T<SCALAR_T>>& wrenchContactsEwrtWorld,
    const ContactArray<pinocchio::FrameIndex>& contactFrames,
    PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface);

///
/// @brief Inverse dynamics of a robot whose base is held still from outside
/// (the gantry): the joint rows of M qdd + nle = S^T tau + J^T W at zero base
/// acceleration, tau_j = M_jj qdd_j + nle_j - (J^T W)_j.
///
/// The base rows are not balanced - whatever holds the base supplies them. With
/// no wrench this is the gravity and Coriolis torque of the free legs, and with
/// the weight on the feet it is g_j(q) - J_j^T W, the static feedforward that
/// holds a posture. For qdd_j = 0 it is exactly what computeJointTorques
/// returned before it included the base coupling.
///
/// Updates M, nle and the frame Jacobians in `pinocchioInterface`'s data.
///
/// @param q Generalized coordinates [base position (3), base quaternion (x, y,
/// z, w) (4), joint angles]: nq = 7 + nj.
/// @param qd Generalized velocities [base linear velocity in the world (3),
/// body angular velocity (3), joint velocities]: nv = 6 + nj.
/// @param qdd_joints Joint accelerations.
/// @param wrenchContactsEwrtWorld The wrench of each contact, as for
/// computeJointTorques.
/// @param contactFrames The frame of each contact, as for
/// computeJointTorques.
/// @param pinocchioInterface Pinocchio interface of the MPC model.
///
/// @return joint torques of same dimension as qdd_joints

template <typename SCALAR_T>
VECTOR_T<SCALAR_T> computeBaseHeldJointTorques(
    const VECTOR_T<SCALAR_T>& q,
    const VECTOR_T<SCALAR_T>& qd,
    const VECTOR_T<SCALAR_T>& qdd_joints,
    const ContactArray<VECTOR6_T<SCALAR_T>>& wrenchContactsEwrtWorld,
    const ContactArray<pinocchio::FrameIndex>& contactFrames,
    PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface);

}  // namespace ocs2::legged
