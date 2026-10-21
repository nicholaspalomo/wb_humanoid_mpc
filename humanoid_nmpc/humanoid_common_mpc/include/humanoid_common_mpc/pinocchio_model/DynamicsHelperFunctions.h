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

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "cppad/cg.hpp"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/PhysicalConstants.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "ocs2_robotic_tools/common/RotationTransforms.h"
#include "pinocchio/algorithm/center-of-mass.hpp"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/pinocchio_model/PinocchioFrameConversions.h"

namespace ocs2::humanoid {

///
/// @brief Updates all frame placement in the PinocchioInterface.
///
/// @tparam SCALAR_T Scalar type [scalar_t/ad_scalar_t].
/// @param q Current generalized coordinates.
/// @param pinocchioInterface Pinocchio interface.
///

template <typename SCALAR_T>
void updateFramePlacements(const VECTOR_T<SCALAR_T>& q, PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface);

///
/// @brief Computes and returns all the contact positions in the inertial frame.
///
/// @tparam SCALAR_T Scalar type [scalar_t/ad_scalar_t].
/// @param q Current generalized coordinates.
/// @param pinocchioInterface Pinocchio interface.
///
/// @return Vector of contact positions in the inertial frame.

template <typename SCALAR_T>
void updateFramePlacements(const VECTOR_T<SCALAR_T>& q, const pinocchio::ModelTpl<SCALAR_T>& model, pinocchio::DataTpl<SCALAR_T>& data);

///
/// @brief Computes and returns all the contact positions in the inertial frame.
///
/// @tparam SCALAR_T Scalar type [scalar_t/ad_scalar_t].
/// @param q Current generalized coordinates.
/// @param pinocchioInterface Pinocchio interface.
///
/// @return Vector of contact positions in the inertial frame.

template <typename SCALAR_T>
std::vector<VECTOR3_T<SCALAR_T>> computeContactPositions(const VECTOR_T<SCALAR_T>& q,
                                                         PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface,
                                                         const MpcRobotModelBase<SCALAR_T>& mpcRobotModel);

///
/// @brief Returns all the contact positions in the inertial frame.
/// Assumes that the frame placements are up to date (e.g. updateFramePlacements has been called).
///
/// @tparam SCALAR_T Scalar type [scalar_t/ad_scalar_t].
/// @param pinocchioInterface Pinocchio interface.
///
/// @return Vector of contact positions in the inertial frame.

template <typename SCALAR_T>
std::vector<VECTOR3_T<SCALAR_T>> getContactPositions(const PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface,
                                                     const MpcRobotModelBase<SCALAR_T>& mpcRobotModel);

///
/// @brief Computes and returns all the frame positions in the inertial frame.
///
/// @tparam SCALAR_T Scalar type [scalar_t/ad_scalar_t].
/// @param q Current generalized coordinates.
/// @param pinocchioInterface Pinocchio interface.
/// @param frameNames Names of the frames for which the positions should be computed.
///
/// @return Vector of contact positions in the inertial frame.

template <typename SCALAR_T>
std::vector<VECTOR3_T<SCALAR_T>> computeFramePositions(const VECTOR_T<SCALAR_T>& q,
                                                       PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface,
                                                       const std::vector<std::string>& frameNames);

///
/// @brief Returns all the frame positions in the inertial frame.
/// Assumes that the frame placements are up to date (e.g. updateFramePlacements has been called).
///
/// @tparam SCALAR_T Scalar type [scalar_t/ad_scalar_t].
/// @param pinocchioInterface Pinocchio interface.
/// @param frameNames Names of the frames for which the positions should be returned.
///
/// @return Vector of contact positions in the inertial frame.

template <typename SCALAR_T>
std::vector<VECTOR3_T<SCALAR_T>> getFramePositions(const PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface,
                                                   std::vector<std::string> frameNames);

///
/// @brief The pendulum length of the robot: its whole-body center of mass above the mean height of its contact frames.
///
/// The one definition of the linear inverted pendulum's height in this repository. Every consumer that needs a nominal
/// pendulum length evaluates it at the task file's `initialState` - the contact planner's `shared.comHeight` and the DCM
/// terminal cost's `dcm_terminal_cost.comHeight` when they are 0 (CentroidalMpcInterface::getNominalComHeight), and the
/// locomotion heuristics' nominal CoM height - so that they cannot disagree; the capture-point heuristic also measures
/// it at every solve. tools/locomotion_heuristics/derive_parameters.py computes the same quantity the same way.
///
/// Above the mean foot height rather than above the world origin, so that it is the same number on a robot standing on
/// a box as on one standing on the floor. The contact frames are the sole centers (createPinocchioModel adds them).
/// Updates the center of mass and the frame placements of `pinocchioInterface`'s data.
///
/// @param q Generalized coordinates (MpcRobotModelBase::getGeneralizedCoordinates of a state).
/// @param pinocchioInterface Pinocchio interface of the MPC model.
/// @param mpcRobotModel The MPC robot model, which names the contact frames.
///
/// @return [m] the center of mass height above the mean height of the contact frames.

scalar_t computeComHeightAboveFeet(const vector_t& q,
                                   PinocchioInterface& pinocchioInterface,
                                   const MpcRobotModelBase<scalar_t>& mpcRobotModel);

///
/// @brief Return amount of legs in contact
///
/// @return number of clodes leg contacts as unsigned integer

inline size_t numberOfLegsInContacts(const contact_flag_t& contactFlags) {
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
/// @brief The magnitude g [m/s^2] of the gravity of `model` (pinocchio::Model::gravity): the gravity the robot's
/// dynamics use, and the one every pendulum of the MPC (the DCM terminal cost, the contact planner, the capture point)
/// is made with, so that none of them is a configuration key that could disagree with the model.
///
/// @return InvalidArgument unless the gravity points straight down the world z axis with a finite, positive magnitude,
///         which the pendulums' sqrt(g / z) assumes.
///
absl::StatusOr<scalar_t> modelGravity(const pinocchio::ModelTpl<scalar_t>& model);

///
/// @brief The weight of the robot that `pinocchioInterface` models [N]: its total mass times kGravitationalAcceleration.
///
/// Evaluated per call rather than cached in a function-static: a static is initialized by the first model that reaches
/// it and then answers for every other model in the process (the test binaries hold several robots at once).
///
inline scalar_t computeRobotWeight(const PinocchioInterface& pinocchioInterface) {
  return computeTotalMass(pinocchioInterface.getModel()) * kGravitationalAcceleration;
}

inline vector_t weightCompensatingInput(const PinocchioInterface& pinocchioInterface,
                                        const contact_flag_t& contactFlags,
                                        const MpcRobotModelBase<scalar_t>& mpcRobotModel) {
  const scalar_t weight = computeRobotWeight(pinocchioInterface);
  const size_t numStanceLegs = numberOfLegsInContacts(contactFlags);
  vector_t input = vector_t::Zero(mpcRobotModel.getInputDim());
  if (numStanceLegs > 0) {
    // NOTE: uses the input-only setter, i.e. the force is written in the frame of the input parameterization
    // (world frame for wrench-space models, local contact frame for basis-vector inputs). For a flat foot the
    // vertical force is identical in both frames. Prefer the state-aware overload below when a state is available.
    const vector3_t forceInInertialFrame(0.0, 0.0, weight / numStanceLegs);
    for (size_t i = 0; i < contactFlags.size(); ++i) {
      if (contactFlags[i]) {
        mpcRobotModel.setContactForce(input, forceInInertialFrame, i);
      }
    }
  }
  return input;
}

///
/// @brief Computes the weight-compensating input (evenly distributed vertical world-frame forces) for the
/// given state. The state is used to express the world-frame force correctly in the input parameterization
/// (e.g. rotated into the local contact frame for basis-vector inputs).
///
/// @param pinocchioInterface Pinocchio interface.
/// @param contactFlags Contact flags.
/// @param mpcRobotModel MPC robot model.
/// @param state Current MPC state.
///
/// @return input with weight-compensating contact forces and zero joint velocities.

inline vector_t weightCompensatingInput(const PinocchioInterface& pinocchioInterface,
                                        const contact_flag_t& contactFlags,
                                        const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                                        const vector_t& state) {
  const scalar_t weight = computeRobotWeight(pinocchioInterface);
  const size_t numStanceLegs = numberOfLegsInContacts(contactFlags);
  vector_t input = vector_t::Zero(mpcRobotModel.getInputDim());
  if (numStanceLegs > 0) {
    const vector3_t forceInInertialFrame(0.0, 0.0, weight / numStanceLegs);
    for (size_t i = 0; i < contactFlags.size(); ++i) {
      if (contactFlags[i]) {
        mpcRobotModel.setContactForceInWorldFrame(state, input, forceInInertialFrame, i);
      }
    }
  }
  return input;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

template <typename SCALAR_T>
inline pinocchio::FrameIndex getContactFrameIndex(const PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface,
                                                  const MpcRobotModelBase<SCALAR_T>& mpcRobotModel,
                                                  size_t contactIndex) {
  return pinocchioInterface.getModel().getFrameId(mpcRobotModel.modelSettings.contactNames[contactIndex]);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

template <typename SCALAR_T>
inline std::vector<pinocchio::FrameIndex> getContactFrameIndices(const PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface,
                                                                 const MpcRobotModelBase<SCALAR_T>& mpcRobotModel) {
  // Sized, not reserved: reserve leaves size() at 0, so indexing below would be out of bounds and the returned
  // vector would be empty - which silently turned every range-for over these indices into a no-op.
  std::vector<pinocchio::FrameIndex> contactFrameIndices(kNumContacts);
  for (size_t i = 0; i < kNumContacts; ++i) {
    contactFrameIndices[i] = getContactFrameIndex<SCALAR_T>(pinocchioInterface, mpcRobotModel, i);
  }
  return contactFrameIndices;
}

/**
 * Returns the Pinocchio frame of each contact of `mpcRobotModel` (modelSettings.contactNames), in the order of the
 * contacts: the sole frames the inverse dynamics below apply the contact wrenches at. The frames must exist
 * (checkContactFramesExist(), which the MPC interfaces call once at construction).
 */
template <typename SCALAR_T>
inline feet_array_t<pinocchio::FrameIndex> getContactFrameIndexArray(const PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface,
                                                                     const MpcRobotModelBase<SCALAR_T>& mpcRobotModel) {
  feet_array_t<pinocchio::FrameIndex> contactFrameIndices{};
  for (size_t i = 0; i < kNumContacts; ++i) {
    contactFrameIndices[i] = getContactFrameIndex<SCALAR_T>(pinocchioInterface, mpcRobotModel, i);
  }
  return contactFrameIndices;
}

/**
 * Checks that `model` has a frame of each of the first kNumContacts names of `contactNames` (modelSettings.contactNames).
 *
 * @return InvalidArgument when there are fewer than kNumContacts names; NotFound naming the first contact frame that
 *         `model` does not have, which every term that reads a contact frame by its index would otherwise read out of
 *         bounds (pinocchio::Model::getFrameId() returns nframes for an unknown name).
 */
absl::Status checkContactFramesExist(const pinocchio::Model& model, absl::Span<const std::string> contactNames);

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

///
/// @brief Computes the center of pressure (CoP) in the inertial frame.
///
/// @warning Assumes that the frame placements are up to date. Does not work when frame is not in contact -> f_z = 0 results in NaN.
///
/// @tparam SCALAR_T Scalar type [scalar_t/ad_scalar_t].
/// @param input Current input.
/// @param pinocchioInterface Pinocchio interface.
///
/// @return Location of center of pressure in the inertial frame.

template <typename SCALAR_T>
inline VECTOR3_T<SCALAR_T> computeContactCoP(const VECTOR_T<SCALAR_T>& input,
                                             const PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface,
                                             size_t contactIndex,
                                             const MpcRobotModelBase<scalar_t>& mpcRobotModel) {
  const VECTOR6_T<SCALAR_T> localContactWrench =
      rotateVectorWorldToLocal<SCALAR_T>(mpcRobotModel.getContactWrench(input, contactIndex), pinocchioInterface.getData(),
                                         getContactFrameIndex(pinocchioInterface, mpcRobotModel, contactIndex));
  // The wrench is [force, torque]: the CoP is (-tau_y, tau_x) / f_z.
  SCALAR_T copX = -localContactWrench[kWrenchTorqueYIndex] / localContactWrench[kWrenchForceZIndex];
  SCALAR_T copY = localContactWrench[kWrenchTorqueXIndex] / localContactWrench[kWrenchForceZIndex];
  VECTOR3_T<SCALAR_T> copInLocalFrame(copX, copY, /*z=*/0.0);
  return transformPointLocalToWorld(copInLocalFrame, pinocchioInterface.getData(),
                                    getContactFrameIndex(pinocchioInterface, mpcRobotModel, contactIndex));
}

///
/// @brief Computes the center of pressure (CoP) in the inertial frame using the state-aware world-frame wrench
/// accessor, which is frame-correct for every input parameterization (wrench-space and basis-vector inputs).
///
/// @warning Assumes that the frame placements are up to date. Does not work when frame is not in contact -> f_z = 0 results in NaN.
///
/// @tparam SCALAR_T Scalar type [scalar_t/ad_scalar_t].
/// @param state Current state.
/// @param input Current input.
/// @param pinocchioInterface Pinocchio interface.
///
/// @return Location of center of pressure in the inertial frame.

template <typename SCALAR_T>
inline VECTOR3_T<SCALAR_T> computeContactCoP(const VECTOR_T<SCALAR_T>& state,
                                             const VECTOR_T<SCALAR_T>& input,
                                             const PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface,
                                             size_t contactIndex,
                                             const MpcRobotModelBase<SCALAR_T>& mpcRobotModel) {
  const pinocchio::FrameIndex frameIndex = getContactFrameIndex(pinocchioInterface, mpcRobotModel, contactIndex);
  const VECTOR6_T<SCALAR_T> localContactWrench = rotateVectorWorldToLocal<SCALAR_T>(
      mpcRobotModel.getContactWrenchInWorldFrame(state, input, contactIndex), pinocchioInterface.getData(), frameIndex);
  // The wrench is [force, torque]: the CoP is (-tau_y, tau_x) / f_z.
  SCALAR_T copX = -localContactWrench[kWrenchTorqueYIndex] / localContactWrench[kWrenchForceZIndex];
  SCALAR_T copY = localContactWrench[kWrenchTorqueXIndex] / localContactWrench[kWrenchForceZIndex];
  VECTOR3_T<SCALAR_T> copInLocalFrame(copX, copY, /*z=*/0.0);
  return transformPointLocalToWorld(copInLocalFrame, pinocchioInterface.getData(), frameIndex);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

///
/// @brief Computes the base acceleration
///
/// @param M Given mass matrix
/// @param nle nonlinear effects in joint space inlcuding gravity comp, centrifugal, corriolis
/// @param qdd_joints Generalized accelerations
/// @param externalForcesInJointSpace Sum of J^T F_ext
///
/// @return linear and angular base acceleration.

template <typename SCALAR_T>
VECTOR6_T<SCALAR_T> computeBaseAcceleration(const MATRIX_T<SCALAR_T>& M,
                                            const VECTOR_T<SCALAR_T>& nle,
                                            const VECTOR_T<SCALAR_T>& qdd_joints,
                                            const VECTOR_T<SCALAR_T>& externalForcesInJointSpace);

///
/// @brief Floating-base inverse dynamics: the joint torques that produce the joint accelerations `qdd_joints` while the
/// feet push on the ground with `footWrenches`, the base being free.
///
/// The unactuated base rows of M qdd + nle = S^T tau + J^T W fix the base acceleration those wrenches produce
/// (computeBaseAcceleration), and the joint rows then return tau_j = M_jb a_b + M_jj qdd_j + nle_j - (J^T W)_j: the joint
/// rows of pinocchio::rnea() at that base acceleration and with those external forces. It used to multiply the joint rows
/// of crba()'s upper-triangular M and so left out the base coupling M_jb a_b (and the lower triangle of M_jj);
/// testJointTorqueInverseDynamics pins it against RNEA.
///
/// With no wrench at all the base is in free fall and gravity loads no joint; for a robot whose base is held from outside
/// (the gantry) use computeBaseHeldJointTorques.
///
/// Updates M, nle and the frame Jacobians in `pinocchioInterface`'s data.
///
/// @param q Generalized coordinates [base position (3), base quaternion (x, y, z, w) (4), joint angles]: nq = 7 + nj.
/// @param qd Generalized velocities [base linear velocity in the world (3), body angular velocity (3), joint
/// velocities]: nv = 6 + nj.
/// @param qdd_joints Joint accelerations.
/// @param footWrenches [W_left, W_right], world frame, at the origins of the sole frames `contactFrames`.
/// @param contactFrames The sole frame of each contact, in the order of the contacts (getContactFrameIndexArray()); each
/// must be a frame of the model.
/// @param pinocchioInterface Pinocchio interface of the MPC model.
///
/// @return joint torques of same dimension as qdd_joints

template <typename SCALAR_T>
VECTOR_T<SCALAR_T> computeJointTorques(const VECTOR_T<SCALAR_T>& q,
                                       const VECTOR_T<SCALAR_T>& qd,
                                       const VECTOR_T<SCALAR_T>& qdd_joints,
                                       const feet_array_t<VECTOR6_T<SCALAR_T>>& footWrenches,
                                       const feet_array_t<pinocchio::FrameIndex>& contactFrames,
                                       PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface);

///
/// @brief Inverse dynamics of a robot whose base is held still from outside (the gantry): the joint rows of
/// M qdd + nle = S^T tau + J^T W at zero base acceleration, tau_j = M_jj qdd_j + nle_j - (J^T W)_j.
///
/// The base rows are not balanced - whatever holds the base supplies them. With no wrench this is the gravity and
/// Coriolis torque of the free legs, and with the weight on the feet it is g_j(q) - J_j^T W, the static feedforward that
/// holds a posture. For qdd_j = 0 it is exactly what computeJointTorques returned before it included the base coupling.
///
/// Updates M, nle and the frame Jacobians in `pinocchioInterface`'s data.
///
/// @param q Generalized coordinates [base position (3), base quaternion (x, y, z, w) (4), joint angles]: nq = 7 + nj.
/// @param qd Generalized velocities [base linear velocity in the world (3), body angular velocity (3), joint
/// velocities]: nv = 6 + nj.
/// @param qdd_joints Joint accelerations.
/// @param footWrenches [W_left, W_right], world frame, at the origins of the sole frames `contactFrames`.
/// @param contactFrames The sole frame of each contact, as for computeJointTorques.
/// @param pinocchioInterface Pinocchio interface of the MPC model.
///
/// @return joint torques of same dimension as qdd_joints

template <typename SCALAR_T>
VECTOR_T<SCALAR_T> computeBaseHeldJointTorques(const VECTOR_T<SCALAR_T>& q,
                                               const VECTOR_T<SCALAR_T>& qd,
                                               const VECTOR_T<SCALAR_T>& qdd_joints,
                                               const feet_array_t<VECTOR6_T<SCALAR_T>>& footWrenches,
                                               const feet_array_t<pinocchio::FrameIndex>& contactFrames,
                                               PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface);

}  // namespace ocs2::humanoid
