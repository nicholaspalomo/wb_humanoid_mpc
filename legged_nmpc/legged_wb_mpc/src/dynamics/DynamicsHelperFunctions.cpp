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

#include "legged_wb_mpc/dynamics/DynamicsHelperFunctions.h"

#include <cstddef>

#include "ocs2_core/Dimensions.h"
#include "ocs2_core/manifold/UnitQuaternionMath.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"

#include "legged_common_mpc/common/ContactArray.h"
#include "legged_common_mpc/common/Types.h"
#include "legged_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "legged_wb_mpc/common/WBAccelMpcRobotModel.h"

// Pinnochio
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/multibody/fwd.hpp"

namespace ocs2::legged {
namespace {

/** Returns the world-frame wrench of every contact that `state` and `input`
 * carry, in contact order (one per modelSettings.contactNames entry). */
template <typename SCALAR_T>
ContactArray<VECTOR6_T<SCALAR_T>> computeWrenchContactsEwrtWorld(
    const VECTOR_T<SCALAR_T>& state,
    const VECTOR_T<SCALAR_T>& input,
    const WBAccelMpcRobotModel<SCALAR_T>& mpcRobotModel) {
  ContactArray<VECTOR6_T<SCALAR_T>> wrenchContactsEwrtWorld(
      mpcRobotModel.modelSettings.contactNames.size());
  for (size_t i = 0; i < wrenchContactsEwrtWorld.size(); ++i) {
    wrenchContactsEwrtWorld[i] =
        mpcRobotModel.getWrenchContactEwrtWorld(state, input,
                                                /*contactIndex=*/i);
  }
  return wrenchContactsEwrtWorld;
}

}  // namespace

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

template <typename SCALAR_T>
VECTOR6_T<SCALAR_T> computeBaseGeneralizedAcceleration(
    const VECTOR_T<SCALAR_T>& state,
    const VECTOR_T<SCALAR_T>& input,
    const PinocchioInterfaceTpl<SCALAR_T>& pinInterface,
    const WBAccelMpcRobotModel<SCALAR_T>& mpcRobotModel) {
  const typename PinocchioInterfaceTpl<SCALAR_T>::Model& model =
      pinInterface.getModel();
  // A copy: the interface is const here, and crba() / nonLinearEffects() write
  // their results into the data.
  typename PinocchioInterfaceTpl<SCALAR_T>::Data data = pinInterface.getData();
  const VECTOR_T<SCALAR_T> q = mpcRobotModel.getGeneralizedCoordinates(state);
  const VECTOR_T<SCALAR_T> qd =
      mpcRobotModel.getGeneralizedVelocities(state, input);
  const VECTOR_T<SCALAR_T> qdd_joints =
      mpcRobotModel.getJointAccelerations(input);

  data.M.fill(SCALAR_T(0.0));
  pinocchio::crba(model, data, q);
  pinocchio::nonLinearEffects(model, data, q, qd);

  // The base columns [dp_W, w_B] - the head of v - of the LOCAL_WORLD_ALIGNED
  // Jacobian of each configured contact frame (modelSettings.contactNames),
  // which take world-frame wrenches. The state-aware accessor is frame-correct
  // for every input parameterization; for the wrench-space WB model it is the
  // input wrench itself.
  VECTOR6_T<SCALAR_T> baseGeneralizedExternalForce =
      VECTOR6_T<SCALAR_T>::Zero();
  for (size_t i = 0; i < mpcRobotModel.modelSettings.contactNames.size(); ++i) {
    MATRIX_T<SCALAR_T> jacContactRtWorldEwrtWorld =
        MATRIX_T<SCALAR_T>::Zero(kTwistDim, mpcRobotModel.getVelocityDim());
    pinocchio::computeFrameJacobian(
        model, data, q,
        getContactFrameIndex<SCALAR_T>(pinInterface, mpcRobotModel, i),
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
        jacContactRtWorldEwrtWorld);
    const MATRIX6_T<SCALAR_T> baseColsJacContactRtWorldEwrtWorld =
        jacContactRtWorldEwrtWorld.leftCols(kPoseTangentDim);
    baseGeneralizedExternalForce +=
        baseColsJacContactRtWorldEwrtWorld.transpose() *
        mpcRobotModel.getWrenchContactEwrtWorld(state, input,
                                                /*contactIndex=*/i);
  }

  return computeBaseGeneralizedAcceleration<SCALAR_T>(
      data.M, data.nle, qdd_joints, baseGeneralizedExternalForce);
}
template ad_vector6_t computeBaseGeneralizedAcceleration(
    const ad_vector_t& state,
    const ad_vector_t& input,
    const PinocchioInterfaceTpl<ad_scalar_t>& pinInterface,
    const WBAccelMpcRobotModel<ad_scalar_t>& mpcRobotModel);
template vector6_t computeBaseGeneralizedAcceleration(
    const vector_t& state,
    const vector_t& input,
    const PinocchioInterfaceTpl<scalar_t>& pinInterface,
    const WBAccelMpcRobotModel<scalar_t>& mpcRobotModel);

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

template <typename SCALAR_T>
VECTOR_T<SCALAR_T> computeGeneralizedAccelerations(
    const VECTOR_T<SCALAR_T>& state,
    const VECTOR_T<SCALAR_T>& input,
    const PinocchioInterfaceTpl<SCALAR_T>& pinInterface,
    const WBAccelMpcRobotModel<SCALAR_T>& mpcRobotModel) {
  // Generalized Accelerations = [ddq_base, ddq_joints], the derivative of the
  // Pinocchio velocity v (nv entries)
  VECTOR_T<SCALAR_T> generalizedAccelerations =
      VECTOR_T<SCALAR_T>::Zero(mpcRobotModel.getVelocityDim());
  generalizedAccelerations.head(kPoseTangentDim) =
      computeBaseGeneralizedAcceleration<SCALAR_T>(state, input, pinInterface,
                                                   mpcRobotModel);
  generalizedAccelerations.tail(mpcRobotModel.getJointDim()) =
      mpcRobotModel.getJointAccelerations(input);
  return generalizedAccelerations;
}
template ad_vector_t computeGeneralizedAccelerations(
    const ad_vector_t& state,
    const ad_vector_t& input,
    const PinocchioInterfaceTpl<ad_scalar_t>& pinInterface,
    const WBAccelMpcRobotModel<ad_scalar_t>& mpcRobotModel);
template vector_t computeGeneralizedAccelerations(
    const vector_t& state,
    const vector_t& input,
    const PinocchioInterfaceTpl<scalar_t>& pinInterface,
    const WBAccelMpcRobotModel<scalar_t>& mpcRobotModel);

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

template <typename SCALAR_T>
VECTOR_T<SCALAR_T> computeStateDerivative(
    const VECTOR_T<SCALAR_T>& state,
    const VECTOR_T<SCALAR_T>& input,
    const PinocchioInterfaceTpl<SCALAR_T>& pinInterface,
    const WBAccelMpcRobotModel<SCALAR_T>& mpcRobotModel) {
  // State derivative = [dq/dt; ddq_base, ddq_joints] of the state [q (nq
  // entries), v (nv entries)]: v starts at nq. The configuration rate is [dp_W,
  // 1/2 G(xi) w_B, dq_j] (design section 2.5), with the raw xi of the state, so
  // that the quaternion's norm is constant to first order (xi' G(xi) = 0).
  const Eigen::Index velocityStart =
      static_cast<Eigen::Index>(mpcRobotModel.getConfigurationDim());
  const Eigen::Index orientationStart =
      static_cast<Eigen::Index>(mpcRobotModel.getBaseOrientationStartIndex());
  VECTOR_T<SCALAR_T> state_derivative =
      VECTOR_T<SCALAR_T>::Zero(mpcRobotModel.getStateDim());
  state_derivative.segment(mpcRobotModel.getBaseStartindex(), kPositionDim) =
      state.segment(velocityStart, kPositionDim);
  const VECTOR4_T<SCALAR_T> quatBaseRtWorldXyzw =
      state.template segment<kQuaternionCoefficients>(orientationStart);
  const VECTOR3_T<SCALAR_T> angvelBaseRtWorldEwrtBase =
      state.template segment<kRotationTangentDim>(velocityStart + kPositionDim);
  state_derivative.template segment<kQuaternionCoefficients>(orientationStart) =
      SCALAR_T(kUnitQuaternionHalfAngleFactor) *
      (quaternionRateMatrix(quatBaseRtWorldXyzw) * angvelBaseRtWorldEwrtBase);
  state_derivative.segment(mpcRobotModel.getJointStartindex(),
                           mpcRobotModel.getJointDim()) =
      mpcRobotModel.getJointVelocities(state, input);
  state_derivative.tail(mpcRobotModel.getVelocityDim()) =
      computeGeneralizedAccelerations<SCALAR_T>(state, input, pinInterface,
                                                mpcRobotModel);
  return state_derivative;
}
template ad_vector_t computeStateDerivative(
    const ad_vector_t& state,
    const ad_vector_t& input,
    const PinocchioInterfaceTpl<ad_scalar_t>& pinInterface,
    const WBAccelMpcRobotModel<ad_scalar_t>& mpcRobotModel);
template vector_t computeStateDerivative(
    const vector_t& state,
    const vector_t& input,
    const PinocchioInterfaceTpl<scalar_t>& pinInterface,
    const WBAccelMpcRobotModel<scalar_t>& mpcRobotModel);

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

template <typename SCALAR_T>
VECTOR_T<SCALAR_T> computeJointTorques(
    const VECTOR_T<SCALAR_T>& state,
    const VECTOR_T<SCALAR_T>& input,
    PinocchioInterfaceTpl<SCALAR_T>& pinInterface,
    const WBAccelMpcRobotModel<SCALAR_T>& mpcRobotModel) {
  const VECTOR_T<SCALAR_T> q = mpcRobotModel.getGeneralizedCoordinates(state);
  const VECTOR_T<SCALAR_T> qd =
      mpcRobotModel.getGeneralizedVelocities(state, input);
  const VECTOR_T<SCALAR_T> qdd_joints =
      mpcRobotModel.getJointAccelerations(input);

  // World-frame wrenches for the LOCAL_WORLD_ALIGNED Jacobians of
  // computeJointTorques (see computeBaseGeneralizedAcceleration).
  const ContactArray<VECTOR6_T<SCALAR_T>> wrenchContactsEwrtWorld =
      computeWrenchContactsEwrtWorld<SCALAR_T>(state, input, mpcRobotModel);
  return computeJointTorques<SCALAR_T>(
      q, qd, qdd_joints, wrenchContactsEwrtWorld,
      getContactFrameIndexArray<SCALAR_T>(pinInterface, mpcRobotModel),
      pinInterface);
}
template ad_vector_t computeJointTorques(
    const ad_vector_t& state,
    const ad_vector_t& input,
    PinocchioInterfaceTpl<ad_scalar_t>& pinInterface,
    const WBAccelMpcRobotModel<ad_scalar_t>& mpcRobotModel);
template vector_t computeJointTorques(
    const vector_t& state,
    const vector_t& input,
    PinocchioInterfaceTpl<scalar_t>& pinInterface,
    const WBAccelMpcRobotModel<scalar_t>& mpcRobotModel);

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

template <typename SCALAR_T>
VECTOR_T<SCALAR_T> computeBaseHeldJointTorques(
    const VECTOR_T<SCALAR_T>& state,
    const VECTOR_T<SCALAR_T>& input,
    PinocchioInterfaceTpl<SCALAR_T>& pinInterface,
    const WBAccelMpcRobotModel<SCALAR_T>& mpcRobotModel) {
  return computeBaseHeldJointTorques<SCALAR_T>(
      mpcRobotModel.getGeneralizedCoordinates(state),
      mpcRobotModel.getGeneralizedVelocities(state, input),
      mpcRobotModel.getJointAccelerations(input),
      computeWrenchContactsEwrtWorld<SCALAR_T>(state, input, mpcRobotModel),
      getContactFrameIndexArray<SCALAR_T>(pinInterface, mpcRobotModel),
      pinInterface);
}
template vector_t computeBaseHeldJointTorques(
    const vector_t& state,
    const vector_t& input,
    PinocchioInterfaceTpl<scalar_t>& pinInterface,
    const WBAccelMpcRobotModel<scalar_t>& mpcRobotModel);

}  // namespace ocs2::legged
