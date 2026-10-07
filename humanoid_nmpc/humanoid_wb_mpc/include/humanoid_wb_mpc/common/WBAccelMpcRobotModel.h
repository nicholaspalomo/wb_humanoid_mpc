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

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "absl/base/nullability.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid {

/******************************************************************************************************/
/* State Vector definition

  Define the state vector x = [q_b_lin q_b_ang, q_j, qd_b_lin qd_b_ang, qd_j]^T

    Base linear position = [p_base_x, p_base_y, p_base_z] in world frame
    Base euler angles = [euler_z, euler_y, euler_x] from world to local frame?
    Joint angles q_j
    Base linear velocity = [v_base_x, v_base_y, v_base_z] in world frame
    Base euler angle derivatives = [euler_d_z, euler_d_y, euler_D_x]
    Joint velocities qd_j

*/
/******************************************************************************************************/

/******************************************************************************************************/
/* Input Vector definition

  Define the input vector u = [W_l, W_r, qdd_j]^T

    Left contact Wrench W_l in inertial frame
    Right contact Wrench W_r in inertial frame
    Joint accelerations qdd_j

*/
/******************************************************************************************************/

/**
 * The state and input layout of the whole-body MPC (above): getters and setters of the base pose, the joint angles and
 * velocities, the joint accelerations and the contact wrenches in its state and input vectors. Holds a reference to
 * `modelSettings`, which must outlive it; otherwise stateless, so its const methods are safe to call concurrently.
 * Copied only by clone(). Every accessor takes a state of getStateDim() and an input of getInputDim() entries, unchecked:
 * the solver and the control thread call them on every evaluation.
 */
template <typename SCALAR_T>
class WBAccelMpcRobotModel : public MpcRobotModelBase<SCALAR_T> {
 public:
  explicit WBAccelMpcRobotModel(const ModelSettings& modelSettings)
      : MpcRobotModelBase<SCALAR_T>(modelSettings, 2 * (6 + modelSettings.mpc_joint_dim), 6 * kNumContacts + modelSettings.mpc_joint_dim) {}
  ~WBAccelMpcRobotModel() override = default;
  WBAccelMpcRobotModel& operator=(const WBAccelMpcRobotModel&) = delete;
  WBAccelMpcRobotModel(WBAccelMpcRobotModel&&) = delete;
  WBAccelMpcRobotModel& operator=(WBAccelMpcRobotModel&&) = delete;
  WBAccelMpcRobotModel* absl_nonnull clone() const override { return new WBAccelMpcRobotModel(*this); }

  /******************************************************************************************************/
  /*                                          Start indices                                             */
  /******************************************************************************************************/

  // LINT.IfChange(state_input_indices)
  size_t getBaseStartindex() const override { return 0; };
  size_t getJointStartindex() const override { return 6; };
  // Be careful, the joint Velocities are part of the state vector here.
  size_t getJointVelocitiesStartindex() const override { return 12 + this->modelSettings.mpc_joint_dim; };
  size_t getJointAccelerationsStartindex() const { return 6 * kNumContacts; }

  // Assumes contact wrench [f_x, f_y, f_z, M_x, M_y, M_z]^T
  size_t getContactWrenchStartIndices(size_t contactIndex) const override { return 6 * contactIndex; };
  size_t getContactForceStartIndices(size_t contactIndex) const override { return getContactWrenchStartIndices(contactIndex); };
  size_t getContactMomentStartIndices(size_t contactIndex) const override { return getContactWrenchStartIndices(contactIndex) + 3; };
  // LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/src/config/weights/StateInputWeightsFromConfig.cpp:state_input_offsets)

  /******************************************************************************************************/
  /*                                     Generalized coordinates                                        */
  /******************************************************************************************************/

  VECTOR_T<SCALAR_T> getGeneralizedCoordinates(const VECTOR_T<SCALAR_T>& state) const override {
    return state.head(6 + this->modelSettings.mpc_joint_dim);
  };

  VECTOR6_T<SCALAR_T> getBasePose(const VECTOR_T<SCALAR_T>& state) const override { return state.head(6); };

  VECTOR3_T<SCALAR_T> getBasePosition(const VECTOR_T<SCALAR_T>& state) const override { return state.head(3); }

  VECTOR3_T<SCALAR_T> getBaseOrientationEulerZYX(const VECTOR_T<SCALAR_T>& state) const override { return state.segment(3, 3); }

  VECTOR3_T<SCALAR_T> getBaseComLinearVelocity(const VECTOR_T<SCALAR_T>& state) const override {
    return state.segment((6 + this->modelSettings.mpc_joint_dim), 3);
  }

  // Contains Euler angle derivatives, not angular velocity!
  VECTOR6_T<SCALAR_T> getBaseComVelocity(const VECTOR_T<SCALAR_T>& state) const override {
    return state.segment((6 + this->modelSettings.mpc_joint_dim), 6);
  };

  void setBaseComLinearVelocity(VECTOR_T<SCALAR_T>& state, const VECTOR3_T<SCALAR_T>& velocity) const override {
    state.segment((6 + this->modelSettings.mpc_joint_dim), 3) = velocity;
  };

  VECTOR_T<SCALAR_T> getJointAngles(const VECTOR_T<SCALAR_T>& state) const override {
    return state.segment(getJointStartindex(), this->modelSettings.mpc_joint_dim);
  };

  VECTOR_T<SCALAR_T> getJointVelocities(const VECTOR_T<SCALAR_T>& state, const VECTOR_T<SCALAR_T>& /*input*/) const override {
    return state.tail(this->modelSettings.mpc_joint_dim);
  };

  VECTOR_T<SCALAR_T> getGeneralizedVelocities(const VECTOR_T<SCALAR_T>& state, const VECTOR_T<SCALAR_T>& /*input*/) override {
    return state.tail(6 + this->modelSettings.mpc_joint_dim);
  };

  VECTOR_T<SCALAR_T> getJointAccelerations(const VECTOR_T<SCALAR_T>& input) const { return input.tail(this->modelSettings.mpc_joint_dim); }

  void setGeneralizedCoordinates(VECTOR_T<SCALAR_T>& state, const VECTOR_T<SCALAR_T>& generalizedCorrdinates) const override {
    state.head(6 + this->modelSettings.mpc_joint_dim) = generalizedCorrdinates;
  }

  void setBasePose(VECTOR_T<SCALAR_T>& state, const VECTOR6_T<SCALAR_T>& basePose) const override { state.head(6) = basePose; }

  void setBasePosition(VECTOR_T<SCALAR_T>& state, const VECTOR3_T<SCALAR_T>& position) const override { state.head(3) = position; }

  void setBaseOrientationEulerZYX(VECTOR_T<SCALAR_T>& state, const VECTOR3_T<SCALAR_T>& eulerAnglesZYX) const override {
    state.segment(3, 3) = eulerAnglesZYX;
  }

  void setBaseLinearVelocity(VECTOR_T<SCALAR_T>& state, const VECTOR3_T<SCALAR_T>& velocity) const {
    state.segment((6 + this->modelSettings.mpc_joint_dim), 3) = velocity;
  }

  void setBaseOrientationEulerZYXDerivatives(VECTOR_T<SCALAR_T>& state, const VECTOR3_T<SCALAR_T>& eulerAnglesZYXDerivative) const {
    state.segment((6 + this->modelSettings.mpc_joint_dim) + 3, 3) = eulerAnglesZYXDerivative;
  }

  void setJointAngles(VECTOR_T<SCALAR_T>& state, const VECTOR_T<SCALAR_T>& jointAngles) const override {
    state.segment(getJointStartindex(), this->modelSettings.mpc_joint_dim) = jointAngles;
  }

  void setJointVelocities(VECTOR_T<SCALAR_T>& state,
                          VECTOR_T<SCALAR_T>& /*input*/,
                          const VECTOR_T<SCALAR_T>& jointVelocities) const override {
    state.tail(this->modelSettings.mpc_joint_dim) = jointVelocities;
  }

  void adaptBasePoseHeight(VECTOR_T<SCALAR_T>& state, scalar_t heightChange) const override { state[2] += heightChange; }

  /******************************************************************************************************/
  /*                                          Contacts                                                  */
  /******************************************************************************************************/

  VECTOR6_T<SCALAR_T> getContactWrench(const VECTOR_T<SCALAR_T>& input, size_t contactIndex) const override {
    return input.segment(getContactWrenchStartIndices(contactIndex), 6);
  };

  VECTOR3_T<SCALAR_T> getContactForce(const VECTOR_T<SCALAR_T>& input, size_t contactIndex) const override {
    return input.segment(getContactForceStartIndices(contactIndex), 3);
  };

  VECTOR3_T<SCALAR_T> getContactMoment(const VECTOR_T<SCALAR_T>& input, size_t contactIndex) const override {
    return input.segment(getContactMomentStartIndices(contactIndex), 3);
  };

  void setContactWrench(VECTOR_T<SCALAR_T>& input, const VECTOR6_T<SCALAR_T>& wrench, size_t contactIndex) const override {
    input.segment(getContactWrenchStartIndices(contactIndex), 6) = wrench;
  };

  void setContactForce(VECTOR_T<SCALAR_T>& input, const VECTOR3_T<SCALAR_T>& force, size_t contactIndex) const override {
    input.segment(getContactForceStartIndices(contactIndex), 3) = force;
  };

  void setContactMoment(VECTOR_T<SCALAR_T>& input, const VECTOR3_T<SCALAR_T>& moment, size_t contactIndex) const override {
    input.segment(getContactMomentStartIndices(contactIndex), 3) = moment;
  };

 protected:
  // For clone(), and for the clone() of a derived model.
  WBAccelMpcRobotModel(const WBAccelMpcRobotModel& rhs) : MpcRobotModelBase<SCALAR_T>(rhs) {}
};

}  // namespace ocs2::humanoid
